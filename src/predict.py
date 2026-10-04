"""
Инференс на тестовой выборке по одному (или нескольким) чекпоинтам.
Можно запускать сразу после того, как готов первый fold*_best.pt,
не дожидаясь остальных фолдов.

Одиночный чекпоинт:
    python src/predict.py \
        --checkpoints models/v1/fold0_best.pt \
        --test_csv data/test.csv \
        --images_dir data/images \
        --out submission.csv

Ансамбль из нескольких (когда дообучатся остальные фолды —
просто перечислите все через запятую, скрипт усреднит прогнозы):
    python src/predict.py \
        --checkpoints models/v1/fold0_best.pt,models/v1/fold1_best.pt,models/v1/fold2_best.pt \
        --test_csv data/test.csv \
        --images_dir data/images \
        --out submission.csv

С TTA (усреднение по флипу + паре кропов, снижает шум прогноза):
    python src/predict.py --checkpoints models/v1/fold0_best.pt --tta \
        --test_csv data/test.csv --images_dir data/images --out submission.csv
"""

import os
import argparse
import logging

import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataset import load_image_bgr, get_val_transforms, IMAGE_SIZE
from model import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TestDataset(Dataset):
    """
    Для test.csv эталонов нет — только image_id. Битые/отсутствующие
    изображения не выбрасываем (в отличие от train), а помечаем и
    подставляем дефолтный прогноз, чтобы submission.csv покрывал
    ВСЕ 307 id, как требует задание.
    """

    def __init__(self, df: pd.DataFrame, images_dir: str, transform, tta: bool = False):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        self.tta = tta

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.images_dir, f"{row['image_id']}.jpg")
        img = load_image_bgr(path)
        is_valid = img is not None
        if not is_valid:
            img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_t = self.transform(image=img)["image"]
        if self.tta:
            img_flip = self.transform(image=np.ascontiguousarray(img[:, ::-1, :]))["image"]
            return img_t, img_flip, is_valid, str(row["image_id"])
        return img_t, is_valid, str(row["image_id"])


@torch.no_grad()
def predict_with_checkpoint(ckpt_path: str, loader: DataLoader, device, tta: bool):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(ckpt["backbone"], pretrained=False, pooling=ckpt.get("pooling", "avg")).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info("Загружен %s (val_MAE на своём фолде = %.3f)", ckpt_path, ckpt.get("val_mae", float("nan")))

    all_ids, all_preds, all_valid = [], [], []
    for batch in tqdm(loader, desc=f"predict[{os.path.basename(ckpt_path)}]"):
        if tta:
            imgs, imgs_flip, is_valid, ids = batch
            imgs = imgs.to(device)
            imgs_flip = imgs_flip.to(device)
            preds = (model(imgs) + model(imgs_flip)) / 2.0
        else:
            imgs, is_valid, ids = batch
            imgs = imgs.to(device)
            preds = model(imgs)

        all_ids.extend(ids)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_valid.extend([bool(v) for v in is_valid])

    return all_ids, np.array(all_preds), all_valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True,
                         help="один путь или несколько через запятую (ансамбль)")
    parser.add_argument("--test_csv", default="data/test.csv")
    parser.add_argument("--images_dir", default="data/images")
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--tta", action="store_true", help="усреднение прогноза с горизонтальным флипом")
    parser.add_argument("--fallback_value", type=float, default=50.0,
                         help="прогноз для битых/отсутствующих файлов (нейтральное значение)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Устройство: %s", device)

    test_df = pd.read_csv(args.test_csv)
    assert "image_id" in test_df.columns, f"Ожидалась колонка image_id, получено: {test_df.columns.tolist()}"
    logger.info("Тестовых id: %d", len(test_df))

    transform = get_val_transforms(IMAGE_SIZE)
    ds = TestDataset(test_df, args.images_dir, transform, tta=args.tta)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    ckpt_paths = [c.strip() for c in args.checkpoints.split(",") if c.strip()]

    ensemble_preds = None
    ref_ids, ref_valid = None, None
    for ckpt_path in ckpt_paths:
        ids, preds, valid = predict_with_checkpoint(ckpt_path, loader, device, args.tta)
        if ref_ids is None:
            ref_ids, ref_valid = ids, valid
        else:
            assert ids == ref_ids, "Порядок id разошёлся между чекпоинтами — не должно происходить при shuffle=False"
        ensemble_preds = preds if ensemble_preds is None else ensemble_preds + preds

    ensemble_preds = ensemble_preds / len(ckpt_paths)
    ensemble_preds = np.clip(ensemble_preds, 0, 100)

    # для битых/отсутствующих файлов подставляем нейтральный fallback, а не прогноз модели на чёрном кадре
    n_invalid = sum(1 for v in ref_valid if not v)
    if n_invalid:
        logger.warning("%d изображений не читаются, подставляем fallback_value=%.1f", n_invalid, args.fallback_value)
        ensemble_preds = np.array([
            args.fallback_value if not v else p for p, v in zip(ensemble_preds, ref_valid)
        ])

    submission = pd.DataFrame({"image_id": ref_ids, "load_pct": ensemble_preds})

    # подстраховка: гарантируем что покрыты ВСЕ id из test.csv, в исходном порядке
    submission = test_df[["image_id"]].astype(str).merge(
        submission.assign(image_id=submission["image_id"].astype(str)),
        on="image_id", how="left"
    )
    missing = submission["load_pct"].isna().sum()
    if missing:
        logger.warning("%d id не получили прогноз (не должно происходить), заполняем fallback", missing)
        submission["load_pct"] = submission["load_pct"].fillna(args.fallback_value)

    submission.to_csv(args.out, index=False)
    logger.info("Сохранено: %s (%d строк, использовано чекпоинтов: %d, TTA=%s)",
                args.out, len(submission), len(ckpt_paths), args.tta)


if __name__ == "__main__":
    main()
