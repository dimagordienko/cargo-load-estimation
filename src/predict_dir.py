"""Превращает папку с моделью, обученной src/train.py, в runs/<имя>/ для ensemble.py:
   runs/<имя>/test_pred.csv  - прогноз теста (среднее по fold*_best.pt, hflip-TTA, свой image_size у каждого чекпоинта)
   runs/<имя>/oof.csv        - копия oof_predictions.csv из папки модели

    python src/predict_dir.py --model_dir models/n_b2_384 --run_dir runs/n_b2_384 --test_images_dir data/test_images
"""
import os
import glob
import shutil
import argparse

import cv2
import numpy as np
import pandas as pd
import torch

from dataset import load_image_bgr, get_val_transforms
from model import build_model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict_ckpt(path, ids, images_dir, device, bs):
    ck = torch.load(path, map_location=device)
    size = int(ck.get("image_size", 384))
    model = build_model(ck["backbone"], pretrained=False, pooling=ck.get("pooling", "avg")).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    tf = get_val_transforms(size)

    imgs, valid = [], []
    for i in ids:
        im = load_image_bgr(os.path.join(images_dir, f"{i}.jpg"))
        valid.append(im is not None)
        im = np.zeros((size, size, 3), np.uint8) if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        imgs.append(im)

    out = []
    for s in range(0, len(imgs), bs):
        x = torch.stack([tf(image=im)["image"] for im in imgs[s:s + bs]]).to(device)
        p = (model(x) + model(torch.flip(x, dims=[3]))) / 2.0
        out.append(p.float().cpu().numpy())
    print(f"  {os.path.basename(path)}: size={size}, val_mae чекпоинта={ck.get('val_mae', float('nan')):.3f}")
    return np.concatenate(out), valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--test_csv", default="data/test.csv")
    ap.add_argument("--test_images_dir", default="data/test_images")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--fallback_value", type=float, default=50.0)
    a = ap.parse_args()

    ckpts = sorted(glob.glob(os.path.join(a.model_dir, "fold*_best.pt")))
    assert ckpts, f"нет fold*_best.pt в {a.model_dir}"
    ids = pd.read_csv(a.test_csv)["image_id"].astype(str).tolist()
    device = get_device()
    print(f"предсказание теста: {len(ckpts)} чекпоинтов, {len(ids)} фото, устройство {device}")

    preds, valid = [], None
    for c in ckpts:
        p, v = predict_ckpt(c, ids, a.test_images_dir, device, a.batch_size)
        preds.append(p)
        valid = v
    final = np.clip(np.mean(preds, axis=0), 0, 100)
    final = np.where(np.array(valid), final, a.fallback_value)
    if not all(valid):
        print(f"[WARN] не читаются {int(np.sum(~np.array(valid)))} фото, подставлено {a.fallback_value}")

    os.makedirs(a.run_dir, exist_ok=True)
    pd.DataFrame({"image_id": ids, "load_pct": final}).to_csv(os.path.join(a.run_dir, "test_pred.csv"), index=False)
    oof_src = os.path.join(a.model_dir, "oof_predictions.csv")
    if os.path.exists(oof_src):
        shutil.copy(oof_src, os.path.join(a.run_dir, "oof.csv"))
    else:
        print(f"[WARN] нет {oof_src} - обучение не дошло до конца, в ансамбль эту модель брать нельзя")
    print("готово ->", a.run_dir)


if __name__ == "__main__":
    main()
