"""
Обучение регрессора загрузки кузова с валидацией по GroupKFold
(группы из train_groups.csv не разрезаются между train/val).

Запуск:
    python src/train.py \
        --train_csv data/train.csv \
        --groups_csv data/train_groups.csv \
        --images_dir data/images \
        --out_dir models/v1 \
        --backbone efficientnet_b0 \
        --epochs 25 \
        --folds 5

Дообучение с чекпоинта (используется тем же скриптом):
    python src/train.py \
        --train_csv data/new_labeled.csv \
        --groups_csv data/new_labeled_groups.csv \
        --images_dir data/images \
        --out_dir models/v2 \
        --resume models/v1/fold0_best.pt \
        --epochs 10 --lr 1e-5
"""

import os
import argparse
import logging
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from dataset import (
    CargoLoadDataset,
    build_valid_dataframe,
    merge_groups,
    get_train_transforms,
    get_val_transforms,
)
from model import build_model
from metrics import mae, accuracy_at_10

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_one_epoch(model, loader, optimizer, criterion, device, train: bool, scaler=None):
    model.train() if train else model.eval()
    total_loss = 0.0
    n_samples = 0
    all_true, all_pred = [], []
    use_amp = scaler is not None and device.type == "cuda"

    torch.set_grad_enabled(train)
    for imgs, targets in tqdm(loader, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad()

        with torch.autocast(device_type="cuda", enabled=use_amp):
            preds = model(imgs)
            loss = criterion(preds, targets)

        if train:
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        n_samples += bs
        all_true.append(targets.detach().cpu().numpy())
        all_pred.append(preds.detach().cpu().numpy())

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    return total_loss / n_samples, mae(all_true, all_pred), accuracy_at_10(all_true, all_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data/train.csv")
    parser.add_argument("--groups_csv", default="data/train_groups.csv")
    parser.add_argument("--images_dir", default="data/images")
    parser.add_argument("--out_dir", default="models/v1")
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--patience", type=int, default=6, help="early stopping по val MAE")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None, help="путь к чекпоинту для дообучения")
    parser.add_argument("--pooling", choices=["avg", "gem"], default="gem",
                         help="gem лучше ловит ЧАСТИЧНУЮ занятость (см. анализ ошибок)")
    parser.add_argument("--balance_middle", action="store_true", default=True,
                         help="взвешенный сэмплер: чаще показывать модели середину диапазона 20-80%%, "
                              "которой в датасете меньше и на которой модель ошибается сильнее всего")
    parser.add_argument("--use_amp", action="store_true", default=True,
                         help="mixed precision - ускоряет обучение на GPU в 1.5-3 раза без потери качества")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Apple Silicon (M1/M2/M3) GPU через Metal. Без этой ветки PyTorch
        # молча падает на CPU, что и давало ~5 минут/эпоху вместо секунд.
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Используем устройство: %s", device)

    # ---------- 1. Готовим датасет ----------
    logger.info("Читаем и валидируем изображения...")
    df = build_valid_dataframe(args.train_csv, args.images_dir)
    df = merge_groups(df, args.groups_csv)
    logger.info("Итоговый датасет: %d строк, %d уникальных групп", len(df), df["group_id"].nunique())

    # ---------- 2. GroupKFold, чтобы фото одной группы не утекали между train/val ----------
    gkf = GroupKFold(n_splits=args.folds)
    fold_maes = []
    fold_acc10 = []
    oof_records = []  # для стекинга и разбора ошибок: image_id, true, pred, fold

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df, groups=df["group_id"])):
        logger.info("===== Фолд %d/%d =====", fold_idx + 1, args.folds)
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        # проверка на утечку групп между train/val
        leak = set(train_df["group_id"]) & set(val_df["group_id"])
        assert not leak, f"Утечка групп между train/val: {leak}"

        train_ds = CargoLoadDataset(train_df, transform=get_train_transforms(args.image_size))
        val_ds = CargoLoadDataset(val_df, transform=get_val_transforms(args.image_size))

        if args.balance_middle:
            # веса обратно пропорциональны частоте "зоны" -> середина 20-80% чаще попадает в батч
            zone = pd.cut(train_df["load_pct"], bins=[0, 20, 80, 100.01], right=False, labels=[0, 1, 2])
            zone_counts = zone.value_counts()
            sample_weights = zone.map(lambda z: 1.0 / zone_counts[z]).values.astype(float)
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_df), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                       num_workers=args.num_workers, pin_memory=True, drop_last=True)
        else:
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)

        model = build_model(args.backbone, pretrained=(args.resume is None), pooling=args.pooling).to(device)
        if args.resume:
            logger.info("Загружаем веса для дообучения из %s", args.resume)
            state = torch.load(args.resume, map_location=device)
            model.load_state_dict(state["model_state"])

        criterion = nn.L1Loss()  # L1 == MAE, совпадает с целевой метрикой
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=(args.use_amp and device.type == "cuda"))

        best_val_mae = float("inf")
        epochs_no_improve = 0
        best_ckpt_path = os.path.join(args.out_dir, f"fold{fold_idx}_best.pt")

        for epoch in range(args.epochs):
            train_loss, train_mae, train_acc10 = run_one_epoch(
                model, train_loader, optimizer, criterion, device, train=True, scaler=scaler
            )
            val_loss, val_mae, val_acc10 = run_one_epoch(
                model, val_loader, optimizer, criterion, device, train=False
            )
            scheduler.step()

            logger.info(
                "Fold %d Epoch %d/%d | train_loss=%.3f train_MAE=%.3f | val_loss=%.3f val_MAE=%.3f val_Acc@10=%.3f",
                fold_idx, epoch + 1, args.epochs, train_loss, train_mae, val_loss, val_mae, val_acc10
            )

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_val_acc10 = val_acc10
                epochs_no_improve = 0
                torch.save({
                    "model_state": model.state_dict(),
                    "backbone": args.backbone,
                    "pooling": args.pooling,
                    "image_size": args.image_size,
                    "val_mae": val_mae,
                    "val_acc10": val_acc10,
                    "fold": fold_idx,
                }, best_ckpt_path)
                logger.info("  -> новый лучший чекпоинт сохранён: %s (val_MAE=%.3f)", best_ckpt_path, val_mae)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    logger.info("  -> early stopping (нет улучшения %d эпох)", args.patience)
                    break

        fold_maes.append(best_val_mae)
        fold_acc10.append(best_val_acc10)
        logger.info("Фолд %d завершён: лучший val_MAE=%.3f, val_Acc@10=%.3f", fold_idx, best_val_mae, best_val_acc10)

        # ---- собираем OOF-предсказания лучшим чекпоинтом фолда (нужно для стекинга и разбора ошибок) ----
        best_state = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_state["model_state"])
        model.eval()
        with torch.no_grad():
            for i, (imgs, targets) in enumerate(val_loader):
                imgs = imgs.to(device)
                preds = model(imgs).cpu().numpy()
                start = i * val_loader.batch_size
                batch_ids = val_df["image_id"].iloc[start:start + len(preds)].tolist()
                batch_true = targets.numpy()
                for img_id, t, p in zip(batch_ids, batch_true, preds):
                    oof_records.append({"image_id": img_id, "load_pct": float(t),
                                         "oof_pred": float(p), "fold": fold_idx})

    # ---------- 3. Итоговая сводка по кросс-валидации ----------
    logger.info("===== Итог по %d фолдам =====", args.folds)
    logger.info("MAE по фолдам: %s", [round(m, 3) for m in fold_maes])
    logger.info("Средний CV MAE: %.3f +- %.3f", float(np.mean(fold_maes)), float(np.std(fold_maes)))
    logger.info("Средний CV Accuracy@10: %.3f", float(np.mean(fold_acc10)))

    summary_path = os.path.join(args.out_dir, "cv_summary.csv")
    pd.DataFrame({
        "fold": list(range(len(fold_maes))),
        "val_mae": fold_maes,
        "val_acc10": fold_acc10,
    }).to_csv(summary_path, index=False)
    logger.info("Сводка сохранена: %s", summary_path)

    # OOF-предсказания всего датасета (каждое фото предсказано моделью, которая его не видела при обучении)
    oof_path = os.path.join(args.out_dir, "oof_predictions.csv")
    oof_df = pd.DataFrame(oof_records)
    oof_df.to_csv(oof_path, index=False)
    overall_oof_mae = mae(oof_df["load_pct"].values, oof_df["oof_pred"].values)
    logger.info("OOF-предсказания сохранены: %s (честный общий MAE по всем данным = %.3f)",
                oof_path, overall_oof_mae)


if __name__ == "__main__":
    main()
