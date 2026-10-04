"""Дообучение бэкбона (5 фолдов, EMA, TTA) -> OOF + прогноз теста в одном запуске.

Отличия от train.py:
  * 5 фолдов вместо 2 (модель видит 80% данных, а не 50%);
  * картинка ужимается в квадрат целиком (без чёрных полей), декодирование JPEG один раз;
  * EMA весов, без отбора эпохи по val (OOF честный), берётся последняя эпоха;
  * hflip-TTA; тест предсказывается прямо здесь, каждым фолдом;
  * каждый фолд сохраняется отдельно -> можно прервать и продолжить (готовые фолды пропускаются).

Примеры:
  python src/train_v4.py --backbone convnext_tiny.fb_in22k_ft_in1k --image_size 384 --epochs 20 --out_dir runs/cnxt_t_384
  python src/train_v4.py --backbone vit_small_patch14_dinov2.lvd142m --image_size 392 --lr 5e-5 --out_dir runs/dino_s_392
  # замер скорости: python src/train_v4.py ... --only_folds 0 --epochs 2 --out_dir runs/tmp
Чтобы Mac не засыпал:  caffeinate -i python src/train_v4.py ...
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse
import copy
import math
import random
import time

import albumentations as A
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from common import load_train_with_folds, read_rgb, mae, acc10


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


class Net(nn.Module):
    def __init__(self, name, size, drop_path=0.1, pretrained=True):
        super().__init__()
        kw = dict(pretrained=pretrained, num_classes=0)
        if drop_path > 0:
            kw["drop_path_rate"] = drop_path
        try:
            self.backbone = timm.create_model(name, img_size=size, **kw)
        except TypeError:
            self.backbone = timm.create_model(name, **kw)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.backbone.num_features, 1))
        nn.init.zeros_(self.head[1].weight)  # старт с константы 0.5: у ViT (DINOv2) признаки крупные и без этого выход 'взрывается'
        nn.init.constant_(self.head[1].bias, 0.5)

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)  # шкала 0..1


class EMA:
    def __init__(self, model, decay):
        self.m = copy.deepcopy(model).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.decay, self.n = decay, 0

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for e, p in zip(self.m.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(p.detach(), alpha=1 - d)
            else:
                e.copy_(p)


class ArrDS(Dataset):
    def __init__(self, imgs, y, tf):
        self.imgs, self.y, self.tf = imgs, y, tf

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.tf(image=self.imgs[i])["image"], np.float32(self.y[i] / 100.0)


def make_tf(mean, std):
    norm = [A.Normalize(mean=mean, std=std), ToTensorV2()]
    train = A.Compose([
        A.Affine(scale=(0.92, 1.08), translate_percent=(-0.04, 0.04), rotate=(-6, 6), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=15, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.15),
    ] + norm)
    return train, A.Compose(norm)


@torch.no_grad()
def predict(model, imgs, tf, device, bs, tta):
    model.eval()
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.stack([tf(image=im)["image"] for im in imgs[i:i + bs]]).to(device)
        p = model(x)
        if tta:
            p = (p + model(torch.flip(x, dims=[3]))) / 2
        out.append(p.float().cpu().numpy())
    return np.clip(np.concatenate(out) * 100.0, 0, 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/train.csv")
    ap.add_argument("--groups_csv", default="data/train_groups.csv")
    ap.add_argument("--test_csv", default="data/test.csv")
    ap.add_argument("--images_dir", default="data/images")
    ap.add_argument("--test_images_dir", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--backbone", default="convnext_tiny.fb_in22k_ft_in1k")
    ap.add_argument("--image_size", type=int, default=384, help="высота (и ширина, если не задан --image_width)")
    ap.add_argument("--image_width", type=int, default=None, help="ширина; для портретных фото 512x384: --image_size 512 --image_width 384")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4, help="бэкбон; голова = 10x")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--drop_path", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.98)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only_folds", default=None, help="например 0,1")
    ap.add_argument("--balance_middle", action="store_true", help="сэмплер в пользу 20-80%% (A/B-тест, по умолчанию выкл)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    device = get_device()
    print("device:", device)

    df = load_train_with_folds(a.train_csv, a.groups_csv, a.folds)
    te = pd.read_csv(a.test_csv)
    te["image_id"] = te["image_id"].astype(str)
    tdir = a.test_images_dir or a.images_dir

    t0 = time.time()
    tr_imgs = [read_rgb(os.path.join(a.images_dir, f"{i}.jpg"), a.image_size, a.image_width) for i in df["image_id"]]
    te_imgs = [read_rgb(os.path.join(tdir, f"{i}.jpg"), a.image_size, a.image_width) for i in te["image_id"]]
    print(f"картинки загружены за {time.time() - t0:.0f}s")

    probe = timm.create_model(a.backbone, pretrained=False, num_classes=0)
    cfg = timm.data.resolve_data_config({}, model=probe)
    del probe
    tf_train, tf_val = make_tf(cfg["mean"], cfg["std"])

    y = df["load_pct"].values
    folds = [int(f) for f in a.only_folds.split(",")] if a.only_folds else list(range(a.folds))

    for k in folds:
        f_oof = os.path.join(a.out_dir, f"oof_fold{k}.csv")
        f_test = os.path.join(a.out_dir, f"test_fold{k}.npy")
        if os.path.exists(f_oof) and os.path.exists(f_test):
            print(f"fold {k}: уже готов, пропускаю")
            continue
        set_seed(a.seed + k)
        tr_idx = np.where(df["fold"].values != k)[0]
        va_idx = np.where(df["fold"].values == k)[0]

        ds = ArrDS([tr_imgs[i] for i in tr_idx], y[tr_idx], tf_train)
        if a.balance_middle:
            z = pd.cut(pd.Series(y[tr_idx]), [-1, 20, 80, 101], right=False, labels=False)
            w = 1.0 / z.map(z.value_counts()).values.astype(float)
            loader = DataLoader(ds, batch_size=a.batch_size, drop_last=True,
                                sampler=WeightedRandomSampler(w, len(w), replacement=True))
        else:
            loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True, drop_last=True)

        model = Net(a.backbone, a.image_size, a.drop_path).to(device)
        ema = EMA(model, a.ema)
        opt = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": a.lr},
            {"params": model.head.parameters(), "lr": a.lr * 10},
        ], weight_decay=a.wd)
        steps = len(loader)
        total, warm = a.epochs * steps, steps
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
        crit = nn.SmoothL1Loss(beta=0.05)

        va_imgs = [tr_imgs[i] for i in va_idx]
        for ep in range(a.epochs):
            t1 = time.time()
            model.train()
            run = 0.0
            for x, t in loader:
                x, t = x.to(device), t.to(device)
                loss = crit(model(x), t)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                sched.step()
                ema.update(model)
                run += loss.item()
            p = predict(ema.m, va_imgs, tf_val, device, a.batch_size, tta=False)
            print(f"fold {k} ep {ep + 1}/{a.epochs} loss={run / steps:.4f} val_MAE(EMA)={mae(y[va_idx], p):.3f} "
                  f"acc@10={acc10(y[va_idx], p):.3f} [{time.time() - t1:.0f}s]", flush=True)

        oof = predict(ema.m, va_imgs, tf_val, device, a.batch_size, tta=True)
        tp = predict(ema.m, te_imgs, tf_val, device, a.batch_size, tta=True)
        print(f"=== fold {k}: финальный val_MAE (EMA+TTA) = {mae(y[va_idx], oof):.3f}", flush=True)
        torch.save({"state": ema.m.state_dict(), "backbone": a.backbone, "size": a.image_size}, os.path.join(a.out_dir, f"fold{k}.pt"))
        pd.DataFrame({"image_id": df["image_id"].values[va_idx], "load_pct": y[va_idx], "oof_pred": oof, "fold": k}).to_csv(f_oof, index=False)
        np.save(f_test, tp)
        del model, ema, opt
        if device.type == "mps":
            torch.mps.empty_cache()

    have = [k for k in range(a.folds) if os.path.exists(os.path.join(a.out_dir, f"oof_fold{k}.csv"))]
    if len(have) == a.folds:
        oof = pd.concat([pd.read_csv(os.path.join(a.out_dir, f"oof_fold{k}.csv")) for k in have]).reset_index(drop=True)
        oof.to_csv(os.path.join(a.out_dir, "oof.csv"), index=False)
        tp = np.mean([np.load(os.path.join(a.out_dir, f"test_fold{k}.npy")) for k in have], axis=0)
        pd.DataFrame({"image_id": te["image_id"], "load_pct": tp}).to_csv(os.path.join(a.out_dir, "test_pred.csv"), index=False)
        print(f"\nOOF MAE = {mae(oof['load_pct'], oof['oof_pred']):.3f}  Acc@10 = {acc10(oof['load_pct'], oof['oof_pred']):.3f}")
        print("готово:", a.out_dir)
    else:
        print(f"готовы фолды {have} из {a.folds}; oof.csv/test_pred.csv соберутся, когда будут все")


if __name__ == "__main__":
    main()
