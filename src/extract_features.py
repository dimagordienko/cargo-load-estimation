"""Быстрый эксперимент: ЗАМОРОЖЕННЫЙ предобученный бэкбон -> признаки -> лёгкая голова (fit_head.py).
Обучения нет, поэтому на M3 это минуты, а не часы. Даёт быстрый ответ, какое семейство моделей лучше.

    python src/extract_features.py --model vit_base_patch14_dinov2.lvd142m --size 448
    python src/extract_features.py --model vit_small_patch14_dinov2.lvd142m --size 448
    python src/extract_features.py --model convnext_small.fb_in22k_ft_in1k --size 384

Признаки: среднее по карте + сетка 2x2 (сохраняет "где стоит груз") + CLS (для ViT). Плюс версия с hflip.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
from tqdm import tqdm

from common import read_rgb


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_model(name, size):
    try:
        m = timm.create_model(name, pretrained=True, num_classes=0, img_size=size)
    except TypeError:  # CNN не принимают img_size
        m = timm.create_model(name, pretrained=True, num_classes=0)
    return m.eval()


@torch.no_grad()
def encode(model, x):
    f = model.forward_features(x)
    cls = None
    if f.ndim == 3:  # ViT: (B, N, C)
        npref = getattr(model, "num_prefix_tokens", 1)
        cls = f[:, 0]
        tok = f[:, npref:]
        h = int(round(tok.shape[1] ** 0.5))
        f = tok.transpose(1, 2).reshape(tok.shape[0], -1, h, h)
    parts = [f.mean((2, 3)), F.adaptive_avg_pool2d(f, 2).flatten(1)]
    if cls is not None:
        parts.append(cls)
    return torch.cat(parts, 1)


def run(model, paths, size, mean, std, device, bs):
    mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=device).view(1, 3, 1, 1)
    A, B = [], []
    for i in tqdm(range(0, len(paths), bs)):
        imgs = np.stack([read_rgb(p, size) for p in paths[i:i + bs]])
        x = torch.from_numpy(imgs).to(device).permute(0, 3, 1, 2).float() / 255.0
        x = (x - mean) / std
        A.append(encode(model, x).float().cpu().numpy())
        B.append(encode(model, torch.flip(x, dims=[3])).float().cpu().numpy())
    return np.concatenate(A), np.concatenate(B)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--train_csv", default="data/train.csv")
    ap.add_argument("--test_csv", default="data/test.csv")
    ap.add_argument("--images_dir", default="data/images")
    ap.add_argument("--test_images_dir", default=None, help="по умолчанию = images_dir")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out_dir", default="features")
    a = ap.parse_args()

    device = get_device()
    print("device:", device)
    model = make_model(a.model, a.size).to(device)
    cfg = timm.data.resolve_data_config({}, model=model)
    mean, std = cfg["mean"], cfg["std"]

    ids_tr = pd.read_csv(a.train_csv)["image_id"].astype(str).tolist()
    ids_te = pd.read_csv(a.test_csv)["image_id"].astype(str).tolist()
    tdir = a.test_images_dir or a.images_dir
    p_tr = [os.path.join(a.images_dir, f"{i}.jpg") for i in ids_tr]
    p_te = [os.path.join(tdir, f"{i}.jpg") for i in ids_te]

    Xtr, Xtr_f = run(model, p_tr, a.size, mean, std, device, a.batch_size)
    Xte, Xte_f = run(model, p_te, a.size, mean, std, device, a.batch_size)

    os.makedirs(a.out_dir, exist_ok=True)
    tag = f"{a.model.split('.')[0]}_{a.size}"
    out = os.path.join(a.out_dir, f"{tag}.npz")
    np.savez_compressed(out, ids_tr=np.array(ids_tr), ids_te=np.array(ids_te),
                        Xtr=Xtr, Xtr_f=Xtr_f, Xte=Xte, Xte_f=Xte_f)
    print("saved", out, "dim =", Xtr.shape[1])


if __name__ == "__main__":
    main()
