"""Есть ли у тестовых фото почти-дубликаты в train? + kNN-базовая линия. Секунды, только numpy.
Работает с ViT-признаками (DINOv2), берёт CLS-часть.
    python src/check_neighbors.py --features features/vit_base_patch14_dinov2_448.npz
"""
import argparse

import numpy as np

from common import load_train_with_folds, mae


def l2n(a):
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--train_csv", default="data/train.csv")
    ap.add_argument("--groups_csv", default="data/train_groups.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()

    z = np.load(a.features, allow_pickle=True)
    df = load_train_with_folds(a.train_csv, a.groups_csv, a.folds)
    pos = {i: k for k, i in enumerate(z["ids_tr"])}
    order = np.array([pos[i] for i in df["image_id"]])
    C = z["Xtr"].shape[1] // 6  # avg(C) + сетка 2x2 (4C) + CLS(C)
    Xtr = l2n(z["Xtr"][order][:, -C:])
    Xte = l2n(z["Xte"][:, -C:])
    g = np.asarray(df["group_id"].tolist())
    y, fold = df["load_pct"].to_numpy(), df["fold"].to_numpy()

    S = Xtr @ Xtr.T
    np.fill_diagonal(S, -1)
    same = g[:, None] == g[None, :]
    s_same = np.where(same, S, -1).max(1)[same.sum(1) > 1]
    s_other = np.where(~same, S, -1).max(1)
    s_test = (Xte @ Xtr.T).max(1)

    q = lambda v: np.percentile(v, [10, 50, 90]).round(3).tolist()
    print("косинусная близость ближайшего соседа (10%, 50%, 90% квантили):")
    print(f"  train -> та же группа:        {q(s_same)}   (n={len(s_same)})")
    print(f"  train -> другая группа:       {q(s_other)}")
    print(f"  test  -> ближайший из train:  {q(s_test)}")
    print("Если test похож на 'та же группа' - в train есть почти-дубликаты тестовых фото.")
    print("Если test похож на 'другая группа' - тест новый, GroupKFold-оценка честная.")

    oof = np.zeros(len(y))
    for f in np.unique(fold):
        va, tr = fold == f, fold != f
        idx = np.argsort(-(Xtr[va] @ Xtr[tr].T), axis=1)[:, :a.k]
        oof[va] = y[tr][idx].mean(1)
    print(f"\nkNN (k={a.k}, по группам) OOF MAE = {mae(y, oof):.3f}")


if __name__ == "__main__":
    main()
