"""Оценка "потолка" MAE из-за шума разметки: у почти одинаковых фото метки должны совпадать.
Если у визуально почти идентичных фото метки заметно расходятся, ниже некоторого MAE не спуститься никакой моделью.
Для гауссова шума: MAE_пол ~ mean|d| / sqrt(2). Оценка ПЕСИМИСТИЧНАЯ: похожие снимки могут иметь реально разную загрузку.
    python src/label_noise.py --features features/vit_base_patch14_dinov2_448.npz
"""
import argparse

import numpy as np

from common import load_train_with_folds


def l2n(a):
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--train_csv", default="data/train.csv")
    ap.add_argument("--groups_csv", default="data/train_groups.csv")
    a = ap.parse_args()

    z = np.load(a.features, allow_pickle=True)
    df = load_train_with_folds(a.train_csv, a.groups_csv, 5)
    pos = {i: k for k, i in enumerate(z["ids_tr"])}
    order = np.array([pos[i] for i in df["image_id"]])
    C = z["Xtr"].shape[1] // 6
    X = l2n(z["Xtr"][order][:, -C:])
    y = df["load_pct"].to_numpy()

    S = X @ X.T
    iu = np.triu_indices(len(y), 1)
    sims, d = S[iu], np.abs(y[iu[0]] - y[iu[1]])
    print(f"всего пар: {len(sims)}; средняя |разница меток| у случайной пары: {d.mean():.1f}\n")
    print(f"{'сходство >=':<12}{'пар':>8}{'mean|d|':>10}{'median|d|':>11}{'~MAE-пол':>10}")
    for thr in (0.90, 0.93, 0.95, 0.97, 0.98):
        m = sims >= thr
        if m.sum() < 5:
            print(f"{thr:<12}{int(m.sum()):>8}   мало пар")
            continue
        print(f"{thr:<12}{int(m.sum()):>8}{d[m].mean():>10.1f}{np.median(d[m]):>11.1f}{d[m].mean() / np.sqrt(2):>10.1f}")


if __name__ == "__main__":
    main()
