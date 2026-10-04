"""Обучение лёгкой головы (Ridge / SVR) на замороженных признаках + честная оценка по GroupKFold.
    python src/fit_head.py --features features/vit_base_patch14_dinov2_448.npz
Пишет runs/head_<tag>/oof.csv и test_pred.csv (формат для ensemble.py).
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from common import load_train_with_folds, mae, acc10


def configs():
    out = []
    for al in (30, 100, 300, 1e3, 3e3, 1e4, 3e4):
        out.append((f"ridge_a{al:g}", lambda al=al: Ridge(alpha=al)))
    for C in (1, 3, 10, 30):
        out.append((f"svr_C{C}", lambda C=C: SVR(kernel="rbf", C=C, epsilon=0.02, gamma="scale")))
    return out


def fit_scaled(make, X, Xf, y):
    Xa = np.vstack([X, Xf])
    sc = StandardScaler().fit(Xa)
    m = make().fit(sc.transform(Xa), np.concatenate([y, y]) / 100.0)
    return sc, m


def predict_tta(sc, m, X, Xf):
    p = (m.predict(sc.transform(X)) + m.predict(sc.transform(Xf))) / 2
    return np.clip(p * 100.0, 0, 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--train_csv", default="data/train.csv")
    ap.add_argument("--groups_csv", default="data/train_groups.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out_root", default="runs")
    a = ap.parse_args()

    z = np.load(a.features, allow_pickle=True)
    df = load_train_with_folds(a.train_csv, a.groups_csv, a.folds)
    pos = {i: k for k, i in enumerate(z["ids_tr"])}
    order = np.array([pos[i] for i in df["image_id"]])
    X, Xf = z["Xtr"][order], z["Xtr_f"][order]
    y, fold = df["load_pct"].values, df["fold"].values

    best = (1e9, None, None)
    print(f"{'config':<16} OOF-MAE  Acc@10")
    for name, make in configs():
        oof = np.zeros(len(y))
        for k in range(a.folds):
            tr, va = fold != k, fold == k
            sc, m = fit_scaled(make, X[tr], Xf[tr], y[tr])
            oof[va] = predict_tta(sc, m, X[va], Xf[va])
        s = mae(y, oof)
        print(f"{name:<16} {s:7.3f}  {acc10(y, oof):.3f}")
        if s < best[0]:
            best = (s, name, (make, oof))

    s, name, (make, oof) = best
    print(f"\nлучший конфиг: {name}  OOF-MAE={s:.3f}  (выбор по OOF даёт лёгкий оптимизм)")

    sc, m = fit_scaled(make, X, Xf, y)  # финальная модель на всех данных
    test_pred = predict_tta(sc, m, z["Xte"], z["Xte_f"])

    tag = os.path.splitext(os.path.basename(a.features))[0]
    out = os.path.join(a.out_root, f"head_{tag}")
    os.makedirs(out, exist_ok=True)
    pd.DataFrame({"image_id": df["image_id"], "load_pct": y, "oof_pred": oof, "fold": fold}).to_csv(
        os.path.join(out, "oof.csv"), index=False)
    pd.DataFrame({"image_id": z["ids_te"], "load_pct": test_pred}).to_csv(
        os.path.join(out, "test_pred.csv"), index=False)
    print("saved ->", out)


if __name__ == "__main__":
    main()
