"""Смешивание прогнозов из нескольких прогонов (runs/*/oof.csv + test_pred.csv) -> submission.csv.
    python src/ensemble.py --runs runs/v3_cnn runs/n_b2_384 runs/n_b0_s7 --out submission_blend.csv
Три способа: среднее всех, оптимизация весов (Nelder-Mead), жадный отбор (Caruana).
Не-среднее берётся, только если лучше среднего на OOF минимум на --min_gain.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

from common import mae, acc10


def softmax(v):
    e = np.exp(v - v.max())
    return e / e.sum()


def greedy_weights(P, y, iters=30):
    """Жадный отбор с возвратами: каждый шаг добавляет модель, максимально снижающую OOF MAE."""
    n = P.shape[1]
    cur, sel = np.zeros(len(y)), []
    best_m, best_w = 1e9, None
    for t in range(iters):
        cands = [(mae(y, (cur * t + P[:, j]) / (t + 1)), j) for j in range(n)]
        m, j = min(cands)
        cur = (cur * t + P[:, j]) / (t + 1)
        sel.append(j)
        if m < best_m:
            best_m, best_w = m, np.bincount(sel, minlength=n) / len(sel)
    return best_w


def crossfit_iso(b, y, seed):
    out = np.zeros_like(b)
    for tr, va in KFold(5, shuffle=True, random_state=seed).split(b):
        iso = IsotonicRegression(y_min=0, y_max=100, out_of_bounds="clip").fit(b[tr], y[tr])
        out[va] = iso.predict(b[va])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--test_csv", default="data/test.csv")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--min_gain", type=float, default=0.10, help="насколько лучше среднего должен быть способ с весами")
    ap.add_argument("--calibrate", action="store_true", help="изотоническая калибровка смеси (с проверкой кросс-валидацией по OOF)")
    ap.add_argument("--min_gain_cal", type=float, default=0.05)
    a = ap.parse_args()

    oofs = [pd.read_csv(f"{r}/oof.csv").set_index("image_id") for r in a.runs]
    tests = [pd.read_csv(f"{r}/test_pred.csv").set_index("image_id")["load_pct"] for r in a.runs]
    idx = oofs[0].index
    for o in oofs[1:]:
        idx = idx.intersection(o.index)
    y = oofs[0].loc[idx, "load_pct"].values
    P = np.column_stack([o.loc[idx, "oof_pred"].values for o in oofs])

    test_ids = pd.read_csv(a.test_csv)["image_id"].astype(str).tolist()
    T = np.column_stack([t.reindex(test_ids).values for t in tests])
    assert not np.isnan(T).any(), "в test_pred.csv не хватает image_id из test.csv"

    for r, col in zip(a.runs, P.T):
        print(f"{r:<45} OOF MAE={mae(y, col):.3f}  Acc@10={acc10(y, col):.3f}")

    n = P.shape[1]
    res = minimize(lambda v: mae(y, P @ softmax(v)), np.zeros(n), method="Nelder-Mead", options={"maxiter": 3000})
    cands = {
        "mean": np.full(n, 1.0 / n),
        "nelder-mead": softmax(res.x),
        "greedy": greedy_weights(P, y),
    }
    print()
    scores = {}
    for name, w in cands.items():
        scores[name] = mae(y, np.clip(P @ w, 0, 100))
        print(f"{name:<12} MAE={scores[name]:.3f}  Acc@10={acc10(y, np.clip(P @ w, 0, 100)):.3f}  веса={np.round(w, 2).tolist()}")

    best = min(scores, key=scores.get)
    name = best if scores["mean"] - scores[best] >= a.min_gain else "mean"
    final = np.clip(T @ cands[name], 0, 100)
    if a.calibrate:
        b = np.clip(P @ cands[name], 0, 100)
        gains = [mae(y, b) - mae(y, crossfit_iso(b, y, sd)) for sd in range(5)]
        g = float(np.mean(gains))
        print(f"\nизотоническая калибровка: средний выигрыш по 5 разбиениям = {g:+.3f} (мин. {min(gains):+.3f})")
        if g >= a.min_gain_cal and min(gains) > 0:
            iso = IsotonicRegression(y_min=0, y_max=100, out_of_bounds="clip").fit(b, y)
            final = np.clip(iso.predict(final), 0, 100)
            print("калибровка ПРИМЕНЕНА")
        else:
            print("калибровка не применена (выигрыш мал или нестабилен)")
    assert np.isfinite(final).all() and len(final) == len(test_ids)
    pd.DataFrame({"image_id": test_ids, "load_pct": final}).to_csv(a.out, index=False)
    print(f"\nиспользовано: {name}; записано {a.out} ({len(final)} строк)")
    print("(веса подобраны по тем же OOF, поэтому реальный MAE на тесте обычно на ~0.1-0.3 хуже)")


if __name__ == "__main__":
    main()
