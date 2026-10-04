"""Разбор OOF-ошибок: где именно модель теряет MAE и есть ли шум в разметке.
    python src/diagnose_oof.py --oof models/v3/oof_predictions.csv
"""
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="models/v3/oof_predictions.csv")
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args()

    df = pd.read_csv(a.oof)
    df["pred"] = df["oof_pred"].clip(0, 100)
    df["err"] = df["pred"] - df["load_pct"]
    df["ae"] = df["err"].abs()

    print(f"n={len(df)}  MAE={df.ae.mean():.3f}  медиана |err|={df.ae.median():.3f}  Acc@10={(df.ae <= 10).mean():.3f}")
    print("\nпо фолдам:")
    print(df.groupby("fold")["ae"].agg(n="size", mae="mean").round(3))

    df["bin"] = pd.cut(df["load_pct"], [-0.1, 5, 15, 30, 50, 70, 90, 100])
    print("\nпо диапазону истинной загрузки (bias = pred - true):")
    print(df.groupby("bin", observed=True).agg(n=("ae", "size"), mae=("ae", "mean"), bias=("err", "mean")).round(2))

    print("\nвклад грубых ошибок:")
    for thr in (15, 20, 30):
        m = df.ae > thr
        print(f"  |err|>{thr}: {m.mean() * 100:.1f}% фото дают {df.ae[m].sum() / df.ae.sum() * 100:.0f}% суммарной ошибки")
    s = df.ae.sort_values().values
    for drop in (0.03, 0.05, 0.10):
        k = int(len(s) * (1 - drop))
        print(f"  MAE без худших {drop * 100:.0f}% фото: {s[:k].mean():.3f}")

    print("\nчастые значения разметки:")
    print(df["load_pct"].value_counts().head(12).to_string())

    print(f"\nтоп-{a.top} худших (посмотрите эти фото глазами: ошибка модели или шум разметки?):")
    print(df.sort_values("ae", ascending=False).head(a.top)[["image_id", "load_pct", "pred", "fold"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
