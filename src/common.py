"""Общие утилиты: фолды (одинаковые во ВСЕХ скриптах), метрики, чтение картинок."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def load_train_with_folds(train_csv="data/train.csv", groups_csv="data/train_groups.csv", n_folds=5):
    """Возвращает df: image_id, load_pct, group_id, fold. Сортировка по image_id => разбиение
    детерминировано и одинаково для fit_head.py / train_v4.py, поэтому OOF-прогнозы можно смешивать."""
    df = pd.read_csv(train_csv)
    g = pd.read_csv(groups_csv)[["image_id", "group_id"]]
    df = df.merge(g, on="image_id", how="left")
    miss = df["group_id"].isna()
    if miss.any():
        df.loc[miss, "group_id"] = [f"solo_{i}" for i in range(int(miss.sum()))]
    df["group_id"] = df["group_id"].astype(str)
    df["load_pct"] = df["load_pct"].clip(0, 100)
    df = df.sort_values("image_id").reset_index(drop=True)
    df["fold"] = -1
    for k, (_, va) in enumerate(GroupKFold(n_splits=n_folds).split(df, groups=df["group_id"])):
        df.loc[va, "fold"] = k
    assert (df["fold"] >= 0).all()
    return df


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def acc10(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float)) <= 10))


def read_rgb(path, size, width=None):
    """RGB uint8, высота=size, ширина=width (по умолчанию = size, т.е. квадрат). Без чёрных полей, ничего не режем.
    Битый файл -> чёрный кадр + предупреждение (индексы не должны съезжать)."""
    import cv2
    h, w = size, (width or size)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] не читается: {path}")
        return np.zeros((h, w, 3), np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
