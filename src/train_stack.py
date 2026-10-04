"""
Стекинг второго уровня — МАКСИМАЛЬНО ПРОСТОЙ и с автозащитой от переобучения.

Урок из первого прогона: GradientBoosting с 8 признаками на 716 фото
переобучается на шуме эвристических признаков и УХУДШАЕТ результат.
Здесь: Ridge с сильной регуляризацией, всего 1 доп. признак
(occupied_ratio - самый информативный), и скрипт сам сравнивает со
чистым CNN и явно говорит, помогает стекинг или нет - используйте
стекинг только если он реально победил в выводе лога.

Обучение + оценка на train:
    python src/train_stack.py \
        --oof_csv models/v3/oof_predictions.csv \
        --images_dir data/images \
        --out_model models/v3/stack_model.pkl
"""

import os
import argparse
import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from dataset import load_image_bgr
from features import extract_heuristic_features
from metrics import mae, accuracy_at_10

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# берём ТОЛЬКО самый сильный признак - меньше признаков = меньше риск переобучения на 716 фото
STACK_FEATURES = ["occupied_ratio"]


def compute_features_for_df(df: pd.DataFrame, images_dir: str) -> pd.DataFrame:
    rows = []
    for image_id in df["image_id"]:
        path = os.path.join(images_dir, f"{image_id}.jpg")
        img = load_image_bgr(path)
        if img is None:
            feats = {name: 0.0 for name in STACK_FEATURES}
        else:
            full_feats = extract_heuristic_features(img)
            feats = {name: full_feats[name] for name in STACK_FEATURES}
        feats["image_id"] = image_id
        rows.append(feats)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof_csv", default="models/v3/oof_predictions.csv")
    parser.add_argument("--images_dir", default="data/images")
    parser.add_argument("--out_model", default="models/v3/stack_model.pkl")
    args = parser.parse_args()

    oof_df = pd.read_csv(args.oof_csv)
    logger.info("OOF-предсказаний загружено: %d", len(oof_df))

    logger.info("Считаем %s по изображениям...", STACK_FEATURES)
    feat_df = compute_features_for_df(oof_df, args.images_dir)
    data = oof_df.merge(feat_df, on="image_id")

    X_cols = ["oof_pred"] + STACK_FEATURES
    X = data[X_cols].values
    y = data["load_pct"].values
    fold = data["fold"].values

    oof_stack_pred = np.zeros_like(y, dtype=float)

    for f in np.unique(fold):
        train_mask = fold != f
        val_mask = fold == f
        # RidgeCV сам подбирает силу регуляризации по внутренней CV - устойчивее к переобучению,
        # чем фиксированный GBM с кучей деревьев
        meta_model = RidgeCV(alphas=np.logspace(-2, 3, 20))
        meta_model.fit(X[train_mask], y[train_mask])
        oof_stack_pred[val_mask] = meta_model.predict(X[val_mask])

    oof_stack_pred = np.clip(oof_stack_pred, 0, 100)

    cnn_only_mae = mae(y, data["oof_pred"].values)
    stack_mae = mae(y, oof_stack_pred)

    logger.info("MAE только CNN (OOF):      %.3f", cnn_only_mae)
    logger.info("MAE стекинг CNN+признак:   %.3f", stack_mae)

    USE_STACK = stack_mae < cnn_only_mae - 0.05  # небольшой запас, чтобы не ловиться на шум
    if USE_STACK:
        logger.info("Стекинг РЕАЛЬНО улучшил результат (%.3f -> %.3f). Используйте stack_model.pkl",
                    cnn_only_mae, stack_mae)
    else:
        logger.warning("Стекинг НЕ улучшил результат на этих данных. "
                        "НЕ используйте stack_model.pkl - берите чистый CNN-ансамбль.")

    final_model = RidgeCV(alphas=np.logspace(-2, 3, 20))
    final_model.fit(X, y)

    with open(args.out_model, "wb") as f:
        pickle.dump({
            "model": final_model,
            "feature_cols": X_cols,
            "use_stack": USE_STACK,  # predict_stack.py проверит этот флаг
            "cnn_only_mae": cnn_only_mae,
            "stack_mae": stack_mae,
        }, f)
    logger.info("Сохранено: %s (use_stack=%s)", args.out_model, USE_STACK)


if __name__ == "__main__":
    main()

