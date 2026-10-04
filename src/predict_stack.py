"""
Применяет обученную стекинг-модель (train_stack.py) к готовому
CNN-submission на тесте: добавляет ручные признаки и пересчитывает
финальный прогноз через мета-модель.
"""

import argparse
import logging
import pickle

import numpy as np
import pandas as pd

from train_stack import compute_features_for_df

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack_model", default="models/v1/stack_model.pkl")
    parser.add_argument("--cnn_submission", default="submission.csv",
                         help="CSV с колонками image_id,load_pct — результат predict.py (CNN-ансамбль)")
    parser.add_argument("--images_dir", default="data/images")
    parser.add_argument("--out", default="submission_stacked.csv")
    args = parser.parse_args()

    with open(args.stack_model, "rb") as f:
        payload = pickle.load(f)
    model = payload["model"]
    feature_cols = payload["feature_cols"]

    if not payload.get("use_stack", False):
        logger.warning(
            "Этот stack_model.pkl был обучен с use_stack=False (стекинг НЕ улучшал CNN на OOF: "
            "CNN_MAE=%.3f vs stack_MAE=%.3f). Копирую cnn_submission без изменений в %s.",
            payload.get("cnn_only_mae", float("nan")), payload.get("stack_mae", float("nan")), args.out
        )
        cnn_df = pd.read_csv(args.cnn_submission)
        cnn_df.to_csv(args.out, index=False)
        return

    cnn_df = pd.read_csv(args.cnn_submission)
    cnn_df = cnn_df.rename(columns={"load_pct": "oof_pred"})  # тот же смысл: прогноз CNN

    logger.info("Считаем ручные признаки для тестовых изображений...")
    feat_df = compute_features_for_df(cnn_df, args.images_dir)
    data = cnn_df.merge(feat_df, on="image_id")

    X = data[feature_cols].values
    final_pred = np.clip(model.predict(X), 0, 100)

    out_df = pd.DataFrame({"image_id": data["image_id"], "load_pct": final_pred})
    out_df.to_csv(args.out, index=False)
    logger.info("Сохранено: %s (%d строк)", args.out, len(out_df))


if __name__ == "__main__":
    main()
