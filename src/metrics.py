import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Основная метрика задачи: mean(abs(эталон - прогноз))."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def accuracy_at_10(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Доля прогнозов с абсолютной ошибкой <= 10 п.п. (справочный показатель, не метрика отбора)."""
    err = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    return float(np.mean(err <= 10.0))
