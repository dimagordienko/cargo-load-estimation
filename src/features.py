"""
Простые ручные признаки поверх изображения — то, на чём построен
baseline (Ridge). CNN на маленьком датасете (716 фото) часто не
дотягивает до Ridge по этим признакам в отдельных случаях, поэтому
комбинация CNN + такие признаки в стекинге обычно снижает MAE
сильнее, чем каждый подход по отдельности.
"""

import cv2
import numpy as np


def extract_heuristic_features(img_bgr: np.ndarray) -> dict:
    """img_bgr — результат load_image_bgr (BGR, uint8)."""
    img = cv2.resize(img_bgr, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # доля "непустых" (не пол/фон) пикселей по адаптивному порогу яркости
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    occupied_ratio = float(np.mean(thresh > 0))

    # плотность краёв — коробки/палеты дают много границ, пустой пол — мало
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))

    # разброс яркости и насыщенности — texture proxy
    brightness_std = float(np.std(gray))
    saturation_mean = float(np.mean(hsv[:, :, 1]))
    saturation_std = float(np.std(hsv[:, :, 1]))

    # энтропия гистограммы яркости — однородный пустой пол имеет низкую энтропию
    hist, _ = np.histogram(gray, bins=32, range=(0, 255), density=True)
    hist = hist + 1e-9
    entropy = float(-np.sum(hist * np.log2(hist)))

    # доля тёмных пикселей (тени от груза/объёмных предметов)
    dark_ratio = float(np.mean(gray < 60))

    return {
        "occupied_ratio": occupied_ratio,
        "edge_density": edge_density,
        "brightness_std": brightness_std,
        "saturation_mean": saturation_mean,
        "saturation_std": saturation_std,
        "entropy": entropy,
        "dark_ratio": dark_ratio,
    }


FEATURE_NAMES = [
    "occupied_ratio", "edge_density", "brightness_std",
    "saturation_mean", "saturation_std", "entropy", "dark_ratio",
]
