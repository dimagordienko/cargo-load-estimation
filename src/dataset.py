"""
Dataset для оценки загрузки транспорта по фото.

Ожидаемая структура данных (как в train.zip из задания):
    data/
        train.csv          # колонки: image_id, load_pct
        train_groups.csv   # колонки: image_id, group_id
        images/
            <image_id>.jpg

Изображение может отсутствовать/быть битым — такие строки логируются
и пропускаются при построении датасета (в проде это соответствует
ветке "неподдерживаемое изображение" в API).
"""

import os
import logging
from typing import Optional, Callable

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

IMAGE_SIZE = 384  # входной размер сети; EfficientNet-B0 нормально работает и с 384


def load_image_bgr(path: str) -> Optional[np.ndarray]:
    """Безопасное чтение изображения. Возвращает None, если файл битый/не читается."""
    if not os.path.exists(path):
        logger.warning("Файл не найден: %s", path)
        return None
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # cv2 сам вернёт None на битых файлах
    if img is None:
        logger.warning("Не удалось декодировать изображение: %s", path)
        return None
    if img.ndim != 3 or img.shape[2] != 3:
        logger.warning("Неожиданная размерность изображения %s: %s", path, img.shape)
        return None
    return img


def build_valid_dataframe(csv_path: str, images_dir: str) -> pd.DataFrame:
    """
    Читает train.csv и отфильтровывает записи с отсутствующими/битыми фото.
    Это и есть "готовый датасет" — чистый список (image_id, load_pct, path).
    """
    df = pd.read_csv(csv_path)
    assert {"image_id", "load_pct"}.issubset(df.columns), \
        f"Ожидались колонки image_id, load_pct, получено: {df.columns.tolist()}"

    df["path"] = df["image_id"].astype(str).apply(
        lambda x: os.path.join(images_dir, f"{x}.jpg")
    )

    valid_mask = []
    for p in df["path"]:
        img = load_image_bgr(p)
        valid_mask.append(img is not None)
    n_before = len(df)
    df = df[valid_mask].reset_index(drop=True)
    n_after = len(df)
    if n_after < n_before:
        logger.warning("Отброшено %d/%d битых/отсутствующих изображений", n_before - n_after, n_before)

    # целевая переменная должна лежать в [0, 100]
    out_of_range = ((df["load_pct"] < 0) | (df["load_pct"] > 100)).sum()
    if out_of_range:
        logger.warning("%d значений load_pct вне диапазона [0,100], клиппинг", out_of_range)
        df["load_pct"] = df["load_pct"].clip(0, 100)

    return df


def merge_groups(df: pd.DataFrame, groups_csv: str) -> pd.DataFrame:
    """Присоединяет group_id из train_groups.csv для GroupKFold-валидации."""
    groups = pd.read_csv(groups_csv)
    assert {"image_id", "group_id"}.issubset(groups.columns), \
        f"Ожидались колонки image_id, group_id, получено: {groups.columns.tolist()}"
    df = df.merge(groups, on="image_id", how="left")
    missing = df["group_id"].isna().sum()
    if missing:
        # если группа не найдена — считаем фото отдельной группой (не должно теряться при сплите)
        logger.warning("%d фото без group_id, назначаем уникальные группы", missing)
        max_group = df["group_id"].max()
        max_group = 0 if pd.isna(max_group) else max_group
        fill_values = np.arange(max_group + 1, max_group + 1 + missing)
        df.loc[df["group_id"].isna(), "group_id"] = fill_values
    return df


def get_train_transforms(image_size: int = IMAGE_SIZE):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(image_size, image_size, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.75, 1.0), ratio=(0.85, 1.15), p=0.7),
        A.HorizontalFlip(p=0.5),
        A.Perspective(scale=(0.02, 0.06), p=0.3),
        A.Affine(rotate=(-10, 10), translate_percent=(0.0, 0.05), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=15, p=0.4),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=5),
        ], p=0.2),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.05, 0.15),
                        hole_width_range=(0.05, 0.15), p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = IMAGE_SIZE):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(image_size, image_size, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class CargoLoadDataset(Dataset):
    """
    df ожидает колонки: path, load_pct (и опционально image_id для отладки).
    transform — albumentations.Compose, принимает image=np.ndarray (RGB).
    """

    def __init__(self, df: pd.DataFrame, transform: Optional[Callable] = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = load_image_bgr(row["path"])
        if img is None:
            # подстраховка на случай гонки с файловой системой; в норме сюда не попадём,
            # т.к. build_valid_dataframe уже отфильтровал битые файлы
            img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        target = np.float32(row["load_pct"])
        return img, target
