"""
Разово ресайзит все фото до размера, немного большего чем image_size
обучения (запас для RandomResizedCrop), и сохраняет в отдельную папку.
Дальше train.py читает уже маленькие файлы -> декодирование JPEG
перестаёт быть узким местом.

Обычно даёт основной прирост скорости, если исходные фото 3000x4000+ -
декодирование одного такого файла на CPU может занимать 100-300ms,
что при 716 фото x много эпох выливается в те самые "часы" вместо
"20 минут".

Запуск (один раз перед обучением):
    python src/cache_resize.py --src data/images --dst data/images_cached --size 512
    python src/cache_resize.py --src data/test_images --dst data/test_images_cached --size 512

Дальше в train.py/predict.py используйте --images_dir data/images_cached
"""

import os
import argparse
import logging

import cv2
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="папка с оригинальными фото")
    parser.add_argument("--dst", required=True, help="куда сохранить уменьшенные копии")
    parser.add_argument("--size", type=int, default=512,
                         help="целевая длинная сторона (берите чуть больше image_size обучения)")
    parser.add_argument("--quality", type=int, default=90)
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    files = [f for f in os.listdir(args.src) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    logger.info("Найдено %d файлов в %s", len(files), args.src)

    n_ok, n_fail = 0, 0
    for fname in tqdm(files):
        src_path = os.path.join(args.src, fname)
        dst_path = os.path.join(args.dst, os.path.splitext(fname)[0] + ".jpg")

        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Не удалось прочитать: %s", src_path)
            n_fail += 1
            continue

        h, w = img.shape[:2]
        scale = args.size / max(h, w)
        if scale < 1.0:  # уменьшаем только если фото больше целевого размера
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        cv2.imwrite(dst_path, img, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        n_ok += 1

    logger.info("Готово: %d обработано, %d ошибок. Результат в %s", n_ok, n_fail, args.dst)


if __name__ == "__main__":
    main()
