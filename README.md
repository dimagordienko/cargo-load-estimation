<div align="center">

# Cargo Load MVP

**Оценка загрузки кузова по фотографии: модель компьютерного зрения возвращает процент заполненности от 0 до 100.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![timm](https://img.shields.io/badge/timm-backbones-green)
![Task](https://img.shields.io/badge/Task-CV%20Regression-blueviolet)
![License](https://img.shields.io/badge/License-MIT-green)

Хакатон PochaTech

</div>

---

## Задача

Почте нужно понимать, насколько реально загружен транспорт перед отправкой. Бумажный отчёт не всегда отражает фактическую картину: машина может выглядеть пустой по нормативам, но быть почти заполненной объёмным грузом.

По фотографии кузова нужно оценить **процент загрузки (`load_pct`, 0-100)**. В данных: фуры 20/10/5 тонн, газели, Лада Ларгус, вагоны, паллеты, контейнеры, мешки и россыпь. Это **регрессия**, а не классификация: модель возвращает число.

## Видео-демонстрация

()


## Подход

- **Backbones из `timm`** (EfficientNet, ConvNeXt, DINOv2 ViT) с transfer learning, разрешение 384-448 px.
- **GeM pooling** и регрессионная голова с выходом в диапазоне 0-100.
- **GroupKFold.** Фото из одной группы не попадают одновременно в train и val, поэтому метрики не завышены из-за утечки данных.
- **OOF-прогнозы** для честной оценки качества и подбора весов ансамбля.
- **Ансамбль** нескольких моделей в итоговый `submission.csv`.
- **Анализ данных:** поиск похожих фото и шума в разметке (`check_neighbors.py`, `label_noise.py`, `diagnose_oof.py`).

```mermaid
flowchart LR
    A[Фото кузова] --> B[Аугментации<br/>dataset.py]
    B --> C[Backbone timm + GeM<br/>model.py]
    C --> D[Регрессия load_pct 0-100]
    D --> E[GroupKFold + OOF<br/>train.py]
    E --> F[Ансамбль<br/>ensemble.py]
    F --> G[submission.csv]
```

## Метрики

- **MAE**: средняя абсолютная ошибка в процентных пунктах.
- **Accuracy@10**: доля предсказаний с отклонением не более 10 п.п.

## Технологии и навыки

| Область | Что использовано |
|---|---|
| **Computer Vision** | PyTorch, timm, transfer learning, аугментации, GeM pooling |
| **ML-процесс** | GroupKFold, OOF-прогнозы, MAE / Acc@10, ансамблирование |
| **Анализ данных** | Поиск похожих изображений, диагностика шума разметки |
| **Эксперименты** | Прогоны разных архитектур, shell-скрипты для пакетного запуска |

## Структура репозитория

```text
.
├── src/
│   ├── train.py             # обучение по фолдам, чекпоинты, OOF
│   ├── dataset.py           # датасет и аугментации
│   ├── model.py             # backbone + GeM + регрессор
│   ├── metrics.py           # MAE, Accuracy@10
│   ├── common.py            # общие утилиты
│   ├── ensemble.py          # ансамблирование предсказаний
│   ├── predict.py           # предсказание на тесте
│   ├── predict_dir.py       # предсказание для папки с фото
│   └── ...                  # эксперименты и диагностика
├── assets/                  # демо (GIF, скриншоты)
├── requirements.txt
├── .gitignore
└── README.md
```

> Данные, веса моделей и артефакты обучения в git не хранятся. Их нужно добавить локально.

## Быстрый старт

**1. Установка**
```bash
git clone https://github.com/<user>/cargo-load-mvp.git
cd cargo-load-mvp
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Данные**

```text
data/
├── train.csv
├── train_groups.csv
├── test.csv
└── images/
```
Другой путь можно задать параметрами командной строки.

**3. Обучение**
```bash
python src/train.py \
  --train_csv data/train.csv \
  --groups_csv data/train_groups.csv \
  --images_dir data/images \
  --out_dir models/v1 \
  --backbone efficientnet_b0 \
  --epochs 25 \
  --folds 5
```

**4. Ансамбль нескольких прогонов**
```bash
python src/ensemble.py \
  --runs runs/model_a runs/model_b \
  --test_csv data/test.csv \
  --out submission.csv
```

## Ограничения и планы

- [ ] REST API (FastAPI) для приёма фото и возврата `load_pct`.
- [ ] Оценка неопределённости: показывать, когда модель не уверена.
- [ ] Лёгкая модель для быстрого инференса.
- [ ] Больше типов кузовов и условий съёмки в обучающей выборке.
