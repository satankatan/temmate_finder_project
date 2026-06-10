# Teammate Finder — нейросетевая система семантического матчмейкинга

Поиск идеального тиммейта по текстовому описанию стиля игры на основе **собственной нейросети** (BiLSTM + Attention) и **размеченного датасета** с геймерским сленгом.

## Что изменилось (v2)

| Компонент | Было | Стало |
|-----------|------|-------|
| Модель | Sentence-BERT (предобученная) | Собственная LSTM + Attention, обученная на игровых данных |
| Датасет | 30 пар для оценки | `data/gaming_compatibility_dataset.csv` — 50+ размеченных пар |
| Препроцессор | Базовая замена сленга | Расширенный словарь + лемматизация (опционально) |
| UI | Простая форма | Тёмный игровой интерфейс с панелью статуса модели |

## Архитектура модели

```
Текст → Препроцессор (сленг, стоп-слова) → Embedding → BiLSTM → Attention → FC → L2-нормализация (128 dim)
```

Обучение: **Triplet Loss** на парах с `compatibility_score >= 0.7` (positive) и `<= 0.45` (negative).

## Быстрый старт

```bash
# 1. Зависимости
pip install -r requirements.txt

# 2. Разделение на train / test (честная оценка)
python scripts/split_dataset.py

# 3. Обучение только на train.csv
python -m model.train --dataset data/train.csv --epochs 30

# 4. Метрики на test.csv (пары, не виденные при обучении)
python scripts/evaluate_custom_model.py --data data/test.csv

# 3. Запуск API
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 4. Открыть интерфейс
# teammate_finder.html в браузере

# 5. (опционально) Заполнить БД тестовыми профилями
python scripts/seed_database.py
```

## API

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/health` | Статус БД и модели |
| GET | `/model/info` | Информация о собственной модели |
| POST | `/users` | Добавить профиль |
| POST | `/similarity/search` | Поиск тиммейтов |
| GET | `/stats` | Статистика |

## Датасет

Файл: `data/gaming_compatibility_dataset.csv`

Колонки:
- `description1`, `description2` — пары описаний стиля
- `compatibility_score` — экспертная оценка совместимости (0–1)
- `game_type` — dota2, csgo, valorant, general

Исходный датасет из отчёта (`sample_gaming_data.csv`) включён и расширен парами со сленгом.

## Структура проекта

```
api/main.py              — FastAPI
model/network.py         — TextEmbeddingModel (LSTM + Attention)
model/train.py           — обучение на датасете
model/inference.py       — инференс собственной модели
model/dataset.py         — загрузка размеченных пар
preprocessing/           — препроцессор игрового сленга
data/                    — размеченный датасет
models/                  — best_model.pth, vocab.json
teammate_finder.html     — веб-интерфейс
```

## Для отчёта

В тексте отчёта замените упоминания Sentence-BERT на:
- собственную архитектуру Embedding + BiLSTM + Attention
- обучение Triplet Loss на размеченном датасете `gaming_compatibility_dataset.csv`
- препроцессор с нормализацией геймерского сленга
