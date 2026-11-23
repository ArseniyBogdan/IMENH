# Улучшение качества сжатых изображений (qecnn)

Обучение двух моделей: **с аугментацией** и **без аугментации**. Обе обучаются
на всех тайлах (61 600) из 200 кадров `trainrawyuv/BSD500train.yuv`.

## Установка

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Версии зафиксированы в `requirements.txt`. Для CUDA-сборки torch (обучение быстрее)
поставьте его отдельно до остальных пакетов:
`.venv\Scripts\pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121`.

## Команды

| Флаг | Что делает |
|---|---|
| `--prepare` | нарезать YUV в PNG-тайлы |
| `--train-without-augment` | обучить модель без аугментации |
| `--train-with-augment` | обучить модель с аугментацией |
| `--eval` | посчитать метрики моделей |

## Обучение

Нарезка (один раз, ~5–8 мин, ~1.5 ГБ):

```
python qecnn.py --prepare
```

Модель без аугментации:

```
python qecnn.py --train-without-augment
```

Модель с аугментацией:

```
python qecnn.py --train-with-augment
```

Результат: `best_model2026_3_noaug.pth` / `best_model2026_3_aug.pth`.

## Метрики

```
python qecnn.py --eval
```

Считает PSNR обеих моделей, печатает значения и обновляет
`results_3_noaug.json` и `results_3_aug.json`.

## Время

Бюджет обучения — 3000 с на модель (`max_training_seconds` в `DEFAULT_PARAMS`):
на GPU ~50 минут, на CPU дольше.

Готовые модели уже лежат в репозитории: `best_model2026_3_aug.pth` — 24.847 дБ,
`best_model2026_3_noaug.pth` — 24.797 дБ, базовая `best_model2026.pth` — 24.738 дБ.
