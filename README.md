# Улучшение качества сжатых изображений (qecnn)

Обучение двух моделей: **с аугментацией** и **без аугментации**. Обе обучаются
на всех тайлах (61 600) из 200 кадров `trainrawyuv/BSD500train.yuv`.

Готовые модели уже лежат в репозитории: `best_model2026_3_aug.pth` — 24.847 дБ,
`best_model2026_3_noaug.pth` — 24.797 дБ, базовая `best_model2026.pth` — 24.738 дБ.

## Перед запуском

В репозитории только код. Положите рядом с кодом:

```
trainrawyuv/BSD500train.yuv
testrawyuv/BSD500val.yuv
yuvcompression/x265.exe
yuvcompression/TAppDecoder.exe
```

Затем сожмите изображения (x265, CRF 35, all-intra) и декодируйте обратно:

```
mkdir traincompyuv testcompyuv
yuvcompression\x265.exe --input-res 480x320 --fps 30 --input trainrawyuv\BSD500train.yuv -o train.hevc --crf 35 --keyint 1 --preset medium
yuvcompression\TAppDecoder.exe -b train.hevc -o traincompyuv\BSD500trainc.yuv
yuvcompression\x265.exe --input-res 480x320 --fps 30 --input testrawyuv\BSD500val.yuv -o val.hevc --crf 35 --keyint 1 --preset medium
yuvcompression\TAppDecoder.exe -b val.hevc -o testcompyuv\BSD500valc.yuv
```

Получатся `traincompyuv/BSD500trainc.yuv` и `testcompyuv/BSD500valc.yuv` — без них
не работают `--prepare`, `--train-*` и `--eval`. 

Для этой же цели отдельно лежат скрипты для windows в `yuvcompression/*.bat`.

Запускать их из корня репозитория (или двойным щелчком):

```
yuvcompression\run_enc_dec_35_BSDintra_train.bat
yuvcompression\run_enc_dec_35_BSDintra_test.bat
```

Первый создаёт `yuvcompression\train.hevc` и `traincompyuv\BSD500trainc.yuv`,
второй — `yuvcompression\val.hevc` и `testcompyuv\BSD500valc.yuv`.
Скрипты сами создают нужные папки и используют пути относительно своего
расположения, поэтому запускать их можно откуда угодно.

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
