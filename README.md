# OCR Evaluation — практическая часть НИР

Сравнительная оценка трёх современных OCR-моделей на научных
документах (OmniDocBench, страницы типа `academic_literature`).

Репозиторий: <https://github.com/AStrateg2509/ocr_eval>

## Модели

| Модель | HuggingFace | Чем интересна |
|---|---|---|
| LightOnOCR-2-1B | `lightonai/LightOnOCR-2-1B` | Компактная (~1B), быстрая |
| DeepSeek-OCR | `deepseek-ai/DeepSeek-OCR` | grounding-промпт, markdown-вывод |
| olmOCR | `allenai/olmOCR-7B-0225-preview` | Qwen2-VL-7B, обучен на 250K страницах |

> mPlug-DocOwl и MonkeyOCR были в первой версии — исключены из сравнения.
> Их `configs/*.yaml` оставлены как deprecated-стабы; удалить через
> `bash cleanup.sh`.

## Метрики

Считаются на следующем шаге (модуль `metrics/` появится позже):

| Метрика | Что измеряет |
|---|---|
| Overall | агрегированный score |
| Text | F1 / нормализованный edit distance по тексту |
| Table | TEDS (Tree Edit-Distance Similarity на HTML-таблицах) |
| IoU | средний IoU bbox-ов layout |
| Recall@5 | доля GT-блоков, попавших в топ-5 предсказаний |

## Структура репозитория

```
ocr_eval/
├── requirements.txt              ← версии зафиксированы (torch 2.6.0, transformers 4.47.0, ...)
├── README.md
├── cleanup.sh                    ← удалить deprecated-файлы первой итерации
├── configs/
│   ├── lightonocr.yaml
│   ├── deepseek_ocr.yaml
│   ├── olmocr.yaml               ← P100-tuning: float16, sdpa, max_image_long_side
│   ├── mplug_docowl.yaml         ← DEPRECATED
│   └── monkeyocr.yaml            ← DEPRECATED
├── src/
│   ├── dataset_loader.py         ← OmniDocBench loader, chunked download, file-based verify
│   ├── utils.py                  ← конфиги, JSONL с резюме, GPU helpers, Timer
│   └── io_records.py             ← формат записей предсказаний
├── notebooks/
│   ├── 00_kaggle_full_pipeline.ipynb   ← ЕДИНЫЙ ноутбук для Kaggle
│   ├── 01..05_*.ipynb            ← DEPRECATED стабы
└── data/                         ← скачанный OmniDocBench + subset.json
└── results/
    ├── lightonocr/predictions.jsonl
    ├── deepseek_ocr/predictions.jsonl
    └── olmocr/predictions.jsonl
```

## Как запускать в Kaggle

1. **Создать секрет.** `Add-ons → Secrets → New Secret`, имя `HF_TOKEN`,
   значение — read-токен с huggingface.co (Settings → Access Tokens).
2. **Создать новый ноутбук**, accelerator: `GPU P100`. (T4×2 тоже подойдёт,
   но в коде стоит ставить `device_map="auto"`.)
3. **Импортировать `notebooks/00_kaggle_full_pipeline.ipynb`** —
   File → Import Notebook → upload файл из репозитория.
4. **Run All.** Ноутбук:
   * клонирует репо;
   * ставит зависимости + poppler;
   * получает HF_TOKEN из Kaggle Secrets;
   * скачивает OmniDocBench JSON;
   * качает ~1653 картинки **чанками по 200** с file-based верификацией
     (обходим лимит ~1000 запросов/мин на HF Hub);
   * собирает 100-страничный subset (academic_literature, en, seed=42);
   * последовательно прогоняет 3 модели (LightOnOCR → DeepSeek-OCR → olmOCR);
   * сводит выходы в pandas-таблицу.

> **Время выполнения** на P100, subset_size=100:
> чанк-загрузка картинок ~5–10 мин · LightOnOCR ~3 мин · DeepSeek ~10 мин ·
> olmOCR ~25–35 мин (7B, без flash-attn). Всего ~50–60 мин.

## Что починили во второй итерации

* `dataset_loader` — `page_attribute` теперь читается как вложенный объект,
  `page_type` берётся из `data_source`, к `image_path` добавляется
  префикс `images/`.
* **Чанк-загрузка** датасета с file-based verify — обход HF rate-limit.
* **DeepSeek-OCR**: `save_results=True` + чтение `result.mmd` с диска
  (раньше ничего не сохранялось, потому что метод не возвращает строку).
* **olmOCR** под P100 — `float16`, `attn_implementation="sdpa"`,
  ресайз картинок до `max_image_long_side`.
* **LightOnOCR-2-1B** заменил mPlug-DocOwl.
* **Один Kaggle-ноутбук** вместо пяти разрозненных.
* **HF auth через Kaggle Secrets** добавлен в setup-ячейку.
* В `requirements.txt` зафиксированы версии (`torch==2.6.0`,
  `transformers==4.47.0`, `torchaudio==2.6.0`, `addict~=2.4.0`).

## Воспроизводимость

* `subset.json` фиксирует точные 100 страниц (seed=42).
* Все конфиги под версионным контролем.
* Метрики детерминированные: `do_sample=False` (LightOn, DeepSeek), `temperature=0.0` (olmOCR).

## Что дальше

После того как все 3 модели отработают:

1. Модули подсчёта метрик: Text / Table / IoU / Recall@5 / Overall.
2. Сводный CSV `results/summary.csv`.
3. Графики: bar chart, radar chart, error analysis.
4. Раздел в .docx-отчёте НИР (установка → датасет → пайплайн → результаты → выводы).
