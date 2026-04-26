# OCR Evaluation — практическая часть НИР

Сравнительная оценка четырёх современных OCR-моделей на научных
документах (OmniDocBench, страницы типа `academic_literature`).

## Модели

| Модель | HuggingFace | Чем интересна |
|---|---|---|
| DeepSeek-OCR | `deepseek-ai/DeepSeek-OCR` | grounding-промпт, markdown-вывод |
| mPLUG-DocOwl 2 | `mPLUG/DocOwl2` | shape-adaptive cropping для high-res |
| olmOCR (AllenAI) | `allenai/olmOCR-7B-0225-preview` | Qwen2-VL-7B, обучен на 250K страницах |
| MonkeyOCR | `echo840/MonkeyOCR` | Structure → Recognition → Relation (SRR) |

## Метрики

Считаются на следующем шаге (модуль `metrics/` — будет добавлен после
прогона инференса):

| Метрика | Что измеряет |
|---|---|
| Overall | агрегированный score (нормализованная сумма остальных) |
| Text | F1 / нормализованный edit distance по тексту |
| Table | TEDS (Tree Edit-Distance Similarity на HTML-таблицах) |
| IoU | средний IoU bbox-ов layout |
| Recall@5 | доля GT-блоков, попавших в топ-5 предсказаний |

## Структура проекта

```
ocr_eval/
├── requirements.txt              ← зависимости для Colab/Kaggle
├── README.md                     ← этот файл
├── configs/
│   ├── deepseek_ocr.yaml
│   ├── mplug_docowl.yaml
│   ├── olmocr.yaml
│   └── monkeyocr.yaml
├── src/
│   ├── dataset_loader.py         ← OmniDocBench loader
│   ├── utils.py                  ← конфиги, JSONL, GPU, время
│   └── io_records.py             ← формат записей предсказаний
├── notebooks/
│   ├── 01_setup_and_dataset.ipynb     ← общий setup, скачивание, subset
│   ├── 02_deepseek_ocr.ipynb
│   ├── 03_mplug_docowl.ipynb
│   ├── 04_olmocr.ipynb
│   └── 05_monkeyocr.ipynb
├── scripts/
│   └── build_notebooks.py        ← регенерация ноутбуков из шаблонов
├── data/                         ← скачанный OmniDocBench + subset.json
└── results/
    ├── deepseek_ocr/predictions.jsonl
    ├── mplug_docowl/predictions.jsonl
    ├── olmocr/predictions.jsonl
    └── monkeyocr/predictions.jsonl
```

## Как запускать в Colab

1. Открыть `notebooks/01_setup_and_dataset.ipynb`.
   Включить GPU runtime (Runtime → Change runtime type → T4/L4/A100).
2. Прогнать ноутбук целиком — он скачает датасет и сохранит общий
   subset (100 страниц academic_literature, en) в `data/subset.json`.
3. Затем последовательно запускать `02_…`, `03_…`, `04_…`, `05_…`.
   Каждый ноутбук:
   * грузит модель,
   * читает один и тот же `subset.json`,
   * пишет в `results/<model>/predictions.jsonl` построчно
     (поддерживает докатывание после перезапуска runtime),
   * освобождает GPU перед выходом.

> **Совет.** В Colab Free бесплатной T4 хватает на DeepSeek-OCR (3B)
> и MonkeyOCR (3B). Для DocOwl2 (~8B) и olmOCR-7B нужен Colab Pro
> (L4/A100) либо свести `subset_size` в конфиге до 30–50.

## Воспроизводимость

* `subset.json` фиксирует точные 100 страниц (seed=42 в loader'e).
* Все конфиги под версионным контролем.
* Метрики — детерминированные (нет sampling в инференсе:
  `do_sample=False`, `temperature=0.1` для olmOCR).

## Что дальше

После того, как все 4 ноутбука отработают, запускается следующая
порция кода (отдельный issue):

1. `metrics/` — модули подсчёта Text / Table / IoU / Recall@5 / Overall.
2. `06_evaluate.ipynb` — сводный CSV `results/summary.csv`.
3. `07_visualize.ipynb` — bar chart, radar chart, error analysis.
4. Раздел в .docx-отчёте НИР.
