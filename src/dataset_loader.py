"""
Загрузчик OmniDocBench для практической части НИР.

OmniDocBench — open-source бенчмарк (opendatalab/OmniDocBench, v1.6).
В Colab/Kaggle полный датасет распаковывается ≈3.5 ГБ; ноутбуки берут
небольшое подмножество, чтобы уложиться в бесплатные лимиты.

Реальная структура OmniDocBench.json:
    [
      {
        "page_info": {
            "image_path": "page-xxxx.png",   <- только имя файла, без images/
            "page_no": 0,
            "height": 2339, "width": 1653,
            "page_attribute": {
                "data_source": "academic_literature",
                "language": "english",
                "layout": "single_column",
                "subset": "v1.5"
            }
        },
        "layout_dets": [
            {"category_type": "text_block"|"equation_isolated"|"table"|...,
             "poly": [x1,y1,...],
             "text": "...",
             "latex": "...",
             "html": "..."},
        ],
        "extra": {"relation": []}
      }, ...
    ]

Структура репозитория HuggingFace:
    OmniDocBench.json
    images/<filename>     <- все картинки в одной папке, без подпапок
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

from PIL import Image


REPO_ID = "opendatalab/OmniDocBench"


# -------- структуры --------

@dataclass
class GroundTruth:
    """Единый формат ground-truth, удобный для сравнения с предсказаниями моделей."""

    page_id: str
    image_path: str       # относительный путь от корня датасета, напр. "images/page-xxx.png"
    page_type: str
    language: str
    full_text: str = ""
    tables_html: List[str] = field(default_factory=list)
    formulas: List[str] = field(default_factory=list)
    layout_polys: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "page_id": self.page_id,
            "image_path": self.image_path,
            "page_type": self.page_type,
            "language": self.language,
            "full_text": self.full_text,
            "tables_html": self.tables_html,
            "formulas": self.formulas,
            "layout_polys": self.layout_polys,
        }


# -------- скачивание --------

def _hf_download_with_retry(filename: str, local_dir: str,
                             max_retries: int = 8) -> Path:
    """
    Скачивает один файл из репо с экспоненциальным backoff при 429.
    """
    from huggingface_hub import hf_hub_download

    for attempt in range(max_retries):
        try:
            return Path(hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=local_dir,
            ))
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = min(30 * (2 ** attempt), 300)
                print(f"  429 rate limit -- жду {wait}s (попытка {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Не удалось скачать {filename} после {max_retries} попыток")


def download_omnidocbench(target_dir: str | os.PathLike,
                          source: str = "huggingface") -> Path:
    """
    Скачивает OmniDocBench файл за файлом, обходя rate limit HuggingFace (429).

    Стратегия:
      1. Скачиваем OmniDocBench.json (разметка) -- 1 запрос.
      2. Читаем JSON, собираем список нужных image_path.
      3. Скачиваем только те картинки, которых ещё нет на диске.
         При 429 -- ждём с экспоненциальным backoff и продолжаем.

    Прерывание (Ctrl+C) безопасно -- при повторном запуске уже скачанные
    файлы пропускаются автоматически.
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "images").mkdir(exist_ok=True)

    # 1. JSON-разметка
    json_path = target / "OmniDocBench.json"
    if not json_path.exists():
        print("Скачиваем OmniDocBench.json ...")
        _hf_download_with_retry("OmniDocBench.json", str(target))
        print("  OK OmniDocBench.json")
    else:
        print("OmniDocBench.json уже есть, пропускаем.")

    # 2. Список всех картинок из JSON
    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    all_image_names = []
    for page in raw:
        name = page.get("page_info", {}).get("image_path", "").strip()
        if name:
            all_image_names.append(name)

    total = len(all_image_names)
    print(f"Всего страниц в датасете: {total}")

    # 3. Скачиваем только отсутствующие
    missing = [n for n in all_image_names if not (target / "images" / n).exists()]
    already = total - len(missing)
    print(f"Уже скачано: {already}, осталось: {len(missing)}")

    errors = []
    for i, name in enumerate(missing, 1):
        local_path = target / "images" / name
        if local_path.exists():
            continue
        if i % 100 == 0 or i <= 3:
            print(f"  [{i}/{len(missing)}] {name}")
        try:
            _hf_download_with_retry(f"images/{name}", str(target))
        except Exception as e:
            errors.append(name)
            print(f"  ERROR {name}: {e}")

    done = sum(1 for n in all_image_names if (target / "images" / n).exists())
    print(f"\nГотово: {done}/{total} картинок на диске.")
    if errors:
        print(f"Не удалось скачать {len(errors)} файлов: {errors[:5]}{'...' if len(errors) > 5 else ''}")
    return target


# -------- парсинг --------

def _aggregate_page(page: dict) -> GroundTruth:
    info = page.get("page_info", {})
    attr = info.get("page_attribute", {})

    raw_name = info.get("image_path", "").strip()
    # В репо все картинки лежат в images/ без подпапок
    image_path = f"images/{raw_name}" if raw_name else ""

    page_id = f"{info.get('page_no', 0)}_{Path(raw_name).stem}"

    text_chunks: List[str] = []
    tables: List[str] = []
    formulas: List[str] = []
    layout: List[dict] = []

    for det in page.get("layout_dets", []):
        cat = det.get("category_type", "")
        layout.append({
            "category": cat,
            "poly": det.get("poly", []),
            "text": det.get("text", ""),
        })
        if cat in {"text_block", "title", "header", "footer", "caption"}:
            text_chunks.append(det.get("text", ""))
        elif cat == "table":
            tables.append(det.get("html") or det.get("text", ""))
        elif cat in {"equation_isolated", "equation_inline", "equation_semantic",
                     "formula", "isolate_formula", "inline_formula"}:
            formulas.append(det.get("latex") or det.get("text", ""))

    return GroundTruth(
        page_id=page_id,
        image_path=image_path,
        page_type=attr.get("data_source", "unknown"),
        language=attr.get("language", "unknown"),
        full_text="\n".join(t for t in text_chunks if t),
        tables_html=tables,
        formulas=formulas,
        layout_polys=layout,
    )


def load_omnidocbench(root: str | os.PathLike,
                      page_types: Optional[List[str]] = None,
                      languages: Optional[List[str]] = None,
                      subset_size: Optional[int] = None,
                      seed: int = 42) -> List[GroundTruth]:
    """
    Загрузить и отфильтровать OmniDocBench. Не тянет картинки в память --
    возвращает список GroundTruth с относительными путями.

    Параметры:
        root          : путь к распакованному датасету
        page_types    : список значений data_source:
                        {"academic_literature", "book", "PPT2PDF",
                         "exam_paper", "colorful_textbook", "newspaper",
                         "magazine", "research_report", "note",
                         "historical_document"}
                        None -> все
        languages     : ["english", "simplified_chinese", "traditional_chinese",
                         "en_ch_mixed", "other"]
                        None -> все
        subset_size   : ограничение на размер выборки (для Colab)
        seed          : seed для случайной выборки subset_size
    """
    root = Path(root)
    json_path = root / "OmniDocBench.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Не найден {json_path}. Сначала вызовите download_omnidocbench()."
        )

    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    items = [_aggregate_page(p) for p in raw]

    # Оставляем только страницы с реально существующими файлами
    before = len(items)
    items = [x for x in items if x.image_path and (root / x.image_path).exists()]
    missing_count = before - len(items)
    if missing_count:
        print(f"[load_omnidocbench] пропущено {missing_count} страниц (файл не найден)")

    if page_types:
        items = [x for x in items if x.page_type in set(page_types)]
    if languages:
        items = [x for x in items if x.language in set(languages)]

    print(f"[load_omnidocbench] после фильтрации: {len(items)} страниц")

    if subset_size and subset_size < len(items):
        rng = random.Random(seed)
        items = rng.sample(items, subset_size)

    return items


def iter_pages(items: List[GroundTruth],
               root: str | os.PathLike) -> Iterator[tuple[Image.Image, GroundTruth]]:
    """Итератор (PIL.Image, GroundTruth). Пропускает отсутствующие файлы."""
    root = Path(root)
    for gt in items:
        img_path = root / gt.image_path
        if not img_path.exists():
            continue
        with Image.open(img_path) as im:
            yield im.convert("RGB"), gt