"""
Загрузчик OmniDocBench для практической части НИР.

OmniDocBench — open-source бенчмарк (opendatalab/OmniDocBench, v1.6).
В Colab/Kaggle полный датасет распаковывается ≈3.5 ГБ; ноутбуки берут
небольшое подмножество, чтобы уложиться в бесплатные лимиты.

Структура распакованного датасета (HuggingFace):
    OmniDocBench/
        images/                 — PNG-страницы (~1000 страниц)
        OmniDocBench.json       — единый JSON со всей разметкой
            [
              {
                "page_info": {"image_path": "images/...png", "page_no": 0,
                              "language": "english", "page_type": "academic_literature", ...},
                "layout_dets": [
                    {"category_type": "text"|"table"|"formula"|"figure"|...,
                     "poly": [x1,y1,x2,y2,x3,y3,x4,y4],
                     "text": "ground-truth text" | "<html>...</html>",
                     "html": "..." (для таблиц)},
                    ...
                ],
                "extra": {...}
              }, ...
            ]

Этот модуль:
  * скачивает архив (через `datasets` или прямой git-clone/hf hub),
  * фильтрует страницы по категориям (по умолчанию academic_literature),
  * возвращает итератор `(image_pil, ground_truth_dict)`.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

from PIL import Image


# -------- структуры --------

@dataclass
class GroundTruth:
    """Единый формат ground-truth, удобный для сравнения с предсказаниями моделей."""

    page_id: str
    image_path: str
    page_type: str
    language: str
    full_text: str = ""                    # склеенный текст всех text-блоков
    tables_html: List[str] = field(default_factory=list)
    formulas: List[str] = field(default_factory=list)
    layout_polys: List[dict] = field(default_factory=list)  # {"category", "poly", "text"}

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

def download_omnidocbench(target_dir: str | os.PathLike,
                          source: str = "huggingface") -> Path:
    """
    Скачивает OmniDocBench. Поддерживаемые источники:
      * "huggingface" — opendatalab/OmniDocBench  (рекомендуется в Colab)
      * "github"      — clone репозитория и его release-архива

    Возвращает Path к распакованной корневой директории.
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    if (target / "OmniDocBench.json").exists():
        return target  # уже скачано

    if source == "huggingface":
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise RuntimeError(
                "huggingface_hub не установлен. pip install huggingface-hub"
            ) from e

        snapshot_download(
            repo_id="opendatalab/OmniDocBench",
            repo_type="dataset",
            local_dir=str(target),
            local_dir_use_symlinks=False,
        )
        return target

    if source == "github":
        import subprocess
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/opendatalab/OmniDocBench", str(target)],
            check=True,
        )
        return target

    raise ValueError(f"Unknown source: {source}")


# -------- парсинг --------

def _aggregate_page(page: dict) -> GroundTruth:
    info = page.get("page_info", {})
    page_id = f"{info.get('page_no', 0)}_{Path(info.get('image_path', '')).stem}"
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
        elif cat in {"table"}:
            tables.append(det.get("html") or det.get("text", ""))
        elif cat in {"formula", "isolate_formula", "inline_formula"}:
            formulas.append(det.get("text", ""))

    return GroundTruth(
        page_id=page_id,
        image_path=info.get("image_path", ""),
        page_type=info.get("page_type", "unknown"),
        language=info.get("language", "unknown"),
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
    Загрузить и отфильтровать OmniDocBench. Не тянет картинки в память —
    возвращает список GroundTruth с относительными путями.

    Параметры:
        root          : путь к распакованному датасету
        page_types    : список из {"academic_literature", "textbook",
                                   "newspaper", "magazine", "ppt", ...}
                        None  → все
        languages     : ["english", "simplified_chinese", "mixed"]
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

    if page_types:
        items = [x for x in items if x.page_type in set(page_types)]
    if languages:
        items = [x for x in items if x.language in set(languages)]

    if subset_size and subset_size < len(items):
        rng = random.Random(seed)
        items = rng.sample(items, subset_size)

    return items


def iter_pages(items: List[GroundTruth],
               root: str | os.PathLike) -> Iterator[tuple[Image.Image, GroundTruth]]:
    """Итератор (PIL.Image, GroundTruth). Разрешает None для отсутствующих файлов."""
    root = Path(root)
    for gt in items:
        img_path = root / gt.image_path
        if not img_path.exists():
            continue
        with Image.open(img_path) as im:
            yield im.convert("RGB"), gt
