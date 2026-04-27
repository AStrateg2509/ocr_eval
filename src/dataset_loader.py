"""
Загрузчик OmniDocBench для практической части НИР.

OmniDocBench — open-source бенчмарк (opendatalab/OmniDocBench, v1.6).
В Colab/Kaggle полный датасет распаковывается ~3.5 ГБ; ноутбуки берут
небольшое подмножество, чтобы уложиться в бесплатные лимиты.

Структура репозитория HuggingFace:
    OmniDocBench.json
    images/<filename>   <- все картинки в одной папке, без подпапок

Структура OmniDocBench.json:
    [
      {
        "page_info": {
            "image_path": "page-xxxx.png",   <- только имя файла, без images/
            "page_no": 0,
            "page_attribute": {
                "data_source": "academic_literature",
                "language": "english",
            }
        },
        "layout_dets": [
            {"category_type": "text_block"|"table"|"equation_isolated"|...,
             "text": "...", "latex": "...", "html": "..."},
        ],
      }, ...
    ]
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

from PIL import Image


# HuggingFace CDN — прямые ссылки, без API-токенов, без rate limit на токены
HF_BASE = "https://huggingface.co/datasets/opendatalab/OmniDocBench/resolve/main"


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

def _http_download(url: str, dest: Path, max_retries: int = 6,
                   hf_token: Optional[str] = None) -> None:
    """
    Скачивает файл напрямую по HTTP, без HuggingFace Python SDK.
    Обходит rate limit на xet-read-token — используется CDN-ссылка.
    При 429 или 503 — экспоненциальный backoff.
    """
    headers = {"User-Agent": "ocr-eval/1.0"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp, \
                 open(tmp, "wb") as f:
                while chunk := resp.read(1 << 20):  # 1 MB chunks
                    f.write(chunk)
            tmp.rename(dest)
            return
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < max_retries - 1:
                wait = min(15 * (2 ** attempt), 240)  # 15, 30, 60, 120, 240
                print(f"  HTTP {e.code} — жду {wait}s (попытка {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"  Ошибка ({e}) — жду {wait}s (попытка {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)


def download_omnidocbench(target_dir: str | os.PathLike,
                          source: str = "huggingface",
                          hf_token: Optional[str] = None) -> Path:
    """
    Скачивает OmniDocBench через прямые HTTP-ссылки (CDN HuggingFace),
    полностью обходя HF Python API и его rate limit на xet-read-token.

    Прерывание (Ctrl+C) безопасно — повторный запуск продолжит с места остановки.

    Параметры:
        target_dir : куда сохранять датасет
        source     : не используется (оставлен для совместимости)
        hf_token   : токен HF если репо приватный (здесь публичный, не нужен)
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "images").mkdir(exist_ok=True)

    # 1. JSON-разметка
    json_path = target / "OmniDocBench.json"
    if not json_path.exists():
        print("Скачиваем OmniDocBench.json ...")
        url = f"{HF_BASE}/OmniDocBench.json"
        _http_download(url, json_path, hf_token=hf_token)
        print(f"  OK ({json_path.stat().st_size // 1024} KB)")
    else:
        print(f"OmniDocBench.json уже есть ({json_path.stat().st_size // 1024} KB)")

    # 2. Список всех картинок из JSON
    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    all_names = [
        p["page_info"]["image_path"].strip()
        for p in raw
        if p.get("page_info", {}).get("image_path", "").strip()
    ]
    total = len(all_names)
    missing = [n for n in all_names if not (target / "images" / n).exists()]
    print(f"Страниц в датасете: {total} | уже есть: {total - len(missing)} | скачать: {len(missing)}")

    if not missing:
        print("Всё уже скачано!")
        return target

    # 3. Скачиваем отсутствующие напрямую через CDN
    errors = []
    for i, name in enumerate(missing, 1):
        dest = target / "images" / name
        if dest.exists():
            continue
        if i <= 3 or i % 200 == 0:
            print(f"  [{i}/{len(missing)}] {name}")
        url = f"{HF_BASE}/images/{name}"
        try:
            _http_download(url, dest, hf_token=hf_token)
        except Exception as e:
            errors.append(name)
            print(f"  ОШИБКА [{i}] {name}: {e}")

    done = sum(1 for n in all_names if (target / "images" / n).exists())
    print(f"\nГотово: {done}/{total} картинок на диске.")
    if errors:
        print(f"Не удалось скачать {len(errors)}: {errors[:3]}{'...' if len(errors) > 3 else ''}")
    return target


# -------- парсинг --------

def _aggregate_page(page: dict) -> GroundTruth:
    info = page.get("page_info", {})
    attr = info.get("page_attribute", {})

    raw_name = info.get("image_path", "").strip()
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
    Загрузить и отфильтровать OmniDocBench.

    Параметры:
        root          : путь к распакованному датасету
        page_types    : {"academic_literature", "book", "PPT2PDF", "exam_paper",
                         "colorful_textbook", "newspaper", "magazine",
                         "research_report", "note", "historical_document"} | None
        languages     : ["english", "simplified_chinese", "traditional_chinese",
                         "en_ch_mixed", "other"] | None
        subset_size   : ограничение на размер выборки
        seed          : seed для случайной выборки
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

    # Только страницы с реально существующими файлами
    before = len(items)
    items = [x for x in items if x.image_path and (root / x.image_path).exists()]
    if before - len(items):
        print(f"[load_omnidocbench] пропущено {before - len(items)} страниц без файла на диске")

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