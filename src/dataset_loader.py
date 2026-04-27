"""
Загрузчик OmniDocBench для практической части НИР.

Версия 2: исправлены три проблемы первой итерации
  1. `page_attribute` — это вложенный dict, а не плоский набор полей.
  2. `page_type` берётся из `page_attribute["data_source"]`
     (значения: "academic_literature", "book", "PPT2PDF", и т. д.).
  3. В `image_path` JSON хранит только имя файла; реальные пути —
     `images/<filename>`. Loader сам подставляет префикс `images/`,
     поэтому потребители работают с уже корректным относительным путём.

Структура распакованного датасета (HuggingFace `opendatalab/OmniDocBench`):

    OmniDocBench/
        images/                 — ~1653 PNG/JPG страницы
        OmniDocBench.json       — единая разметка
            [
              {
                "page_info": {
                    "image_path": "<filename>.jpg",
                    "page_no": 0,
                    "page_attribute": {
                        "data_source": "academic_literature",
                        "language": "english",
                        "layout": ...,
                        "fancy_titles": ...
                    }
                },
                "layout_dets": [
                    {"category_type": "text"|"table"|"formula"|"figure"|...,
                     "poly": [x1,y1,x2,y2,x3,y3,x4,y4],
                     "text": "...",
                     "html": "..." (для таблиц)}, ...
                ],
                "extra": {...}
              }, ...
            ]
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


# ---------------------------------------------------------------- структуры

@dataclass
class GroundTruth:
    """Унифицированный формат ground-truth, удобный для сравнения с предсказаниями."""

    page_id: str
    image_path: str             # уже с префиксом "images/"
    page_type: str              # data_source из page_attribute
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


# ---------------------------------------------------------------- скачивание

REPO_ID = "opendatalab/OmniDocBench"

#: Имя главного файла разметки внутри репозитория. Опционально могут быть
#: и старые версии — `OmniDocBench1.5.json`. Поиск ведётся по wildcards.
JSON_GLOB = "OmniDocBench*.json"
IMAGES_DIRNAME = "images"


def download_omnidocbench_json(target_dir: str | os.PathLike,
                               token: Optional[str] = None) -> Path:
    """
    Скачать только файлы разметки (JSON) — это очень быстро (несколько МБ).
    Сами картинки качаются отдельно через `download_omnidocbench_images_chunked()`,
    чтобы обойти лимит ~1000 запросов в HF API.
    """
    from huggingface_hub import snapshot_download

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(target),
        local_dir_use_symlinks=False,
        allow_patterns=["*.json", "README*"],
        token=token,
    )
    return target


def list_image_filenames(target_dir: str | os.PathLike) -> list[str]:
    """
    Получить список имён картинок согласно разметке.
    Использует поле page_info.image_path из всех записей JSON.
    """
    target = Path(target_dir)
    json_path = _find_main_json(target)
    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    names = []
    for page in raw:
        info = page.get("page_info", {})
        fname = info.get("image_path", "")
        if fname:
            # JSON может хранить как "name.jpg", так и "images/name.jpg" —
            # унифицируем до bare-имени.
            names.append(Path(fname).name)
    # сохраняем порядок появления, без повторов
    seen = set(); result = []
    for n in names:
        if n not in seen:
            seen.add(n); result.append(n)
    return result


def list_repo_images(token: Optional[str] = None) -> list[str]:
    """
    Альтернативный способ получить список изображений — напрямую из
    репозитория (полезно, если разметка ещё не скачана).
    """
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    return [f for f in files if f.startswith(f"{IMAGES_DIRNAME}/")
            and not f.endswith("/")]


def _find_main_json(root: Path) -> Path:
    """Найти самый «свежий» OmniDocBench*.json в корне."""
    candidates = sorted(root.glob(JSON_GLOB))
    if not candidates:
        raise FileNotFoundError(
            f"Не найден файл разметки {JSON_GLOB} в {root}. "
            f"Сначала вызовите download_omnidocbench_json()."
        )
    # Если их несколько (например, разные версии), берём наибольшую по версии
    return candidates[-1]


# ---------------------------------------------------------------- chunked images

def download_omnidocbench_images_chunked(
    target_dir: str | os.PathLike,
    chunk_size: int = 200,
    sleep_between_chunks: float = 5.0,
    max_retries: int = 5,
    token: Optional[str] = None,
    on_progress=None,
) -> dict:
    """
    Скачать изображения OmniDocBench по чанкам — обходит лимит ~1000 запросов
    HuggingFace Hub API.

    Стратегия:
      * получаем полный список image-файлов в репозитории (один запрос);
      * сравниваем с тем, что уже лежит локально → формируем `to_download`;
      * качаем по `chunk_size` картинок, между чанками `sleep_between_chunks`
        секунд паузы (плюс экспоненциальный backoff при ошибках);
      * после каждого чанка — verify: страница существует на диске
        и читается как изображение (не нулевой размер).

    Возвращает:
      {"total": N_repo, "already": K, "downloaded": M, "failed": [...]}
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError

    target = Path(target_dir)
    images_dir = target / IMAGES_DIRNAME
    images_dir.mkdir(parents=True, exist_ok=True)

    repo_images = list_repo_images(token=token)        # список 'images/xxx.jpg'
    total = len(repo_images)
    if on_progress:
        on_progress("scanned", total=total)

    # уже скачанные — определяем по реальным файлам, не по JSON
    locally_present = {p.name for p in images_dir.iterdir()
                       if p.is_file() and p.stat().st_size > 0}
    to_download = [r for r in repo_images
                   if Path(r).name not in locally_present]

    failed: list[tuple[str, str]] = []
    downloaded = 0

    for i in range(0, len(to_download), chunk_size):
        chunk = to_download[i: i + chunk_size]
        if on_progress:
            on_progress("chunk_start", index=i // chunk_size,
                        size=len(chunk), remaining=len(to_download) - i)

        for repo_path in chunk:
            for attempt in range(max_retries):
                try:
                    hf_hub_download(
                        repo_id=REPO_ID,
                        filename=repo_path,
                        repo_type="dataset",
                        local_dir=str(target),
                        local_dir_use_symlinks=False,
                        token=token,
                    )
                    downloaded += 1
                    break
                except (HfHubHTTPError, OSError) as e:
                    if attempt + 1 == max_retries:
                        failed.append((repo_path, str(e)))
                    else:
                        time.sleep(2 ** attempt)
        # пауза между чанками
        if i + chunk_size < len(to_download):
            time.sleep(sleep_between_chunks)

    summary = {
        "total": total,
        "already": len(locally_present),
        "downloaded": downloaded,
        "failed": failed,
    }
    if on_progress:
        on_progress("done", **summary)
    return summary


def verify_downloaded_images(target_dir: str | os.PathLike,
                             expected: Optional[list[str]] = None) -> dict:
    """
    Файловая верификация — пробегаем по `images/` и проверяем,
    что каждый файл валиден и не пустой. Сравниваем с ожидаемым списком,
    если он задан (можно передать результат `list_repo_images()` или
    `list_image_filenames()`).

    Возвращает {"present": [...], "missing": [...], "broken": [...]}.
    """
    target = Path(target_dir)
    images_dir = target / IMAGES_DIRNAME
    if not images_dir.exists():
        return {"present": [], "missing": expected or [], "broken": []}

    present, broken = [], []
    for p in sorted(images_dir.iterdir()):
        if not p.is_file():
            continue
        if p.stat().st_size == 0:
            broken.append(p.name); continue
        try:
            with Image.open(p) as im:
                im.verify()                      # дешёвая проверка целостности
            present.append(p.name)
        except Exception:
            broken.append(p.name)

    missing: list[str] = []
    if expected:
        present_set = set(present)
        for f in expected:
            name = Path(f).name
            if name not in present_set:
                missing.append(name)

    return {"present": present, "missing": missing, "broken": broken}


# ---------------------------------------------------------------- парсинг разметки

def _aggregate_page(page: dict) -> GroundTruth:
    info = page.get("page_info", {})
    attr = info.get("page_attribute", {}) or {}

    raw_image = info.get("image_path", "") or ""
    image_filename = Path(raw_image).name             # на всякий случай
    image_path = f"{IMAGES_DIRNAME}/{image_filename}" # всегда с префиксом

    page_id = f"{info.get('page_no', 0)}_{Path(image_filename).stem}"
    page_type = attr.get("data_source", "unknown")
    language  = attr.get("language", "unknown")

    text_chunks: List[str] = []
    tables: List[str]      = []
    formulas: List[str]    = []
    layout: List[dict]     = []

    for det in page.get("layout_dets", []):
        cat = det.get("category_type", "")
        layout.append({
            "category": cat,
            "poly": det.get("poly", []),
            "text": det.get("text", ""),
        })
        if cat in {"text_block", "title", "header", "footer", "caption", "text"}:
            text_chunks.append(det.get("text", ""))
        elif cat in {"table"}:
            tables.append(det.get("html") or det.get("text", ""))
        elif cat in {"formula", "isolate_formula", "inline_formula"}:
            formulas.append(det.get("text", ""))

    return GroundTruth(
        page_id=page_id,
        image_path=image_path,
        page_type=page_type,
        language=language,
        full_text="\n".join(t for t in text_chunks if t),
        tables_html=tables,
        formulas=formulas,
        layout_polys=layout,
    )


def load_omnidocbench(root: str | os.PathLike,
                      page_types: Optional[List[str]] = None,
                      languages: Optional[List[str]] = None,
                      subset_size: Optional[int] = None,
                      seed: int = 42,
                      require_image_present: bool = True) -> List[GroundTruth]:
    """
    Загрузить и отфильтровать OmniDocBench.

    Параметры:
        page_types  — фильтр по data_source ("academic_literature", ...).
        languages   — ["english", ...].
        subset_size — ограничение размера выборки.
        require_image_present — отбрасывать записи, для которых картинка
                                ещё не скачана (актуально при чанк-загрузке).
    """
    root = Path(root)
    json_path = _find_main_json(root)

    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    items = [_aggregate_page(p) for p in raw]

    if page_types:
        items = [x for x in items if x.page_type in set(page_types)]
    if languages:
        items = [x for x in items if x.language in set(languages)]

    if require_image_present:
        items = [x for x in items if (root / x.image_path).exists()]

    if subset_size and subset_size < len(items):
        rng = random.Random(seed)
        items = rng.sample(items, subset_size)

    return items


def iter_pages(items: List[GroundTruth],
               root: str | os.PathLike) -> Iterator[tuple[Image.Image, GroundTruth]]:
    """Итератор `(PIL.Image, GroundTruth)`. Пропускает отсутствующие файлы."""
    root = Path(root)
    for gt in items:
        img_path = root / gt.image_path
        if not img_path.exists():
            continue
        with Image.open(img_path) as im:
            yield im.convert("RGB"), gt
