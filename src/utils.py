"""
Общие утилиты: загрузка YAML-конфигов, JSONL-логирование, замер времени,
управление GPU-памятью.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


# -------- конфиг --------

def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    """Загрузить YAML-конфиг."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -------- JSONL --------

class JsonlWriter:
    """
    Запись результатов инференса в .jsonl. Открывает файл в режиме 'a',
    чтобы можно было докатывать выборку поверх предыдущей сессии Kaggle/Colab.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def already_processed_ids(path: str | os.PathLike,
                          include_errors: bool = False) -> set[str]:
    """
    Идентификаторы страниц, уже записанные в JSONL — для возобновления
    после перезапуска runtime.

    По умолчанию ошибочные записи (error != null) НЕ считаются обработанными,
    чтобы при следующем прогоне они переобработались. Если хочется
    держаться за старое поведение (любой page_id = done), передайте
    include_errors=True.
    """
    p = Path(path)
    if not p.exists():
        return set()
    ids: set[str] = set()
    for rec in read_jsonl(p):
        if rec.get("error") and not include_errors:
            continue
        if "page_id" in rec:
            ids.add(rec["page_id"])
    return ids


def drop_error_records(path: str | os.PathLike) -> dict:
    """
    Перезаписать JSONL, исключив строки с error != null.

    Полезно вызывать перед каждым прогоном модели, если предыдущий запуск
    оставил ошибочные записи (например, из-за бага в загрузке весов).
    Возвращает {"kept": K, "removed": R}.

    Если файла нет — no-op, возвращает {"kept":0, "removed":0}.
    """
    p = Path(path)
    if not p.exists():
        return {"kept": 0, "removed": 0}
    recs = read_jsonl(p)
    kept = [r for r in recs if not r.get("error")]
    removed = len(recs) - len(kept)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(p)
    return {"kept": len(kept), "removed": removed}


# -------- время --------

@dataclass
class Timer:
    label: str = "block"
    elapsed: float = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._t0


@contextmanager
def measure(label: str = "block"):
    t0 = time.perf_counter()
    yield
    print(f"[{label}] {time.perf_counter() - t0:.2f}s", file=sys.stderr)


# -------- GPU --------

def cuda_free():
    """Освободить GPU-память. Полезно между сменой моделей в одном ноутбуке."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


def gpu_info() -> str:
    """Сводка по всем доступным GPU (актуально для Kaggle T4 x2)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "no CUDA"
        n = torch.cuda.device_count()
        parts = []
        for i in range(n):
            name = torch.cuda.get_device_name(i)
            total = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
            used = torch.cuda.memory_allocated(i) / 1024 ** 3
            cc = torch.cuda.get_device_capability(i)
            parts.append(f"cuda:{i} {name} sm_{cc[0]}{cc[1]} | {used:.2f}/{total:.1f} GiB")
        return " ; ".join(parts)
    except Exception as e:
        return f"gpu_info error: {e}"


def cuda_capabilities() -> list[tuple[int, int]]:
    """Список compute capabilities всех видимых GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        return [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]
    except Exception:
        return []


def supports_flash_attention() -> bool:
    """sm_80+ — full FA2; sm_75 (T4) поддерживает большинство ядер FA2.
    Возвращаем True для sm_75+, False для P100/V100/Pascal."""
    caps = cuda_capabilities()
    return bool(caps) and all(cc[0] >= 7 and (cc[0] > 7 or cc[1] >= 5) for cc in caps)


# -------- batch helpers --------

def chunked(iterable: Iterable, size: int):
    buf: list = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf
