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
    чтобы можно было докатывать выборку поверх предыдущей сессии Colab.
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


def already_processed_ids(path: str | os.PathLike) -> set[str]:
    """Идентификаторы страниц, уже записанные в JSONL — для возобновления после перезапуска."""
    p = Path(path)
    if not p.exists():
        return set()
    ids: set[str] = set()
    for rec in read_jsonl(p):
        if "page_id" in rec:
            ids.add(rec["page_id"])
    return ids


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
    try:
        import torch
        if not torch.cuda.is_available():
            return "no CUDA"
        i = torch.cuda.current_device()
        name = torch.cuda.get_device_name(i)
        total = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
        used = torch.cuda.memory_allocated(i) / 1024 ** 3
        return f"{name} | used {used:.2f} / {total:.1f} GiB"
    except Exception as e:
        return f"gpu_info error: {e}"


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
