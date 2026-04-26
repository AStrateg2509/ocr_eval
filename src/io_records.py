"""
Единый формат записи в JSONL — то, что предсказывает модель.

Каждая запись содержит page_id и блоки распознанной разметки в максимально
сводимом к OmniDocBench виде, чтобы потом скрипты подсчёта метрик
(будут реализованы на следующем шаге) могли работать единообразно.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class PredictionRecord:
    page_id: str
    model: str
    full_text: str = ""                       # markdown / plain
    tables_html: List[str] = field(default_factory=list)
    formulas: List[str] = field(default_factory=list)
    layout_boxes: List[dict] = field(default_factory=list)  # {"category","bbox","text"}
    raw_output: Optional[str] = None
    inference_time_s: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)
