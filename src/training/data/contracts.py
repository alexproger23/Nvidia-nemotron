from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CompetitionExample:
    example_id: str
    prompt: str
    split: str
    answer: str | None = None


@dataclass(slots=True)
class ReasoningExample:
    source_name: str
    split: str
    prompt: str
    answer: str
    example_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetSummary:
    name: str
    split: str
    rows: int
    rows_with_answers: int
    path: Path
    format: str
    fields: tuple[str, ...] = ()
