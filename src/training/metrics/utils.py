from __future__ import annotations

from typing import Any

from .contracts import MetricFunction


def named_metric(name: str, function: MetricFunction) -> MetricFunction:
    setattr(function, "__name__", name)
    return function


def rows_for_kwargs(kwargs: dict[str, Any]) -> list[Any]:
    for key in ("prompts", "completions", "completion_ids", "completions_ids"):
        rows = kwargs.get(key)
        if isinstance(rows, list):
            return rows
    return []
