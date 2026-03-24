from __future__ import annotations

from typing import Any

from .contracts import RewardFunction


def named_reward(name: str, function: RewardFunction) -> RewardFunction:
    setattr(function, "__name__", name)
    return function


def zeros_for_kwargs(kwargs: dict[str, Any]) -> list[float]:
    return [0.0] * len(rows_for_kwargs(kwargs))


def rows_for_kwargs(kwargs: dict[str, Any]) -> list[Any]:
    for key in ("prompts", "completions", "completion_ids", "completions_ids"):
        rows = kwargs.get(key)
        if isinstance(rows, list):
            return rows
    return []
