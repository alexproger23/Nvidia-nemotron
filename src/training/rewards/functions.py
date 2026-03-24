from __future__ import annotations

from typing import Any

from config.models import RewardComponentConfig

from .contracts import RewardFunction
from .utils import named_reward, rows_for_kwargs


def register_user_rewards(register: Any) -> None:
    """Register custom rewards here.

    Edit only this file when you want to add project-specific rewards.

    Example:
        register("answer_length_bonus", build_answer_length_bonus)
    """

    # register("answer_length_bonus", build_answer_length_bonus)


def build_answer_length_bonus(component: RewardComponentConfig) -> RewardFunction:
    """Example reward.

    Gives positive reward when completion length is within the configured range.
    You can copy this function and adapt it for your own reward logic.
    """

    min_length = int(component.params.get("min_length", 32))
    max_length = int(component.params.get("max_length", 512))
    reward_value = float(component.params.get("reward", 1.0))
    penalty_value = float(component.params.get("penalty", 0.0))

    def reward_function(*, completions: list[Any] | None = None, **kwargs: Any) -> list[float]:
        rows = completions or rows_for_kwargs(kwargs)
        scores: list[float] = []
        for row in rows:
            text = _completion_text(row)
            text_length = len(text.strip())
            if min_length <= text_length <= max_length:
                scores.append(reward_value)
            else:
                scores.append(penalty_value)
        return scores

    return named_reward("answer_length_bonus", reward_function)


def _completion_text(row: Any) -> str:
    if isinstance(row, str):
        return row
    if isinstance(row, dict):
        text = row.get("text")
        return text if isinstance(text, str) else str(text or "")
    return str(row)
