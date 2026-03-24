from __future__ import annotations

from typing import Any

from config.models import RewardComponentConfig

from .contracts import RewardFunction
from .utils import named_reward, rows_for_kwargs


def register_builtin_rewards(register: Any) -> None:
    register("constant", build_constant_reward)


def build_constant_reward(component: RewardComponentConfig) -> RewardFunction:
    value = float(component.params.get("value", 0.0))
    function_name = f"constant_{component.name}"

    def reward_function(**kwargs: Any) -> list[float]:
        return [value] * len(rows_for_kwargs(kwargs))

    return named_reward(function_name, reward_function)
