from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from config.models import RewardComponentConfig

RewardFunction = Callable[..., list[float]]
RewardFactory = Callable[[RewardComponentConfig], RewardFunction]


@dataclass(slots=True)
class RewardStack:
    functions: tuple[RewardFunction, ...]
    weights: tuple[float, ...]
    component_names: tuple[str, ...]
    uses_stub: bool = False
