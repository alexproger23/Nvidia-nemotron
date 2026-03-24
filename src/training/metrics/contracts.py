from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from config.models import MetricComponentConfig

MetricValue = float | None
MetricFunction = Callable[..., list[MetricValue]]
MetricFactory = Callable[[MetricComponentConfig], MetricFunction]


@dataclass(slots=True)
class MetricStack:
    functions: tuple[MetricFunction, ...]
    component_names: tuple[str, ...]
