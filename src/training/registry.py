from __future__ import annotations

from collections.abc import Callable

from training.contracts import Stage

StageFactory = Callable[[], Stage]


class StageRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, StageFactory] = {}

    def register(self, stage_name: str, factory: StageFactory) -> None:
        self._factories[stage_name] = factory

    def create(self, stage_name: str) -> Stage:
        try:
            return self._factories[stage_name]()
        except KeyError as exc:
            raise KeyError(f"Stage '{stage_name}' is not registered") from exc
