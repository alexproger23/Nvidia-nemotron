from __future__ import annotations

from config.models import MetricComponentConfig, MetricProfile

from .contracts import MetricFactory, MetricFunction, MetricStack
from .functions import register_user_metrics


class MetricRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, MetricFactory] = {}

    def register(self, name: str, factory: MetricFactory) -> None:
        self._factories[name] = factory

    def has(self, name: str) -> bool:
        return name in self._factories

    def build(self, component: MetricComponentConfig) -> MetricFunction:
        try:
            factory = self._factories[component.name]
        except KeyError as exc:
            available = ", ".join(sorted(self._factories)) or "<empty>"
            raise KeyError(
                f"Unknown metric component '{component.name}'. Registered components: {available}"
            ) from exc
        return factory(component)


def build_default_metric_registry() -> MetricRegistry:
    registry = MetricRegistry()
    register_user_metrics(registry.register)
    return registry


def build_metric_stack(
    profile: MetricProfile | None,
    registry: MetricRegistry | None = None,
) -> MetricStack:
    active_registry = registry or build_default_metric_registry()
    if profile is None or not profile.components:
        return MetricStack(functions=(), component_names=())

    functions = tuple(active_registry.build(component) for component in profile.components)
    names = tuple(component.name for component in profile.components)
    return MetricStack(functions=functions, component_names=names)
