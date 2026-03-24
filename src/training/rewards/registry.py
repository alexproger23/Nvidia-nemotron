from __future__ import annotations

from config.models import RewardComponentConfig, RewardProfile

from .builtin import register_builtin_rewards
from .contracts import RewardFactory, RewardFunction, RewardStack
from .functions import register_user_rewards
from .utils import named_reward, zeros_for_kwargs


class RewardRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, RewardFactory] = {}

    def register(self, name: str, factory: RewardFactory) -> None:
        self._factories[name] = factory

    def has(self, name: str) -> bool:
        return name in self._factories

    def build(self, component: RewardComponentConfig) -> RewardFunction:
        try:
            factory = self._factories[component.name]
        except KeyError as exc:
            available = ", ".join(sorted(self._factories)) or "<empty>"
            raise KeyError(
                f"Unknown reward component '{component.name}'. Registered components: {available}"
            ) from exc
        return factory(component)


def build_default_reward_registry() -> RewardRegistry:
    registry = RewardRegistry()
    register_builtin_rewards(registry.register)
    register_user_rewards(registry.register)
    return registry


def build_reward_stack(
    profile: RewardProfile | None,
    registry: RewardRegistry | None = None,
) -> RewardStack:
    active_registry = registry or build_default_reward_registry()
    if profile is None or not profile.components:
        zero_reward = named_reward("zero_reward", lambda **kwargs: zeros_for_kwargs(kwargs))
        return RewardStack(
            functions=(zero_reward,),
            weights=(1.0,),
            component_names=("zero_reward",),
            uses_stub=True,
        )

    functions = tuple(active_registry.build(component) for component in profile.components)
    weights = tuple(component.weight for component in profile.components)
    names = tuple(component.name for component in profile.components)
    return RewardStack(functions=functions, weights=weights, component_names=names, uses_stub=False)
