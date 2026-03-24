from __future__ import annotations

from pathlib import Path
from typing import Any

from config.models import RewardProfile, RlStageConfig


def expect_rl_config(stage_config: Any) -> RlStageConfig:
    if not isinstance(stage_config, RlStageConfig):
        raise TypeError(f"RL stage received unexpected config type: {type(stage_config).__name__}")
    return stage_config


def project_root(experiment: Any) -> Path:
    recipe_path = experiment.source_files.get("recipe")
    if recipe_path is None:
        return Path.cwd()
    return recipe_path.parent.parent.parent


def scale_rewards(config: RlStageConfig, reward_profile: RewardProfile | None) -> str | bool:
    if reward_profile is not None:
        return reward_profile.scale_rewards
    return config.scale_rewards


def multi_objective_aggregation(config: RlStageConfig, reward_profile: RewardProfile | None) -> str:
    if reward_profile is not None:
        return reward_profile.multi_objective_aggregation
    return config.multi_objective_aggregation
