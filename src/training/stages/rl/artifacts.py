from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from config.models import MetricProfile, RewardProfile, RlStageConfig
from training.contracts import ArtifactRef
from training.metrics import MetricStack
from training.rewards import RewardStack


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def dataset_summary_payload(
    *,
    summaries: dict[str, list[Any]],
    train_rows: int,
) -> dict[str, Any]:
    return {
        "train_rows_used": train_rows,
        "sources": {
            split: [
                {
                    "name": summary.name,
                    "split": summary.split,
                    "rows": summary.rows,
                    "rows_with_answers": summary.rows_with_answers,
                    "path": str(summary.path),
                    "format": summary.format,
                    "fields": list(summary.fields),
                }
                for summary in split_summaries
            ]
            for split, split_summaries in summaries.items()
        },
    }


def reward_manifest_payload(
    *,
    reward_profile: RewardProfile | None,
    reward_stack: RewardStack,
) -> dict[str, Any]:
    return {
        "reward_profile": None if reward_profile is None else asdict(reward_profile),
        "resolved_components": list(reward_stack.component_names),
        "weights": list(reward_stack.weights),
        "uses_stub_reward": reward_stack.uses_stub,
    }


def metric_manifest_payload(
    *,
    metric_profile: MetricProfile | None,
    metric_stack: MetricStack,
) -> dict[str, Any]:
    return {
        "metric_profile": None if metric_profile is None else asdict(metric_profile),
        "resolved_components": list(metric_stack.component_names),
    }


def create_checkpoint_artifact(
    *,
    checkpoint_dir: Path,
    checkpoint_source: str,
    reward_stack: RewardStack,
    metric_stack: MetricStack,
    base_model: str,
) -> ArtifactRef:
    return ArtifactRef(
        name="rl_checkpoint",
        path=checkpoint_dir,
        kind="checkpoint",
        metadata={
            "checkpoint_source": checkpoint_source,
            "reward_components": list(reward_stack.component_names),
            "metric_components": list(metric_stack.component_names),
            "uses_stub_reward": reward_stack.uses_stub,
            "base_model": base_model,
        },
    )


def create_stage_artifacts(
    *,
    stage_config_path: Path,
    dataset_summary_path: Path,
    reward_manifest_path: Path,
    metric_manifest_path: Path,
    metrics_summary_path: Path,
    trainer_state_path: Path,
    checkpoint_artifact: ArtifactRef,
) -> dict[str, ArtifactRef]:
    return {
        "rl_stage_config": ArtifactRef(
            name="rl_stage_config",
            path=stage_config_path,
            kind="config",
        ),
        "rl_dataset_summary": ArtifactRef(
            name="rl_dataset_summary",
            path=dataset_summary_path,
            kind="dataset_summary",
        ),
        "rl_reward_manifest": ArtifactRef(
            name="rl_reward_manifest",
            path=reward_manifest_path,
            kind="reward_manifest",
        ),
        "rl_metric_manifest": ArtifactRef(
            name="rl_metric_manifest",
            path=metric_manifest_path,
            kind="metric_manifest",
        ),
        "rl_metrics_summary": ArtifactRef(
            name="rl_metrics_summary",
            path=metrics_summary_path,
            kind="metrics",
        ),
        "rl_trainer_state": ArtifactRef(
            name="rl_trainer_state",
            path=trainer_state_path,
            kind="trainer_state",
        ),
        "checkpoint": checkpoint_artifact,
    }


def write_stage_inputs(
    *,
    config: RlStageConfig,
    stage_config_path: Path,
    dataset_summary_path: Path,
    dataset_summary: dict[str, Any],
    reward_manifest_path: Path,
    reward_manifest: dict[str, Any],
    metric_manifest_path: Path,
    metric_manifest: dict[str, Any],
) -> None:
    write_json(stage_config_path, asdict(config))
    write_json(dataset_summary_path, dataset_summary)
    write_json(reward_manifest_path, reward_manifest)
    write_json(metric_manifest_path, metric_manifest)
