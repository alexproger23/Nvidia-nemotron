from __future__ import annotations

from pathlib import Path

from training.contracts import StageContext, StageResult
from training.data import load_reasoning_split, summarize_reasoning_profile
from training.metrics import MetricRegistry, build_default_metric_registry, build_metric_stack
from training.rewards import RewardRegistry, build_default_reward_registry, build_reward_stack

from .artifacts import (
    create_checkpoint_artifact,
    create_stage_artifacts,
    dataset_summary_payload,
    metric_manifest_payload,
    reward_manifest_payload,
    write_json,
    write_stage_inputs,
)
from .common import expect_rl_config, project_root
from .training import run_grpo_training


class RlStage:
    name = "rl"

    def __init__(
        self,
        reward_registry: RewardRegistry | None = None,
        metric_registry: MetricRegistry | None = None,
    ) -> None:
        self.reward_registry = reward_registry or build_default_reward_registry()
        self.metric_registry = metric_registry or build_default_metric_registry()

    def run(self, context: StageContext) -> StageResult:
        config = expect_rl_config(context.stage_config)
        experiment = context.experiment
        if experiment.data is None:
            raise ValueError("RL stage requires a resolved data profile")

        root = project_root(experiment)
        train_examples = load_reasoning_split(
            experiment.data,
            "train",
            project_root=root,
        )
        if config.max_train_samples is not None:
            train_examples = train_examples[: config.max_train_samples]
        if not train_examples:
            raise ValueError("RL stage resolved an empty train split")

        reward_profile = experiment.reward
        reward_stack = build_reward_stack(reward_profile, registry=self.reward_registry)
        metric_profile = experiment.metric
        metric_stack = build_metric_stack(metric_profile, registry=self.metric_registry)
        dataset_summaries = summarize_reasoning_profile(experiment.data, project_root=root)

        stage_paths = _stage_paths(context.output_dir)
        write_stage_inputs(
            config=config,
            stage_config_path=stage_paths["stage_config"],
            dataset_summary_path=stage_paths["dataset_summary"],
            dataset_summary=dataset_summary_payload(
                summaries=dataset_summaries,
                train_rows=len(train_examples),
            ),
            reward_manifest_path=stage_paths["reward_manifest"],
            reward_manifest=reward_manifest_payload(
                reward_profile=reward_profile,
                reward_stack=reward_stack,
            ),
            metric_manifest_path=stage_paths["metric_manifest"],
            metric_manifest=metric_manifest_payload(
                metric_profile=metric_profile,
                metric_stack=metric_stack,
            ),
        )

        training_result = run_grpo_training(
            context=context,
            config=config,
            reward_stack=reward_stack,
            metric_stack=metric_stack,
            train_examples=train_examples,
        )
        write_json(stage_paths["metrics_summary"], training_result["metrics"])
        write_json(stage_paths["trainer_state"], training_result["trainer_state"])

        checkpoint_artifact = create_checkpoint_artifact(
            checkpoint_dir=training_result["checkpoint_dir"],
            checkpoint_source=training_result["checkpoint_source"],
            reward_stack=reward_stack,
            metric_stack=metric_stack,
            base_model=experiment.model.base_model,
        )
        artifacts = create_stage_artifacts(
            stage_config_path=stage_paths["stage_config"],
            dataset_summary_path=stage_paths["dataset_summary"],
            reward_manifest_path=stage_paths["reward_manifest"],
            metric_manifest_path=stage_paths["metric_manifest"],
            metrics_summary_path=stage_paths["metrics_summary"],
            trainer_state_path=stage_paths["trainer_state"],
            checkpoint_artifact=checkpoint_artifact,
        )

        return StageResult(
            stage_name=self.name,
            metrics=training_result["metrics"],
            artifacts=artifacts,
            checkpoint=checkpoint_artifact,
            metadata={
                "reward_components": list(reward_stack.component_names),
                "metric_components": list(metric_stack.component_names),
                "uses_stub_reward": reward_stack.uses_stub,
                "checkpoint_source": training_result["checkpoint_source"],
            },
        )


def _stage_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "stage_config": output_dir / "resolved_stage_config.json",
        "dataset_summary": output_dir / "dataset_summary.json",
        "reward_manifest": output_dir / "reward_manifest.json",
        "metric_manifest": output_dir / "metric_manifest.json",
        "metrics_summary": output_dir / "metrics_summary.json",
        "trainer_state": output_dir / "trainer_state.json",
    }
