from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from config.loader import ConfigLoader
from config.models import DataSourceConfig, MetricComponentConfig, MetricProfile, ResolvedExperiment
from training.data import ReasoningExample, load_reasoning_source, load_reasoning_split
from training.eval import (
    CheckpointEvaluator,
    CheckpointRef,
    VllmCheckpointPredictor,
    write_checkpoint_eval_artifacts,
)
from training.metrics import MetricStack, build_default_metric_registry, build_metric_stack


DEFAULT_KAGGLE_MODEL = "metric/nemotron-3-nano-30b-a3b-bf16"
DEFAULT_BOXED_ANSWER_HINT = (
    "Please put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)


def load_experiment(
    *,
    recipe_name: str,
    config_root: str | Path,
    data_profile: str | None = None,
    metric_profile: str | None = None,
) -> ResolvedExperiment:
    overrides: dict[str, Any] = {}
    if data_profile:
        overrides["refs.data"] = data_profile
    if metric_profile:
        overrides["refs.metric"] = metric_profile

    loader = ConfigLoader(config_root=config_root)
    return loader.resolve(recipe_name, overrides=overrides or None)


def resolve_validation_examples(
    experiment: ResolvedExperiment,
    *,
    validation_path: str | Path | None = None,
    validation_format: str | None = None,
    project_root: str | Path | None = None,
) -> list[ReasoningExample]:
    if validation_path is not None:
        source_path = Path(validation_path)
        source_format = validation_format or source_path.suffix.lstrip(".") or "jsonl"
        source = DataSourceConfig(
            name="adhoc_validation",
            path=str(source_path),
            format=source_format,
            split="validation",
            prompt_field="prompt",
            answer_field="answer",
        )
        return load_reasoning_source(source, split="validation", project_root=project_root)

    if experiment.data is None:
        raise ValueError("Validation runner requires a resolved data profile or an explicit --validation-path.")
    if not experiment.data.validation_sources:
        raise ValueError("Resolved data profile has no validation_sources.")
    return load_reasoning_split(experiment.data, "validation", project_root=project_root)


def resolve_metric_stack(
    experiment: ResolvedExperiment,
    *,
    metric_names: Sequence[str] = (),
) -> MetricStack:
    if metric_names:
        profile = MetricProfile(
            name="adhoc_metric_selection",
            components=tuple(MetricComponentConfig(name=name) for name in metric_names),
        )
    else:
        profile = experiment.metric

    stack = build_metric_stack(profile, registry=build_default_metric_registry())
    if not stack.component_names:
        raise ValueError("Validation runner requires at least one metric. Use recipe metric profile or pass --metric.")
    return stack


def normalize_kaggle_model_handle(handle: str) -> str:
    normalized = handle.strip().strip("/")
    if normalized.count("/") >= 3:
        return normalized
    return f"{normalized}/transformers/default"


def resolve_base_model_path(
    experiment: ResolvedExperiment,
    *,
    model_source: str,
    model_path: str | Path | None = None,
    kaggle_model: str = DEFAULT_KAGGLE_MODEL,
    hf_model_id: str | None = None,
) -> tuple[str, str]:
    if model_path is not None:
        return str(Path(model_path)), "local"
    if model_source == "local":
        raise ValueError("--model-source local requires --model-path.")

    if model_source in {"auto", "kaggle"}:
        try:
            import kagglehub

            resolved = kagglehub.model_download(normalize_kaggle_model_handle(kaggle_model))
            return str(resolved), "kaggle"
        except Exception as exc:
            if model_source == "kaggle":
                raise RuntimeError(f"Failed to resolve Kaggle model '{kaggle_model}': {exc}") from exc

    resolved_hf = hf_model_id or experiment.model.base_model
    return resolved_hf, "hf"


def build_checkpoint_ref(
    experiment: ResolvedExperiment,
    *,
    base_model_path: str,
    checkpoint_path: str | Path | None = None,
) -> CheckpointRef:
    tokenizer_path = _resolve_tokenizer_path(experiment, base_model_path)
    if checkpoint_path is None:
        return CheckpointRef(
            name="base_model",
            model_path=base_model_path,
            tokenizer_path=tokenizer_path,
            revision=experiment.model.revision if not _is_existing_path(base_model_path) else None,
            metadata={"target": "base_model"},
        )

    checkpoint_dir = Path(checkpoint_path)
    if _is_adapter_checkpoint(checkpoint_dir):
        return CheckpointRef(
            name=checkpoint_dir.name,
            model_path=base_model_path,
            tokenizer_path=tokenizer_path,
            revision=experiment.model.revision if not _is_existing_path(base_model_path) else None,
            adapter_path=str(checkpoint_dir),
            metadata={"target": "adapter_checkpoint"},
        )

    full_model_path = str(checkpoint_dir)
    return CheckpointRef(
        name=checkpoint_dir.name,
        model_path=full_model_path,
        tokenizer_path=_resolve_tokenizer_path(experiment, full_model_path),
        metadata={"target": "model_checkpoint"},
    )


def run_validation(
    *,
    experiment: ResolvedExperiment,
    metric_stack: MetricStack,
    checkpoint: CheckpointRef,
    examples: list[ReasoningExample],
    output_dir: str | Path,
    max_samples: int | None,
    seed: int,
    predictor_kwargs: dict[str, Any],
    project_root: str | Path | None = None,
    include_data_profile_summary: bool = True,
) -> dict[str, Any]:
    predictor = VllmCheckpointPredictor(**predictor_kwargs)
    evaluator = CheckpointEvaluator(predictor)
    result = evaluator.evaluate(
        checkpoint,
        examples,
        data_profile=experiment.data if include_data_profile_summary else None,
        project_root=project_root,
        max_samples=max_samples,
        seed=seed,
        metric_stack=metric_stack,
    )

    artifact_paths = write_checkpoint_eval_artifacts(result, output_dir)
    manifest = {
        "recipe": experiment.recipe.name,
        "data_profile": None if experiment.data is None else experiment.data.name,
        "metric_components": list(metric_stack.component_names),
        "checkpoint": asdict(checkpoint),
        "predictor_kwargs": predictor_kwargs,
        "artifacts": {name: str(path) for name, path in artifact_paths.items()},
        "metrics": result.metrics,
    }
    manifest_path = Path(output_dir) / "validation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "metrics": result.metrics,
        "artifact_paths": artifact_paths,
        "manifest_path": manifest_path,
    }


def _resolve_tokenizer_path(experiment: ResolvedExperiment, model_path: str) -> str:
    if _is_existing_path(model_path):
        return model_path
    return experiment.model.tokenizer or model_path


def _is_adapter_checkpoint(path: Path) -> bool:
    return path.exists() and (path / "adapter_config.json").exists()


def _is_existing_path(path: str) -> bool:
    return Path(path).exists()
