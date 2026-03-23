from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AdapterConfig:
    method: str = "lora"
    rank: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ()


@dataclass(slots=True)
class ModelProfile:
    name: str
    base_model: str
    tokenizer: str | None = None
    revision: str | None = None
    dtype: str = "bfloat16"
    max_context_tokens: int = 8192
    adapter: AdapterConfig = field(default_factory=AdapterConfig)


@dataclass(slots=True)
class DataSourceConfig:
    name: str
    path: str
    format: str = "jsonl"
    split: str = "train"
    weight: float = 1.0
    prompt_field: str | None = None
    answer_field: str | None = None


@dataclass(slots=True)
class DataProfile:
    name: str
    dataset_format: str = "chat"
    max_sequence_length: int = 4096
    packing: bool = True
    shuffle: bool = True
    train_sources: tuple[DataSourceConfig, ...] = ()
    validation_sources: tuple[DataSourceConfig, ...] = ()


@dataclass(slots=True)
class TrackingProfile:
    name: str
    mode: str = "offline"
    project: str = "nvidia-kaggle"
    entity: str | None = None
    tags: tuple[str, ...] = ()
    output_root: str = "out/runs"
    parquet_root: str = "out/analytics/parquet"
    duckdb_path: str = "out/analytics/experiments.duckdb"
    log_interval_steps: int = 10
    save_resolved_config: bool = True


@dataclass(slots=True)
class RunConfig:
    seed: int = 42
    output_subdir: str | None = None
    tags: tuple[str, ...] = ()
    notes: str | None = None
    parent_run_id: str | None = None


@dataclass(slots=True)
class StageBaseConfig:
    enabled: bool = True
    depends_on: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    notes: str | None = None


@dataclass(slots=True)
class BaselineEvalStageConfig(StageBaseConfig):
    eval_suite: str = "proxy_reasoning_v1"
    prompt_profile: str = "reasoning_v1"
    max_samples: int | None = None
    reasoning_mode: str | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 2048


@dataclass(slots=True)
class SftStageConfig(StageBaseConfig):
    learning_rate: float = 2e-4
    epochs: float = 1.0
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    save_steps: int = 100
    eval_steps: int = 100
    train_split: str = "train"
    validation_split: str = "validation"


@dataclass(slots=True)
class FinalEvalStageConfig(StageBaseConfig):
    eval_suite: str = "final_suite_v1"
    candidate_source: str = "best_sft"
    compare_to: tuple[str, ...] = ()
    max_samples: int | None = None


StageConfig = BaselineEvalStageConfig | SftStageConfig | FinalEvalStageConfig


@dataclass(slots=True)
class StageDefinition:
    name: str
    config: StageConfig


@dataclass(slots=True)
class RecipeConfig:
    name: str
    description: str | None = None
    refs: dict[str, str] = field(default_factory=dict)
    run: RunConfig = field(default_factory=RunConfig)
    stages: tuple[StageDefinition, ...] = ()

    def get_stage(self, stage_name: str) -> StageDefinition:
        for stage in self.stages:
            if stage.name == stage_name:
                return stage
        raise KeyError(stage_name)


@dataclass(slots=True)
class ResolvedExperiment:
    recipe: RecipeConfig
    model: ModelProfile
    tracking: TrackingProfile
    data: DataProfile | None = None
    source_files: dict[str, Path] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.recipe.name

    def enabled_stages(self) -> tuple[StageDefinition, ...]:
        return tuple(stage for stage in self.recipe.stages if stage.config.enabled)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_files"] = {
            name: str(path)
            for name, path in self.source_files.items()
        }
        return payload

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=False,
        )

    def write_snapshot(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(), encoding="utf-8")
