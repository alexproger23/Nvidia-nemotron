from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from config.errors import ConfigValidationError
from config.models import (
    AdapterConfig,
    BaselineEvalStageConfig,
    DataProfile,
    DataSourceConfig,
    FinalEvalStageConfig,
    ModelProfile,
    RunConfig,
    SftStageConfig,
    StageConfig,
    TrackingProfile,
)

DomainParser = Callable[[str, Mapping[str, Any]], Any]
StageParser = Callable[[str, Mapping[str, Any]], StageConfig]


@dataclass(slots=True)
class DomainSpec:
    name: str
    directory: str
    parser: DomainParser


class ConfigRegistry:
    def __init__(self) -> None:
        self._domains: dict[str, DomainSpec] = {}
        self._stages: dict[str, StageParser] = {}

    def register_domain(self, name: str, directory: str, parser: DomainParser) -> None:
        self._domains[name] = DomainSpec(name=name, directory=directory, parser=parser)

    def register_stage(self, name: str, parser: StageParser) -> None:
        self._stages[name] = parser

    def get_domain(self, name: str) -> DomainSpec:
        try:
            return self._domains[name]
        except KeyError as exc:
            raise ConfigValidationError("registry", f"Unknown config domain: {name}") from exc

    def parse_domain(self, name: str, profile_name: str, raw: Mapping[str, Any]) -> Any:
        return self.get_domain(name).parser(profile_name, raw)

    def parse_stage(self, name: str, raw: Mapping[str, Any]) -> StageConfig:
        try:
            parser = self._stages[name]
        except KeyError as exc:
            raise ConfigValidationError("registry", f"Unknown stage in recipe: {name}") from exc
        return parser(name, raw)

    def parse_run(self, raw: Mapping[str, Any]) -> RunConfig:
        return parse_run_config(raw)


def build_default_registry() -> ConfigRegistry:
    registry = ConfigRegistry()
    registry.register_domain("model", "model", parse_model_profile)
    registry.register_domain("data", "data", parse_data_profile)
    registry.register_domain("tracking", "tracking", parse_tracking_profile)

    registry.register_stage("baseline_eval", parse_baseline_eval_stage)
    registry.register_stage("sft", parse_sft_stage)
    registry.register_stage("final_eval", parse_final_eval_stage)
    return registry


def parse_run_config(raw: Mapping[str, Any]) -> RunConfig:
    context = "recipe.run"
    reject_unknown_keys(raw, {"seed", "output_subdir", "tags", "notes", "parent_run_id"}, context)
    return RunConfig(
        seed=optional_int(raw, "seed", context) or 42,
        output_subdir=optional_str(raw, "output_subdir", context),
        tags=optional_str_list(raw, "tags", context),
        notes=optional_str(raw, "notes", context),
        parent_run_id=optional_str(raw, "parent_run_id", context),
    )


def parse_model_profile(profile_name: str, raw: Mapping[str, Any]) -> ModelProfile:
    context = f"model:{profile_name}"
    reject_unknown_keys(
        raw,
        {"name", "base_model", "tokenizer", "revision", "dtype", "max_context_tokens", "adapter"},
        context,
    )
    return ModelProfile(
        name=optional_str(raw, "name", context) or profile_name,
        base_model=required_str(raw, "base_model", context),
        tokenizer=optional_str(raw, "tokenizer", context),
        revision=optional_str(raw, "revision", context),
        dtype=optional_str(raw, "dtype", context) or "bfloat16",
        max_context_tokens=optional_int(raw, "max_context_tokens", context) or 8192,
        adapter=parse_adapter_config(optional_mapping(raw, "adapter", context)),
    )


def parse_data_profile(profile_name: str, raw: Mapping[str, Any]) -> DataProfile:
    context = f"data:{profile_name}"
    reject_unknown_keys(
        raw,
        {
            "name",
            "dataset_format",
            "max_sequence_length",
            "packing",
            "shuffle",
            "train_sources",
            "validation_sources",
        },
        context,
    )
    return DataProfile(
        name=optional_str(raw, "name", context) or profile_name,
        dataset_format=optional_str(raw, "dataset_format", context) or "chat",
        max_sequence_length=optional_int(raw, "max_sequence_length", context) or 4096,
        packing=optional_bool(raw, "packing", context, True),
        shuffle=optional_bool(raw, "shuffle", context, True),
        train_sources=parse_data_source_list(raw, "train_sources", context),
        validation_sources=parse_data_source_list(raw, "validation_sources", context),
    )


def parse_tracking_profile(profile_name: str, raw: Mapping[str, Any]) -> TrackingProfile:
    context = f"tracking:{profile_name}"
    reject_unknown_keys(
        raw,
        {
            "name",
            "mode",
            "project",
            "entity",
            "tags",
            "output_root",
            "parquet_root",
            "duckdb_path",
            "log_interval_steps",
            "save_resolved_config",
        },
        context,
    )
    return TrackingProfile(
        name=optional_str(raw, "name", context) or profile_name,
        mode=optional_str(raw, "mode", context) or "offline",
        project=optional_str(raw, "project", context) or "nvidia-kaggle",
        entity=optional_str(raw, "entity", context),
        tags=optional_str_list(raw, "tags", context),
        output_root=optional_str(raw, "output_root", context) or "out/runs",
        parquet_root=optional_str(raw, "parquet_root", context) or "out/analytics/parquet",
        duckdb_path=optional_str(raw, "duckdb_path", context) or "out/analytics/experiments.duckdb",
        log_interval_steps=optional_int(raw, "log_interval_steps", context) or 10,
        save_resolved_config=optional_bool(raw, "save_resolved_config", context, True),
    )


def parse_baseline_eval_stage(stage_name: str, raw: Mapping[str, Any]) -> BaselineEvalStageConfig:
    context = f"recipe.stages.{stage_name}"
    reject_unknown_keys(
        raw,
        {
            "enabled",
            "depends_on",
            "tags",
            "notes",
            "eval_suite",
            "prompt_profile",
            "max_samples",
            "reasoning_mode",
            "temperature",
            "top_p",
            "max_new_tokens",
        },
        context,
    )
    return BaselineEvalStageConfig(
        **parse_stage_base_fields(raw, context),
        eval_suite=optional_str(raw, "eval_suite", context) or "proxy_reasoning_v1",
        prompt_profile=optional_str(raw, "prompt_profile", context) or "reasoning_v1",
        max_samples=optional_int(raw, "max_samples", context),
        reasoning_mode=optional_str(raw, "reasoning_mode", context),
        temperature=optional_float(raw, "temperature", context, 0.0),
        top_p=optional_float(raw, "top_p", context, 1.0),
        max_new_tokens=optional_int(raw, "max_new_tokens", context) or 2048,
    )


def parse_sft_stage(stage_name: str, raw: Mapping[str, Any]) -> SftStageConfig:
    context = f"recipe.stages.{stage_name}"
    reject_unknown_keys(
        raw,
        {
            "enabled",
            "depends_on",
            "tags",
            "notes",
            "learning_rate",
            "epochs",
            "per_device_batch_size",
            "gradient_accumulation_steps",
            "max_train_samples",
            "max_eval_samples",
            "save_steps",
            "eval_steps",
            "train_split",
            "validation_split",
        },
        context,
    )
    return SftStageConfig(
        **parse_stage_base_fields(raw, context),
        learning_rate=optional_float(raw, "learning_rate", context, 2e-4),
        epochs=optional_float(raw, "epochs", context, 1.0),
        per_device_batch_size=optional_int(raw, "per_device_batch_size", context) or 1,
        gradient_accumulation_steps=optional_int(raw, "gradient_accumulation_steps", context) or 1,
        max_train_samples=optional_int(raw, "max_train_samples", context),
        max_eval_samples=optional_int(raw, "max_eval_samples", context),
        save_steps=optional_int(raw, "save_steps", context) or 100,
        eval_steps=optional_int(raw, "eval_steps", context) or 100,
        train_split=optional_str(raw, "train_split", context) or "train",
        validation_split=optional_str(raw, "validation_split", context) or "validation",
    )


def parse_final_eval_stage(stage_name: str, raw: Mapping[str, Any]) -> FinalEvalStageConfig:
    context = f"recipe.stages.{stage_name}"
    reject_unknown_keys(
        raw,
        {
            "enabled",
            "depends_on",
            "tags",
            "notes",
            "eval_suite",
            "candidate_source",
            "compare_to",
            "max_samples",
        },
        context,
    )
    return FinalEvalStageConfig(
        **parse_stage_base_fields(raw, context),
        eval_suite=optional_str(raw, "eval_suite", context) or "final_suite_v1",
        candidate_source=optional_str(raw, "candidate_source", context) or "best_sft",
        compare_to=optional_str_list(raw, "compare_to", context),
        max_samples=optional_int(raw, "max_samples", context),
    )


def parse_stage_base_fields(raw: Mapping[str, Any], context: str) -> dict[str, Any]:
    return {
        "enabled": optional_bool(raw, "enabled", context, True),
        "depends_on": optional_str_list(raw, "depends_on", context),
        "tags": optional_str_list(raw, "tags", context),
        "notes": optional_str(raw, "notes", context),
    }


def parse_adapter_config(raw: Mapping[str, Any]) -> AdapterConfig:
    context = "model.adapter"
    reject_unknown_keys(raw, {"method", "rank", "alpha", "dropout", "target_modules"}, context)
    return AdapterConfig(
        method=optional_str(raw, "method", context) or "lora",
        rank=optional_int(raw, "rank", context) or 64,
        alpha=optional_int(raw, "alpha", context) or 128,
        dropout=optional_float(raw, "dropout", context, 0.05),
        target_modules=optional_str_list(raw, "target_modules", context),
    )


def parse_data_source_list(raw: Mapping[str, Any], key: str, context: str) -> tuple[DataSourceConfig, ...]:
    values = raw.get(key, ())
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ConfigValidationError(context, f"'{key}' must be an array of tables")

    items: list[DataSourceConfig] = []
    for index, value in enumerate(values):
        source_raw = expect_mapping(value, f"{context}.{key}[{index}]")
        items.append(parse_data_source(source_raw, f"{context}.{key}[{index}]"))
    return tuple(items)


def parse_data_source(raw: Mapping[str, Any], context: str) -> DataSourceConfig:
    reject_unknown_keys(
        raw,
        {"name", "path", "format", "split", "weight", "prompt_field", "answer_field"},
        context,
    )
    weight = raw.get("weight", 1.0)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise ConfigValidationError(context, "'weight' must be a float")
    return DataSourceConfig(
        name=required_str(raw, "name", context),
        path=required_str(raw, "path", context),
        format=optional_str(raw, "format", context) or "jsonl",
        split=optional_str(raw, "split", context) or "train",
        weight=float(weight),
        prompt_field=optional_str(raw, "prompt_field", context),
        answer_field=optional_str(raw, "answer_field", context),
    )


def expect_mapping(raw: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ConfigValidationError(context, "Expected a table/object")
    return raw


def reject_unknown_keys(raw: Mapping[str, Any], allowed_keys: set[str], context: str) -> None:
    unknown_keys = sorted(set(raw) - allowed_keys)
    if unknown_keys:
        raise ConfigValidationError(context, f"Unknown keys: {', '.join(unknown_keys)}")


def required_str(raw: Mapping[str, Any], key: str, context: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(context, f"'{key}' must be a non-empty string")
    return value


def optional_str(raw: Mapping[str, Any], key: str, context: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigValidationError(context, f"'{key}' must be a string")
    return value


def optional_int(raw: Mapping[str, Any], key: str, context: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigValidationError(context, f"'{key}' must be an integer")
    return value


def optional_float(raw: Mapping[str, Any], key: str, context: str, default: float) -> float:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigValidationError(context, f"'{key}' must be a float")
    return float(value)


def optional_bool(raw: Mapping[str, Any], key: str, context: str, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ConfigValidationError(context, f"'{key}' must be a boolean")
    return value


def optional_str_list(raw: Mapping[str, Any], key: str, context: str) -> tuple[str, ...]:
    value = raw.get(key, ())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigValidationError(context, f"'{key}' must be a list of strings")

    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ConfigValidationError(context, f"'{key}' must contain only non-empty strings")
        items.append(item)
    return tuple(items)


def optional_mapping(raw: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    return expect_mapping(raw.get(key, {}), f"{context}.{key}")
