from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from config.models import DataProfile, DataSourceConfig
from training.data.contracts import DatasetSummary, ReasoningExample


def load_reasoning_split(
    profile: DataProfile,
    split: str,
    *,
    project_root: str | Path | None = None,
) -> list[ReasoningExample]:
    sources = _select_sources(profile, split)
    rows: list[ReasoningExample] = []
    for source in sources:
        rows.extend(load_reasoning_source(source, split=split, project_root=project_root))
    return rows


def load_reasoning_source(
    source: DataSourceConfig,
    *,
    split: str | None = None,
    project_root: str | Path | None = None,
) -> list[ReasoningExample]:
    source_path = _resolve_source_path(source.path, project_root)
    source_split = split or source.split
    prompt_field = source.prompt_field or "prompt"
    answer_field = source.answer_field or "answer"

    rows: list[ReasoningExample] = []
    for raw_row in _iter_source_rows(source_path, source.format):
        prompt = _require_text(raw_row, prompt_field, source.name)
        answer = _require_text(raw_row, answer_field, source.name)
        metadata = {
            key: value
            for key, value in raw_row.items()
            if key not in {prompt_field, answer_field, "id"}
        }
        rows.append(
            ReasoningExample(
                source_name=source.name,
                split=source_split,
                prompt=prompt,
                answer=answer,
                example_id=_optional_text(raw_row.get("id")),
                metadata=metadata,
            )
        )
    return rows


def summarize_reasoning_profile(
    profile: DataProfile,
    *,
    project_root: str | Path | None = None,
) -> dict[str, list[DatasetSummary]]:
    return {
        "train": [
            summarize_reasoning_source(source, project_root=project_root)
            for source in profile.train_sources
        ],
        "validation": [
            summarize_reasoning_source(source, project_root=project_root)
            for source in profile.validation_sources
        ],
    }


def summarize_reasoning_source(
    source: DataSourceConfig,
    *,
    project_root: str | Path | None = None,
) -> DatasetSummary:
    rows = load_reasoning_source(source, project_root=project_root)
    fields = ["id"]
    if source.prompt_field:
        fields.append(source.prompt_field)
    else:
        fields.append("prompt")
    if source.answer_field:
        fields.append(source.answer_field)
    else:
        fields.append("answer")

    return DatasetSummary(
        name=source.name,
        split=source.split,
        rows=len(rows),
        rows_with_answers=len(rows),
        path=_resolve_source_path(source.path, project_root),
        format=source.format,
        fields=tuple(dict.fromkeys(fields)),
    )


def write_reasoning_jsonl(examples: Iterable[ReasoningExample], output_path: str | Path) -> int:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        for example in examples:
            payload: dict[str, Any] = {
                "prompt": example.prompt,
                "answer": example.answer,
                **example.metadata,
            }
            if example.example_id is not None:
                payload["id"] = example.example_id
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def _select_sources(profile: DataProfile, split: str) -> tuple[DataSourceConfig, ...]:
    if split == "train":
        return profile.train_sources
    if split == "validation":
        return profile.validation_sources
    raise ValueError(f"Unsupported reasoning split: {split}")


def _resolve_source_path(path: str, project_root: str | Path | None) -> Path:
    source_path = Path(path)
    if source_path.is_absolute():
        return source_path

    root = Path(project_root) if project_root is not None else Path.cwd()
    return root / source_path


def _iter_source_rows(path: Path, source_format: str) -> Iterator[dict[str, Any]]:
    source_format = source_format.lower()
    if source_format == "jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise ValueError(f"{path}:{line_number} must contain a JSON object")
                yield payload
        return

    if source_format == "csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield row
        return

    raise ValueError(f"Unsupported reasoning source format: {source_format}")


def _require_text(row: dict[str, Any], key: str, source_name: str) -> str:
    value = _optional_text(row.get(key))
    if value is None:
        raise ValueError(f"Source '{source_name}' is missing required text field: {key}")
    return value


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)
