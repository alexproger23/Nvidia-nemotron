from __future__ import annotations

import random
from pathlib import Path

from training.data.competition import discover_competition_data, load_competition_split
from training.data.contracts import ReasoningExample
from training.data.reasoning import write_reasoning_jsonl


def prepare_reasoning_dataset_from_competition(
    source: str | Path = "data/raw",
    *,
    output_dir: str | Path = "data/reasoning",
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Path]:
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in the [0.0, 1.0) interval")

    competition_source = discover_competition_data(source)
    train_rows = load_competition_split(competition_source, "train")
    shuffled_rows = list(train_rows)
    random.Random(seed).shuffle(shuffled_rows)

    validation_count = _validation_count(len(shuffled_rows), validation_ratio)
    validation_rows = shuffled_rows[:validation_count]
    train_rows = shuffled_rows[validation_count:]

    output_root = Path(output_dir)
    train_path = output_root / "train.jsonl"
    validation_path = output_root / "validation.jsonl"

    write_reasoning_jsonl(_to_reasoning_examples(train_rows, split="train"), train_path)
    write_reasoning_jsonl(_to_reasoning_examples(validation_rows, split="validation"), validation_path)

    return {
        "train": train_path,
        "validation": validation_path,
    }


def _to_reasoning_examples(rows: list, *, split: str) -> list[ReasoningExample]:
    return [
        ReasoningExample(
            source_name=f"competition_{split}",
            split=split,
            prompt=row.prompt,
            answer=row.answer or "",
            example_id=row.example_id,
            metadata={"source_split": row.split},
        )
        for row in rows
    ]


def _validation_count(total_rows: int, validation_ratio: float) -> int:
    if total_rows < 2 or validation_ratio == 0.0:
        return 0

    count = int(total_rows * validation_ratio)
    if count == 0:
        count = 1
    if count >= total_rows:
        count = total_rows - 1
    return count
