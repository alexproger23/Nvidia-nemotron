from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from config.models import DataProfile
from training.data import ReasoningExample, summarize_reasoning_profile
from training.eval.contracts import (
    CheckpointEvalResult,
    CheckpointPrediction,
    CheckpointPredictor,
    CheckpointRef,
)


class CheckpointEvaluator:
    def __init__(self, predictor: CheckpointPredictor) -> None:
        self.predictor = predictor

    def evaluate(
        self,
        checkpoint: CheckpointRef,
        examples: list[ReasoningExample],
        *,
        data_profile: DataProfile | None = None,
        project_root: str | Path | None = None,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> CheckpointEvalResult:
        eval_examples = _sample_examples(examples, max_samples=max_samples, seed=seed)
        predictions = self.predictor.predict_many(checkpoint, eval_examples)
        if len(predictions) != len(eval_examples):
            raise ValueError("predictor returned a different number of predictions than examples")

        dataset_summary = (
            _dataset_summary_payload(data_profile, project_root=Path(project_root or Path.cwd()))
            if data_profile is not None
            else {}
        )
        metrics = _compute_metrics(predictions, sampled_examples=len(eval_examples))

        return CheckpointEvalResult(
            checkpoint=checkpoint,
            predictor_name=self.predictor.name,
            metrics=metrics,
            predictions=predictions,
            dataset_summary=dataset_summary,
        )


def _sample_examples(
    examples: list[ReasoningExample],
    *,
    max_samples: int | None,
    seed: int,
) -> list[ReasoningExample]:
    if max_samples is None or max_samples >= len(examples):
        return list(examples)
    return random.Random(seed).sample(examples, k=max_samples)


def _compute_metrics(
    predictions: list[CheckpointPrediction],
    *,
    sampled_examples: int,
) -> dict[str, Any]:
    exact_matches = 0
    non_empty_predictions = 0

    for prediction in predictions:
        normalized_prediction = _normalize_text(prediction.prediction)
        normalized_target = _normalize_text(prediction.target_answer)
        if normalized_prediction:
            non_empty_predictions += 1
        if normalized_target is not None and normalized_prediction == normalized_target:
            exact_matches += 1

    total = len(predictions)
    return {
        "evaluated_examples": total,
        "sampled_examples": sampled_examples,
        "exact_match": round(exact_matches / total, 6) if total else 0.0,
        "non_empty_prediction_rate": round(non_empty_predictions / total, 6) if total else 0.0,
    }


def _dataset_summary_payload(data_profile: DataProfile, *, project_root: Path) -> dict[str, Any]:
    summaries = summarize_reasoning_profile(data_profile, project_root=project_root)
    return {
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
    }


def _normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().lower()
