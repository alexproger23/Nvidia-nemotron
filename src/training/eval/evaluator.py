from __future__ import annotations

import random
from pathlib import Path

from config.models import DataProfile
from training.data import ReasoningExample, summarize_reasoning_profile
from training.eval.contracts import (
    CheckpointEvalResult,
    CheckpointPrediction,
    CheckpointPredictor,
    CheckpointRef,
)
from training.eval.metrics import compute_prediction_metrics
from training.metrics import MetricStack


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
        metric_stack: MetricStack | None = None,
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
        metrics = compute_prediction_metrics(
            predictions,
            sampled_examples=len(eval_examples),
            metric_stack=metric_stack,
        )

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
