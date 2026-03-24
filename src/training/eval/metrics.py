from __future__ import annotations

from typing import Any

from training.eval.contracts import CheckpointPrediction
from training.metrics import MetricStack


def compute_prediction_metrics(
    predictions: list[CheckpointPrediction],
    *,
    sampled_examples: int,
    metric_stack: MetricStack | None = None,
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
    metrics: dict[str, Any] = {
        "evaluated_examples": total,
        "sampled_examples": sampled_examples,
        "exact_match": round(exact_matches / total, 6) if total else 0.0,
        "non_empty_prediction_rate": round(non_empty_predictions / total, 6) if total else 0.0,
    }
    if metric_stack is None or not metric_stack.component_names:
        return metrics

    prompts = [prediction.prompt for prediction in predictions]
    completions = [prediction.raw_output or prediction.prediction for prediction in predictions]
    extracted_predictions = [prediction.prediction for prediction in predictions]
    answers = [prediction.target_answer for prediction in predictions]

    for metric_name, metric_func in zip(metric_stack.component_names, metric_stack.functions, strict=True):
        values = metric_func(
            prompts=prompts,
            completions=completions,
            predictions=extracted_predictions,
            answer=answers,
        )
        if len(values) != len(predictions):
            raise ValueError(
                f"Metric '{metric_name}' returned {len(values)} values for {len(predictions)} predictions."
            )

        valid_values = [float(value) for value in values if value is not None]
        coverage = len(valid_values) / total if total else 0.0
        metrics[f"metrics/{metric_name}/coverage"] = round(coverage, 6)
        if not valid_values:
            metrics[f"metrics/{metric_name}/mean"] = None
            metrics[f"metrics/{metric_name}/std"] = None
            continue

        mean_value = sum(valid_values) / len(valid_values)
        variance = sum((value - mean_value) ** 2 for value in valid_values) / len(valid_values)
        metrics[f"metrics/{metric_name}/mean"] = round(mean_value, 6)
        metrics[f"metrics/{metric_name}/std"] = round(variance**0.5, 6)

    metrics["metric_component_count"] = len(metric_stack.component_names)
    return metrics


def _normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().lower()
