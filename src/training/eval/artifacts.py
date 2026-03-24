from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from training.eval.contracts import CheckpointEvalResult, CheckpointPrediction


def write_checkpoint_eval_artifacts(result: CheckpointEvalResult, output_dir: str | Path) -> dict[str, Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    metrics_path = output_root / "metrics_summary.json"
    predictions_path = output_root / "predictions.jsonl"
    dataset_summary_path = output_root / "dataset_summary.json"
    checkpoint_path = output_root / "checkpoint_ref.json"

    _write_json(metrics_path, result.metrics)
    _write_json(checkpoint_path, _checkpoint_payload(result))
    _write_json(dataset_summary_path, result.dataset_summary)
    _write_predictions(predictions_path, result.predictions)

    return {
        "metrics": metrics_path,
        "predictions": predictions_path,
        "dataset_summary": dataset_summary_path,
        "checkpoint": checkpoint_path,
    }


def _checkpoint_payload(result: CheckpointEvalResult) -> dict[str, Any]:
    payload = asdict(result.checkpoint)
    payload["predictor_name"] = result.predictor_name
    return payload


def _write_predictions(output_path: Path, predictions: list[CheckpointPrediction]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        for prediction in predictions:
            handle.write(json.dumps(asdict(prediction), ensure_ascii=False) + "\n")


def _write_json(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
