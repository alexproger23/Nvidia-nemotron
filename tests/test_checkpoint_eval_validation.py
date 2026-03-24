from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from training.eval.contracts import CheckpointPrediction
from training.eval.metrics import compute_prediction_metrics
from training.eval.validation import normalize_kaggle_model_handle
from training.metrics import build_metric_stack
from config.models import MetricComponentConfig, MetricProfile


class CheckpointEvalMetricsTest(unittest.TestCase):
    def test_compute_prediction_metrics_includes_custom_metric(self) -> None:
        predictions = [
            CheckpointPrediction(
                example_id="1",
                prompt="p1",
                target_answer="42",
                prediction="42",
                raw_output="Reasoning... \\boxed{42}",
            ),
            CheckpointPrediction(
                example_id="2",
                prompt="p2",
                target_answer="17",
                prediction="16",
                raw_output="Reasoning... \\boxed{16}",
            ),
        ]
        metric_stack = build_metric_stack(
            MetricProfile(
                name="adhoc",
                components=(MetricComponentConfig(name="nvidia_metric"),),
            )
        )

        metrics = compute_prediction_metrics(
            predictions,
            sampled_examples=2,
            metric_stack=metric_stack,
        )

        self.assertEqual(metrics["evaluated_examples"], 2)
        self.assertEqual(metrics["exact_match"], 0.5)
        self.assertEqual(metrics["metrics/nvidia_metric/coverage"], 1.0)
        self.assertEqual(metrics["metrics/nvidia_metric/mean"], 0.5)


class ValidationHelpersTest(unittest.TestCase):
    def test_normalize_kaggle_model_handle_adds_default_variant(self) -> None:
        self.assertEqual(
            normalize_kaggle_model_handle("metric/nemotron-3-nano-30b-a3b-bf16"),
            "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
        )
        self.assertEqual(
            normalize_kaggle_model_handle("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"),
            "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
        )


if __name__ == "__main__":
    unittest.main()
