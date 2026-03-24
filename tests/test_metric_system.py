from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config.loader import ConfigLoader
from training.metrics import build_metric_stack, extract_final_answer, verify_answer


class NvidiaMetricHelpersTest(unittest.TestCase):
    def test_extract_final_answer_prefers_last_boxed_answer(self) -> None:
        text = "Reasoning...\nThe answer is \\boxed{42}\nIgnore \\boxed{1}"
        self.assertEqual(extract_final_answer(text), "1")

    def test_verify_answer_uses_numeric_tolerance(self) -> None:
        self.assertTrue(verify_answer("28.55", "28.549"))
        self.assertFalse(verify_answer("28.55", "27.0"))


class MetricConfigResolutionTest(unittest.TestCase):
    def test_recipe_resolves_metric_profile(self) -> None:
        loader = ConfigLoader(config_root=PROJECT_ROOT / "config")
        experiment = loader.resolve("rl_bootstrap")

        self.assertIsNotNone(experiment.metric)
        assert experiment.metric is not None
        self.assertEqual(experiment.metric.name, "nvidia_proxy_v1")

        metric_stack = build_metric_stack(experiment.metric)
        self.assertEqual(metric_stack.component_names, ("nvidia_metric",))
        self.assertEqual(len(metric_stack.functions), 1)


if __name__ == "__main__":
    unittest.main()
