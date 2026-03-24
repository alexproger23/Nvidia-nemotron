from __future__ import annotations

import math
import re
from typing import Any

from config.models import MetricComponentConfig

from .contracts import MetricFunction, MetricValue
from .utils import named_metric, rows_for_kwargs


def register_user_metrics(register: Any) -> None:
    """Register custom metrics here."""

    register("nvidia_metric", build_nvidia_metric)


def build_nvidia_metric(component: MetricComponentConfig) -> MetricFunction:
    relative_tolerance = float(component.params.get("relative_tolerance", 1e-2))
    absolute_tolerance = float(component.params.get("absolute_tolerance", 1e-5))

    def metric_function(
        *,
        completions: list[Any] | None = None,
        answer: list[Any] | None = None,
        **kwargs: Any,
    ) -> list[MetricValue]:
        rows = completions or rows_for_kwargs(kwargs)
        answers = answer or []

        scores: list[MetricValue] = []
        for index, row in enumerate(rows):
            if index >= len(answers) or answers[index] is None:
                scores.append(None)
                continue

            prediction = extract_final_answer(_completion_text(row))
            target = str(answers[index])
            is_match = verify_answer(
                target,
                prediction,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
            scores.append(1.0 if is_match else 0.0)

        return scores

    return named_metric("nvidia_metric", metric_function)


def extract_final_answer(text: str | None) -> str:
    if text is None:
        return "NOT_FOUND"

    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", text)
    if matches:
        non_empty = [match.strip() for match in matches if match.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()

    patterns = (
        r"The final answer is:\s*([^\n]+)",
        r"Final answer is:\s*([^\n]+)",
        r"Final answer\s*:\s*([^\n]+)",
        r"final answer\s*:\s*([^\n]+)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if matches:
        return matches[-1]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "NOT_FOUND"


def verify_answer(
    stored_answer: str,
    predicted: str,
    *,
    relative_tolerance: float = 1e-2,
    absolute_tolerance: float = 1e-5,
) -> bool:
    expected = stored_answer.strip()
    actual = predicted.strip()

    try:
        expected_number = float(expected)
        actual_number = float(actual)
    except Exception:
        return actual.lower() == expected.lower()

    return math.isclose(
        expected_number,
        actual_number,
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    )


def _completion_text(row: Any) -> str:
    if isinstance(row, str):
        return row
    if isinstance(row, dict):
        for key in ("content", "text"):
            value = row.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                return "\n".join(_completion_text(item) for item in value)
        return str(row)
    if isinstance(row, list):
        return "\n".join(_completion_text(item) for item in row)
    return str(row)
