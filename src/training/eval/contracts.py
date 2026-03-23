from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from config.models import ModelProfile
from training.data import ReasoningExample


@dataclass(slots=True)
class CheckpointRef:
    name: str
    model_path: str
    tokenizer_path: str | None = None
    revision: str | None = None
    adapter_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CheckpointPrediction:
    example_id: str | None
    prompt: str
    target_answer: str | None
    prediction: str
    raw_output: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CheckpointEvalResult:
    checkpoint: CheckpointRef
    predictor_name: str
    metrics: dict[str, Any]
    predictions: list[CheckpointPrediction]
    dataset_summary: dict[str, Any]


class CheckpointPredictor(Protocol):
    name: str

    def predict_many(
        self,
        checkpoint: CheckpointRef,
        examples: list[ReasoningExample],
    ) -> list[CheckpointPrediction]:
        """Predict answers for the provided examples."""


def checkpoint_from_model_profile(
    model: ModelProfile,
    *,
    name: str = "base_model",
    adapter_path: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> CheckpointRef:
    return CheckpointRef(
        name=name,
        model_path=model.base_model,
        tokenizer_path=model.tokenizer,
        revision=model.revision,
        adapter_path=adapter_path,
        metadata=metadata or {},
    )
