from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from config.models import ResolvedExperiment, StageConfig

if TYPE_CHECKING:
    from training.tracking import ExperimentLogger


@dataclass(slots=True)
class ArtifactRef:
    name: str
    path: Path
    kind: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StageContext:
    experiment: ResolvedExperiment
    stage_name: str
    stage_config: StageConfig
    run_id: str
    output_dir: Path
    input_artifacts: dict[str, ArtifactRef] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    logger: ExperimentLogger | None = None


@dataclass(slots=True)
class StageResult:
    stage_name: str
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, ArtifactRef] = field(default_factory=dict)
    checkpoint: ArtifactRef | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunResult:
    run_id: str
    stages: dict[str, StageResult] = field(default_factory=dict)


class Stage(Protocol):
    name: str

    def run(self, context: StageContext) -> StageResult:
        """Execute a stage and return its structured result."""
