from __future__ import annotations

from pathlib import Path

from config.models import ResolvedExperiment
from training.contracts import ArtifactRef, RunResult, StageContext
from training.registry import StageRegistry
from training.tracking import ExperimentLogger


class RecipeRunner:
    def __init__(self, stage_registry: StageRegistry | None = None) -> None:
        self.stage_registry = stage_registry or StageRegistry()

    def run(self, experiment: ResolvedExperiment, run_id: str) -> RunResult:
        results = RunResult(run_id=run_id)
        artifacts: dict[str, ArtifactRef] = {}

        # Создаём логгер на весь run
        logger = ExperimentLogger(experiment=experiment, run_id=run_id)
        logger.start()

        try:
            for stage_index, stage in enumerate(experiment.enabled_stages()):
                stage_impl = self.stage_registry.create(stage.name)
                output_dir = self._stage_output_dir(experiment, run_id, stage.name)
                output_dir.mkdir(parents=True, exist_ok=True)

                # Создаём logger для конкретной стадии
                stage_logger = ExperimentLogger(
                    experiment=experiment,
                    run_id=run_id,
                    stage_name=stage.name,
                )
                # Используем тот же local_store, но start() вызываем для стадии
                stage_logger.start()

                context = StageContext(
                    experiment=experiment,
                    stage_name=stage.name,
                    stage_config=stage.config,
                    run_id=run_id,
                    output_dir=output_dir,
                    input_artifacts=dict(artifacts),
                    logger=stage_logger,
                )
                result = stage_impl.run(context)
                results.stages[stage.name] = result
                artifacts.update(result.artifacts)
                if result.checkpoint is not None:
                    artifacts["checkpoint"] = result.checkpoint

                # Логирование итогов стадии и завершение
                logger.log(result.metrics, step=stage_index)
                stage_logger.finish()

        finally:
            logger.finish()

        return results

    def _stage_output_dir(self, experiment: ResolvedExperiment, run_id: str, stage_name: str) -> Path:
        root = Path(experiment.tracking.output_root)
        run_subdir = experiment.recipe.run.output_subdir or experiment.name
        return root / run_subdir / run_id / stage_name
