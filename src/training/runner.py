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
        # Ранняя валидация: все стадии должны быть зарегистрированы
        self._validate_enabled_stages(experiment)

        results = RunResult(run_id=run_id)
        artifacts: dict[str, ArtifactRef] = {}

        # Один logger на весь run
        logger = ExperimentLogger(experiment=experiment, run_id=run_id)
        logger.start()

        try:
            for stage_index, stage in enumerate(experiment.enabled_stages()):
                # Регистрация старта стадии
                logger._local_store.register_stage_start(
                    run_id=run_id,
                    stage_name=stage.name,
                    stage_index=stage_index,
                )

                stage_impl = self.stage_registry.create(stage.name)
                output_dir = self._stage_output_dir(experiment, run_id, stage.name)
                output_dir.mkdir(parents=True, exist_ok=True)

                context = StageContext(
                    experiment=experiment,
                    stage_name=stage.name,
                    stage_config=stage.config,
                    run_id=run_id,
                    output_dir=output_dir,
                    input_artifacts=dict(artifacts),
                    logger=logger,  # Передаём один logger на все стадии
                )
                result = stage_impl.run(context)
                results.stages[stage.name] = result
                artifacts.update(result.artifacts)
                if result.checkpoint is not None:
                    artifacts["checkpoint"] = result.checkpoint

                # Логирование итогов стадии с указанием имени стадии
                logger.log(result.metrics, step=stage_index, stage=stage.name)

                # Регистрация завершения стадии
                logger._local_store.register_stage_finish(
                    run_id=run_id,
                    stage_name=stage.name,
                    status="completed",
                )

        finally:
            logger.finish()

        return results

    def _validate_enabled_stages(self, experiment: ResolvedExperiment) -> None:
        """Проверить что все включённые стадии зарегистрированы."""
        missing = []
        for stage in experiment.enabled_stages():
            if stage.name not in self.stage_registry._factories:
                missing.append(stage.name)
        if missing:
            raise KeyError(f"Stages not registered: {', '.join(missing)}")

    def _stage_output_dir(self, experiment: ResolvedExperiment, run_id: str, stage_name: str) -> Path:
        root = Path(experiment.tracking.output_root)
        run_subdir = experiment.recipe.run.output_subdir or experiment.name
        return root / run_subdir / run_id / stage_name
