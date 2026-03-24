from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from config.models import ResolvedExperiment
from training.tracking.local_store import LocalStore


class ExperimentLogger:
    """
    Логгер экспериментов с записью в W&B и локально (Parquet/DuckDB).
    
    W&B — основное хранилище метрик.
    Локальное хранилище — резервная копия и быстрые запросы.
    
    Один экземпляр на run. Для стадий используется stage_name в метках.
    """

    def __init__(
        self,
        experiment: ResolvedExperiment,
        run_id: str,
    ) -> None:
        self.experiment = experiment
        self.run_id = run_id
        self.tracking = experiment.tracking

        self._local_store = LocalStore(
            parquet_root=Path(self.tracking.parquet_root),
            duckdb_path=Path(self.tracking.duckdb_path),
        )
        self._wandb_initialized = False
        self._started = False
        self._wandb_url: str | None = None

    def start(self) -> None:
        """Инициализировать логгер (start run)."""
        if self._started:
            return

        # Получаем git commit
        git_commit = self._get_git_commit()

        # Собираем список стадий
        stages = [s.name for s in self.experiment.enabled_stages()]

        # Теги
        tags = list(self.tracking.tags) + [self.experiment.name]

        # Инициализация локального хранилища с lineage
        self._local_store.init_run(self.run_id, "global")
        self._local_store.register_run_start(
            run_id=self.run_id,
            recipe=self.experiment.name,
            parent_run_id=self.experiment.recipe.run.parent_run_id,
            git_commit=git_commit,
            seed=self.experiment.recipe.run.seed,
            config_path=str(self.experiment.source_files.get("recipe", "")),
            stages=stages,
            tags=tags,
        )

        # Инициализация W&B
        self._init_wandb()

        self._started = True

    def _get_git_commit(self) -> str | None:
        """Получить текущий git commit."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except Exception:
            return None

    def _init_wandb(self) -> None:
        """Инициализировать W&B run."""
        try:
            import wandb
        except ImportError:
            return  # W&B не установлен, работаем только локально

        mode = self.tracking.mode
        if mode == "disabled":
            return

        # Определяем режим wandb
        if mode == "offline":
            wandb_mode = "offline"
        elif mode == "online":
            wandb_mode = "online"
        else:
            wandb_mode = "offline"  # по умолчанию offline

        # Собираем конфиг для логирования
        config_dict = self.experiment.to_dict()

        # Теги
        tags = list(self.tracking.tags) + [self.experiment.name]

        # Инициализируем run
        wandb.init(
            project=self.tracking.project,
            entity=self.tracking.entity,
            id=self.run_id,
            name=f"{self.experiment.name}/{self.run_id}",
            tags=tags,
            config=config_dict,
            mode=wandb_mode,
            resume="allow",
        )

        self._wandb_initialized = True

        # Сохраняем URL для логирования
        try:
            self._wandb_url = wandb.run.get_url()
        except Exception:
            pass

        # Обновляем local store с wandb_url
        if self._wandb_url:
            self._local_store.register_run_start(
                run_id=self.run_id,
                recipe=self.experiment.name,
                wandb_url=self._wandb_url,
            )

    def log(
        self,
        metrics: dict[str, float],
        step: int | None = None,
        stage: str | None = None,
    ) -> None:
        """
        Записать метрики в W&B и локально.
        
        Args:
            metrics: Словарь метрик {name: value}.
            step: Опциональный номер шага.
            stage: Опциональное имя стадии для локальной записи.
        """
        if not self._started:
            self.start()

        stage_name = stage or "global"

        # Локальная запись
        self._local_store.write_metrics(
            run_id=self.run_id,
            stage=stage_name,
            metrics=metrics,
            step=step,
        )

        # Обновление summary
        self._local_store.update_metrics_summary(
            run_id=self.run_id,
            stage=stage_name,
            metrics=metrics,
            step=step,
        )

        # W&B запись
        if self._wandb_initialized:
            try:
                import wandb

                # Добавляем префикс стадии к метрикам
                prefixed_metrics = metrics
                if stage_name != "global":
                    prefixed_metrics = {
                        f"{stage_name}/{k}": v for k, v in metrics.items()
                    }

                wandb.log(prefixed_metrics, step=step)
            except Exception:
                pass  # Игнорируем ошибки W&B, локально всё записано

    def log_artifact(
        self,
        path: Path,
        name: str,
        artifact_type: str,
        metadata: dict[str, Any] | None = None,
        stage: str | None = None,
    ) -> None:
        """
        Залогировать артефакт.
        
        Args:
            path: Путь к файлу артефакта.
            name: Имя артефакта.
            artifact_type: Тип артефакта (например, "model", "data").
            metadata: Дополнительные метаданные.
            stage: Опциональное имя стадии.
        """
        stage_name = stage or "global"

        # Локальная регистрация
        self._local_store.log_artifact(
            run_id=self.run_id,
            stage=stage_name,
            name=name,
            path=path,
            artifact_type=artifact_type,
            metadata=metadata,
        )

        # W&B артефакт
        if self._wandb_initialized:
            try:
                import wandb

                artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata)
                artifact.add_file(str(path))
                wandb.log_artifact(artifact)
            except Exception:
                pass  # Игнорируем ошибки W&B

    def log_checkpoint(
        self,
        path: Path,
        step: int,
        is_best: bool = False,
        stage: str | None = None,
    ) -> None:
        """
        Залогировать checkpoint модели.
        
        Args:
            path: Путь к checkpoint.
            step: Номер шага.
            is_best: Флаг лучшего checkpoint.
            stage: Опциональное имя стадии.
        """
        stage_name = stage or "global"

        # Локальная регистрация checkpoint
        self._local_store.log_checkpoint(
            run_id=self.run_id,
            stage=stage_name,
            step=step,
            path=path,
            is_best=is_best,
        )

        # Artifact для W&B
        if self._wandb_initialized:
            try:
                import wandb

                metadata = {"step": step, "is_best": is_best, "stage": stage_name}
                artifact = wandb.Artifact(name=f"checkpoint-{step}", type="model", metadata=metadata)
                artifact.add_file(str(path))
                wandb.log_artifact(artifact)
            except Exception:
                pass  # Игнорируем ошибки W&B

    def update_config(self, updates: dict[str, Any]) -> None:
        """Обновить конфиг в W&B."""
        if self._wandb_initialized:
            try:
                import wandb

                wandb.config.update(updates, allow_val_change=True)
            except Exception:
                pass

    def add_tags(self, tags: list[str]) -> None:
        """Добавить теги к run."""
        if self._wandb_initialized:
            try:
                import wandb

                wandb.run.tags = tuple(list(wandb.run.tags or []) + tags)
            except Exception:
                pass

    def finish(self, status: str = "completed") -> None:
        """Завершить логгер (finish run)."""
        if not self._started:
            return

        # Завершение W&B
        if self._wandb_initialized:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass

        # Завершение локального хранилища
        self._local_store.register_run_finish(self.run_id, status)
        self._local_store.close()

        self._started = False
        self._wandb_initialized = False
