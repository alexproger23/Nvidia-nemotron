from __future__ import annotations

from pathlib import Path
from typing import Any

from config.models import ResolvedExperiment
from training.tracking.local_store import LocalStore


class ExperimentLogger:
    """
    Логгер экспериментов с записью в W&B и локально (Parquet/DuckDB).
    
    W&B — основное хранилище метрик.
    Локальное хранилище — резервная копия и быстрые запросы.
    """

    def __init__(
        self,
        experiment: ResolvedExperiment,
        run_id: str,
        stage_name: str = "unknown",
    ) -> None:
        self.experiment = experiment
        self.run_id = run_id
        self.stage_name = stage_name
        self.tracking = experiment.tracking

        self._local_store = LocalStore(
            parquet_root=Path(self.tracking.parquet_root),
            duckdb_path=Path(self.tracking.duckdb_path),
        )
        self._wandb_initialized = False
        self._started = False

    def start(self) -> None:
        """Инициализировать логгер (start run)."""
        if self._started:
            return

        # Инициализация локального хранилища
        self._local_store.init_run(self.run_id, self.stage_name)
        self._local_store.register_run_start(self.run_id, self.experiment.name)

        # Инициализация W&B
        self._init_wandb()

        self._started = True

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

    def log(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """
        Записать метрики в W&B и локально.
        
        Args:
            metrics: Словарь метрик {name: value}.
            step: Опциональный номер шага.
        """
        if not self._started:
            self.start()

        # Локальная запись
        self._local_store.write_metrics(
            run_id=self.run_id,
            stage=self.stage_name,
            metrics=metrics,
            step=step,
        )

        # W&B запись
        if self._wandb_initialized:
            try:
                import wandb

                wandb.log(metrics, step=step)
            except Exception:
                pass  # Игнорируем ошибки W&B, локально всё записано

    def log_artifact(
        self,
        path: Path,
        name: str,
        artifact_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Залогировать артефакт.
        
        Args:
            path: Путь к файлу артефакта.
            name: Имя артефакта.
            artifact_type: Тип артефакта (например, "model", "data").
            metadata: Дополнительные метаданные.
        """
        # Локальная регистрация
        self._local_store.log_artifact(
            run_id=self.run_id,
            stage=self.stage_name,
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
    ) -> None:
        """
        Залогировать checkpoint модели.
        
        Args:
            path: Путь к checkpoint.
            step: Номер шага.
            is_best: Флаг лучшего checkpoint.
        """
        metadata = {"step": step, "is_best": is_best}
        self.log_artifact(
            path=path,
            name=f"checkpoint-{step}",
            artifact_type="model",
            metadata=metadata,
        )

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
