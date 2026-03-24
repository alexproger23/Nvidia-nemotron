from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb


@dataclass(slots=True)
class MetricBuffer:
    """Буфер для одной метрики перед записью в Parquet."""
    run_id: str
    stage: str
    metric: str
    values: list[float] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)


@dataclass(slots=True)
class LocalStore:
    """Локальное хранилище метрик в Parquet + DuckDB."""

    parquet_root: Path
    duckdb_path: Path
    _conn: duckdb.DuckDBPyConnection | None = field(init=False, default=None)
    _buffers: dict[tuple[str, str, str], MetricBuffer] = field(init=False, default_factory=dict)
    _buffer_size: int = field(default=100)

    def __post_init__(self) -> None:
        self.parquet_root.mkdir(parents=True, exist_ok=True)
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    def init_run(self, run_id: str, stage: str) -> None:
        """Инициализировать хранилище для run + stage."""
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Создать таблицы если не существуют."""
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id VARCHAR PRIMARY KEY,
                recipe VARCHAR,
                parent_run_id VARCHAR,
                git_commit VARCHAR,
                seed INTEGER,
                config_path VARCHAR,
                wandb_url VARCHAR,
                stages VARCHAR,
                started_at TIMESTAMP,
                finished_at TIMESTAMP,
                status VARCHAR
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS run_tags (
                run_id VARCHAR,
                tag VARCHAR,
                PRIMARY KEY (run_id, tag)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stages (
                run_id VARCHAR,
                stage_name VARCHAR,
                stage_index INTEGER,
                status VARCHAR,
                started_at TIMESTAMP,
                finished_at TIMESTAMP,
                PRIMARY KEY (run_id, stage_name)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics_timeseries (
                run_id VARCHAR,
                stage VARCHAR,
                metric VARCHAR,
                value DOUBLE,
                step INTEGER,
                timestamp TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics_summary (
                run_id VARCHAR,
                stage VARCHAR,
                metric VARCHAR,
                min_value DOUBLE,
                max_value DOUBLE,
                last_value DOUBLE,
                last_step INTEGER,
                PRIMARY KEY (run_id, stage, metric)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS artifacts (
                run_id VARCHAR,
                stage VARCHAR,
                name VARCHAR,
                path VARCHAR,
                type VARCHAR,
                metadata VARCHAR,
                timestamp TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                run_id VARCHAR,
                stage VARCHAR,
                step INTEGER,
                path VARCHAR,
                is_best BOOLEAN,
                timestamp TIMESTAMP,
                PRIMARY KEY (run_id, stage, step)
            )
        """)
        conn.commit()

    def write_metrics(
        self,
        run_id: str,
        stage: str,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Записать метрики в буфер + DuckDB. Parquet — при сбросе."""
        timestamp = datetime.now(timezone.utc)
        step_value = step if step is not None else -1

        # Запись в DuckDB (синхронно)
        conn = self._get_connection()
        rows = [
            (run_id, stage, name, value, step_value, timestamp)
            for name, value in metrics.items()
        ]
        conn.executemany(
            """
            INSERT INTO metrics_timeseries (run_id, stage, metric, value, step, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()

        # Буферизация для Parquet
        for metric_name, value in metrics.items():
            key = (run_id, stage, metric_name)
            if key not in self._buffers:
                self._buffers[key] = MetricBuffer(
                    run_id=run_id,
                    stage=stage,
                    metric=metric_name,
                )
            buf = self._buffers[key]
            buf.values.append(value)
            buf.steps.append(step_value)
            buf.timestamps.append(timestamp)

            # Сброс при заполнении
            if len(buf.values) >= self._buffer_size:
                self._flush_buffer(key)

    def flush(self) -> None:
        """Сбросить все буферы в Parquet."""
        keys = list(self._buffers.keys())
        for key in keys:
            self._flush_buffer(key)

    def _flush_buffer(self, key: tuple[str, str, str]) -> None:
        """Сбросить буфер в Parquet файл."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        buf = self._buffers.pop(key, None)
        if buf is None or len(buf.values) == 0:
            return

        parquet_dir = self.parquet_root / buf.run_id / buf.stage
        parquet_dir.mkdir(parents=True, exist_ok=True)
        parquet_file = parquet_dir / f"{buf.metric}.parquet"

        n = len(buf.values)
        table = pa.table({
            "run_id": [buf.run_id] * n,
            "stage": [buf.stage] * n,
            "metric": [buf.metric] * n,
            "value": buf.values,
            "step": buf.steps,
            "timestamp": buf.timestamps,
        })

        if parquet_file.exists():
            existing = pq.read_table(parquet_file)
            combined = pa.concat_tables([existing, table])
            pq.write_table(combined, parquet_file)
        else:
            pq.write_table(table, parquet_file)

    def register_run_start(
        self,
        run_id: str,
        recipe: str,
        parent_run_id: str | None = None,
        git_commit: str | None = None,
        seed: int | None = None,
        config_path: str | None = None,
        wandb_url: str | None = None,
        stages: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Зарегистрировать старт run с lineage информацией."""
        conn = self._get_connection()

        # INSERT OR IGNORE для runs
        conn.execute(
            """
            INSERT OR IGNORE INTO runs (
                run_id, recipe, parent_run_id, git_commit, seed,
                config_path, wandb_url, stages, started_at, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                recipe,
                parent_run_id,
                git_commit,
                seed,
                config_path,
                wandb_url,
                ",".join(stages or []) if stages else None,
                datetime.now(timezone.utc),
                "running",
            ),
        )

        # Теги
        if tags:
            for tag in tags:
                conn.execute(
                    "INSERT OR IGNORE INTO run_tags (run_id, tag) VALUES (?, ?)",
                    (run_id, tag),
                )

        conn.commit()

    def register_stage_start(self, run_id: str, stage_name: str, stage_index: int) -> None:
        """Зарегистрировать старт стадии."""
        conn = self._get_connection()
        conn.execute(
            """
            INSERT OR REPLACE INTO stages (run_id, stage_name, stage_index, status, started_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, stage_name, stage_index, "running", datetime.now(timezone.utc)),
        )
        conn.commit()

    def register_stage_finish(self, run_id: str, stage_name: str, status: str = "completed") -> None:
        """Зарегистрировать завершение стадии."""
        conn = self._get_connection()
        conn.execute(
            """
            UPDATE stages SET finished_at = ?, status = ?
            WHERE run_id = ? AND stage_name = ?
            """,
            (datetime.now(timezone.utc), status, run_id, stage_name),
        )
        conn.commit()

    def register_run_finish(self, run_id: str, status: str = "completed") -> None:
        """Зарегистрировать завершение run."""
        conn = self._get_connection()
        conn.execute(
            """
            UPDATE runs 
            SET finished_at = ?, status = ?
            WHERE run_id = ?
            """,
            (datetime.now(timezone.utc), status, run_id),
        )
        conn.commit()

    def log_checkpoint(
        self,
        run_id: str,
        stage: str,
        step: int,
        path: Path,
        is_best: bool = False,
    ) -> None:
        """Залогировать checkpoint."""
        conn = self._get_connection()
        conn.execute(
            """
            INSERT INTO checkpoints (run_id, stage, step, path, is_best, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, stage, step, str(path), is_best, datetime.now(timezone.utc)),
        )
        conn.commit()

    def update_metrics_summary(
        self,
        run_id: str,
        stage: str,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Обновить summary метрик (min/max/last)."""
        conn = self._get_connection()
        step_value = step if step is not None else -1

        for metric_name, value in metrics.items():
            # Читаем текущее summary
            cur = conn.execute(
                """
                SELECT min_value, max_value, last_value, last_step
                FROM metrics_summary
                WHERE run_id = ? AND stage = ? AND metric = ?
                """,
                (run_id, stage, metric_name),
            ).fetchone()

            if cur is None:
                # Новая метрика
                conn.execute(
                    """
                    INSERT INTO metrics_summary
                    (run_id, stage, metric, min_value, max_value, last_value, last_step)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (run_id, stage, metric_name, value, value, value, step_value),
                )
            else:
                # Обновляем существующую
                min_val = min(cur[0], value)
                max_val = max(cur[1], value)
                conn.execute(
                    """
                    UPDATE metrics_summary
                    SET min_value = ?, max_value = ?, last_value = ?, last_step = ?
                    WHERE run_id = ? AND stage = ? AND metric = ?
                    """,
                    (min_val, max_val, value, step_value, run_id, stage, metric_name),
                )

        conn.commit()

    def log_artifact(
        self,
        run_id: str,
        stage: str,
        name: str,
        path: Path,
        artifact_type: str,
        metadata: dict | None = None,
    ) -> None:
        """Залогировать артефакт."""
        import json

        conn = self._get_connection()
        conn.execute(
            """
            INSERT INTO artifacts (run_id, stage, name, path, type, metadata, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                stage,
                name,
                str(path),
                artifact_type,
                json.dumps(metadata or {}),
                datetime.now(timezone.utc),
            ),
        )
        conn.commit()

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Получить соединение с DuckDB."""
        if self._conn is None:
            self._conn = duckdb.connect(str(self.duckdb_path))
        return self._conn

    def close(self) -> None:
        """Закрыть соединение и сбросить буферы."""
        self.flush()
        if self._conn is not None:
            self._conn.close()
            self._conn = None
