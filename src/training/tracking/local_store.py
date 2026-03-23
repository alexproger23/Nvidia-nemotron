from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import duckdb


@dataclass(slots=True)
class LocalStore:
    """Локальное хранилище метрик в Parquet + DuckDB."""

    parquet_root: Path
    duckdb_path: Path
    _conn: duckdb.DuckDBPyConnection | None = field(init=False, default=None)

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
                run_id VARCHAR,
                recipe VARCHAR,
                started_at TIMESTAMP,
                finished_at TIMESTAMP,
                status VARCHAR,
                PRIMARY KEY (run_id)
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
        conn.commit()

    def write_metrics(
        self,
        run_id: str,
        stage: str,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Записать метрики в Parquet и DuckDB."""
        timestamp = datetime.now(timezone.utc)
        step_value = step if step is not None else -1

        # Запись в DuckDB
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

        # Запись в Parquet (дописываем в файл run_id/stage/metrics.parquet)
        self._write_parquet(run_id, stage, metrics, step_value, timestamp)

    def _write_parquet(
        self,
        run_id: str,
        stage: str,
        metrics: dict[str, float],
        step: int,
        timestamp: datetime,
    ) -> None:
        """Дописать метрики в Parquet файл."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        parquet_dir = self.parquet_root / run_id / stage
        parquet_dir.mkdir(parents=True, exist_ok=True)
        parquet_file = parquet_dir / "metrics.parquet"

        # Создаём таблицу для одной записи
        table = pa.table({
            "run_id": [run_id],
            "stage": [stage],
            "metric": list(metrics.keys()),
            "value": list(metrics.values()),
            "step": [step] * len(metrics),
            "timestamp": [timestamp] * len(metrics),
        })

        if parquet_file.exists():
            # Читаем существующий и дописываем
            existing = pq.read_table(parquet_file)
            combined = pa.concat_tables([existing, table])
            pq.write_table(combined, parquet_file)
        else:
            pq.write_table(table, parquet_file)

    def register_run_start(self, run_id: str, recipe: str) -> None:
        """Зарегистрировать старт run."""
        conn = self._get_connection()
        conn.execute(
            """
            INSERT OR REPLACE INTO runs (run_id, recipe, started_at, status)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, recipe, datetime.now(timezone.utc), "running"),
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
        """Закрыть соединение."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
