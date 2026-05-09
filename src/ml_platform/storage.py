from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    path: Path
    created_at: str
    config: dict[str, Any]


@dataclass(frozen=True)
class RecordedRun:
    run_id: str
    created_at: str
    path: Path
    status: str
    config: dict[str, Any]
    metrics: dict[str, Any]


class RunStorage:
    def __init__(self, runs_dir: str | Path = "runs") -> None:
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.runs_dir / "runs.sqlite"
        self._init_db()
        logger.info("RunStorage initialized runs_dir=%s", self.runs_dir.resolve())

    def create_run(self, config: dict[str, Any]) -> RunRecord:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        path = self.runs_dir / run_id
        path.mkdir(parents=True, exist_ok=False)
        created_at = datetime.now(timezone.utc).isoformat()
        self.save_json(RunRecord(run_id, path, created_at, config), "config.json", config)
        logger.info("Run created run_id=%s target=%s", run_id, config.get("target"))
        return RunRecord(run_id=run_id, path=path, created_at=created_at, config=config)

    def save_json(self, run: RunRecord, name: str, payload: Any) -> Path:
        output = run.path / name
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.debug("Saved JSON run_id=%s name=%s", run.run_id, name)
        return output

    def save_text(self, run: RunRecord, name: str, payload: str) -> Path:
        output = run.path / name
        output.write_text(payload, encoding="utf-8")
        logger.debug("Saved text run_id=%s name=%s", run.run_id, name)
        return output

    def save_model(self, run: RunRecord, model: Any) -> Path:
        output = run.path / "model.joblib"
        joblib.dump(model, output)
        logger.info("Model saved run_id=%s size_bytes=%d", run.run_id, output.stat().st_size)
        return output

    def save_predictions(self, run: RunRecord, predictions: pd.DataFrame) -> Path:
        output = run.path / "prediction_sample.csv"
        predictions.to_csv(output, index=False)
        logger.debug("Predictions saved run_id=%s rows=%d", run.run_id, len(predictions))
        return output

    def record_run(self, run: RunRecord, metrics: dict[str, Any], status: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO runs (run_id, created_at, path, status, config_json, metrics_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.created_at,
                    str(run.path),
                    status,
                    json.dumps(run.config, ensure_ascii=False),
                    json.dumps(metrics, ensure_ascii=False),
                ),
            )
        logger.info("Run recorded in db run_id=%s status=%s", run.run_id, status)

    def list_runs(self, limit: int = 50) -> list[RecordedRun]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT run_id, created_at, path, status, config_json, metrics_json
                FROM runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [
            RecordedRun(
                run_id=row[0],
                created_at=row[1],
                path=Path(row[2]),
                status=row[3],
                config=json.loads(row[4]),
                metrics=json.loads(row[5]),
            )
            for row in rows
        ]

    def load_json(self, run_path: str | Path, name: str) -> Any | None:
        path = Path(run_path) / name
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def load_text(self, run_path: str | Path, name: str) -> str | None:
        path = Path(run_path) / name
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL
                )
                """
            )
