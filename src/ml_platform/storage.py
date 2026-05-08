from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    path: Path
    created_at: str
    config: dict[str, Any]


class RunStorage:
    def __init__(self, runs_dir: str | Path = "runs") -> None:
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.runs_dir / "runs.sqlite"
        self._init_db()

    def create_run(self, config: dict[str, Any]) -> RunRecord:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        path = self.runs_dir / run_id
        path.mkdir(parents=True, exist_ok=False)
        created_at = datetime.now(timezone.utc).isoformat()
        self.save_json(RunRecord(run_id, path, created_at, config), "config.json", config)
        return RunRecord(run_id=run_id, path=path, created_at=created_at, config=config)

    def save_json(self, run: RunRecord, name: str, payload: Any) -> Path:
        output = run.path / name
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return output

    def save_text(self, run: RunRecord, name: str, payload: str) -> Path:
        output = run.path / name
        output.write_text(payload, encoding="utf-8")
        return output

    def save_model(self, run: RunRecord, model: Any) -> Path:
        output = run.path / "model.joblib"
        joblib.dump(model, output)
        return output

    def save_predictions(self, run: RunRecord, predictions: pd.DataFrame) -> Path:
        output = run.path / "prediction_sample.csv"
        predictions.to_csv(output, index=False)
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
