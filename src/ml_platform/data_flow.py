from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


DEFAULT_PREVIEW_LIMIT = 20
MAX_COLUMN_DIFF_ITEMS = 100


@dataclass(frozen=True)
class DataPreview:
    columns: list[str]
    rows: list[dict[str, object]]
    truncated: bool = False


@dataclass(frozen=True)
class DataFlowSnapshot:
    step: str
    label: str
    stage: str
    partition: str = "full"
    data_kind: str = "dataframe"
    rows: int | None = None
    columns: int | None = None
    memory_mb: float | None = None
    rows_delta: int | None = None
    columns_added: list[str] = field(default_factory=list)
    columns_removed: list[str] = field(default_factory=list)
    preview: DataPreview | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DataFlowTrace:
    target: str
    snapshots: list[DataFlowSnapshot] = field(default_factory=list)


class DataFlowTracker:
    def __init__(self, target: str, preview_limit: int = DEFAULT_PREVIEW_LIMIT) -> None:
        self.target = target
        self.preview_limit = preview_limit
        self.snapshots: list[DataFlowSnapshot] = []
        self._column_history: list[list[str] | None] = []

    def snapshot_dataframe(
        self,
        step: str,
        label: str,
        stage: str,
        df: pd.DataFrame,
        *,
        partition: str = "full",
        preview: bool = False,
        metadata: Mapping[str, object] | None = None,
    ) -> DataFlowSnapshot:
        column_names = [str(column) for column in df.columns]
        snapshot = self._snapshot(
            step=step,
            label=label,
            stage=stage,
            partition=partition,
            data_kind="dataframe",
            rows=len(df),
            columns=len(column_names),
            memory_mb=_frame_memory_mb(df),
            column_names=column_names,
            preview_payload=_build_preview(df, self.preview_limit) if preview else None,
            metadata=metadata,
        )
        return snapshot

    def snapshot_matrix(
        self,
        step: str,
        label: str,
        stage: str,
        *,
        matrix: Any | None = None,
        rows: int | None = None,
        columns: int | None = None,
        column_names: Iterable[Any] | None = None,
        partition: str = "train",
        metadata: Mapping[str, object] | None = None,
    ) -> DataFlowSnapshot:
        if matrix is not None:
            shape = getattr(matrix, "shape", None)
            if not shape or len(shape) < 2:
                raise ValueError("matrix must expose a 2D shape.")
            rows = int(shape[0])
            columns = int(shape[1])
            derived_metadata = {
                "density": _matrix_density(matrix),
            }
            memory_mb = _matrix_memory_mb(matrix)
        else:
            if rows is None or columns is None:
                raise ValueError("rows and columns are required when matrix is not provided.")
            derived_metadata = {}
            memory_mb = None

        names = [str(name) for name in column_names] if column_names is not None else None
        merged_metadata = dict(derived_metadata)
        if metadata:
            merged_metadata.update(dict(metadata))
        return self._snapshot(
            step=step,
            label=label,
            stage=stage,
            partition=partition,
            data_kind="matrix",
            rows=rows,
            columns=columns,
            memory_mb=memory_mb,
            column_names=names,
            preview_payload=None,
            metadata=merged_metadata,
        )

    def snapshot_artifact(
        self,
        step: str,
        label: str,
        stage: str,
        *,
        partition: str = "full",
        metadata: Mapping[str, object] | None = None,
    ) -> DataFlowSnapshot:
        return self._snapshot(
            step=step,
            label=label,
            stage=stage,
            partition=partition,
            data_kind="artifact",
            rows=None,
            columns=None,
            memory_mb=None,
            column_names=None,
            preview_payload=None,
            metadata=metadata,
        )

    def to_trace(self) -> DataFlowTrace:
        return DataFlowTrace(target=self.target, snapshots=list(self.snapshots))

    def _snapshot(
        self,
        *,
        step: str,
        label: str,
        stage: str,
        partition: str,
        data_kind: str,
        rows: int | None,
        columns: int | None,
        memory_mb: float | None,
        column_names: list[str] | None,
        preview_payload: DataPreview | None,
        metadata: Mapping[str, object] | None,
    ) -> DataFlowSnapshot:
        previous_snapshot, previous_columns = self._previous_for_partition(partition)
        rows_delta = None
        if rows is not None and previous_snapshot is not None and previous_snapshot.rows is not None:
            rows_delta = rows - previous_snapshot.rows

        columns_added: list[str] = []
        columns_removed: list[str] = []
        safe_metadata = _json_safe_value(dict(metadata or {}))
        if not isinstance(safe_metadata, dict):
            safe_metadata = {"value": safe_metadata}

        if column_names is not None and previous_columns is not None:
            added = [name for name in column_names if name not in previous_columns]
            removed = [name for name in previous_columns if name not in column_names]
            columns_added = added[:MAX_COLUMN_DIFF_ITEMS]
            columns_removed = removed[:MAX_COLUMN_DIFF_ITEMS]
            if len(added) > len(columns_added):
                safe_metadata["columns_added_total"] = len(added)
            if len(removed) > len(columns_removed):
                safe_metadata["columns_removed_total"] = len(removed)

        snapshot = DataFlowSnapshot(
            step=step,
            label=label,
            stage=stage,
            partition=partition,
            data_kind=data_kind,
            rows=rows,
            columns=columns,
            memory_mb=memory_mb,
            rows_delta=rows_delta,
            columns_added=columns_added,
            columns_removed=columns_removed,
            preview=preview_payload,
            metadata=safe_metadata,
        )
        self.snapshots.append(snapshot)
        self._column_history.append(column_names)
        return snapshot

    def _previous_for_partition(self, partition: str) -> tuple[DataFlowSnapshot | None, list[str] | None]:
        for index in range(len(self.snapshots) - 1, -1, -1):
            snapshot = self.snapshots[index]
            if snapshot.partition == partition:
                return snapshot, self._column_history[index]
        if partition != "full":
            for index in range(len(self.snapshots) - 1, -1, -1):
                snapshot = self.snapshots[index]
                if snapshot.partition == "full":
                    return snapshot, self._column_history[index]
        return None, None


def _build_preview(df: pd.DataFrame, limit: int) -> DataPreview:
    preview_df = df.head(limit).copy()
    rows = preview_df.to_dict(orient="records")
    safe_rows = []
    for row in rows:
        safe_rows.append({str(key): _json_safe_value(value) for key, value in row.items()})
    return DataPreview(
        columns=[str(column) for column in preview_df.columns],
        rows=safe_rows,
        truncated=len(df) > limit,
    )


def _frame_memory_mb(df: pd.DataFrame) -> float:
    return round(float(df.memory_usage(index=True, deep=True).sum()) / (1024 * 1024), 4)


def _matrix_density(matrix: Any) -> float | None:
    shape = getattr(matrix, "shape", None)
    if not shape or len(shape) < 2 or int(shape[0]) * int(shape[1]) == 0:
        return None
    total_cells = int(shape[0]) * int(shape[1])
    if hasattr(matrix, "nnz"):
        return round(float(matrix.nnz) / total_cells, 6)
    try:
        nonzero = int(np.count_nonzero(matrix))
    except Exception:
        return None
    return round(float(nonzero) / total_cells, 6)


def _matrix_memory_mb(matrix: Any) -> float | None:
    try:
        if hasattr(matrix, "data") and hasattr(matrix, "indptr") and hasattr(matrix, "indices"):
            bytes_used = int(matrix.data.nbytes + matrix.indptr.nbytes + matrix.indices.nbytes)
        elif hasattr(matrix, "nbytes"):
            bytes_used = int(matrix.nbytes)
        else:
            return None
    except Exception:
        return None
    return round(bytes_used / (1024 * 1024), 4)


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe_value(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, (pd.Series, pd.Index)):
        return [_json_safe_value(item) for item in value.tolist()]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (np.ndarray,)):
        return [_json_safe_value(item) for item in value.tolist()]
    return str(value)
