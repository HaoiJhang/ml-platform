from __future__ import annotations

import inspect
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ml_platform.cleaning import CleanedData
from ml_platform.data_flow import DataFlowTracker

logger = logging.getLogger(__name__)


@dataclass
class TrainedModel:
    model: Any
    trainer_name: str
    feature_importance: list[dict[str, Any]]
    optimization_metric_used: str | None = None
    training_notes: list[str] | None = None
    leaderboard: list[dict[str, Any]] = field(default_factory=list)
    fit_summary: dict[str, Any] = field(default_factory=dict)
    model_path: Path | None = None


def train_model(
    cleaned: CleanedData,
    time_budget: int = 30,
    metric_preference: str = "auto",
    tracker: DataFlowTracker | None = None,
    output_path: str | Path | None = None,
    presets: str | None = None,
) -> TrainedModel:
    logger.info(
        "Starting AutoGluon training rows=%d features=%d task=%s time_budget=%d",
        len(cleaned.X_train),
        len(cleaned.feature_columns),
        cleaned.config.task_type,
        time_budget,
    )
    return _train_with_autogluon(
        cleaned,
        time_budget=time_budget,
        metric_preference=metric_preference,
        tracker=tracker,
        output_path=output_path,
        presets=presets or cleaned.config.autogluon_presets,
    )


def _train_with_autogluon(
    cleaned: CleanedData,
    time_budget: int,
    metric_preference: str,
    tracker: DataFlowTracker | None = None,
    output_path: str | Path | None = None,
    presets: str = "medium_quality",
) -> TrainedModel:
    try:
        from autogluon.features.generators import AutoMLPipelineFeatureGenerator
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise RuntimeError(
            "AutoGluon is required for training. Install autogluon.tabular[all] and rerun."
        ) from exc

    train_data = _autogluon_frame(cleaned, partition="train")
    test_data = _autogluon_frame(cleaned, partition="test")
    feature_names = [column for column in train_data.columns if column != cleaned.config.target]
    problem_type = _autogluon_problem_type(cleaned)
    autogluon_metric, metric_note = _map_metric_preference(cleaned.config.task_type, metric_preference)
    predictor_path = _resolve_predictor_path(output_path)
    feature_generator = _build_feature_generator(
        AutoMLPipelineFeatureGenerator,
        cleaned.config.autogluon_feature_generator_params or {},
    )

    if tracker is not None:
        tracker.snapshot_dataframe(
            "autogluon_training_frame",
            "AutoGluon training frame",
            "training",
            train_data,
            partition="train",
            metadata={
                "feature_count": len(feature_names),
                "label": cleaned.config.target,
                "problem_type": problem_type,
                "presets": presets,
            },
        )
        tracker.snapshot_dataframe(
            "autogluon_test_frame",
            "AutoGluon test frame",
            "training",
            test_data,
            partition="test",
            metadata={"feature_count": len(feature_names), "label": cleaned.config.target},
        )
        tracker.snapshot_artifact(
            "autogluon_feature_generator_config",
            "AutoGluon feature generator config",
            "training",
            metadata=dict(cleaned.config.autogluon_feature_generator_params or {}),
        )

    predictor_kwargs: dict[str, Any] = {
        "label": cleaned.config.target,
        "problem_type": problem_type,
        "path": str(predictor_path),
    }
    if autogluon_metric is not None:
        predictor_kwargs["eval_metric"] = autogluon_metric
    predictor = TabularPredictor(**predictor_kwargs)
    fit_kwargs: dict[str, Any] = {
        "train_data": train_data,
        "time_limit": max(int(time_budget), 1),
        "presets": presets,
        "verbosity": 0,
        "feature_generator": feature_generator,
    }
    predictor.fit(**fit_kwargs)

    notes = [metric_note] if metric_note else []
    leaderboard, leaderboard_note = _leaderboard_rows(predictor, test_data)
    if leaderboard_note:
        notes.append(leaderboard_note)
    importance, importance_note = _feature_importance_rows(predictor, test_data, feature_names)
    if importance_note:
        notes.append(importance_note)
    fit_summary, summary_note = _fit_summary(predictor)
    if summary_note:
        notes.append(summary_note)

    trained = TrainedModel(
        model=predictor,
        trainer_name="autogluon_tabular",
        feature_importance=importance,
        optimization_metric_used=autogluon_metric,
        training_notes=notes,
        leaderboard=leaderboard,
        fit_summary=fit_summary,
        model_path=Path(getattr(predictor, "path", predictor_path)),
    )
    _record_training_artifact(tracker, trained)
    return trained


def _autogluon_frame(cleaned: CleanedData, *, partition: str) -> pd.DataFrame:
    if partition == "train":
        X = cleaned.X_train_prepared if cleaned.X_train_prepared is not None else cleaned.X_train
        y = cleaned.y_train
    elif partition == "test":
        X = cleaned.X_test_prepared if cleaned.X_test_prepared is not None else cleaned.X_test
        y = cleaned.y_test
    else:
        raise ValueError("partition must be 'train' or 'test'.")

    frame = _ensure_dataframe(X, cleaned.prepared_feature_names or cleaned.feature_columns).copy()
    frame[cleaned.config.target] = y.to_numpy()
    return frame


def _ensure_dataframe(value: Any, columns: list[str]) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.reset_index(drop=True)
    return pd.DataFrame(value, columns=columns).reset_index(drop=True)


def _autogluon_problem_type(cleaned: CleanedData) -> str:
    if cleaned.config.task_type == "regression":
        return "regression"
    class_count = int(cleaned.y_train.nunique(dropna=True))
    return "binary" if class_count == 2 else "multiclass"


def _resolve_predictor_path(output_path: str | Path | None) -> Path:
    if output_path is None:
        return Path(tempfile.mkdtemp(prefix="ml-platform-autogluon-")) / "predictor"
    return Path(output_path)


def _build_feature_generator(generator_class: Any, params: dict[str, Any]) -> Any:
    signature = inspect.signature(generator_class)
    supported = {
        key: value
        for key, value in params.items()
        if key in signature.parameters
    }
    return generator_class(**supported)


def _leaderboard_rows(predictor: Any, test_data: pd.DataFrame) -> tuple[list[dict[str, Any]], str | None]:
    try:
        leaderboard = predictor.leaderboard(test_data, silent=True)
    except Exception as exc:
        return [], f"AutoGluon leaderboard unavailable: {exc}"
    return _dataframe_records(leaderboard), None


def _feature_importance_rows(
    predictor: Any,
    test_data: pd.DataFrame,
    fallback_features: list[str],
) -> tuple[list[dict[str, Any]], str | None]:
    try:
        importance = predictor.feature_importance(test_data, silent=True)
    except Exception as exc:
        rows = [{"feature": feature, "importance": None} for feature in fallback_features[:50]]
        return rows, f"AutoGluon feature importance unavailable: {exc}"
    if not isinstance(importance, pd.DataFrame) or importance.empty:
        rows = [{"feature": feature, "importance": None} for feature in fallback_features[:50]]
        return rows, "AutoGluon feature importance returned no rows."
    feature_column = importance.index.name or "index"
    rows = importance.reset_index().rename(columns={feature_column: "feature"})
    if "feature" not in rows.columns:
        rows = rows.rename(columns={rows.columns[0]: "feature"})
    return _dataframe_records(rows.head(50)), None


def _fit_summary(predictor: Any) -> tuple[dict[str, Any], str | None]:
    try:
        summary = predictor.fit_summary(verbosity=0)
    except Exception as exc:
        return {}, f"AutoGluon fit summary unavailable: {exc}"
    return _json_safe(summary), None


def _dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_json_safe(row) for row in frame.to_dict(orient="records")]


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return _dataframe_records(value)
    if isinstance(value, pd.Series):
        return [_json_safe(item) for item in value.to_list()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return str(value)
    return value


def _record_training_artifact(tracker: DataFlowTracker | None, trained: TrainedModel) -> None:
    if tracker is None:
        return
    tracker.snapshot_artifact(
        "trained_model",
        "Trained model",
        "training",
        metadata={
            "trainer_name": trained.trainer_name,
            "optimization_metric_used": trained.optimization_metric_used,
            "training_notes": trained.training_notes or [],
            "feature_importance_count": len(trained.feature_importance),
            "leaderboard_rows": len(trained.leaderboard),
            "model_path": str(trained.model_path) if trained.model_path else None,
        },
    )


def _map_metric_preference(task_type: str, metric_preference: str) -> tuple[str | None, str | None]:
    if task_type == "classification":
        mapping = {
            "accuracy": "accuracy",
            "f1_weighted": "f1_weighted",
            "precision_weighted": "precision_weighted",
            "recall_weighted": "recall_weighted",
            "roc_auc": "roc_auc",
        }
    else:
        mapping = {
            "rmse": "root_mean_squared_error",
            "mae": "mean_absolute_error",
            "r2": "r2",
        }
    if metric_preference in mapping:
        return mapping[metric_preference], None
    if metric_preference != "auto":
        return None, f"AutoGluon optimization metric mapping was unavailable for requested metric: {metric_preference}."
    return None, None
