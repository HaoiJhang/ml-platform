from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline

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


def train_model(
    cleaned: CleanedData,
    time_budget: int = 30,
    metric_preference: str = "auto",
    tracker: DataFlowTracker | None = None,
) -> TrainedModel:
    logger.info("Starting training rows=%d features=%d task=%s time_budget=%d", len(cleaned.X_train), len(cleaned.feature_columns), cleaned.config.task_type, time_budget)
    try:
        return _train_with_flaml(cleaned, time_budget=time_budget, metric_preference=metric_preference, tracker=tracker)
    except Exception as exc:
        logger.warning("FLAML training failed, falling back to sklearn: %s", exc)
        return _train_with_sklearn(cleaned, metric_preference=metric_preference, fallback_reason=str(exc), tracker=tracker)


def _train_with_flaml(
    cleaned: CleanedData,
    time_budget: int,
    metric_preference: str,
    tracker: DataFlowTracker | None = None,
) -> TrainedModel:
    from flaml import AutoML

    preprocessor, X_train, X_test_probe, feature_names = _resolved_training_inputs(cleaned)
    if tracker is not None:
        tracker.snapshot_matrix(
            "train_matrix",
            "Train matrix",
            "training",
            matrix=X_train,
            partition="train",
            column_names=feature_names,
            metadata={"feature_count": len(feature_names) if feature_names is not None else X_train.shape[1]},
        )
        tracker.snapshot_matrix(
            "test_matrix_probe",
            "Test matrix probe",
            "training",
            matrix=X_test_probe,
            partition="test",
            column_names=feature_names,
            metadata={"probe_rows": int(X_test_probe.shape[0])},
        )
    task = "classification" if cleaned.config.task_type == "classification" else "regression"
    flaml_metric, metric_note = _map_metric_preference(cleaned.config.task_type, metric_preference)
    automl = AutoML()
    fit_kwargs: dict[str, Any] = {
        "task": task,
        "time_budget": time_budget,
        "verbose": 0,
    }
    if flaml_metric is not None:
        fit_kwargs["metric"] = flaml_metric
    automl.fit(
        X_train,
        cleaned.y_train,
        **fit_kwargs,
    )

    model = Pipeline([("preprocessor", preprocessor), ("estimator", automl)])
    importance = _feature_importance_from_pipeline(model, cleaned.feature_columns)
    notes = [metric_note] if metric_note else []
    trained = TrainedModel(
        model=model,
        trainer_name="flaml",
        feature_importance=importance,
        optimization_metric_used=flaml_metric,
        training_notes=notes,
    )
    _record_training_artifact(tracker, trained)
    return trained


def _train_with_sklearn(
    cleaned: CleanedData,
    metric_preference: str,
    fallback_reason: str | None = None,
    tracker: DataFlowTracker | None = None,
) -> TrainedModel:
    logger.info("Training sklearn RandomForest features=%d", len(cleaned.feature_columns))
    if cleaned.config.task_type == "classification":
        estimator = RandomForestClassifier(n_estimators=200, random_state=cleaned.config.random_state, n_jobs=-1)
    else:
        estimator = RandomForestRegressor(n_estimators=200, random_state=cleaned.config.random_state, n_jobs=-1)

    preprocessor, X_train, X_test_probe, feature_names = _resolved_training_inputs(cleaned)
    estimator.fit(X_train, cleaned.y_train)
    model = Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])
    if tracker is not None:
        tracker.snapshot_matrix(
            "train_matrix",
            "Train matrix",
            "training",
            matrix=X_train,
            partition="train",
            column_names=feature_names,
            metadata={
                "feature_count": len(feature_names) if feature_names is not None else len(cleaned.feature_columns),
                "materialized_after_fit": cleaned.fitted_preprocessor is None,
            },
        )
        tracker.snapshot_matrix(
            "test_matrix_probe",
            "Test matrix probe",
            "training",
            matrix=X_test_probe,
            partition="test",
            column_names=feature_names,
            metadata={"probe_rows": int(X_test_probe.shape[0])},
        )
    importance = _feature_importance_from_pipeline(model, cleaned.feature_columns)
    notes = []
    if fallback_reason:
        notes.append(f"Used sklearn fallback because FLAML was unavailable or failed: {fallback_reason}")
        importance.insert(
            0,
            {
                "feature": "__trainer_note__",
                "importance": 0.0,
                "note": notes[-1],
            },
        )
    if metric_preference != "auto":
        notes.append(f"Sklearn fallback ignores metric preference during optimization: {metric_preference}.")
    trained = TrainedModel(
        model=model,
        trainer_name="sklearn_random_forest",
        feature_importance=importance,
        optimization_metric_used=None,
        training_notes=notes,
    )
    _record_training_artifact(tracker, trained)
    return trained


def _feature_importance_from_pipeline(model: Pipeline, fallback_features: list[str]) -> list[dict[str, Any]]:
    estimator = model.named_steps["estimator"]
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None and hasattr(estimator, "model") and hasattr(estimator.model, "estimator"):
        importances = getattr(estimator.model.estimator, "feature_importances_", None)
    if importances is None:
        return [{"feature": feature, "importance": None} for feature in fallback_features[:50]]

    try:
        names = model.named_steps["preprocessor"].get_feature_names_out()
    except Exception:
        names = fallback_features
    rows = [
        {"feature": str(name), "importance": float(value)}
        for name, value in zip(names, importances)
    ]
    return sorted(rows, key=lambda row: row["importance"], reverse=True)[:50]


def _preprocessor_feature_names(preprocessor: Any) -> list[str] | None:
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:
        return None


def _resolved_training_inputs(cleaned: CleanedData) -> tuple[Any, Any, Any, list[str] | None]:
    if cleaned.fitted_preprocessor is not None and cleaned.X_train_prepared is not None and cleaned.X_test_prepared is not None:
        probe = cleaned.X_test_prepared[:1]
        return cleaned.fitted_preprocessor, cleaned.X_train_prepared, probe, cleaned.prepared_feature_names

    preprocessor = cleaned.preprocessor
    X_train = preprocessor.fit_transform(cleaned.X_train)
    X_test_probe = preprocessor.transform(cleaned.X_test.head(1))
    if X_test_probe.shape[1] == 0:
        raise ValueError("Preprocessor produced no features.")
    feature_names = _preprocessor_feature_names(preprocessor)
    return preprocessor, X_train, X_test_probe, feature_names


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
        },
    )


def _map_metric_preference(task_type: str, metric_preference: str) -> tuple[str | None, str | None]:
    if task_type == "classification":
        mapping = {
            "accuracy": "accuracy",
            "roc_auc": "roc_auc",
        }
    else:
        mapping = {
            "rmse": "rmse",
            "mae": "mae",
            "r2": "r2",
        }
    if metric_preference in mapping:
        return mapping[metric_preference], None
    if metric_preference != "auto":
        return None, f"FLAML optimization metric mapping was unavailable for requested metric: {metric_preference}."
    return None, None
