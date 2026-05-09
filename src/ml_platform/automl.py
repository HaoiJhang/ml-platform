from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline

from ml_platform.cleaning import CleanedData

logger = logging.getLogger(__name__)


@dataclass
class TrainedModel:
    model: Any
    trainer_name: str
    feature_importance: list[dict[str, Any]]
    optimization_metric_used: str | None = None
    training_notes: list[str] | None = None


def train_model(cleaned: CleanedData, time_budget: int = 30, metric_preference: str = "auto") -> TrainedModel:
    logger.info("Starting training rows=%d features=%d task=%s time_budget=%d", len(cleaned.X_train), len(cleaned.feature_columns), cleaned.config.task_type, time_budget)
    try:
        return _train_with_flaml(cleaned, time_budget=time_budget, metric_preference=metric_preference)
    except Exception as exc:
        logger.warning("FLAML training failed, falling back to sklearn: %s", exc)
        return _train_with_sklearn(cleaned, metric_preference=metric_preference, fallback_reason=str(exc))


def _train_with_flaml(cleaned: CleanedData, time_budget: int, metric_preference: str) -> TrainedModel:
    from flaml import AutoML

    preprocessor = cleaned.preprocessor
    X_train = preprocessor.fit_transform(cleaned.X_train)
    X_test_probe = preprocessor.transform(cleaned.X_test.head(1))
    if X_test_probe.shape[1] == 0:
        raise ValueError("Preprocessor produced no features.")

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
    return TrainedModel(
        model=model,
        trainer_name="flaml",
        feature_importance=importance,
        optimization_metric_used=flaml_metric,
        training_notes=notes,
    )


def _train_with_sklearn(
    cleaned: CleanedData,
    metric_preference: str,
    fallback_reason: str | None = None,
) -> TrainedModel:
    logger.info("Training sklearn RandomForest features=%d", len(cleaned.feature_columns))
    if cleaned.config.task_type == "classification":
        estimator = RandomForestClassifier(n_estimators=200, random_state=cleaned.config.random_state, n_jobs=-1)
    else:
        estimator = RandomForestRegressor(n_estimators=200, random_state=cleaned.config.random_state, n_jobs=-1)

    model = Pipeline([("preprocessor", cleaned.preprocessor), ("estimator", estimator)])
    model.fit(cleaned.X_train, cleaned.y_train)
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
    return TrainedModel(
        model=model,
        trainer_name="sklearn_random_forest",
        feature_importance=importance,
        optimization_metric_used=None,
        training_notes=notes,
    )


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
