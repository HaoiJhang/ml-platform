from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline

from ml_platform.cleaning import CleanedData


@dataclass
class TrainedModel:
    model: Any
    trainer_name: str
    feature_importance: list[dict[str, Any]]


def train_model(cleaned: CleanedData, time_budget: int = 30) -> TrainedModel:
    try:
        return _train_with_flaml(cleaned, time_budget=time_budget)
    except Exception as exc:
        return _train_with_sklearn(cleaned, fallback_reason=str(exc))


def _train_with_flaml(cleaned: CleanedData, time_budget: int) -> TrainedModel:
    from flaml import AutoML

    preprocessor = cleaned.preprocessor
    X_train = preprocessor.fit_transform(cleaned.X_train)
    X_test_probe = preprocessor.transform(cleaned.X_test.head(1))
    if X_test_probe.shape[1] == 0:
        raise ValueError("Preprocessor produced no features.")

    task = "classification" if cleaned.config.task_type == "classification" else "regression"
    automl = AutoML()
    automl.fit(
        X_train,
        cleaned.y_train,
        task=task,
        time_budget=time_budget,
        verbose=0,
    )

    model = Pipeline([("preprocessor", preprocessor), ("estimator", automl)])
    importance = _feature_importance_from_pipeline(model, cleaned.feature_columns)
    return TrainedModel(model=model, trainer_name="flaml", feature_importance=importance)


def _train_with_sklearn(cleaned: CleanedData, fallback_reason: str | None = None) -> TrainedModel:
    if cleaned.config.task_type == "classification":
        estimator = RandomForestClassifier(n_estimators=200, random_state=cleaned.config.random_state, n_jobs=-1)
    else:
        estimator = RandomForestRegressor(n_estimators=200, random_state=cleaned.config.random_state, n_jobs=-1)

    model = Pipeline([("preprocessor", cleaned.preprocessor), ("estimator", estimator)])
    model.fit(cleaned.X_train, cleaned.y_train)
    importance = _feature_importance_from_pipeline(model, cleaned.feature_columns)
    if fallback_reason:
        importance.insert(
            0,
            {
                "feature": "__trainer_note__",
                "importance": 0.0,
                "note": f"Used sklearn fallback because FLAML was unavailable or failed: {fallback_reason}",
            },
        )
    return TrainedModel(model=model, trainer_name="sklearn_random_forest", feature_importance=importance)


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
