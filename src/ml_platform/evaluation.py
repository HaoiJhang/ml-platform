from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from ml_platform.cleaning import CleanedData
from ml_platform.data_flow import DataFlowTracker

logger = logging.getLogger(__name__)


def evaluate_model(
    model: Any,
    cleaned: CleanedData,
    task_type: str,
    tracker: DataFlowTracker | None = None,
) -> tuple[dict[str, float | None], pd.DataFrame]:
    logger.info("Evaluating model task_type=%s test_rows=%d", task_type, len(cleaned.X_test))
    X_test = _feature_frame(cleaned, partition="test")
    X_train = _feature_frame(cleaned, partition="train")
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    if task_type == "classification":
        metrics = _classification_metrics(model, X_test, cleaned.y_test, y_pred)
        metrics.update(_classification_metrics(model, X_train, cleaned.y_train, y_pred_train, prefix="train_"))
    elif task_type == "regression":
        metrics = _regression_metrics(cleaned.y_test, y_pred)
        metrics.update(_regression_metrics(cleaned.y_train, y_pred_train, prefix="train_"))
    else:
        raise ValueError("task_type must be 'classification' or 'regression'.")

    sample = X_test.copy().head(20)
    sample["actual"] = cleaned.y_test.head(20).to_numpy()
    sample["prediction"] = pd.Series(y_pred[:20], index=sample.index).to_numpy()
    if tracker is not None:
        tracker.snapshot_dataframe(
            "prediction_sample",
            "Prediction sample",
            "evaluation",
            sample,
            partition="test",
            preview=True,
            metadata={"task_type": task_type},
        )
    return metrics, sample


def _classification_metrics(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    y_pred: np.ndarray,
    prefix: str = "",
) -> dict[str, float | None]:
    return {
        f"{prefix}accuracy": float(accuracy_score(y, y_pred)),
        f"{prefix}f1_weighted": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
        f"{prefix}precision_weighted": float(precision_score(y, y_pred, average="weighted", zero_division=0)),
        f"{prefix}recall_weighted": float(recall_score(y, y_pred, average="weighted", zero_division=0)),
        f"{prefix}roc_auc": _safe_roc_auc(model, X, y),
    }


def _regression_metrics(y: pd.Series, y_pred: np.ndarray, prefix: str = "") -> dict[str, float | None]:
    mse = float(mean_squared_error(y, y_pred))
    return {
        f"{prefix}rmse": float(np.sqrt(mse)),
        f"{prefix}mae": float(mean_absolute_error(y, y_pred)),
        f"{prefix}r2": float(r2_score(y, y_pred)),
    }


def _safe_roc_auc(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> float | None:
    if y_test.nunique(dropna=True) != 2 or not hasattr(model, "predict_proba"):
        return None
    try:
        probabilities = model.predict_proba(X_test)
        if isinstance(probabilities, pd.Series):
            return float(roc_auc_score(y_test, probabilities))
        if probabilities.shape[1] != 2:
            return None
        if isinstance(probabilities, pd.DataFrame):
            scores = probabilities.iloc[:, 1]
        else:
            scores = probabilities[:, 1]
        return float(roc_auc_score(y_test, scores))
    except Exception:
        return None


def _feature_frame(cleaned: CleanedData, *, partition: str) -> pd.DataFrame:
    if partition == "train":
        value = cleaned.X_train_prepared if cleaned.X_train_prepared is not None else cleaned.X_train
        columns = cleaned.prepared_feature_names or cleaned.feature_columns
    elif partition == "test":
        value = cleaned.X_test_prepared if cleaned.X_test_prepared is not None else cleaned.X_test
        columns = cleaned.prepared_feature_names or cleaned.feature_columns
    else:
        raise ValueError("partition must be 'train' or 'test'.")
    if isinstance(value, pd.DataFrame):
        return value.reset_index(drop=True)
    return pd.DataFrame(value, columns=columns).reset_index(drop=True)
