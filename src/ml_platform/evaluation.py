from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

from ml_platform.cleaning import CleanedData


def evaluate_model(model: Any, cleaned: CleanedData, task_type: str) -> tuple[dict[str, float | None], pd.DataFrame]:
    y_pred = model.predict(cleaned.X_test)
    if task_type == "classification":
        metrics: dict[str, float | None] = {
            "accuracy": float(accuracy_score(cleaned.y_test, y_pred)),
            "f1_weighted": float(f1_score(cleaned.y_test, y_pred, average="weighted", zero_division=0)),
            "roc_auc": _safe_roc_auc(model, cleaned.X_test, cleaned.y_test),
        }
    elif task_type == "regression":
        mse = float(mean_squared_error(cleaned.y_test, y_pred))
        metrics = {
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(cleaned.y_test, y_pred)),
            "r2": float(r2_score(cleaned.y_test, y_pred)),
        }
    else:
        raise ValueError("task_type must be 'classification' or 'regression'.")

    sample = cleaned.X_test.copy().head(20)
    sample["actual"] = cleaned.y_test.head(20).to_numpy()
    sample["prediction"] = pd.Series(y_pred[:20], index=sample.index).to_numpy()
    return metrics, sample


def _safe_roc_auc(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> float | None:
    if y_test.nunique(dropna=True) != 2 or not hasattr(model, "predict_proba"):
        return None
    try:
        probabilities = model.predict_proba(X_test)
        if probabilities.shape[1] != 2:
            return None
        return float(roc_auc_score(y_test, probabilities[:, 1]))
    except Exception:
        return None
