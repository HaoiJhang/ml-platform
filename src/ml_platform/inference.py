from __future__ import annotations

from typing import Any

import pandas as pd

from ml_platform.cleaning import CleanedData


def predict_with_trained_model(
    *,
    model: Any,
    cleaned: CleanedData,
    new_data: pd.DataFrame,
    task_type: str,
) -> pd.DataFrame:
    """Run current-session inference with the same feature shape used for training."""
    if task_type not in {"classification", "regression"}:
        raise ValueError("task_type must be 'classification' or 'regression'.")
    if new_data.empty:
        raise ValueError("Inference input is empty.")

    missing_features = [
        column for column in cleaned.feature_columns if column not in new_data.columns
    ]
    if missing_features:
        raise ValueError(
            "Inference input is missing required feature columns: "
            + ", ".join(missing_features)
        )

    output = new_data.copy()
    model_input = new_data[cleaned.feature_columns].copy()
    if cleaned.fitted_preprocessor is not None:
        model_input = cleaned.fitted_preprocessor.transform(model_input)
        model_input = _ensure_dataframe(model_input, cleaned.prepared_feature_names)

    predictions = model.predict(model_input)
    output["prediction"] = pd.Series(predictions).to_numpy()

    if task_type == "classification" and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(model_input)
        probability_frame = _probability_frame(probabilities)
        for column in probability_frame.columns:
            output[f"probability__{column}"] = probability_frame[column].to_numpy()

    return output


def _ensure_dataframe(value: Any, columns: list[str] | None) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.reset_index(drop=True)
    return pd.DataFrame(value, columns=columns).reset_index(drop=True)


def _probability_frame(probabilities: Any) -> pd.DataFrame:
    if isinstance(probabilities, pd.DataFrame):
        return probabilities.reset_index(drop=True).rename(columns=str)
    if isinstance(probabilities, pd.Series):
        return pd.DataFrame({"positive": probabilities.reset_index(drop=True)})
    frame = pd.DataFrame(probabilities).reset_index(drop=True)
    return frame.rename(columns=lambda column: str(column))
