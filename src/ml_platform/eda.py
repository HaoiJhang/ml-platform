from __future__ import annotations

from typing import Any
import warnings

import pandas as pd


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def infer_column_types(df: pd.DataFrame) -> dict[str, list[str]]:
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    datetime_like = []
    categorical = []
    for column in df.columns:
        if column in numeric:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(df[column].dropna().head(100), errors="coerce")
        if len(parsed) > 0 and parsed.notna().mean() >= 0.8:
            datetime_like.append(column)
        else:
            categorical.append(column)
    return {"numeric": numeric, "categorical": categorical, "datetime_like": datetime_like}


def generate_eda_summary(df: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
    column_types = infer_column_types(df)
    rows = len(df)
    columns: dict[str, Any] = {}

    for column in df.columns:
        series = df[column]
        missing_count = int(series.isna().sum())
        item: dict[str, Any] = {
            "dtype": str(series.dtype),
            "missing_count": missing_count,
            "missing_rate": float(missing_count / rows) if rows else 0.0,
            "unique_count": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            item["stats"] = {key: _json_safe(value) for key, value in desc.to_dict().items()}
        else:
            top_values = series.astype("string").value_counts(dropna=True).head(10)
            item["top_values"] = {str(key): int(value) for key, value in top_values.to_dict().items()}
        columns[column] = item

    summary: dict[str, Any] = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "duplicate_rows": int(df.duplicated().sum()),
        "column_types": column_types,
        "columns": columns,
        "quality_warnings": quality_warnings(df, target=target),
    }

    if target and target in df.columns:
        target_series = df[target]
        summary["target"] = {
            "name": target,
            "missing_count": int(target_series.isna().sum()),
            "unique_count": int(target_series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(target_series):
            summary["target"]["stats"] = {
                key: _json_safe(value) for key, value in target_series.describe().to_dict().items()
            }
        else:
            summary["target"]["top_values"] = {
                str(key): int(value)
                for key, value in target_series.astype("string").value_counts(dropna=True).head(20).to_dict().items()
            }
    return summary


def quality_warnings(df: pd.DataFrame, target: str | None = None) -> list[str]:
    warnings: list[str] = []
    if df.duplicated().any():
        warnings.append("Dataset contains duplicate rows.")
    high_missing = [
        column for column in df.columns if column != target and float(df[column].isna().mean()) >= 0.5
    ]
    if high_missing:
        warnings.append(f"Columns with at least 50% missing values: {', '.join(high_missing[:20])}.")
    constant_columns = [
        column for column in df.columns if column != target and int(df[column].nunique(dropna=True)) <= 1
    ]
    if constant_columns:
        warnings.append(f"Constant or near-empty columns: {', '.join(constant_columns[:20])}.")
    if target and target in df.columns and df[target].isna().any():
        warnings.append("Target column contains missing values; affected rows will be removed before training.")
    return warnings
