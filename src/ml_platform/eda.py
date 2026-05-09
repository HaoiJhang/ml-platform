from __future__ import annotations

import logging
from typing import Any
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
    logger.info("Generating EDA summary rows=%d columns=%d target=%s", len(df), len(df.columns), target)
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
            item["skew"] = _json_safe(series.skew())
            item["outlier_count_iqr"] = _iqr_outlier_count(series)
            item["zero_rate"] = float((series == 0).mean()) if rows else 0.0
        else:
            top_values = series.astype("string").value_counts(dropna=True).head(10)
            item["top_values"] = {str(key): int(value) for key, value in top_values.to_dict().items()}
            item["top_value_rate"] = float(top_values.iloc[0] / rows) if rows and len(top_values) else 0.0
        columns[column] = item

    summary: dict[str, Any] = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "duplicate_rows": int(df.duplicated().sum()),
        "column_types": column_types,
        "columns": columns,
        "missingness": missingness_summary(df),
        "correlations": correlation_summary(df, target=target),
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
        summary["target_relationships"] = target_relationship_summary(df, target)
    logger.info("EDA summary complete warnings=%d", len(summary["quality_warnings"]))
    return summary


def _iqr_outlier_count(series: pd.Series) -> int:
    clean = series.dropna()
    if clean.empty:
        return 0
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0 or pd.isna(iqr):
        return 0
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((clean < lower) | (clean > upper)).sum())


def missingness_summary(df: pd.DataFrame) -> dict[str, Any]:
    missing_rate = df.isna().mean().sort_values(ascending=False)
    top_missing = {
        column: float(rate)
        for column, rate in missing_rate[missing_rate > 0].head(20).to_dict().items()
    }
    missing_indicators = df.isna().astype(int)
    pairs: list[dict[str, Any]] = []
    if missing_indicators.shape[1] >= 2:
        corr = missing_indicators.corr().replace([np.inf, -np.inf], np.nan)
        for left in corr.columns:
            for right in corr.columns:
                if left >= right:
                    continue
                value = corr.loc[left, right]
                if pd.notna(value) and abs(value) >= 0.5:
                    pairs.append({"left": left, "right": right, "correlation": float(value)})
    pairs = sorted(pairs, key=lambda row: abs(row["correlation"]), reverse=True)[:20]
    return {
        "rows_with_any_missing": int(df.isna().any(axis=1).sum()),
        "top_missing_columns": top_missing,
        "correlated_missing_pairs": pairs,
    }


def correlation_summary(df: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
    numeric = df.select_dtypes(include=["number"])
    if numeric.shape[1] < 2:
        return {"top_numeric_pairs": [], "target_numeric_correlations": []}

    corr = numeric.corr(numeric_only=True).replace([np.inf, -np.inf], np.nan)
    pairs: list[dict[str, Any]] = []
    columns = list(corr.columns)
    for index, left in enumerate(columns):
        for right in columns[index + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value):
                pairs.append({"left": left, "right": right, "correlation": float(value)})
    pairs = sorted(pairs, key=lambda row: abs(row["correlation"]), reverse=True)[:20]

    target_corr: list[dict[str, Any]] = []
    if target and target in corr.columns:
        for feature, value in corr[target].drop(labels=[target]).dropna().items():
            target_corr.append({"feature": feature, "correlation": float(value)})
        target_corr = sorted(target_corr, key=lambda row: abs(row["correlation"]), reverse=True)[:20]

    return {"top_numeric_pairs": pairs, "target_numeric_correlations": target_corr}


def target_relationship_summary(df: pd.DataFrame, target: str) -> dict[str, Any]:
    target_series = df[target]
    features = [column for column in df.columns if column != target]
    numeric_features = df[features].select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [column for column in features if column not in numeric_features]
    relationships: dict[str, Any] = {"numeric_features": [], "categorical_features": []}

    if pd.api.types.is_numeric_dtype(target_series):
        for column in numeric_features[:50]:
            corr = df[[column, target]].corr(numeric_only=True).iloc[0, 1]
            relationships["numeric_features"].append(
                {"feature": column, "target_correlation": _json_safe(corr)}
            )
    else:
        for column in numeric_features[:50]:
            grouped = df.groupby(target, dropna=False)[column].agg(["mean", "median", "count"]).head(20)
            relationships["numeric_features"].append(
                {
                    "feature": column,
                    "by_target": {
                        str(index): {key: _json_safe(value) for key, value in row.items()}
                        for index, row in grouped.iterrows()
                    },
                }
            )

    for column in categorical_features[:30]:
        crosstab = pd.crosstab(df[column].astype("string"), target_series.astype("string"), dropna=False)
        normalized = crosstab.div(crosstab.sum(axis=1).replace(0, np.nan), axis=0).head(20)
        relationships["categorical_features"].append(
            {
                "feature": column,
                "target_distribution_by_value": {
                    str(index): {str(key): _json_safe(value) for key, value in row.items()}
                    for index, row in normalized.iterrows()
                },
            }
        )

    return relationships


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
    numeric_columns = df.select_dtypes(include=["number"]).columns
    outlier_columns = [
        column
        for column in numeric_columns
        if column != target and _iqr_outlier_count(df[column]) > max(10, int(len(df) * 0.05))
    ]
    if outlier_columns:
        warnings.append(f"Numeric columns with many IQR outliers: {', '.join(outlier_columns[:20])}.")
    return warnings
