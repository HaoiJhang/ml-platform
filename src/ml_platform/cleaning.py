from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class CleanConfig:
    target: str
    task_type: str
    test_size: float = 0.2
    random_state: int = 42
    high_missing_threshold: float = 0.9
    standardize_numeric: bool = True


@dataclass
class CleanedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    feature_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    cleaning_log: list[dict[str, Any]]
    config: CleanConfig


def _make_one_hot_encoder() -> OneHotEncoder:
    kwargs: dict[str, Any] = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        kwargs["sparse_output"] = False
    else:
        kwargs["sparse"] = False
    return OneHotEncoder(**kwargs)


def clean_and_split(df: pd.DataFrame, config: CleanConfig) -> CleanedData:
    if config.target not in df.columns:
        raise ValueError(f"Target column not found: {config.target}")
    if config.task_type not in {"classification", "regression"}:
        raise ValueError("task_type must be 'classification' or 'regression'.")

    working = df.copy()
    log: list[dict[str, Any]] = []

    before_rows = len(working)
    working = working.dropna(subset=[config.target])
    log.append(
        {
            "step": "drop_missing_target",
            "rows_before": before_rows,
            "rows_after": len(working),
            "rows_removed": before_rows - len(working),
        }
    )
    if working.empty:
        raise ValueError("No rows remain after dropping missing target values.")

    feature_columns = [column for column in working.columns if column != config.target]
    high_missing_columns = [
        column
        for column in feature_columns
        if float(working[column].isna().mean()) > config.high_missing_threshold
    ]
    if high_missing_columns:
        working = working.drop(columns=high_missing_columns)
    log.append(
        {
            "step": "drop_high_missing_features",
            "threshold": config.high_missing_threshold,
            "columns": high_missing_columns,
        }
    )

    feature_columns = [column for column in working.columns if column != config.target]
    constant_columns = [
        column for column in feature_columns if int(working[column].nunique(dropna=True)) <= 1
    ]
    if constant_columns:
        working = working.drop(columns=constant_columns)
    log.append({"step": "drop_constant_features", "columns": constant_columns})

    feature_columns = [column for column in working.columns if column != config.target]
    if not feature_columns:
        raise ValueError("No feature columns remain after cleaning.")

    X = working[feature_columns]
    y = working[config.target]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if config.standardize_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_features:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_features))
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _make_one_hot_encoder()),
                    ]
                ),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
    stratify = None
    if config.task_type == "classification" and y.nunique(dropna=True) > 1:
        class_counts = y.value_counts()
        if int(class_counts.min()) >= 2:
            stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify,
    )
    log.append(
        {
            "step": "split_train_test",
            "test_size": config.test_size,
            "random_state": config.random_state,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "stratified": stratify is not None,
        }
    )
    log.append(
        {
            "step": "build_preprocessor",
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "standardize_numeric": config.standardize_numeric,
        }
    )

    return CleanedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        feature_columns=feature_columns,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        cleaning_log=log,
        config=config,
    )
