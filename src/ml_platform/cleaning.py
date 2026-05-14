from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from ml_platform.artifacts import FeatureEngineeringOperation, PreprocessingPlan, artifact_to_dict
from ml_platform.data_flow import DataFlowTracker
from ml_platform.feature_engineering import FeatureEngineeringTransformer

logger = logging.getLogger(__name__)
MISSING_TOKEN = "__missing__"


@dataclass(frozen=True)
class CleanConfig:
    target: str
    task_type: str
    test_size: float = 0.2
    random_state: int = 42
    high_missing_threshold: float = 0.9
    numeric_imputation_strategy: str = "median"
    categorical_imputation_strategy: str = "most_frequent"
    categorical_encoding_strategy: str = "one_hot"
    standardize_numeric: bool = True
    feature_engineering_operations: list[FeatureEngineeringOperation] | None = None


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
    fitted_preprocessor: Any | None = None
    X_train_prepared: Any | None = None
    X_test_prepared: Any | None = None
    prepared_feature_names: list[str] | None = None


def _make_one_hot_encoder() -> OneHotEncoder:
    kwargs: dict[str, Any] = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        kwargs["sparse_output"] = False
    else:
        kwargs["sparse"] = False
    return OneHotEncoder(**kwargs)


def _make_ordinal_encoder() -> OrdinalEncoder:
    kwargs: dict[str, Any] = {"handle_unknown": "use_encoded_value", "unknown_value": -1}
    if "encoded_missing_value" in inspect.signature(OrdinalEncoder).parameters:
        kwargs["encoded_missing_value"] = -1
    return OrdinalEncoder(**kwargs)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X: Any, y: Any = None) -> "FrequencyEncoder":
        frame = pd.DataFrame(X)
        self.frequency_maps_: list[dict[str, float]] = []
        for column in frame.columns:
            values = _string_values(frame[column])
            self.frequency_maps_.append(values.value_counts(normalize=True, dropna=False).to_dict())
        return self

    def transform(self, X: Any) -> np.ndarray:
        frame = pd.DataFrame(X)
        encoded_columns: list[np.ndarray] = []
        for index, column in enumerate(frame.columns):
            values = _string_values(frame[column])
            mapping = self.frequency_maps_[index] if index < len(self.frequency_maps_) else {}
            encoded_columns.append(values.map(mapping).fillna(0.0).astype(float).to_numpy())
        if not encoded_columns:
            return np.empty((len(frame), 0), dtype=float)
        return np.column_stack(encoded_columns)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is None:
            return np.asarray([], dtype=object)
        return np.asarray([str(feature) for feature in input_features], dtype=object)


def _numeric_imputer(strategy_name: str) -> SimpleImputer:
    if strategy_name == "constant_zero":
        return SimpleImputer(strategy="constant", fill_value=0.0)
    return SimpleImputer(strategy=strategy_name)


def _categorical_imputer(strategy_name: str) -> SimpleImputer:
    if strategy_name == "constant_missing":
        return SimpleImputer(strategy="constant", fill_value="missing")
    return SimpleImputer(strategy=strategy_name)


def _categorical_encoder(strategy_name: str) -> Any:
    if strategy_name == "one_hot":
        return _make_one_hot_encoder()
    if strategy_name == "ordinal":
        return _make_ordinal_encoder()
    if strategy_name == "frequency":
        return FrequencyEncoder()
    raise ValueError("Unsupported categorical_encoding_strategy.")


def _preprocessor_feature_names(preprocessor: Any) -> list[str] | None:
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:
        return None


def preprocessing_plan_to_clean_config(
    plan: PreprocessingPlan | dict[str, Any],
    *,
    target: str,
    task_type: str,
) -> CleanConfig:
    payload = artifact_to_dict(plan)
    global_params = payload.get("global_params", {}) if isinstance(payload, dict) else {}
    raw_steps = payload.get("steps", []) if isinstance(payload, dict) else []
    applied_step_ids = payload.get("applied_step_ids", []) if isinstance(payload, dict) else []
    applied_ids = {str(item) for item in applied_step_ids if str(item).strip()}

    resolved_steps: dict[str, dict[str, Any]] = {}
    for raw_step in raw_steps:
        if not isinstance(raw_step, dict):
            continue
        step_id = str(raw_step.get("id") or "")
        if applied_ids and step_id not in applied_ids:
            continue
        if not bool(raw_step.get("enabled", True)):
            continue
        kind = str(raw_step.get("kind") or "").strip()
        if not kind:
            continue
        resolved_steps[kind] = raw_step

    missing_step = resolved_steps.get("missing_value", {})
    encoding_step = resolved_steps.get("categorical_encoding", {})
    scaling_step = resolved_steps.get("numeric_scaling", {})
    feature_step = resolved_steps.get("feature_engineering", {})

    missing_params = missing_step.get("params", {}) if isinstance(missing_step, dict) else {}
    encoding_params = encoding_step.get("params", {}) if isinstance(encoding_step, dict) else {}
    scaling_params = scaling_step.get("params", {}) if isinstance(scaling_step, dict) else {}
    feature_params = feature_step.get("params", {}) if isinstance(feature_step, dict) else {}

    feature_operations: list[FeatureEngineeringOperation] = []
    if bool(feature_params.get("enabled", True)):
        for raw_operation in feature_params.get("operations", []) or []:
            if not isinstance(raw_operation, dict):
                continue
            feature_operations.append(
                FeatureEngineeringOperation(
                    operation=str(raw_operation.get("operation", "")),
                    source_column=str(raw_operation.get("source_column") or "") or None,
                    columns=[str(item) for item in raw_operation.get("columns", []) if str(item).strip()],
                    parts=[str(item) for item in raw_operation.get("parts", []) if str(item).strip()],
                    operator=str(raw_operation.get("operator") or "") or None,
                    bins=int(raw_operation["bins"]) if raw_operation.get("bins") is not None else None,
                    mapping={
                        str(key): str(value)
                        for key, value in dict(raw_operation.get("mapping", {})).items()
                        if str(key).strip()
                    },
                    default_value=str(raw_operation.get("default_value") or "other"),
                    rationale=str(raw_operation.get("rationale") or ""),
                )
            )

    return CleanConfig(
        target=target,
        task_type=task_type,
        test_size=float(global_params.get("test_size", 0.2)),
        random_state=int(global_params.get("random_state", 42)),
        high_missing_threshold=float(missing_params.get("high_missing_threshold", 0.9)),
        numeric_imputation_strategy=str(missing_params.get("numeric_imputation_strategy", "median")),
        categorical_imputation_strategy=str(missing_params.get("categorical_imputation_strategy", "most_frequent")),
        categorical_encoding_strategy=str(encoding_params.get("strategy", "one_hot")),
        standardize_numeric=bool(scaling_params.get("standardize_numeric", True)),
        feature_engineering_operations=feature_operations or None,
    )


def clean_and_split(df: pd.DataFrame, config: CleanConfig, tracker: DataFlowTracker | None = None) -> CleanedData:
    if config.target not in df.columns:
        raise ValueError(f"Target column not found: {config.target}")
    if config.task_type not in {"classification", "regression"}:
        raise ValueError("task_type must be 'classification' or 'regression'.")
    if config.numeric_imputation_strategy not in {"median", "mean", "most_frequent", "constant_zero"}:
        raise ValueError("Unsupported numeric_imputation_strategy.")
    if config.categorical_imputation_strategy not in {"most_frequent", "constant_missing"}:
        raise ValueError("Unsupported categorical_imputation_strategy.")
    if config.categorical_encoding_strategy not in {"one_hot", "ordinal", "frequency"}:
        raise ValueError("Unsupported categorical_encoding_strategy.")

    logger.info("Cleaning dataset rows=%d columns=%d target=%s task_type=%s", len(df), len(df.columns), config.target, config.task_type)
    working = df.copy()
    log: list[dict[str, Any]] = []
    if tracker is not None:
        tracker.snapshot_dataframe(
            "cleaning_input",
            "Cleaning input",
            "cleaning",
            working,
            metadata={"target": config.target, "task_type": config.task_type},
        )

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
    if tracker is not None:
        tracker.snapshot_dataframe(
            "after_target_drop",
            "After target drop",
            "cleaning",
            working,
            metadata={"rows_removed": before_rows - len(working)},
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
        logger.info("Dropped high-missing features threshold=%.2f count=%d", config.high_missing_threshold, len(high_missing_columns))
    log.append(
        {
            "step": "drop_high_missing_features",
            "threshold": config.high_missing_threshold,
            "columns": high_missing_columns,
        }
    )
    if tracker is not None:
        tracker.snapshot_dataframe(
            "after_high_missing_drop",
            "After high-missing drop",
            "cleaning",
            working,
            metadata={
                "high_missing_threshold": config.high_missing_threshold,
                "dropped_columns": high_missing_columns,
            },
        )

    feature_columns = [column for column in working.columns if column != config.target]
    constant_columns = [
        column for column in feature_columns if int(working[column].nunique(dropna=True)) <= 1
    ]
    if constant_columns:
        working = working.drop(columns=constant_columns)
        logger.info("Dropped constant features count=%d", len(constant_columns))
    log.append({"step": "drop_constant_features", "columns": constant_columns})
    if tracker is not None:
        tracker.snapshot_dataframe(
            "after_constant_drop",
            "After constant drop",
            "cleaning",
            working,
            metadata={"dropped_columns": constant_columns},
        )

    feature_columns = [column for column in working.columns if column != config.target]
    if not feature_columns:
        raise ValueError("No feature columns remain after cleaning.")

    X = working[feature_columns]
    y = working[config.target]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    numeric_steps: list[tuple[str, Any]] = [("imputer", _numeric_imputer(config.numeric_imputation_strategy))]
    if config.standardize_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers: list[tuple[str, Any, Any]] = [
        ("numeric", Pipeline(numeric_steps), make_column_selector(dtype_include="number")),
        (
            "categorical",
            Pipeline(
                [
                    ("imputer", _categorical_imputer(config.categorical_imputation_strategy)),
                    ("encoder", _categorical_encoder(config.categorical_encoding_strategy)),
                ]
            ),
            make_column_selector(dtype_exclude="number"),
        )
    ]
    column_preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
    feature_operations = list(config.feature_engineering_operations or [])
    if feature_operations:
        preprocessor = Pipeline(
            [
                ("feature_engineering", FeatureEngineeringTransformer(feature_operations)),
                ("columns", column_preprocessor),
            ]
        )
    else:
        preprocessor = column_preprocessor
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
    logger.info("Split complete train=%d test=%d stratified=%s", len(X_train), len(X_test), stratify is not None)
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
            "numeric_imputation_strategy": config.numeric_imputation_strategy,
            "categorical_imputation_strategy": config.categorical_imputation_strategy,
            "categorical_encoding_strategy": config.categorical_encoding_strategy,
            "standardize_numeric": config.standardize_numeric,
            "feature_engineering_operations": [operation.operation for operation in feature_operations],
        }
    )
    if tracker is not None:
        tracker.snapshot_dataframe(
            "train_split",
            "Train split",
            "cleaning",
            X_train,
            partition="train",
            metadata={"target_rows": len(y_train)},
        )
        tracker.snapshot_dataframe(
            "test_split",
            "Test split",
            "cleaning",
            X_test,
            partition="test",
            metadata={"target_rows": len(y_test), "stratified": stratify is not None},
        )
        tracker.snapshot_artifact(
            "preprocessor_plan",
            "Preprocessor plan",
            "cleaning",
            metadata={
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
                "numeric_imputation_strategy": config.numeric_imputation_strategy,
                "categorical_imputation_strategy": config.categorical_imputation_strategy,
                "categorical_encoding_strategy": config.categorical_encoding_strategy,
                "standardize_numeric": config.standardize_numeric,
                "feature_engineering_operations": [operation.operation for operation in feature_operations],
            },
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


def prepare_for_training(cleaned: CleanedData, tracker: DataFlowTracker | None = None) -> CleanedData:
    fitted_preprocessor = clone(cleaned.preprocessor)
    X_train_prepared = fitted_preprocessor.fit_transform(cleaned.X_train)
    X_test_prepared = fitted_preprocessor.transform(cleaned.X_test)
    feature_names = _preprocessor_feature_names(fitted_preprocessor)
    feature_count = len(feature_names) if feature_names is not None else X_train_prepared.shape[1]
    if feature_count == 0:
        raise ValueError("Preprocessor produced no features.")

    preparation_log = list(cleaned.cleaning_log)
    preparation_log.append(
        {
            "step": "prepare_training_data",
            "prepared_feature_count": feature_count,
            "numeric_imputation_strategy": cleaned.config.numeric_imputation_strategy,
            "categorical_imputation_strategy": cleaned.config.categorical_imputation_strategy,
            "categorical_encoding_strategy": cleaned.config.categorical_encoding_strategy,
            "standardize_numeric": cleaned.config.standardize_numeric,
        }
    )

    if tracker is not None:
        tracker.snapshot_matrix(
            "prepared_train_matrix",
            "Prepared train matrix",
            "preparation",
            matrix=X_train_prepared,
            partition="train",
            column_names=feature_names,
            metadata={"feature_count": feature_count},
        )
        tracker.snapshot_matrix(
            "prepared_test_matrix",
            "Prepared test matrix",
            "preparation",
            matrix=X_test_prepared,
            partition="test",
            column_names=feature_names,
            metadata={"feature_count": feature_count},
        )

    return CleanedData(
        X_train=cleaned.X_train,
        X_test=cleaned.X_test,
        y_train=cleaned.y_train,
        y_test=cleaned.y_test,
        preprocessor=cleaned.preprocessor,
        feature_columns=cleaned.feature_columns,
        numeric_features=cleaned.numeric_features,
        categorical_features=cleaned.categorical_features,
        cleaning_log=preparation_log,
        config=cleaned.config,
        fitted_preprocessor=fitted_preprocessor,
        X_train_prepared=X_train_prepared,
        X_test_prepared=X_test_prepared,
        prepared_feature_names=feature_names,
    )


def _string_values(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna(MISSING_TOKEN).astype(str)
