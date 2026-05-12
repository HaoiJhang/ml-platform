from __future__ import annotations

import json
from importlib import import_module
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ml_platform.artifacts import FeatureEngineeringOperation, FeatureEngineeringPlan
from ml_platform.config import Settings

logger = logging.getLogger(__name__)

ALLOWED_OPERATIONS = {
    "date_parts",
    "frequency_encoding",
    "numeric_binning",
    "numeric_interaction",
    "categorical_mapping",
}
ALLOWED_DATE_PARTS = {"year", "month", "day", "dayofweek", "quarter", "is_weekend"}
ALLOWED_INTERACTIONS = {"ratio", "difference", "sum", "product"}
MAX_OPERATIONS = 12
MAX_BINS = 10
MISSING_TOKEN = "__missing__"


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item) for item in value if isinstance(item, str) and str(item).strip()]
    return []


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def suggest_feature_engineering_plan(
    df: pd.DataFrame,
    target: str,
    eda_summary: dict[str, Any],
    settings: Settings,
    user_brief: str = "",
) -> FeatureEngineeringPlan:
    rule_based = _rule_based_plan(df, target, eda_summary)
    if not settings.llm_enabled or not user_brief.strip():
        return rule_based

    try:
        llm_plan = _generate_llm_feature_plan(df, target, eda_summary, settings, user_brief)
    except Exception as exc:
        logger.warning("LLM feature plan generation failed, using rule-based plan: %s", exc)
        return rule_based

    validated = validate_feature_engineering_plan(llm_plan.operations, df, target)
    operations = validated.operations or rule_based.operations
    notes = (llm_plan.notes or []) + rule_based.notes
    return FeatureEngineeringPlan(
        planner_name="llm+local_whitelist",
        operations=operations,
        rejected_operations=validated.rejected_operations,
        notes=notes,
    )


def validate_feature_engineering_plan(
    operations: list[FeatureEngineeringOperation],
    df: pd.DataFrame,
    target: str,
) -> FeatureEngineeringPlan:
    accepted: list[FeatureEngineeringOperation] = []
    rejected: list[str] = []
    columns = set(df.columns)
    feature_columns = columns - {target}
    numeric_columns = set(df[list(feature_columns)].select_dtypes(include=["number"]).columns)

    for operation in operations[:MAX_OPERATIONS]:
        name = operation.operation
        if name not in ALLOWED_OPERATIONS:
            rejected.append(f"Rejected unsupported operation: {name}.")
            continue

        source = operation.source_column
        if name in {"date_parts", "frequency_encoding", "numeric_binning", "categorical_mapping"}:
            if not source or source not in feature_columns:
                rejected.append(f"Rejected {name}: source column is missing, unknown, or target-like.")
                continue

        if name == "date_parts":
            parts = [part for part in operation.parts if part in ALLOWED_DATE_PARTS]
            if not parts:
                rejected.append(f"Rejected date_parts for {source}: no allowed parts were requested.")
                continue
            accepted.append(FeatureEngineeringOperation(name, source_column=source, parts=parts, rationale=operation.rationale))
            continue

        if name == "frequency_encoding":
            accepted.append(FeatureEngineeringOperation(name, source_column=source, rationale=operation.rationale))
            continue

        if name == "numeric_binning":
            if source not in numeric_columns:
                rejected.append(f"Rejected numeric_binning for {source}: source is not numeric.")
                continue
            bins = min(max(int(operation.bins or 5), 2), MAX_BINS)
            accepted.append(FeatureEngineeringOperation(name, source_column=source, bins=bins, rationale=operation.rationale))
            continue

        if name == "numeric_interaction":
            if len(operation.columns) != 2 or any(column not in numeric_columns for column in operation.columns):
                rejected.append("Rejected numeric_interaction: both columns must be known numeric feature columns.")
                continue
            operator = operation.operator or ""
            if operator not in ALLOWED_INTERACTIONS:
                rejected.append(f"Rejected numeric_interaction: unsupported operator {operator}.")
                continue
            accepted.append(
                FeatureEngineeringOperation(
                    name,
                    columns=list(operation.columns),
                    operator=operator,
                    rationale=operation.rationale,
                )
            )
            continue

        if name == "categorical_mapping":
            if source in numeric_columns:
                rejected.append(f"Rejected categorical_mapping for {source}: source is numeric.")
                continue
            clean_mapping = {
                str(key)[:100]: str(value)[:100]
                for key, value in list(operation.mapping.items())[:50]
                if str(key).strip() and str(value).strip()
            }
            if not clean_mapping:
                rejected.append(f"Rejected categorical_mapping for {source}: mapping is empty.")
                continue
            accepted.append(
                FeatureEngineeringOperation(
                    name,
                    source_column=source,
                    mapping=clean_mapping,
                    default_value=str(operation.default_value or "other")[:100],
                    rationale=operation.rationale,
                )
            )

    return FeatureEngineeringPlan(
        planner_name="local_whitelist",
        operations=accepted,
        rejected_operations=rejected,
        notes=[],
    )


def _rule_based_plan(df: pd.DataFrame, target: str, eda_summary: dict[str, Any]) -> FeatureEngineeringPlan:
    operations: list[FeatureEngineeringOperation] = []
    feature_columns = [column for column in df.columns if column != target]
    column_types = eda_summary.get("column_types", {})

    for column in column_types.get("datetime_like", []):
        if column in feature_columns:
            operations.append(
                FeatureEngineeringOperation(
                    operation="date_parts",
                    source_column=column,
                    parts=["year", "month", "dayofweek", "is_weekend"],
                    rationale="Datetime-like columns can expose calendar seasonality.",
                )
            )

    for column in feature_columns:
        series = df[column]
        non_null = max(len(series.dropna()), 1)
        unique_count = int(series.nunique(dropna=True))
        if not pd.api.types.is_numeric_dtype(series) and unique_count >= 10 and unique_count / non_null >= 0.4:
            operations.append(
                FeatureEngineeringOperation(
                    operation="frequency_encoding",
                    source_column=column,
                    rationale="High-cardinality categorical columns can benefit from train-fitted frequency signals.",
                )
            )

    return validate_feature_engineering_plan(operations, df, target)


def _generate_llm_feature_plan(
    df: pd.DataFrame,
    target: str,
    eda_summary: dict[str, Any],
    settings: Settings,
    user_brief: str,
) -> FeatureEngineeringPlan:
    client_cls = getattr(import_module("open" "ai"), "Open" "AI")
    client = client_cls(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    schema = {
        "allowed_operations": {
            "date_parts": {"source_column": "string", "parts": sorted(ALLOWED_DATE_PARTS)},
            "frequency_encoding": {"source_column": "string"},
            "numeric_binning": {"source_column": "numeric string", "bins": "integer 2-10"},
            "numeric_interaction": {"columns": ["numeric column", "numeric column"], "operator": sorted(ALLOWED_INTERACTIONS)},
            "categorical_mapping": {"source_column": "categorical string", "mapping": {"raw_value": "group"}, "default_value": "string"},
        },
        "rules": [
            "Use only provided feature columns. Never use the target column.",
            "Return only JSON. Do not write code.",
            "Prefer a small plan. Avoid speculative operations without column evidence.",
        ],
    }
    preview = {
        "target": target,
        "columns": [column for column in df.columns if column != target],
        "shape": eda_summary.get("shape", {}),
        "column_types": eda_summary.get("column_types", {}),
        "quality_warnings": eda_summary.get("quality_warnings", []),
    }
    response = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You propose safe tabular feature engineering plans as strict JSON for a local whitelist executor.",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_brief": user_brief,
                        "dataset_preview": preview,
                        "schema": schema,
                        "return_shape": {"operations": [], "notes": []},
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    )
    payload = json.loads(response.choices[0].message.content or "{}")
    operations = [_operation_from_payload(item) for item in _dict_list(payload.get("operations"))]
    notes = _string_list(payload.get("notes"))
    return FeatureEngineeringPlan(planner_name="llm", operations=operations, notes=notes)


def _operation_from_payload(payload: dict[str, Any]) -> FeatureEngineeringOperation:
    return FeatureEngineeringOperation(
        operation=str(payload.get("operation", "")),
        source_column=str(payload["source_column"]) if "source_column" in payload else None,
        columns=_string_list(payload.get("columns")),
        parts=_string_list(payload.get("parts")),
        operator=str(payload["operator"]) if "operator" in payload else None,
        bins=int(payload["bins"]) if isinstance(payload.get("bins"), int) else None,
        mapping={str(key): str(value) for key, value in payload.get("mapping", {}).items()}
        if isinstance(payload.get("mapping"), dict)
        else {},
        default_value=str(payload.get("default_value", "other")),
        rationale=str(payload.get("rationale", "")),
    )


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, operations: list[FeatureEngineeringOperation] | None = None):
        self.operations = operations or []

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FeatureEngineeringTransformer":
        self.input_features_ = [str(column) for column in X.columns]
        self.generated_features_ = self._generated_feature_names(X)
        self.frequency_maps_: dict[str, dict[str, float]] = {}
        self.bin_edges_: dict[str, np.ndarray] = {}

        for operation in self.operations:
            if operation.operation == "frequency_encoding" and operation.source_column in X.columns:
                values = _string_values(X[operation.source_column])
                self.frequency_maps_[operation.source_column] = values.value_counts(normalize=True, dropna=False).to_dict()
            elif operation.operation == "numeric_binning" and operation.source_column in X.columns:
                numeric = pd.to_numeric(X[operation.source_column], errors="coerce").dropna()
                if numeric.nunique() >= 2:
                    quantiles = np.linspace(0, 1, int(operation.bins or 5) + 1)
                    self.bin_edges_[operation.source_column] = np.unique(numeric.quantile(quantiles).to_numpy())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        output = X.copy()
        for operation in self.operations:
            if operation.operation == "date_parts":
                self._apply_date_parts(output, operation)
            elif operation.operation == "frequency_encoding":
                self._apply_frequency_encoding(output, operation)
            elif operation.operation == "numeric_binning":
                self._apply_numeric_binning(output, operation)
            elif operation.operation == "numeric_interaction":
                self._apply_numeric_interaction(output, operation)
            elif operation.operation == "categorical_mapping":
                self._apply_categorical_mapping(output, operation)
        return output

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        base = list(input_features) if input_features is not None else list(getattr(self, "input_features_", []))
        return np.asarray(base + list(getattr(self, "generated_features_", [])), dtype=object)

    def _generated_feature_names(self, X: pd.DataFrame) -> list[str]:
        names: list[str] = []
        existing = set(str(column) for column in X.columns)
        for operation in self.operations:
            if operation.operation == "date_parts" and operation.source_column in X.columns:
                for part in operation.parts:
                    names.append(_unique_name(f"fe__{operation.source_column}__{part}", existing | set(names)))
            elif operation.operation == "frequency_encoding" and operation.source_column in X.columns:
                names.append(_unique_name(f"fe__{operation.source_column}__freq", existing | set(names)))
            elif operation.operation == "numeric_binning" and operation.source_column in X.columns:
                names.append(_unique_name(f"fe__{operation.source_column}__bin", existing | set(names)))
            elif operation.operation == "numeric_interaction" and all(column in X.columns for column in operation.columns):
                left, right = operation.columns
                names.append(_unique_name(f"fe__{left}__{operation.operator}__{right}", existing | set(names)))
            elif operation.operation == "categorical_mapping" and operation.source_column in X.columns:
                names.append(_unique_name(f"fe__{operation.source_column}__mapped", existing | set(names)))
        return names

    def _apply_date_parts(self, output: pd.DataFrame, operation: FeatureEngineeringOperation) -> None:
        source = operation.source_column
        if source not in output.columns:
            return
        parsed = pd.to_datetime(output[source], errors="coerce")
        existing = set(output.columns)
        for part in operation.parts:
            name = _unique_name(f"fe__{source}__{part}", existing)
            existing.add(name)
            if part == "year":
                output[name] = parsed.dt.year
            elif part == "month":
                output[name] = parsed.dt.month
            elif part == "day":
                output[name] = parsed.dt.day
            elif part == "dayofweek":
                output[name] = parsed.dt.dayofweek
            elif part == "quarter":
                output[name] = parsed.dt.quarter
            elif part == "is_weekend":
                output[name] = parsed.dt.dayofweek.isin([5, 6]).astype(float)

    def _apply_frequency_encoding(self, output: pd.DataFrame, operation: FeatureEngineeringOperation) -> None:
        source = operation.source_column
        if source not in output.columns:
            return
        name = _unique_name(f"fe__{source}__freq", set(output.columns))
        mapping = self.frequency_maps_.get(source, {})
        output[name] = _string_values(output[source]).map(mapping).fillna(0.0).astype(float)

    def _apply_numeric_binning(self, output: pd.DataFrame, operation: FeatureEngineeringOperation) -> None:
        source = operation.source_column
        if source not in output.columns:
            return
        name = _unique_name(f"fe__{source}__bin", set(output.columns))
        edges = self.bin_edges_.get(source)
        numeric = pd.to_numeric(output[source], errors="coerce")
        if edges is None or len(edges) < 2:
            output[name] = -1.0
            return
        output[name] = pd.cut(numeric, bins=edges, labels=False, include_lowest=True).astype(float).fillna(-1.0)

    def _apply_numeric_interaction(self, output: pd.DataFrame, operation: FeatureEngineeringOperation) -> None:
        if len(operation.columns) != 2 or any(column not in output.columns for column in operation.columns):
            return
        left, right = operation.columns
        name = _unique_name(f"fe__{left}__{operation.operator}__{right}", set(output.columns))
        left_values = pd.to_numeric(output[left], errors="coerce")
        right_values = pd.to_numeric(output[right], errors="coerce")
        if operation.operator == "ratio":
            values = left_values / right_values.replace(0, np.nan)
        elif operation.operator == "difference":
            values = left_values - right_values
        elif operation.operator == "sum":
            values = left_values + right_values
        else:
            values = left_values * right_values
        output[name] = values.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _apply_categorical_mapping(self, output: pd.DataFrame, operation: FeatureEngineeringOperation) -> None:
        source = operation.source_column
        if source not in output.columns:
            return
        name = _unique_name(f"fe__{source}__mapped", set(output.columns))
        output[name] = _string_values(output[source]).map(operation.mapping).fillna(operation.default_value)


def _string_values(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna(MISSING_TOKEN).astype(str)


def _unique_name(base: str, existing: set[str]) -> str:
    clean = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(base))
    name = clean[:120]
    counter = 2
    while name in existing:
        suffix = f"__{counter}"
        name = f"{clean[: 120 - len(suffix)]}{suffix}"
        counter += 1
    return name
