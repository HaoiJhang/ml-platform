from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import pandas as pd

from ml_platform.artifacts import FeatureEngineeringOperation, NextRunPlan
from ml_platform.config import Settings
from ml_platform.feature_engineering import validate_feature_engineering_plan

logger = logging.getLogger(__name__)

VALID_PRIORITY_METRICS = {
    "auto",
    "accuracy",
    "f1_weighted",
    "precision_weighted",
    "recall_weighted",
    "roc_auc",
    "rmse",
    "mae",
    "r2",
}


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


def dataset_fingerprint(df: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes()
    schema = json.dumps(
        {
            "columns": [str(column) for column in df.columns],
            "dtypes": [str(dtype) for dtype in df.dtypes],
            "rows": len(df),
        },
        ensure_ascii=True,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(schema + payload).hexdigest()


def suggest_next_run_plan(
    df: pd.DataFrame,
    target: str,
    current_config: dict[str, Any],
    previous_run: dict[str, Any],
    settings: Settings,
    user_brief: str = "",
) -> NextRunPlan:
    rule_based = _rule_based_next_run(df, target, current_config, previous_run)
    if not settings.llm_enabled:
        return rule_based

    try:
        llm_plan = _generate_openai_next_run_plan(df, target, current_config, previous_run, settings, user_brief)
    except Exception as exc:
        logger.warning("OpenAI next-run planning failed, using rule-based plan: %s", exc)
        return rule_based
    return validate_next_run_plan(llm_plan, df, target, current_config, previous_run)


def validate_next_run_plan(
    plan: NextRunPlan,
    df: pd.DataFrame,
    target: str,
    current_config: dict[str, Any],
    previous_run: dict[str, Any],
) -> NextRunPlan:
    rejected: list[str] = list(plan.rejected_changes)
    columns = set(df.columns)
    feature_columns = columns - {target}
    current_excluded = set(current_config.get("excluded_columns", []))

    priority_metric = plan.priority_metric if plan.priority_metric in VALID_PRIORITY_METRICS else "auto"
    if plan.priority_metric not in VALID_PRIORITY_METRICS:
        rejected.append(f"Rejected unsupported priority metric: {plan.priority_metric}.")

    excluded_columns_add = []
    for column in plan.excluded_columns_add:
        if column not in feature_columns:
            rejected.append(f"Rejected exclusion for unknown or target column: {column}.")
            continue
        if column in current_excluded:
            continue
        excluded_columns_add.append(column)

    time_budget = None
    if plan.time_budget is not None:
        budget = int(plan.time_budget)
        if 5 <= budget <= 600:
            time_budget = budget
        else:
            rejected.append(f"Rejected time budget outside allowed range: {budget}.")

    high_missing_threshold = None
    if plan.high_missing_threshold is not None:
        threshold = float(plan.high_missing_threshold)
        if 0.5 <= threshold <= 1.0:
            high_missing_threshold = threshold
        else:
            rejected.append(f"Rejected high missing threshold outside allowed range: {threshold}.")

    validated_features = validate_feature_engineering_plan(plan.feature_engineering_operations, df, target)
    rejected.extend(validated_features.rejected_operations)

    return NextRunPlan(
        planner_name=plan.planner_name if validated_features.operations or excluded_columns_add or time_budget or high_missing_threshold or priority_metric != "auto" else "local_whitelist",
        parent_run_id=plan.parent_run_id or str(previous_run.get("run_id", "")),
        target=target,
        priority_metric=priority_metric,
        excluded_columns_add=excluded_columns_add,
        time_budget=time_budget,
        high_missing_threshold=high_missing_threshold,
        feature_engineering_operations=validated_features.operations,
        notes=plan.notes,
        risk_flags=plan.risk_flags,
        rejected_changes=rejected,
    )


def _rule_based_next_run(
    df: pd.DataFrame,
    target: str,
    current_config: dict[str, Any],
    previous_run: dict[str, Any],
) -> NextRunPlan:
    preflight = previous_run.get("validation_pre") or {}
    postrun = previous_run.get("validation_post") or {}
    recommendations = previous_run.get("recommendations") or {}
    metrics = previous_run.get("metrics") or {}

    excluded = list(dict.fromkeys((preflight.get("recommended_excluded_columns") or []) + (preflight.get("detected_leakage_columns") or [])))
    notes: list[str] = []
    risk_flags: list[str] = list(recommendations.get("summary", []))

    if excluded:
        notes.append(f"Previous validation surfaced columns worth excluding: {', '.join(excluded[:5])}.")

    priority_metric = current_config.get("priority_metric", "auto")
    if any(issue.get("code") == "class_imbalance" for issue in preflight.get("issues", [])) and priority_metric in {"auto", "accuracy"}:
        priority_metric = "recall_weighted"
        notes.append("Switched priority metric toward recall because the previous run showed class imbalance.")

    time_budget = None
    if any(issue.get("code") == "generalization_gap" for issue in postrun.get("issues", [])) and int(current_config.get("time_budget", 30)) < 60:
        time_budget = min(int(current_config.get("time_budget", 30)) * 2, 60)
        notes.append("Increased time budget modestly to compare a second baseline under the same split.")

    high_missing_threshold = None
    if preflight.get("feature_count", 0) < 5 and float(current_config.get("high_missing_threshold", 0.9)) < 0.95:
        high_missing_threshold = 0.95
        notes.append("Relaxed the missingness threshold because very few features remained.")

    feature_plan = validate_feature_engineering_plan([], df, target)
    if not previous_run.get("feature_engineering_plan"):
        feature_plan = validate_feature_engineering_plan(
            _default_feature_engineering_candidates(df, target),
            df,
            target,
        )
        if feature_plan.operations:
            notes.append("Added a conservative whitelist feature plan because the previous run did not use engineered features.")

    if metrics and not notes and not risk_flags:
        notes.append("No strong local iteration change was detected from the previous run artifacts.")

    return NextRunPlan(
        planner_name="local_whitelist",
        parent_run_id=str(previous_run.get("run_id", "")),
        target=target,
        priority_metric=str(priority_metric),
        excluded_columns_add=excluded,
        time_budget=time_budget,
        high_missing_threshold=high_missing_threshold,
        feature_engineering_operations=feature_plan.operations,
        notes=notes,
        risk_flags=risk_flags,
        rejected_changes=feature_plan.rejected_operations,
    )


def _default_feature_engineering_candidates(df: pd.DataFrame, target: str) -> list[FeatureEngineeringOperation]:
    operations: list[FeatureEngineeringOperation] = []
    features = [column for column in df.columns if column != target]
    for column in features:
        series = df[column]
        non_null = max(len(series.dropna()), 1)
        unique_count = int(series.nunique(dropna=True))
        if not pd.api.types.is_numeric_dtype(series) and unique_count >= 10 and unique_count / non_null >= 0.4:
            operations.append(FeatureEngineeringOperation(operation="frequency_encoding", source_column=column, rationale="High-cardinality categorical signal."))
    return operations


def _generate_openai_next_run_plan(
    df: pd.DataFrame,
    target: str,
    current_config: dict[str, Any],
    previous_run: dict[str, Any],
    settings: Settings,
    user_brief: str,
) -> NextRunPlan:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    payload = {
        "user_brief": user_brief,
        "target": target,
        "current_config": current_config,
        "dataset_preview": {
            "columns": [column for column in df.columns if column != target],
            "shape": {"rows": len(df), "columns": len(df.columns)},
        },
        "previous_run": {
            "run_id": previous_run.get("run_id"),
            "config": previous_run.get("config"),
            "metrics": previous_run.get("metrics"),
            "validation_pre": previous_run.get("validation_pre"),
            "validation_post": previous_run.get("validation_post"),
            "recommendations": previous_run.get("recommendations"),
            "feature_engineering_plan": previous_run.get("feature_engineering_plan"),
            "report": previous_run.get("report", "")[:4000],
        },
        "allowed_changes": {
            "priority_metric": sorted(VALID_PRIORITY_METRICS),
            "excluded_columns_add": "feature columns only",
            "time_budget": "integer 5-600",
            "high_missing_threshold": "float 0.5-1.0",
            "feature_engineering_operations": "same whitelist schema as existing local feature engineering plan",
        },
        "rules": [
            "Return only JSON.",
            "Do not suggest code.",
            "Do not use the target column as a feature.",
            "Keep the plan conservative and executable in one run.",
        ],
    }
    response = client.chat.completions.create(
        model=settings.openai_model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You propose the next local tabular ML run as strict JSON under a fixed whitelist.",
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    plan = json.loads(response.choices[0].message.content or "{}")
    return NextRunPlan(
        planner_name="openai+local_whitelist",
        parent_run_id=str(plan.get("parent_run_id") or previous_run.get("run_id", "")),
        target=target,
        priority_metric=str(plan.get("priority_metric", "auto")),
        excluded_columns_add=_string_list(plan.get("excluded_columns_add")),
        time_budget=int(plan["time_budget"]) if isinstance(plan.get("time_budget"), int) else None,
        high_missing_threshold=float(plan["high_missing_threshold"])
        if isinstance(plan.get("high_missing_threshold"), (int, float))
        else None,
        feature_engineering_operations=[
            FeatureEngineeringOperation(
                operation=str(item.get("operation", "")),
                source_column=str(item["source_column"]) if "source_column" in item else None,
                columns=_string_list(item.get("columns")),
                parts=_string_list(item.get("parts")),
                operator=str(item["operator"]) if "operator" in item else None,
                bins=int(item["bins"]) if isinstance(item.get("bins"), int) else None,
                mapping={str(key): str(value) for key, value in item.get("mapping", {}).items()}
                if isinstance(item.get("mapping"), dict)
                else {},
                default_value=str(item.get("default_value", "other")),
                rationale=str(item.get("rationale", "")),
            )
            for item in _dict_list(plan.get("feature_engineering_operations"))
        ],
        notes=_string_list(plan.get("notes")),
        risk_flags=_string_list(plan.get("risk_flags")),
        rejected_changes=[],
    )
