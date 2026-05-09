from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from ml_platform.artifacts import PlanSuggestion
from ml_platform.config import Settings


METRIC_KEYWORDS: list[tuple[str, str]] = [
    ("roc_auc", "roc_auc"),
    ("roc auc", "roc_auc"),
    ("auc", "roc_auc"),
    ("recall", "recall_weighted"),
    ("sensitivity", "recall_weighted"),
    ("precision", "precision_weighted"),
    ("f1", "f1_weighted"),
    ("accuracy", "accuracy"),
    ("rmse", "rmse"),
    ("mae", "mae"),
    ("r2", "r2"),
]


def suggest_plan(
    df: pd.DataFrame,
    eda_summary: dict[str, Any],
    settings: Settings,
    user_brief: str = "",
) -> PlanSuggestion:
    rule_based = _rule_based_plan(df, eda_summary, user_brief)
    if not settings.llm_enabled or not user_brief.strip():
        return rule_based

    try:
        llm_plan = _generate_openai_plan(df, eda_summary, settings, user_brief)
    except Exception:
        return rule_based
    return _merge_plan(rule_based, llm_plan, df.columns.tolist())


def _rule_based_plan(df: pd.DataFrame, eda_summary: dict[str, Any], user_brief: str) -> PlanSuggestion:
    notes: list[str] = []
    risks = list(eda_summary.get("quality_warnings", []))
    suggested_targets = _match_targets(df, user_brief)
    if suggested_targets:
        notes.append(f"Inferred target candidates from the brief: {', '.join(suggested_targets[:3])}.")
    else:
        suggested_targets = [df.columns[-1]]
        notes.append(f"No target was found in the brief, so the last column was suggested: {df.columns[-1]}.")

    primary_target = suggested_targets[0]
    task_type = _infer_task_type(df, primary_target)
    notes.append(f"Suggested task type for {primary_target}: {task_type}.")

    excluded_columns = _suggest_excluded_columns(df, primary_target)
    if excluded_columns:
        notes.append(f"Potential identifier columns were suggested for exclusion: {', '.join(excluded_columns[:5])}.")

    metric = _infer_priority_metric(user_brief, task_type)
    if metric != "auto":
        notes.append(f"Detected a priority metric preference from the brief: {metric}.")

    if len(df) < 50:
        risks.append("Very small dataset; treat the run as exploratory and expect unstable holdout metrics.")

    return PlanSuggestion(
        planner_name="rule_based",
        user_brief=user_brief,
        suggested_targets=suggested_targets[:3],
        suggested_task_type=task_type,
        suggested_excluded_columns=excluded_columns[:10],
        priority_metric=metric,
        notes=notes,
        risk_flags=risks[:10],
    )


def _generate_openai_plan(
    df: pd.DataFrame,
    eda_summary: dict[str, Any],
    settings: Settings,
    user_brief: str,
) -> PlanSuggestion:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    preview = {
        "columns": df.columns.tolist(),
        "shape": eda_summary.get("shape", {}),
        "quality_warnings": eda_summary.get("quality_warnings", []),
    }
    response = client.chat.completions.create(
        model=settings.openai_model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You help configure a local tabular ML experiment. "
                    "Return strict JSON with keys: suggested_targets, suggested_task_type, "
                    "suggested_excluded_columns, priority_metric, notes, risk_flags. "
                    "Only use values grounded in the provided schema and brief."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"user_brief": user_brief, "dataset_preview": preview},
                    ensure_ascii=False,
                ),
            },
        ],
    )
    content = response.choices[0].message.content or "{}"
    payload = json.loads(content)
    return PlanSuggestion(
        planner_name="openai",
        user_brief=user_brief,
        suggested_targets=[str(item) for item in payload.get("suggested_targets", []) if isinstance(item, str)],
        suggested_task_type=str(payload.get("suggested_task_type", "auto")),
        suggested_excluded_columns=[
            str(item) for item in payload.get("suggested_excluded_columns", []) if isinstance(item, str)
        ],
        priority_metric=str(payload.get("priority_metric", "auto")),
        notes=[str(item) for item in payload.get("notes", []) if isinstance(item, str)],
        risk_flags=[str(item) for item in payload.get("risk_flags", []) if isinstance(item, str)],
    )


def _merge_plan(rule_based: PlanSuggestion, llm_plan: PlanSuggestion, columns: list[str]) -> PlanSuggestion:
    valid_columns = set(columns)
    llm_targets = [column for column in llm_plan.suggested_targets if column in valid_columns]
    llm_excluded = [column for column in llm_plan.suggested_excluded_columns if column in valid_columns]
    task_type = llm_plan.suggested_task_type if llm_plan.suggested_task_type in {"auto", "classification", "regression"} else rule_based.suggested_task_type
    metric = llm_plan.priority_metric if llm_plan.priority_metric != "auto" else rule_based.priority_metric
    return PlanSuggestion(
        planner_name="openai+rule_based",
        user_brief=rule_based.user_brief,
        suggested_targets=llm_targets or rule_based.suggested_targets,
        suggested_task_type=task_type,
        suggested_excluded_columns=llm_excluded or rule_based.suggested_excluded_columns,
        priority_metric=metric,
        notes=(llm_plan.notes or []) + rule_based.notes,
        risk_flags=(llm_plan.risk_flags or []) + rule_based.risk_flags,
    )


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _match_targets(df: pd.DataFrame, user_brief: str) -> list[str]:
    if not user_brief.strip():
        return []
    normalized_brief = f" {_normalize(user_brief)} "
    matches: list[str] = []
    identifier_matches: list[str] = []
    for column in df.columns:
        normalized_column = _normalize(str(column))
        if normalized_column and f" {normalized_column} " in normalized_brief:
            if _is_identifier_name(str(column)):
                identifier_matches.append(column)
            else:
                matches.append(column)
    return matches or identifier_matches


def _suggest_excluded_columns(df: pd.DataFrame, target: str) -> list[str]:
    suggestions: list[str] = []
    for column in df.columns:
        if column == target:
            continue
        name = str(column).lower()
        unique_rate = float(df[column].nunique(dropna=True) / max(len(df[column].dropna()), 1))
        if _is_identifier_name(name):
            suggestions.append(column)
            continue
        if unique_rate >= 0.98 and len(df) >= 20 and not pd.api.types.is_numeric_dtype(df[column]):
            suggestions.append(column)
    return suggestions


def _infer_priority_metric(user_brief: str, task_type: str) -> str:
    normalized = _normalize(user_brief)
    for keyword, metric in METRIC_KEYWORDS:
        if keyword in normalized:
            return metric
    if task_type == "classification":
        return "f1_weighted"
    if task_type == "regression":
        return "rmse"
    return "auto"


def _infer_task_type(df: pd.DataFrame, target: str) -> str:
    series = df[target].dropna()
    if not pd.api.types.is_numeric_dtype(series):
        return "classification"
    unique_count = int(series.nunique())
    if unique_count <= max(20, int(len(series) * 0.05)):
        return "classification"
    return "regression"


def _is_identifier_name(name: str) -> bool:
    lowered = name.lower()
    return lowered == "id" or lowered.endswith("_id") or "uuid" in lowered
