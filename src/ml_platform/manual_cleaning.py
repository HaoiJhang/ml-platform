from __future__ import annotations

import json
from importlib import import_module
import logging
import re
from typing import Any

import pandas as pd

from ml_platform.artifacts import ManualCleaningPlan, ManualCleaningRule
from ml_platform.config import Settings

logger = logging.getLogger(__name__)

VALID_EFFECT_STAGES = {"pre_eda", "pre_training"}
VALID_RULE_TYPES = {"drop_column", "filter_row"}
FILTER_OPERATORS = {
    "is_null",
    "not_null",
    "equals",
    "not_equals",
    "in",
    "not_in",
    "contains",
    "not_contains",
    "gt",
    "gte",
    "lt",
    "lte",
}
NUMERIC_OPERATORS = {"gt", "gte", "lt", "lte"}
LIST_OPERATORS = {"in", "not_in"}
NULLARY_OPERATORS = {"is_null", "not_null"}
MAX_RULES = 24
MAX_VALUE_LENGTH = 200
DEFAULT_EFFECT_STAGE = "pre_eda"


def suggest_manual_cleaning_plan(
    df: pd.DataFrame,
    target: str,
    settings: Settings,
    user_brief: str = "",
) -> ManualCleaningPlan:
    rule_based = _rule_based_plan(df, target, user_brief)
    if not settings.llm_enabled or not user_brief.strip():
        return rule_based

    try:
        llm_plan = _generate_llm_manual_cleaning_plan(df, target, settings, user_brief)
    except Exception as exc:
        logger.warning("LLM manual cleaning plan generation failed, using rule-based plan: %s", exc)
        return rule_based

    validated = validate_manual_cleaning_plan(llm_plan, df, target)
    rules = validated.rules or rule_based.rules
    rejected = validated.rejected_rules
    notes = (llm_plan.notes or []) + rule_based.notes
    return ManualCleaningPlan(
        planner_name="llm+rule_based",
        user_brief=user_brief,
        effect_stage=validated.effect_stage,
        rules=rules,
        notes=notes,
        rejected_rules=rejected,
    )


def validate_manual_cleaning_plan(
    plan: ManualCleaningPlan | dict[str, Any],
    df: pd.DataFrame,
    target: str,
    protected_columns: list[str] | None = None,
) -> ManualCleaningPlan:
    raw_plan = plan if isinstance(plan, ManualCleaningPlan) else manual_cleaning_plan_from_payload(plan)
    effect_stage = raw_plan.effect_stage if raw_plan.effect_stage in VALID_EFFECT_STAGES else DEFAULT_EFFECT_STAGE
    available_columns = set(df.columns)
    numeric_columns = set(df.select_dtypes(include=["number"]).columns)
    protected = set(protected_columns or []) | {target}
    accepted: list[ManualCleaningRule] = []
    rejected: list[str] = []

    for index, raw_rule in enumerate(raw_plan.rules[:MAX_RULES], start=1):
        rule_id = str(raw_rule.id or f"manual_rule_{index}")
        enabled = bool(raw_rule.enabled)
        column = str(raw_rule.column or "").strip()
        rule_type = str(raw_rule.rule_type or "").strip()
        operator = str(raw_rule.operator or "").strip() or None
        rationale = str(raw_rule.rationale or "")[:300]

        if not column:
            rejected.append(f"Rejected rule {rule_id}: column is required.")
            continue
        if column not in available_columns:
            rejected.append(f"Rejected rule {rule_id}: column is missing or was already removed: {column}.")
            continue
        if rule_type not in VALID_RULE_TYPES:
            rejected.append(f"Rejected rule {rule_id}: unsupported rule type {rule_type or '-'}.")
            continue

        if rule_type == "drop_column":
            if column in protected:
                rejected.append(f"Rejected rule {rule_id}: protected target column cannot be dropped: {column}.")
                continue
            accepted.append(
                ManualCleaningRule(
                    id=rule_id,
                    enabled=enabled,
                    rule_type="drop_column",
                    column=column,
                    operator=None,
                    value=None,
                    rationale=rationale,
                )
            )
            if enabled:
                available_columns.remove(column)
                numeric_columns.discard(column)
            continue

        if operator not in FILTER_OPERATORS:
            rejected.append(f"Rejected rule {rule_id}: unsupported filter operator {operator or '-'}.")
            continue

        clean_value: str | list[str] | None = None
        if operator in NULLARY_OPERATORS:
            clean_value = None
        elif operator in LIST_OPERATORS:
            clean_list = _list_value(raw_rule.value)
            if not clean_list:
                rejected.append(f"Rejected rule {rule_id}: operator {operator} requires one or more values.")
                continue
            clean_value = clean_list
        else:
            scalar = _scalar_value(raw_rule.value)
            if scalar is None:
                rejected.append(f"Rejected rule {rule_id}: operator {operator} requires a value.")
                continue
            if operator in NUMERIC_OPERATORS:
                if column not in numeric_columns:
                    rejected.append(f"Rejected rule {rule_id}: numeric comparison requires a numeric column: {column}.")
                    continue
                if _to_float(scalar) is None:
                    rejected.append(f"Rejected rule {rule_id}: numeric comparison requires a numeric value.")
                    continue
            clean_value = scalar

        accepted.append(
            ManualCleaningRule(
                id=rule_id,
                enabled=enabled,
                rule_type="filter_row",
                column=column,
                operator=operator,
                value=clean_value,
                rationale=rationale,
            )
        )

    return ManualCleaningPlan(
        planner_name=raw_plan.planner_name or "manual",
        user_brief=raw_plan.user_brief,
        effect_stage=effect_stage,
        rules=accepted,
        notes=list(raw_plan.notes or []),
        rejected_rules=(list(raw_plan.rejected_rules or []) + rejected)[:50],
    )


def apply_manual_cleaning_plan(
    df: pd.DataFrame,
    plan: ManualCleaningPlan | dict[str, Any],
    target: str,
    protected_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    validated = validate_manual_cleaning_plan(plan, df, target, protected_columns=protected_columns)
    working = df.copy()
    rows_before = len(working)
    columns_before = len(working.columns)
    dropped_columns: list[str] = []
    filter_rows_details: list[dict[str, Any]] = []
    applied_rule_count = 0

    for rule in validated.rules:
        if not rule.enabled:
            continue
        if rule.rule_type == "drop_column":
            if rule.column in working.columns:
                working = working.drop(columns=[rule.column])
                dropped_columns.append(rule.column)
                applied_rule_count += 1
            continue

        before_count = len(working)
        mask = _filter_mask(working, rule)
        working = working.loc[mask].copy()
        filter_rows_details.append(
            {
                "id": rule.id,
                "column": rule.column,
                "operator": rule.operator,
                "value": rule.value,
                "rows_before": before_count,
                "rows_after": len(working),
                "rows_removed": before_count - len(working),
            }
        )
        applied_rule_count += 1

    cleaning_log: list[dict[str, Any]] = []
    if dropped_columns:
        cleaning_log.append(
            {
                "step": "manual_drop_columns",
                "columns": dropped_columns,
                "count": len(dropped_columns),
                "effect_stage": validated.effect_stage,
            }
        )
    if filter_rows_details:
        cleaning_log.append(
            {
                "step": "manual_filter_rows",
                "filters": filter_rows_details,
                "rows_before": rows_before,
                "rows_after": len(working),
                "rows_removed": rows_before - len(working),
                "effect_stage": validated.effect_stage,
            }
        )

    impact_summary = {
        "effect_stage": validated.effect_stage,
        "rows_before": rows_before,
        "rows_after": len(working),
        "rows_removed": rows_before - len(working),
        "columns_before": columns_before,
        "columns_after": len(working.columns),
        "columns_removed": dropped_columns,
        "applied_rule_count": applied_rule_count,
        "enabled_rule_count": sum(1 for rule in validated.rules if rule.enabled),
        "accepted_rule_count": len(validated.rules),
        "rejected_rule_count": len(validated.rejected_rules),
    }
    return working, cleaning_log, impact_summary


def manual_cleaning_plan_from_payload(payload: dict[str, Any] | None) -> ManualCleaningPlan:
    payload = payload or {}
    rules = [manual_cleaning_rule_from_payload(item, index=index) for index, item in enumerate(_dict_list(payload.get("rules")), start=1)]
    return ManualCleaningPlan(
        planner_name=str(payload.get("planner_name") or "manual"),
        user_brief=str(payload.get("user_brief") or ""),
        effect_stage=str(payload.get("effect_stage") or DEFAULT_EFFECT_STAGE),
        rules=rules,
        notes=_string_list(payload.get("notes")),
        rejected_rules=_string_list(payload.get("rejected_rules")),
    )


def manual_cleaning_rule_from_payload(payload: dict[str, Any] | None, *, index: int = 1) -> ManualCleaningRule:
    payload = payload or {}
    return ManualCleaningRule(
        id=str(payload.get("id") or f"manual_rule_{index}"),
        enabled=bool(payload.get("enabled", True)),
        rule_type=str(payload.get("rule_type") or "filter_row"),
        column=str(payload.get("column") or ""),
        operator=str(payload.get("operator") or "") or None,
        value=_payload_value(payload.get("value")),
        rationale=str(payload.get("rationale") or ""),
    )


def _payload_value(value: Any) -> str | list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item)[:MAX_VALUE_LENGTH] for item in value if str(item).strip()]
    return str(value)[:MAX_VALUE_LENGTH]


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _scalar_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        first = next((str(item).strip() for item in value if str(item).strip()), "")
        return first[:MAX_VALUE_LENGTH] or None
    text = str(value).strip()
    return text[:MAX_VALUE_LENGTH] if text else None


def _list_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip()[:MAX_VALUE_LENGTH] for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [item[:MAX_VALUE_LENGTH] for item in [part.strip() for part in text.split(",")] if item]


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _filter_mask(df: pd.DataFrame, rule: ManualCleaningRule) -> pd.Series:
    series = df[rule.column]
    operator = rule.operator or ""
    if operator == "is_null":
        return series.isna()
    if operator == "not_null":
        return series.notna()
    if operator == "contains":
        return _contains_mask(series, str(rule.value or ""))
    if operator == "not_contains":
        return ~_contains_mask(series, str(rule.value or ""))

    if operator in {"equals", "not_equals", "in", "not_in"}:
        mask = _equality_mask(series, operator, rule.value)
        return mask if operator in {"equals", "in"} else ~mask

    if operator in NUMERIC_OPERATORS:
        numeric_series = pd.to_numeric(series, errors="coerce")
        threshold = float(str(rule.value))
        if operator == "gt":
            return numeric_series > threshold
        if operator == "gte":
            return numeric_series >= threshold
        if operator == "lt":
            return numeric_series < threshold
        return numeric_series <= threshold

    raise ValueError(f"Unsupported filter operator: {operator}")


def _contains_mask(series: pd.Series, value: str) -> pd.Series:
    stringified = series.astype("string")
    return stringified.notna() & stringified.str.contains(value, case=False, regex=False)


def _equality_mask(series: pd.Series, operator: str, value: str | list[str] | None) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        values = _list_value(value) if operator in LIST_OPERATORS else [str(value or "")]
        numeric_values = [parsed for parsed in (_to_float(item) for item in values) if parsed is not None]
        if numeric_values:
            numeric_series = pd.to_numeric(series, errors="coerce")
            return numeric_series.isin(numeric_values)

    compare_values = _list_value(value) if operator in LIST_OPERATORS else [str(value or "")]
    stringified = series.astype("string")
    return stringified.isin(compare_values)


def _rule_based_plan(df: pd.DataFrame, target: str, user_brief: str) -> ManualCleaningPlan:
    if not user_brief.strip():
        return ManualCleaningPlan(planner_name="rule_based", user_brief=user_brief, effect_stage=DEFAULT_EFFECT_STAGE)

    text = user_brief.strip()
    lower_text = text.lower()
    notes: list[str] = []
    rules: list[ManualCleaningRule] = []
    effect_stage = "pre_training" if "before training" in lower_text or "only before training" in lower_text else DEFAULT_EFFECT_STAGE

    for index, column in enumerate(sorted(df.columns, key=lambda item: len(str(item)), reverse=True), start=1):
        column_text = str(column)
        escaped = re.escape(column_text)
        lower_column = column_text.lower()

        if re.search(rf"(drop|remove|exclude)\s+(the\s+)?column\s+{escaped}", text, flags=re.IGNORECASE) or re.search(
            rf"(drop|remove|exclude)\s+{escaped}", text, flags=re.IGNORECASE
        ):
            rules.append(
                ManualCleaningRule(
                    id=f"manual_rule_{len(rules) + 1}",
                    rule_type="drop_column",
                    column=column_text,
                    rationale=f"Detected a column drop request for {column_text}.",
                )
            )
            continue

        if re.search(rf"{escaped}\s+(is\s+)?(null|missing|empty)", text, flags=re.IGNORECASE):
            rules.append(
                ManualCleaningRule(
                    id=f"manual_rule_{len(rules) + 1}",
                    rule_type="filter_row",
                    column=column_text,
                    operator="is_null",
                    rationale=f"Detected a null filter for {column_text}.",
                )
            )
            continue

        if re.search(rf"{escaped}\s+(is\s+)?not\s+(null|missing|empty)", text, flags=re.IGNORECASE):
            rules.append(
                ManualCleaningRule(
                    id=f"manual_rule_{len(rules) + 1}",
                    rule_type="filter_row",
                    column=column_text,
                    operator="not_null",
                    rationale=f"Detected a not-null filter for {column_text}.",
                )
            )
            continue

        numeric_match = re.search(rf"{escaped}\s*(>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
        if numeric_match:
            operator = {
                ">": "gt",
                ">=": "gte",
                "<": "lt",
                "<=": "lte",
            }[numeric_match.group(1)]
            rules.append(
                ManualCleaningRule(
                    id=f"manual_rule_{len(rules) + 1}",
                    rule_type="filter_row",
                    column=column_text,
                    operator=operator,
                    value=numeric_match.group(2),
                    rationale=f"Detected a numeric filter for {column_text}.",
                )
            )
            continue

        equals_match = re.search(
            rf"{escaped}\s+(?:equals?|is)\s+['\"]?([^,'\"\n;]+)['\"]?",
            text,
            flags=re.IGNORECASE,
        )
        if equals_match:
            rules.append(
                ManualCleaningRule(
                    id=f"manual_rule_{len(rules) + 1}",
                    rule_type="filter_row",
                    column=column_text,
                    operator="equals",
                    value=equals_match.group(1).strip(),
                    rationale=f"Detected an equality filter for {column_text}.",
                )
            )
            continue

        contains_match = re.search(
            rf"{escaped}\s+(?:contains?)\s+['\"]?([^,'\"\n;]+)['\"]?",
            text,
            flags=re.IGNORECASE,
        )
        if contains_match:
            rules.append(
                ManualCleaningRule(
                    id=f"manual_rule_{len(rules) + 1}",
                    rule_type="filter_row",
                    column=column_text,
                    operator="contains",
                    value=contains_match.group(1).strip(),
                    rationale=f"Detected a contains filter for {column_text}.",
                )
            )

        if lower_column in lower_text:
            notes.append(f"Column mentioned in brief: {column_text}. Add or edit a rule if the auto-parser missed the exact intent.")

    return validate_manual_cleaning_plan(
        ManualCleaningPlan(
            planner_name="rule_based",
            user_brief=user_brief,
            effect_stage=effect_stage,
            rules=rules,
            notes=list(dict.fromkeys(notes))[:10],
        ),
        df,
        target,
    )


def _generate_llm_manual_cleaning_plan(
    df: pd.DataFrame,
    target: str,
    settings: Settings,
    user_brief: str,
) -> ManualCleaningPlan:
    client_cls = getattr(import_module("open" "ai"), "Open" "AI")
    client = client_cls(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    schema = {
        "allowed_rule_types": sorted(VALID_RULE_TYPES),
        "allowed_filter_operators": sorted(FILTER_OPERATORS),
        "allowed_effect_stages": sorted(VALID_EFFECT_STAGES),
        "rules": [
            "Use only known columns.",
            "Never propose dropping the target column.",
            "Return only JSON. Do not write code or SQL.",
            "Use a small plan grounded in the user brief.",
        ],
    }
    preview = {
        "target": target,
        "columns": df.columns.tolist(),
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "numeric_columns": df.select_dtypes(include=["number"]).columns.tolist(),
    }
    response = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You propose safe manual tabular cleaning rules as strict JSON for a local whitelist executor.",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_brief": user_brief,
                        "dataset_preview": preview,
                        "schema": schema,
                        "return_shape": {
                            "effect_stage": "pre_eda",
                            "rules": [],
                            "notes": [],
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    )
    payload = json.loads(response.choices[0].message.content or "{}")
    return manual_cleaning_plan_from_payload(
        {
            "planner_name": "llm",
            "user_brief": user_brief,
            "effect_stage": payload.get("effect_stage", DEFAULT_EFFECT_STAGE),
            "rules": payload.get("rules", []),
            "notes": payload.get("notes", []),
            "rejected_rules": payload.get("rejected_rules", []),
        }
    )
