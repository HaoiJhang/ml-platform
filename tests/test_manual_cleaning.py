from pathlib import Path

import pandas as pd

from ml_platform.config import Settings
from ml_platform.manual_cleaning import apply_manual_cleaning_plan, suggest_manual_cleaning_plan, validate_manual_cleaning_plan


def _settings() -> Settings:
    return Settings(
        runs_dir=Path("runs"),
        data_dir=Path("data"),
        openai_api_key=None,
        openai_base_url=None,
        openai_model="gpt-4o-mini",
    )


def test_validate_manual_cleaning_plan_rejects_protected_target_columns() -> None:
    df = pd.DataFrame(
        {
            "target": [0, 1, 0],
            "secondary_target": [1, 0, 1],
            "segment": ["a", "b", "a"],
            "amount": [10.0, 20.0, 30.0],
        }
    )
    plan = validate_manual_cleaning_plan(
        {
            "planner_name": "manual",
            "effect_stage": "pre_eda",
            "rules": [
                {"id": "r1", "rule_type": "drop_column", "column": "target"},
                {"id": "r2", "rule_type": "drop_column", "column": "secondary_target"},
                {"id": "r3", "rule_type": "filter_row", "column": "segment", "operator": "contains", "value": "a"},
            ],
        },
        df,
        "target",
        protected_columns=["target", "secondary_target"],
    )

    assert len(plan.rules) == 1
    assert plan.rules[0].column == "segment"
    assert len(plan.rejected_rules) == 2


def test_apply_manual_cleaning_plan_applies_drop_and_filter_rules_in_order() -> None:
    df = pd.DataFrame(
        {
            "target": [0, 1, 1, 0],
            "customer_id": ["c1", "c2", "c3", "c4"],
            "segment": ["pro", "basic", "pro-plus", "pro"],
            "amount": [10.0, 20.0, 35.0, 50.0],
        }
    )
    cleaned, cleaning_log, impact = apply_manual_cleaning_plan(
        df,
        {
            "planner_name": "manual",
            "effect_stage": "pre_training",
            "rules": [
                {"id": "drop_id", "rule_type": "drop_column", "column": "customer_id"},
                {"id": "contains", "rule_type": "filter_row", "column": "segment", "operator": "contains", "value": "pro"},
                {"id": "gt", "rule_type": "filter_row", "column": "amount", "operator": "gt", "value": "30"},
            ],
        },
        "target",
    )

    assert "customer_id" not in cleaned.columns
    assert cleaned["amount"].tolist() == [35.0, 50.0]
    assert [entry["step"] for entry in cleaning_log] == ["manual_drop_columns", "manual_filter_rows"]
    assert impact["rows_removed"] == 2
    assert impact["columns_removed"] == ["customer_id"]


def test_apply_manual_cleaning_plan_supports_equals_in_and_numeric_filters() -> None:
    df = pd.DataFrame(
        {
            "target": [1, 1, 0, 1],
            "status": ["active", "paused", "active", "active"],
            "tier": ["gold", "silver", "gold", "platinum"],
            "score": [0.8, 0.2, 0.9, 0.95],
        }
    )
    cleaned, _, impact = apply_manual_cleaning_plan(
        df,
        {
            "planner_name": "manual",
            "rules": [
                {"id": "equals", "rule_type": "filter_row", "column": "status", "operator": "equals", "value": "active"},
                {"id": "in", "rule_type": "filter_row", "column": "tier", "operator": "in", "value": ["gold", "platinum"]},
                {"id": "gte", "rule_type": "filter_row", "column": "score", "operator": "gte", "value": "0.9"},
            ],
        },
        "target",
    )

    assert cleaned["tier"].tolist() == ["gold", "platinum"]
    assert impact["rows_after"] == 2


def test_suggest_manual_cleaning_plan_parses_simple_rule_based_brief() -> None:
    df = pd.DataFrame(
        {
            "target": [0, 1, 0],
            "customer_id": ["c1", "c2", "c3"],
            "segment": ["a", "b", "a"],
            "amount": [10.0, 20.0, 30.0],
        }
    )
    plan = suggest_manual_cleaning_plan(
        df=df,
        target="target",
        settings=_settings(),
        user_brief="drop customer_id and keep rows where amount > 15 and segment equals a",
    )

    assert any(rule.rule_type == "drop_column" and rule.column == "customer_id" for rule in plan.rules)
    assert any(rule.operator == "gt" and rule.column == "amount" for rule in plan.rules)
    assert any(rule.operator == "equals" and rule.column == "segment" for rule in plan.rules)
