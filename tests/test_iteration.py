from pathlib import Path

import pandas as pd

from ml_platform.artifacts import FeatureEngineeringOperation, NextRunPlan
from ml_platform.config import Settings
from ml_platform.iteration import dataset_fingerprint, suggest_next_run_plan, validate_next_run_plan


def test_dataset_fingerprint_is_stable_and_changes_with_data() -> None:
    df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    same = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    changed = pd.DataFrame({"x": [1, 3], "y": ["a", "b"]})

    assert dataset_fingerprint(df) == dataset_fingerprint(same)
    assert dataset_fingerprint(df) != dataset_fingerprint(changed)


def test_validate_next_run_plan_rejects_invalid_changes_and_keeps_safe_ones(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "target": [0, 1, 0, 1],
            "amount": [10.0, 20.0, 30.0, 40.0],
            "city": ["a", "b", "a", "c"],
        }
    )
    current_config = {"excluded_columns": ["city"], "priority_metric": "accuracy", "time_budget": 30, "high_missing_threshold": 0.9}
    previous_run = {"run_id": "run-1"}
    plan = validate_next_run_plan(
        NextRunPlan(
            planner_name="openai+local_whitelist",
            parent_run_id="run-1",
            target="target",
            priority_metric="made_up_metric",
            excluded_columns_add=["target", "city", "amount"],
            time_budget=1000,
            high_missing_threshold=0.2,
            feature_engineering_operations=[
                FeatureEngineeringOperation(operation="numeric_binning", source_column="amount", bins=4),
                FeatureEngineeringOperation(operation="python_eval", source_column="city"),
            ],
        ),
        df=df,
        target="target",
        current_config=current_config,
        previous_run=previous_run,
    )

    assert plan.priority_metric == "auto"
    assert plan.excluded_columns_add == ["amount"]
    assert plan.time_budget is None
    assert plan.high_missing_threshold is None
    assert [item.operation for item in plan.feature_engineering_operations] == ["numeric_binning"]
    assert len(plan.rejected_changes) >= 4


def test_local_next_run_plan_uses_previous_artifacts(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "target": ["yes", "no"] * 20,
            "customer_id": [f"c{i}" for i in range(40)],
            "city": [f"city_{i % 12}" for i in range(40)],
            "amount": [float(10 + i) for i in range(40)],
        }
    )
    settings = Settings(
        runs_dir=tmp_path,
        data_dir=tmp_path / "data",
        openai_api_key=None,
        openai_base_url=None,
        openai_model="test",
    )
    previous_run = {
        "run_id": "run-2",
        "config": {"priority_metric": "accuracy"},
        "metrics": {"accuracy": 0.7},
        "validation_pre": {
            "recommended_excluded_columns": ["customer_id"],
            "detected_leakage_columns": [],
            "issues": [{"code": "class_imbalance"}],
            "feature_count": 3,
        },
        "validation_post": {"issues": [{"code": "generalization_gap"}]},
        "recommendations": {"summary": ["Train and test metrics diverge materially."]},
    }

    plan = suggest_next_run_plan(
        df=df,
        target="target",
        current_config={
            "priority_metric": "accuracy",
            "excluded_columns": [],
            "time_budget": 30,
            "high_missing_threshold": 0.9,
        },
        previous_run=previous_run,
        settings=settings,
        user_brief="",
    )

    assert plan.parent_run_id == "run-2"
    assert plan.priority_metric == "recall_weighted"
    assert "customer_id" in plan.excluded_columns_add
    assert plan.time_budget == 60
