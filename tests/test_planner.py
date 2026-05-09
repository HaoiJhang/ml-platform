import pandas as pd

from ml_platform.artifacts import PlanSuggestion
from ml_platform.config import Settings
from ml_platform.eda import generate_eda_summary
from ml_platform.planner import suggest_plan


def test_planner_suggests_target_metric_and_identifier_exclusion(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(30)],
            "tenure_days": list(range(30)),
            "monthly_spend": [50 + i for i in range(30)],
            "churn": ["yes" if i % 3 == 0 else "no" for i in range(30)],
        }
    )
    settings = Settings(
        runs_dir=tmp_path,
        data_dir=tmp_path / "data",
        openai_api_key=None,
        openai_base_url=None,
        openai_model="test",
    )

    plan = suggest_plan(
        df=df,
        eda_summary=generate_eda_summary(df, target="churn"),
        settings=settings,
        user_brief="predict churn and optimize recall while ignoring customer_id style columns",
    )

    assert plan.suggested_targets == ["churn"]
    assert plan.priority_metric == "recall_weighted"
    assert "customer_id" in plan.suggested_excluded_columns


def test_planner_merges_llm_and_rule_based_exclusions_and_normalizes_metric(tmp_path, monkeypatch) -> None:
    df = pd.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(30)],
            "signup_date": [f"2024-01-{(i % 28) + 1:02d}" for i in range(30)],
            "monthly_spend": [50 + i for i in range(30)],
            "churn": ["yes" if i % 3 == 0 else "no" for i in range(30)],
        }
    )
    settings = Settings(
        runs_dir=tmp_path,
        data_dir=tmp_path / "data",
        openai_api_key="test-key",
        openai_base_url=None,
        openai_model="test",
    )

    def _fake_openai_plan(*args, **kwargs) -> PlanSuggestion:
        return PlanSuggestion(
            planner_name="openai",
            user_brief="predict churn and optimize recall",
            suggested_targets=["churn"],
            suggested_task_type="classification",
            suggested_excluded_columns=["signup_date"],
            priority_metric="recall",
            notes=["Exclude signup_date and customer_id."],
            risk_flags=[],
        )

    monkeypatch.setattr("ml_platform.planner._generate_openai_plan", _fake_openai_plan)

    plan = suggest_plan(
        df=df,
        eda_summary=generate_eda_summary(df, target="churn"),
        settings=settings,
        user_brief="predict churn and optimize recall",
    )

    assert plan.priority_metric == "recall_weighted"
    assert "signup_date" in plan.suggested_excluded_columns
    assert "customer_id" in plan.suggested_excluded_columns
