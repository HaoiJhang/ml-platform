import pandas as pd

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
