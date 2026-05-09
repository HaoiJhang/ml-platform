import pandas as pd

from ml_platform.artifacts import FeatureEngineeringOperation
from ml_platform.automl import train_model
from ml_platform.cleaning import CleanConfig, clean_and_split
from ml_platform.evaluation import evaluate_model
from ml_platform.feature_engineering import FeatureEngineeringTransformer, validate_feature_engineering_plan


def test_feature_plan_validation_rejects_target_and_unknown_operations() -> None:
    df = pd.DataFrame(
        {
            "target": [0, 1, 0, 1],
            "amount": [10.0, 20.0, 30.0, 40.0],
            "segment": ["a", "b", "a", "c"],
        }
    )

    plan = validate_feature_engineering_plan(
        [
            FeatureEngineeringOperation(operation="frequency_encoding", source_column="target"),
            FeatureEngineeringOperation(operation="python_eval", source_column="amount"),
            FeatureEngineeringOperation(operation="numeric_binning", source_column="amount", bins=99),
            FeatureEngineeringOperation(operation="categorical_mapping", source_column="amount", mapping={"10": "low"}),
        ],
        df,
        target="target",
    )

    assert [operation.operation for operation in plan.operations] == ["numeric_binning"]
    assert plan.operations[0].bins == 10
    assert len(plan.rejected_operations) == 3


def test_frequency_encoding_is_fit_on_training_data_only() -> None:
    transformer = FeatureEngineeringTransformer(
        [FeatureEngineeringOperation(operation="frequency_encoding", source_column="segment")]
    )
    train = pd.DataFrame({"segment": ["a", "a", "b"]})
    test = pd.DataFrame({"segment": ["a", "c"]})

    transformed = transformer.fit(train).transform(test)

    assert transformed["fe__segment__freq"].tolist() == [2 / 3, 0.0]


def test_cleaning_pipeline_applies_whitelisted_feature_engineering() -> None:
    df = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1],
            "signup_date": pd.date_range("2024-01-01", periods=6, freq="D").astype(str),
            "amount": [10.0, 12.0, 50.0, 55.0, 90.0, 95.0],
            "visits": [1.0, 2.0, 5.0, 5.0, 9.0, 10.0],
            "segment": ["a", "a", "b", "b", "c", "c"],
        }
    )
    cleaned = clean_and_split(
        df,
        CleanConfig(
            target="target",
            task_type="classification",
            test_size=0.33,
            random_state=7,
            feature_engineering_operations=[
                FeatureEngineeringOperation(
                    operation="date_parts",
                    source_column="signup_date",
                    parts=["month", "dayofweek"],
                ),
                FeatureEngineeringOperation(operation="numeric_binning", source_column="amount", bins=3),
                FeatureEngineeringOperation(
                    operation="numeric_interaction",
                    columns=["amount", "visits"],
                    operator="ratio",
                ),
                FeatureEngineeringOperation(operation="frequency_encoding", source_column="segment"),
            ],
        ),
    )

    transformed = cleaned.preprocessor.fit_transform(cleaned.X_train)

    assert transformed.shape[0] == len(cleaned.X_train)
    assert any(
        step.get("step") == "build_preprocessor" and step.get("feature_engineering_operations")
        for step in cleaned.cleaning_log
    )


def test_training_pipeline_runs_with_feature_engineering_plan() -> None:
    df = pd.DataFrame(
        {
            "target": [0, 1] * 20,
            "signup_date": pd.date_range("2024-01-01", periods=40, freq="D").astype(str),
            "amount": [float(10 + i) for i in range(40)],
            "visits": [float(1 + (i % 7)) for i in range(40)],
            "segment": [f"s{i % 8}" for i in range(40)],
        }
    )
    cleaned = clean_and_split(
        df,
        CleanConfig(
            target="target",
            task_type="classification",
            test_size=0.25,
            random_state=11,
            feature_engineering_operations=[
                FeatureEngineeringOperation(
                    operation="date_parts",
                    source_column="signup_date",
                    parts=["month", "dayofweek"],
                ),
                FeatureEngineeringOperation(operation="frequency_encoding", source_column="segment"),
                FeatureEngineeringOperation(operation="numeric_binning", source_column="amount", bins=4),
                FeatureEngineeringOperation(
                    operation="numeric_interaction",
                    columns=["amount", "visits"],
                    operator="ratio",
                ),
            ],
        ),
    )

    trained = train_model(cleaned, time_budget=1)
    metrics, predictions = evaluate_model(trained.model, cleaned, "classification")

    assert metrics["accuracy"] is not None
    assert not predictions.empty
    assert trained.feature_importance
