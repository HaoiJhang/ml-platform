import pandas as pd

from ml_platform.artifacts import PreprocessingPlan, PreprocessingStep, artifact_to_dict
from ml_platform.cleaning import CleanConfig, clean_and_split, prepare_for_training, preprocessing_plan_to_clean_config
from ml_platform.data_flow import DataFlowTracker


def test_cleaning_preserves_target_and_drops_unusable_features() -> None:
    df = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, None],
            "num": [1.0, None, 3.0, 4.0, 5.0],
            "cat": ["a", "b", None, "b", "a"],
            "constant": [1, 1, 1, 1, 1],
            "mostly_missing": [None, None, None, None, 1],
        }
    )

    cleaned = clean_and_split(
        df,
        CleanConfig(
            target="target",
            task_type="classification",
            test_size=0.5,
            high_missing_threshold=0.75,
        ),
    )

    assert "target" not in cleaned.feature_columns
    assert "constant" not in cleaned.feature_columns
    assert "mostly_missing" not in cleaned.feature_columns
    assert set(cleaned.y_train.dropna().unique()).issubset({0.0, 1.0})
    assert cleaned.numeric_features == ["num"]
    assert cleaned.categorical_features == ["cat"]


def test_cleaning_handles_all_categorical_features() -> None:
    df = pd.DataFrame(
        {
            "target": ["yes", "no", "yes", "no", "yes", "no"],
            "plan": ["basic", "pro", "basic", "team", "pro", "team"],
            "region": ["east", "west", "east", "north", "west", "north"],
        }
    )

    cleaned = clean_and_split(
        df,
        CleanConfig(target="target", task_type="classification", test_size=0.33, random_state=3),
    )
    prepared = prepare_for_training(cleaned)

    assert prepared.X_train_prepared is not None
    assert prepared.X_train_prepared.shape[0] == len(cleaned.X_train)
    assert list(prepared.X_train_prepared.columns) == ["plan", "region"]
    assert cleaned.numeric_features == []
    assert cleaned.categorical_features == ["plan", "region"]


def test_cleaning_supports_ordinal_categorical_encoding() -> None:
    df = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1],
            "segment": ["a", "b", "a", "c", "b", "c"],
            "region": ["east", "west", "east", "north", "west", "north"],
        }
    )

    cleaned = clean_and_split(
        df,
        CleanConfig(
            target="target",
            task_type="classification",
            test_size=0.33,
            random_state=3,
            categorical_encoding_strategy="ordinal",
        ),
    )
    prepared = prepare_for_training(cleaned)

    assert prepared.X_train_prepared is not None
    assert prepared.X_train_prepared.shape[0] == len(prepared.X_train)
    assert prepared.X_train_prepared.shape[1] == len(prepared.categorical_features)
    assert prepared.prepared_feature_names == ["segment", "region"]


def test_cleaning_supports_frequency_categorical_encoding() -> None:
    df = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1],
            "segment": ["a", "b", "a", "c", "b", "c"],
            "region": ["east", "west", "east", "north", "west", "north"],
        }
    )

    cleaned = clean_and_split(
        df,
        CleanConfig(
            target="target",
            task_type="classification",
            test_size=0.33,
            random_state=3,
            categorical_encoding_strategy="frequency",
        ),
    )
    prepared = prepare_for_training(cleaned)

    assert prepared.X_train_prepared is not None
    assert prepared.X_train_prepared.shape[0] == len(prepared.X_train)
    assert prepared.X_train_prepared.shape[1] == len(prepared.categorical_features)
    assert prepared.prepared_feature_names == ["segment", "region"]


def test_cleaning_handles_all_numeric_features() -> None:
    df = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1],
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "visits": [1.0, 2.0, 1.0, 3.0, 5.0, 8.0],
        }
    )

    cleaned = clean_and_split(
        df,
        CleanConfig(target="target", task_type="classification", test_size=0.33, random_state=3),
    )
    prepared = prepare_for_training(cleaned)

    assert prepared.X_train_prepared is not None
    assert prepared.X_train_prepared.shape[0] == len(cleaned.X_train)
    assert list(prepared.X_train_prepared.columns) == ["amount", "visits"]
    assert cleaned.numeric_features == ["amount", "visits"]
    assert cleaned.categorical_features == []


def test_cleaning_tracker_captures_row_and_column_changes() -> None:
    df = pd.DataFrame(
        {
            "target": [1, 0, 1, None, 0],
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0],
            "constant": [1, 1, 1, 1, 1],
            "mostly_missing": [None, None, None, None, 99.0],
            "segment": ["a", "b", "a", "c", "b"],
        }
    )
    tracker = DataFlowTracker(target="target")

    cleaned = clean_and_split(
        df,
        CleanConfig(
            target="target",
            task_type="classification",
            test_size=0.5,
            random_state=7,
            high_missing_threshold=0.7,
        ),
        tracker=tracker,
    )
    payload = artifact_to_dict(tracker.to_trace())
    snapshots = {snapshot["step"]: snapshot for snapshot in payload["snapshots"]}

    assert snapshots["after_target_drop"]["rows_delta"] == -1
    assert "mostly_missing" in snapshots["after_high_missing_drop"]["columns_removed"]
    assert "constant" in snapshots["after_constant_drop"]["columns_removed"]
    assert snapshots["train_split"]["rows"] + snapshots["test_split"]["rows"] == snapshots["after_constant_drop"]["rows"]
    assert snapshots["preprocessor_plan"]["metadata"]["numeric_features"] == ["amount"]
    assert cleaned.feature_columns == ["amount", "segment"]


def test_cleaning_uses_user_selected_imputation_strategies() -> None:
    df = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1],
            "amount": [10.0, None, 30.0, None, 50.0, 60.0],
            "segment": ["a", None, "b", "b", None, "c"],
        }
    )

    cleaned = clean_and_split(
        df,
        CleanConfig(
            target="target",
            task_type="classification",
            test_size=0.33,
            random_state=3,
            numeric_imputation_strategy="mean",
            categorical_imputation_strategy="constant_missing",
            standardize_numeric=False,
        ),
    )

    prepared = prepare_for_training(cleaned)

    assert prepared.X_train_prepared is not None
    assert prepared.X_train_prepared.shape[0] == len(cleaned.X_train)
    assert prepared.X_train_prepared.isna().any().any()
    assert any(
        step.get("step") == "build_preprocessor"
        and step.get("trainer_preprocessing") == "autogluon"
        for step in cleaned.cleaning_log
    )


def test_prepare_for_training_materializes_preprocessed_matrices() -> None:
    df = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1],
            "amount": [10.0, None, 30.0, None, 50.0, 60.0],
            "segment": ["a", None, "b", "b", None, "c"],
        }
    )

    cleaned = clean_and_split(
        df,
        CleanConfig(target="target", task_type="classification", test_size=0.33, random_state=3),
    )
    prepared = prepare_for_training(cleaned)

    assert prepared.fitted_preprocessor is None
    assert prepared.X_train_prepared is not None
    assert prepared.X_test_prepared is not None
    assert prepared.X_train_prepared.shape[0] == len(prepared.X_train)
    assert prepared.X_test_prepared.shape[0] == len(prepared.X_test)
    assert prepared.prepared_feature_names == ["amount", "segment"]
    assert any(step.get("step") == "prepare_training_data" for step in prepared.cleaning_log)


def test_preprocessing_plan_maps_to_clean_config() -> None:
    plan = PreprocessingPlan(
        global_params={"test_size": 0.3, "random_state": 9},
        applied_step_ids=[
            "missing_value",
            "categorical_encoding",
            "numeric_scaling",
            "feature_engineering",
        ],
        steps=[
            PreprocessingStep(
                id="missing_value",
                kind="missing_value",
                params={
                    "high_missing_threshold": 0.75,
                    "numeric_imputation_strategy": "mean",
                    "categorical_imputation_strategy": "constant_missing",
                },
            ),
            PreprocessingStep(
                id="categorical_encoding",
                kind="categorical_encoding",
                params={"strategy": "one_hot"},
            ),
            PreprocessingStep(
                id="numeric_scaling",
                kind="numeric_scaling",
                params={"standardize_numeric": False},
            ),
            PreprocessingStep(
                id="feature_engineering",
                kind="feature_engineering",
                params={
                    "operations": [
                        {
                            "operation": "frequency_encoding",
                            "source_column": "segment",
                            "rationale": "Encode a high-cardinality feature.",
                        }
                    ]
                },
            ),
        ],
    )

    config = preprocessing_plan_to_clean_config(
        plan,
        target="target",
        task_type="classification",
    )

    assert config.test_size == 0.3
    assert config.random_state == 9
    assert config.high_missing_threshold == 0.75
    assert config.numeric_imputation_strategy == "mean"
    assert config.categorical_imputation_strategy == "constant_missing"
    assert config.categorical_encoding_strategy == "one_hot"
    assert config.standardize_numeric is False
    assert config.feature_engineering_operations is not None
    assert config.feature_engineering_operations[0].operation == "frequency_encoding"


def test_preprocessing_plan_preserves_selected_categorical_encoding_strategy() -> None:
    plan = PreprocessingPlan(
        applied_step_ids=["categorical_encoding"],
        steps=[
            PreprocessingStep(
                id="categorical_encoding",
                kind="categorical_encoding",
                params={"strategy": "frequency"},
            ),
        ],
    )

    config = preprocessing_plan_to_clean_config(
        plan,
        target="target",
        task_type="classification",
    )

    assert config.categorical_encoding_strategy == "frequency"
