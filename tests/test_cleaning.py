import pandas as pd

from ml_platform.artifacts import artifact_to_dict
from ml_platform.cleaning import CleanConfig, clean_and_split
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
    transformed = cleaned.preprocessor.fit_transform(cleaned.X_train)

    assert transformed.shape[0] == len(cleaned.X_train)
    assert cleaned.numeric_features == []
    assert cleaned.categorical_features == ["plan", "region"]


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
    transformed = cleaned.preprocessor.fit_transform(cleaned.X_train)

    assert transformed.shape[0] == len(cleaned.X_train)
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
