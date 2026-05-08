import pandas as pd

from ml_platform.cleaning import CleanConfig, clean_and_split


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
