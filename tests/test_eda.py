import pandas as pd

from ml_platform.eda import generate_eda_summary


def test_deeper_eda_includes_missingness_correlations_and_target_relationships() -> None:
    df = pd.DataFrame(
        {
            "num_a": [1, 2, 3, 4, 100],
            "num_b": [2, 4, 6, 8, 200],
            "cat": ["x", "x", "y", "y", "y"],
            "target": ["no", "no", "yes", "yes", "yes"],
            "missing_a": [None, None, 1, 1, 1],
            "missing_b": [None, None, 2, 2, 2],
        }
    )

    summary = generate_eda_summary(df, target="target")

    assert summary["columns"]["num_a"]["outlier_count_iqr"] == 1
    assert summary["missingness"]["rows_with_any_missing"] == 2
    assert summary["missingness"]["correlated_missing_pairs"]
    assert summary["correlations"]["top_numeric_pairs"][0]["left"] == "num_a"
    assert summary["target_relationships"]["numeric_features"]
    assert summary["target_relationships"]["categorical_features"]
