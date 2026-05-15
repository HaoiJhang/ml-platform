import pandas as pd

from ml_platform.eda import (
    build_distribution_overview,
    build_distribution_plot_data,
    build_target_compare_groups,
    choose_default_distribution_feature,
    generate_eda_summary,
)


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


def test_distribution_plot_data_uses_histogram_for_continuous_numeric() -> None:
    df = pd.DataFrame(
        {
            "feature": list(range(30)),
            "target": ["low"] * 15 + ["high"] * 15,
        }
    )

    plot_spec = build_distribution_plot_data(df, "feature", target_column="target")

    assert plot_spec["display_mode"] == "continuous_numeric"
    assert plot_spec["chart_type"] == "histogram"
    assert plot_spec["compare_enabled"] is True
    assert set(plot_spec["plot_data"]["target_group"]) == {"low", "high"}


def test_distribution_plot_data_uses_grouped_bar_for_discrete_numeric() -> None:
    df = pd.DataFrame(
        {
            "tickets": [0, 1, 1, 2, 2, 2],
            "target": ["no", "no", "yes", "yes", "no", "yes"],
        }
    )

    plot_spec = build_distribution_plot_data(df, "tickets", target_column="target")

    assert plot_spec["display_mode"] == "discrete_numeric"
    assert plot_spec["chart_type"] == "grouped_bar"
    assert sorted(plot_spec["plot_data"]["feature_value"].unique().tolist()) == [0, 1, 2]


def test_distribution_plot_data_keeps_top_categories_and_groups_rest_as_other() -> None:
    df = pd.DataFrame(
        {
            "category": [f"c{i}" for i in range(22)],
            "target": ["yes", "no"] * 11,
        }
    )

    plot_spec = build_distribution_plot_data(df, "category", target_column="target")
    values = set(plot_spec["plot_data"]["feature_value"].tolist())

    assert plot_spec["display_mode"] == "categorical"
    assert "Other" in values
    assert len(values) == 21


def test_distribution_plot_data_buckets_datetime_columns_by_day_or_month() -> None:
    short_df = pd.DataFrame(
        {
            "signup_at": pd.date_range("2024-01-01", periods=10, freq="D").astype(str),
            "target": ["yes", "no"] * 5,
        }
    )
    long_df = pd.DataFrame(
        {
            "signup_at": pd.date_range("2024-01-01", periods=180, freq="D").astype(str),
            "target": ["yes", "no"] * 90,
        }
    )

    short_plot = build_distribution_plot_data(short_df, "signup_at", target_column="target")
    long_plot = build_distribution_plot_data(long_df, "signup_at", target_column="target")

    assert short_plot["display_mode"] == "datetime_like"
    assert short_plot["datetime_bucket_unit"] == "day"
    assert long_plot["datetime_bucket_unit"] == "month"


def test_numeric_target_compare_groups_use_quartiles_or_fallback() -> None:
    stable_target = pd.Series(range(40))
    unstable_target = pd.Series([0] * 50 + list(range(11)))

    grouped, grouped_meta = build_target_compare_groups(stable_target)
    failed_grouped, failed_meta = build_target_compare_groups(unstable_target)

    assert grouped is not None
    assert grouped_meta["mode"] == "quartile"
    assert set(grouped.dropna().unique().tolist()) == {"Q1", "Q2", "Q3", "Q4"}
    assert failed_grouped is None
    assert failed_meta["reason"] == "unstable_quartiles"


def test_distribution_overview_and_default_feature_prefer_first_continuous_numeric() -> None:
    df = pd.DataFrame(
        {
            "segment": (["a", "b", "a", "c"] * 6) + ["a"],
            "score": list(range(25)),
            "visits": [0, 1, 1, 2] * 6 + [3],
        }
    )

    overview = build_distribution_overview(df, ["segment", "score", "visits"])

    assert overview["display_mode"].tolist() == [
        "categorical",
        "continuous_numeric",
        "discrete_numeric",
    ]
    assert choose_default_distribution_feature(df, ["segment", "score", "visits"]) == "score"
