from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import streamlit as st

from ml_platform.automl import train_model
from ml_platform.cleaning import CleanConfig, clean_and_split
from ml_platform.config import load_settings
from ml_platform.data_io import read_csv
from ml_platform.eda import generate_eda_summary
from ml_platform.evaluation import evaluate_model
from ml_platform.llm_report import generate_report
from ml_platform.storage import RunStorage


st.set_page_config(page_title="ML Platform Prototype", layout="wide")


def _infer_task_type(df: pd.DataFrame, target: str) -> str:
    series = df[target].dropna()
    if not pd.api.types.is_numeric_dtype(series):
        return "classification"
    unique_count = int(series.nunique())
    if unique_count <= max(20, int(len(series) * 0.05)):
        return "classification"
    return "regression"


def main() -> None:
    settings = load_settings()

    st.title("ML Platform Research Prototype")
    st.caption("Local CSV EDA, cleaning, AutoML training, evaluation, storage, and report generation.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV file to start a local experiment.")
        return

    df = read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head(50), use_container_width=True)

    columns = list(df.columns)
    target = st.selectbox("Target variable", columns, index=len(columns) - 1)
    default_task = _infer_task_type(df, target)
    task_type = st.radio(
        "Task type",
        ["classification", "regression"],
        index=0 if default_task == "classification" else 1,
        horizontal=True,
    )

    with st.expander("Cleaning options", expanded=False):
        test_size = st.slider("Test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        high_missing_threshold = st.slider(
            "Drop feature when missing rate is above",
            min_value=0.5,
            max_value=1.0,
            value=0.9,
            step=0.05,
        )
        random_state = st.number_input("Random state", value=42, step=1)
        time_budget = st.number_input("Training time budget seconds", value=30, min_value=5, step=5)

    eda_summary = generate_eda_summary(df, target=target)
    st.subheader("EDA Summary")
    left, right = st.columns(2)
    with left:
        st.metric("Rows", eda_summary["shape"]["rows"])
        st.metric("Columns", eda_summary["shape"]["columns"])
        st.metric("Duplicate rows", eda_summary["duplicate_rows"])
    with right:
        st.dataframe(pd.DataFrame(eda_summary["columns"]).T, use_container_width=True)

    if target in df.columns and eda_summary.get("target"):
        st.write("Target summary")
        st.json(eda_summary["target"], expanded=False)

    eda_tabs = st.tabs(["Missingness", "Correlations", "Target relationships", "Quality warnings"])
    with eda_tabs[0]:
        st.metric("Rows with any missing value", eda_summary["missingness"]["rows_with_any_missing"])
        top_missing = eda_summary["missingness"]["top_missing_columns"]
        if top_missing:
            st.dataframe(
                pd.DataFrame(
                    [{"column": column, "missing_rate": rate} for column, rate in top_missing.items()]
                ),
                use_container_width=True,
            )
        correlated_missing = eda_summary["missingness"]["correlated_missing_pairs"]
        if correlated_missing:
            st.write("Correlated missingness pairs")
            st.dataframe(pd.DataFrame(correlated_missing), use_container_width=True)
    with eda_tabs[1]:
        top_pairs = eda_summary["correlations"]["top_numeric_pairs"]
        target_corr = eda_summary["correlations"]["target_numeric_correlations"]
        if target_corr:
            st.write("Numeric correlations with target")
            st.dataframe(pd.DataFrame(target_corr), use_container_width=True)
        if top_pairs:
            st.write("Strong numeric feature correlations")
            st.dataframe(pd.DataFrame(top_pairs), use_container_width=True)
    with eda_tabs[2]:
        st.json(eda_summary.get("target_relationships", {}), expanded=False)
    with eda_tabs[3]:
        for warning in eda_summary["quality_warnings"]:
            st.warning(warning)

    if not st.button("Run training", type="primary"):
        return

    config = CleanConfig(
        target=target,
        task_type=task_type,
        test_size=float(test_size),
        random_state=int(random_state),
        high_missing_threshold=float(high_missing_threshold),
    )

    with st.spinner("Cleaning data and training model locally..."):
        cleaned = clean_and_split(df, config)
        trained = train_model(cleaned, time_budget=int(time_budget))
        metrics, prediction_sample = evaluate_model(trained.model, cleaned, task_type=task_type)

        report = generate_report(
            eda_summary=eda_summary,
            cleaning_log=cleaned.cleaning_log,
            metrics=metrics,
            feature_importance=trained.feature_importance,
            settings=settings,
        )

        storage = RunStorage(settings.runs_dir)
        run = storage.create_run(
            config={
                "target": target,
                "task_type": task_type,
                "test_size": test_size,
                "high_missing_threshold": high_missing_threshold,
                "random_state": random_state,
                "time_budget": time_budget,
                "trainer": trained.trainer_name,
            }
        )
        storage.save_json(run, "eda_summary.json", eda_summary)
        storage.save_json(run, "cleaning_log.json", cleaned.cleaning_log)
        storage.save_json(run, "metrics.json", metrics)
        storage.save_json(run, "feature_importance.json", trained.feature_importance)
        storage.save_text(run, "report.md", report)
        storage.save_predictions(run, prediction_sample)
        storage.save_model(run, trained.model)
        storage.record_run(run, metrics=metrics, status="completed")

    st.success(f"Run completed: {run.run_id}")
    st.subheader("Metrics")
    st.json(metrics)
    st.subheader("Feature Importance")
    st.dataframe(pd.DataFrame(trained.feature_importance), use_container_width=True)
    st.subheader("Analysis Report")
    st.markdown(report)
    st.caption(f"Artifacts saved to {Path(run.path).resolve()}")


if __name__ == "__main__":
    main()
