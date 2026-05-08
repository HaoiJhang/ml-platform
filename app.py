from __future__ import annotations

import base64
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
from ml_platform.config import Settings, load_settings
from ml_platform.data_io import read_csv
from ml_platform.eda import generate_eda_summary
from ml_platform.evaluation import evaluate_model
from ml_platform.llm_report import generate_report
from ml_platform.storage import RunStorage


st.set_page_config(page_title="ML Platform", layout="wide")

HERO_IMAGE_PATH = Path("/Users/haoyi/Pictures/彩虹.jpg")


def _load_hero_background() -> str:
    if not HERO_IMAGE_PATH.exists():
        return (
            "linear-gradient(100deg, rgba(40, 64, 88, 0.98) 0 38%, "
            "rgba(39, 70, 101, 0.86) 38% 62%, rgba(42, 53, 67, 0.92) 62%)"
        )
    encoded = base64.b64encode(HERO_IMAGE_PATH.read_bytes()).decode("ascii")
    return (
        "linear-gradient(100deg, rgba(18, 31, 43, 0.78) 0 36%, "
        "rgba(18, 31, 43, 0.54) 36% 68%, rgba(18, 31, 43, 0.34) 68%), "
        f'url("data:image/jpeg;base64,{encoded}")'
    )


def _apply_design_system() -> None:
    hero_background = _load_hero_background()
    st.markdown(
        """
        <style>
            @import url("https://fonts.googleapis.com/css2?family=Exo:wght@400;500;600;700;800;900&display=swap");

            :root {
                --lab-ink: #171717;
                --lab-muted: #5f6b7a;
                --lab-faint: #8994a3;
                --lab-panel: #ffffff;
                --lab-panel-strong: #f4f6f8;
                --lab-line: #dfe4eb;
                --lab-accent: #5c6672;
                --lab-accent-soft: #eef1f4;
                --lab-cyan: #5c6672;
                --lab-warn: #d86b35;
                --lab-bg: #f2f5f8;
            }

            html,
            body,
            .stApp,
            .stApp *,
            [class^="st-"],
            [class*=" st-"],
            [data-testid],
            [data-testid] *,
            button,
            input,
            textarea,
            select {
                font-family: Exo, "Helvetica Neue", Helvetica, Arial, sans-serif !important;
            }

            .stApp {
                color: var(--lab-ink);
                background:
                    linear-gradient(180deg, #ffffff 0, #f7f8fb 34rem, var(--lab-bg) 100%),
                    var(--lab-bg);
                font-family: Exo, "Helvetica Neue", Helvetica, Arial, sans-serif;
            }

            .stApp::before {
                content: "";
                position: fixed;
                inset: 0;
                pointer-events: none;
                background-image:
                    linear-gradient(120deg, transparent 0 64%, rgba(47, 143, 199, 0.045) 64% 66%, transparent 66%),
                    linear-gradient(90deg, rgba(23, 23, 23, 0.025) 1px, transparent 1px);
                background-size: 280px 140px, 72px 44px;
                mask-image: linear-gradient(to bottom, black, transparent 56%);
            }

            .block-container {
                max-width: 1200px;
                padding-top: 1rem;
                padding-bottom: 4rem;
            }

            [data-testid="stHeader"] {
                display: none;
            }

            [data-testid="stToolbar"],
            [data-testid="stDecoration"],
            [data-testid="stStatusWidget"],
            .stDeployButton {
                display: none !important;
            }

            [data-testid="stSidebar"] {
                background: #f7f8fa;
                border-right: 1px solid var(--lab-line);
            }

            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3 {
                color: var(--lab-ink);
                font-family: Exo, "Helvetica Neue", Helvetica, Arial, sans-serif;
                font-weight: 800;
                letter-spacing: 0;
            }

            h1, h2, h3 {
                color: var(--lab-ink);
                font-family: Exo, "Helvetica Neue", Helvetica, Arial, sans-serif;
                font-weight: 800;
                letter-spacing: 0;
            }

            h1 {
                font-size: clamp(2.45rem, 5vw, 5.6rem) !important;
                line-height: 0.98 !important;
                max-width: 780px;
            }

            h2 {
                margin-top: 1.55rem;
                padding-top: 1rem;
                border-top: 1px solid var(--lab-line);
            }

            h3 {
                font-size: 1.45rem !important;
            }

            p, li, label, .stMarkdown, [data-testid="stCaptionContainer"] {
                color: var(--lab-muted);
                font-family: Exo, "Helvetica Neue", Helvetica, Arial, sans-serif;
            }

            .lab-hero {
                position: relative;
                overflow: hidden;
                min-height: 325px;
                padding: 2.7rem 2.9rem 2.25rem;
                margin-bottom: 1.55rem;
                border: 1px solid #cfd6df;
                background: __HERO_BACKGROUND__;
                background-position: center;
                background-size: cover;
                border-radius: 0;
                box-shadow: 0 10px 24px rgba(21, 37, 54, 0.14);
                animation: labRise 500ms ease-out both;
            }

            .lab-hero::after {
                content: "";
                position: absolute;
                right: 9%;
                top: 0;
                z-index: 0;
                width: 24rem;
                height: 100%;
                border: 0;
                border-radius: 0;
                transform: skewX(-11deg);
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.01));
            }

            .lab-kicker {
                display: inline-flex;
                align-items: center;
                gap: 0.55rem;
                margin-bottom: 1rem;
                color: #ffffff;
                font-size: 0.9rem;
                font-weight: 800;
                letter-spacing: 0.02em;
                text-transform: none;
                font-family: Exo, "Helvetica Neue", Helvetica, Arial, sans-serif;
            }

            .lab-kicker::before {
                content: none;
            }

            .lab-hero h1 {
                position: relative;
                z-index: 2;
                margin: 0 0 1rem;
                color: #ffffff;
                text-shadow: 0 2px 14px rgba(0, 0, 0, 0.18);
            }

            .lab-hero p {
                position: relative;
                z-index: 2;
                max-width: 700px;
                margin: 0;
                color: #edf4fb;
                font-size: 1.08rem;
                line-height: 1.62;
                font-weight: 400;
            }

            .lab-rail {
                display: flex;
                position: relative;
                z-index: 2;
                flex-wrap: wrap;
                gap: 0.65rem;
                margin-top: 1.6rem;
            }

            .lab-pill {
                border: 1px solid var(--lab-line);
                color: #ffffff;
                background: rgba(255, 255, 255, 0.18);
                border-radius: 7px;
                padding: 0.5rem 0.75rem;
                font-size: 0.76rem;
                font-family: Exo, "Helvetica Neue", Helvetica, Arial, sans-serif;
                text-transform: none;
                letter-spacing: 0;
                font-weight: 700;
                backdrop-filter: blur(4px);
            }

            .lab-empty {
                padding: 1.35rem 1.5rem;
                border: 1px dashed #c7ced8;
                background: #ffffff;
                color: #4d5562;
                border-radius: 9px;
                font-family: Exo, "Helvetica Neue", Helvetica, Arial, sans-serif;
            }

            .lab-caption {
                margin: -0.35rem 0 0.75rem;
                color: var(--lab-muted);
            }

            [data-testid="stMetric"] {
                padding: 1rem 1.1rem;
                border: 1px solid var(--lab-line);
                background: var(--lab-panel);
                border-radius: 9px;
                box-shadow: 0 6px 18px rgba(21, 37, 54, 0.06);
            }

            [data-testid="stMetricLabel"] {
                color: var(--lab-muted);
                font-family: Exo, "Helvetica Neue", Helvetica, Arial, sans-serif;
            }

            [data-testid="stMetricValue"] {
                color: var(--lab-accent);
                font-family: Exo, "Helvetica Neue", Helvetica, Arial, sans-serif;
                font-weight: 800;
            }

            [data-testid="stFileUploader"] section,
            [data-testid="stDataFrame"],
            [data-testid="stJson"],
            [data-testid="stExpander"],
            div[data-testid="stTabs"] [role="tabpanel"] {
                border: 1px solid var(--lab-line);
                background: var(--lab-panel);
                border-radius: 9px;
                box-shadow: 0 6px 18px rgba(21, 37, 54, 0.05);
            }

            [data-testid="stFileUploader"] section {
                padding: 1.1rem;
                border-radius: 9px;
            }

            [data-testid="stFileUploader"] section * {
                color: #4d5562 !important;
            }

            [data-testid="stFileUploader"] section svg {
                color: #5c6672 !important;
            }

            [data-testid="stFileUploader"] button {
                color: #ffffff !important;
                background: #5c6672 !important;
                border: 1px solid #5c6672 !important;
                border-radius: 6px !important;
            }

            input,
            textarea,
            [data-baseweb="select"] > div {
                color: var(--lab-ink) !important;
                background: #ffffff !important;
                border-color: #c7ced8 !important;
            }

            button[kind="primary"],
            .stDownloadButton button {
                border: 1px solid #b9481c !important;
                background: #5c6672 !important;
                color: #ffffff !important;
                font-weight: 800 !important;
                box-shadow: 0 8px 18px rgba(21, 37, 54, 0.14);
                transition: transform 160ms ease, box-shadow 160ms ease;
            }

            button[kind="primary"]:hover,
            .stDownloadButton button:hover {
                transform: translateY(-1px);
                box-shadow: 0 12px 24px rgba(21, 37, 54, 0.2);
            }

            button,
            input,
            textarea,
            [data-baseweb="select"] > div {
                border-radius: 6px !important;
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 0.35rem;
                border-bottom: 1px solid var(--lab-line);
            }

            .stTabs [data-baseweb="tab"] {
                color: var(--lab-muted);
                background: #f8f9fb;
                border: 1px solid var(--lab-line);
                border-radius: 6px;
                font-weight: 700;
            }

            .stTabs [aria-selected="true"] {
                color: #ffffff;
                background: var(--lab-accent);
                border-color: var(--lab-accent);
            }

            [data-testid="stAlert"] {
                border-radius: 9px;
                border: 1px solid #b9edf5;
                background: var(--lab-accent-soft);
            }

            @keyframes labRise {
                from { opacity: 0; transform: translateY(14px); }
                to { opacity: 1; transform: translateY(0); }
            }

            @media (max-width: 720px) {
                .block-container {
                    padding-left: 1rem;
                    padding-right: 1rem;
                }

                .lab-hero {
                    min-height: 260px;
                    padding: 1.5rem;
                }

                .lab-hero::after {
                    opacity: 0.42;
                }
            }
        </style>
        """.replace("__HERO_BACKGROUND__", hero_background),
        unsafe_allow_html=True,
    )


def _render_hero() -> None:
    st.markdown(
        """
        <section class="lab-hero">
            <h1>ML Platform</h1>
            <p>
                Upload a tabular dataset, inspect data quality, train a local baseline,
                and export the artifacts from one compact experiment surface.
            </p>
            <div class="lab-rail">
                <span class="lab-pill">Dates</span>
                <span class="lab-pill">Submit</span>
                <span class="lab-pill">Experiments</span>
                <span class="lab-pill">Artifacts</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _section_caption(text: str) -> None:
    st.markdown(f'<p class="lab-caption">{text}</p>', unsafe_allow_html=True)


def _configure_llm_settings(settings: Settings) -> Settings:
    with st.sidebar:
        st.header("Report engine")
        st.caption("Optional LLM configuration. Leave empty to use the local rule-based report.")
        api_key = st.text_input(
            "API key",
            value="",
            type="password",
            placeholder="Uses OPENAI_API_KEY if empty",
        )
        base_url = st.text_input(
            "Base URL",
            value=settings.openai_base_url or "",
            placeholder="OpenAI default or compatible API URL",
        )
        model = st.text_input("Model", value=settings.openai_model)

    return Settings(
        runs_dir=settings.runs_dir,
        data_dir=settings.data_dir,
        openai_api_key=api_key or settings.openai_api_key,
        openai_base_url=base_url or None,
        openai_model=model or settings.openai_model,
    )


def _infer_task_type(df: pd.DataFrame, target: str) -> str:
    series = df[target].dropna()
    if not pd.api.types.is_numeric_dtype(series):
        return "classification"
    unique_count = int(series.nunique())
    if unique_count <= max(20, int(len(series) * 0.05)):
        return "classification"
    return "regression"


def main() -> None:
    _apply_design_system()
    settings = _configure_llm_settings(load_settings())

    _render_hero()

    st.subheader("Dataset intake")
    _section_caption("Start with one local CSV. The app keeps the experiment artifacts under the configured runs directory.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded_file is None:
        st.markdown(
            """
            <div class="lab-empty">
                Drop a CSV here to unlock schema inspection, missingness checks,
                training controls, and exportable run artifacts.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    df = read_csv(uploaded_file)
    columns = list(df.columns)

    st.subheader("Experiment setup")
    _section_caption("Choose the prediction target and tune the small number of parameters that affect the local run.")
    setup_cols = st.columns([1.2, 0.85, 1.15])
    with setup_cols[0]:
        target = st.selectbox("Target variable", columns, index=len(columns) - 1)
    with setup_cols[1]:
        default_task = _infer_task_type(df, target)
        task_type = st.radio(
            "Task type",
            ["classification", "regression"],
            index=0 if default_task == "classification" else 1,
            horizontal=True,
        )
    with setup_cols[2]:
        time_budget = st.number_input("Training time budget seconds", value=30, min_value=5, step=5)

    config_cols = st.columns(3)
    with config_cols[0]:
        test_size = st.slider("Test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    with config_cols[1]:
        high_missing_threshold = st.slider(
            "Drop feature when missing rate is above",
            min_value=0.5,
            max_value=1.0,
            value=0.9,
            step=0.05,
        )
    with config_cols[2]:
        random_state = st.number_input("Random state", value=42, step=1)

    st.subheader("Data preview")
    _section_caption("First 50 rows are shown for quick sanity checks before training.")
    st.dataframe(df.head(50), use_container_width=True)

    eda_summary = generate_eda_summary(df, target=target)
    st.subheader("EDA summary")
    _section_caption("A compact quality audit for shape, duplicates, missingness, correlations, and target behavior.")
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Rows", eda_summary["shape"]["rows"])
    with metric_cols[1]:
        st.metric("Columns", eda_summary["shape"]["columns"])
    with metric_cols[2]:
        st.metric("Duplicate rows", eda_summary["duplicate_rows"])
    with metric_cols[3]:
        st.metric("Rows with missing", eda_summary["missingness"]["rows_with_any_missing"])

    st.write("Column profile")
    st.dataframe(pd.DataFrame(eda_summary["columns"]).T, use_container_width=True)

    if target in df.columns and eda_summary.get("target"):
        st.write("Target profile")
        st.json(eda_summary["target"], expanded=False)

    eda_tabs = st.tabs(["Missingness", "Correlations", "Target relationships", "Quality warnings"])
    with eda_tabs[0]:
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

    st.subheader("Training run")
    _section_caption("Launch the local pipeline after reviewing the setup and data audit.")
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
        report_path = storage.save_text(run, "report.md", report)
        prediction_path = storage.save_predictions(run, prediction_sample)
        model_path = storage.save_model(run, trained.model)
        storage.record_run(run, metrics=metrics, status="completed")

    st.success(f"Run completed: {run.run_id}")
    st.subheader("Artifacts")
    _section_caption("Export the model, generated analysis report, and prediction sample for downstream review.")
    download_cols = st.columns(3)
    with download_cols[0]:
        st.download_button(
            "Download model",
            data=model_path.read_bytes(),
            file_name=f"{run.run_id}_model.joblib",
            mime="application/octet-stream",
        )
    with download_cols[1]:
        st.download_button(
            "Download report",
            data=report_path.read_text(encoding="utf-8"),
            file_name=f"{run.run_id}_report.md",
            mime="text/markdown",
        )
    with download_cols[2]:
        st.download_button(
            "Download predictions",
            data=prediction_path.read_text(encoding="utf-8"),
            file_name=f"{run.run_id}_prediction_sample.csv",
            mime="text/csv",
        )

    st.subheader("Run results")
    _section_caption("Metrics, feature importance, and the generated report are shown below for immediate review.")
    st.write("Metrics")
    st.json(metrics)
    st.write("Feature importance")
    st.dataframe(pd.DataFrame(trained.feature_importance), use_container_width=True)
    st.write("Analysis report")
    st.markdown(report)
    st.caption(f"Artifacts saved to {Path(run.path).resolve()}")


if __name__ == "__main__":
    main()
