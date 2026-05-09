from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import streamlit as st

from ml_platform.artifacts import artifact_to_dict
from ml_platform.automl import train_model
from ml_platform.cleaning import CleanConfig, clean_and_split
from ml_platform.config import Settings, load_settings
from ml_platform.data_io import read_csv
from ml_platform.eda import generate_eda_summary
from ml_platform.evaluation import evaluate_model
from ml_platform.llm_report import generate_report_result
from ml_platform.planner import suggest_plan
from ml_platform.storage import RunStorage
from ml_platform.validation import build_recommendations, resolve_priority_metric, validate_postrun, validate_preflight

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(PROJECT_ROOT / "app.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("ml_platform")

st.set_page_config(page_title="ML Platform", layout="wide")

HERO_IMAGE_PATH = Path("/Users/haoyi/Pictures/彩虹.jpg")
LOCAL_LLM_CONFIG_PATH = PROJECT_ROOT / ".ml_platform.local.json"


def _load_local_llm_config() -> dict[str, str]:
    if not LOCAL_LLM_CONFIG_PATH.exists():
        return {}
    try:
        raw_config = json.loads(LOCAL_LLM_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Unable to read local LLM config: %s", exc)
        return {}
    if not isinstance(raw_config, dict):
        return {}
    return {
        key: value
        for key, value in raw_config.items()
        if key in {"openai_api_key", "openai_base_url", "openai_model"} and isinstance(value, str)
    }


def _save_local_llm_config(config: dict[str, str]) -> None:
    try:
        LOCAL_LLM_CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
        os.chmod(LOCAL_LLM_CONFIG_PATH, 0o600)
    except OSError as exc:
        logger.warning("Unable to write local LLM config: %s", exc)


def _delete_local_llm_config() -> None:
    try:
        LOCAL_LLM_CONFIG_PATH.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Unable to delete local LLM config: %s", exc)


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
            .stApp {
                font-family: Exo, "Helvetica Neue", Helvetica, Arial, sans-serif !important;
            }

            button,
            input,
            textarea,
            select,
            [data-baseweb="select"] > div,
            [data-testid="stFileUploader"] section * {
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
                width: 2.2rem !important;
                height: 2.2rem !important;
            }

            /* Icon sizing for all interactive components */
            svg {
                display: inline-block;
                flex-shrink: 0;
            }

            [data-testid="stExpander"] svg {
                width: 1rem !important;
                height: 1rem !important;
            }

            [data-baseweb="select"] svg,
            [data-baseweb="menu"] svg {
                width: 1.1rem !important;
                height: 1.1rem !important;
            }

            .stAlert svg {
                width: 1.1rem !important;
                height: 1.1rem !important;
            }

            [data-testid="stMetric"] svg {
                width: 1.3rem !important;
                height: 1.3rem !important;
            }

            [data-testid="stSpinner"] svg {
                width: 1.5rem !important;
                height: 1.5rem !important;
            }

            button [data-testid="stBaseButton-icon"] svg,
            .stDownloadButton button svg {
                width: 1rem !important;
                height: 1rem !important;
            }

            .stCheckbox svg,
            .stRadio svg {
                width: 1.05rem !important;
                height: 1.05rem !important;
            }

            [data-testid="stStatusWidget"] svg {
                width: 1.4rem !important;
                height: 1.4rem !important;
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
                border: 1px solid #4e5863 !important;
                background: #4e5863 !important;
                color: #ffffff !important;
                font-weight: 800 !important;
                box-shadow: 0 8px 18px rgba(21, 37, 54, 0.14);
                transition: transform 160ms ease, box-shadow 160ms ease;
            }

            button[kind="primary"] *,
            .stDownloadButton button * {
                color: #ffffff !important;
            }

            button[kind="primary"]:disabled,
            button[kind="primary"][disabled] {
                border-color: #c3cad3 !important;
                background: #e3e7ec !important;
                color: #7b8592 !important;
                box-shadow: none;
                opacity: 1 !important;
            }

            button[kind="primary"]:disabled *,
            button[kind="primary"][disabled] * {
                color: #7b8592 !important;
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
                color: #ffffff !important;
                background: var(--lab-accent);
                border-color: var(--lab-accent);
            }

            .stTabs [aria-selected="true"] *,
            .stTabs [aria-selected="true"] p,
            .stTabs [aria-selected="true"] span {
                color: #ffffff !important;
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
    saved_config = _load_local_llm_config()
    saved_api_key = saved_config.get("openai_api_key", "")
    saved_base_url = saved_config.get("openai_base_url", "")
    saved_model = saved_config.get("openai_model", "")

    st.subheader("Report engine")
    _section_caption(
        "Optional LLM configuration for plan and report generation. Save the key locally to keep it after page reloads."
    )
    config_cols = st.columns(3)
    with config_cols[0]:
        api_key = st.text_input(
            "API key",
            value=saved_api_key,
            type="password",
            placeholder="Uses OPENAI_API_KEY if empty",
        )
    with config_cols[1]:
        base_url = st.text_input(
            "Base URL",
            value=saved_base_url or settings.openai_base_url or "",
            placeholder="OpenAI default or compatible API URL",
        )
    with config_cols[2]:
        model = st.text_input("Model", value=saved_model or settings.openai_model)

    remember_config = st.checkbox("Remember LLM settings on this device", value=bool(saved_api_key))
    if remember_config and api_key:
        _save_local_llm_config(
            {
                "openai_api_key": api_key,
                "openai_base_url": base_url,
                "openai_model": model,
            }
        )
    elif not remember_config and saved_config:
        _delete_local_llm_config()
        saved_config = {}

    if saved_config and st.button("Forget saved LLM settings"):
        _delete_local_llm_config()
        st.rerun()

    return Settings(
        runs_dir=settings.runs_dir,
        data_dir=settings.data_dir,
        openai_api_key=api_key or saved_api_key or settings.openai_api_key,
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


def _target_task_types(df: pd.DataFrame, targets: list[str], task_type_choice: str) -> dict[str, str]:
    if task_type_choice in {"classification", "regression"}:
        return {target: task_type_choice for target in targets}
    return {target: _infer_task_type(df, target) for target in targets}


def _initialize_experiment_state(columns: list[str]) -> None:
    signature = tuple(columns)
    if st.session_state.get("_dataset_signature") == signature:
        return
    st.session_state["_dataset_signature"] = signature
    st.session_state["planner_brief"] = ""
    st.session_state["target_columns"] = [columns[-1]]
    st.session_state["task_type_choice"] = "auto"
    st.session_state["time_budget"] = 30
    st.session_state["time_budget_text"] = "30"
    st.session_state["priority_metric_choice"] = "auto"
    st.session_state["excluded_columns"] = []
    st.session_state["test_size"] = 0.2
    st.session_state["high_missing_threshold"] = 0.9
    st.session_state["random_state"] = 42
    st.session_state["random_state_text"] = "42"


def _queue_plan_suggestion(plan_suggestion: dict[str, object], columns: list[str]) -> None:
    suggested_targets = [column for column in plan_suggestion.get("suggested_targets", []) if column in columns]
    suggested_task_type = plan_suggestion.get("suggested_task_type")
    pending_targets = suggested_targets or st.session_state.get("target_columns", [])
    current_targets = set(pending_targets)
    excluded = [
        column
        for column in plan_suggestion.get("suggested_excluded_columns", [])
        if column in columns and column not in current_targets
    ]
    metric = plan_suggestion.get("priority_metric")
    st.session_state["_pending_plan_suggestion"] = {
        "target_columns": suggested_targets,
        "task_type_choice": suggested_task_type,
        "excluded_columns": excluded,
        "priority_metric_choice": metric,
    }


def _consume_pending_plan_suggestion(columns: list[str]) -> None:
    pending = st.session_state.pop("_pending_plan_suggestion", None)
    if not pending:
        return

    pending_targets = [column for column in pending.get("target_columns", []) if column in columns]
    if pending_targets:
        st.session_state["target_columns"] = pending_targets

    suggested_task_type = pending.get("task_type_choice")
    if suggested_task_type in {"auto", "classification", "regression"}:
        st.session_state["task_type_choice"] = suggested_task_type

    excluded_columns = [
        column
        for column in pending.get("excluded_columns", [])
        if column in columns and column not in set(st.session_state.get("target_columns", []))
    ]
    st.session_state["excluded_columns"] = excluded_columns

    metric = pending.get("priority_metric_choice")
    if metric in {"auto", "accuracy", "f1_weighted", "precision_weighted", "recall_weighted", "roc_auc", "rmse", "mae", "r2"}:
        st.session_state["priority_metric_choice"] = metric


def _render_text_items(title: str, items: list[object], empty_text: str) -> None:
    st.write(title)
    clean_items = [str(item) for item in items if str(item).strip()]
    if not clean_items:
        st.caption(empty_text)
        return
    for item in clean_items:
        st.write(f"- {item}")


def _render_planner_suggestion(plan_data: dict[str, object]) -> None:
    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.metric("Planner", str(plan_data.get("planner_name") or "unknown"))
    with summary_cols[1]:
        st.metric("Task type", str(plan_data.get("suggested_task_type") or "auto"))
    with summary_cols[2]:
        st.metric("Priority metric", str(plan_data.get("priority_metric") or "auto"))

    target_items = [str(item) for item in plan_data.get("suggested_targets", []) if str(item).strip()]
    excluded_items = [str(item) for item in plan_data.get("suggested_excluded_columns", []) if str(item).strip()]
    detail_cols = st.columns(2)
    with detail_cols[0]:
        st.write("Suggested targets")
        if target_items:
            st.dataframe(pd.DataFrame({"target": target_items}), hide_index=True, use_container_width=True)
        else:
            st.caption("No target suggestion.")
    with detail_cols[1]:
        st.write("Suggested exclusions")
        if excluded_items:
            st.dataframe(pd.DataFrame({"column": excluded_items}), hide_index=True, use_container_width=True)
        else:
            st.caption("No excluded columns suggested.")

    notes_cols = st.columns(2)
    with notes_cols[0]:
        _render_text_items("Notes", list(plan_data.get("notes", [])), "No notes.")
    with notes_cols[1]:
        _render_text_items("Risk flags", list(plan_data.get("risk_flags", [])), "No risk flags.")

    with st.expander("Raw planner JSON", expanded=False):
        st.json(plan_data, expanded=True)


def _render_target_profile(target_data: dict[str, object]) -> None:
    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.metric("Target", str(target_data.get("name") or "-"))
    with summary_cols[1]:
        st.metric("Missing rows", int(target_data.get("missing_count") or 0))
    with summary_cols[2]:
        st.metric("Unique values", int(target_data.get("unique_count") or 0))

    stats = target_data.get("stats", {})
    if isinstance(stats, dict) and stats:
        stats_rows = [{"stat": key, "value": value} for key, value in stats.items()]
        st.dataframe(pd.DataFrame(stats_rows), hide_index=True, use_container_width=True)

    top_values = target_data.get("top_values", {})
    if isinstance(top_values, dict) and top_values:
        st.write("Top target values")
        st.dataframe(
            pd.DataFrame([{"value": key, "count": value} for key, value in top_values.items()]),
            hide_index=True,
            use_container_width=True,
        )


def _flatten_target_relationships(relationships: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric_correlation_rows: list[dict[str, object]] = []
    numeric_group_rows: list[dict[str, object]] = []
    categorical_rows: list[dict[str, object]] = []

    for item in relationships.get("numeric_features", []):
        if not isinstance(item, dict):
            continue
        feature = str(item.get("feature") or "")
        if "target_correlation" in item:
            numeric_correlation_rows.append(
                {"feature": feature, "target_correlation": item.get("target_correlation")}
            )
        by_target = item.get("by_target", {})
        if isinstance(by_target, dict):
            for target_value, stats in by_target.items():
                if not isinstance(stats, dict):
                    continue
                numeric_group_rows.append(
                    {
                        "feature": feature,
                        "target_value": str(target_value),
                        "mean": stats.get("mean"),
                        "median": stats.get("median"),
                        "count": stats.get("count"),
                    }
                )

    for item in relationships.get("categorical_features", []):
        if not isinstance(item, dict):
            continue
        feature = str(item.get("feature") or "")
        distribution = item.get("target_distribution_by_value", {})
        if not isinstance(distribution, dict):
            continue
        for feature_value, target_rates in distribution.items():
            if not isinstance(target_rates, dict):
                continue
            for target_value, rate in target_rates.items():
                categorical_rows.append(
                    {
                        "feature": feature,
                        "feature_value": str(feature_value),
                        "target_value": str(target_value),
                        "share": rate,
                    }
                )

    return (
        pd.DataFrame(numeric_correlation_rows),
        pd.DataFrame(numeric_group_rows),
        pd.DataFrame(categorical_rows),
    )


def _render_target_relationships(relationships: dict[str, object]) -> None:
    numeric_correlations, numeric_groups, categorical_distribution = _flatten_target_relationships(relationships)

    if not numeric_correlations.empty:
        st.write("Numeric feature relationships")
        st.dataframe(numeric_correlations, hide_index=True, use_container_width=True)

    if not numeric_groups.empty:
        st.write("Numeric feature distribution by target")
        st.dataframe(numeric_groups, hide_index=True, use_container_width=True)

    if not categorical_distribution.empty:
        st.write("Categorical feature target distribution")
        st.dataframe(categorical_distribution, hide_index=True, use_container_width=True)

    if numeric_correlations.empty and numeric_groups.empty and categorical_distribution.empty:
        st.caption("No target relationship summary available.")


def _render_metrics(metrics: dict[str, object], priority_metric: str | None = None) -> None:
    test_rows: list[dict[str, object]] = []
    train_rows: list[dict[str, object]] = []
    for metric_name, value in metrics.items():
        phase = "train" if str(metric_name).startswith("train_") else "test"
        label = str(metric_name).removeprefix("train_")
        row = {"metric": label, "value": value}
        if phase == "train":
            train_rows.append(row)
        else:
            test_rows.append(row)

    if priority_metric:
        resolved_value = metrics.get(priority_metric)
        priority_cols = st.columns(2)
        with priority_cols[0]:
            st.metric("Priority metric", priority_metric)
        with priority_cols[1]:
            st.metric("Priority value", "-" if resolved_value is None else f"{float(resolved_value):.4f}")

    metric_cols = st.columns(2)
    with metric_cols[0]:
        st.write("Test metrics")
        if test_rows:
            st.dataframe(pd.DataFrame(test_rows), hide_index=True, use_container_width=True)
        else:
            st.caption("No test metrics.")
    with metric_cols[1]:
        st.write("Train metrics")
        if train_rows:
            st.dataframe(pd.DataFrame(train_rows), hide_index=True, use_container_width=True)
        else:
            st.caption("No train metrics.")


def _render_validation_summary(
    preflight: dict[str, object],
    postrun: dict[str, object],
    recommendations: dict[str, object],
    priority_metric: str,
) -> None:
    summary_cols = st.columns(4)
    with summary_cols[0]:
        st.metric("Requested metric", priority_metric)
    with summary_cols[1]:
        st.metric("Preflight", "OK" if preflight.get("ok_to_run") else "Blocked")
    with summary_cols[2]:
        st.metric("Postrun", "OK" if postrun.get("ok") else "Check issues")
    with summary_cols[3]:
        st.metric("Trainer", str(postrun.get("trainer_name") or "-"))

    preflight_detail_cols = st.columns(3)
    with preflight_detail_cols[0]:
        st.metric("Feature count", int(preflight.get("feature_count") or 0))
    with preflight_detail_cols[1]:
        st.metric("Dropped target rows", int(preflight.get("dropped_target_rows") or 0))
    with preflight_detail_cols[2]:
        st.metric("Report mode", str(postrun.get("report_mode") or "-"))

    generalization_gap = postrun.get("generalization_gap", {})
    if isinstance(generalization_gap, dict) and generalization_gap:
        gap_rows = [{"metric": key, "value": value} for key, value in generalization_gap.items()]
        st.write("Generalization gap")
        st.dataframe(pd.DataFrame(gap_rows), hide_index=True, use_container_width=True)

    class_balance = preflight.get("class_balance", {})
    if isinstance(class_balance, dict) and class_balance:
        st.write("Class balance")
        st.dataframe(
            pd.DataFrame([{"class": key, "share": value} for key, value in class_balance.items()]),
            hide_index=True,
            use_container_width=True,
        )

    recommended_exclusions = preflight.get("recommended_excluded_columns", [])
    detected_leakage = preflight.get("detected_leakage_columns", [])
    detail_cols = st.columns(2)
    with detail_cols[0]:
        _render_text_items("Recommended exclusions", list(recommended_exclusions), "No extra exclusions suggested.")
    with detail_cols[1]:
        _render_text_items("Potential leakage columns", list(detected_leakage), "No leakage columns detected.")

    issue_cols = st.columns(2)
    with issue_cols[0]:
        st.write("Preflight issues")
        _render_issue_table(list(preflight.get("issues", [])))
    with issue_cols[1]:
        st.write("Postrun issues")
        _render_issue_table(list(postrun.get("issues", [])))

    recommendation_cols = st.columns(2)
    with recommendation_cols[0]:
        _render_text_items("Recommendation summary", list(recommendations.get("summary", [])), "No summary available.")
    with recommendation_cols[1]:
        _render_text_items("Next steps", list(recommendations.get("next_steps", [])), "No next steps available.")


def _render_issue_table(issues: list[dict[str, object]]) -> None:
    if not issues:
        st.caption("No issues surfaced.")
        return
    st.dataframe(pd.DataFrame(issues), use_container_width=True)


def _render_integer_input(label: str, state_key: str, min_value: int | None = None) -> int:
    text_key = f"{state_key}_text"
    raw_value = st.text_input(label, key=text_key)
    current_value = int(st.session_state.get(state_key, 0))
    candidate = raw_value.strip()
    try:
        parsed_value = int(candidate)
    except ValueError:
        st.caption(f"Enter a whole number. Using {current_value} until corrected.")
        return current_value

    if min_value is not None and parsed_value < min_value:
        st.caption(f"Enter a value greater than or equal to {min_value}. Using {current_value} until corrected.")
        return current_value

    st.session_state[state_key] = parsed_value
    return parsed_value


def main() -> None:
    _apply_design_system()
    _render_hero()
    settings = _configure_llm_settings(load_settings())

    st.subheader("Dataset intake")
    _section_caption("Start with one local CSV. The app keeps the experiment artifacts under the configured runs directory.")

    demo_csvs = sorted(Path(p).name for p in PROJECT_ROOT.glob("data/*.csv") if p.is_file())

    data_source = st.radio(
        "Data source",
        ["Upload CSV", *([f"Demo: {name}" for name in demo_csvs] if demo_csvs else [])],
        horizontal=True,
        index=0,
    )

    df: pd.DataFrame
    if data_source.startswith("Demo: "):
        demo_name = data_source.removeprefix("Demo: ")
        demo_path = PROJECT_ROOT / "data" / demo_name
        df = read_csv(demo_path)
        st.info(f"Loaded demo dataset `{demo_name}` ({len(df)} rows, {len(df.columns)} columns).")
    else:
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
    _initialize_experiment_state(columns)
    _consume_pending_plan_suggestion(columns)

    st.subheader("Experiment setup")
    _section_caption("Choose the prediction target and tune the small number of parameters that affect the local run.")
    setup_cols = st.columns([1.2, 0.85, 1.15])
    with setup_cols[0]:
        target_columns = st.multiselect(
            "Target variables",
            columns,
            key="target_columns",
            help="Select one or more targets. Multi-target runs train one model per target.",
        )
    with setup_cols[1]:
        task_type_choice = st.radio(
            "Task type",
            ["auto", "classification", "regression"],
            horizontal=True,
            key="task_type_choice",
        )
    with setup_cols[2]:
        time_budget = _render_integer_input("Training time budget seconds", "time_budget", min_value=5)

    if not target_columns:
        st.warning("Select at least one target variable to continue.")
        return

    exclude_options = [column for column in columns if column not in target_columns]
    excluded_columns = st.multiselect(
        "Exclude columns from EDA and training features",
        exclude_options,
        key="excluded_columns",
        help="Excluded columns are removed before EDA and are not used as model features.",
    )
    analysis_columns = [column for column in columns if column not in excluded_columns]
    analysis_df = df[analysis_columns].copy()
    primary_target = target_columns[0]
    task_types = _target_task_types(analysis_df, target_columns, task_type_choice)
    if len(target_columns) > 1:
        st.caption(
            "Multi-target mode trains and stores one independent run per target. "
            "Other selected targets are excluded from each model's feature set."
        )

    config_cols = st.columns(4)
    with config_cols[0]:
        test_size = st.slider("Test size", min_value=0.1, max_value=0.5, step=0.05, key="test_size")
    with config_cols[1]:
        high_missing_threshold = st.slider(
            "Drop feature when missing rate is above",
            min_value=0.5,
            max_value=1.0,
            step=0.05,
            key="high_missing_threshold",
        )
    with config_cols[2]:
        random_state = _render_integer_input("Random state", "random_state")
    with config_cols[3]:
        priority_metric_choice = st.selectbox(
            "Priority metric",
            ["auto", "accuracy", "f1_weighted", "precision_weighted", "recall_weighted", "roc_auc", "rmse", "mae", "r2"],
            key="priority_metric_choice",
        )

    eda_summary = generate_eda_summary(analysis_df, target=primary_target)
    planner_brief = st.text_area(
        "Planning brief",
        key="planner_brief",
        placeholder="Example: predict churn, optimize recall, ignore customer_id-like fields, keep this as a quick baseline.",
        help="Optional natural-language brief used to suggest targets, task type, exclusions, and a priority metric.",
    )
    plan_suggestion = suggest_plan(
        df=analysis_df,
        eda_summary=eda_summary,
        settings=settings,
        user_brief=planner_brief,
    )
    plan_data = artifact_to_dict(plan_suggestion)
    st.subheader("Execution plan")
    _section_caption("Use the brief-driven suggestion as a starting point, then confirm the explicit controls before running.")
    with st.expander("Planner suggestion", expanded=bool(planner_brief.strip())):
        _render_planner_suggestion(plan_data)
        if st.button("Apply planner suggestions"):
            _queue_plan_suggestion(plan_data, columns)
            st.rerun()

    priority_metrics = {
        target: resolve_priority_metric(_target_task_types(analysis_df, [target], task_type_choice)[target], priority_metric_choice)
        for target in target_columns
    }
    preflight_by_target = {
        target: validate_preflight(
            df=df,
            target=target,
            task_type=task_types[target],
            excluded_columns=excluded_columns,
            priority_metric=priority_metric_choice,
            high_missing_threshold=float(high_missing_threshold),
        )
        for target in target_columns
    }
    with st.expander("Preflight validation", expanded=True):
        for target in target_columns:
            validation = artifact_to_dict(preflight_by_target[target])
            st.write(f"{target} ({task_types[target]})")
            st.caption(f"Resolved priority metric: {priority_metrics[target]}")
            _render_issue_table(validation.get("issues", []))

    st.subheader("Data preview")
    _section_caption("First 50 rows are shown for quick sanity checks before training.")
    st.dataframe(analysis_df.head(50), use_container_width=True)

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

    if primary_target in analysis_df.columns and eda_summary.get("target"):
        label = "Primary target profile" if len(target_columns) > 1 else "Target profile"
        st.write(label)
        _render_target_profile(eda_summary["target"])
    if len(target_columns) > 1:
        st.write("Target task types")
        st.dataframe(
            pd.DataFrame(
                [{"target": target, "task_type": task_type} for target, task_type in task_types.items()]
            ),
            use_container_width=True,
        )

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
        _render_target_relationships(eda_summary.get("target_relationships", {}))
    with eda_tabs[3]:
        for warning in eda_summary["quality_warnings"]:
            st.warning(warning)

    st.subheader("Training run")
    _section_caption("Launch the local pipeline after reviewing the setup and data audit.")
    if not st.button("Run training", type="primary"):
        return

    results = []
    storage = RunStorage(settings.runs_dir)
    with st.spinner("Cleaning data and training model locally..."):
        failing_targets = [target for target, validation in preflight_by_target.items() if not validation.ok_to_run]
        if failing_targets:
            logger.error("Blocking preflight issues targets=%s", failing_targets)
            st.error(f"Resolve blocking preflight issues before training: {', '.join(failing_targets)}")
            return

        feature_columns = [column for column in analysis_df.columns if column not in target_columns]
        if not feature_columns:
            logger.error("No feature columns remain after exclusions")
            st.error("No feature columns remain after excluding selected target and ignored columns.")
            return

        logger.info("Starting training pipeline targets=%s task_types=%s", target_columns, task_types)
        for target in target_columns:
            logger.info("Training target=%s task_type=%s", target, task_types[target])
            task_type = task_types[target]
            priority_metric = priority_metrics[target]
            preflight_validation = preflight_by_target[target]
            target_df = analysis_df[feature_columns + [target]].copy()
            target_eda_summary = generate_eda_summary(target_df, target=target)
            config = CleanConfig(
                target=target,
                task_type=task_type,
                test_size=float(test_size),
                random_state=int(random_state),
                high_missing_threshold=float(high_missing_threshold),
            )
            cleaned = clean_and_split(target_df, config)
            trained = train_model(cleaned, time_budget=int(time_budget), metric_preference=priority_metric)
            metrics, prediction_sample = evaluate_model(trained.model, cleaned, task_type=task_type)
            planned_report_mode = "openai" if settings.llm_enabled else "rule_based"
            postrun_validation = validate_postrun(
                metrics=metrics,
                prediction_sample=prediction_sample,
                task_type=task_type,
                priority_metric=priority_metric,
                trainer_name=trained.trainer_name,
                optimization_metric_used=trained.optimization_metric_used,
                feature_importance=trained.feature_importance,
                report_mode=planned_report_mode,
            )
            recommendations = build_recommendations(preflight_validation, postrun_validation)

            report, report_mode = generate_report_result(
                eda_summary=target_eda_summary,
                cleaning_log=cleaned.cleaning_log,
                metrics=metrics,
                feature_importance=trained.feature_importance,
                settings=settings,
                plan_suggestion=plan_suggestion,
                preflight_validation=preflight_validation,
                postrun_validation=postrun_validation,
                recommendations=recommendations,
            )
            postrun_validation = validate_postrun(
                metrics=metrics,
                prediction_sample=prediction_sample,
                task_type=task_type,
                priority_metric=priority_metric,
                trainer_name=trained.trainer_name,
                optimization_metric_used=trained.optimization_metric_used,
                feature_importance=trained.feature_importance,
                report_mode=report_mode,
            )
            recommendations = build_recommendations(preflight_validation, postrun_validation)
            if report_mode != planned_report_mode:
                report, report_mode = generate_report_result(
                    eda_summary=target_eda_summary,
                    cleaning_log=cleaned.cleaning_log,
                    metrics=metrics,
                    feature_importance=trained.feature_importance,
                    settings=settings,
                    plan_suggestion=plan_suggestion,
                    preflight_validation=preflight_validation,
                    postrun_validation=postrun_validation,
                    recommendations=recommendations,
                )

            run = storage.create_run(
                config={
                    "target": target,
                    "target_columns": target_columns,
                    "task_type": task_type,
                    "task_type_choice": task_type_choice,
                    "excluded_columns": excluded_columns,
                    "test_size": test_size,
                    "high_missing_threshold": high_missing_threshold,
                    "random_state": random_state,
                    "time_budget": time_budget,
                    "priority_metric": priority_metric,
                    "trainer": trained.trainer_name,
                    "planner_name": plan_suggestion.planner_name,
                }
            )
            storage.save_json(run, "plan.json", artifact_to_dict(plan_suggestion))
            storage.save_json(run, "eda_summary.json", target_eda_summary)
            storage.save_json(run, "cleaning_log.json", cleaned.cleaning_log)
            storage.save_json(run, "metrics.json", metrics)
            storage.save_json(run, "feature_importance.json", trained.feature_importance)
            storage.save_json(run, "validation_pre.json", artifact_to_dict(preflight_validation))
            storage.save_json(run, "validation_post.json", artifact_to_dict(postrun_validation))
            storage.save_json(run, "recommendations.json", artifact_to_dict(recommendations))
            storage.save_json(
                run,
                "training_summary.json",
                {
                    "trainer_name": trained.trainer_name,
                    "optimization_metric_used": trained.optimization_metric_used,
                    "training_notes": trained.training_notes or [],
                },
            )
            report_path = storage.save_text(run, "report.md", report)
            prediction_path = storage.save_predictions(run, prediction_sample)
            model_path = storage.save_model(run, trained.model)
            storage.record_run(run, metrics=metrics, status="completed")
            results.append(
                {
                    "target": target,
                    "task_type": task_type,
                    "run": run,
                    "trained": trained,
                    "metrics": metrics,
                    "report": report,
                    "preflight_validation": preflight_validation,
                    "postrun_validation": postrun_validation,
                    "recommendations": recommendations,
                    "priority_metric": priority_metric,
                    "report_path": report_path,
                    "prediction_path": prediction_path,
                    "model_path": model_path,
                }
            )

    completed_ids = ", ".join(str(result["run"].run_id) for result in results)
    logger.info("Training pipeline complete run_ids=%s", completed_ids)
    st.success(f"Run completed: {completed_ids}")
    st.subheader("Artifacts")
    _section_caption("Export the model, generated analysis report, and prediction sample for downstream review.")
    for result in results:
        run = result["run"]
        label = f"{result['target']} ({result['task_type']})"
        with st.expander(label, expanded=len(results) == 1):
            download_cols = st.columns(3)
            with download_cols[0]:
                st.download_button(
                    "Download model",
                    data=result["model_path"].read_bytes(),
                    file_name=f"{run.run_id}_{result['target']}_model.joblib",
                    mime="application/octet-stream",
                )
            with download_cols[1]:
                st.download_button(
                    "Download report",
                    data=result["report_path"].read_text(encoding="utf-8"),
                    file_name=f"{run.run_id}_{result['target']}_report.md",
                    mime="text/markdown",
                )
            with download_cols[2]:
                st.download_button(
                    "Download predictions",
                    data=result["prediction_path"].read_text(encoding="utf-8"),
                    file_name=f"{run.run_id}_{result['target']}_prediction_sample.csv",
                    mime="text/csv",
                )

    st.subheader("Run results")
    _section_caption("Metrics, feature importance, and the generated report are shown below for immediate review.")
    for result in results:
        run = result["run"]
        with st.expander(f"{result['target']} results", expanded=len(results) == 1):
            st.write("Metrics")
            _render_metrics(result["metrics"], priority_metric=result["priority_metric"])
            st.write("Validation")
            _render_validation_summary(
                preflight=artifact_to_dict(result["preflight_validation"]),
                postrun=artifact_to_dict(result["postrun_validation"]),
                recommendations=artifact_to_dict(result["recommendations"]),
                priority_metric=result["priority_metric"],
            )
            st.write("Feature importance")
            st.dataframe(pd.DataFrame(result["trained"].feature_importance), use_container_width=True)
            st.write("Analysis report")
            st.markdown(result["report"])
            st.caption(f"Artifacts saved to {Path(run.path).resolve()}")


if __name__ == "__main__":
    main()
