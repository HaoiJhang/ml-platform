from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

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
from ml_platform.data_flow import DataFlowTracker
from ml_platform.data_io import read_csv
from ml_platform.eda import generate_eda_summary
from ml_platform.evaluation import evaluate_model
from ml_platform.feature_engineering import suggest_feature_engineering_plan
from ml_platform.llm_report import generate_report_result
from ml_platform.manual_cleaning import (
    DEFAULT_EFFECT_STAGE,
    FILTER_OPERATORS,
    VALID_EFFECT_STAGES,
    VALID_RULE_TYPES,
    apply_manual_cleaning_plan,
    suggest_manual_cleaning_plan,
    validate_manual_cleaning_plan,
)
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

HERO_IMAGE_CANDIDATES = (
    PROJECT_ROOT / "data" / "hero.jpg",
    PROJECT_ROOT / "assets" / "hero.jpg",
    PROJECT_ROOT / "assets" / "hero.jpeg",
    PROJECT_ROOT / "assets" / "hero.png",
    PROJECT_ROOT / "彩虹.jpg",
)
LOCAL_LLM_CONFIG_PATH = PROJECT_ROOT / ".ml_platform.local.json"
_LEGACY_PROVIDER_TOKEN = "open" + "ai"
_LEGACY_LOCAL_LLM_KEYS = {
    "llm_api_key": f"{_LEGACY_PROVIDER_TOKEN}_api_key",
    "llm_base_url": f"{_LEGACY_PROVIDER_TOKEN}_base_url",
    "llm_model": f"{_LEGACY_PROVIDER_TOKEN}_model",
}
PLANNER_CACHE_VERSION = 1
FEATURE_PLAN_CACHE_VERSION = 2
MANUAL_CLEANING_PLAN_CACHE_VERSION = 1


def _local_llm_config_enabled() -> bool:
    return os.getenv("ML_PLATFORM_ALLOW_LOCAL_LLM_CONFIG", "1") != "0"


def _load_local_llm_config() -> dict[str, str]:
    if not _local_llm_config_enabled():
        return {}
    if not LOCAL_LLM_CONFIG_PATH.exists():
        return {}
    try:
        raw_config = json.loads(LOCAL_LLM_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Unable to read local LLM config: %s", exc)
        return {}
    if not isinstance(raw_config, dict):
        return {}
    normalized: dict[str, str] = {}
    for key in ("llm_api_key", "llm_base_url", "llm_model"):
        value = raw_config.get(key)
        if not isinstance(value, str):
            value = raw_config.get(_LEGACY_LOCAL_LLM_KEYS[key])
        if isinstance(value, str):
            normalized[key] = value
    return normalized


def _save_local_llm_config(config: dict[str, str]) -> None:
    if not _local_llm_config_enabled():
        return
    try:
        LOCAL_LLM_CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
        os.chmod(LOCAL_LLM_CONFIG_PATH, 0o600)
    except OSError as exc:
        logger.warning("Unable to write local LLM config: %s", exc)


def _delete_local_llm_config() -> None:
    if not _local_llm_config_enabled():
        return
    try:
        LOCAL_LLM_CONFIG_PATH.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Unable to delete local LLM config: %s", exc)


def _dataset_fingerprint(df: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes()
    schema = json.dumps(
        {
            "columns": [str(column) for column in df.columns],
            "dtypes": [str(dtype) for dtype in df.dtypes],
            "rows": len(df),
        },
        ensure_ascii=True,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(schema + payload).hexdigest()


def _load_hero_background() -> str:
    hero_image_path = next((path for path in HERO_IMAGE_CANDIDATES if path.exists()), None)
    if hero_image_path is None:
        return (
            "linear-gradient(100deg, rgba(40, 64, 88, 0.98) 0 38%, "
            "rgba(39, 70, 101, 0.86) 38% 62%, rgba(42, 53, 67, 0.92) 62%)"
        )
    encoded = base64.b64encode(hero_image_path.read_bytes()).decode("ascii")
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
            [data-baseweb="select"] > div {
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

            [data-testid="stFileUploader"] section,
            [data-testid="stFileUploader"] small,
            [data-testid="stFileUploader"] p,
            [data-testid="stFileUploader"] label,
            [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] * {
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


def _render_help_center() -> None:
    st.subheader("Quick start")
    _section_caption("This page is organized as a guided first run. Advanced settings stay out of the way until you need them.")
    st.info(
        "Upload a dataset, choose the column to predict, review the checks, then run training. "
        "Optional AI help and advanced adjustments can stay closed for a first pass."
    )


def _section_caption(text: str) -> None:
    st.markdown(f'<p class="lab-caption">{text}</p>', unsafe_allow_html=True)


def _render_step_status(current_action: str, next_action: str, level: str = "info") -> None:
    message = f"Now: {current_action} Next: {next_action}"
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    else:
        st.info(message)


def _configure_llm_settings(settings: Settings) -> Settings:
    allow_local_llm_config = _local_llm_config_enabled()
    saved_config = _load_local_llm_config()
    saved_api_key = saved_config.get("llm_api_key", "")
    saved_base_url = saved_config.get("llm_base_url", "")
    saved_model = saved_config.get("llm_model", "")
    _section_caption(
        "Optional AI help: you can finish the full local training flow without any API key. "
        "Add one only if you want AI-generated suggestions and a more natural-language report."
    )

    api_key = saved_api_key
    base_url = saved_base_url or settings.llm_base_url or ""
    model = saved_model or settings.llm_model

    with st.expander("Optional AI help", expanded=False):
        if allow_local_llm_config:
            st.caption(
                "This environment can remember settings locally. Hosted deployments can also use "
                "`LLM_API_KEY` or Streamlit Secrets."
            )
        else:
            st.caption(
                "Local persistence is disabled here, so enter a key per session or configure "
                "`LLM_API_KEY` / Streamlit Secrets."
            )

        config_cols = st.columns(3)
        with config_cols[0]:
            api_key = st.text_input(
                "API key",
                value=saved_api_key,
                key="_llm_api_key",
                type="password",
                placeholder="Uses LLM_API_KEY if empty",
            )
        with config_cols[1]:
            base_url = st.text_input(
                "Base URL",
                value=saved_base_url or settings.llm_base_url or "",
                key="_llm_base_url",
                placeholder="LLM default or compatible API URL",
            )
        with config_cols[2]:
            model = st.text_input("Model", value=saved_model or settings.llm_model, key="_llm_model")

        if allow_local_llm_config:
            remember_config = st.checkbox("Remember LLM settings on this device", value=bool(saved_api_key))
            if remember_config and api_key:
                _save_local_llm_config(
                    {
                        "llm_api_key": api_key,
                        "llm_base_url": base_url,
                        "llm_model": model,
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
        llm_api_key=api_key or saved_api_key or settings.llm_api_key,
        llm_base_url=base_url or settings.llm_base_url or None,
        llm_model=model or settings.llm_model,
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


def _experiment_signature(
    *,
    dataset_fingerprint: str,
    target_columns: list[str],
    task_type_choice: str,
    time_budget: int,
    excluded_columns: list[str],
    test_size: float,
    high_missing_threshold: float,
    random_state: int,
    priority_metric_choice: str,
    planner_brief: str,
    feature_plan: Any,
    manual_cleaning_plan: Any,
) -> str:
    payload = {
        "dataset_fingerprint": dataset_fingerprint,
        "target_columns": target_columns,
        "task_type_choice": task_type_choice,
        "time_budget": time_budget,
        "excluded_columns": excluded_columns,
        "test_size": test_size,
        "high_missing_threshold": high_missing_threshold,
        "random_state": random_state,
        "priority_metric_choice": priority_metric_choice,
        "planner_brief": planner_brief.strip(),
        "feature_engineering_operations": _feature_plan_operations(feature_plan) if feature_plan else [],
        "manual_cleaning_plan": artifact_to_dict(manual_cleaning_plan) if manual_cleaning_plan else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def _initialize_experiment_state(dataset_signature: str, columns: list[str]) -> None:
    signature = (dataset_signature, tuple(columns))
    if st.session_state.get("_dataset_signature") == signature:
        return
    st.session_state["_dataset_signature"] = signature
    st.session_state["planner_brief"] = ""
    st.session_state["target_columns"] = []
    st.session_state["task_type_choice"] = "auto"
    st.session_state["time_budget"] = 30
    st.session_state["time_budget_text"] = "30"
    st.session_state["priority_metric_choice"] = "auto"
    st.session_state["apply_feature_engineering"] = False
    st.session_state["excluded_columns"] = []
    st.session_state["test_size"] = 0.2
    st.session_state["high_missing_threshold"] = 0.9
    st.session_state["random_state"] = 42
    st.session_state["random_state_text"] = "42"
    st.session_state["_planner_signature"] = None
    st.session_state["_planner_suggestion"] = None
    st.session_state["_feature_engineering_plan_signature"] = None
    st.session_state["_feature_engineering_plan"] = None
    st.session_state["_feature_engineering_override_plan"] = None
    st.session_state["manual_cleaning_brief"] = ""
    st.session_state["_manual_cleaning_plan_signature"] = None
    st.session_state["_manual_cleaning_suggested_plan"] = None
    st.session_state["_manual_cleaning_plan"] = None
    st.session_state["_manual_cleaning_override_plan"] = None
    st.session_state["_latest_results_signature"] = None
    st.session_state["_latest_results"] = []


def _render_run_outputs(results: list[dict[str, object]]) -> None:
    st.subheader("5. Review results")
    _section_caption("Training has finished. Start with the short summary below, then open details or download files.")
    completed_targets = ", ".join(str(result["target"]) for result in results)
    _render_step_status(
        f"Training finished for: {completed_targets}.",
        "Check the validation notes first, then download the report, model, or prediction sample you need.",
        level="success",
    )
    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.metric("Completed runs", len(results))
    with summary_cols[1]:
        st.metric("Targets trained", len({str(result["target"]) for result in results}))
    with summary_cols[2]:
        st.metric("Result files per run", 3)
    summary_rows = [
        {
            "target": str(result["target"]),
            "task_type": str(result["task_type"]),
            "priority_metric": str(result["priority_metric"]),
            "trainer": str(result["trained"].trainer_name),
        }
        for result in results
    ]
    st.write("Run summary")
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)
    st.write("Download files")
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

    st.write("Detailed results")
    for result in results:
        run = result["run"]
        with st.expander(f"{result['target']} results", expanded=len(results) == 1):
            st.write("Metrics")
            _render_metrics(result["metrics"], priority_metric=result["priority_metric"])
            st.write("Validation checks")
            _render_validation_summary(
                preflight=artifact_to_dict(result["preflight_validation"]),
                postrun=artifact_to_dict(result["postrun_validation"]),
                recommendations=artifact_to_dict(result["recommendations"]),
                priority_metric=result["priority_metric"],
            )
            st.write("Data flow")
            _render_data_flow(result["data_flow"])
            st.write("Feature importance")
            st.dataframe(pd.DataFrame(result["trained"].feature_importance), use_container_width=True)
            st.write("Analysis report")
            st.markdown(result["report"])
            st.caption(f"Artifacts saved to {Path(run.path).resolve()}")


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
    normalized_items = items if isinstance(items, list) else [items]
    clean_items = [str(item) for item in normalized_items if str(item).strip()]
    if not clean_items:
        st.caption(empty_text)
        return
    for item in clean_items:
        st.write(f"- {item}")


def _normalize_item_list(value: Any) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


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
        _render_text_items("Notes", _normalize_item_list(plan_data.get("notes")), "No notes.")
    with notes_cols[1]:
        _render_text_items("Risk flags", _normalize_item_list(plan_data.get("risk_flags")), "No risk flags.")

    with st.expander("Raw planner JSON", expanded=False):
        st.json(plan_data, expanded=True)


def _get_planner_suggestion(
    df: pd.DataFrame,
    eda_summary: dict[str, Any],
    settings: Settings,
    user_brief: str,
    dataset_fingerprint: str,
):
    signature = (
        PLANNER_CACHE_VERSION,
        dataset_fingerprint,
        tuple(str(column) for column in df.columns),
        tuple(str(dtype) for dtype in df.dtypes),
        len(df),
        user_brief.strip(),
        bool(settings.llm_api_key),
        settings.llm_base_url or "",
        settings.llm_model,
    )
    if st.session_state.get("_planner_signature") == signature:
        return st.session_state.get("_planner_suggestion")

    plan = suggest_plan(
        df=df,
        eda_summary=eda_summary,
        settings=settings,
        user_brief=user_brief,
    )
    st.session_state["_planner_signature"] = signature
    st.session_state["_planner_suggestion"] = plan
    return plan


def _render_feature_engineering_plan(plan_data: dict[str, object]) -> None:
    operations = _normalize_item_list(plan_data.get("operations"))
    rejected = _normalize_item_list(plan_data.get("rejected_operations"))
    notes = _normalize_item_list(plan_data.get("notes"))

    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.metric("Planner", str(plan_data.get("planner_name") or "local_whitelist"))
    with summary_cols[1]:
        st.metric("Accepted ops", len(operations))
    with summary_cols[2]:
        st.metric("Rejected ops", len(rejected))

    if operations:
        rows = []
        for operation in operations:
            if not isinstance(operation, dict):
                continue
            rows.append(
                {
                    "operation": operation.get("operation"),
                    "source": operation.get("source_column") or ", ".join(operation.get("columns", [])),
                    "detail": operation.get("operator") or ", ".join(operation.get("parts", [])) or operation.get("bins") or "",
                    "rationale": operation.get("rationale", ""),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.caption("No feature engineering operations were accepted.")

    detail_cols = st.columns(2)
    with detail_cols[0]:
        _render_text_items("Notes", notes, "No notes.")
    with detail_cols[1]:
        _render_text_items("Rejected operations", rejected, "No rejected operations.")


def _feature_plan_operations(plan_data: Any) -> list[Any]:
    if isinstance(plan_data, dict):
        return list(plan_data.get("operations", []))
    return list(getattr(plan_data, "operations", []))


def _get_feature_engineering_plan(
    df: pd.DataFrame,
    target: str,
    settings: Settings,
    user_brief: str,
):
    signature = (
        FEATURE_PLAN_CACHE_VERSION,
        tuple(str(column) for column in df.columns),
        tuple(str(dtype) for dtype in df.dtypes),
        len(df),
        target,
        user_brief.strip(),
        bool(settings.llm_api_key),
        settings.llm_base_url or "",
        settings.llm_model,
    )
    if st.session_state.get("_feature_engineering_plan_signature") == signature:
        return st.session_state.get("_feature_engineering_plan")

    plan = suggest_feature_engineering_plan(
        df=df,
        target=target,
        eda_summary=generate_eda_summary(df, target=target),
        settings=settings,
        user_brief=user_brief,
    )
    st.session_state["_feature_engineering_plan_signature"] = signature
    st.session_state["_feature_engineering_plan"] = plan
    return plan


def _clone_json_data(value: Any) -> Any:
    return json.loads(json.dumps(artifact_to_dict(value), ensure_ascii=False))


def _new_manual_cleaning_rule(column: str = "", *, rule_type: str = "filter_row") -> dict[str, object]:
    return {
        "id": f"manual_rule_{os.urandom(4).hex()}",
        "enabled": True,
        "rule_type": rule_type,
        "column": column,
        "operator": "equals" if rule_type == "filter_row" else None,
        "value": None,
        "rationale": "",
    }


def _blank_manual_cleaning_plan(user_brief: str = "") -> dict[str, object]:
    return {
        "planner_name": "manual",
        "user_brief": user_brief,
        "effect_stage": DEFAULT_EFFECT_STAGE,
        "rules": [_new_manual_cleaning_rule()],
        "notes": [],
        "rejected_rules": [],
    }


def _manual_cleaning_rule_value_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if str(item).strip())
    return str(value)


def _manual_cleaning_effect_stage_label(effect_stage: str) -> str:
    if effect_stage == "pre_training":
        return "Apply only before training"
    return "Apply before EDA and training"


def _manual_cleaning_rule_rows(plan_data: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rule in plan_data.get("rules", []):
        if not isinstance(rule, dict):
            continue
        rows.append(
            {
                "enabled": bool(rule.get("enabled", True)),
                "rule_type": rule.get("rule_type"),
                "column": rule.get("column"),
                "operator": rule.get("operator") or "-",
                "value": _manual_cleaning_rule_value_text(rule.get("value")) or "-",
                "rationale": rule.get("rationale") or "",
            }
        )
    return rows


def _get_manual_cleaning_plan(
    df: pd.DataFrame,
    target: str,
    settings: Settings,
    user_brief: str,
):
    signature = (
        MANUAL_CLEANING_PLAN_CACHE_VERSION,
        tuple(str(column) for column in df.columns),
        tuple(str(dtype) for dtype in df.dtypes),
        len(df),
        target,
        user_brief.strip(),
        bool(settings.llm_api_key),
        settings.llm_base_url or "",
        settings.llm_model,
    )
    if st.session_state.get("_manual_cleaning_plan_signature") == signature:
        return st.session_state.get("_manual_cleaning_suggested_plan")

    plan = suggest_manual_cleaning_plan(
        df=df,
        target=target,
        settings=settings,
        user_brief=user_brief,
    )
    st.session_state["_manual_cleaning_plan_signature"] = signature
    st.session_state["_manual_cleaning_suggested_plan"] = plan
    return plan


def _render_manual_cleaning_editor(
    draft_plan: dict[str, object],
    *,
    available_columns: list[str],
) -> dict[str, object]:
    current_rules = [item for item in draft_plan.get("rules", []) if isinstance(item, dict)]
    effect_stage_value = str(draft_plan.get("effect_stage") or DEFAULT_EFFECT_STAGE)
    if effect_stage_value not in VALID_EFFECT_STAGES:
        effect_stage_value = DEFAULT_EFFECT_STAGE
    effect_stage_options = ["pre_eda", "pre_training"]
    effect_stage = st.radio(
        "Rule effect stage",
        effect_stage_options,
        index=effect_stage_options.index(effect_stage_value),
        format_func=_manual_cleaning_effect_stage_label,
        horizontal=True,
    )

    updated_rules: list[dict[str, object]] = []
    delete_rule_id: str | None = None
    for index, rule in enumerate(current_rules, start=1):
        rule_id = str(rule.get("id") or f"manual_rule_{index}")
        enabled_default = bool(rule.get("enabled", True))
        rule_type_default = str(rule.get("rule_type") or "filter_row")
        if rule_type_default not in VALID_RULE_TYPES:
            rule_type_default = "filter_row"
        column_default = str(rule.get("column") or "")
        operator_default = str(rule.get("operator") or "equals") if rule_type_default == "filter_row" else ""
        if operator_default not in FILTER_OPERATORS:
            operator_default = "equals"
        value_default = _manual_cleaning_rule_value_text(rule.get("value"))
        rationale_default = str(rule.get("rationale") or "")

        title = f"Rule {index}: {column_default or 'Select column'}"
        with st.expander(title, expanded=len(current_rules) == 1):
            top_cols = st.columns([0.8, 1.0, 1.3, 0.9])
            with top_cols[0]:
                enabled = st.checkbox("Enabled", value=enabled_default, key=f"manual_rule_enabled_{rule_id}")
            with top_cols[1]:
                rule_type = st.selectbox(
                    "Rule type",
                    ["drop_column", "filter_row"],
                    index=["drop_column", "filter_row"].index(rule_type_default),
                    key=f"manual_rule_type_{rule_id}",
                )
            with top_cols[2]:
                column_options = ["", *available_columns]
                column_index = column_options.index(column_default) if column_default in column_options else 0
                column = st.selectbox(
                    "Column",
                    column_options,
                    index=column_index,
                    key=f"manual_rule_column_{rule_id}",
                )
            with top_cols[3]:
                delete_clicked = st.button("Delete rule", key=f"manual_rule_delete_{rule_id}")
                if delete_clicked:
                    delete_rule_id = rule_id

            operator: str | None = None
            parsed_value: str | list[str] | None = None
            if rule_type == "filter_row":
                operator_cols = st.columns([1.1, 1.9])
                with operator_cols[0]:
                    operator_options = [
                        "is_null",
                        "not_null",
                        "equals",
                        "not_equals",
                        "in",
                        "not_in",
                        "contains",
                        "not_contains",
                        "gt",
                        "gte",
                        "lt",
                        "lte",
                    ]
                    operator = st.selectbox(
                        "Operator",
                        operator_options,
                        index=operator_options.index(operator_default),
                        key=f"manual_rule_operator_{rule_id}",
                    )
                with operator_cols[1]:
                    if operator in {"is_null", "not_null"}:
                        st.caption("No value is needed for this operator.")
                    else:
                        label = "Values (comma-separated)" if operator in {"in", "not_in"} else "Value"
                        raw_value = st.text_input(label, value=value_default, key=f"manual_rule_value_{rule_id}")
                        if operator in {"in", "not_in"}:
                            parsed_value = [item.strip() for item in raw_value.split(",") if item.strip()]
                        else:
                            parsed_value = raw_value.strip() or None
            else:
                st.caption("This rule drops the selected column before downstream processing.")

            rationale = st.text_input("Rationale", value=rationale_default, key=f"manual_rule_rationale_{rule_id}")
            updated_rules.append(
                {
                    "id": rule_id,
                    "enabled": enabled,
                    "rule_type": rule_type,
                    "column": column,
                    "operator": operator,
                    "value": parsed_value,
                    "rationale": rationale,
                }
            )

    if delete_rule_id:
        updated_rules = [rule for rule in updated_rules if str(rule.get("id")) != delete_rule_id]

    controls = st.columns(2)
    add_blank_rule = controls[0].button("Add blank rule")
    reset_draft = controls[1].button("Reset draft")

    updated_plan = {
        "planner_name": str(draft_plan.get("planner_name") or "manual"),
        "user_brief": str(draft_plan.get("user_brief") or ""),
        "effect_stage": effect_stage,
        "rules": updated_rules,
        "notes": list(draft_plan.get("notes", [])) if isinstance(draft_plan.get("notes"), list) else [],
        "rejected_rules": [],
    }
    if add_blank_rule:
        updated_plan["rules"].append(_new_manual_cleaning_rule())
    if reset_draft:
        applied = st.session_state.get("_manual_cleaning_plan")
        st.session_state["_manual_cleaning_override_plan"] = _clone_json_data(applied) if applied else None
        st.rerun()

    st.session_state["_manual_cleaning_override_plan"] = updated_plan
    if delete_rule_id or add_blank_rule:
        st.rerun()
    return updated_plan


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


def _preview_to_frame(preview: dict[str, object]) -> pd.DataFrame:
    rows = preview.get("rows", [])
    columns = [str(column) for column in preview.get("columns", [])]
    if not isinstance(rows, list):
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    ordered_columns = [column for column in columns if column in frame.columns]
    remaining_columns = [column for column in frame.columns if column not in ordered_columns]
    return frame[ordered_columns + remaining_columns] if not frame.empty or ordered_columns else frame


def _shape_label(snapshot: dict[str, object]) -> str:
    rows = snapshot.get("rows")
    columns = snapshot.get("columns")
    if rows is None and columns is None:
        return "-"
    if rows is None:
        return f"? x {columns}"
    if columns is None:
        return f"{rows} x ?"
    return f"{rows} x {columns}"


def _render_data_flow(trace_data: dict[str, object]) -> None:
    snapshots = [item for item in trace_data.get("snapshots", []) if isinstance(item, dict)]
    if not snapshots:
        st.caption("No data flow trace available.")
        return

    target_name = str(trace_data.get("target") or "run")
    for start in range(0, len(snapshots), 4):
        chunk = snapshots[start : start + 4]
        columns = st.columns(len(chunk))
        for index, snapshot in enumerate(chunk, start=start + 1):
            with columns[index - start - 1]:
                st.caption(f"{index}. {snapshot.get('label') or snapshot.get('step')}")
                st.metric("Shape", _shape_label(snapshot))
                partition = str(snapshot.get("partition") or "full")
                stage = str(snapshot.get("stage") or "-")
                delta = snapshot.get("rows_delta")
                delta_text = "-" if delta is None else f"{int(delta):+d}"
                st.caption(f"{stage} | {partition} | delta {delta_text}")

    summary_rows = []
    for index, snapshot in enumerate(snapshots, start=1):
        summary_rows.append(
            {
                "index": index,
                "step": snapshot.get("step"),
                "label": snapshot.get("label"),
                "stage": snapshot.get("stage"),
                "partition": snapshot.get("partition"),
                "kind": snapshot.get("data_kind"),
                "shape": _shape_label(snapshot),
                "rows_delta": snapshot.get("rows_delta"),
                "added": len(snapshot.get("columns_added", [])),
                "removed": len(snapshot.get("columns_removed", [])),
            }
        )
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

    options = [
        f"{index}. {snapshot.get('label') or snapshot.get('step')} [{snapshot.get('partition') or 'full'}]"
        for index, snapshot in enumerate(snapshots, start=1)
    ]
    selected = st.selectbox(
        "Inspect data flow step",
        options,
        key=f"data_flow_step_{target_name}",
    )
    selected_index = options.index(selected)
    selected_snapshot = snapshots[selected_index]

    metric_cols = st.columns(5)
    with metric_cols[0]:
        st.metric("Stage", str(selected_snapshot.get("stage") or "-"))
    with metric_cols[1]:
        st.metric("Partition", str(selected_snapshot.get("partition") or "-"))
    with metric_cols[2]:
        st.metric("Kind", str(selected_snapshot.get("data_kind") or "-"))
    with metric_cols[3]:
        st.metric("Shape", _shape_label(selected_snapshot))
    with metric_cols[4]:
        memory = selected_snapshot.get("memory_mb")
        memory_text = "-" if memory is None else f"{float(memory):.4f} MB"
        st.metric("Memory", memory_text)

    delta_cols = st.columns(3)
    with delta_cols[0]:
        delta = selected_snapshot.get("rows_delta")
        st.metric("Rows delta", "-" if delta is None else f"{int(delta):+d}")
    with delta_cols[1]:
        st.metric("Columns added", len(selected_snapshot.get("columns_added", [])))
    with delta_cols[2]:
        st.metric("Columns removed", len(selected_snapshot.get("columns_removed", [])))

    metadata = selected_snapshot.get("metadata", {})
    if isinstance(metadata, dict) and metadata:
        st.write("Metadata")
        st.json(metadata, expanded=True)

    detail_cols = st.columns(2)
    with detail_cols[0]:
        added = [str(item) for item in selected_snapshot.get("columns_added", []) if str(item).strip()]
        _render_text_items("Columns added", added, "No columns added.")
    with detail_cols[1]:
        removed = [str(item) for item in selected_snapshot.get("columns_removed", []) if str(item).strip()]
        _render_text_items("Columns removed", removed, "No columns removed.")

    preview = selected_snapshot.get("preview")
    if isinstance(preview, dict):
        st.write("Preview")
        preview_frame = _preview_to_frame(preview)
        st.dataframe(preview_frame, use_container_width=True)
        if preview.get("truncated"):
            st.caption("Preview truncated to the first rows.")


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
    _render_help_center()
    settings = _configure_llm_settings(load_settings())
    storage = RunStorage(settings.runs_dir)

    st.subheader("1. Upload data")
    _section_caption("Start with one CSV file or a demo dataset. The app saves each run under the configured runs directory.")

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
        _render_step_status(
            f"Loaded demo dataset `{demo_name}` with {len(df)} rows and {len(df.columns)} columns.",
            "Choose the column you want to predict.",
            level="success",
        )
    else:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if uploaded_file is None:
            _render_step_status(
                "No dataset has been loaded yet.",
                "Upload a CSV or pick a demo dataset to unlock the next step.",
            )
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
        _render_step_status(
            f"Loaded `{uploaded_file.name}` with {len(df)} rows and {len(df.columns)} columns.",
            "Choose the column you want to predict.",
            level="success",
        )
    current_dataset_fingerprint = _dataset_fingerprint(df)
    columns = list(df.columns)
    _initialize_experiment_state(current_dataset_fingerprint, columns)
    _consume_pending_plan_suggestion(columns)

    st.subheader("2. Choose what to predict")
    _section_caption("Pick the column you want the app to predict. The app can infer the task type automatically.")
    setup_cols = st.columns([1.5, 1.0])
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

    if not target_columns:
        _render_step_status(
            "Your dataset is ready, but no prediction target has been selected yet.",
            "Select at least one target column to continue to the checks step.",
            level="warning",
        )
        return

    selected_targets = ", ".join(str(target) for target in target_columns)
    _render_step_status(
        f"Selected target column{'s' if len(target_columns) > 1 else ''}: {selected_targets}.",
        "Review the data checks before starting training.",
        level="success",
    )

    exclude_options = [column for column in columns if column not in target_columns]
    with st.expander("Advanced experiment settings", expanded=False):
        st.caption("Most first runs can keep the defaults here. Open this only if you want more control.")
        excluded_columns = st.multiselect(
            "Exclude columns from EDA and training features",
            exclude_options,
            key="excluded_columns",
            help="Excluded columns are removed before EDA and are not used as model features.",
        )
        top_advanced_cols = st.columns(2)
        with top_advanced_cols[0]:
            time_budget = _render_integer_input("Training time budget seconds", "time_budget", min_value=5)
        with top_advanced_cols[1]:
            priority_metric_choice = st.selectbox(
                "Priority metric",
                ["auto", "accuracy", "f1_weighted", "precision_weighted", "recall_weighted", "roc_auc", "rmse", "mae", "r2"],
                key="priority_metric_choice",
                help="This is the score the trainer treats as most important when choosing the best baseline.",
            )

        config_cols = st.columns(3)
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

    analysis_columns = [column for column in columns if column not in excluded_columns]
    base_analysis_df = df[analysis_columns].copy()
    primary_target = target_columns[0]
    applied_manual_cleaning_payload = st.session_state.get("_manual_cleaning_plan")
    manual_cleaning_plan = None
    analysis_manual_cleaning_log: list[dict[str, object]] = []
    analysis_manual_cleaning_impact: dict[str, object] = {}
    analysis_df = base_analysis_df
    if isinstance(applied_manual_cleaning_payload, dict):
        manual_cleaning_plan = validate_manual_cleaning_plan(
            applied_manual_cleaning_payload,
            base_analysis_df,
            primary_target,
            protected_columns=target_columns,
        )
        if any(rule.enabled for rule in manual_cleaning_plan.rules) and manual_cleaning_plan.effect_stage == "pre_eda":
            analysis_df, analysis_manual_cleaning_log, analysis_manual_cleaning_impact = apply_manual_cleaning_plan(
                base_analysis_df,
                manual_cleaning_plan,
                primary_target,
                protected_columns=target_columns,
            )

    candidate_feature_columns = [column for column in analysis_df.columns if column not in target_columns]
    task_types = _target_task_types(analysis_df, target_columns, task_type_choice)
    if len(target_columns) > 1:
        st.caption(
            "Multi-target mode trains and stores one independent run per target. "
            "Other selected targets are excluded from each model's feature set."
        )

    eda_summary = generate_eda_summary(analysis_df, target=primary_target)
    st.subheader("3. Check data before training")
    _section_caption("Use the brief, validation checks, and data summary to catch issues before you spend time training.")
    st.write("Planning help")
    _section_caption("This optional brief lets you describe your goal in plain language so the app can suggest a sensible first setup.")
    planner_brief = st.text_area(
        "Planning brief",
        key="planner_brief",
        placeholder="Example: predict churn, treat customer_id as reference only, and keep this as a quick first pass.",
        help="Optional natural-language brief used to suggest targets, task type, exclusions, and a priority metric.",
    )
    plan_suggestion = _get_planner_suggestion(
        df=analysis_df,
        eda_summary=eda_summary,
        settings=settings,
        user_brief=planner_brief,
        dataset_fingerprint=current_dataset_fingerprint,
    )
    plan_data = artifact_to_dict(plan_suggestion)
    with st.expander("Planner suggestion", expanded=bool(planner_brief.strip())):
        _render_planner_suggestion(plan_data)
        if st.button("Apply planner suggestions"):
            _queue_plan_suggestion(plan_data, columns)
            st.rerun()

    feature_plan = None
    with st.expander("Advanced adjustments", expanded=False):
        st.caption(
            "Most first runs can skip this section. Open it only if you want to clean rows or columns manually, "
            "or add extra local feature transformations."
        )
        st.write("Manual cleaning rules")
        _section_caption("If you already know some rows or columns should be filtered out, draft the rules here before training.")
        manual_cleaning_brief = st.text_area(
            "Cleaning rules brief",
            key="manual_cleaning_brief",
            placeholder="Example: drop customer_id and keep rows where monthly_spend > 20 and churn equals 1.",
            help="Natural-language rules are converted into a structured draft. Nothing is applied until you confirm.",
        )
        manual_rule_controls = st.columns(3)
        with manual_rule_controls[0]:
            if st.button("Generate cleaning rules"):
                suggested_manual_plan = _get_manual_cleaning_plan(
                    df=base_analysis_df,
                    target=primary_target,
                    settings=settings,
                    user_brief=manual_cleaning_brief,
                )
                st.session_state["_manual_cleaning_override_plan"] = _clone_json_data(suggested_manual_plan)
                st.rerun()
        with manual_rule_controls[1]:
            if st.button("Start with blank rule"):
                st.session_state["_manual_cleaning_override_plan"] = _blank_manual_cleaning_plan(manual_cleaning_brief)
                st.rerun()
        with manual_rule_controls[2]:
            if st.session_state.get("_manual_cleaning_plan") and st.button("Clear applied manual rules"):
                st.session_state["_manual_cleaning_plan"] = None
                st.session_state["_manual_cleaning_override_plan"] = None
                st.rerun()

        if st.session_state.get("_manual_cleaning_override_plan") is None and manual_cleaning_plan is not None:
            st.session_state["_manual_cleaning_override_plan"] = _clone_json_data(manual_cleaning_plan)

        draft_manual_plan = st.session_state.get("_manual_cleaning_override_plan")
        if isinstance(draft_manual_plan, dict):
            draft_manual_plan["user_brief"] = manual_cleaning_brief
            edited_manual_plan = _render_manual_cleaning_editor(
                draft_manual_plan,
                available_columns=list(base_analysis_df.columns),
            )
            validated_manual_preview = validate_manual_cleaning_plan(
                edited_manual_plan,
                base_analysis_df,
                primary_target,
                protected_columns=target_columns,
            )
            preview_df, preview_log, preview_impact = apply_manual_cleaning_plan(
                base_analysis_df,
                validated_manual_preview,
                primary_target,
                protected_columns=target_columns,
            )
            preview_plan_data = artifact_to_dict(validated_manual_preview)
            summary_cols = st.columns(5)
            with summary_cols[0]:
                st.metric("Planner", str(preview_plan_data.get("planner_name") or "manual"))
            with summary_cols[1]:
                st.metric("Effect stage", _manual_cleaning_effect_stage_label(validated_manual_preview.effect_stage))
            with summary_cols[2]:
                st.metric("Accepted rules", len(preview_plan_data.get("rules", [])))
            with summary_cols[3]:
                st.metric("Rejected rules", len(preview_plan_data.get("rejected_rules", [])))
            with summary_cols[4]:
                st.metric("Rows removed", int(preview_impact.get("rows_removed") or 0))

            rule_rows = _manual_cleaning_rule_rows(preview_plan_data)
            if rule_rows:
                st.dataframe(pd.DataFrame(rule_rows), hide_index=True, use_container_width=True)
            else:
                st.caption("No manual cleaning rules are in the current draft.")

            impact_cols = st.columns(3)
            with impact_cols[0]:
                st.metric("Columns removed", len(preview_impact.get("columns_removed", [])))
            with impact_cols[1]:
                st.metric("Rows after", int(preview_impact.get("rows_after") or len(base_analysis_df)))
            with impact_cols[2]:
                st.metric("Columns after", int(preview_impact.get("columns_after") or len(base_analysis_df.columns)))

            detail_cols = st.columns(2)
            with detail_cols[0]:
                _render_text_items("Notes", preview_plan_data.get("notes", []), "No notes.")
            with detail_cols[1]:
                _render_text_items("Rejected rules", preview_plan_data.get("rejected_rules", []), "No rejected rules.")

            with st.expander("Manual cleaning preview impact", expanded=False):
                st.json(preview_impact, expanded=True)
                st.dataframe(preview_df.head(20), use_container_width=True)
                if preview_log:
                    st.write("Planned cleaning log")
                    st.json(preview_log, expanded=True)

            apply_cols = st.columns(2)
            with apply_cols[0]:
                if st.button("Apply manual cleaning rules", type="primary"):
                    st.session_state["_manual_cleaning_plan"] = preview_plan_data
                    st.session_state["_manual_cleaning_override_plan"] = _clone_json_data(preview_plan_data)
                    st.rerun()
            with apply_cols[1]:
                if st.session_state.get("_manual_cleaning_plan"):
                    st.caption("Applied rules remain active until you clear them or apply a different draft.")
        else:
            st.caption("Generate rules from a brief or start with a blank rule to configure manual cleaning.")

        apply_feature_engineering = st.checkbox(
            "Apply local whitelist feature engineering",
            key="apply_feature_engineering",
            help="LLM can propose a structured plan, but only local whitelisted transformations are executed inside the training pipeline.",
        )
        if apply_feature_engineering:
            feature_plan_override = st.session_state.get("_feature_engineering_override_plan")
            if isinstance(feature_plan_override, dict) and feature_plan_override.get("feature_engineering_operations"):
                feature_plan = feature_plan_override
            else:
                feature_plan_df = analysis_df[candidate_feature_columns + [primary_target]].copy()
                feature_plan = _get_feature_engineering_plan(
                    df=feature_plan_df,
                    target=primary_target,
                    settings=settings,
                    user_brief=planner_brief,
                )
            with st.expander("Feature engineering plan", expanded=True):
                _render_feature_engineering_plan(artifact_to_dict(feature_plan))

    current_experiment_signature = _experiment_signature(
        dataset_fingerprint=current_dataset_fingerprint,
        target_columns=target_columns,
        task_type_choice=task_type_choice,
        time_budget=int(time_budget),
        excluded_columns=excluded_columns,
        test_size=float(test_size),
        high_missing_threshold=float(high_missing_threshold),
        random_state=int(random_state),
        priority_metric_choice=priority_metric_choice,
        planner_brief=planner_brief,
        feature_plan=feature_plan,
        manual_cleaning_plan=manual_cleaning_plan,
    )

    priority_metrics = {
        target: resolve_priority_metric(_target_task_types(analysis_df, [target], task_type_choice)[target], priority_metric_choice)
        for target in target_columns
    }
    preflight_by_target: dict[str, object] = {}
    for target in target_columns:
        preflight_input = analysis_df[[column for column in analysis_df.columns if column not in target_columns or column == target]].copy()
        if manual_cleaning_plan is not None and any(rule.enabled for rule in manual_cleaning_plan.rules) and manual_cleaning_plan.effect_stage == "pre_training":
            preflight_input, _, _ = apply_manual_cleaning_plan(
                preflight_input,
                manual_cleaning_plan,
                target,
                protected_columns=target_columns,
            )
        preflight_by_target[target] = validate_preflight(
            df=preflight_input,
            target=target,
            task_type=task_types[target],
            excluded_columns=[],
            priority_metric=priority_metric_choice,
            high_missing_threshold=float(high_missing_threshold),
        )
    failing_targets = [target for target, validation in preflight_by_target.items() if not validation.ok_to_run]
    if failing_targets:
        _render_step_status(
            "The app found blocking issues in the current setup.",
            f"Fix the checks for: {', '.join(failing_targets)} before starting training.",
            level="warning",
        )
    else:
        _render_step_status(
            "The dataset and target selection passed the current checks.",
            "You can start training after this review, or adjust the setup first.",
            level="success",
        )
    st.write("Preflight validation")
    _section_caption("This check looks for blocking issues before training, such as missing target values or no usable feature columns.")
    with st.expander("Preflight validation", expanded=True):
        for target in target_columns:
            validation = artifact_to_dict(preflight_by_target[target])
            st.write(f"{target} ({task_types[target]})")
            st.caption(f"Resolved priority metric: {priority_metrics[target]}")
            _render_issue_table(validation.get("issues", []))

    st.write("Data preview")
    _section_caption("First 50 rows are shown for a quick sanity check before training.")
    if manual_cleaning_plan is not None and any(rule.enabled for rule in manual_cleaning_plan.rules) and manual_cleaning_plan.effect_stage == "pre_training":
        st.info("Manual cleaning rules are set to apply only before training. The data preview and EDA below still show the pre-cleaning analysis subset.")
    st.dataframe(analysis_df.head(50), use_container_width=True)

    st.write("EDA summary")
    _section_caption("EDA means a quick health check for the dataset: shape, duplicates, missing values, correlations, and target behavior.")
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

    st.subheader("4. Start training")
    _section_caption("Launch the local baseline after you have reviewed the target, checks, and data summary.")
    if failing_targets:
        _render_step_status(
            "Training is blocked by validation issues.",
            f"Resolve the flagged issues for: {', '.join(failing_targets)}.",
            level="warning",
        )
    else:
        _render_step_status(
            "The run is ready to start.",
            "Click Run training to build the local baseline and unlock the results step.",
            level="success",
        )
    results: list[dict[str, object]] = []
    if st.session_state.get("_latest_results_signature") == current_experiment_signature:
        results = list(st.session_state.get("_latest_results", []))

    if st.button("Run training", type="primary"):
        results = []
        with st.spinner("Cleaning data and training model locally..."):
            if failing_targets:
                logger.error("Blocking preflight issues targets=%s", failing_targets)
                st.error(f"Resolve blocking preflight issues before training: {', '.join(failing_targets)}")
                return

            feature_columns = candidate_feature_columns
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
                tracker = DataFlowTracker(target=target)
                tracker.snapshot_dataframe(
                    "raw_dataset",
                    "Raw dataset",
                    "intake",
                    df,
                    preview=True,
                    metadata={"source_columns": list(df.columns)},
                )
                tracker.snapshot_dataframe(
                    "analysis_subset",
                    "Analysis subset",
                    "intake",
                    base_analysis_df,
                    metadata={"excluded_columns": excluded_columns},
                )
                if manual_cleaning_plan is not None and any(rule.enabled for rule in manual_cleaning_plan.rules) and manual_cleaning_plan.effect_stage == "pre_eda":
                    tracker.snapshot_dataframe(
                        "after_manual_cleaning_pre_eda",
                        "After manual cleaning (EDA + training)",
                        "intake",
                        analysis_df,
                        metadata=analysis_manual_cleaning_impact,
                    )

                target_df = analysis_df[feature_columns + [target]].copy()
                tracker.snapshot_dataframe(
                    "target_dataset",
                    "Target dataset",
                    "intake",
                    target_df,
                    preview=True,
                    metadata={"target": target, "feature_columns": feature_columns},
                )
                training_input_df = target_df
                manual_cleaning_log = list(analysis_manual_cleaning_log)
                manual_cleaning_impact = dict(analysis_manual_cleaning_impact)
                if manual_cleaning_plan is not None and any(rule.enabled for rule in manual_cleaning_plan.rules) and manual_cleaning_plan.effect_stage == "pre_training":
                    target_manual_plan = validate_manual_cleaning_plan(
                        manual_cleaning_plan,
                        target_df,
                        target,
                        protected_columns=target_columns,
                    )
                    training_input_df, manual_cleaning_log, manual_cleaning_impact = apply_manual_cleaning_plan(
                        target_df,
                        target_manual_plan,
                        target,
                        protected_columns=target_columns,
                    )
                    tracker.snapshot_dataframe(
                        "after_manual_cleaning_pre_training",
                        "After manual cleaning (training only)",
                        "intake",
                        training_input_df,
                        metadata=manual_cleaning_impact,
                    )

                target_eda_summary = generate_eda_summary(target_df, target=target)
                config = CleanConfig(
                    target=target,
                    task_type=task_type,
                    test_size=float(test_size),
                    random_state=int(random_state),
                    high_missing_threshold=float(high_missing_threshold),
                    feature_engineering_operations=_feature_plan_operations(feature_plan) if feature_plan else None,
                )
                cleaned = clean_and_split(training_input_df, config, tracker=tracker)
                if manual_cleaning_log:
                    cleaned.cleaning_log = manual_cleaning_log + cleaned.cleaning_log
                trained = train_model(cleaned, time_budget=int(time_budget), metric_preference=priority_metric, tracker=tracker)
                metrics, prediction_sample = evaluate_model(trained.model, cleaned, task_type=task_type, tracker=tracker)
                tracker.snapshot_artifact(
                    "metrics_summary",
                    "Metrics summary",
                    "evaluation",
                    metadata={"metrics": metrics, "priority_metric": priority_metric},
                )
                planned_report_mode = "llm" if settings.llm_enabled else "rule_based"
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
                        "feature_engineering_enabled": bool(feature_plan),
                        "manual_cleaning_enabled": bool(manual_cleaning_plan and any(rule.enabled for rule in manual_cleaning_plan.rules)),
                        "manual_cleaning_effect_stage": manual_cleaning_plan.effect_stage if manual_cleaning_plan else None,
                        "dataset_fingerprint": current_dataset_fingerprint,
                    }
                )
                storage.save_json(run, "plan.json", artifact_to_dict(plan_suggestion))
                if feature_plan:
                    storage.save_json(run, "feature_engineering_plan.json", artifact_to_dict(feature_plan))
                if manual_cleaning_plan:
                    storage.save_json(run, "manual_cleaning_plan.json", artifact_to_dict(manual_cleaning_plan))
                storage.save_json(run, "eda_summary.json", target_eda_summary)
                storage.save_json(run, "cleaning_log.json", cleaned.cleaning_log)
                storage.save_json(run, "metrics.json", metrics)
                data_flow_payload = artifact_to_dict(tracker.to_trace())
                storage.save_json(run, "data_flow.json", data_flow_payload)
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
                        "manual_cleaning_effect_stage": manual_cleaning_plan.effect_stage if manual_cleaning_plan else None,
                        "manual_cleaning_applied_rules": 0
                        if not manual_cleaning_plan
                        else sum(1 for rule in manual_cleaning_plan.rules if rule.enabled),
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
                        "data_flow": data_flow_payload,
                        "report_path": report_path,
                        "prediction_path": prediction_path,
                        "model_path": model_path,
                    }
                )

        st.session_state["_latest_results_signature"] = current_experiment_signature
        st.session_state["_latest_results"] = list(results)
        completed_ids = ", ".join(str(result["run"].run_id) for result in results)
        logger.info("Training pipeline complete run_ids=%s", completed_ids)
        st.success(f"Run completed: {completed_ids}")

    if results:
        _render_run_outputs(results)


if __name__ == "__main__":
    main()
