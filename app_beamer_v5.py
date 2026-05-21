from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urlencode

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import streamlit as st
import altair as alt

from ml_platform.artifacts import artifact_to_dict
from ml_platform.automl import train_model
from ml_platform.cleaning import (
    clean_and_split,
    prepare_for_training,
    preprocessing_plan_to_clean_config,
)
from ml_platform.config import Settings, load_settings
from ml_platform.data_flow import DataFlowTracker
from ml_platform.data_io import read_csv
from ml_platform.eda import (
    build_distribution_overview,
    build_distribution_plot_data,
    choose_default_distribution_feature,
    generate_eda_summary,
)
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
from ml_platform.ui_i18n import translate_ui_text
from ml_platform.validation import (
    build_recommendations,
    resolve_priority_metric,
    validate_postrun,
    validate_preflight,
)

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
    PROJECT_ROOT / "彩虹.jpg",
    PROJECT_ROOT / "data" / "hero.jpg",
    PROJECT_ROOT / "assets" / "hero.jpg",
    PROJECT_ROOT / "assets" / "hero.jpeg",
    PROJECT_ROOT / "assets" / "hero.png",
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
PREPROCESSING_PLAN_VERSION = 2
AUTOGLUON_DEFAULT_PRESETS = "medium_quality"
AUTOGLUON_FEATURE_GENERATOR_DEFAULTS = {
    "enable_numeric_features": True,
    "enable_categorical_features": True,
    "enable_datetime_features": True,
    "enable_text_special_features": True,
    "enable_text_ngram_features": True,
    "enable_raw_text_features": False,
    "enable_vision_features": False,
}
DEFAULT_UI_LANGUAGE = "zh-CN"
UI_LANGUAGE_OPTIONS = ("en", "zh-CN")
UI_LANGUAGE_LABELS = {
    "en": "English",
    "zh-CN": "简体中文",
}
BEAMER_SANS_FONT_STACK = (
    '"LXGW WenKai", "LXGW WenKai GB", "LXGW WenKai Screen", '
    '"Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif'
)
BEAMER_MONO_FONT_STACK = (
    '"SFMono-Regular", "SF Mono", "Cascadia Mono", "JetBrains Mono", Menlo, '
    'Consolas, "Liberation Mono", monospace'
)
PREPROCESSING_STEP_IDS = {
    "column_selection": "column_selection",
    "manual_cleaning": "manual_cleaning",
    "missing_value": "missing_value",
    "categorical_encoding": "categorical_encoding",
    "numeric_scaling": "numeric_scaling",
    "autogluon_feature_generator": "autogluon_feature_generator",
    "feature_engineering": "feature_engineering",
}
WIZARD_STEPS = (
    "upload",
    "target",
    "check",
    "prepare",
    "train",
    "results",
)
WIZARD_STEP_LABELS = {
    "upload": "Upload data",
    "target": "Choose target",
    "check": "Data preprocessing",
    "prepare": "Prepare training",
    "train": "Start training",
    "results": "Review results",
}

BEAMER_SECTION_LABELS = {
    "dataset": "Dataset",
    "task": "Task",
    "preprocess": "Preprocess",
    "training": "Training",
    "results": "Results",
}

BEAMER_NAV_SECTIONS = {
    "dataset": ("Source", "Schema", "Profile"),
    "task": ("Target", "Metric", "Budget"),
    "preprocess": (
        "Field health",
        "Preprocessing details",
        "EDA profile",
        "Preflight validation",
    ),
    "training": ("Prepare batches", "Fit models", "Save artifacts"),
    "results": ("Summary", "Metrics", "Validation", "Importance", "Downloads"),
}

BEAMER_STEP_TO_SECTION_FRAME = {
    "upload": ("dataset", 0),
    "target": ("task", 0),
    "check": ("preprocess", 0),
    "prepare": ("training", 0),
    "train": ("training", 1),
    "results": ("results", 0),
}

BEAMER_NAV_STEP_TARGETS = {
    "dataset": ("upload", "upload", "upload"),
    "task": ("target", "target", "target"),
    "preprocess": ("check", "check", "check", "check"),
    "training": ("prepare", "train", "train"),
    "results": ("results", "results", "results", "results", "results"),
}

SLIDE_TITLES = {
    "upload": "Dataset Upload",
    "target": "Task Definition",
    "check": "Data Preprocessing",
    "prepare": "Training Preparation",
    "train": "Model Training",
    "results": "Evaluation & Export",
}

SLIDE_SUBTITLES = {
    "upload": "Load a dataset and unlock schema inspection.",
    "target": "Choose the outcome, task type, metric, and first-run budget.",
    "check": "Move through field health, preprocessing details, and preflight validation as separate frames.",
    "prepare": "Materialize the applied preprocessing plan before model fitting.",
    "train": "Run AutoML training and collect artifacts in a reproducible run folder.",
    "results": "Compare metrics, inspect validation notes, and download artifacts.",
}


def _ui_language() -> str:
    return str(st.session_state.get("ui_language", DEFAULT_UI_LANGUAGE))


def _t(text: str, **kwargs: Any) -> str:
    return translate_ui_text(_ui_language(), text, **kwargs)


def _render_language_switcher() -> None:
    topbar_cols = st.columns([0.82, 0.18])
    with topbar_cols[1]:
        st.caption(_t("Interface language"))
        current_language = _ui_language()
        if current_language not in UI_LANGUAGE_OPTIONS:
            current_language = DEFAULT_UI_LANGUAGE
        st.selectbox(
            _t("Interface language"),
            UI_LANGUAGE_OPTIONS,
            index=UI_LANGUAGE_OPTIONS.index(current_language),
            key="ui_language",
            label_visibility="collapsed",
            format_func=lambda code: UI_LANGUAGE_LABELS.get(code, code),
        )


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


def _preprocessing_step(
    kind: str,
    *,
    enabled: bool = True,
    params: dict[str, Any] | None = None,
    summary: str = "",
    execution_mode: str = "materialize_before_training",
) -> dict[str, Any]:
    return {
        "id": PREPROCESSING_STEP_IDS[kind],
        "kind": kind,
        "enabled": enabled,
        "params": dict(params or {}),
        "summary": summary,
        "execution_mode": execution_mode,
    }


def _default_preprocessing_plan() -> dict[str, Any]:
    steps = [
        _preprocessing_step(
            "column_selection",
            params={"excluded_columns": []},
            summary="No excluded columns.",
        ),
        _preprocessing_step(
            "manual_cleaning",
            enabled=False,
            params={"plan": None},
            summary="No manual cleaning rules applied.",
        ),
        _preprocessing_step(
            "missing_value",
            params={
                "high_missing_threshold": 0.9,
            },
            summary="Auto-drop high-missing columns above 0.90.",
        ),
        _preprocessing_step(
            "autogluon_feature_generator",
            params=dict(AUTOGLUON_FEATURE_GENERATOR_DEFAULTS),
            summary="AutoGluon feature generation is enabled for numeric, categorical, datetime, and text features.",
        ),
        _preprocessing_step(
            "feature_engineering",
            enabled=False,
            params={
                "planner_name": "local_whitelist",
                "operations": [],
                "rejected_operations": [],
                "notes": [],
            },
            summary="Feature engineering is disabled.",
        ),
    ]
    return {
        "version": PREPROCESSING_PLAN_VERSION,
        "global_params": {
            "test_size": 0.2,
            "random_state": 42,
            "autogluon_presets": AUTOGLUON_DEFAULT_PRESETS,
        },
        "steps": steps,
        "applied_step_ids": [str(step["id"]) for step in steps],
        "notes": [],
    }


def _preprocessing_steps(plan_data: Any) -> list[dict[str, Any]]:
    payload = (
        artifact_to_dict(plan_data) if not isinstance(plan_data, dict) else plan_data
    )
    return [step for step in payload.get("steps", []) if isinstance(step, dict)]


def _preprocessing_step_payload(plan_data: Any, kind: str) -> dict[str, Any] | None:
    for step in _preprocessing_steps(plan_data):
        if str(step.get("kind")) == kind:
            return step
    return None


def _preprocessing_step_params(plan_data: Any, kind: str) -> dict[str, Any]:
    step = _preprocessing_step_payload(plan_data, kind)
    if not isinstance(step, dict):
        return {}
    params = step.get("params", {})
    return dict(params) if isinstance(params, dict) else {}


def _preprocessing_step_enabled(plan_data: Any, kind: str) -> bool:
    step = _preprocessing_step_payload(plan_data, kind)
    return bool(step and step.get("enabled", True))


def _autogluon_feature_generator_params(raw_params: Any = None) -> dict[str, bool]:
    resolved = dict(AUTOGLUON_FEATURE_GENERATOR_DEFAULTS)
    if isinstance(raw_params, dict):
        for key in resolved:
            if key in raw_params:
                resolved[key] = bool(raw_params[key])
    return resolved


def _autogluon_enabled_feature_names(params: dict[str, bool]) -> list[str]:
    labels = {
        "enable_numeric_features": "numeric",
        "enable_categorical_features": "categorical",
        "enable_datetime_features": "datetime",
        "enable_text_special_features": "text_special",
        "enable_text_ngram_features": "text_ngram",
        "enable_raw_text_features": "raw_text",
        "enable_vision_features": "vision",
    }
    return [label for key, label in labels.items() if bool(params.get(key))]


def _upsert_preprocessing_step(
    plan_data: dict[str, Any], step_data: dict[str, Any]
) -> dict[str, Any]:
    updated = json.loads(json.dumps(plan_data, ensure_ascii=False))
    steps = [step for step in updated.get("steps", []) if isinstance(step, dict)]
    replaced = False
    for index, existing in enumerate(steps):
        if str(existing.get("kind")) == str(step_data.get("kind")):
            steps[index] = step_data
            replaced = True
            break
    if not replaced:
        steps.append(step_data)
    updated["steps"] = steps
    updated["applied_step_ids"] = [
        str(step.get("id")) for step in steps if str(step.get("id", "")).strip()
    ]
    return updated


def _update_applied_preprocessing_step(step_data: dict[str, Any]) -> None:
    applied_plan = (
        st.session_state.get("_preprocessing_plan_applied")
        or _default_preprocessing_plan()
    )
    st.session_state["_preprocessing_plan_applied"] = _upsert_preprocessing_step(
        applied_plan, step_data
    )


def _preprocessing_plan_global_params(plan_data: Any) -> dict[str, Any]:
    payload = (
        artifact_to_dict(plan_data) if not isinstance(plan_data, dict) else plan_data
    )
    params = payload.get("global_params", {})
    return dict(params) if isinstance(params, dict) else {}


def _set_applied_preprocessing_global_params(
    *, test_size: float, random_state: int
) -> None:
    applied_plan = (
        st.session_state.get("_preprocessing_plan_applied")
        or _default_preprocessing_plan()
    )
    updated = json.loads(json.dumps(applied_plan, ensure_ascii=False))
    updated["global_params"] = {
        "test_size": float(test_size),
        "random_state": int(random_state),
        "autogluon_presets": str(
            _preprocessing_plan_global_params(applied_plan).get(
                "autogluon_presets", AUTOGLUON_DEFAULT_PRESETS
            )
        ),
    }
    st.session_state["_preprocessing_plan_applied"] = updated


def _json_equal(left: Any, right: Any) -> bool:
    return json.dumps(
        artifact_to_dict(left), sort_keys=True, ensure_ascii=True
    ) == json.dumps(artifact_to_dict(right), sort_keys=True, ensure_ascii=True)


def _html_escape(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def _safe_widget_key(*parts: object) -> str:
    raw = "_".join(str(part) for part in parts if str(part).strip())
    return "w_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _render_slide_title(step: str, title: str | None = None, subtitle: str | None = None) -> None:
    resolved_title = _t(title or SLIDE_TITLES.get(step, WIZARD_STEP_LABELS.get(step, step)))
    resolved_subtitle = _t(subtitle or SLIDE_SUBTITLES.get(step, ""))
    subtitle_html = ""
    if resolved_subtitle:
        subtitle_html = '<div class="beamer-slide-subtitle">' + _html_escape(resolved_subtitle) + '</div>'
    st.markdown(
        '<section class="beamer-slide-titlebar">'
        + '<div class="beamer-slide-title">' + _html_escape(resolved_title) + '</div>'
        + subtitle_html
        + '</section>',
        unsafe_allow_html=True,
    )


def _render_explanation_strip(text: str) -> None:
    st.markdown(
        '<div class="beamer-explanation-strip">' + _html_escape(_t(text)) + '</div>',
        unsafe_allow_html=True,
    )


def _section_title(text: str) -> None:
    st.markdown(
        '<div class="beamer-section-title">' + _html_escape(str(text)) + "</div>",
        unsafe_allow_html=True,
    )


def _panel_title(text: str) -> None:
    st.markdown(
        '<div class="beamer-panel-title">' + _html_escape(str(text)) + "</div>",
        unsafe_allow_html=True,
    )


def _frame_note(text: str, *, tone: str = "info") -> None:
    tone_class = {
        "info": "beamer-frame-note-info",
        "warning": "beamer-frame-note-warning",
        "success": "beamer-frame-note-success",
    }.get(tone, "beamer-frame-note-info")
    st.markdown(
        '<div class="beamer-frame-note ' + tone_class + '">'
        + _html_escape(str(text))
        + "</div>",
        unsafe_allow_html=True,
    )


def _render_task_type_card(label: str, value: str) -> None:
    st.markdown(
        '<div class="beamer-task-type-card">'
        + '<div class="beamer-task-type-label">'
        + _html_escape(str(label))
        + '</div><div class="beamer-task-type-value">'
        + _html_escape(str(value))
        + "</div></div>",
        unsafe_allow_html=True,
    )


def _load_hero_background() -> str:
    hero_image_path = next(
        (path for path in HERO_IMAGE_CANDIDATES if path.exists()), None
    )
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
            @import url("https://fonts.googleapis.com/css2?family=LXGW+WenKai+Mono+TC&display=swap");

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
                font-family: "LXGW WenKai Mono", "LXGW WenKai Mono GB", "LXGW WenKai Mono TC", "霞鹜文楷等宽", monospace !important;
            }

            button,
            input,
            textarea,
            select,
            [data-baseweb="select"] > div {
                font-family: "LXGW WenKai Mono", "LXGW WenKai Mono GB", "LXGW WenKai Mono TC", "霞鹜文楷等宽", monospace !important;
            }

            .stApp {
                color: var(--lab-ink);
                background:
                    linear-gradient(180deg, #ffffff 0, #f7f8fb 34rem, var(--lab-bg) 100%),
                    var(--lab-bg);
                font-family: "LXGW WenKai Mono", "LXGW WenKai Mono GB", "LXGW WenKai Mono TC", "霞鹜文楷等宽", monospace;
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
                font-family: "LXGW WenKai Mono", "LXGW WenKai Mono GB", "LXGW WenKai Mono TC", "霞鹜文楷等宽", monospace;
                font-weight: 800;
                letter-spacing: 0;
            }

            h1, h2, h3 {
                color: var(--lab-ink);
                font-family: "LXGW WenKai Mono", "LXGW WenKai Mono GB", "LXGW WenKai Mono TC", "霞鹜文楷等宽", monospace;
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
                font-family: "LXGW WenKai Mono", "LXGW WenKai Mono GB", "LXGW WenKai Mono TC", "霞鹜文楷等宽", monospace;
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
                font-family: "LXGW WenKai Mono", "LXGW WenKai Mono GB", "LXGW WenKai Mono TC", "霞鹜文楷等宽", monospace;
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
                font-family: "LXGW WenKai Mono", "LXGW WenKai Mono GB", "LXGW WenKai Mono TC", "霞鹜文楷等宽", monospace;
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
                font-family: "LXGW WenKai Mono", "LXGW WenKai Mono GB", "LXGW WenKai Mono TC", "霞鹜文楷等宽", monospace;
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
                font-family: "LXGW WenKai Mono", "LXGW WenKai Mono GB", "LXGW WenKai Mono TC", "霞鹜文楷等宽", monospace;
            }

            [data-testid="stMetricValue"] {
                color: var(--lab-accent);
                font-family: "LXGW WenKai Mono", "LXGW WenKai Mono GB", "LXGW WenKai Mono TC", "霞鹜文楷等宽", monospace;
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



            .beamer-training-progress {
                margin: 0.8rem 0 1rem;
                padding: 0.9rem 1rem;
                border: 1px solid var(--beamer-line);
                background: var(--beamer-paper);
                border-radius: 7px;
            }

            .beamer-preprocess-frame-note {
                margin: 0.4rem 0 1rem;
                color: var(--beamer-muted);
                font-size: 0.9rem;
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



def _apply_beamer_design() -> None:
    """Minimal beamer-like layer on top of the existing Streamlit app."""
    st.markdown(
        """
        <style>
            @import url("https://fonts.googleapis.com/css2?family=LXGW+WenKai&family=Noto+Sans+SC:wght@400;500;600;700&display=swap");

            :root {
                --beamer-bg: #FAFAF8;
                --beamer-paper: #FFFFFF;
                --beamer-titlebar: #E9E3DC;
                --beamer-blue: #1F4EAA;
                --beamer-ink: #222222;
                --beamer-muted: #78829A;
                --beamer-faint: #B8C0D4;
                --beamer-line: #D8D3CC;
                --beamer-burgundy: #7A0019;
                --beamer-sans: __BEAMER_SANS__;
                --beamer-mono: __BEAMER_MONO__;
                --beamer-fs-body: 14px;
                --beamer-fs-meta: 12px;
                --beamer-fs-panel: 18px;
                --beamer-fs-slide: 24px;
                --beamer-fs-hero: 28px;
            }

            html, body, .stApp, .stApp *,
            button, input, textarea, select, option, label, p, span, div,
            h1, h2, h3, h4, h5, h6, code, pre, table, th, td,
            [data-testid], [data-baseweb], [data-baseweb="select"] > div {
                font-family: var(--beamer-sans) !important;
                letter-spacing: 0 !important;
            }

            code,
            pre,
            kbd,
            samp,
            [data-testid="stMetricValue"],
            [data-testid="stCodeBlock"] * {
                font-family: var(--beamer-mono) !important;
            }

            .material-symbols-rounded,
            .material-symbols-outlined,
            .material-icons,
            span[class*="material-symbols"],
            span[class*="material-icons"],
            [data-testid="stIconMaterial"],
            [data-testid="stIconMaterial"] * {
                font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
                font-weight: normal !important;
                font-style: normal !important;
                line-height: 1 !important;
                letter-spacing: normal !important;
                text-transform: none !important;
                white-space: nowrap !important;
                word-wrap: normal !important;
                direction: ltr !important;
                -webkit-font-feature-settings: "liga" !important;
                -webkit-font-smoothing: antialiased !important;
            }

            .stApp {
                color: var(--beamer-ink);
                background: var(--beamer-bg) !important;
            }

            .stApp::before {
                background-image: radial-gradient(rgba(31, 78, 170, 0.11) 1px, transparent 1px) !important;
                background-size: 22px 22px !important;
                opacity: 0.32 !important;
                mask-image: linear-gradient(to bottom, black, transparent 82%) !important;
            }

            .block-container {
                max-width: 1120px !important;
                padding-top: 1.1rem !important;
                padding-bottom: 3rem !important;
            }

            [data-testid="stHeader"],
            [data-testid="stToolbar"],
            [data-testid="stDecoration"],
            [data-testid="stStatusWidget"],
            .stDeployButton,
            footer {
                display: none !important;
                visibility: hidden !important;
            }

            h1, h2, h3 {
                border: 0 !important;
                padding-top: 0 !important;
                margin-top: 0 !important;
                color: var(--beamer-ink) !important;
            }

            p,
            li,
            label,
            .stMarkdown,
            [data-testid="stCaptionContainer"] {
                color: var(--beamer-ink) !important;
                font-size: var(--beamer-fs-body) !important;
                line-height: 1.6 !important;
            }

            .beamer-app-headline {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 1rem;
                margin: 0 0 0.75rem;
                padding: 0.2rem 0 0.55rem;
            }

            .beamer-app-title {
                color: var(--beamer-blue);
                font-size: clamp(1.6rem, 2vw, 1.75rem);
                line-height: 1.25;
                font-weight: 700;
            }

            .beamer-app-subtitle {
                color: var(--beamer-muted);
                font-size: var(--beamer-fs-meta);
                line-height: 1.45;
                margin-top: 0.2rem;
            }

            .beamer-nav {
                width: 100%;
                display: grid;
                grid-template-columns: repeat(5, 1fr);
                align-items: stretch;
                border: 2px solid #111111;
                background: var(--beamer-paper);
                margin: 0.25rem 0 0;
                box-shadow: none;
            }

            .beamer-nav-section {
                padding: 0.5rem 0.8rem;
                min-width: 0;
                border-right: 1px solid var(--beamer-line);
            }

            .beamer-nav-section:last-child {
                border-right: 0;
            }

            .beamer-nav-title {
                color: #9DA7BF;
                font-size: var(--beamer-fs-meta);
                line-height: 1.45;
                white-space: normal;
                overflow: visible;
                text-overflow: clip;
                margin-bottom: 0.2rem;
            }

            .beamer-nav-title.active {
                color: var(--beamer-blue);
                font-weight: 700;
            }

            .beamer-dots {
                display: flex;
                align-items: center;
                gap: 4px;
                min-height: 10px;
                flex-wrap: wrap;
            }

            .beamer-dot {
                width: 6px;
                height: 6px;
                border-radius: 999px;
                border: 1px solid #A9B3CA;
                background: transparent;
                box-sizing: border-box;
            }

            .beamer-dot-link {
                display: inline-block;
                text-decoration: none !important;
                cursor: pointer;
            }

            .beamer-dot.done {
                background: #A9B3CA;
            }

            .beamer-dot.current {
                border-color: var(--beamer-blue);
                background: var(--beamer-blue);
            }

            .beamer-slide-titlebar {
                background: var(--beamer-titlebar);
                border-left: 6px solid var(--beamer-blue);
                padding: 0.9rem 1.05rem 0.85rem;
                margin: 0 0 1.35rem;
            }

            .beamer-slide-title {
                color: var(--beamer-blue);
                font-size: var(--beamer-fs-slide);
                line-height: 1.25;
                font-weight: 700;
            }

            .beamer-slide-subtitle {
                color: var(--beamer-muted);
                font-size: var(--beamer-fs-meta);
                line-height: 1.45;
                margin-top: 0.3rem;
            }

            .beamer-help-strip,
            .beamer-explanation-strip,
            .beamer-frame-note {
                margin: 0.8rem 0 1rem;
                border-left: 5px solid var(--beamer-burgundy);
                background: #F0EEEA;
                padding: 0.8rem 1rem;
                color: #333333;
                font-size: var(--beamer-fs-meta);
                line-height: 1.45;
            }

            .beamer-frame-note-info {
                border-left-color: var(--beamer-blue);
                background: #F4F7FD;
            }

            .beamer-frame-note-warning {
                border-left-color: #A24C26;
                background: #FAF2EC;
            }

            .beamer-frame-note-success {
                border-left-color: #25684A;
                background: #EEF7F2;
            }

            .beamer-jump-caption {
                color: var(--beamer-muted);
                font-size: var(--beamer-fs-meta);
                line-height: 1.45;
                margin: 0.45rem 0 0.35rem;
            }

            .beamer-section-title,
            .beamer-panel-title {
                margin: 0 0 0.45rem;
                color: var(--beamer-ink);
                font-size: var(--beamer-fs-panel);
                line-height: 1.25;
                font-weight: 700;
            }

            .beamer-section-caption {
                margin: 0 0 0.8rem;
                color: var(--beamer-muted);
                font-size: var(--beamer-fs-meta);
                line-height: 1.45;
            }

            .beamer-task-type-card {
                border: 1px solid var(--beamer-line);
                border-radius: 6px;
                background: #FFFFFF;
                padding: 0.85rem 1rem;
                margin: 0.1rem 0 1rem;
            }

            .beamer-task-type-label {
                color: var(--beamer-muted);
                font-size: var(--beamer-fs-meta);
                line-height: 1.45;
                margin-bottom: 0.3rem;
            }

            .beamer-task-type-value {
                color: var(--beamer-blue);
                font-size: var(--beamer-fs-panel);
                line-height: 1.25;
                font-weight: 700;
            }

            .beamer-frame-selector-title,
            .beamer-frame-progress {
                color: var(--beamer-muted);
                font-size: var(--beamer-fs-meta);
                line-height: 1.45;
            }

            .beamer-frame-selector-title {
                margin: 0 0 0.55rem;
                color: var(--beamer-blue);
                font-weight: 600;
            }

            .beamer-frame-progress {
                text-align: center;
                padding-top: 0.45rem;
            }

            .beamer-empty {
                padding: 1rem 1.05rem;
                border: 1px dashed var(--beamer-line);
                background: #FFFFFF;
                color: var(--beamer-muted);
                border-radius: 6px;
                font-size: var(--beamer-fs-body);
                line-height: 1.6;
            }

            [data-testid="stMetric"],
            [data-testid="stDataFrame"],
            [data-testid="stJson"],
            [data-testid="stExpander"],
            [data-testid="stFileUploader"] section {
                border: 1px solid var(--beamer-line) !important;
                background: var(--beamer-paper) !important;
                border-radius: 6px !important;
                box-shadow: none !important;
            }

            div[data-testid="stVerticalBlockBorderWrapper"] {
                border: 1px solid var(--beamer-line) !important;
                background: var(--beamer-paper) !important;
                border-radius: 6px !important;
                padding: 0.18rem 0.32rem !important;
                box-shadow: none !important;
            }

            div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlockBorderWrapper"] {
                border: 0 !important;
                background: transparent !important;
                padding: 0 !important;
                box-shadow: none !important;
            }

            [data-testid="stMetricValue"] {
                color: var(--beamer-blue) !important;
                font-weight: 700 !important;
            }

            [data-testid="stMetricLabel"],
            [data-testid="stCaptionContainer"] {
                color: var(--beamer-muted);
                font-size: var(--beamer-fs-meta) !important;
                line-height: 1.45 !important;
            }

            button[kind="primary"],
            .stDownloadButton button {
                background: var(--beamer-blue) !important;
                border-color: var(--beamer-blue) !important;
                color: #FFFFFF !important;
                border-radius: 4px !important;
                box-shadow: none !important;
                transform: none !important;
                min-height: 2.25rem !important;
                font-size: var(--beamer-fs-body) !important;
            }

            div[data-testid="stDownloadButton"] button,
            div[data-testid="stDownloadButton"] button * {
                background: var(--beamer-blue) !important;
                border-color: var(--beamer-blue) !important;
                color: #FFFFFF !important;
                opacity: 1 !important;
            }

            button[kind="secondary"] {
                color: var(--beamer-blue) !important;
                background: #FFFFFF !important;
                border: 1px solid #D4D9E8 !important;
                border-radius: 4px !important;
                box-shadow: none !important;
                min-height: 2.25rem !important;
                font-size: var(--beamer-fs-body) !important;
            }

            button[kind="primary"] *,
            .stDownloadButton button * {
                color: #FFFFFF !important;
            }

            input, textarea, [data-baseweb="select"] > div {
                border-radius: 4px !important;
                border-color: #CCD2E0 !important;
                background: #FFFFFF !important;
                font-size: var(--beamer-fs-body) !important;
            }

            .stTabs [data-baseweb="tab-list"] {
                border-bottom: 1px solid var(--beamer-line) !important;
                gap: 0.4rem !important;
            }

            .stTabs [data-baseweb="tab"] {
                border-radius: 4px 4px 0 0 !important;
                box-shadow: none !important;
                font-size: var(--beamer-fs-body) !important;
                line-height: 1.45 !important;
            }

            .beamer-bottom-shell {
                margin-top: 2.2rem;
                padding: 0.75rem 0 0.15rem;
                border-top: 1px solid var(--beamer-line);
            }

            .beamer-bottom-caption {
                text-align: center;
                color: var(--beamer-muted);
                font-size: var(--beamer-fs-meta);
                line-height: 1.45;
                padding-top: 0.35rem;
            }

            .beamer-roadmap {
                display: grid;
                grid-template-columns: repeat(5, minmax(0, 1fr));
                gap: 0.55rem;
                margin: -0.35rem 0 1.25rem;
            }

            .beamer-roadmap-card {
                background: #FFFFFF;
                border: 1px solid var(--beamer-line);
                padding: 0.65rem 0.72rem;
                min-height: 72px;
                border-radius: 6px;
            }

            .beamer-roadmap-index {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 1.15rem;
                height: 1.15rem;
                background: var(--beamer-blue);
                color: #FFFFFF;
                font-size: var(--beamer-fs-meta);
                margin-bottom: 0.35rem;
            }

            .beamer-roadmap-title {
                color: var(--beamer-ink);
                font-weight: 700;
                font-size: var(--beamer-fs-meta);
                line-height: 1.45;
            }

            .beamer-roadmap-note {
                color: var(--beamer-muted);
                font-size: var(--beamer-fs-meta);
                line-height: 1.45;
                margin-top: 0.25rem;
            }

            @media (max-width: 860px) {
                .beamer-nav {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
                .beamer-app-headline {
                    flex-direction: column;
                }
                .beamer-roadmap {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }

            @media (max-width: 560px) {
                .beamer-nav {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """.replace("__BEAMER_SANS__", BEAMER_SANS_FONT_STACK).replace(
            "__BEAMER_MONO__", BEAMER_MONO_FONT_STACK
        ),
        unsafe_allow_html=True,
    )

def _render_hero() -> None:
    hero_description = _t(
        "Upload a tabular dataset, inspect data quality, train a local baseline, and export reproducible artifacts."
    )
    st.markdown(
        '<section class="beamer-app-headline">'
        + '<div><div class="beamer-app-title">ML Platform</div>'
        + '<div class="beamer-app-subtitle">' + _html_escape(hero_description) + '</div></div>'
        + '<div class="beamer-app-subtitle">' + _html_escape(_t("Guided AutoML workflow")) + '</div>'
        + '</section>',
        unsafe_allow_html=True,
    )


def _render_help_center() -> None:
    st.markdown(
        '<div class="beamer-help-strip">'
        + _html_escape(_t("This interface is organized as a beamer-style guided run: one section, one task, one compact explanation. Advanced settings stay folded until needed."))
        + '</div>',
        unsafe_allow_html=True,
    )



def _section_caption(text: str) -> None:
    st.markdown(
        '<p class="beamer-section-caption">' + _html_escape(str(text)) + "</p>",
        unsafe_allow_html=True,
    )


def _render_step_status(
    current_action: str, next_action: str, level: str = "info"
) -> None:
    message = _t(
        "Now: {current_action} Next: {next_action}",
        current_action=_t(current_action),
        next_action=_t(next_action),
    )
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    else:
        st.info(message)


def _query_param_value(name: str) -> str | None:
    value = st.query_params.get(name)
    if isinstance(value, list):
        return str(value[0]) if value else None
    if value is None:
        return None
    return str(value)


def _write_navigation_query_params(step: str, frame_index: int | None = None) -> None:
    current_step = _query_param_value("step")
    if current_step != step:
        st.query_params["step"] = step
    if step == "check" and frame_index is not None:
        frame_value = str(frame_index)
        if _query_param_value("frame") != frame_value:
            st.query_params["frame"] = frame_value
    elif "frame" in st.query_params:
        del st.query_params["frame"]


def _sync_navigation_state_from_query_params() -> None:
    query_step = _query_param_value("step")
    if query_step in WIZARD_STEPS:
        st.session_state["active_step"] = query_step
        if query_step == "check":
            query_frame = _query_param_value("frame")
            try:
                frame_index = int(query_frame) if query_frame is not None else 0
            except ValueError:
                frame_index = 0
            st.session_state["preprocess_frame_idx"] = frame_index
        else:
            st.session_state["preprocess_frame_idx"] = 0
        return
    _write_navigation_query_params(
        _active_step(),
        _preprocess_frame_index() if _active_step() == "check" else None,
    )


def _wizard_step_index(step: str) -> int:
    return WIZARD_STEPS.index(step) if step in WIZARD_STEPS else 0


def _set_active_step(step: str) -> None:
    if step in WIZARD_STEPS:
        st.session_state["active_step"] = step
        if step != "check":
            st.session_state["preprocess_frame_idx"] = 0
        _write_navigation_query_params(
            step,
            _preprocess_frame_index() if step == "check" else None,
        )


def _active_step() -> str:
    step = str(st.session_state.get("active_step", "upload"))
    return step if step in WIZARD_STEPS else "upload"


def _preprocess_frame_index() -> int:
    """Current beamer mini-frame inside the data preprocessing section."""
    frame_count = len(BEAMER_NAV_SECTIONS["preprocess"])
    try:
        frame_index = int(st.session_state.get("preprocess_frame_idx", 0))
    except (TypeError, ValueError):
        frame_index = 0
    return max(0, min(frame_index, frame_count - 1))


def _set_preprocess_frame_index(frame_index: int) -> None:
    frame_count = len(BEAMER_NAV_SECTIONS["preprocess"])
    resolved_index = max(0, min(int(frame_index), frame_count - 1))
    st.session_state["preprocess_frame_idx"] = resolved_index
    if _active_step() == "check":
        _write_navigation_query_params("check", resolved_index)


def _render_preprocessing_roadmap() -> None:
    """Render local controls for the separated preprocessing mini-frames."""
    frames = list(BEAMER_NAV_SECTIONS["preprocess"])
    current_idx = _preprocess_frame_index()

    st.markdown(
        '<div class="beamer-frame-selector-title">'
        + _html_escape(_t("Preprocessing frames"))
        + '</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(len(frames))
    for idx, frame_name in enumerate(frames):
        with cols[idx]:
            button_type = "primary" if idx == current_idx else "secondary"
            if st.button(
                _t(frame_name),
                key=f"preprocess_frame_select_{idx}",
                use_container_width=True,
                type=button_type,
            ):
                _set_preprocess_frame_index(idx)
                st.rerun()

    nav_left, nav_mid, nav_right = st.columns([1, 3, 1])
    with nav_left:
        if st.button(
            _t("← Previous frame"),
            key="preprocess_frame_prev",
            use_container_width=True,
            disabled=current_idx <= 0,
        ):
            _set_preprocess_frame_index(current_idx - 1)
            st.rerun()
    with nav_mid:
        st.markdown(
            '<div class="beamer-frame-progress">'
            + _html_escape(
                _t(
                    "Frame {current} / {total}: {frame}",
                    current=current_idx + 1,
                    total=len(frames),
                    frame=_t(frames[current_idx]),
                )
            )
            + '</div>',
            unsafe_allow_html=True,
        )
    with nav_right:
        if st.button(
            _t("Next frame →"),
            key="preprocess_frame_next",
            use_container_width=True,
            disabled=current_idx >= len(frames) - 1,
        ):
            _set_preprocess_frame_index(current_idx + 1)
            st.rerun()


def _nav_target_for(section_name: str, frame_index: int) -> tuple[str, int | None]:
    steps = BEAMER_NAV_STEP_TARGETS.get(section_name, ())
    if not steps:
        return "upload", None
    safe_index = max(0, min(frame_index, len(steps) - 1))
    step = steps[safe_index]
    if step == "check":
        return step, safe_index
    return step, None


def _nav_href_for(section_name: str, frame_index: int) -> str:
    step, query_frame = _nav_target_for(section_name, frame_index)
    params: dict[str, str] = {"step": step}
    if query_frame is not None:
        params["frame"] = str(query_frame)
    return "?" + urlencode(params)


def _render_wizard_nav(
    *,
    dataset_loaded: bool,
    target_selected: bool,
    checks_passed: bool,
    prepared_ready: bool,
    training_finished: bool,
) -> None:
    active_step = _active_step()
    active_section, active_frame_index = BEAMER_STEP_TO_SECTION_FRAME.get(
        active_step, ("dataset", 0)
    )
    if active_step == "check":
        active_frame_index = _preprocess_frame_index()
    section_names = list(BEAMER_NAV_SECTIONS)
    active_section_index = section_names.index(active_section)
    completed_sections = {
        "dataset": dataset_loaded,
        "task": target_selected,
        "preprocess": bool(checks_passed or prepared_ready),
        "training": bool(training_finished),
        "results": bool(training_finished),
    }
    enabled_sections = {
        "dataset": True,
        "task": dataset_loaded,
        "preprocess": dataset_loaded and target_selected,
        "training": dataset_loaded and target_selected,
        "results": training_finished,
    }

    html = ['<nav class="beamer-nav" aria-label="AutoML workflow navigation">']
    for section_index, (section_name, frames) in enumerate(BEAMER_NAV_SECTIONS.items()):
        section_active = section_name == active_section
        section_finished = bool(completed_sections.get(section_name)) or (
            section_index < active_section_index
        )
        section_enabled = bool(enabled_sections.get(section_name))
        title_class = "beamer-nav-title active" if section_active else "beamer-nav-title"
        html.append('<div class="beamer-nav-section">')
        html.append(
            f'<div class="{title_class}">{_html_escape(_t(BEAMER_SECTION_LABELS[section_name]))}</div>'
        )
        html.append('<div class="beamer-dots">')
        for frame_index, frame_name in enumerate(frames):
            if section_active and frame_index == active_frame_index:
                dot_class = "beamer-dot current"
            elif section_finished or (section_active and frame_index < active_frame_index):
                dot_class = "beamer-dot done"
            else:
                dot_class = "beamer-dot"
            if section_enabled:
                dot_href = _nav_href_for(section_name, frame_index)
                dot_label = _html_escape(
                    f"{_t(BEAMER_SECTION_LABELS[section_name])} · {_t(frame_name)}"
                )
                html.append(
                    f'<a class="{dot_class} beamer-dot-link" href="{dot_href}" target="_self" aria-label="{dot_label}" title="{dot_label}"></a>'
                )
            else:
                html.append(f'<span class="{dot_class}"></span>')
        html.append('</div></div>')
    html.append('</nav>')
    st.markdown("".join(html), unsafe_allow_html=True)


def _render_bottom_workflow_nav() -> None:
    """No-op: global navigation is kept only in the beamer-style headline."""
    return


def _cache_current_dataset(df: pd.DataFrame, *, source_label: str) -> None:
    st.session_state["_current_df"] = df.copy()
    st.session_state["_current_source_label"] = source_label


def _cached_current_dataset() -> pd.DataFrame | None:
    cached = st.session_state.get("_current_df")
    return cached.copy() if isinstance(cached, pd.DataFrame) else None


def _clear_upload_revisit_guard() -> None:
    st.session_state.pop("_suppress_upload_auto_advance", None)
    st.session_state.pop("_upload_revisit_signature", None)


def _request_upload_revisit() -> None:
    cached = _cached_current_dataset()
    if cached is not None:
        st.session_state["_suppress_upload_auto_advance"] = True
        st.session_state["_upload_revisit_signature"] = _dataset_fingerprint(cached)
    _set_active_step("upload")


def _handle_loaded_upload_dataset(
    df: pd.DataFrame,
    *,
    source_label: str,
    status_text: str,
    continue_key: str,
) -> None:
    dataset_signature = _dataset_fingerprint(df)
    revisit_same_dataset = (
        bool(st.session_state.get("_suppress_upload_auto_advance"))
        and st.session_state.get("_upload_revisit_signature") == dataset_signature
    )
    _cache_current_dataset(df, source_label=source_label)
    _render_step_status(
        status_text,
        _t("Choose the column you want to predict."),
        level="success",
    )
    if revisit_same_dataset:
        if st.button(_t("Next: choose target"), key=continue_key):
            _clear_upload_revisit_guard()
            _set_active_step("target")
            st.rerun()
        return
    _clear_upload_revisit_guard()
    _set_active_step("target")
    st.rerun()


def _render_message(level: str, text: str) -> None:
    if level == "success":
        st.success(text)
    elif level == "warning":
        st.warning(text)
    else:
        st.info(text)


def _configure_llm_settings(settings: Settings) -> Settings:
    allow_local_llm_config = _local_llm_config_enabled()
    saved_config = _load_local_llm_config()
    saved_api_key = saved_config.get("llm_api_key", "")
    saved_base_url = saved_config.get("llm_base_url", "")
    saved_model = saved_config.get("llm_model", "")
    _section_caption(
        _t(
            "Optional AI help: you can finish the full local training flow without any API key. Add one only if you want AI-generated suggestions and a more natural-language report."
        )
    )

    api_key = saved_api_key
    base_url = saved_base_url or settings.llm_base_url or ""
    model = saved_model or settings.llm_model

    with st.expander(_t("Optional AI help"), expanded=False):
        if allow_local_llm_config:
            st.caption(
                _t(
                    "This environment can remember settings locally. Hosted deployments can also use `LLM_API_KEY` or Streamlit Secrets."
                )
            )
        else:
            st.caption(
                _t(
                    "Local persistence is disabled here, so enter a key per session or configure `LLM_API_KEY` / Streamlit Secrets."
                )
            )

        config_cols = st.columns(3)
        with config_cols[0]:
            api_key = st.text_input(
                _t("API key"),
                value=saved_api_key,
                key="_llm_api_key",
                type="password",
                placeholder=_t("Uses LLM_API_KEY if empty"),
            )
        with config_cols[1]:
            base_url = st.text_input(
                _t("Base URL"),
                value=saved_base_url or settings.llm_base_url or "",
                key="_llm_base_url",
                placeholder=_t("LLM default or compatible API URL"),
            )
        with config_cols[2]:
            model = st.text_input(
                _t("Model"), value=saved_model or settings.llm_model, key="_llm_model"
            )

        if allow_local_llm_config:
            remember_config = st.checkbox(
                _t("Remember LLM settings on this device"), value=bool(saved_api_key)
            )
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

            if saved_config and st.button(_t("Forget saved LLM settings")):
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


def _target_task_types(
    df: pd.DataFrame, targets: list[str], task_type_choice: str
) -> dict[str, str]:
    if task_type_choice in {"classification", "regression"}:
        return {target: task_type_choice for target in targets}
    return {target: _infer_task_type(df, target) for target in targets}


def _experiment_signature(
    *,
    dataset_fingerprint: str,
    target_columns: list[str],
    task_type_choice: str,
    time_budget: int,
    priority_metric_choice: str,
    planner_brief: str,
    preprocessing_plan: Any,
) -> str:
    payload = {
        "dataset_fingerprint": dataset_fingerprint,
        "target_columns": target_columns,
        "task_type_choice": task_type_choice,
        "time_budget": time_budget,
        "priority_metric_choice": priority_metric_choice,
        "planner_brief": planner_brief.strip(),
        "preprocessing_plan": artifact_to_dict(preprocessing_plan),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _initialize_experiment_state(dataset_signature: str, columns: list[str]) -> None:
    signature = (dataset_signature, tuple(columns))
    if st.session_state.get("_dataset_signature") == signature:
        return
    st.session_state["_dataset_signature"] = signature
    st.session_state["planner_brief"] = ""
    st.session_state["target_columns"] = []
    st.session_state["_selected_target_columns"] = []
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
    st.session_state["numeric_imputation_strategy"] = "median"
    st.session_state["categorical_imputation_strategy"] = "most_frequent"
    st.session_state["categorical_encoding_strategy"] = "one_hot"
    st.session_state["standardize_numeric"] = True
    for key, value in AUTOGLUON_FEATURE_GENERATOR_DEFAULTS.items():
        st.session_state[key] = value
    st.session_state["_planner_signature"] = None
    st.session_state["_planner_suggestion"] = None
    st.session_state["_feature_engineering_plan_signature"] = None
    st.session_state["_feature_engineering_plan"] = None
    st.session_state["_feature_engineering_override_plan"] = None
    st.session_state["_feature_engineering_applied_plan"] = None
    st.session_state["manual_cleaning_brief"] = ""
    st.session_state["_manual_cleaning_plan_signature"] = None
    st.session_state["_manual_cleaning_suggested_plan"] = None
    st.session_state["_manual_cleaning_plan"] = None
    st.session_state["_manual_cleaning_override_plan"] = None
    st.session_state["_preprocessing_plan_applied"] = _default_preprocessing_plan()
    st.session_state["_preprocessing_plan_draft"] = _default_preprocessing_plan()
    st.session_state["_latest_prepared_signature"] = None
    st.session_state["_latest_prepared_batches"] = []
    st.session_state["_latest_results_signature"] = None
    st.session_state["_latest_results"] = []
    if _active_step() == "upload":
        st.session_state["active_step"] = "target"


def _render_run_outputs(results: list[dict[str, object]]) -> None:
    _render_slide_title(
        "results",
        "Evaluation & Export",
        "Results are separated into summary, metrics, validation, importance, and downloads.",
    )
    if not results:
        st.info(_t("No completed training results are available yet."))
        return

    completed_targets = ", ".join(str(result["target"]) for result in results)
    _render_step_status(
        _t(
            "Training finished for: {completed_targets}.",
            completed_targets=completed_targets,
        ),
        _t("Review summary first, then open the detailed result frames."),
        level="success",
    )

    summary_tab, metrics_tab, validation_tab, importance_tab, downloads_tab = st.tabs(
        [
            _t("Summary"),
            _t("Metrics"),
            _t("Validation"),
            _t("Feature importance"),
            _t("Downloads"),
        ]
    )

    with summary_tab:
        _panel_title(_t("Run summary"))
        summary_cols = st.columns(3)
        with summary_cols[0]:
            st.metric(_t("Completed runs"), len(results))
        with summary_cols[1]:
            st.metric(
                _t("Targets trained"),
                len({str(result["target"]) for result in results}),
            )
        with summary_cols[2]:
            st.metric(_t("Result files per run"), 3)

        summary_rows = [
            {
                "target": str(result["target"]),
                "task_type": _t(str(result["task_type"])),
                "priority_metric": str(result["priority_metric"]),
                "trainer": str(result["trained"].trainer_name),
            }
            for result in results
        ]
        summary_frame = pd.DataFrame(summary_rows).rename(
            columns={
                "target": _t("Target"),
                "task_type": _t("Task type"),
                "priority_metric": _t("Priority metric"),
                "trainer": _t("Trainer"),
            }
        )
        st.dataframe(summary_frame, hide_index=True, use_container_width=True)
        _render_explanation_strip(
            "Summary is the first frame: it tells you which targets were trained and which metric each run optimized."
        )

    with metrics_tab:
        for result in results:
            with st.container(border=True):
                _panel_title(_t("{target} metrics", target=result["target"]))
                result_level, result_summary = _result_translation_summary(result)
                _render_message(result_level, result_summary)
                _render_metrics(
                    result["metrics"], priority_metric=result["priority_metric"]
                )
                if result["trained"].leaderboard:
                    _section_title(_t("AutoGluon leaderboard"))
                    st.dataframe(
                        pd.DataFrame(result["trained"].leaderboard),
                        use_container_width=True,
                    )
        _render_explanation_strip(
            "Metrics are isolated from downloads and reports so the result page does not become a single dense block."
        )

    with validation_tab:
        for result in results:
            with st.container(border=True):
                _panel_title(_t("{target} validation", target=result["target"]))
                _render_validation_summary(
                    preflight=artifact_to_dict(result["preflight_validation"]),
                    postrun=artifact_to_dict(result["postrun_validation"]),
                    recommendations=artifact_to_dict(result["recommendations"]),
                    priority_metric=result["priority_metric"],
                )
                _section_title(_t("Data flow"))
                _render_data_flow(result["data_flow"])
        _render_explanation_strip(
            "Validation is separated from score comparison because it answers a different question: whether this run is safe to trust."
        )

    with importance_tab:
        for result in results:
            with st.container(border=True):
                _panel_title(_t("{target} feature importance", target=result["target"]))
                importance_frame = pd.DataFrame(result["trained"].feature_importance)
                if importance_frame.empty:
                    st.caption(_t("No feature importance was returned for this run."))
                else:
                    st.dataframe(importance_frame, use_container_width=True)
        _render_explanation_strip(
            "Feature importance is kept in its own frame so model explanation does not compete with model comparison."
        )

    with downloads_tab:
        _panel_title(_t("Download files"))
        for result_index, result in enumerate(results):
            run = result["run"]
            label = f"{result['target']} ({_t(str(result['task_type']))})"
            with st.container(border=True):
                _section_title(label)
                download_cols = st.columns(3)
                with download_cols[0]:
                    model_path = Path(result["model_path"])
                    st.download_button(
                        label=_t("Download model"),
                        data=model_path.read_bytes(),
                        file_name=f"{run.run_id}_{result['target']}_{model_path.name}",
                        mime="application/octet-stream",
                        key=_safe_widget_key("download", "model", run.run_id, result_index),
                        use_container_width=True,
                    )
                with download_cols[1]:
                    st.download_button(
                        label=_t("Download report"),
                        data=result["report_path"].read_text(encoding="utf-8"),
                        file_name=f"{run.run_id}_{result['target']}_report.md",
                        mime="text/markdown",
                        key=_safe_widget_key("download", "report", run.run_id, result_index),
                        use_container_width=True,
                    )
                with download_cols[2]:
                    st.download_button(
                        label=_t("Download predictions"),
                        data=result["prediction_path"].read_text(encoding="utf-8"),
                        file_name=f"{run.run_id}_{result['target']}_prediction_sample.csv",
                        mime="text/csv",
                        key=_safe_widget_key("download", "prediction", run.run_id, result_index),
                        use_container_width=True,
                    )
                with st.expander(_t("Analysis report"), expanded=False):
                    st.markdown(result["report"])
                st.caption(_t("Artifacts saved to {path}", path=Path(run.path).resolve()))
        _render_explanation_strip(
            "Downloads are the final frame: model, report, and prediction sample are grouped by target."
        )


def _queue_plan_suggestion(
    plan_suggestion: dict[str, object], columns: list[str]
) -> None:
    suggested_targets = [
        column
        for column in plan_suggestion.get("suggested_targets", [])
        if column in columns
    ]
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

    pending_targets = [
        column for column in pending.get("target_columns", []) if column in columns
    ]
    if pending_targets:
        st.session_state["target_columns"] = pending_targets

    suggested_task_type = pending.get("task_type_choice")
    if suggested_task_type in {"auto", "classification", "regression"}:
        st.session_state["task_type_choice"] = suggested_task_type

    excluded_columns = [
        column
        for column in pending.get("excluded_columns", [])
        if column in columns
        and column not in set(st.session_state.get("target_columns", []))
    ]
    st.session_state["excluded_columns"] = excluded_columns

    metric = pending.get("priority_metric_choice")
    if metric in {
        "auto",
        "accuracy",
        "f1_weighted",
        "precision_weighted",
        "recall_weighted",
        "roc_auc",
        "rmse",
        "mae",
        "r2",
    }:
        st.session_state["priority_metric_choice"] = metric


def _render_text_items(title: str, items: list[object], empty_text: str) -> None:
    _section_title(_t(title))
    normalized_items = items if isinstance(items, list) else [items]
    clean_items = [str(item) for item in normalized_items if str(item).strip()]
    if not clean_items:
        st.caption(_t(empty_text))
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
        st.metric(_t("Planner"), str(plan_data.get("planner_name") or "unknown"))
    with summary_cols[1]:
        st.metric(_t("Task type"), str(plan_data.get("suggested_task_type") or "auto"))
    with summary_cols[2]:
        st.metric(
            _t("Priority metric"), str(plan_data.get("priority_metric") or "auto")
        )

    target_items = [
        str(item)
        for item in plan_data.get("suggested_targets", [])
        if str(item).strip()
    ]
    excluded_items = [
        str(item)
        for item in plan_data.get("suggested_excluded_columns", [])
        if str(item).strip()
    ]
    detail_cols = st.columns(2)
    with detail_cols[0]:
        st.write(_t("Suggested targets"))
        if target_items:
            st.dataframe(
                pd.DataFrame({_t("Target"): target_items}),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.caption(_t("No target suggestion."))
    with detail_cols[1]:
        st.write(_t("Suggested exclusions"))
        if excluded_items:
            st.dataframe(
                pd.DataFrame({_t("Column"): excluded_items}),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.caption(_t("No excluded columns suggested."))

    notes_cols = st.columns(2)
    with notes_cols[0]:
        _render_text_items(
            "Notes", _normalize_item_list(plan_data.get("notes")), "No notes."
        )
    with notes_cols[1]:
        _render_text_items(
            "Risk flags",
            _normalize_item_list(plan_data.get("risk_flags")),
            "No risk flags.",
        )

    with st.expander(_t("Raw planner JSON"), expanded=False):
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
        st.metric(
            _t("Planner"), str(plan_data.get("planner_name") or "local_whitelist")
        )
    with summary_cols[1]:
        st.metric(_t("Accepted ops"), len(operations))
    with summary_cols[2]:
        st.metric(_t("Rejected ops"), len(rejected))

    if operations:
        rows = []
        for operation in operations:
            if not isinstance(operation, dict):
                continue
            rows.append(
                {
                    _t("Operation"): operation.get("operation"),
                    _t("Source"): operation.get("source_column")
                    or ", ".join(operation.get("columns", [])),
                    _t("Detail"): operation.get("operator")
                    or ", ".join(operation.get("parts", []))
                    or operation.get("bins")
                    or "",
                    _t("Rationale"): operation.get("rationale", ""),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.caption(_t("No feature engineering operations were accepted."))

    detail_cols = st.columns(2)
    with detail_cols[0]:
        _render_text_items("Notes", notes, "No notes.")
    with detail_cols[1]:
        _render_text_items("Rejected operations", rejected, "No rejected operations.")


def _feature_plan_operations(plan_data: Any) -> list[Any]:
    if isinstance(plan_data, dict):
        return list(plan_data.get("operations", []))
    return list(getattr(plan_data, "operations", []))


def _enabled_manual_rule_count(plan_data: Any) -> int:
    payload = artifact_to_dict(plan_data) if plan_data is not None else {}
    rules = payload.get("rules", []) if isinstance(payload, dict) else []
    return sum(
        1
        for rule in rules
        if isinstance(rule, dict) and bool(rule.get("enabled", True))
    )


def _preprocessing_manual_plan(plan_data: Any) -> dict[str, Any] | None:
    params = _preprocessing_step_params(plan_data, "manual_cleaning")
    manual_plan = params.get("plan")
    return dict(manual_plan) if isinstance(manual_plan, dict) else None


def _preprocessing_feature_plan(plan_data: Any) -> dict[str, Any] | None:
    params = _preprocessing_step_params(plan_data, "feature_engineering")
    if not bool(_preprocessing_step_enabled(plan_data, "feature_engineering")):
        return None
    operations = params.get("operations", [])
    if not isinstance(operations, list) or not operations:
        return None
    return {
        "planner_name": str(params.get("planner_name") or "local_whitelist"),
        "operations": operations,
        "rejected_operations": list(params.get("rejected_operations", [])),
        "notes": list(params.get("notes", [])),
    }


def _build_preprocessing_plan(
    *,
    base_analysis_df: pd.DataFrame,
    target_columns: list[str],
    excluded_columns: list[str],
    test_size: float,
    random_state: int,
    high_missing_threshold: float,
    numeric_imputation_strategy: str,
    categorical_imputation_strategy: str,
    categorical_encoding_strategy: str,
    standardize_numeric: bool,
    autogluon_feature_generator_params: dict[str, Any] | None = None,
    manual_cleaning_plan: Any = None,
    feature_plan: Any = None,
) -> dict[str, Any]:
    visible_columns = [
        column for column in base_analysis_df.columns if column not in excluded_columns
    ]
    preview_df = (
        base_analysis_df[visible_columns].copy()
        if visible_columns
        else base_analysis_df.iloc[:, :0].copy()
    )
    feature_columns = [
        column for column in preview_df.columns if column not in target_columns
    ]
    feature_df = (
        preview_df[feature_columns].copy()
        if feature_columns
        else preview_df.iloc[:, :0].copy()
    )
    numeric_columns = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [
        column for column in feature_df.columns if column not in numeric_columns
    ]
    high_missing_columns = [
        column
        for column in feature_columns
        if float(feature_df[column].isna().mean()) > float(high_missing_threshold)
    ]

    if categorical_encoding_strategy == "one_hot":
        encoded_feature_estimate = 0
        for column in categorical_columns:
            encoded_feature_estimate += max(int(feature_df[column].dropna().nunique()), 1)
    else:
        encoded_feature_estimate = len(categorical_columns)

    manual_plan_data = (
        artifact_to_dict(manual_cleaning_plan)
        if manual_cleaning_plan is not None
        else None
    )
    manual_rule_count = _enabled_manual_rule_count(manual_plan_data)
    feature_plan_data = (
        artifact_to_dict(feature_plan) if feature_plan is not None else None
    )
    feature_operations = (
        _feature_plan_operations(feature_plan_data) if feature_plan_data else []
    )
    autogluon_params = _autogluon_feature_generator_params(
        autogluon_feature_generator_params
    )

    steps = [
        _preprocessing_step(
            "column_selection",
            params={"excluded_columns": list(excluded_columns)},
            summary=f"Excluded columns: {len(excluded_columns)}.",
        ),
        _preprocessing_step(
            "manual_cleaning",
            enabled=manual_rule_count > 0,
            params={"plan": manual_plan_data},
            summary="No manual cleaning rules applied."
            if manual_rule_count == 0
            else f"Manual cleaning rules enabled: {manual_rule_count}.",
        ),
        _preprocessing_step(
            "missing_value",
            params={
                "high_missing_threshold": float(high_missing_threshold),
                "high_missing_columns": high_missing_columns,
            },
            summary=(
                f"Auto-drop high-missing columns: {len(high_missing_columns)}."
            ),
        ),
        _preprocessing_step(
            "autogluon_feature_generator",
            params={
                **autogluon_params,
                "numeric_feature_count": len(numeric_columns),
                "categorical_feature_count": len(categorical_columns),
                "estimated_one_hot_features_if_legacy": encoded_feature_estimate,
            },
            summary=(
                "AutoGluon feature generation enabled: "
                f"{', '.join(_autogluon_enabled_feature_names(autogluon_params))}."
            ),
        ),
        _preprocessing_step(
            "feature_engineering",
            enabled=bool(feature_plan_data and feature_operations),
            params={
                "planner_name": str(
                    (feature_plan_data or {}).get("planner_name") or "local_whitelist"
                ),
                "operations": feature_operations,
                "rejected_operations": list(
                    (feature_plan_data or {}).get("rejected_operations", [])
                ),
                "notes": list((feature_plan_data or {}).get("notes", [])),
            },
            summary=(
                "Feature engineering is disabled."
                if not feature_operations
                else f"Feature engineering operations: {len(feature_operations)}."
            ),
        ),
    ]
    return {
        "version": PREPROCESSING_PLAN_VERSION,
        "global_params": {
            "test_size": float(test_size),
            "random_state": int(random_state),
            "autogluon_presets": AUTOGLUON_DEFAULT_PRESETS,
        },
        "steps": steps,
        "applied_step_ids": [str(step["id"]) for step in steps],
        "notes": [],
    }


def _render_preprocessing_step_status(
    title: str, draft_step: dict[str, Any], applied_plan: Any
) -> None:
    applied_step = _preprocessing_step_payload(
        applied_plan, str(draft_step.get("kind"))
    )
    if applied_step is None:
        _render_step_status(
            title, "This step has not been applied yet.", level="warning"
        )
        return
    if _json_equal(applied_step, draft_step):
        _render_step_status(title, "Current settings are applied.", level="success")
        return
    _render_step_status(title, "Draft changes are not applied yet.", level="info")


def _preprocessing_status_text(draft_plan: Any, applied_plan: Any) -> str:
    if _json_equal(draft_plan, applied_plan):
        return _t("Current settings are applied.")
    return _t("Draft changes are not applied yet.")


def _preprocessing_summary_text(
    *,
    test_size: float,
    high_missing_threshold: float,
    autogluon_feature_generator_params: dict[str, bool],
    manual_cleaning_enabled: bool,
    feature_engineering_enabled: bool,
    draft_plan: Any,
    applied_plan: Any,
) -> str:
    enabled_features = ", ".join(
        _autogluon_enabled_feature_names(autogluon_feature_generator_params)
    )
    parts = [
        f"{_t('Test size')} {test_size:.2f}",
        f"{_t('Drop feature when missing rate is above')} {high_missing_threshold:.2f}",
        f"{_t('AutoGluon feature generation')} {enabled_features}",
        f"{_t('Manual cleaning')} {_t('Yes') if manual_cleaning_enabled else _t('No')}",
        f"{_t('Feature engineering')} {_t('Yes') if feature_engineering_enabled else _t('No')}",
        _preprocessing_status_text(draft_plan, applied_plan),
    ]
    return " | ".join(parts)


def _task_type_plain_language(task_type: str) -> str:
    if task_type == "classification":
        return _t("predicting labels such as yes/no or named categories")
    return _t("predicting numbers such as price, spend, or duration")


def _target_selection_summary(
    *,
    target_columns: list[str],
    task_type_choice: str,
    inferred_task_types: dict[str, str],
) -> tuple[str, str]:
    if task_type_choice == "auto":
        if len(target_columns) == 1:
            target = str(target_columns[0])
            inferred = inferred_task_types.get(target, "regression")
            return (
                "info",
                _t(
                    "Your target is the result you want to predict. The app currently reads `{target}` as {task_type_explanation}. This is only a suggestion, and you can still change Task type manually.",
                    target=target,
                    task_type_explanation=_task_type_plain_language(inferred),
                ),
            )
        return (
            "info",
            _t(
                "Each selected target will train as a separate run. With automatic task detection, the app will decide for each target whether it looks more like labels or numbers."
            ),
        )

    return (
        "success",
        _t(
            "Your target is the result you want to predict. You set the task type manually, so the app will treat the selected target as {task_type_explanation}.",
            task_type_explanation=_task_type_plain_language(task_type_choice),
        ),
    )


def _preflight_explanation_items(validation: dict[str, object]) -> list[str]:
    explanations: list[str] = []
    seen: set[str] = set()
    for issue in validation.get("issues", []):
        if not isinstance(issue, dict):
            continue
        code = str(issue.get("code") or "")
        if code in seen:
            continue
        seen.add(code)
        if code == "identifier_candidates":
            explanations.append(
                _t(
                    "Some columns look like record IDs. They help identify rows, but they usually do not help the model generalize, so excluding them is often safer."
                )
            )
        elif code == "possible_leakage":
            explanations.append(
                _t(
                    "Some columns look too close to the target. That can leak the answer into training and make the model look better than it really is."
                )
            )
        elif code == "high_missing_features":
            explanations.append(
                _t(
                    "Some features are missing too often. The app can still continue, but dropping or revisiting those columns usually makes the baseline easier to trust."
                )
            )
        elif code == "small_dataset":
            explanations.append(
                _t(
                    "Only a small number of rows are available for training and testing, so the score can swing a lot. Treat this run as an early signal, not a final answer."
                )
            )
        elif code == "class_imbalance":
            explanations.append(
                _t(
                    "One class is much rarer than the others. The model can still run, but overall scores may hide weak performance on the rare cases you care about."
                )
            )
        elif code == "missing_target":
            explanations.append(
                _t(
                    "The target column could not be found, so the app does not know what outcome to train on yet."
                )
            )
        elif code == "empty_after_target_drop":
            explanations.append(
                _t(
                    "Rows with missing target values are dropped before training. Here, that leaves no usable rows, so training cannot continue."
                )
            )
        elif code == "no_features":
            explanations.append(
                _t(
                    "No usable feature columns remain after exclusions. The app needs at least one input column to learn from."
                )
            )
    if int(validation.get("dropped_target_rows") or 0) > 0:
        explanations.append(
            _t(
                "Some rows are missing the target value and will be removed before training. If too many rows are dropped, trust the result more cautiously."
            )
        )
    return explanations


def _preflight_explanation_summary(validation: dict[str, object]) -> tuple[str, str]:
    issues = [
        issue for issue in validation.get("issues", []) if isinstance(issue, dict)
    ]
    has_errors = any(str(issue.get("severity")) == "error" for issue in issues)
    has_warnings = any(
        str(issue.get("severity")) in {"warning", "info"} for issue in issues
    )
    if has_errors:
        return (
            "warning",
            _t(
                "These checks can stop training. Fix the blocking items first, then run again."
            ),
        )
    if has_warnings:
        return (
            "info",
            _t(
                "These checks will not stop training, but they can make the result less reliable. Review the notes below before you trust the model."
            ),
        )
    return (
        "success",
        _t(
            "No blocking or major caution signals were found for this target. You can use this run as a first baseline."
        ),
    )


def _field_type_label(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return _t("Numeric")
    parsed = pd.to_datetime(series.dropna().head(50), errors="coerce", format="mixed")
    if not parsed.empty and float(parsed.notna().mean()) >= 0.8:
        return _t("Datetime-like")
    return _t("Categorical")


def build_field_health_rows(
    df: pd.DataFrame,
    *,
    target_columns: list[str],
    preflight_by_target: dict[str, object],
    high_missing_threshold: float,
) -> list[dict[str, object]]:
    recommended_exclusions: set[str] = set()
    leakage_columns: set[str] = set()
    for validation in preflight_by_target.values():
        payload = artifact_to_dict(validation)
        recommended_exclusions.update(
            str(column)
            for column in payload.get("recommended_excluded_columns", [])
            if str(column).strip()
        )
        leakage_columns.update(
            str(column)
            for column in payload.get("detected_leakage_columns", [])
            if str(column).strip()
        )

    rows: list[dict[str, object]] = []
    targets = set(target_columns)
    for column in df.columns:
        series = df[column]
        missing_rate = float(series.isna().mean()) if len(series) else 0.0
        unique_count = int(series.nunique(dropna=True))
        is_target = column in targets
        is_high_missing = missing_rate > high_missing_threshold and not is_target
        is_identifier = column in recommended_exclusions
        is_leakage = column in leakage_columns
        if is_target:
            suggestion = _t("Target column")
        elif is_leakage:
            suggestion = _t("Review or exclude: possible target leakage")
        elif is_identifier:
            suggestion = _t("Consider excluding: identifier-like")
        elif is_high_missing:
            suggestion = _t("Review missing values")
        else:
            suggestion = _t("Use by default")
        rows.append(
            {
                _t("Column"): str(column),
                _t("Role"): _t("Target") if is_target else _t("Feature"),
                _t("System type"): _field_type_label(series),
                _t("Missing rate"): round(missing_rate, 4),
                _t("Unique values"): unique_count,
                _t("ID-like"): _t("Yes") if is_identifier else _t("No"),
                _t("Leakage risk"): _t("Yes") if is_leakage else _t("No"),
                _t("Suggested action"): suggestion,
            }
        )
    return rows


def _render_field_health_table(
    df: pd.DataFrame,
    *,
    target_columns: list[str],
    preflight_by_target: dict[str, object],
    high_missing_threshold: float,
) -> None:
    _panel_title(_t("Field health"))
    _section_caption(
        _t(
            "Review the column type, missing rate, and risk flags before training. The app keeps safe defaults unless you change advanced settings."
        )
    )
    rows = build_field_health_rows(
        df,
        target_columns=target_columns,
        preflight_by_target=preflight_by_target,
        high_missing_threshold=high_missing_threshold,
    )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _priority_metric_plain_language(task_type: str, priority_metric: str) -> str:
    metric = resolve_priority_metric(task_type, priority_metric)
    metric_meanings = {
        "accuracy": _t("the share of predictions that were correct"),
        "f1_weighted": _t(
            "the balance between catching each class and avoiding wrong labels"
        ),
        "precision_weighted": _t(
            "how often the predicted labels were right when the model made that call"
        ),
        "recall_weighted": _t(
            "how often the model found the cases it was supposed to catch"
        ),
        "roc_auc": _t(
            "how well the model separates classes across different decision thresholds"
        ),
        "rmse": _t("the typical prediction error size, where lower is better"),
        "mae": _t("the average absolute prediction error, where lower is better"),
        "r2": _t("how much of the target variation the model explains"),
    }
    return metric_meanings.get(metric, metric)


def _result_translation_summary(result: dict[str, object]) -> tuple[str, str]:
    task_type = str(result["task_type"])
    priority_metric = str(result["priority_metric"])
    preflight = artifact_to_dict(result["preflight_validation"])
    postrun = artifact_to_dict(result["postrun_validation"])
    preflight_issues = [
        issue for issue in preflight.get("issues", []) if isinstance(issue, dict)
    ]
    postrun_issues = [
        issue for issue in postrun.get("issues", []) if isinstance(issue, dict)
    ]
    if any(str(issue.get("severity")) == "error" for issue in postrun_issues):
        level = "warning"
        reliability = _t(
            "This run completed, but there are blocking result issues. Read the warnings carefully before you rely on it."
        )
    elif preflight_issues or postrun_issues:
        level = "info"
        reliability = _t(
            "This run is best treated as a baseline. It is useful for direction, but you should read the warnings before trusting it for decisions."
        )
    else:
        level = "success"
        reliability = _t(
            "This run looks like a clean first baseline. It is still a good idea to compare it with another run before relying on it."
        )
    return (
        level,
        _t(
            "This model is {task_type_explanation}. The priority metric here is `{priority_metric}`, which tells you {metric_explanation}. {reliability}",
            task_type_explanation=_task_type_plain_language(task_type),
            priority_metric=resolve_priority_metric(task_type, priority_metric),
            metric_explanation=_priority_metric_plain_language(
                task_type, priority_metric
            ),
            reliability=reliability,
        ),
    )


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


def _new_manual_cleaning_rule(
    column: str = "", *, rule_type: str = "filter_row"
) -> dict[str, object]:
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
        return _t("Apply only before training")
    return _t("Apply before EDA and training")


def _manual_cleaning_rule_rows(plan_data: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rule in plan_data.get("rules", []):
        if not isinstance(rule, dict):
            continue
        rows.append(
            {
                _t("Enabled"): bool(rule.get("enabled", True)),
                _t("Rule type"): _t(str(rule.get("rule_type"))),
                _t("Column"): rule.get("column"),
                _t("Operator"): _t(str(rule.get("operator") or "-")),
                _t("Value"): _manual_cleaning_rule_value_text(rule.get("value")) or "-",
                _t("Rationale"): rule.get("rationale") or "",
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
    current_rules = [
        item for item in draft_plan.get("rules", []) if isinstance(item, dict)
    ]
    effect_stage_value = str(draft_plan.get("effect_stage") or DEFAULT_EFFECT_STAGE)
    if effect_stage_value not in VALID_EFFECT_STAGES:
        effect_stage_value = DEFAULT_EFFECT_STAGE
    effect_stage_options = ["pre_eda", "pre_training"]
    effect_stage_widget_key = "manual_cleaning_effect_stage"
    effect_stage_option_labels = {
        option: _manual_cleaning_effect_stage_label(option)
        for option in effect_stage_options
    }
    effect_stage_display_options = [
        effect_stage_option_labels[option] for option in effect_stage_options
    ]
    display_to_effect_stage = {
        label: option for option, label in effect_stage_option_labels.items()
    }
    stored_effect_stage = st.session_state.get(effect_stage_widget_key)
    if isinstance(stored_effect_stage, str):
        if stored_effect_stage in effect_stage_options:
            st.session_state[effect_stage_widget_key] = effect_stage_option_labels[
                stored_effect_stage
            ]
        elif stored_effect_stage not in effect_stage_display_options:
            st.session_state.pop(effect_stage_widget_key, None)
    selected_effect_stage_label = st.radio(
        _t("Rule effect stage"),
        effect_stage_display_options,
        index=effect_stage_display_options.index(
            effect_stage_option_labels[effect_stage_value]
        ),
        key=effect_stage_widget_key,
        horizontal=True,
    )
    effect_stage = display_to_effect_stage[selected_effect_stage_label]

    updated_rules: list[dict[str, object]] = []
    delete_rule_id: str | None = None
    for index, rule in enumerate(current_rules, start=1):
        rule_id = str(rule.get("id") or f"manual_rule_{index}")
        enabled_default = bool(rule.get("enabled", True))
        rule_type_default = str(rule.get("rule_type") or "filter_row")
        if rule_type_default not in VALID_RULE_TYPES:
            rule_type_default = "filter_row"
        rule_type_widget_key = f"manual_rule_type_{rule_id}"
        rule_type_option_labels = {
            "drop_column": _t("Drop column"),
            "filter_row": _t("Filter rows"),
        }
        rule_type_display_options = [
            rule_type_option_labels["drop_column"],
            rule_type_option_labels["filter_row"],
        ]
        display_to_rule_type = {
            label: option for option, label in rule_type_option_labels.items()
        }
        stored_rule_type = st.session_state.get(rule_type_widget_key)
        if isinstance(stored_rule_type, str):
            if stored_rule_type in rule_type_option_labels:
                st.session_state[rule_type_widget_key] = rule_type_option_labels[
                    stored_rule_type
                ]
            elif stored_rule_type not in rule_type_display_options:
                st.session_state.pop(rule_type_widget_key, None)
        column_default = str(rule.get("column") or "")
        operator_default = (
            str(rule.get("operator") or "equals")
            if rule_type_default == "filter_row"
            else ""
        )
        if operator_default not in FILTER_OPERATORS:
            operator_default = "equals"
        value_default = _manual_cleaning_rule_value_text(rule.get("value"))
        rationale_default = str(rule.get("rationale") or "")

        title = _t(
            "Rule {index}: {column}",
            index=index,
            column=column_default or _t("Select column"),
        )
        with st.expander(title, expanded=len(current_rules) == 1):
            top_cols = st.columns([0.8, 1.0, 1.3, 0.9])
            with top_cols[0]:
                enabled = st.checkbox(
                    _t("Enabled"),
                    value=enabled_default,
                    key=f"manual_rule_enabled_{rule_id}",
                )
            with top_cols[1]:
                selected_rule_type_label = st.selectbox(
                    _t("Rule type"),
                    rule_type_display_options,
                    index=rule_type_display_options.index(
                        rule_type_option_labels[rule_type_default]
                    ),
                    key=rule_type_widget_key,
                )
                rule_type = display_to_rule_type[selected_rule_type_label]
            with top_cols[2]:
                column_options = ["", *available_columns]
                column_index = (
                    column_options.index(column_default)
                    if column_default in column_options
                    else 0
                )
                column = st.selectbox(
                    _t("Column"),
                    column_options,
                    index=column_index,
                    key=f"manual_rule_column_{rule_id}",
                    format_func=lambda value: (
                        _t("Select column") if value == "" else value
                    ),
                )
            with top_cols[3]:
                delete_clicked = st.button(
                    _t("Delete rule"), key=f"manual_rule_delete_{rule_id}"
                )
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
                    operator_widget_key = f"manual_rule_operator_{rule_id}"
                    operator_option_labels = {
                        option: _t(option) for option in operator_options
                    }
                    operator_display_options = [
                        operator_option_labels[option] for option in operator_options
                    ]
                    display_to_operator = {
                        label: option
                        for option, label in operator_option_labels.items()
                    }
                    stored_operator = st.session_state.get(operator_widget_key)
                    if isinstance(stored_operator, str):
                        if stored_operator in operator_option_labels:
                            st.session_state[operator_widget_key] = (
                                operator_option_labels[stored_operator]
                            )
                        elif stored_operator not in operator_display_options:
                            st.session_state.pop(operator_widget_key, None)
                    selected_operator_label = st.selectbox(
                        _t("Operator"),
                        operator_display_options,
                        index=operator_display_options.index(
                            operator_option_labels[operator_default]
                        ),
                        key=operator_widget_key,
                    )
                    operator = display_to_operator[selected_operator_label]
                with operator_cols[1]:
                    if operator in {"is_null", "not_null"}:
                        st.caption(_t("No value is needed for this operator."))
                    else:
                        label = (
                            _t("Values (comma-separated)")
                            if operator in {"in", "not_in"}
                            else _t("Value")
                        )
                        raw_value = st.text_input(
                            label,
                            value=value_default,
                            key=f"manual_rule_value_{rule_id}",
                        )
                        if operator in {"in", "not_in"}:
                            parsed_value = [
                                item.strip()
                                for item in raw_value.split(",")
                                if item.strip()
                            ]
                        else:
                            parsed_value = raw_value.strip() or None
            else:
                st.caption(
                    _t(
                        "This rule drops the selected column before downstream processing."
                    )
                )

            rationale = st.text_input(
                _t("Rationale"),
                value=rationale_default,
                key=f"manual_rule_rationale_{rule_id}",
            )
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
        updated_rules = [
            rule for rule in updated_rules if str(rule.get("id")) != delete_rule_id
        ]

    controls = st.columns(2)
    add_blank_rule = controls[0].button(_t("Add blank rule"))
    reset_draft = controls[1].button(_t("Reset draft"))

    updated_plan = {
        "planner_name": str(draft_plan.get("planner_name") or "manual"),
        "user_brief": str(draft_plan.get("user_brief") or ""),
        "effect_stage": effect_stage,
        "rules": updated_rules,
        "notes": list(draft_plan.get("notes", []))
        if isinstance(draft_plan.get("notes"), list)
        else [],
        "rejected_rules": [],
    }
    if add_blank_rule:
        updated_plan["rules"].append(_new_manual_cleaning_rule())
    if reset_draft:
        applied = st.session_state.get("_manual_cleaning_plan")
        st.session_state["_manual_cleaning_override_plan"] = (
            _clone_json_data(applied) if applied else None
        )
        st.rerun()

    st.session_state["_manual_cleaning_override_plan"] = updated_plan
    if delete_rule_id or add_blank_rule:
        st.rerun()
    return updated_plan


def _render_target_profile(target_data: dict[str, object]) -> None:
    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.metric(_t("Target"), str(target_data.get("name") or "-"))
    with summary_cols[1]:
        st.metric(_t("Missing rows"), int(target_data.get("missing_count") or 0))
    with summary_cols[2]:
        st.metric(_t("Unique values"), int(target_data.get("unique_count") or 0))

    stats = target_data.get("stats", {})
    if isinstance(stats, dict) and stats:
        stats_rows = [
            {_t("Stat"): key, _t("Value"): value} for key, value in stats.items()
        ]
        st.dataframe(
            pd.DataFrame(stats_rows), hide_index=True, use_container_width=True
        )

    top_values = target_data.get("top_values", {})
    if isinstance(top_values, dict) and top_values:
        _section_title(_t("Top target values"))
        st.dataframe(
            pd.DataFrame(
                [
                    {_t("Value"): key, _t("Count"): value}
                    for key, value in top_values.items()
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )


def _flatten_target_relationships(
    relationships: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric_correlation_rows: list[dict[str, object]] = []
    numeric_group_rows: list[dict[str, object]] = []
    categorical_rows: list[dict[str, object]] = []

    for item in relationships.get("numeric_features", []):
        if not isinstance(item, dict):
            continue
        feature = str(item.get("feature") or "")
        if "target_correlation" in item:
            numeric_correlation_rows.append(
                {
                    "feature": feature,
                    "target_correlation": item.get("target_correlation"),
                }
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
    numeric_correlations, numeric_groups, categorical_distribution = (
        _flatten_target_relationships(relationships)
    )

    if not numeric_correlations.empty:
        _section_title(_t("Numeric feature relationships"))
        st.dataframe(numeric_correlations, hide_index=True, use_container_width=True)

    if not numeric_groups.empty:
        _section_title(_t("Numeric feature distribution by target"))
        st.dataframe(numeric_groups, hide_index=True, use_container_width=True)

    if not categorical_distribution.empty:
        _section_title(_t("Categorical feature target distribution"))
        st.dataframe(
            categorical_distribution, hide_index=True, use_container_width=True
        )

    if (
        numeric_correlations.empty
        and numeric_groups.empty
        and categorical_distribution.empty
    ):
        st.caption(_t("No target relationship summary available."))


def _distribution_mode_label(mode: str) -> str:
    labels = {
        "continuous_numeric": "Continuous numeric",
        "discrete_numeric": "Discrete numeric",
        "categorical": "Categorical",
        "datetime_like": "Datetime-like",
    }
    return _t(labels.get(mode, mode))


def _distribution_compare_message(
    plot_spec: dict[str, object],
) -> tuple[str, str] | None:
    target_column = str(plot_spec.get("target_column") or "")
    if not target_column:
        return None
    if bool(plot_spec.get("compare_enabled")):
        if plot_spec.get("compare_mode") == "quartile":
            return (
                "info",
                _t(
                    "Comparing feature values against quartiles of `{target}` (Q1 to Q4).",
                    target=target_column,
                ),
            )
        return (
            "info",
            _t(
                "Comparing feature values against target groups from `{target}`.",
                target=target_column,
            ),
        )

    reason = str(plot_spec.get("compare_reason") or "")
    if reason == "too_many_groups":
        return (
            "caption",
            _t(
                "Target `{target}` has {group_count} distinct values, so this chart shows the overall distribution only.",
                target=target_column,
                group_count=int(plot_spec.get("compare_group_count") or 0),
            ),
        )
    if reason == "unstable_quartiles":
        return (
            "caption",
            _t(
                "Target `{target}` could not be split into four stable quartiles, so this chart shows the overall distribution only.",
                target=target_column,
            ),
        )
    if reason == "no_target_values":
        return (
            "caption",
            _t(
                "Target `{target}` has no usable values for comparison.",
                target=target_column,
            ),
        )
    return None


def _build_distribution_chart(plot_spec: dict[str, object]) -> alt.Chart | alt.FacetChart | None:
    plot_data = plot_spec.get("plot_data")
    if not isinstance(plot_data, pd.DataFrame) or plot_data.empty:
        return None

    display_mode = str(plot_spec.get("display_mode") or "")
    compare_enabled = bool(plot_spec.get("compare_enabled"))
    target_group_title = _t("Target group")
    count_title = _t("Count")
    share_title = _t("Share")
    feature_column = str(plot_spec.get("column") or "")
    base = alt.Chart(plot_data)

    if display_mode == "continuous_numeric":
        histogram = base.mark_bar(color="#4C78A8").encode(
            x=alt.X(
                "feature_value:Q",
                bin=alt.Bin(maxbins=30),
                title=feature_column,
            ),
            y=alt.Y("count():Q", title=count_title),
        )
        if compare_enabled:
            return histogram.properties(height=220).facet(
                column=alt.Column(f"{'target_group'}:N", title=target_group_title)
            )
        return histogram.properties(height=320)

    if display_mode == "discrete_numeric":
        discrete = base.mark_bar().encode(
            x=alt.X("feature_value:O", title=feature_column, sort="ascending"),
            y=alt.Y("count():Q", title=count_title),
        )
        if compare_enabled:
            discrete = discrete.encode(
                color=alt.Color("target_group:N", title=target_group_title)
            )
        return discrete.properties(height=320)

    if display_mode == "datetime_like":
        datetime_chart = base.mark_bar(color="#72B7B2").encode(
            x=alt.X("time_bucket:T", title=feature_column),
            y=alt.Y("count():Q", title=count_title),
        )
        if compare_enabled:
            return datetime_chart.properties(height=220).facet(
                column=alt.Column("target_group:N", title=target_group_title)
            )
        return datetime_chart.properties(height=320)

    categorical = base.mark_bar().encode(
        x=alt.X("feature_value:N", title=feature_column, sort="-y"),
        y=alt.Y(
            "count():Q",
            title=share_title if compare_enabled else count_title,
            stack="normalize" if compare_enabled else None,
        ),
    )
    if compare_enabled:
        categorical = categorical.encode(
            color=alt.Color("target_group:N", title=target_group_title)
        )
    return categorical.properties(height=320)


def _render_distribution_explorer(
    analysis_df: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_columns: list[str],
    primary_target: str,
) -> None:
    _panel_title(_t("Distribution explorer"))
    _section_caption(
        _t(
            "Use this view to scan feature distributions and compare them against the selected target."
        )
    )
    if not feature_columns:
        st.info(_t("No feature columns available for distribution charts."))
        return

    overview = build_distribution_overview(analysis_df, feature_columns)
    if overview.empty:
        st.info(_t("No feature columns available for distribution charts."))
        return

    default_feature = choose_default_distribution_feature(analysis_df, feature_columns)
    default_index = 0
    if default_feature and default_feature in feature_columns:
        default_index = feature_columns.index(default_feature)

    control_cols = st.columns(2 if len(target_columns) > 1 else 1)
    with control_cols[0]:
        selected_feature = st.selectbox(
            _t("Feature to inspect"),
            options=feature_columns,
            index=default_index,
            key="distribution_feature_column",
        )
    selected_target = primary_target
    if len(target_columns) > 1:
        with control_cols[1]:
            selected_target = st.selectbox(
                _t("Compare target"),
                options=target_columns,
                index=target_columns.index(primary_target),
                key="distribution_compare_target",
            )

    overview_display = overview.copy()
    overview_display["display_mode"] = overview_display["display_mode"].map(
        _distribution_mode_label
    )
    overview_display = overview_display.rename(
        columns={
            "column": _t("Column"),
            "dtype": _t("Dtype"),
            "display_mode": _t("Display mode"),
            "non_null_count": _t("Non-null rows"),
            "missing_rate": _t("Missing rate"),
            "unique_count": _t("Unique values"),
        }
    )
    _section_title(_t("Distribution overview"))
    st.dataframe(overview_display, hide_index=True, use_container_width=True)

    plot_spec = build_distribution_plot_data(
        analysis_df,
        feature_column=selected_feature,
        target_column=selected_target,
    )
    detail_cols = st.columns(4)
    with detail_cols[0]:
        st.metric(_t("Display mode"), _distribution_mode_label(str(plot_spec["display_mode"])))
    with detail_cols[1]:
        st.metric(_t("Missing rate"), f"{float(plot_spec['missing_rate']):.1%}")
    with detail_cols[2]:
        st.metric(_t("Unique values"), int(plot_spec["unique_count"]))
    with detail_cols[3]:
        st.metric(_t("Rows used in chart"), int(plot_spec["sample_count"]))

    compare_message = _distribution_compare_message(plot_spec)
    if compare_message is not None:
        level, text = compare_message
        if level == "info":
            st.info(text)
        else:
            st.caption(text)

    chart = _build_distribution_chart(plot_spec)
    if chart is None:
        st.info(_t("No rows available for the selected distribution chart."))
        return
    st.altair_chart(chart, use_container_width=True)


def _render_metrics(
    metrics: dict[str, object], priority_metric: str | None = None
) -> None:
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
            st.metric(_t("Priority metric"), priority_metric)
        with priority_cols[1]:
            st.metric(
                _t("Priority value"),
                "-" if resolved_value is None else f"{float(resolved_value):.4f}",
            )

    metric_cols = st.columns(2)
    with metric_cols[0]:
        _section_title(_t("Test metrics"))
        if test_rows:
            st.dataframe(
                pd.DataFrame(test_rows), hide_index=True, use_container_width=True
            )
        else:
            st.caption(_t("No test metrics."))
    with metric_cols[1]:
        _section_title(_t("Train metrics"))
        if train_rows:
            st.dataframe(
                pd.DataFrame(train_rows), hide_index=True, use_container_width=True
            )
        else:
            st.caption(_t("No train metrics."))


def _preview_to_frame(preview: dict[str, object]) -> pd.DataFrame:
    rows = preview.get("rows", [])
    columns = [str(column) for column in preview.get("columns", [])]
    if not isinstance(rows, list):
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    ordered_columns = [column for column in columns if column in frame.columns]
    remaining_columns = [
        column for column in frame.columns if column not in ordered_columns
    ]
    return (
        frame[ordered_columns + remaining_columns]
        if not frame.empty or ordered_columns
        else frame
    )


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
    snapshots = [
        item for item in trace_data.get("snapshots", []) if isinstance(item, dict)
    ]
    if not snapshots:
        st.caption(_t("No data flow trace available."))
        return

    target_name = str(trace_data.get("target") or "run")
    for start in range(0, len(snapshots), 4):
        chunk = snapshots[start : start + 4]
        columns = st.columns(len(chunk))
        for index, snapshot in enumerate(chunk, start=start + 1):
            with columns[index - start - 1]:
                st.caption(
                    _t(
                        "{index}. {label}",
                        index=index,
                        label=snapshot.get("label") or snapshot.get("step"),
                    )
                )
                st.metric(_t("Shape"), _shape_label(snapshot))
                partition = str(snapshot.get("partition") or "full")
                stage = str(snapshot.get("stage") or "-")
                delta = snapshot.get("rows_delta")
                delta_text = "-" if delta is None else f"{int(delta):+d}"
                st.caption(
                    _t(
                        "{stage} | {partition} | delta {delta_text}",
                        stage=stage,
                        partition=partition,
                        delta_text=delta_text,
                    )
                )

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
        _t("Inspect data flow step"),
        options,
        key=f"data_flow_step_{target_name}",
    )
    selected_index = options.index(selected)
    selected_snapshot = snapshots[selected_index]

    metric_cols = st.columns(5)
    with metric_cols[0]:
        st.metric(_t("Stage"), str(selected_snapshot.get("stage") or "-"))
    with metric_cols[1]:
        st.metric(_t("Partition"), str(selected_snapshot.get("partition") or "-"))
    with metric_cols[2]:
        st.metric(_t("Kind"), str(selected_snapshot.get("data_kind") or "-"))
    with metric_cols[3]:
        st.metric(_t("Shape"), _shape_label(selected_snapshot))
    with metric_cols[4]:
        memory = selected_snapshot.get("memory_mb")
        memory_text = "-" if memory is None else f"{float(memory):.4f} MB"
        st.metric(_t("Memory"), memory_text)

    delta_cols = st.columns(3)
    with delta_cols[0]:
        delta = selected_snapshot.get("rows_delta")
        st.metric(_t("Rows delta"), "-" if delta is None else f"{int(delta):+d}")
    with delta_cols[1]:
        st.metric(_t("Columns added"), len(selected_snapshot.get("columns_added", [])))
    with delta_cols[2]:
        st.metric(
            _t("Columns removed"), len(selected_snapshot.get("columns_removed", []))
        )

    metadata = selected_snapshot.get("metadata", {})
    if isinstance(metadata, dict) and metadata:
        _section_title(_t("Metadata"))
        st.json(metadata, expanded=True)

    detail_cols = st.columns(2)
    with detail_cols[0]:
        added = [
            str(item)
            for item in selected_snapshot.get("columns_added", [])
            if str(item).strip()
        ]
        _render_text_items("Columns added", added, "No columns added.")
    with detail_cols[1]:
        removed = [
            str(item)
            for item in selected_snapshot.get("columns_removed", [])
            if str(item).strip()
        ]
        _render_text_items("Columns removed", removed, "No columns removed.")

    preview = selected_snapshot.get("preview")
    if isinstance(preview, dict):
        _section_title(_t("Preview"))
        preview_frame = _preview_to_frame(preview)
        st.dataframe(preview_frame, use_container_width=True)
        if preview.get("truncated"):
            st.caption(_t("Preview truncated to the first rows."))


def _render_validation_summary(
    preflight: dict[str, object],
    postrun: dict[str, object],
    recommendations: dict[str, object],
    priority_metric: str,
) -> None:
    summary_cols = st.columns(4)
    with summary_cols[0]:
        st.metric(_t("Requested metric"), priority_metric)
    with summary_cols[1]:
        st.metric(
            _t("Preflight"), _t("OK") if preflight.get("ok_to_run") else _t("Blocked")
        )
    with summary_cols[2]:
        st.metric(_t("Postrun"), _t("OK") if postrun.get("ok") else _t("Check issues"))
    with summary_cols[3]:
        st.metric(_t("Trainer"), str(postrun.get("trainer_name") or "-"))

    preflight_detail_cols = st.columns(3)
    with preflight_detail_cols[0]:
        st.metric(_t("Feature count"), int(preflight.get("feature_count") or 0))
    with preflight_detail_cols[1]:
        st.metric(
            _t("Dropped target rows"), int(preflight.get("dropped_target_rows") or 0)
        )
    with preflight_detail_cols[2]:
        st.metric(_t("Report mode"), str(postrun.get("report_mode") or "-"))

    generalization_gap = postrun.get("generalization_gap", {})
    if isinstance(generalization_gap, dict) and generalization_gap:
        gap_rows = [
            {"metric": key, "value": value} for key, value in generalization_gap.items()
        ]
        _section_title(_t("Generalization gap"))
        st.dataframe(pd.DataFrame(gap_rows), hide_index=True, use_container_width=True)

    class_balance = preflight.get("class_balance", {})
    if isinstance(class_balance, dict) and class_balance:
        _section_title(_t("Class balance"))
        st.dataframe(
            pd.DataFrame(
                [{"class": key, "share": value} for key, value in class_balance.items()]
            ),
            hide_index=True,
            use_container_width=True,
        )

    recommended_exclusions = preflight.get("recommended_excluded_columns", [])
    detected_leakage = preflight.get("detected_leakage_columns", [])
    detail_cols = st.columns(2)
    with detail_cols[0]:
        _render_text_items(
            "Recommended exclusions",
            list(recommended_exclusions),
            "No extra exclusions suggested.",
        )
    with detail_cols[1]:
        _render_text_items(
            "Potential leakage columns",
            list(detected_leakage),
            "No leakage columns detected.",
        )

    issue_cols = st.columns(2)
    with issue_cols[0]:
        _section_title(_t("Preflight issues"))
        _render_issue_table(list(preflight.get("issues", [])))
    with issue_cols[1]:
        _section_title(_t("Postrun issues"))
        _render_issue_table(list(postrun.get("issues", [])))

    recommendation_cols = st.columns(2)
    with recommendation_cols[0]:
        _render_text_items(
            "Recommendation summary",
            list(recommendations.get("summary", [])),
            "No summary available.",
        )
    with recommendation_cols[1]:
        _render_text_items(
            "Next steps",
            list(recommendations.get("next_steps", [])),
            "No next steps available.",
        )


def _render_issue_table(issues: list[dict[str, object]]) -> None:
    if not issues:
        st.caption(_t("No issues surfaced."))
        return
    st.dataframe(pd.DataFrame(issues), use_container_width=True)


def _render_integer_input(
    label: str, state_key: str, min_value: int | None = None
) -> int:
    text_key = f"{state_key}_text"
    raw_value = st.text_input(label, key=text_key)
    current_value = int(st.session_state.get(state_key, 0))
    candidate = raw_value.strip()
    try:
        parsed_value = int(candidate)
    except ValueError:
        st.caption(
            _t(
                "Enter a whole number. Using {current_value} until corrected.",
                current_value=current_value,
            )
        )
        return current_value

    if min_value is not None and parsed_value < min_value:
        st.caption(
            _t(
                "Enter a value greater than or equal to {min_value}. Using {current_value} until corrected.",
                min_value=min_value,
                current_value=current_value,
            )
        )
        return current_value

    st.session_state[state_key] = parsed_value
    return parsed_value


def _prepare_target_batches(
    *,
    df: pd.DataFrame,
    base_analysis_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    analysis_manual_cleaning_log: list[dict[str, object]],
    analysis_manual_cleaning_impact: dict[str, object],
    candidate_feature_columns: list[str],
    excluded_columns: list[str],
    target_columns: list[str],
    task_types: dict[str, str],
    priority_metrics: dict[str, str],
    preflight_by_target: dict[str, object],
    preprocessing_plan: Any,
    manual_cleaning_plan: Any,
) -> list[dict[str, object]]:
    if not candidate_feature_columns:
        raise ValueError(
            "No feature columns remain after excluding selected target and ignored columns."
        )

    prepared_batches: list[dict[str, object]] = []
    for target in target_columns:
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
        if (
            manual_cleaning_plan is not None
            and any(rule.enabled for rule in manual_cleaning_plan.rules)
            and manual_cleaning_plan.effect_stage == "pre_eda"
        ):
            tracker.snapshot_dataframe(
                "after_manual_cleaning_pre_eda",
                "After manual cleaning (EDA + training)",
                "intake",
                analysis_df,
                metadata=analysis_manual_cleaning_impact,
            )

        target_df = analysis_df[candidate_feature_columns + [target]].copy()
        tracker.snapshot_dataframe(
            "target_dataset",
            "Target dataset",
            "intake",
            target_df,
            preview=True,
            metadata={"target": target, "feature_columns": candidate_feature_columns},
        )
        training_input_df = target_df
        manual_cleaning_log = list(analysis_manual_cleaning_log)
        manual_cleaning_impact = dict(analysis_manual_cleaning_impact)
        if (
            manual_cleaning_plan is not None
            and any(rule.enabled for rule in manual_cleaning_plan.rules)
            and manual_cleaning_plan.effect_stage == "pre_training"
        ):
            target_manual_plan = validate_manual_cleaning_plan(
                manual_cleaning_plan,
                target_df,
                target,
                protected_columns=target_columns,
            )
            training_input_df, manual_cleaning_log, manual_cleaning_impact = (
                apply_manual_cleaning_plan(
                    target_df,
                    target_manual_plan,
                    target,
                    protected_columns=target_columns,
                )
            )
            tracker.snapshot_dataframe(
                "after_manual_cleaning_pre_training",
                "After manual cleaning (training only)",
                "intake",
                training_input_df,
                metadata=manual_cleaning_impact,
            )

        config = preprocessing_plan_to_clean_config(
            preprocessing_plan,
            target=target,
            task_type=task_type,
        )
        tracker.snapshot_artifact(
            "preprocessing_plan",
            "Preprocessing plan",
            "intake",
            metadata=artifact_to_dict(preprocessing_plan),
        )
        cleaned = clean_and_split(training_input_df, config, tracker=tracker)
        if manual_cleaning_log:
            cleaned.cleaning_log = manual_cleaning_log + cleaned.cleaning_log
        prepared = prepare_for_training(cleaned, tracker=tracker)
        prepared_batches.append(
            {
                "target": target,
                "task_type": task_type,
                "priority_metric": priority_metric,
                "preflight_validation": preflight_validation,
                "cleaned": prepared,
                "target_eda_summary": generate_eda_summary(target_df, target=target),
                "tracker": tracker,
            }
        )
    return prepared_batches


def _materialize_prepared_batches(
    *,
    current_experiment_signature: str,
    df: pd.DataFrame,
    base_analysis_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    analysis_manual_cleaning_log: list[dict[str, object]],
    analysis_manual_cleaning_impact: dict[str, object],
    candidate_feature_columns: list[str],
    excluded_columns: list[str],
    target_columns: list[str],
    task_types: dict[str, str],
    priority_metrics: dict[str, str],
    preflight_by_target: dict[str, object],
    preprocessing_plan: Any,
    manual_cleaning_plan: Any,
) -> list[dict[str, object]]:
    prepared_batches = _prepare_target_batches(
        df=df,
        base_analysis_df=base_analysis_df,
        analysis_df=analysis_df,
        analysis_manual_cleaning_log=analysis_manual_cleaning_log,
        analysis_manual_cleaning_impact=analysis_manual_cleaning_impact,
        candidate_feature_columns=candidate_feature_columns,
        excluded_columns=excluded_columns,
        target_columns=target_columns,
        task_types=task_types,
        priority_metrics=priority_metrics,
        preflight_by_target=preflight_by_target,
        preprocessing_plan=preprocessing_plan,
        manual_cleaning_plan=manual_cleaning_plan,
    )
    st.session_state["_latest_prepared_signature"] = current_experiment_signature
    st.session_state["_latest_prepared_batches"] = list(prepared_batches)
    return prepared_batches


def main() -> None:
    _sync_navigation_state_from_query_params()
    _apply_beamer_design()
    _render_language_switcher()
    _render_hero()
    _render_help_center()
    settings = _configure_llm_settings(load_settings())
    storage = RunStorage(settings.runs_dir)
    active_step = _active_step()
    active_index = _wizard_step_index(active_step)
    _render_wizard_nav(
        dataset_loaded=bool(st.session_state.get("_dataset_signature"))
        or _cached_current_dataset() is not None,
        target_selected=bool(
            st.session_state.get(
                "_selected_target_columns", st.session_state.get("target_columns")
            )
        ),
        checks_passed=bool(
            st.session_state.get(
                "_selected_target_columns", st.session_state.get("target_columns")
            )
        )
        and active_index >= _wizard_step_index("prepare"),
        prepared_ready=bool(st.session_state.get("_latest_prepared_batches")),
        training_finished=bool(st.session_state.get("_latest_results")),
    )

    df = _cached_current_dataset()
    if active_step == "upload" or df is None:
        _render_slide_title("upload", "Dataset Upload", "Start with one CSV file or a demo dataset.")
        with st.container(border=True):
            _section_caption(
                _t(
                    "Start with one CSV file or a demo dataset. The app saves each run under the configured runs directory."
                )
            )

            demo_csvs = sorted(
                Path(p).name for p in PROJECT_ROOT.glob("data/*.csv") if p.is_file()
            )

            upload_label = _t("Upload CSV")
            demo_label_to_name = {
                _t("Demo: {name}", name=name): name for name in demo_csvs
            }
            data_source = st.radio(
                _t("Data source"),
                [upload_label, *list(demo_label_to_name.keys())],
                horizontal=True,
                index=0,
                key="data_source_choice",
            )

            if data_source in demo_label_to_name:
                demo_name = demo_label_to_name[data_source]
                demo_path = PROJECT_ROOT / "data" / demo_name
                df = read_csv(demo_path)
                _handle_loaded_upload_dataset(
                    df,
                    source_label=demo_name,
                    status_text=_t(
                        "Loaded demo dataset `{demo_name}` with {rows} rows and {columns} columns.",
                        demo_name=demo_name,
                        rows=len(df),
                        columns=len(df.columns),
                    ),
                    continue_key="next_after_demo_upload_revisit",
                )
            else:
                uploaded_file = st.file_uploader(
                    _t("Upload CSV"), type=["csv"], label_visibility="collapsed"
                )
                if uploaded_file is None:
                    if df is not None:
                        source_label = str(
                            st.session_state.get("_current_source_label", "dataset.csv")
                        )
                        _render_step_status(
                            _t(
                                "Loaded `{file_name}` with {rows} rows and {columns} columns.",
                                file_name=source_label,
                                rows=len(df),
                                columns=len(df.columns),
                            ),
                            _t("Choose the column you want to predict."),
                            level="success",
                        )
                        if st.button(
                            _t("Next: choose target"),
                            key="next_after_cached_upload_revisit",
                        ):
                            _clear_upload_revisit_guard()
                            _set_active_step("target")
                            st.rerun()
                        _render_bottom_workflow_nav()
                        return
                    _render_step_status(
                        _t("No dataset has been loaded yet."),
                        _t(
                            "Upload a CSV or pick a demo dataset to unlock the next step."
                        ),
                    )
                    st.markdown(
                        f"""
                        <div class="beamer-empty">
                            {_t("Drop a CSV here to unlock schema inspection, missingness checks, training controls, and exportable run artifacts.")}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    _render_bottom_workflow_nav()
                    return
                df = read_csv(uploaded_file)
                _handle_loaded_upload_dataset(
                    df,
                    source_label=uploaded_file.name,
                    status_text=_t(
                        "Loaded `{file_name}` with {rows} rows and {columns} columns.",
                        file_name=uploaded_file.name,
                        rows=len(df),
                        columns=len(df.columns),
                    ),
                    continue_key="next_after_file_upload_revisit",
                )
        _render_bottom_workflow_nav()
        return
    current_dataset_fingerprint = _dataset_fingerprint(df)
    columns = list(df.columns)
    _initialize_experiment_state(current_dataset_fingerprint, columns)
    _consume_pending_plan_suggestion(columns)

    task_type_labels = {
        "auto": _t("auto"),
        "classification": _t("classification"),
        "regression": _t("regression"),
    }
    current_task_type_choice = str(st.session_state.get("task_type_choice", "auto"))
    task_type_choice = (
        current_task_type_choice
        if current_task_type_choice in task_type_labels
        else "auto"
    )
    priority_metric_choice = str(st.session_state.get("priority_metric_choice", "auto"))
    time_budget = int(st.session_state.get("time_budget", 30))
    stored_targets = st.session_state.get(
        "_selected_target_columns", st.session_state.get("target_columns", [])
    )
    target_columns = [column for column in stored_targets if column in columns]

    if active_step == "target":
        _render_slide_title("target", "Task Definition", "Choose target variables, task type, metric, and first-run budget.")
        with st.container(border=True):
            back_cols = st.columns([0.22, 0.78])
            with back_cols[0]:
                if st.button(_t("Back: upload data"), key="back_to_upload_from_target"):
                    _request_upload_revisit()
                    st.rerun()
            _section_caption(
                _t(
                    "Pick the column you want the app to predict. The app can infer the task type automatically."
                )
            )
            st.caption(
                _t(
                    "Your target column is the outcome you want the model to predict. Labels like yes/no usually mean classification, while numbers like price or spend usually mean regression."
                )
            )
            setup_cols = st.columns([1.05, 0.95])
            with setup_cols[0]:
                target_columns = st.multiselect(
                    _t("Target variables"),
                    columns,
                    key="target_columns",
                    help=_t(
                        "Select one or more targets. Multi-target runs train one model per target."
                    ),
                )
                st.session_state["_selected_target_columns"] = list(target_columns)
            with setup_cols[1]:
                if not target_columns:
                    st.caption(_t("System recommendation"))
                    st.info(
                        _t(
                            "Choose a target first. The app will suggest classification or regression automatically."
                        )
                    )

            if not target_columns:
                _render_step_status(
                    _t(
                        "Your dataset is ready, but no prediction target has been selected yet."
                    ),
                    _t(
                        "Select at least one target column to continue to the checks step."
                    ),
                    level="warning",
                )
                _render_bottom_workflow_nav()
                return

            selected_targets = ", ".join(str(target) for target in target_columns)
            selection_task_types = _target_task_types(
                df, target_columns, task_type_choice
            )
            with setup_cols[1]:
                if len(selection_task_types) == 1:
                    _render_task_type_card(
                        _t("Detected task type"),
                        _t(next(iter(selection_task_types.values()))),
                    )
                else:
                    _panel_title(_t("Detected task type"))
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {_t("Target"): target, _t("Task type"): _t(task_type)}
                                for target, task_type in selection_task_types.items()
                            ]
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )
                _section_caption(_t("Manual task type override"))
                task_type_choice = st.radio(
                    _t("Task type"),
                    list(task_type_labels.keys()),
                    horizontal=True,
                    key="task_type_choice",
                    format_func=lambda value: task_type_labels.get(str(value), str(value)),
                    help=_t("Use auto for inference, or force classification/regression manually."),
                )
                selection_task_types = _target_task_types(
                    df, target_columns, str(task_type_choice)
                )

            with st.expander(_t("Advanced experiment settings"), expanded=False):
                st.caption(
                    _t(
                        "Most first runs can keep the defaults here. Open this only if you want more control."
                    )
                )
                top_advanced_cols = st.columns(2)
                with top_advanced_cols[0]:
                    time_budget = _render_integer_input(
                        _t("Training time budget seconds"), "time_budget", min_value=5
                    )
                with top_advanced_cols[1]:
                    priority_metric_choice = st.selectbox(
                        _t("Priority metric"),
                        [
                            "auto",
                            "accuracy",
                            "f1_weighted",
                            "precision_weighted",
                            "recall_weighted",
                            "roc_auc",
                            "rmse",
                            "mae",
                            "r2",
                        ],
                        key="priority_metric_choice",
                        help=_t(
                            "This is the score the trainer treats as most important when choosing the best baseline."
                        ),
                    )
            _render_step_status(
                _t(
                    "Selected target columns: {selected_targets}.",
                    selected_targets=selected_targets,
                )
                if len(target_columns) > 1
                else _t(
                    "Selected target column: {selected_targets}.",
                    selected_targets=selected_targets,
                ),
                _t("Review the data checks before starting training."),
                level="success",
            )
            target_explanation_level, target_explanation = _target_selection_summary(
                target_columns=target_columns,
                task_type_choice=task_type_choice,
                inferred_task_types=selection_task_types,
            )
            _render_message(target_explanation_level, target_explanation)

            if st.button(_t("Next: data preprocessing"), key="next_after_target"):
                _set_active_step("check")
                st.rerun()
            _render_bottom_workflow_nav()
            return

    if not target_columns:
        _set_active_step("target")
        st.rerun()
    selection_task_types = _target_task_types(df, target_columns, task_type_choice)
    exclude_options = [column for column in columns if column not in target_columns]

    applied_preprocessing_plan = (
        st.session_state.get("_preprocessing_plan_applied")
        or _default_preprocessing_plan()
    )
    applied_global_params = _preprocessing_plan_global_params(
        applied_preprocessing_plan
    )
    applied_excluded_columns = [
        str(column)
        for column in _preprocessing_step_params(
            applied_preprocessing_plan, "column_selection"
        ).get("excluded_columns", [])
        if str(column).strip()
    ]
    if isinstance(
        applied_manual_cleaning_payload := _preprocessing_manual_plan(
            applied_preprocessing_plan
        ),
        dict,
    ):
        st.session_state["_manual_cleaning_plan"] = _clone_json_data(
            applied_manual_cleaning_payload
        )
    applied_feature_plan_payload = _preprocessing_feature_plan(
        applied_preprocessing_plan
    )
    if isinstance(applied_feature_plan_payload, dict):
        st.session_state["_feature_engineering_applied_plan"] = _clone_json_data(
            applied_feature_plan_payload
        )
    analysis_columns = [
        column for column in columns if column not in applied_excluded_columns
    ]
    base_analysis_df = df[analysis_columns].copy()

    primary_target = target_columns[0]
    if applied_manual_cleaning_payload is None and isinstance(
        st.session_state.get("_manual_cleaning_plan"), dict
    ):
        applied_manual_cleaning_payload = dict(
            st.session_state["_manual_cleaning_plan"]
        )
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
        if (
            any(rule.enabled for rule in manual_cleaning_plan.rules)
            and manual_cleaning_plan.effect_stage == "pre_eda"
        ):
            (
                analysis_df,
                analysis_manual_cleaning_log,
                analysis_manual_cleaning_impact,
            ) = apply_manual_cleaning_plan(
                base_analysis_df,
                manual_cleaning_plan,
                primary_target,
                protected_columns=target_columns,
            )

    candidate_feature_columns = [
        column for column in analysis_df.columns if column not in target_columns
    ]
    task_types = _target_task_types(analysis_df, target_columns, task_type_choice)
    if len(target_columns) > 1:
        st.caption(
            _t(
                "Multi-target mode trains and stores one independent run per target. Other selected targets are excluded from each model's feature set."
            )
        )

    eda_summary = generate_eda_summary(analysis_df, target=primary_target)
    planner_brief = str(st.session_state.get("planner_brief", ""))
    plan_suggestion = _get_planner_suggestion(
        df=analysis_df,
        eda_summary=eda_summary,
        settings=settings,
        user_brief=planner_brief,
        dataset_fingerprint=current_dataset_fingerprint,
    )
    preprocessing_summary_test_size = float(
        st.session_state.get("test_size", applied_global_params.get("test_size", 0.2))
    )
    preprocessing_summary_random_state = int(
        st.session_state.get(
            "random_state", applied_global_params.get("random_state", 42)
        )
    )
    preprocessing_summary_high_missing_threshold = float(
        st.session_state.get("high_missing_threshold", 0.9)
    )
    preprocessing_summary_numeric_imputation_strategy = str(
        st.session_state.get("numeric_imputation_strategy", "median")
    )
    preprocessing_summary_categorical_imputation_strategy = str(
        st.session_state.get("categorical_imputation_strategy", "most_frequent")
    )
    preprocessing_summary_categorical_encoding_strategy = str(
        st.session_state.get("categorical_encoding_strategy", "one_hot")
    )
    preprocessing_summary_standardize_numeric = bool(
        st.session_state.get("standardize_numeric", True)
    )
    applied_autogluon_params = _preprocessing_step_params(
        applied_preprocessing_plan, "autogluon_feature_generator"
    )
    preprocessing_summary_autogluon_params = _autogluon_feature_generator_params(
        {
            key: st.session_state.get(key, applied_autogluon_params.get(key, value))
            for key, value in AUTOGLUON_FEATURE_GENERATOR_DEFAULTS.items()
        }
    )
    preprocessing_summary_excluded_columns = [
        column
        for column in st.session_state.get("excluded_columns", [])
        if column in columns and column not in target_columns
    ]
    preprocessing_summary_columns = [
        column
        for column in columns
        if column not in preprocessing_summary_excluded_columns
    ]
    preprocessing_summary_base_df = df[preprocessing_summary_columns].copy()
    preprocessing_summary_manual_plan = st.session_state.get("_manual_cleaning_plan")
    preprocessing_summary_feature_plan = st.session_state.get(
        "_feature_engineering_applied_plan"
    )
    preprocessing_summary_draft_plan = _build_preprocessing_plan(
        base_analysis_df=preprocessing_summary_base_df,
        target_columns=target_columns,
        excluded_columns=[],
        test_size=preprocessing_summary_test_size,
        random_state=preprocessing_summary_random_state,
        high_missing_threshold=preprocessing_summary_high_missing_threshold,
        numeric_imputation_strategy=preprocessing_summary_numeric_imputation_strategy,
        categorical_imputation_strategy=preprocessing_summary_categorical_imputation_strategy,
        categorical_encoding_strategy=preprocessing_summary_categorical_encoding_strategy,
        standardize_numeric=preprocessing_summary_standardize_numeric,
        autogluon_feature_generator_params=preprocessing_summary_autogluon_params,
        manual_cleaning_plan=preprocessing_summary_manual_plan,
        feature_plan=preprocessing_summary_feature_plan,
    )
    preprocessing_summary_draft_plan = _upsert_preprocessing_step(
        preprocessing_summary_draft_plan,
        _preprocessing_step(
            "column_selection",
            params={"excluded_columns": list(preprocessing_summary_excluded_columns)},
            summary=f"Excluded columns: {len(preprocessing_summary_excluded_columns)}.",
        ),
    )
    preprocessing_summary = _preprocessing_summary_text(
        test_size=preprocessing_summary_test_size,
        high_missing_threshold=preprocessing_summary_high_missing_threshold,
        autogluon_feature_generator_params=preprocessing_summary_autogluon_params,
        manual_cleaning_enabled=_enabled_manual_rule_count(
            preprocessing_summary_manual_plan
        )
        > 0,
        feature_engineering_enabled=bool(
            _feature_plan_operations(preprocessing_summary_feature_plan)
        ),
        draft_plan=preprocessing_summary_draft_plan,
        applied_plan=applied_preprocessing_plan,
    )

    if active_step != "check":
        draft_excluded_columns = list(preprocessing_summary_excluded_columns)
        draft_analysis_columns = [
            column for column in columns if column not in draft_excluded_columns
        ]
        draft_base_analysis_df = df[draft_analysis_columns].copy()
        test_size = float(preprocessing_summary_test_size)
        high_missing_threshold = float(preprocessing_summary_high_missing_threshold)
        random_state = int(preprocessing_summary_random_state)
        numeric_imputation_strategy = preprocessing_summary_numeric_imputation_strategy
        categorical_imputation_strategy = preprocessing_summary_categorical_imputation_strategy
        categorical_encoding_strategy = preprocessing_summary_categorical_encoding_strategy
        standardize_numeric = bool(preprocessing_summary_standardize_numeric)
        autogluon_feature_generator_params = dict(
            preprocessing_summary_autogluon_params
        )
        preview_plan_data = None
        draft_feature_plan = st.session_state.get("_feature_engineering_applied_plan")
        draft_column_step = _preprocessing_step(
            "column_selection",
            params={"excluded_columns": list(draft_excluded_columns)},
            summary=f"Excluded columns: {len(draft_excluded_columns)}.",
        )
        draft_autogluon_plan = _build_preprocessing_plan(
            base_analysis_df=draft_base_analysis_df,
            target_columns=target_columns,
            excluded_columns=[],
            test_size=float(test_size),
            random_state=int(random_state),
            high_missing_threshold=float(high_missing_threshold),
            numeric_imputation_strategy=numeric_imputation_strategy,
            categorical_imputation_strategy=categorical_imputation_strategy,
            categorical_encoding_strategy=categorical_encoding_strategy,
            standardize_numeric=bool(standardize_numeric),
            autogluon_feature_generator_params=autogluon_feature_generator_params,
            manual_cleaning_plan=st.session_state.get("_manual_cleaning_plan"),
            feature_plan=draft_feature_plan,
        )
        draft_autogluon_step = _preprocessing_step_payload(
            draft_autogluon_plan, "autogluon_feature_generator"
        ) or _preprocessing_step("autogluon_feature_generator")
    else:
        preprocess_frame_idx = _preprocess_frame_index()
        preprocess_frame_label = BEAMER_NAV_SECTIONS["preprocess"][preprocess_frame_idx]
        _render_slide_title(
            "check",
            "Data Preprocessing",
            _t("Preprocess frame: {frame}.", frame=_t(preprocess_frame_label)),
        )
        _render_preprocessing_roadmap()
        draft_excluded_columns = list(preprocessing_summary_excluded_columns)
        draft_analysis_columns = [
            column for column in columns if column not in draft_excluded_columns
        ]
        draft_base_analysis_df = df[draft_analysis_columns].copy()
        test_size = float(preprocessing_summary_test_size)
        high_missing_threshold = float(preprocessing_summary_high_missing_threshold)
        random_state = int(preprocessing_summary_random_state)
        numeric_imputation_strategy = preprocessing_summary_numeric_imputation_strategy
        categorical_imputation_strategy = preprocessing_summary_categorical_imputation_strategy
        categorical_encoding_strategy = preprocessing_summary_categorical_encoding_strategy
        standardize_numeric = bool(preprocessing_summary_standardize_numeric)
        autogluon_feature_generator_params = dict(preprocessing_summary_autogluon_params)
        preview_plan_data = None
        draft_feature_plan = st.session_state.get("_feature_engineering_applied_plan")
        draft_column_step = _preprocessing_step(
            "column_selection",
            params={"excluded_columns": list(draft_excluded_columns)},
            summary=f"Excluded columns: {len(draft_excluded_columns)}.",
        )
        draft_autogluon_plan = _build_preprocessing_plan(
            base_analysis_df=draft_base_analysis_df,
            target_columns=target_columns,
            excluded_columns=[],
            test_size=float(test_size),
            random_state=int(random_state),
            high_missing_threshold=float(high_missing_threshold),
            numeric_imputation_strategy=numeric_imputation_strategy,
            categorical_imputation_strategy=categorical_imputation_strategy,
            categorical_encoding_strategy=categorical_encoding_strategy,
            standardize_numeric=bool(standardize_numeric),
            autogluon_feature_generator_params=autogluon_feature_generator_params,
            manual_cleaning_plan=st.session_state.get("_manual_cleaning_plan"),
            feature_plan=draft_feature_plan,
        )
        draft_autogluon_step = _preprocessing_step_payload(
            draft_autogluon_plan, "autogluon_feature_generator"
        ) or _preprocessing_step("autogluon_feature_generator")
        if preprocess_frame_idx == 1:
            with st.container(border=True):
                _section_caption(
                    _t(
                        "Default preprocessing is ready for a first pass. Expand this section only if you want to fine-tune data preparation."
                    )
                )
                st.success(
                    _t(
                        "Recommended preprocessing is already selected. You can continue without opening advanced settings."
                    )
                )
                st.caption(
                    _t(
                        "AutoGluon handles missing values, categorical encoding, datetime features, text features, model search, and ensembling during training."
                    )
                )
                st.caption(preprocessing_summary)
                with st.expander(_t("Open preprocessing details"), expanded=False):
                    with st.container(border=True):
                        st.write(_t("Planning help"))
                        _section_caption(
                            _t(
                                "This optional brief lets you describe your goal in plain language so the app can suggest a sensible first setup."
                            )
                        )
                        planner_brief = st.text_area(
                            _t("Planning brief"),
                            key="planner_brief",
                            placeholder=_t(
                                "Example: predict churn, treat customer_id as reference only, and keep this as a quick first pass."
                            ),
                            help=_t(
                                "Optional natural-language brief used to suggest targets, task type, exclusions, and a priority metric."
                            ),
                        )
                        plan_data = artifact_to_dict(plan_suggestion)
                        with st.expander(
                            _t("Planner suggestion"), expanded=bool(planner_brief.strip())
                        ):
                            _render_planner_suggestion(plan_data)
                            if st.button(_t("Apply planner suggestions")):
                                _queue_plan_suggestion(plan_data, columns)
                                st.rerun()

                        draft_excluded_columns = st.multiselect(
                            _t("Exclude columns from EDA and training features"),
                            exclude_options,
                            key="excluded_columns",
                            help=_t(
                                "Excluded columns are removed before EDA and are not used as model features."
                            ),
                        )
                        draft_analysis_columns = [
                            column for column in columns if column not in draft_excluded_columns
                        ]
                        draft_base_analysis_df = df[draft_analysis_columns].copy()

                    split_box = st.container(border=True)
                    with split_box:
                        st.write(_t("Split settings"))
                        split_cols = st.columns(3)
                        with split_cols[0]:
                            test_size = st.slider(
                                _t("Test size"),
                                min_value=0.1,
                                max_value=0.5,
                                step=0.01,
                                format="%.2f",
                                key="test_size",
                            )
                        with split_cols[1]:
                            high_missing_threshold = st.slider(
                                _t("Drop feature when missing rate is above"),
                                min_value=0.5,
                                max_value=1.0,
                                step=0.01,
                                format="%.2f",
                                key="high_missing_threshold",
                            )
                        with split_cols[2]:
                            random_state = _render_integer_input(
                                _t("Random state"), "random_state"
                            )
                        if float(applied_global_params.get("test_size", 0.2)) == float(
                            test_size
                        ) and int(
                            applied_global_params.get("random_state", 42)
                        ) == int(random_state):
                            _render_step_status(
                                "Split settings",
                                "Current settings are applied.",
                                level="success",
                            )
                        else:
                            _render_step_status(
                                "Split settings",
                                "Draft changes are not applied yet.",
                                level="info",
                            )
                        if st.button(_t("Apply split settings")):
                            _set_applied_preprocessing_global_params(
                                test_size=float(test_size), random_state=int(random_state)
                            )
                            st.rerun()

                    with st.container(border=True):
                        st.write(_t("Step 3.1: Column selection and manual cleaning"))
                        draft_column_step = _preprocessing_step(
                            "column_selection",
                            params={"excluded_columns": list(draft_excluded_columns)},
                            summary=f"Excluded columns: {len(draft_excluded_columns)}.",
                        )
                        _render_preprocessing_step_status(
                            "Column selection", draft_column_step, applied_preprocessing_plan
                        )
                        column_cols = st.columns(3)
                        with column_cols[0]:
                            st.metric(_t("Excluded columns"), len(draft_excluded_columns))
                        with column_cols[1]:
                            st.metric(_t("Columns after exclusion"), len(draft_analysis_columns))
                        with column_cols[2]:
                            if st.button(_t("Apply column selection")):
                                _update_applied_preprocessing_step(draft_column_step)
                                st.rerun()

                        st.write(_t("Manual cleaning rules"))
                        _section_caption(
                            _t(
                                "If you already know some rows or columns should be filtered out, draft the rules here before training."
                            )
                        )
                        manual_cleaning_brief = st.text_area(
                            _t("Cleaning rules brief"),
                            key="manual_cleaning_brief",
                            placeholder=_t(
                                "Example: drop customer_id and keep rows where monthly_spend > 20 and churn equals 1."
                            ),
                            help=_t(
                                "Natural-language rules are converted into a structured draft. Nothing is applied until you confirm."
                            ),
                        )
                        manual_rule_controls = st.columns(3)
                        with manual_rule_controls[0]:
                            if st.button(_t("Generate cleaning rules")):
                                suggested_manual_plan = _get_manual_cleaning_plan(
                                    df=draft_base_analysis_df,
                                    target=primary_target,
                                    settings=settings,
                                    user_brief=manual_cleaning_brief,
                                )
                                st.session_state["_manual_cleaning_override_plan"] = (
                                    _clone_json_data(suggested_manual_plan)
                                )
                                st.rerun()
                        with manual_rule_controls[1]:
                            if st.button(_t("Start with blank rule")):
                                st.session_state["_manual_cleaning_override_plan"] = (
                                    _blank_manual_cleaning_plan(manual_cleaning_brief)
                                )
                                st.rerun()
                        with manual_rule_controls[2]:
                            if st.session_state.get("_manual_cleaning_plan") and st.button(
                                _t("Clear applied manual rules")
                            ):
                                st.session_state["_manual_cleaning_plan"] = None
                                st.session_state["_manual_cleaning_override_plan"] = None
                                _update_applied_preprocessing_step(
                                    _preprocessing_step(
                                        "manual_cleaning",
                                        enabled=False,
                                        params={"plan": None},
                                        summary="No manual cleaning rules applied.",
                                    )
                                )
                                st.rerun()

                        if (
                            st.session_state.get("_manual_cleaning_override_plan") is None
                            and manual_cleaning_plan is not None
                        ):
                            st.session_state["_manual_cleaning_override_plan"] = (
                                _clone_json_data(manual_cleaning_plan)
                            )

                        validated_manual_preview = None
                        preview_plan_data = None
                        if isinstance(st.session_state.get("_manual_cleaning_override_plan"), dict):
                            draft_manual_plan = st.session_state["_manual_cleaning_override_plan"]
                            draft_manual_plan["user_brief"] = manual_cleaning_brief
                            edited_manual_plan = _render_manual_cleaning_editor(
                                draft_manual_plan,
                                available_columns=list(draft_base_analysis_df.columns),
                            )
                            validated_manual_preview = validate_manual_cleaning_plan(
                                edited_manual_plan,
                                draft_base_analysis_df,
                                primary_target,
                                protected_columns=target_columns,
                            )
                            preview_df, preview_log, preview_impact = apply_manual_cleaning_plan(
                                draft_base_analysis_df,
                                validated_manual_preview,
                                primary_target,
                                protected_columns=target_columns,
                            )
                            preview_plan_data = artifact_to_dict(validated_manual_preview)
                            draft_manual_step = _preprocessing_step(
                                "manual_cleaning",
                                enabled=_enabled_manual_rule_count(preview_plan_data) > 0,
                                params={"plan": preview_plan_data},
                                summary="No manual cleaning rules applied."
                                if _enabled_manual_rule_count(preview_plan_data) == 0
                                else f"Manual cleaning rules enabled: {_enabled_manual_rule_count(preview_plan_data)}.",
                            )
                            _render_preprocessing_step_status(
                                "Manual cleaning",
                                draft_manual_step,
                                applied_preprocessing_plan,
                            )

                            summary_cols = st.columns(5)
                            with summary_cols[0]:
                                st.metric(
                                    _t("Planner"),
                                    str(preview_plan_data.get("planner_name") or "manual"),
                                )
                            with summary_cols[1]:
                                st.metric(
                                    _t("Effect stage"),
                                    _manual_cleaning_effect_stage_label(
                                        validated_manual_preview.effect_stage
                                    ),
                                )
                            with summary_cols[2]:
                                st.metric(
                                    _t("Accepted rules"),
                                    len(preview_plan_data.get("rules", [])),
                                )
                            with summary_cols[3]:
                                st.metric(
                                    _t("Rejected rules"),
                                    len(preview_plan_data.get("rejected_rules", [])),
                                )
                            with summary_cols[4]:
                                st.metric(
                                    _t("Rows removed"),
                                    int(preview_impact.get("rows_removed") or 0),
                                )

                            rule_rows = _manual_cleaning_rule_rows(preview_plan_data)
                            if rule_rows:
                                st.dataframe(
                                    pd.DataFrame(rule_rows),
                                    hide_index=True,
                                    use_container_width=True,
                                )
                            else:
                                st.caption(
                                    _t("No manual cleaning rules are in the current draft.")
                                )

                            impact_cols = st.columns(3)
                            with impact_cols[0]:
                                st.metric(
                                    _t("Columns removed"),
                                    len(preview_impact.get("columns_removed", [])),
                                )
                            with impact_cols[1]:
                                st.metric(
                                    _t("Rows after"),
                                    int(
                                        preview_impact.get("rows_after")
                                        or len(draft_base_analysis_df)
                                    ),
                                )
                            with impact_cols[2]:
                                st.metric(
                                    _t("Columns after"),
                                    int(
                                        preview_impact.get("columns_after")
                                        or len(draft_base_analysis_df.columns)
                                    ),
                                )

                            detail_cols = st.columns(2)
                            with detail_cols[0]:
                                _render_text_items(
                                    "Notes", preview_plan_data.get("notes", []), "No notes."
                                )
                            with detail_cols[1]:
                                _render_text_items(
                                    "Rejected rules",
                                    preview_plan_data.get("rejected_rules", []),
                                    "No rejected rules.",
                                )

                            with st.expander(
                                _t("Manual cleaning preview impact"), expanded=False
                            ):
                                st.json(preview_impact, expanded=True)
                                st.dataframe(
                                    preview_df.head(20), use_container_width=True
                                )
                                if preview_log:
                                    st.write(_t("Planned cleaning log"))
                                    st.json(preview_log, expanded=True)

                            apply_cols = st.columns(2)
                            with apply_cols[0]:
                                if st.button(
                                    _t("Apply manual cleaning rules"), type="primary"
                                ):
                                    st.session_state["_manual_cleaning_plan"] = (
                                        preview_plan_data
                                    )
                                    st.session_state["_manual_cleaning_override_plan"] = (
                                        _clone_json_data(preview_plan_data)
                                    )
                                    _update_applied_preprocessing_step(draft_manual_step)
                                    st.rerun()
                            with apply_cols[1]:
                                if st.session_state.get("_manual_cleaning_plan"):
                                    st.caption(
                                        _t(
                                            "Applied rules remain active until you clear them or apply a different draft."
                                        )
                                    )
                        else:
                            st.caption(
                                _t(
                                    "Generate rules from a brief or start with a blank rule to configure manual cleaning."
                                )
                            )

                    with st.container(border=True):
                        st.write(_t("Step 3.2: Feature cleanup"))
                        _section_caption(
                            _t(
                                "The platform only removes unusable columns before AutoGluon. Missing values, categorical encoding, datetime features, and text features are handled by AutoGluon during training."
                            )
                        )
                        numeric_imputation_strategy = "median"
                        categorical_imputation_strategy = "most_frequent"
                        categorical_encoding_strategy = "autogluon"
                        standardize_numeric = False
                        draft_missing_plan = _build_preprocessing_plan(
                            base_analysis_df=draft_base_analysis_df,
                            target_columns=target_columns,
                            excluded_columns=[],
                            test_size=float(test_size),
                            random_state=int(random_state),
                            high_missing_threshold=float(high_missing_threshold),
                            numeric_imputation_strategy=numeric_imputation_strategy,
                            categorical_imputation_strategy=categorical_imputation_strategy,
                            categorical_encoding_strategy=categorical_encoding_strategy,
                            standardize_numeric=bool(standardize_numeric),
                            autogluon_feature_generator_params=preprocessing_summary_autogluon_params,
                            manual_cleaning_plan=preview_plan_data
                            if preview_plan_data is not None
                            else st.session_state.get("_manual_cleaning_plan"),
                            feature_plan=st.session_state.get(
                                "_feature_engineering_applied_plan"
                            ),
                        )
                        draft_missing_step = _preprocessing_step_payload(
                            draft_missing_plan, "missing_value"
                        ) or _preprocessing_step("missing_value")
                        _render_preprocessing_step_status(
                            "Missing-value handling",
                            draft_missing_step,
                            applied_preprocessing_plan,
                        )
                        missing_params = draft_missing_step.get("params", {})
                        missing_cols = st.columns(3)
                        with missing_cols[0]:
                            st.metric(
                                _t("Feature columns"),
                                len(
                                    [
                                        column
                                        for column in draft_base_analysis_df.columns
                                        if column not in target_columns
                                    ]
                                ),
                            )
                        with missing_cols[1]:
                            st.metric(
                                _t("High-missing columns"),
                                len(missing_params.get("high_missing_columns", [])),
                            )
                        with missing_cols[2]:
                            if st.button(_t("Apply missing-value step")):
                                _update_applied_preprocessing_step(draft_missing_step)
                                st.rerun()
                        high_missing_preview = missing_params.get("high_missing_columns", [])
                        if high_missing_preview:
                            st.dataframe(
                                pd.DataFrame({"column": list(high_missing_preview)}),
                                hide_index=True,
                                use_container_width=True,
                            )
                        else:
                            st.caption(
                                _t(
                                    "No feature columns will be auto-dropped by the current high-missing threshold."
                                )
                            )

                    with st.container(border=True):
                        st.write(_t("Step 3.3: AutoGluon feature generation"))
                        _section_caption(
                            _t(
                                "These options are passed to AutoGluon's AutoMLPipelineFeatureGenerator."
                            )
                        )
                        for key, value in AUTOGLUON_FEATURE_GENERATOR_DEFAULTS.items():
                            st.session_state.setdefault(key, value)
                        ag_cols = st.columns(3)
                        with ag_cols[0]:
                            enable_numeric_features = st.checkbox(
                                _t("Enable numeric features"),
                                key="enable_numeric_features",
                            )
                            enable_categorical_features = st.checkbox(
                                _t("Enable categorical features"),
                                key="enable_categorical_features",
                            )
                            enable_datetime_features = st.checkbox(
                                _t("Enable datetime features"),
                                key="enable_datetime_features",
                            )
                        with ag_cols[1]:
                            enable_text_special_features = st.checkbox(
                                _t("Enable text special features"),
                                key="enable_text_special_features",
                            )
                            enable_text_ngram_features = st.checkbox(
                                _t("Enable text ngram features"),
                                key="enable_text_ngram_features",
                            )
                        with ag_cols[2]:
                            enable_raw_text_features = st.checkbox(
                                _t("Enable raw text features"),
                                key="enable_raw_text_features",
                            )
                            enable_vision_features = st.checkbox(
                                _t("Enable vision features"),
                                key="enable_vision_features",
                            )
                        autogluon_feature_generator_params = _autogluon_feature_generator_params(
                            {
                                "enable_numeric_features": enable_numeric_features,
                                "enable_categorical_features": enable_categorical_features,
                                "enable_datetime_features": enable_datetime_features,
                                "enable_text_special_features": enable_text_special_features,
                                "enable_text_ngram_features": enable_text_ngram_features,
                                "enable_raw_text_features": enable_raw_text_features,
                                "enable_vision_features": enable_vision_features,
                            }
                        )
                        draft_autogluon_plan = _build_preprocessing_plan(
                            base_analysis_df=draft_base_analysis_df,
                            target_columns=target_columns,
                            excluded_columns=[],
                            test_size=float(test_size),
                            random_state=int(random_state),
                            high_missing_threshold=float(high_missing_threshold),
                            numeric_imputation_strategy=numeric_imputation_strategy,
                            categorical_imputation_strategy=categorical_imputation_strategy,
                            categorical_encoding_strategy=categorical_encoding_strategy,
                            standardize_numeric=bool(standardize_numeric),
                            autogluon_feature_generator_params=autogluon_feature_generator_params,
                            manual_cleaning_plan=preview_plan_data
                            if preview_plan_data is not None
                            else st.session_state.get("_manual_cleaning_plan"),
                            feature_plan=st.session_state.get(
                                "_feature_engineering_applied_plan"
                            ),
                        )
                        draft_autogluon_step = _preprocessing_step_payload(
                            draft_autogluon_plan, "autogluon_feature_generator"
                        ) or _preprocessing_step("autogluon_feature_generator")
                        _render_preprocessing_step_status(
                            "AutoGluon feature generation",
                            draft_autogluon_step,
                            applied_preprocessing_plan,
                        )
                        autogluon_metrics = st.columns(3)
                        with autogluon_metrics[0]:
                            st.metric(
                                _t("Categorical columns"),
                                int(
                                    draft_autogluon_step["params"].get(
                                        "categorical_feature_count", 0
                                    )
                                ),
                            )
                        with autogluon_metrics[1]:
                            st.metric(
                                _t("Numeric columns"),
                                int(
                                    draft_autogluon_step["params"].get(
                                        "numeric_feature_count", 0
                                    )
                                ),
                            )
                        with autogluon_metrics[2]:
                            st.metric(
                                _t("Enabled generators"),
                                len(_autogluon_enabled_feature_names(autogluon_feature_generator_params)),
                            )
                        if st.button(_t("Apply AutoGluon feature generation")):
                            _update_applied_preprocessing_step(draft_autogluon_step)
                            st.rerun()

                    feature_plan = _preprocessing_feature_plan(applied_preprocessing_plan)
                    with st.container(border=True):
                        st.write(_t("Step 3.5: Feature engineering"))
                        apply_feature_engineering = st.checkbox(
                            _t("Apply local whitelist feature engineering"),
                            key="apply_feature_engineering",
                            help=_t(
                                "LLM can propose a structured plan, but only local whitelisted transformations are executed inside the training pipeline."
                            ),
                        )
                        draft_feature_plan = None
                        if apply_feature_engineering:
                            feature_plan_override = st.session_state.get(
                                "_feature_engineering_override_plan"
                            )
                            if isinstance(feature_plan_override, dict) and feature_plan_override.get(
                                "operations"
                            ):
                                draft_feature_plan = feature_plan_override
                            else:
                                feature_source_columns = [
                                    column
                                    for column in draft_base_analysis_df.columns
                                    if column not in target_columns
                                ]
                                feature_plan_df = draft_base_analysis_df[
                                    feature_source_columns + [primary_target]
                                ].copy()
                                draft_feature_plan = artifact_to_dict(
                                    _get_feature_engineering_plan(
                                        df=feature_plan_df,
                                        target=primary_target,
                                        settings=settings,
                                        user_brief=planner_brief,
                                    )
                                )
                        draft_feature_step = _preprocessing_step(
                            "feature_engineering",
                            enabled=bool(
                                draft_feature_plan
                                and _feature_plan_operations(draft_feature_plan)
                            ),
                            params={
                                "planner_name": str(
                                    (draft_feature_plan or {}).get("planner_name")
                                    or "local_whitelist"
                                ),
                                "operations": _feature_plan_operations(draft_feature_plan)
                                if draft_feature_plan
                                else [],
                                "rejected_operations": list(
                                    (draft_feature_plan or {}).get("rejected_operations", [])
                                ),
                                "notes": list((draft_feature_plan or {}).get("notes", [])),
                            },
                            summary=(
                                "Feature engineering is disabled."
                                if not draft_feature_plan
                                or not _feature_plan_operations(draft_feature_plan)
                                else f"Feature engineering operations: {len(_feature_plan_operations(draft_feature_plan))}."
                            ),
                        )
                        _render_preprocessing_step_status(
                            "Feature engineering",
                            draft_feature_step,
                            applied_preprocessing_plan,
                        )
                        if draft_feature_plan:
                            with st.expander(_t("Feature engineering plan"), expanded=True):
                                _render_feature_engineering_plan(draft_feature_plan)
                        else:
                            st.caption(_t("Feature engineering is currently disabled."))
                        feature_cols = st.columns(2)
                        with feature_cols[0]:
                            if st.button(_t("Apply feature engineering step")):
                                st.session_state["_feature_engineering_applied_plan"] = (
                                    draft_feature_plan
                                )
                                _update_applied_preprocessing_step(draft_feature_step)
                                st.rerun()
                        with feature_cols[1]:
                            if feature_plan:
                                st.caption(
                                    _t(
                                        "Applied feature engineering remains active until you apply a different draft."
                                    )
                                )

        draft_preprocessing_plan = _build_preprocessing_plan(
            base_analysis_df=draft_base_analysis_df,
            target_columns=target_columns,
            excluded_columns=[],
            test_size=float(test_size),
            random_state=int(random_state),
            high_missing_threshold=float(high_missing_threshold),
            numeric_imputation_strategy=numeric_imputation_strategy,
            categorical_imputation_strategy=categorical_imputation_strategy,
            categorical_encoding_strategy=categorical_encoding_strategy,
            standardize_numeric=bool(standardize_numeric),
            autogluon_feature_generator_params=autogluon_feature_generator_params,
            manual_cleaning_plan=preview_plan_data
            if preview_plan_data is not None
            else st.session_state.get("_manual_cleaning_plan"),
            feature_plan=draft_feature_plan,
        )
        draft_preprocessing_plan = _upsert_preprocessing_step(
            draft_preprocessing_plan, draft_column_step
        )
        draft_preprocessing_plan = _upsert_preprocessing_step(
            draft_preprocessing_plan, draft_autogluon_step
        )
        st.session_state["_preprocessing_plan_draft"] = draft_preprocessing_plan
    applied_preprocessing_plan = (
        st.session_state.get("_preprocessing_plan_applied")
        or applied_preprocessing_plan
    )
    feature_plan = _preprocessing_feature_plan(applied_preprocessing_plan)

    current_experiment_signature = _experiment_signature(
        dataset_fingerprint=current_dataset_fingerprint,
        target_columns=target_columns,
        task_type_choice=task_type_choice,
        time_budget=int(time_budget),
        priority_metric_choice=priority_metric_choice,
        planner_brief=planner_brief,
        preprocessing_plan=applied_preprocessing_plan,
    )
    applied_missing_params = _preprocessing_step_params(
        applied_preprocessing_plan, "missing_value"
    )

    priority_metrics = {
        target: resolve_priority_metric(
            _target_task_types(analysis_df, [target], task_type_choice)[target],
            priority_metric_choice,
        )
        for target in target_columns
    }
    preflight_by_target: dict[str, object] = {}
    for target in target_columns:
        preflight_input = analysis_df[
            [
                column
                for column in analysis_df.columns
                if column not in target_columns or column == target
            ]
        ].copy()
        if (
            manual_cleaning_plan is not None
            and any(rule.enabled for rule in manual_cleaning_plan.rules)
            and manual_cleaning_plan.effect_stage == "pre_training"
        ):
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
            high_missing_threshold=float(
                applied_missing_params.get("high_missing_threshold", 0.9)
            ),
        )
    failing_targets = [
        target
        for target, validation in preflight_by_target.items()
        if not validation.ok_to_run
    ]
    st.session_state["_beamer_failing_targets"] = list(failing_targets)
    if active_step == "check":
        preprocess_frame_idx = _preprocess_frame_index()
        if preprocess_frame_idx in {0, 3}:
            if failing_targets:
                _render_step_status(
                    _t("The app found blocking issues in the current setup."),
                    _t(
                        "Fix the checks for: {failing_targets} before starting training.",
                        failing_targets=", ".join(failing_targets),
                    ),
                    level="warning",
                )
            else:
                _render_step_status(
                    _t("The dataset and target selection passed the current checks."),
                    _t("Continue through preprocessing frames, then prepare training."),
                    level="success",
                )

        if preprocess_frame_idx == 0:
            _frame_note(
                _t(
                    "Field health is separated from preprocessing controls. Use this frame to inspect schema, missingness, ID-like columns, and leakage warnings."
                )
            )
            _render_field_health_table(
                analysis_df,
                target_columns=target_columns,
                preflight_by_target=preflight_by_target,
                high_missing_threshold=float(
                    applied_missing_params.get("high_missing_threshold", 0.9)
                ),
            )

        elif preprocess_frame_idx == 1:
            _frame_note(
                _t(
                    "This frame contains preprocessing controls only. Field health, EDA profile, and preflight validation are intentionally separated into different frames."
                )
            )

        elif preprocess_frame_idx == 2:
            _frame_note(
                _t(
                    "This frame contains data preview and EDA diagnostics. It does not contain preprocessing controls or final preflight checks."
                )
            )
            with st.container(border=True):
                _panel_title(_t("Data preview"))
                _section_caption(
                    _t("First 50 rows are shown for a quick sanity check before training.")
                )
                if (
                    manual_cleaning_plan is not None
                    and any(rule.enabled for rule in manual_cleaning_plan.rules)
                    and manual_cleaning_plan.effect_stage == "pre_training"
                ):
                    st.info(
                        _t(
                            "Manual cleaning rules are set to apply only before training. The data preview and EDA below still show the pre-cleaning analysis subset."
                        )
                    )
                st.dataframe(analysis_df.head(50), use_container_width=True)

            with st.container(border=True):
                _panel_title(_t("EDA summary"))
                _section_caption(
                    _t(
                        "EDA means a quick health check for the dataset: shape, duplicates, missing values, correlations, and target behavior."
                    )
                )
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric(_t("Rows"), eda_summary["shape"]["rows"])
                with metric_cols[1]:
                    st.metric(_t("Columns"), eda_summary["shape"]["columns"])
                with metric_cols[2]:
                    st.metric(_t("Duplicate rows"), eda_summary["duplicate_rows"])
                with metric_cols[3]:
                    st.metric(
                        _t("Rows with missing"),
                        eda_summary["missingness"]["rows_with_any_missing"],
                    )

                _section_title(_t("Column profile"))
                st.dataframe(pd.DataFrame(eda_summary["columns"]).T, use_container_width=True)

                if primary_target in analysis_df.columns and eda_summary.get("target"):
                    label = (
                        _t("Primary target profile")
                        if len(target_columns) > 1
                        else _t("Target profile")
                    )
                    _section_title(label)
                    _render_target_profile(eda_summary["target"])
                if len(target_columns) > 1:
                    _section_title(_t("Target task types"))
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {_t("Target"): target, _t("Task type"): _t(task_type)}
                                for target, task_type in task_types.items()
                            ]
                        ),
                        use_container_width=True,
                    )

                distribution_feature_columns = [
                    column for column in analysis_df.columns if column not in target_columns
                ]
                eda_tabs = st.tabs(
                    [
                        _t("Missingness"),
                        _t("Correlations"),
                        _t("Target relationships"),
                        _t("Distributions"),
                        _t("Quality warnings"),
                    ]
                )
                with eda_tabs[0]:
                    top_missing = eda_summary["missingness"]["top_missing_columns"]
                    if top_missing:
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {_t("Column"): column, _t("Missing rate"): rate}
                                    for column, rate in top_missing.items()
                                ]
                            ),
                            use_container_width=True,
                        )
                    correlated_missing = eda_summary["missingness"]["correlated_missing_pairs"]
                    if correlated_missing:
                        _section_title(_t("Correlated missingness pairs"))
                        st.dataframe(pd.DataFrame(correlated_missing), use_container_width=True)
                with eda_tabs[1]:
                    top_pairs = eda_summary["correlations"]["top_numeric_pairs"]
                    target_corr = eda_summary["correlations"]["target_numeric_correlations"]
                    if target_corr:
                        _section_title(_t("Numeric correlations with target"))
                        st.dataframe(pd.DataFrame(target_corr), use_container_width=True)
                    if top_pairs:
                        _section_title(_t("Strong numeric feature correlations"))
                        st.dataframe(pd.DataFrame(top_pairs), use_container_width=True)
                with eda_tabs[2]:
                    _render_target_relationships(eda_summary.get("target_relationships", {}))
                with eda_tabs[3]:
                    _render_distribution_explorer(
                        analysis_df,
                        feature_columns=distribution_feature_columns,
                        target_columns=target_columns,
                        primary_target=primary_target,
                    )
                with eda_tabs[4]:
                    for warning in eda_summary["quality_warnings"]:
                        st.warning(warning)

        else:
            _frame_note(
                _t(
                    "Preflight validation is the final gate before materializing train/test batches."
                )
            )
            with st.container(border=True):
                _panel_title(_t("Preflight validation"))
                _section_caption(
                    _t(
                        "This check looks for blocking issues before training, such as missing target values or no usable feature columns."
                    )
                )
                st.caption(
                    _t(
                        "These checks explain why training can continue or why it should pause. Blocking items stop the run; caution items let you continue but make the result less trustworthy."
                    )
                )
                for target in target_columns:
                    validation = artifact_to_dict(preflight_by_target[target])
                    _section_title(f"{target} ({_t(task_types[target])})")
                    st.caption(
                        _t(
                            "Resolved priority metric: {priority_metric}",
                            priority_metric=priority_metrics[target],
                        )
                    )
                    validation_level, validation_summary = _preflight_explanation_summary(
                        validation
                    )
                    _render_message(validation_level, validation_summary)
                    explanation_items = _preflight_explanation_items(validation)
                    if explanation_items:
                        _render_text_items(
                            "What this means",
                            explanation_items,
                            "No extra explanation available.",
                        )
                    _render_issue_table(validation.get("issues", []))

            if st.button(_t("Next: prepare training"), key="next_after_checks", disabled=bool(failing_targets)):
                _set_active_step("prepare")
                st.rerun()

        _render_bottom_workflow_nav()
        return

    elif _wizard_step_index(active_step) <= _wizard_step_index("check"):
        _render_bottom_workflow_nav()
        return

    prepared_batches: list[dict[str, object]] = []
    prepared_ready = (
        st.session_state.get("_latest_prepared_signature")
        == current_experiment_signature
    )
    if prepared_ready:
        prepared_batches = list(st.session_state.get("_latest_prepared_batches", []))

    if _active_step() == "prepare":
        _render_slide_title(
            "prepare",
            "Training Preparation",
            "Apply the current preprocessing plan and materialize train/test batches.",
        )
        with st.container(border=True):
            _section_caption(
                _t(
                    "Apply the selected cleaning and preprocessing methods first, then train models as a separate step."
                )
            )
            if failing_targets:
                _render_step_status(
                    "Data preparation is blocked by validation issues.",
                    f"Resolve the flagged issues for: {', '.join(failing_targets)} before preparing data.",
                    level="warning",
                )
            elif prepared_ready:
                _render_step_status(
                    "Data preparation is complete for the current setup.",
                    "Review the prepared train/test summary below, then continue to model training.",
                    level="success",
                )
            else:
                _render_step_status(
                    "Prepared data is not cached yet.",
                    "Prepare data to preview the train/test batches, or start training directly and let the app prepare them automatically.",
                    level="info",
                )

            prepare_cols = st.columns([0.8, 1.2])
            with prepare_cols[0]:
                prepare_clicked = st.button(
                    _t("Prepare data"), type="primary", disabled=bool(failing_targets)
                )
            with prepare_cols[1]:
                if prepared_ready:
                    st.caption(
                        _t(
                            "Prepared targets: {prepared_targets}",
                            prepared_targets=", ".join(
                                str(batch["target"]) for batch in prepared_batches
                            ),
                        )
                    )
            if prepare_clicked:
                try:
                    prepared_batches = _materialize_prepared_batches(
                        current_experiment_signature=current_experiment_signature,
                        df=df,
                        base_analysis_df=base_analysis_df,
                        analysis_df=analysis_df,
                        analysis_manual_cleaning_log=analysis_manual_cleaning_log,
                        analysis_manual_cleaning_impact=analysis_manual_cleaning_impact,
                        candidate_feature_columns=candidate_feature_columns,
                        excluded_columns=applied_excluded_columns,
                        target_columns=target_columns,
                        task_types=task_types,
                        priority_metrics=priority_metrics,
                        preflight_by_target=preflight_by_target,
                        preprocessing_plan=applied_preprocessing_plan,
                        manual_cleaning_plan=manual_cleaning_plan,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                    _render_bottom_workflow_nav()
                    return

                prepared_ready = True
                _set_active_step("train")
                st.session_state["_latest_prepared_signature"] = current_experiment_signature
                st.session_state["_latest_prepared_batches"] = list(prepared_batches)
                st.success(
                    _t(
                        "Data preparation completed: {prepared_targets}",
                        prepared_targets=", ".join(
                            str(batch["target"]) for batch in prepared_batches
                        ),
                    )
                )
                st.rerun()

            if prepared_ready and prepared_batches:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                _t("prepared target"): str(batch["target"]),
                                _t("prepared task type"): str(batch["task_type"]),
                                _t("prepared train rows"): len(batch["cleaned"].X_train),
                                _t("prepared test rows"): len(batch["cleaned"].X_test),
                                _t("prepared features"): len(
                                    batch["cleaned"].prepared_feature_names or []
                                ),
                            }
                            for batch in prepared_batches
                        ]
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
                if st.button(_t("Next: start training"), key="next_after_prepare"):
                    _set_active_step("train")
                    st.rerun()

        _render_bottom_workflow_nav()
        return

    results: list[dict[str, object]] = []
    if (
        st.session_state.get("_latest_results_signature")
        == current_experiment_signature
    ):
        results = list(st.session_state.get("_latest_results", []))

    if _active_step() == "results":
        _render_run_outputs(results)
        _render_bottom_workflow_nav()
        return

    _render_slide_title(
        "train",
        "Model Training",
        "Run AutoML training and write reproducible artifacts to storage.",
    )

    with st.container(border=True):
        _section_caption(
            _t(
                "Training now uses the prepared train/test data from the previous step instead of rerunning preprocessing inside the fit step."
            )
        )
        if failing_targets:
            _render_step_status(
                _t("Training is blocked by validation issues."),
                _t(
                    "Resolve the flagged issues for: {failing_targets}.",
                    failing_targets=", ".join(failing_targets),
                ),
                level="warning",
            )
        elif not prepared_ready:
            _render_step_status(
                "Training is waiting for data preparation.",
                "Run training will prepare data automatically with the applied preprocessing settings if needed.",
                level="info",
            )
        else:
            _render_step_status(
                _t("The run is ready to start."),
                "Click Run training to fit models on the prepared data and unlock the results step.",
                level="success",
            )
        st.caption(
            _t(
                "Run training will prepare data automatically with the applied preprocessing settings if needed."
            )
        )

        if st.button(
            _t("Run training"),
            type="primary",
            disabled=bool(failing_targets),
        ):
            results = []
            spinner_text = (
                _t("Training models on the prepared data...")
                if prepared_ready
                else _t(
                    "Preparing data and training models with the applied preprocessing settings..."
                )
            )
            st.markdown('<div class="beamer-training-progress">', unsafe_allow_html=True)
            training_progress = st.progress(0)
            training_status = st.empty()
            training_status.info(spinner_text)
            st.markdown('</div>', unsafe_allow_html=True)
            with st.spinner(spinner_text):
                if failing_targets:
                    logger.error(
                        "Blocking preflight issues targets=%s", failing_targets
                    )
                    st.error(
                        _t(
                            "Resolve blocking preflight issues before training: {failing_targets}",
                            failing_targets=", ".join(failing_targets),
                        )
                    )
                    _render_bottom_workflow_nav()
                    return

                if not prepared_ready:
                    training_progress.progress(8)
                    training_status.info(_t("Preparing train/test batches with the applied preprocessing plan..."))
                    try:
                        prepared_batches = _materialize_prepared_batches(
                            current_experiment_signature=current_experiment_signature,
                            df=df,
                            base_analysis_df=base_analysis_df,
                            analysis_df=analysis_df,
                            analysis_manual_cleaning_log=analysis_manual_cleaning_log,
                            analysis_manual_cleaning_impact=analysis_manual_cleaning_impact,
                            candidate_feature_columns=candidate_feature_columns,
                            excluded_columns=applied_excluded_columns,
                            target_columns=target_columns,
                            task_types=task_types,
                            priority_metrics=priority_metrics,
                            preflight_by_target=preflight_by_target,
                            preprocessing_plan=applied_preprocessing_plan,
                            manual_cleaning_plan=manual_cleaning_plan,
                        )
                    except ValueError as exc:
                        logger.error("Automatic data preparation failed: %s", exc)
                        st.error(str(exc))
                        _render_bottom_workflow_nav()
                        return
                    prepared_ready = True
                    training_progress.progress(18)
                    training_status.info(_t("Prepared data batches. Starting AutoML fitting..."))

                if not prepared_batches:
                    logger.error("Training requested without prepared batches")
                    st.error(_t("Prepare data before training."))
                    _render_bottom_workflow_nav()
                    return

                logger.info(
                    "Starting training pipeline targets=%s task_types=%s",
                    target_columns,
                    task_types,
                )
                total_batches = max(len(prepared_batches), 1)
                for batch_index, batch in enumerate(prepared_batches):
                    target = str(batch["target"])
                    batch_base_progress = 20 + int(60 * batch_index / total_batches)
                    training_progress.progress(min(batch_base_progress, 90))
                    training_status.info(
                        _t(
                            "Training target {current}/{total}: {target}",
                            current=batch_index + 1,
                            total=total_batches,
                            target=target,
                        )
                    )
                    task_type = str(batch["task_type"])
                    priority_metric = str(batch["priority_metric"])
                    preflight_validation = batch["preflight_validation"]
                    tracker = batch["tracker"]
                    target_eda_summary = batch["target_eda_summary"]
                    cleaned = batch["cleaned"]
                    logger.info("Training target=%s task_type=%s", target, task_type)
                    run = storage.create_run(
                        config={
                            "target": target,
                            "target_columns": target_columns,
                            "task_type": task_type,
                            "task_type_choice": task_type_choice,
                            "excluded_columns": applied_excluded_columns,
                            "test_size": cleaned.config.test_size,
                            "high_missing_threshold": cleaned.config.high_missing_threshold,
                            "random_state": cleaned.config.random_state,
                            "autogluon_presets": cleaned.config.autogluon_presets,
                            "autogluon_feature_generator_params": dict(
                                cleaned.config.autogluon_feature_generator_params or {}
                            ),
                            "time_budget": time_budget,
                            "priority_metric": priority_metric,
                            "trainer": "autogluon_tabular",
                            "planner_name": plan_suggestion.planner_name,
                            "feature_engineering_enabled": bool(feature_plan),
                            "manual_cleaning_enabled": bool(
                                manual_cleaning_plan
                                and any(
                                    rule.enabled for rule in manual_cleaning_plan.rules
                                )
                            ),
                            "manual_cleaning_effect_stage": manual_cleaning_plan.effect_stage
                            if manual_cleaning_plan
                            else None,
                            "dataset_fingerprint": current_dataset_fingerprint,
                            "prepared_before_training": True,
                        }
                    )
                    trained = train_model(
                        cleaned,
                        time_budget=int(time_budget),
                        metric_preference=priority_metric,
                        tracker=tracker,
                        output_path=run.path / "autogluon_predictor",
                        presets=AUTOGLUON_DEFAULT_PRESETS,
                    )
                    training_progress.progress(min(batch_base_progress + int(35 / total_batches), 94))
                    training_status.info(
                        _t(
                            "Evaluating target {current}/{total}: {target}",
                            current=batch_index + 1,
                            total=total_batches,
                            target=target,
                        )
                    )
                    metrics, prediction_sample = evaluate_model(
                        trained.model, cleaned, task_type=task_type, tracker=tracker
                    )
                    tracker.snapshot_artifact(
                        "metrics_summary",
                        "Metrics summary",
                        "evaluation",
                        metadata={
                            "metrics": metrics,
                            "priority_metric": priority_metric,
                        },
                    )
                    planned_report_mode = (
                        "llm" if settings.llm_enabled else "rule_based"
                    )
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
                    recommendations = build_recommendations(
                        preflight_validation, postrun_validation
                    )

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
                    recommendations = build_recommendations(
                        preflight_validation, postrun_validation
                    )
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

                    training_progress.progress(min(batch_base_progress + int(50 / total_batches), 96))
                    training_status.info(
                        _t(
                            "Saving artifacts for target {current}/{total}: {target}",
                            current=batch_index + 1,
                            total=total_batches,
                            target=target,
                        )
                    )
                    storage.save_json(
                        run, "plan.json", artifact_to_dict(plan_suggestion)
                    )
                    storage.save_json(
                        run,
                        "preprocessing_plan.json",
                        artifact_to_dict(applied_preprocessing_plan),
                    )
                    if feature_plan:
                        storage.save_json(
                            run,
                            "feature_engineering_plan.json",
                            artifact_to_dict(feature_plan),
                        )
                    if manual_cleaning_plan:
                        storage.save_json(
                            run,
                            "manual_cleaning_plan.json",
                            artifact_to_dict(manual_cleaning_plan),
                        )
                    storage.save_json(run, "eda_summary.json", target_eda_summary)
                    storage.save_json(run, "cleaning_log.json", cleaned.cleaning_log)
                    storage.save_json(run, "metrics.json", metrics)
                    data_flow_payload = artifact_to_dict(tracker.to_trace())
                    storage.save_json(run, "data_flow.json", data_flow_payload)
                    storage.save_json(
                        run, "feature_importance.json", trained.feature_importance
                    )
                    storage.save_json(run, "leaderboard.json", trained.leaderboard)
                    if trained.leaderboard:
                        pd.DataFrame(trained.leaderboard).to_csv(
                            run.path / "leaderboard.csv", index=False
                        )
                    storage.save_json(run, "fit_summary.json", trained.fit_summary)
                    storage.save_json(
                        run,
                        "validation_pre.json",
                        artifact_to_dict(preflight_validation),
                    )
                    storage.save_json(
                        run,
                        "validation_post.json",
                        artifact_to_dict(postrun_validation),
                    )
                    storage.save_json(
                        run, "recommendations.json", artifact_to_dict(recommendations)
                    )
                    storage.save_json(
                        run,
                        "training_summary.json",
                        {
                            "trainer_name": trained.trainer_name,
                            "optimization_metric_used": trained.optimization_metric_used,
                            "training_notes": trained.training_notes or [],
                            "manual_cleaning_effect_stage": manual_cleaning_plan.effect_stage
                            if manual_cleaning_plan
                            else None,
                            "manual_cleaning_applied_rules": 0
                            if not manual_cleaning_plan
                            else sum(
                                1 for rule in manual_cleaning_plan.rules if rule.enabled
                            ),
                            "prepared_before_training": True,
                            "leaderboard_rows": len(trained.leaderboard),
                            "model_path": str(trained.model_path)
                            if trained.model_path
                            else None,
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

            training_progress.progress(100)
            training_status.success(_t("Training complete. Results and artifacts are ready."))
            st.session_state["_latest_results_signature"] = current_experiment_signature
            st.session_state["_latest_results"] = list(results)
            _set_active_step("results")
            completed_ids = ", ".join(str(result["run"].run_id) for result in results)
            logger.info("Training pipeline complete run_ids=%s", completed_ids)
            st.success(
                _t("Run completed: {completed_ids}", completed_ids=completed_ids)
            )
            st.rerun()

    _render_bottom_workflow_nav()


if __name__ == "__main__":
    main()
