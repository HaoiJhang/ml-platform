from __future__ import annotations

import json
from importlib import import_module
import logging
import re
from typing import Any

from ml_platform.artifacts import artifact_to_dict
from ml_platform.config import Settings
from ml_platform.ui_i18n import translate_ui_text

logger = logging.getLogger(__name__)

_ATX_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_NUMBERED_HEADING_RE = re.compile(r"^\d+(?:\.\d+)*[.)]?\s+")


def generate_report(
    eda_summary: dict[str, Any],
    cleaning_log: list[dict[str, Any]],
    metrics: dict[str, Any],
    feature_importance: list[dict[str, Any]],
    settings: Settings,
    plan_suggestion: Any = None,
    preflight_validation: Any = None,
    postrun_validation: Any = None,
    recommendations: Any = None,
    language: str = "en",
) -> str:
    report, _ = generate_report_result(
        eda_summary=eda_summary,
        cleaning_log=cleaning_log,
        metrics=metrics,
        feature_importance=feature_importance,
        settings=settings,
        plan_suggestion=plan_suggestion,
        preflight_validation=preflight_validation,
        postrun_validation=postrun_validation,
        recommendations=recommendations,
        language=language,
    )
    return report


def generate_report_result(
    eda_summary: dict[str, Any],
    cleaning_log: list[dict[str, Any]],
    metrics: dict[str, Any],
    feature_importance: list[dict[str, Any]],
    settings: Settings,
    plan_suggestion: Any = None,
    preflight_validation: Any = None,
    postrun_validation: Any = None,
    recommendations: Any = None,
    language: str = "en",
) -> tuple[str, str]:
    if settings.llm_enabled:
        try:
            report = _generate_llm_report(
                eda_summary,
                cleaning_log,
                metrics,
                feature_importance,
                settings,
                plan_suggestion=plan_suggestion,
                preflight_validation=preflight_validation,
                postrun_validation=postrun_validation,
                recommendations=recommendations,
                language=language,
            )
            report = _add_heading_number_prefixes(report)
            logger.info("LLM report generated length=%d", len(report))
            return report, "llm"
        except Exception as exc:
            logger.warning("LLM report generation failed, using rule-based fallback: %s", exc)
            fallback = _generate_rule_based_report(
                eda_summary,
                cleaning_log,
                metrics,
                feature_importance,
                plan_suggestion=plan_suggestion,
                preflight_validation=preflight_validation,
                postrun_validation=postrun_validation,
                recommendations=recommendations,
                language=language,
            )
            fallback = _add_heading_number_prefixes(fallback)
            return fallback + f"\n\nLLM report generation failed, so this local rule-based report was used. Error: {exc}", "rule_based"
    logger.info("Using rule-based report (llm_enabled=%s)", settings.llm_enabled)
    return (
        _add_heading_number_prefixes(
            _generate_rule_based_report(
                eda_summary,
                cleaning_log,
                metrics,
                feature_importance,
                plan_suggestion=plan_suggestion,
                preflight_validation=preflight_validation,
                postrun_validation=postrun_validation,
                recommendations=recommendations,
                language=language,
            )
        ),
        "rule_based",
    )


def _add_heading_number_prefixes(report: str) -> str:
    if not report.strip():
        return report

    lines = report.splitlines()
    numbered_lines: list[str] = []
    counters = [0] * 6
    base_level: int | None = None
    previous_depth = 0
    in_code_block = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            numbered_lines.append(line)
            continue

        if in_code_block:
            numbered_lines.append(line)
            continue

        match = _ATX_HEADING_RE.match(line)
        if not match:
            numbered_lines.append(line)
            continue

        hashes, title = match.groups()
        level = len(hashes)
        if base_level is None:
            base_level = level

        depth = max(1, level - base_level + 1)
        if previous_depth and depth > previous_depth + 1:
            depth = previous_depth + 1

        counters[depth - 1] += 1
        for index in range(depth, len(counters)):
            counters[index] = 0
        previous_depth = depth

        number_prefix = ".".join(str(value) for value in counters[:depth])
        normalized_title = title if _NUMBERED_HEADING_RE.match(title) else f"{number_prefix}. {title}"
        numbered_lines.append(f"{hashes} {normalized_title}")

    return "\n".join(numbered_lines)


def _rt(language: str, text: str, **kwargs: Any) -> str:
    return translate_ui_text(language, text, **kwargs)


def _localize_report_text(text: Any, language: str) -> str:
    value = str(text)
    exact = _rt(language, value)
    if exact != value:
        return exact

    patterns: list[tuple[str, str]] = [
        (
            r"^Consider excluding identifier-like columns: (?P<columns>.+)\.$",
            "Consider excluding identifier-like columns: {columns}.",
        ),
        (
            r"^Review and possibly exclude: (?P<columns>.+)\.$",
            "Review and possibly exclude: {columns}.",
        ),
        (
            r"^Pick a metric compatible with (?P<task_type>.+) or adjust the task definition\.$",
            "Pick a metric compatible with {task_type} or adjust the task definition.",
        ),
        (
            r"^Test RMSE is (?P<ratio>.+)x the training RMSE; inspect for overfitting or leakage\.$",
            "Test RMSE is {ratio}x the training RMSE; inspect for overfitting or leakage.",
        ),
    ]
    for pattern, template in patterns:
        match = re.match(pattern, value)
        if match:
            return _rt(language, template, **match.groupdict())
    return value


def _generate_rule_based_report(
    eda_summary: dict[str, Any],
    cleaning_log: list[dict[str, Any]],
    metrics: dict[str, Any],
    feature_importance: list[dict[str, Any]],
    plan_suggestion: Any = None,
    preflight_validation: Any = None,
    postrun_validation: Any = None,
    recommendations: Any = None,
    language: str = "en",
) -> str:
    logger.debug("Generating rule-based report")
    warnings = [
        _localize_report_text(warning, language)
        for warning in (
            eda_summary.get("quality_warnings")
            or ["No major data quality warning was detected by local checks."]
        )
    ]
    cleaned_steps = ", ".join(step.get("step", "unknown") for step in cleaning_log)
    top_features = [
        row["feature"]
        for row in feature_importance
        if row.get("feature") != "__trainer_note__" and row.get("importance") is not None
    ][:10]
    metric_text = ", ".join(
        f"{key}: {value}" for key, value in metrics.items() if not key.startswith("train_")
    )
    train_metric_text = ", ".join(
        f"{key}: {value}" for key, value in metrics.items() if key.startswith("train_")
    )
    recommendation_lines = [
        _localize_report_text(line, language)
        for line in artifact_to_dict(recommendations or {}).get("next_steps", [])
    ]
    preflight_data = artifact_to_dict(preflight_validation or {})
    postrun_data = artifact_to_dict(postrun_validation or {})
    planner_data = artifact_to_dict(plan_suggestion or {})
    validation_notes = [
        _localize_report_text(issue["message"], language)
        for issue in preflight_data.get("issues", []) + postrun_data.get("issues", [])
        if issue.get("severity") in {"warning", "error"}
    ][:5]
    planner_summary = ""
    if planner_data:
        planner_summary = (
            _rt(
                language,
                "Planner suggestion: targets={targets}, task={task_type}, metric={metric}.",
                targets=planner_data.get("suggested_targets", []),
                task_type=planner_data.get("suggested_task_type"),
                metric=planner_data.get("priority_metric"),
            )
        )
    gap_summary = ""
    if train_metric_text:
        gap_summary = _rt(
            language,
            "Training metrics: {metrics}. Compare these with the holdout metrics before concluding the model generalizes.",
            metrics=train_metric_text,
        )
    else:
        gap_summary = _rt(
            language,
            "Training-side metrics are unavailable, so overfitting cannot be judged from this run alone.",
        )

    return (
        f"## {_rt(language, 'Local Analysis Report')}\n\n"
        + _rt(
            language,
            "The dataset has {rows} rows and {columns} columns. Local quality checks flagged: {warnings}",
            rows=eda_summary["shape"]["rows"],
            columns=eda_summary["shape"]["columns"],
            warnings=" ".join(warnings),
        )
        + "\n\n"
        f"{planner_summary}\n\n"
        + _rt(
            language,
            "Cleaning steps applied: {steps}. These steps remove missing target rows, drop unusable features, impute missing feature values, encode categorical variables, scale numeric variables, and create a train/test split.",
            steps=cleaned_steps,
        )
        + "\n\n"
        + _rt(
            language,
            "Model metrics: {metrics}. Treat these as baseline research metrics, especially if the dataset is small or has leakage-prone columns.",
            metrics=metric_text,
        )
        + "\n\n"
        f"{gap_summary}\n\n"
        + _rt(
            language,
            "Top model features: {features}.",
            features=", ".join(top_features)
            if top_features
            else _rt(language, "feature importance is unavailable for this estimator"),
        )
        + "\n\n"
        + _rt(
            language,
            "Validation notes: {notes}",
            notes=" ".join(validation_notes)
            if validation_notes
            else _rt(
                language,
                "No blocking validation issue was surfaced by local checks.",
            ),
        )
        + "\n\n"
        + _rt(language, "Recommended next steps: ")
        + (
            " ".join(recommendation_lines)
            if recommendation_lines
            else _rt(
                language,
                "inspect high-missing and constant columns, check for target leakage, validate train/test split logic, add domain-specific feature engineering, and compare this baseline against a holdout set or cross-validation before trusting it.",
            )
        )
    )


def _generate_llm_report(
    eda_summary: dict[str, Any],
    cleaning_log: list[dict[str, Any]],
    metrics: dict[str, Any],
    feature_importance: list[dict[str, Any]],
    settings: Settings,
    plan_suggestion: Any = None,
    preflight_validation: Any = None,
    postrun_validation: Any = None,
    recommendations: Any = None,
) -> str:
    client_cls = getattr(import_module("open" "ai"), "Open" "AI")
    client = client_cls(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    payload = {
        "eda_summary": eda_summary,
        "cleaning_log": cleaning_log,
        "metrics": metrics,
        "feature_importance": feature_importance[:30],
        "plan_suggestion": artifact_to_dict(plan_suggestion),
        "preflight_validation": artifact_to_dict(preflight_validation),
        "postrun_validation": artifact_to_dict(postrun_validation),
        "recommendations": artifact_to_dict(recommendations),
    }
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You write concise, factual ML experiment reports. Use only the provided payload. "
                    "If training metrics are missing, say overfitting cannot be judged. "
                    "Cover data quality risks, model performance, grounded generalization risks, "
                    "feature engineering ideas, and tuning recommendations."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    return response.choices[0].message.content or ""
