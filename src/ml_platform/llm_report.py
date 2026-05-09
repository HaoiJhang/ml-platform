from __future__ import annotations

import json
import logging
from typing import Any

from ml_platform.artifacts import artifact_to_dict
from ml_platform.config import Settings

logger = logging.getLogger(__name__)


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
) -> tuple[str, str]:
    if settings.llm_enabled:
        try:
            report = _generate_openai_report(
                eda_summary,
                cleaning_log,
                metrics,
                feature_importance,
                settings,
                plan_suggestion=plan_suggestion,
                preflight_validation=preflight_validation,
                postrun_validation=postrun_validation,
                recommendations=recommendations,
            )
            logger.info("OpenAI report generated length=%d", len(report))
            return report, "openai"
        except Exception as exc:
            logger.warning("OpenAI report generation failed, using rule-based fallback: %s", exc)
            fallback = _generate_rule_based_report(
                eda_summary,
                cleaning_log,
                metrics,
                feature_importance,
                plan_suggestion=plan_suggestion,
                preflight_validation=preflight_validation,
                postrun_validation=postrun_validation,
                recommendations=recommendations,
            )
            return fallback + f"\n\nLLM report generation failed, so this local rule-based report was used. Error: {exc}", "rule_based"
    logger.info("Using rule-based report (llm_enabled=%s)", settings.llm_enabled)
    return (
        _generate_rule_based_report(
            eda_summary,
            cleaning_log,
            metrics,
            feature_importance,
            plan_suggestion=plan_suggestion,
            preflight_validation=preflight_validation,
            postrun_validation=postrun_validation,
            recommendations=recommendations,
        ),
        "rule_based",
    )


def _generate_rule_based_report(
    eda_summary: dict[str, Any],
    cleaning_log: list[dict[str, Any]],
    metrics: dict[str, Any],
    feature_importance: list[dict[str, Any]],
    plan_suggestion: Any = None,
    preflight_validation: Any = None,
    postrun_validation: Any = None,
    recommendations: Any = None,
) -> str:
    logger.debug("Generating rule-based report")
    warnings = eda_summary.get("quality_warnings") or ["No major data quality warning was detected by local checks."]
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
    recommendation_lines = artifact_to_dict(recommendations or {}).get("next_steps", [])
    preflight_data = artifact_to_dict(preflight_validation or {})
    postrun_data = artifact_to_dict(postrun_validation or {})
    planner_data = artifact_to_dict(plan_suggestion or {})
    validation_notes = [
        issue["message"]
        for issue in preflight_data.get("issues", []) + postrun_data.get("issues", [])
        if issue.get("severity") in {"warning", "error"}
    ][:5]
    planner_summary = ""
    if planner_data:
        planner_summary = (
            f"Planner suggestion: targets={planner_data.get('suggested_targets', [])}, "
            f"task={planner_data.get('suggested_task_type')}, metric={planner_data.get('priority_metric')}."
        )
    gap_summary = ""
    if train_metric_text:
        gap_summary = f"Training metrics: {train_metric_text}. Compare these with the holdout metrics before concluding the model generalizes."
    else:
        gap_summary = "Training-side metrics are unavailable, so overfitting cannot be judged from this run alone."

    return (
        "## Local Analysis Report\n\n"
        f"The dataset has {eda_summary['shape']['rows']} rows and {eda_summary['shape']['columns']} columns. "
        f"Local quality checks flagged: {' '.join(warnings)}\n\n"
        f"{planner_summary}\n\n"
        f"Cleaning steps applied: {cleaned_steps}. These steps remove missing target rows, drop unusable features, "
        "impute missing feature values, encode categorical variables, scale numeric variables, and create a train/test split.\n\n"
        f"Model metrics: {metric_text}. Treat these as baseline research metrics, especially if the dataset is small or has leakage-prone columns.\n\n"
        f"{gap_summary}\n\n"
        f"Top model features: {', '.join(top_features) if top_features else 'feature importance is unavailable for this estimator'}.\n\n"
        f"Validation notes: {' '.join(validation_notes) if validation_notes else 'No blocking validation issue was surfaced by local checks.'}\n\n"
        "Recommended next steps: "
        + (
            " ".join(recommendation_lines)
            if recommendation_lines
            else "inspect high-missing and constant columns, check for target leakage, validate train/test split logic, add domain-specific feature engineering, and compare this baseline against a holdout set or cross-validation before trusting it."
        )
    )


def _generate_openai_report(
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
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
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
        model=settings.openai_model,
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
