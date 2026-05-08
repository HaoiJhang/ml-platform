from __future__ import annotations

import json
from typing import Any

from ml_platform.config import Settings


def generate_report(
    eda_summary: dict[str, Any],
    cleaning_log: list[dict[str, Any]],
    metrics: dict[str, Any],
    feature_importance: list[dict[str, Any]],
    settings: Settings,
) -> str:
    if settings.llm_enabled:
        try:
            return _generate_openai_report(eda_summary, cleaning_log, metrics, feature_importance, settings)
        except Exception as exc:
            fallback = _generate_rule_based_report(eda_summary, cleaning_log, metrics, feature_importance)
            return fallback + f"\n\nLLM report generation failed, so this local rule-based report was used. Error: {exc}"
    return _generate_rule_based_report(eda_summary, cleaning_log, metrics, feature_importance)


def _generate_rule_based_report(
    eda_summary: dict[str, Any],
    cleaning_log: list[dict[str, Any]],
    metrics: dict[str, Any],
    feature_importance: list[dict[str, Any]],
) -> str:
    warnings = eda_summary.get("quality_warnings") or ["No major data quality warning was detected by local checks."]
    cleaned_steps = ", ".join(step.get("step", "unknown") for step in cleaning_log)
    top_features = [
        row["feature"]
        for row in feature_importance
        if row.get("feature") != "__trainer_note__" and row.get("importance") is not None
    ][:10]
    metric_text = ", ".join(f"{key}: {value}" for key, value in metrics.items())

    return (
        "## Local Analysis Report\n\n"
        f"The dataset has {eda_summary['shape']['rows']} rows and {eda_summary['shape']['columns']} columns. "
        f"Local quality checks flagged: {' '.join(warnings)}\n\n"
        f"Cleaning steps applied: {cleaned_steps}. These steps remove missing target rows, drop unusable features, "
        "impute missing feature values, encode categorical variables, scale numeric variables, and create a train/test split.\n\n"
        f"Model metrics: {metric_text}. Treat these as baseline research metrics, especially if the dataset is small or has leakage-prone columns.\n\n"
        f"Top model features: {', '.join(top_features) if top_features else 'feature importance is unavailable for this estimator'}.\n\n"
        "Recommended next steps: inspect high-missing and constant columns, check for target leakage, validate train/test split logic, "
        "add domain-specific feature engineering, and compare this baseline against a holdout set or cross-validation before trusting it."
    )


def _generate_openai_report(
    eda_summary: dict[str, Any],
    cleaning_log: list[dict[str, Any]],
    metrics: dict[str, Any],
    feature_importance: list[dict[str, Any]],
    settings: Settings,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    payload = {
        "eda_summary": eda_summary,
        "cleaning_log": cleaning_log,
        "metrics": metrics,
        "feature_importance": feature_importance[:30],
    }
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You write concise, factual ML experiment reports. Cover data quality risks, "
                    "model performance, overfitting or underfitting risks, feature engineering ideas, "
                    "and tuning recommendations."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    return response.choices[0].message.content or ""
