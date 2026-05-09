from __future__ import annotations

from typing import Any

import pandas as pd

from ml_platform.artifacts import PostRunValidation, PreflightValidation, RecommendationSet, ValidationIssue


VALID_CLASSIFICATION_METRICS = {"accuracy", "f1_weighted", "precision_weighted", "recall_weighted", "roc_auc"}
VALID_REGRESSION_METRICS = {"rmse", "mae", "r2"}


def validate_preflight(
    df: pd.DataFrame,
    target: str,
    task_type: str,
    excluded_columns: list[str],
    priority_metric: str,
    high_missing_threshold: float,
) -> PreflightValidation:
    issues: list[ValidationIssue] = []
    if target not in df.columns:
        issues.append(ValidationIssue("error", "missing_target", f"Target column was not found: {target}.", target))
        return PreflightValidation(
            ok_to_run=False,
            task_type=task_type,
            priority_metric=priority_metric,
            issues=issues,
        )

    kept_columns = [column for column in df.columns if column not in excluded_columns]
    if target not in kept_columns:
        issues.append(ValidationIssue("error", "excluded_target", "Target column cannot be excluded.", target))
        return PreflightValidation(
            ok_to_run=False,
            task_type=task_type,
            priority_metric=priority_metric,
            issues=issues,
        )

    working = df[kept_columns].copy()
    dropped_target_rows = int(working[target].isna().sum())
    working = working.dropna(subset=[target])
    if working.empty:
        issues.append(ValidationIssue("error", "empty_after_target_drop", "No rows remain after dropping missing target values.", target))

    feature_columns = [column for column in working.columns if column != target]
    if not feature_columns:
        issues.append(ValidationIssue("error", "no_features", "No feature columns remain after exclusions.", target))

    high_missing = [
        column for column in feature_columns if float(working[column].isna().mean()) > high_missing_threshold
    ]
    if high_missing:
        issues.append(
            ValidationIssue(
                "warning",
                "high_missing_features",
                f"Features exceed the missing-rate threshold and will be dropped: {', '.join(high_missing[:10])}.",
            )
        )

    constant_columns = [column for column in feature_columns if int(working[column].nunique(dropna=True)) <= 1]
    if constant_columns:
        issues.append(
            ValidationIssue(
                "warning",
                "constant_features",
                f"Constant or nearly empty features were detected: {', '.join(constant_columns[:10])}.",
            )
        )

    id_like = _detect_identifier_columns(working[feature_columns])
    if id_like:
        issues.append(
            ValidationIssue(
                "info",
                "identifier_candidates",
                f"Identifier-like columns may not generalize well: {', '.join(id_like[:10])}.",
            )
        )

    leakage_columns = _detect_leakage_columns(working, target)
    if leakage_columns:
        issues.append(
            ValidationIssue(
                "warning",
                "possible_leakage",
                f"Features look suspiciously close to the target and may leak label information: {', '.join(leakage_columns[:10])}.",
            )
        )

    class_balance: dict[str, float] = {}
    if task_type == "classification" and not working.empty:
        class_counts = working[target].astype("string").value_counts(normalize=True, dropna=False)
        class_balance = {str(label): float(rate) for label, rate in class_counts.to_dict().items()}
        if int(working[target].nunique(dropna=True)) < 2:
            issues.append(ValidationIssue("error", "single_class_target", "Classification target must have at least two classes.", target))
        elif float(class_counts.min()) < 0.1:
            issues.append(
                ValidationIssue(
                    "warning",
                    "class_imbalance",
                    f"Target is imbalanced; the smallest class accounts for {class_counts.min():.1%} of rows.",
                    target,
                )
            )
    elif task_type == "regression" and not working.empty and int(working[target].nunique(dropna=True)) < 5:
        issues.append(
            ValidationIssue(
                "warning",
                "low_target_cardinality",
                "Regression target has very few unique values; classification may be more appropriate.",
                target,
            )
        )

    if len(working) < 30:
        issues.append(
            ValidationIssue(
                "warning",
                "small_dataset",
                "Very few rows remain after filtering; holdout metrics may be unstable.",
            )
        )

    if priority_metric != "auto" and not _metric_is_compatible(task_type, priority_metric):
        issues.append(
            ValidationIssue(
                "warning",
                "metric_incompatible",
                f"Priority metric {priority_metric} does not match task type {task_type}; reporting will fall back to task defaults.",
            )
        )

    ok_to_run = not any(issue.severity == "error" for issue in issues)
    return PreflightValidation(
        ok_to_run=ok_to_run,
        task_type=task_type,
        priority_metric=priority_metric,
        issues=issues,
        dropped_target_rows=dropped_target_rows,
        feature_count=max(len(feature_columns), 0),
        recommended_excluded_columns=id_like,
        detected_leakage_columns=leakage_columns,
        class_balance=class_balance,
    )


def validate_postrun(
    metrics: dict[str, float | None],
    prediction_sample: pd.DataFrame,
    task_type: str,
    priority_metric: str,
    trainer_name: str,
    optimization_metric_used: str | None,
    feature_importance: list[dict[str, Any]],
    report_mode: str,
) -> PostRunValidation:
    issues: list[ValidationIssue] = []
    resolved_metric = resolve_priority_metric(task_type, priority_metric)
    metric_value = metrics.get(resolved_metric)
    if metric_value is None:
        issues.append(
            ValidationIssue(
                "warning",
                "priority_metric_unavailable",
                f"Priority metric {resolved_metric} is unavailable for this run.",
            )
        )

    if prediction_sample.empty:
        issues.append(ValidationIssue("error", "empty_predictions", "Prediction sample is empty."))

    if not feature_importance or all(row.get("importance") is None for row in feature_importance):
        issues.append(
            ValidationIssue(
                "warning",
                "feature_importance_unavailable",
                "Feature importance is unavailable for the trained estimator.",
            )
        )

    trainer_note = next((row.get("note") for row in feature_importance if row.get("feature") == "__trainer_note__"), None)
    if trainer_note:
        issues.append(ValidationIssue("info", "trainer_fallback", str(trainer_note)))

    generalization_gap = _compute_generalization_gap(metrics, task_type)
    if task_type == "classification" and generalization_gap.get("accuracy_gap") is not None and generalization_gap["accuracy_gap"] > 0.15:
        issues.append(
            ValidationIssue(
                "warning",
                "generalization_gap",
                f"Train/test accuracy gap is {generalization_gap['accuracy_gap']:.3f}; inspect for overfitting or leakage.",
            )
        )
    if task_type == "regression" and generalization_gap.get("rmse_ratio") is not None and generalization_gap["rmse_ratio"] > 1.5:
        issues.append(
            ValidationIssue(
                "warning",
                "generalization_gap",
                f"Test RMSE is {generalization_gap['rmse_ratio']:.2f}x the training RMSE; inspect for overfitting or leakage.",
            )
        )

    if report_mode not in {"", "pending", "openai"}:
        issues.append(
            ValidationIssue(
                "info",
                "rule_based_report",
                "The run used the local rule-based report generator.",
            )
        )

    ok = not any(issue.severity == "error" for issue in issues)
    return PostRunValidation(
        ok=ok,
        task_type=task_type,
        priority_metric=resolved_metric,
        issues=issues,
        trainer_name=trainer_name,
        optimization_metric_used=optimization_metric_used,
        report_mode=report_mode,
        generalization_gap=generalization_gap,
    )


def build_recommendations(
    preflight: PreflightValidation,
    postrun: PostRunValidation | None = None,
) -> RecommendationSet:
    summary: list[str] = []
    next_steps: list[str] = []

    if preflight.detected_leakage_columns:
        summary.append("Potential leakage signals were detected before training.")
        next_steps.append(f"Review and possibly exclude: {', '.join(preflight.detected_leakage_columns[:5])}.")
    if preflight.recommended_excluded_columns:
        next_steps.append(
            f"Consider excluding identifier-like columns: {', '.join(preflight.recommended_excluded_columns[:5])}."
        )
    if any(issue.code == "class_imbalance" for issue in preflight.issues):
        next_steps.append("Compare the baseline against class-weighted models or resampling for the minority class.")
    if any(issue.code == "small_dataset" for issue in preflight.issues):
        next_steps.append("Prefer cross-validation or repeated holdout over trusting a single small test split.")

    if postrun is not None:
        if any(issue.code == "generalization_gap" for issue in postrun.issues):
            summary.append("Train and test metrics diverge materially.")
            next_steps.append("Inspect feature leakage, reduce model complexity, and compare with cross-validation.")
        if any(issue.code == "priority_metric_unavailable" for issue in postrun.issues):
            next_steps.append(f"Pick a metric compatible with {postrun.task_type} or adjust the task definition.")
        if postrun.report_mode not in {"", "pending", "openai"}:
            next_steps.append("Provide an API key if you want a narrative report beyond the local rules.")

    if not summary:
        summary.append("No blocking validation issues were found in the current run configuration.")
    if not next_steps:
        next_steps.append("Review the stored artifacts and compare this baseline against a second run before trusting the model.")
    return RecommendationSet(summary=summary, next_steps=next_steps)


def resolve_priority_metric(task_type: str, priority_metric: str) -> str:
    if _metric_is_compatible(task_type, priority_metric):
        return priority_metric
    return "f1_weighted" if task_type == "classification" else "rmse"


def _metric_is_compatible(task_type: str, priority_metric: str) -> bool:
    if priority_metric == "auto":
        return False
    if task_type == "classification":
        return priority_metric in VALID_CLASSIFICATION_METRICS
    return priority_metric in VALID_REGRESSION_METRICS


def _detect_identifier_columns(df: pd.DataFrame) -> list[str]:
    candidates: list[str] = []
    for column in df.columns:
        series = df[column]
        non_null = max(len(series.dropna()), 1)
        unique_rate = float(series.nunique(dropna=True) / non_null)
        name = str(column).lower()
        if name == "id" or name.endswith("_id") or "uuid" in name:
            candidates.append(column)
        elif unique_rate >= 0.98 and len(series) >= 20 and not pd.api.types.is_numeric_dtype(series):
            candidates.append(column)
    return candidates


def _detect_leakage_columns(df: pd.DataFrame, target: str) -> list[str]:
    leakage: list[str] = []
    target_series = df[target]
    for column in df.columns:
        if column == target:
            continue
        series = df[column]
        try:
            if series.astype("string").equals(target_series.astype("string")):
                leakage.append(column)
                continue
        except Exception:
            pass
        if pd.api.types.is_numeric_dtype(series) and pd.api.types.is_numeric_dtype(target_series):
            corr = df[[column, target]].corr(numeric_only=True).iloc[0, 1]
            if pd.notna(corr) and abs(float(corr)) >= 0.995:
                leakage.append(column)
    return leakage


def _compute_generalization_gap(metrics: dict[str, float | None], task_type: str) -> dict[str, float | None]:
    if task_type == "classification":
        train_accuracy = metrics.get("train_accuracy")
        test_accuracy = metrics.get("accuracy")
        return {
            "accuracy_gap": None if train_accuracy is None or test_accuracy is None else float(train_accuracy - test_accuracy),
            "f1_gap": None
            if metrics.get("train_f1_weighted") is None or metrics.get("f1_weighted") is None
            else float(metrics["train_f1_weighted"] - metrics["f1_weighted"]),
        }
    train_rmse = metrics.get("train_rmse")
    test_rmse = metrics.get("rmse")
    return {
        "rmse_gap": None if train_rmse is None or test_rmse is None else float(test_rmse - train_rmse),
        "rmse_ratio": None if train_rmse in (None, 0) or test_rmse is None else float(test_rmse / train_rmse),
    }
