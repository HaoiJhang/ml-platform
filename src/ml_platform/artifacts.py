from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    column: str | None = None


@dataclass(frozen=True)
class PlanSuggestion:
    planner_name: str
    user_brief: str
    suggested_targets: list[str] = field(default_factory=list)
    suggested_task_type: str = "auto"
    suggested_excluded_columns: list[str] = field(default_factory=list)
    priority_metric: str = "auto"
    notes: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PreflightValidation:
    ok_to_run: bool
    task_type: str
    priority_metric: str
    issues: list[ValidationIssue] = field(default_factory=list)
    dropped_target_rows: int = 0
    feature_count: int = 0
    recommended_excluded_columns: list[str] = field(default_factory=list)
    detected_leakage_columns: list[str] = field(default_factory=list)
    class_balance: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PostRunValidation:
    ok: bool
    task_type: str
    priority_metric: str
    issues: list[ValidationIssue] = field(default_factory=list)
    trainer_name: str = ""
    optimization_metric_used: str | None = None
    report_mode: str = "rule_based"
    generalization_gap: dict[str, float | None] = field(default_factory=dict)


@dataclass(frozen=True)
class RecommendationSet:
    summary: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)


def artifact_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: artifact_to_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [artifact_to_dict(item) for item in value]
    return value
