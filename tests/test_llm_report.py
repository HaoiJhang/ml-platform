from pathlib import Path

from ml_platform.config import Settings
from ml_platform.llm_report import _add_heading_number_prefixes, generate_report_result


def test_add_heading_number_prefixes_for_nested_markdown() -> None:
    report = "\n".join(
        [
            "## Summary",
            "Text",
            "### Metrics",
            "More text",
            "### Risks",
            "#### Leakage",
            "## Next steps",
        ]
    )

    numbered = _add_heading_number_prefixes(report)

    assert "## 1. Summary" in numbered
    assert "### 1.1. Metrics" in numbered
    assert "### 1.2. Risks" in numbered
    assert "#### 1.2.1. Leakage" in numbered
    assert "## 2. Next steps" in numbered


def test_add_heading_number_prefixes_ignores_fenced_code_blocks() -> None:
    report = "\n".join(
        [
            "## Summary",
            "```python",
            "## not a heading",
            "```",
            "### Metrics",
        ]
    )

    numbered = _add_heading_number_prefixes(report)

    assert "## 1. Summary" in numbered
    assert "## not a heading" in numbered
    assert "### 1.1. Metrics" in numbered


def test_rule_based_report_uses_requested_language() -> None:
    settings = Settings(
        runs_dir=Path("runs"),
        data_dir=Path("data"),
        llm_api_key=None,
        llm_base_url=None,
        llm_model="test-model",
    )

    report, mode = generate_report_result(
        eda_summary={"shape": {"rows": 10, "columns": 3}, "quality_warnings": []},
        cleaning_log=[{"step": "split_train_test"}],
        metrics={"r2": 0.8, "train_r2": 0.9},
        feature_importance=[{"feature": "x1", "importance": 0.4}],
        settings=settings,
        plan_suggestion={
            "suggested_targets": ["target"],
            "suggested_task_type": "regression",
            "priority_metric": "r2",
        },
        recommendations={
            "next_steps": [
                "Consider excluding identifier-like columns: customer_id."
            ]
        },
        language="zh-CN",
    )

    assert mode == "rule_based"
    assert "本地分析报告" in report
    assert "数据集共有 10 行、3 列" in report
    assert "建议排除疑似标识符列：customer_id。" in report
    assert "Local Analysis Report" not in report
    assert "The dataset has" not in report
