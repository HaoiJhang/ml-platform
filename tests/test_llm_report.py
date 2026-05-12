from ml_platform.llm_report import _add_heading_number_prefixes


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
