from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app as app_module
import pandas as pd
from streamlit.testing.v1 import AppTest

from ml_platform.validation import validate_preflight


def _latest_run_config(runs_dir: Path) -> dict[str, object]:
    run_dirs = [path for path in runs_dir.iterdir() if path.is_dir()]
    latest_run = max(run_dirs, key=lambda path: path.name)
    return json.loads((latest_run / "config.json").read_text(encoding="utf-8"))


def _switch_to_english(app: AppTest) -> None:
    app.selectbox(key="ui_language").set_value("en")
    app.run(timeout=120)


def _click_button(app: AppTest, label: str) -> None:
    button = next(button for button in app.button if button.label == label)
    button.click()
    app.run(timeout=120)


def _advance_to_check(app: AppTest, label: str = "Next: check data") -> None:
    _click_button(app, label)


def _advance_to_prepare(app: AppTest, label: str = "Next: prepare training") -> None:
    _click_button(app, label)


def _choose_demo(app: AppTest, label: str = "Demo: demo_customer_churn.csv") -> None:
    app.radio[0].set_value(label)
    app.run(timeout=120)
    next_label = "下一步：选择预测目标" if label.startswith("示例") else "Next: choose target"
    _click_button(app, next_label)


def test_local_llm_config_enabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("ML_PLATFORM_ALLOW_LOCAL_LLM_CONFIG", raising=False)
    assert app_module._local_llm_config_enabled() is True


def test_local_llm_config_can_be_explicitly_disabled(monkeypatch) -> None:
    monkeypatch.setenv("ML_PLATFORM_ALLOW_LOCAL_LLM_CONFIG", "0")
    assert app_module._local_llm_config_enabled() is False


def test_optional_ai_help_is_collapsed_by_default(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ML_PLATFORM_RUNS_DIR", str(tmp_path / "runs"))

    app = AppTest.from_file("app.py")
    app.run(timeout=120)

    _switch_to_english(app)

    ai_help = next(expander for expander in app.expander if expander.label == "Optional AI help")
    assert ai_help.proto.expanded is False
    assert not any(subheader.value == "2. Report engine" for subheader in app.subheader)


def test_field_health_rows_flag_target_id_leakage_and_missingness() -> None:
    df = pd.DataFrame(
        {
            "customer_id": [f"c-{index}" for index in range(30)],
            "target": [0, 1] * 15,
            "target_copy": [0, 1] * 15,
            "mostly_missing": [None] * 25 + [1, 2, 3, 4, 5],
        }
    )
    preflight = validate_preflight(
        df=df,
        target="target",
        task_type="classification",
        excluded_columns=[],
        priority_metric="auto",
        high_missing_threshold=0.5,
    )

    rows = app_module.build_field_health_rows(
        df,
        target_columns=["target"],
        preflight_by_target={"target": preflight},
        high_missing_threshold=0.5,
    )
    column_key = app_module._t("Column")
    suggested_action_key = app_module._t("Suggested action")
    id_like_key = app_module._t("ID-like")
    leakage_risk_key = app_module._t("Leakage risk")
    by_column = {row[column_key]: row for row in rows}

    assert by_column["target"][suggested_action_key] == app_module._t("Target column")
    assert by_column["customer_id"][id_like_key] == app_module._t("Yes")
    assert by_column["target_copy"][leakage_risk_key] == app_module._t("Yes")
    assert by_column["mostly_missing"][suggested_action_key] == app_module._t(
        "Review missing values"
    )


def test_can_switch_ui_language_to_chinese(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ML_PLATFORM_RUNS_DIR", str(tmp_path / "runs"))

    app = AppTest.from_file("app.py")
    app.run(timeout=120)

    assert any(subheader.value == "快速开始" for subheader in app.subheader)
    assert any(subheader.value == "1. 上传数据" for subheader in app.subheader)

    app.selectbox(key="ui_language").set_value("en")
    app.run(timeout=120)

    assert any(subheader.value == "Quick start" for subheader in app.subheader)
    assert any(subheader.value == "1. Upload data" for subheader in app.subheader)


def test_distribution_tab_shows_default_feature_and_excludes_target(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("ML_PLATFORM_RUNS_DIR", str(tmp_path / "runs"))

    app = AppTest.from_file("app.py")
    app.run(timeout=120)

    _switch_to_english(app)

    _choose_demo(app)

    app.multiselect(key="target_columns").set_value(["churn"])
    app.run(timeout=120)
    _advance_to_check(app)

    assert any(tab.label == "Distributions" for tab in app.tabs)
    distribution_feature = app.selectbox(key="distribution_feature_column")
    assert distribution_feature.label == "Feature to inspect"
    assert distribution_feature.value == "tenure_months"
    assert "churn" not in distribution_feature.options
    assert not any(selectbox.key == "distribution_compare_target" for selectbox in app.selectbox)
    assert any(
        caption.value
        == "Use this view to scan feature distributions and compare them against the selected target."
        for caption in app.caption
    )


def test_categorical_encoding_strategy_offers_three_options(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ML_PLATFORM_RUNS_DIR", str(tmp_path / "runs"))

    app = AppTest.from_file("app.py")
    app.run(timeout=120)

    _switch_to_english(app)

    _choose_demo(app)

    app.multiselect(key="target_columns").set_value(["churn"])
    app.run(timeout=120)
    _advance_to_check(app)

    categorical_encoding = app.selectbox(key="_categorical_encoding_strategy_label")
    assert categorical_encoding.options == ["one_hot", "ordinal", "frequency"]


def test_preprocessing_section_is_collapsed_by_default(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ML_PLATFORM_RUNS_DIR", str(tmp_path / "runs"))

    app = AppTest.from_file("app.py")
    app.run(timeout=120)

    _switch_to_english(app)

    _choose_demo(app)

    app.multiselect(key="target_columns").set_value(["churn"])
    app.run(timeout=120)
    _advance_to_check(app)

    preprocessing_expander = next(
        expander for expander in app.expander if expander.label == "Open preprocessing details"
    )
    assert preprocessing_expander.proto.expanded is False
    assert any("Categorical encoding strategy" in caption.value for caption in app.caption)


def test_chinese_ui_covers_preprocessing_and_results_labels(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ML_PLATFORM_RUNS_DIR", str(tmp_path / "runs"))

    app = AppTest.from_file("app.py")
    app.run(timeout=120)

    app.selectbox(key="ui_language").set_value("zh-CN")
    app.run(timeout=120)

    _choose_demo(app, "示例：demo_customer_churn.csv")

    app.multiselect(key="target_columns").set_value(["churn"])
    app.run(timeout=120)
    app.text_input(key="time_budget_text").set_value("5")
    app.run(timeout=120)
    _advance_to_check(app, "下一步：检查数据")

    assert any(subheader.value == "3. 配置数据预处理" for subheader in app.subheader)
    assert not any(subheader.value == "4. 准备数据" for subheader in app.subheader)
    assert any("类别编码策略" in caption.value for caption in app.caption)
    assert any("这些检查会解释为什么当前可以继续训练" in caption.value for caption in app.caption)
    assert any("这些检查项不会阻止训练" in alert.value for alert in app.info)
    _advance_to_prepare(app, "下一步：准备训练")
    assert any(subheader.value == "4. 准备数据" for subheader in app.subheader)
    assert not any(subheader.value == "5. 开始训练" for subheader in app.subheader)
    _click_button(app, "准备数据")
    assert any(subheader.value == "5. 开始训练" for subheader in app.subheader)
    assert any(button.label == "准备数据" for button in app.button)
    assert any(button.label == "开始训练" for button in app.button)

    run_training = next(button for button in app.button if button.label == "开始训练")
    run_training.click()
    app.run(timeout=120)

    assert any(subheader.value == "6. 查看结果" for subheader in app.subheader)
    assert any(metric.label == "已完成运行数" for metric in app.metric)
    assert any(selectbox.label == "查看数据处理流程步骤" for selectbox in app.selectbox)
    assert any(caption.value == "如何看结果" for caption in app.caption)
    assert any("建议先看运行摘要和校验说明" in alert.value for alert in app.info)
    assert any(caption.value == "结果解读" for caption in app.caption)


def test_chinese_ui_translates_distribution_tab_and_controls(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("ML_PLATFORM_RUNS_DIR", str(tmp_path / "runs"))

    app = AppTest.from_file("app.py")
    app.run(timeout=120)

    app.selectbox(key="ui_language").set_value("zh-CN")
    app.run(timeout=120)

    _choose_demo(app, "示例：demo_customer_churn.csv")

    app.multiselect(key="target_columns").set_value(["churn"])
    app.run(timeout=120)
    _advance_to_check(app, "下一步：检查数据")

    assert any(tab.label == "分布可视化" for tab in app.tabs)
    distribution_feature = app.selectbox(key="distribution_feature_column")
    assert distribution_feature.label == "选择要观察的特征列"
    assert any(
        caption.value == "这个视图用于快速扫一遍特征分布，并和当前选择的目标列做对比。"
        for caption in app.caption
    )


def test_workflow_waits_for_target_selection(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ML_PLATFORM_RUNS_DIR", str(tmp_path / "runs"))

    app = AppTest.from_file("app.py")
    app.run(timeout=120)

    _switch_to_english(app)

    assert any(subheader.value == "1. Upload data" for subheader in app.subheader)
    assert not any(subheader.value == "2. Choose what to predict" for subheader in app.subheader)

    _choose_demo(app)

    assert any(subheader.value == "2. Choose what to predict" for subheader in app.subheader)
    assert any("no prediction target has been selected yet" in alert.value.lower() for alert in app.warning)
    assert not any(button.label == "Run training" for button in app.button)

    app.multiselect(key="target_columns").set_value(["churn"])
    app.run(timeout=120)

    advanced_settings = next(expander for expander in app.expander if expander.label == "Advanced experiment settings")
    assert advanced_settings.proto.expanded is False
    assert not any(subheader.value == "3. Configure preprocessing" for subheader in app.subheader)
    assert not any(button.label == "Run training" for button in app.button)
    _advance_to_check(app)
    assert any(subheader.value == "3. Configure preprocessing" for subheader in app.subheader)
    assert not any(subheader.value == "4. Prepare data" for subheader in app.subheader)
    assert any("Field health" in markdown.value for markdown in app.markdown)
    assert any("These checks explain why training can continue" in caption.value for caption in app.caption)
    assert any("These checks will not stop training" in alert.value for alert in app.info)


def test_data_flow_selection_does_not_drop_latest_results(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ML_PLATFORM_RUNS_DIR", str(tmp_path / "runs"))

    app = AppTest.from_file("app.py")
    app.run(timeout=120)

    _switch_to_english(app)

    _choose_demo(app)

    app.multiselect(key="target_columns").set_value(["churn"])
    app.run(timeout=120)
    app.text_input(key="time_budget_text").set_value("5")
    app.run(timeout=120)
    _advance_to_check(app)
    _advance_to_prepare(app)

    _click_button(app, "Prepare data")
    _click_button(app, "Run training")

    assert any(subheader.value == "6. Review results" for subheader in app.subheader)
    assert any(metric.label == "Completed runs" for metric in app.metric)
    assert any(caption.value == "How to read the results" for caption in app.caption)
    assert any("Start with the run summary and validation notes." in alert.value for alert in app.info)
    assert any(caption.value == "Result translation" for caption in app.caption)

    data_flow_select = next(selectbox for selectbox in app.selectbox if selectbox.label == "Inspect data flow step")
    data_flow_select.set_value(data_flow_select.options[1])
    app.run(timeout=120)

    assert any(subheader.value == "6. Review results" for subheader in app.subheader)
    selected_again = next(selectbox for selectbox in app.selectbox if selectbox.label == "Inspect data flow step")
    assert selected_again.value == selected_again.options[1]


def test_manual_cleaning_rules_only_change_eda_after_apply(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ML_PLATFORM_RUNS_DIR", str(tmp_path / "runs"))

    app = AppTest.from_file("app.py")
    app.run(timeout=120)

    _choose_demo(app, "示例：demo_customer_churn.csv")

    app.multiselect(key="target_columns").set_value(["churn"])
    app.run(timeout=120)
    _advance_to_check(app, "下一步：检查数据")

    def rows_metric_value() -> str:
        return next(metric.value for metric in app.metric if metric.label == "行数")

    assert rows_metric_value() == "24"

    app.text_area(key="manual_cleaning_brief").set_value("keep rows where churn equals yes")
    app.run(timeout=120)

    generate_button = next(button for button in app.button if button.label == "生成清洗规则")
    generate_button.click()
    app.run(timeout=120)

    assert rows_metric_value() == "24"

    apply_button = next(button for button in app.button if button.label == "应用手动清洗规则")
    apply_button.click()
    app.run(timeout=120)

    assert rows_metric_value() == "10"


def test_applied_preprocessing_step_invalidates_prepared_data(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ML_PLATFORM_RUNS_DIR", str(tmp_path / "runs"))

    app = AppTest.from_file("app.py")
    app.run(timeout=120)

    _switch_to_english(app)

    _choose_demo(app)

    app.multiselect(key="target_columns").set_value(["churn"])
    app.run(timeout=120)
    app.text_input(key="time_budget_text").set_value("5")
    app.run(timeout=120)
    _advance_to_check(app)
    _advance_to_prepare(app)

    prepare_data = next(button for button in app.button if button.label == "Prepare data")
    prepare_data.click()
    app.run(timeout=120)

    run_training = next(button for button in app.button if button.label == "Run training")
    assert run_training.disabled is False

    _click_button(app, "Done: Check data")
    app.selectbox(key="numeric_imputation_strategy").set_value("mean")
    app.run(timeout=120)

    apply_missing_step = next(button for button in app.button if button.label == "Apply missing-value step")
    apply_missing_step.click()
    app.run(timeout=120)

    _advance_to_prepare(app)
    _click_button(app, "Prepare data")
    _click_button(app, "Run training")

    run_config = _latest_run_config(tmp_path / "runs")
    assert run_config["numeric_imputation_strategy"] == "mean"


def test_blocking_preflight_explanation_is_shown_when_no_features_remain(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ML_PLATFORM_RUNS_DIR", str(tmp_path / "runs"))

    app = AppTest.from_file("app.py")
    app.run(timeout=120)

    _switch_to_english(app)

    _choose_demo(app)

    app.multiselect(key="target_columns").set_value(["churn"])
    app.run(timeout=120)
    _advance_to_check(app)

    app.multiselect(key="excluded_columns").set_value(
        [
            "customer_id",
            "signup_date",
            "tenure_months",
            "monthly_spend",
            "support_tickets_90d",
            "contract_type",
            "region",
            "autopay",
            "late_payments_6m",
            "nps_score",
        ]
    )
    app.run(timeout=120)

    apply_column_selection = next(
        button for button in app.button if button.label == "Apply column selection"
    )
    apply_column_selection.click()
    app.run(timeout=120)

    assert any("These checks can stop training." in alert.value for alert in app.warning)
    assert any(
        "The app needs at least one input column to learn from." in markdown.value
        for markdown in app.markdown
    )


def test_run_training_uses_applied_preprocessing_not_unapplied_draft(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ML_PLATFORM_RUNS_DIR", str(tmp_path / "runs"))

    app = AppTest.from_file("app.py")
    app.run(timeout=120)

    _switch_to_english(app)

    _choose_demo(app)

    app.multiselect(key="target_columns").set_value(["churn"])
    app.run(timeout=120)
    _advance_to_check(app)

    app.selectbox(key="numeric_imputation_strategy").set_value("mean")
    app.run(timeout=120)
    _advance_to_prepare(app)

    _click_button(app, "Prepare data")
    _click_button(app, "Run training")

    run_config = _latest_run_config(tmp_path / "runs")
    assert run_config["numeric_imputation_strategy"] == "median"
