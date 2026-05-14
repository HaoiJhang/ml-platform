from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app as app_module
from streamlit.testing.v1 import AppTest


def _latest_run_config(runs_dir: Path) -> dict[str, object]:
    run_dirs = [path for path in runs_dir.iterdir() if path.is_dir()]
    latest_run = max(run_dirs, key=lambda path: path.name)
    return json.loads((latest_run / "config.json").read_text(encoding="utf-8"))


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

    ai_help = next(expander for expander in app.expander if expander.label == "Optional AI help")
    assert ai_help.proto.expanded is False
    assert not any(subheader.value == "2. Report engine" for subheader in app.subheader)


def test_can_switch_ui_language_to_chinese(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ML_PLATFORM_RUNS_DIR", str(tmp_path / "runs"))

    app = AppTest.from_file("app.py")
    app.run(timeout=120)

    assert any(subheader.value == "Quick start" for subheader in app.subheader)

    app.selectbox(key="ui_language").set_value("zh-CN")
    app.run(timeout=120)

    assert any(subheader.value == "快速开始" for subheader in app.subheader)
    assert any(subheader.value == "1. 上传数据" for subheader in app.subheader)


def test_categorical_encoding_strategy_offers_three_options(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ML_PLATFORM_RUNS_DIR", str(tmp_path / "runs"))

    app = AppTest.from_file("app.py")
    app.run(timeout=120)

    app.radio[0].set_value("Demo: demo_customer_churn.csv")
    app.run(timeout=120)

    app.multiselect(key="target_columns").set_value(["churn"])
    app.run(timeout=120)

    categorical_encoding = app.selectbox(key="_categorical_encoding_strategy_label")
    assert categorical_encoding.options == ["one_hot", "ordinal", "frequency"]


def test_preprocessing_section_is_collapsed_by_default(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ML_PLATFORM_RUNS_DIR", str(tmp_path / "runs"))

    app = AppTest.from_file("app.py")
    app.run(timeout=120)

    app.radio[0].set_value("Demo: demo_customer_churn.csv")
    app.run(timeout=120)

    app.multiselect(key="target_columns").set_value(["churn"])
    app.run(timeout=120)

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

    app.radio[0].set_value("示例：demo_customer_churn.csv")
    app.run(timeout=120)

    app.multiselect(key="target_columns").set_value(["churn"])
    app.run(timeout=120)

    app.text_input(key="time_budget_text").set_value("5")
    app.run(timeout=120)

    assert any(subheader.value == "3. 配置数据预处理" for subheader in app.subheader)
    assert any(subheader.value == "4. 准备数据" for subheader in app.subheader)
    assert any(subheader.value == "5. 开始训练" for subheader in app.subheader)
    assert any(button.label == "准备数据" for button in app.button)
    assert any(button.label == "开始训练" for button in app.button)
    assert any("类别编码策略" in caption.value for caption in app.caption)

    run_training = next(button for button in app.button if button.label == "开始训练")
    run_training.click()
    app.run(timeout=120)

    assert any(subheader.value == "6. 查看结果" for subheader in app.subheader)
    assert any(metric.label == "已完成运行数" for metric in app.metric)
    assert any(selectbox.label == "查看数据处理流程步骤" for selectbox in app.selectbox)


def test_workflow_waits_for_target_selection(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ML_PLATFORM_RUNS_DIR", str(tmp_path / "runs"))

    app = AppTest.from_file("app.py")
    app.run(timeout=120)

    assert any(subheader.value == "1. Upload data" for subheader in app.subheader)
    assert not any(subheader.value == "2. Choose what to predict" for subheader in app.subheader)

    app.radio[0].set_value("Demo: demo_customer_churn.csv")
    app.run(timeout=120)

    assert any(subheader.value == "2. Choose what to predict" for subheader in app.subheader)
    assert any("no prediction target has been selected yet" in alert.value.lower() for alert in app.warning)
    assert not any(button.label == "Run training" for button in app.button)

    app.multiselect(key="target_columns").set_value(["churn"])
    app.run(timeout=120)

    advanced_settings = next(expander for expander in app.expander if expander.label == "Advanced experiment settings")
    assert advanced_settings.proto.expanded is False
    assert any(subheader.value == "3. Configure preprocessing" for subheader in app.subheader)
    assert any(subheader.value == "4. Prepare data" for subheader in app.subheader)
    assert any(subheader.value == "5. Start training" for subheader in app.subheader)
    run_training = next(button for button in app.button if button.label == "Run training")
    assert run_training.disabled is False


def test_data_flow_selection_does_not_drop_latest_results(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ML_PLATFORM_RUNS_DIR", str(tmp_path / "runs"))

    app = AppTest.from_file("app.py")
    app.run(timeout=120)

    app.radio[0].set_value("Demo: demo_customer_churn.csv")
    app.run(timeout=120)

    app.multiselect(key="target_columns").set_value(["churn"])
    app.run(timeout=120)

    app.text_input(key="time_budget_text").set_value("5")
    app.run(timeout=120)

    run_training = next(button for button in app.button if button.label == "Run training")
    run_training.click()
    app.run(timeout=120)

    assert any(subheader.value == "6. Review results" for subheader in app.subheader)
    assert any(metric.label == "Completed runs" for metric in app.metric)

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

    app.radio[0].set_value("Demo: demo_customer_churn.csv")
    app.run(timeout=120)

    app.multiselect(key="target_columns").set_value(["churn"])
    app.run(timeout=120)

    def rows_metric_value() -> str:
        return next(metric.value for metric in app.metric if metric.label == "Rows")

    assert rows_metric_value() == "24"

    app.text_area(key="manual_cleaning_brief").set_value("keep rows where churn equals yes")
    app.run(timeout=120)

    generate_button = next(button for button in app.button if button.label == "Generate cleaning rules")
    generate_button.click()
    app.run(timeout=120)

    assert rows_metric_value() == "24"

    apply_button = next(button for button in app.button if button.label == "Apply manual cleaning rules")
    apply_button.click()
    app.run(timeout=120)

    assert rows_metric_value() == "10"


def test_applied_preprocessing_step_invalidates_prepared_data(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ML_PLATFORM_RUNS_DIR", str(tmp_path / "runs"))

    app = AppTest.from_file("app.py")
    app.run(timeout=120)

    app.radio[0].set_value("Demo: demo_customer_churn.csv")
    app.run(timeout=120)

    app.multiselect(key="target_columns").set_value(["churn"])
    app.run(timeout=120)

    app.text_input(key="time_budget_text").set_value("5")
    app.run(timeout=120)

    prepare_data = next(button for button in app.button if button.label == "Prepare data")
    prepare_data.click()
    app.run(timeout=120)

    run_training = next(button for button in app.button if button.label == "Run training")
    assert run_training.disabled is False

    app.selectbox(key="numeric_imputation_strategy").set_value("mean")
    app.run(timeout=120)
    run_training = next(button for button in app.button if button.label == "Run training")
    assert run_training.disabled is False

    apply_missing_step = next(button for button in app.button if button.label == "Apply missing-value step")
    apply_missing_step.click()
    app.run(timeout=120)

    run_training = next(button for button in app.button if button.label == "Run training")
    assert run_training.disabled is False
    run_training.click()
    app.run(timeout=120)

    run_config = _latest_run_config(tmp_path / "runs")
    assert run_config["numeric_imputation_strategy"] == "mean"


def test_run_training_uses_applied_preprocessing_not_unapplied_draft(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ML_PLATFORM_RUNS_DIR", str(tmp_path / "runs"))

    app = AppTest.from_file("app.py")
    app.run(timeout=120)

    app.radio[0].set_value("Demo: demo_customer_churn.csv")
    app.run(timeout=120)

    app.multiselect(key="target_columns").set_value(["churn"])
    app.run(timeout=120)

    app.selectbox(key="numeric_imputation_strategy").set_value("mean")
    app.run(timeout=120)

    app.text_input(key="time_budget_text").set_value("5")
    app.run(timeout=120)

    run_training = next(button for button in app.button if button.label == "Run training")
    run_training.click()
    app.run(timeout=120)

    run_config = _latest_run_config(tmp_path / "runs")
    assert run_config["numeric_imputation_strategy"] == "median"
