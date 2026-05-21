from __future__ import annotations

import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app_beamer_v5 as app_module
from streamlit.testing.v1 import AppTest

from ml_platform.ui_i18n import UI_TRANSLATIONS


def _markdown_blob(app: AppTest) -> str:
    parts = [item.value for item in app.markdown]
    parts.extend(item.value for item in app.caption)
    parts.extend(item.value for item in app.info)
    parts.extend(item.value for item in app.success)
    parts.extend(item.label for item in app.button)
    parts.extend(item.label for item in app.radio)
    parts.extend(item.label for item in app.selectbox)
    return "\n".join(str(part) for part in parts)


def _click_button(app: AppTest, label: str) -> None:
    button = next(button for button in app.button if button.label == label)
    button.click()
    app.run(timeout=120)


def _choose_demo_and_target(app: AppTest, *, language: str) -> None:
    demo_label = (
        "示例：demo_customer_churn.csv"
        if language == "zh-CN"
        else "Demo: demo_customer_churn.csv"
    )
    next_check_label = (
        "下一步：数据预处理" if language == "zh-CN" else "Next: data preprocessing"
    )

    app.radio[0].set_value(demo_label)
    app.run(timeout=120)
    app.multiselect(key="target_columns").set_value(["churn"])
    app.run(timeout=120)
    _click_button(app, next_check_label)


def _seed_cached_result_state(app: AppTest, tmp_path: Path, *, active_step: str) -> None:
    df = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5, 6],
            "feature_b": [10, 11, 12, 13, 14, 15],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    preprocessing_plan = app_module._default_preprocessing_plan()
    dataset_fingerprint = app_module._dataset_fingerprint(df)
    experiment_signature = app_module._experiment_signature(
        dataset_fingerprint=dataset_fingerprint,
        target_columns=["target"],
        task_type_choice="classification",
        time_budget=30,
        priority_metric_choice="accuracy",
        planner_brief="",
        preprocessing_plan=preprocessing_plan,
    )

    run_path = tmp_path / "runs" / "seed-run"
    run_path.mkdir(parents=True, exist_ok=True)
    report_path = run_path / "report.md"
    prediction_path = run_path / "prediction_sample.csv"
    model_path = run_path / "model.bin"
    report_path.write_text("# seeded report\n", encoding="utf-8")
    prediction_path.write_text("prediction\n0\n", encoding="utf-8")
    model_path.write_bytes(b"seed-model")

    result = {
        "target": "target",
        "task_type": "classification",
        "run": SimpleNamespace(run_id="seed-run", path=run_path),
        "trained": SimpleNamespace(
            trainer_name="autogluon",
            leaderboard=[],
            feature_importance=[],
            fit_summary={},
            training_notes=[],
            optimization_metric_used="accuracy",
            model_path=model_path,
        ),
        "metrics": {"accuracy": 0.9, "train_accuracy": 0.95},
        "report": "# seeded report\n",
        "preflight_validation": {
            "ok_to_run": True,
            "feature_count": 2,
            "dropped_target_rows": 0,
            "class_balance": {},
            "issues": [],
            "recommended_excluded_columns": [],
            "detected_leakage_columns": [],
        },
        "postrun_validation": {
            "ok": True,
            "trainer_name": "autogluon",
            "report_mode": "rule_based",
            "generalization_gap": {},
            "issues": [],
        },
        "recommendations": {"summary": [], "next_steps": []},
        "priority_metric": "accuracy",
        "data_flow": {"snapshots": []},
        "report_path": report_path,
        "prediction_path": prediction_path,
        "model_path": model_path,
    }

    app.session_state["active_step"] = active_step
    app.session_state["_current_df"] = df
    app.session_state["_current_source_label"] = "seed.csv"
    app.session_state["_dataset_signature"] = (dataset_fingerprint, tuple(df.columns))
    app.session_state["planner_brief"] = ""
    app.session_state["target_columns"] = ["target"]
    app.session_state["_selected_target_columns"] = ["target"]
    app.session_state["task_type_choice"] = "classification"
    app.session_state["time_budget"] = 30
    app.session_state["time_budget_text"] = "30"
    app.session_state["priority_metric_choice"] = "accuracy"
    app.session_state["apply_feature_engineering"] = False
    app.session_state["excluded_columns"] = []
    app.session_state["test_size"] = 0.2
    app.session_state["high_missing_threshold"] = 0.9
    app.session_state["random_state"] = 42
    app.session_state["random_state_text"] = "42"
    app.session_state["numeric_imputation_strategy"] = "median"
    app.session_state["categorical_imputation_strategy"] = "most_frequent"
    app.session_state["categorical_encoding_strategy"] = "one_hot"
    app.session_state["standardize_numeric"] = True
    for key, value in app_module.AUTOGLUON_FEATURE_GENERATOR_DEFAULTS.items():
        app.session_state[key] = value
    app.session_state["_preprocessing_plan_applied"] = preprocessing_plan
    app.session_state["_preprocessing_plan_draft"] = preprocessing_plan
    app.session_state["_latest_prepared_signature"] = experiment_signature
    app.session_state["_latest_prepared_batches"] = []
    app.session_state["_latest_results_signature"] = experiment_signature
    app.session_state["_latest_results"] = [result]


def _helper_literal_strings(module: ast.Module) -> set[str]:
    helper_arg_counts = {
        "_render_slide_title": 3,
        "_render_step_status": 2,
        "_render_text_items": 3,
        "_render_explanation_strip": 1,
    }
    values: set[str] = set()
    for node in ast.walk(module):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in helper_arg_counts
        ):
            for arg in node.args[: helper_arg_counts[node.func.id]]:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    values.add(arg.value)
    return values


def _metadata_strings(module: ast.Module) -> set[str]:
    values: set[str] = set()
    metadata_names = {
        "BEAMER_SECTION_LABELS",
        "BEAMER_NAV_SECTIONS",
        "SLIDE_TITLES",
        "SLIDE_SUBTITLES",
        "WIZARD_STEP_LABELS",
    }
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if getattr(target, "id", "") not in metadata_names:
                continue
            data = ast.literal_eval(node.value)
            if isinstance(data, dict):
                for value in data.values():
                    if isinstance(value, tuple):
                        values.update(item for item in value if isinstance(item, str))
                    elif isinstance(value, str):
                        values.add(value)
    return values


def test_beamer_v5_zh_top_level_copy_is_localized(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ML_PLATFORM_RUNS_DIR", str(tmp_path / "runs"))

    app = AppTest.from_file("app_beamer_v5.py")
    app.run(timeout=120)

    text_blob = _markdown_blob(app)
    assert app.selectbox(key="ui_language").label == "界面语言"
    assert "数据集上传" in text_blob
    assert "引导式 AutoML 工作流" in text_blob
    assert any('class="beamer-app-headline"' in item.value for item in app.markdown)
    assert not any('class="beamer-help-strip"' in item.value for item in app.markdown)
    assert "Dataset Upload" not in text_blob
    assert "Guided AutoML workflow" not in text_blob
    assert "Interface language" not in text_blob


def test_beamer_v5_english_switches_top_level_copy(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ML_PLATFORM_RUNS_DIR", str(tmp_path / "runs"))

    app = AppTest.from_file("app_beamer_v5.py")
    app.run(timeout=120)
    app.selectbox(key="ui_language").set_value("en")
    app.run(timeout=120)

    text_blob = _markdown_blob(app)
    assert app.selectbox(key="ui_language").label == "Interface language"
    assert "Dataset Upload" in text_blob
    assert "Guided AutoML workflow" in text_blob
    assert any('class="beamer-app-headline"' in item.value for item in app.markdown)
    assert not any('class="beamer-help-strip"' in item.value for item in app.markdown)
    assert "数据集上传" not in text_blob
    assert "界面语言" not in text_blob


def test_beamer_v5_preprocess_frame_copy_is_localized(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ML_PLATFORM_RUNS_DIR", str(tmp_path / "runs"))

    app = AppTest.from_file("app_beamer_v5.py")
    app.run(timeout=120)

    _choose_demo_and_target(app, language="zh-CN")

    text_blob = _markdown_blob(app)
    assert "数据预处理" in text_blob
    assert "预处理 frame：字段健康表。" in text_blob
    assert "预处理 frame" in text_blob
    assert "第 1 / 4 个 frame：字段健康表" in text_blob
    assert "字段健康与预处理控制已分开" in text_blob
    assert "字段健康表" in [button.label for button in app.button]
    assert "预处理细节" in [button.label for button in app.button]
    assert "EDA 概览" in [button.label for button in app.button]
    assert "Preprocess frame:" not in text_blob
    assert "Field health is separated from preprocessing controls." not in text_blob


def test_beamer_v5_nav_dots_are_progress_only(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ML_PLATFORM_RUNS_DIR", str(tmp_path / "runs"))

    app = AppTest.from_file("app_beamer_v5.py")
    app.run(timeout=120)
    _choose_demo_and_target(app, language="zh-CN")

    nav_markup = next(
        markdown.value
        for markdown in app.markdown
        if '<nav class="beamer-nav"' in markdown.value
    )
    nav_keys = {button.key for button in app.button if button.key}
    assert "wizard_nav_dataset_0" not in nav_keys
    assert nav_markup.count('aria-label="数据集 · 来源"') == 1
    assert 'aria-label="数据集 · 结构"' not in nav_markup
    assert 'aria-label="任务 · 预算"' not in nav_markup


def test_beamer_v5_preprocess_can_return_to_target(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ML_PLATFORM_RUNS_DIR", str(tmp_path / "runs"))

    app = AppTest.from_file("app_beamer_v5.py")
    app.run(timeout=120)
    _choose_demo_and_target(app, language="zh-CN")

    text_blob = _markdown_blob(app)
    assert "数据预处理" in text_blob
    app.session_state["target_columns"] = ["churn"]
    _click_button(app, "返回：选择目标")

    text_blob = _markdown_blob(app)
    assert "任务定义" in text_blob
    assert "预处理 frame" not in text_blob


def test_beamer_v5_upload_ready_advances_and_can_return(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ML_PLATFORM_RUNS_DIR", str(tmp_path / "runs"))

    app = AppTest.from_file("app_beamer_v5.py")
    app.run(timeout=120)
    app.radio[0].set_value("示例：demo_customer_churn.csv")
    app.run(timeout=120)

    text_blob = _markdown_blob(app)
    assert "任务定义" in text_blob
    assert "数据集上传" not in text_blob

    _click_button(app, "返回：上传数据")

    text_blob = _markdown_blob(app)
    assert "数据集上传" in text_blob
    assert "任务定义" not in text_blob
    assert "下一步：选择预测目标" in [button.label for button in app.button]


def test_beamer_v5_train_page_does_not_render_results(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ML_PLATFORM_RUNS_DIR", str(tmp_path / "runs"))

    app = AppTest.from_file("app_beamer_v5.py")
    _seed_cached_result_state(app, tmp_path, active_step="train")
    app.run(timeout=120)

    text_blob = _markdown_blob(app)
    assert "模型训练" in text_blob
    assert "评估与导出" not in text_blob
    assert "下载文件" not in text_blob
    assert "本次运行摘要" not in text_blob


def test_beamer_v5_results_page_does_not_render_training(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ML_PLATFORM_RUNS_DIR", str(tmp_path / "runs"))

    app = AppTest.from_file("app_beamer_v5.py")
    _seed_cached_result_state(app, tmp_path, active_step="results")
    app.run(timeout=120)

    text_blob = _markdown_blob(app)
    assert "评估与导出" in text_blob
    assert "结果 frame" in text_blob
    assert "摘要" in [button.label for button in app.button]
    assert "下载" in [button.label for button in app.button]
    _click_button(app, "下载")
    text_blob = _markdown_blob(app)
    assert "下载文件" in text_blob
    assert "模型训练" not in text_blob
    assert "训练现在会直接使用上一步准备好的 train/test 数据" not in text_blob


def test_beamer_v5_translation_audit_covers_metadata_and_helper_literals() -> None:
    source = Path("app_beamer_v5.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    zh_keys = set(UI_TRANSLATIONS["zh-CN"].keys())

    t_literals = {
        node.args[0].value
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_t"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    metadata_literals = _metadata_strings(module)
    helper_literals = _helper_literal_strings(module)
    ignored_internal_values = {"upload", "target", "check", "prepare", "train", "results"}

    expected_keys = (
        t_literals | metadata_literals | helper_literals
    ) - ignored_internal_values
    missing = sorted(expected_keys - zh_keys)
    assert missing == []
