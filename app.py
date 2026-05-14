from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import streamlit as st

from ml_platform.artifacts import artifact_to_dict
from ml_platform.automl import train_model
from ml_platform.cleaning import CleanConfig, clean_and_split
from ml_platform.config import Settings, load_settings
from ml_platform.data_flow import DataFlowTracker
from ml_platform.data_io import read_csv
from ml_platform.eda import generate_eda_summary
from ml_platform.evaluation import evaluate_model
from ml_platform.feature_engineering import suggest_feature_engineering_plan
from ml_platform.llm_report import generate_report_result
from ml_platform.manual_cleaning import (
    DEFAULT_EFFECT_STAGE,
    FILTER_OPERATORS,
    VALID_EFFECT_STAGES,
    VALID_RULE_TYPES,
    apply_manual_cleaning_plan,
    suggest_manual_cleaning_plan,
    validate_manual_cleaning_plan,
)
from ml_platform.planner import suggest_plan
from ml_platform.storage import RunStorage
from ml_platform.validation import build_recommendations, resolve_priority_metric, validate_postrun, validate_preflight

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(PROJECT_ROOT / "app.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("ml_platform")

st.set_page_config(page_title="ML Platform", layout="wide")

HERO_IMAGE_CANDIDATES = (
    PROJECT_ROOT / "data" / "hero.jpg",
    PROJECT_ROOT / "assets" / "hero.jpg",
    PROJECT_ROOT / "assets" / "hero.jpeg",
    PROJECT_ROOT / "assets" / "hero.png",
    PROJECT_ROOT / "彩虹.jpg",
)
LOCAL_LLM_CONFIG_PATH = PROJECT_ROOT / ".ml_platform.local.json"
_LEGACY_PROVIDER_TOKEN = "open" + "ai"
_LEGACY_LOCAL_LLM_KEYS = {
    "llm_api_key": f"{_LEGACY_PROVIDER_TOKEN}_api_key",
    "llm_base_url": f"{_LEGACY_PROVIDER_TOKEN}_base_url",
    "llm_model": f"{_LEGACY_PROVIDER_TOKEN}_model",
}
PLANNER_CACHE_VERSION = 1
FEATURE_PLAN_CACHE_VERSION = 2
MANUAL_CLEANING_PLAN_CACHE_VERSION = 1
DEFAULT_UI_LANGUAGE = "en"
UI_LANGUAGE_OPTIONS = ("en", "zh-CN")
UI_LANGUAGE_LABELS = {
    "en": "English",
    "zh-CN": "中文",
}
UI_TRANSLATIONS = {
    "zh-CN": {
        "Upload a tabular dataset, inspect data quality, train a local baseline,\n                and export the artifacts from one compact experiment surface.": "上传表格数据集，检查数据质量，训练本地基线模型，并在一个紧凑的实验界面中导出产物。",
        "Dates": "数据",
        "Submit": "提交",
        "Experiments": "实验",
        "Artifacts": "产物",
        "Quick start": "快速开始",
        "This page is organized as a guided first run. Advanced settings stay out of the way until you need them.": "这个页面按照首次使用向导组织。高级设置会先收起，等你需要时再展开。",
        "Upload a dataset, choose the column to predict, review the checks, then run training. Optional AI help and advanced adjustments can stay closed for a first pass.": "先上传数据，选择要预测的列，检查校验结果，然后开始训练。第一次使用时，可选 AI 帮助和高级调整都可以先不展开。",
        "Now: {current_action} Next: {next_action}": "当前：{current_action} 下一步：{next_action}",
        "Optional AI help: you can finish the full local training flow without any API key. Add one only if you want AI-generated suggestions and a more natural-language report.": "可选 AI 帮助：即使没有 API key，你也可以完成完整的本地训练流程。只有在你希望获得 AI 建议和更自然语言的报告时，才需要填写。",
        "Optional AI help": "可选 AI 帮助",
        "This environment can remember settings locally. Hosted deployments can also use `LLM_API_KEY` or Streamlit Secrets.": "当前环境可以在本地记住这些设置。部署到线上时，也可以使用 `LLM_API_KEY` 或 Streamlit Secrets。",
        "Local persistence is disabled here, so enter a key per session or configure `LLM_API_KEY` / Streamlit Secrets.": "这里已禁用本地持久化，所以需要每次会话重新输入，或者通过 `LLM_API_KEY` / Streamlit Secrets 配置。",
        "API key": "API key",
        "Uses LLM_API_KEY if empty": "留空时使用 LLM_API_KEY",
        "Base URL": "Base URL",
        "LLM default or compatible API URL": "LLM 默认地址或兼容 API 地址",
        "Model": "模型",
        "Remember LLM settings on this device": "在当前设备记住 LLM 设置",
        "Forget saved LLM settings": "清除已保存的 LLM 设置",
        "5. Review results": "5. 查看结果",
        "Training has finished. Start with the short summary below, then open details or download files.": "训练已经完成。先看下面的简要总结，再按需展开详情或下载文件。",
        "Training finished for: {completed_targets}.": "训练已完成，目标列：{completed_targets}。",
        "Check the validation notes first, then download the report, model, or prediction sample you need.": "建议先看校验说明，再下载你需要的报告、模型或预测样本。",
        "Completed runs": "已完成运行",
        "Targets trained": "已训练目标",
        "Result files per run": "每次运行可下载文件数",
        "Run summary": "运行摘要",
        "Download files": "下载文件",
        "Download model": "下载模型",
        "Download report": "下载报告",
        "Download predictions": "下载预测结果",
        "Detailed results": "详细结果",
        "{target} results": "{target} 结果",
        "Metrics": "指标",
        "Validation checks": "校验检查",
        "Data flow": "数据流",
        "Feature importance": "特征重要性",
        "Analysis report": "分析报告",
        "Artifacts saved to {path}": "产物已保存到 {path}",
        "Planner": "规划器",
        "Task type": "任务类型",
        "Priority metric": "优先指标",
        "Suggested targets": "建议目标列",
        "No target suggestion.": "暂无目标列建议。",
        "Suggested exclusions": "建议排除列",
        "No excluded columns suggested.": "暂无排除列建议。",
        "Notes": "备注",
        "No notes.": "暂无备注。",
        "Risk flags": "风险提示",
        "No risk flags.": "暂无风险提示。",
        "Raw planner JSON": "原始 planner JSON",
        "Accepted ops": "已接受操作",
        "Rejected ops": "已拒绝操作",
        "No feature engineering operations were accepted.": "没有被接受的特征工程操作。",
        "Rejected operations": "已拒绝操作",
        "No rejected operations.": "没有被拒绝的操作。",
        "Operation": "操作",
        "Source": "来源",
        "Detail": "细节",
        "Apply only before training": "仅在训练前应用",
        "Apply before EDA and training": "在 EDA 和训练前应用",
        "Rule effect stage": "规则生效阶段",
        "Rule {index}: {column}": "规则 {index}：{column}",
        "Select column": "选择列",
        "Enabled": "启用",
        "Rule type": "规则类型",
        "Drop column": "删除列",
        "Filter rows": "筛选行",
        "Column": "列",
        "Delete rule": "删除规则",
        "Operator": "运算符",
        "No value is needed for this operator.": "这个运算符不需要填写值。",
        "Values (comma-separated)": "多个值（用逗号分隔）",
        "Value": "值",
        "This rule drops the selected column before downstream processing.": "这条规则会在后续处理前删除所选列。",
        "Rationale": "说明理由",
        "Add blank rule": "新增空白规则",
        "Reset draft": "重置草稿",
        "Effect stage": "生效阶段",
        "Accepted rules": "已接受规则",
        "Target": "目标列",
        "Missing rows": "缺失行数",
        "Unique values": "唯一值数量",
        "Stat": "统计项",
        "Count": "数量",
        "Top target values": "目标列高频值",
        "Numeric feature relationships": "数值特征关系",
        "Numeric feature distribution by target": "按目标列划分的数值特征分布",
        "Categorical feature target distribution": "类别特征的目标分布",
        "No target relationship summary available.": "暂无目标关系摘要。",
        "Priority value": "优先指标值",
        "Test metrics": "测试集指标",
        "No test metrics.": "暂无测试集指标。",
        "Train metrics": "训练集指标",
        "No train metrics.": "暂无训练集指标。",
        "No data flow trace available.": "暂无数据流追踪。",
        "{index}. {label}": "{index}. {label}",
        "Shape": "形状",
        "{stage} | {partition} | delta {delta_text}": "{stage} | {partition} | 变化 {delta_text}",
        "Inspect data flow step": "查看数据流步骤",
        "Stage": "阶段",
        "Partition": "分区",
        "Kind": "类型",
        "Memory": "内存",
        "Rows delta": "行数变化",
        "Columns added": "新增列",
        "Columns removed": "删除列",
        "Columns after": "处理后列数",
        "Metadata": "元数据",
        "No columns added.": "没有新增列。",
        "No columns removed.": "没有删除列。",
        "Preview": "预览",
        "Preview truncated to the first rows.": "预览只显示前几行。",
        "Requested metric": "请求指标",
        "Preflight": "训练前检查",
        "OK": "正常",
        "Blocked": "阻断",
        "Postrun": "训练后检查",
        "Check issues": "需检查问题",
        "Trainer": "训练器",
        "Feature count": "特征数量",
        "Dropped target rows": "被丢弃的目标行数",
        "Report mode": "报告模式",
        "Generalization gap": "泛化差距",
        "Class balance": "类别分布",
        "Recommended exclusions": "建议排除列",
        "No extra exclusions suggested.": "暂无额外排除建议。",
        "Potential leakage columns": "潜在泄漏列",
        "No leakage columns detected.": "未检测到泄漏列。",
        "Preflight issues": "训练前问题",
        "Postrun issues": "训练后问题",
        "Recommendation summary": "建议摘要",
        "No summary available.": "暂无摘要。",
        "Next steps": "下一步建议",
        "No next steps available.": "暂无下一步建议。",
        "No issues surfaced.": "未发现问题。",
        "Enter a whole number. Using {current_value} until corrected.": "请输入整数。在修正前将继续使用 {current_value}。",
        "Enter a value greater than or equal to {min_value}. Using {current_value} until corrected.": "请输入大于等于 {min_value} 的值。在修正前将继续使用 {current_value}。",
        "1. Upload data": "1. 上传数据",
        "Start with one CSV file or a demo dataset. The app saves each run under the configured runs directory.": "先选择一个 CSV 文件或示例数据集。应用会把每次运行结果保存到配置好的 runs 目录。",
        "Data source": "数据来源",
        "Upload CSV": "上传 CSV",
        "Demo: {name}": "示例：{name}",
        "Loaded demo dataset `{demo_name}` with {rows} rows and {columns} columns.": "已加载示例数据集 `{demo_name}`，共 {rows} 行、{columns} 列。",
        "Choose the column you want to predict.": "选择你想预测的列。",
        "No dataset has been loaded yet.": "当前还没有加载数据集。",
        "Upload a CSV or pick a demo dataset to unlock the next step.": "上传 CSV 或选择示例数据集后，才能进入下一步。",
        "Drop a CSV here to unlock schema inspection, missingness checks, training controls, and exportable run artifacts.": "把 CSV 拖到这里后，就可以查看字段结构、缺失情况、训练设置以及可导出的运行产物。",
        "Loaded `{file_name}` with {rows} rows and {columns} columns.": "已加载 `{file_name}`，共 {rows} 行、{columns} 列。",
        "2. Choose what to predict": "2. 选择要预测的内容",
        "Pick the column you want the app to predict. The app can infer the task type automatically.": "选择你希望应用预测的列。应用可以自动判断任务类型。",
        "Target variables": "目标列",
        "Select one or more targets. Multi-target runs train one model per target.": "选择一个或多个目标列。多目标模式会为每个目标列分别训练一个模型。",
        "Your dataset is ready, but no prediction target has been selected yet.": "数据已经准备好，但你还没有选择预测目标。",
        "Select at least one target column to continue to the checks step.": "至少选择一个目标列后，才能进入检查步骤。",
        "Selected target column: {selected_targets}.": "已选择目标列：{selected_targets}。",
        "Selected target columns: {selected_targets}.": "已选择目标列：{selected_targets}。",
        "Review the data checks before starting training.": "开始训练前，先看一下数据检查结果。",
        "Advanced experiment settings": "高级实验设置",
        "Most first runs can keep the defaults here. Open this only if you want more control.": "第一次运行通常保留默认值就可以。只有在你想要更多控制时，再展开这里。",
        "Exclude columns from EDA and training features": "从 EDA 和训练特征中排除列",
        "Excluded columns are removed before EDA and are not used as model features.": "被排除的列会在 EDA 前移除，也不会作为模型特征使用。",
        "Training time budget seconds": "训练时间预算（秒）",
        "This is the score the trainer treats as most important when choosing the best baseline.": "训练器会把这个分数当作选择最佳 baseline 时最重要的指标。",
        "Test size": "测试集比例",
        "Drop feature when missing rate is above": "当缺失率高于该值时删除特征",
        "Random state": "随机种子",
        "Multi-target mode trains and stores one independent run per target. Other selected targets are excluded from each model's feature set.": "多目标模式会为每个目标列分别训练并保存一次独立运行。其他已选目标列不会进入对应模型的特征集合。",
        "3. Check data before training": "3. 训练前检查数据",
        "Use the brief, validation checks, and data summary to catch issues before you spend time training.": "先通过需求描述、校验结果和数据摘要发现问题，再决定是否开始训练。",
        "Planning help": "规划帮助",
        "This optional brief lets you describe your goal in plain language so the app can suggest a sensible first setup.": "这里是可选的自然语言描述框。你可以直接说出目标，让应用给出更合理的初始建议。",
        "Planning brief": "需求描述",
        "Example: predict churn, treat customer_id as reference only, and keep this as a quick first pass.": "例如：预测 churn，把 customer_id 只当作参考字段，并把这次训练当成快速初版。",
        "Optional natural-language brief used to suggest targets, task type, exclusions, and a priority metric.": "可选的自然语言描述，用来建议目标列、任务类型、排除列和优先指标。",
        "Planner suggestion": "规划建议",
        "Apply planner suggestions": "应用规划建议",
        "Advanced adjustments": "高级调整",
        "Most first runs can skip this section. Open it only if you want to clean rows or columns manually, or add extra local feature transformations.": "第一次运行通常可以跳过这一节。只有在你想手动清洗行/列，或者添加额外本地特征变换时，再展开。",
        "Manual cleaning rules": "手动清洗规则",
        "If you already know some rows or columns should be filtered out, draft the rules here before training.": "如果你已经知道某些行或列需要被筛掉，可以在训练前先在这里写规则。",
        "Cleaning rules brief": "清洗规则描述",
        "Example: drop customer_id and keep rows where monthly_spend > 20 and churn equals 1.": "例如：删除 customer_id，并只保留 monthly_spend > 20 且 churn 等于 1 的行。",
        "Natural-language rules are converted into a structured draft. Nothing is applied until you confirm.": "自然语言规则会被转换成结构化草稿。在你确认之前，不会真正应用。",
        "Generate cleaning rules": "生成清洗规则",
        "Start with blank rule": "从空白规则开始",
        "Clear applied manual rules": "清除已应用的手动规则",
        "No manual cleaning rules are in the current draft.": "当前草稿里还没有手动清洗规则。",
        "Rows removed": "删除行数",
        "Rows after": "处理后行数",
        "Rejected rules": "被拒绝的规则",
        "Manual cleaning preview impact": "手动清洗预览影响",
        "Planned cleaning log": "计划中的清洗日志",
        "Apply manual cleaning rules": "应用手动清洗规则",
        "Applied rules remain active until you clear them or apply a different draft.": "已应用的规则会一直生效，直到你清除它们或应用另一份草稿。",
        "Generate rules from a brief or start with a blank rule to configure manual cleaning.": "你可以根据描述生成规则，也可以从空白规则开始配置手动清洗。",
        "Apply local whitelist feature engineering": "应用本地白名单特征工程",
        "LLM can propose a structured plan, but only local whitelisted transformations are executed inside the training pipeline.": "LLM 可以提出结构化方案，但训练流程里真正会执行的只有本地白名单中的变换。",
        "Feature engineering plan": "特征工程方案",
        "The app found blocking issues in the current setup.": "当前配置里存在阻断问题。",
        "Fix the checks for: {failing_targets} before starting training.": "开始训练前，请先解决这些目标列的检查问题：{failing_targets}。",
        "The dataset and target selection passed the current checks.": "当前数据和目标列选择已通过检查。",
        "You can start training after this review, or adjust the setup first.": "看完这部分后，你可以直接开始训练，或者先调整设置。",
        "Preflight validation": "训练前校验",
        "This check looks for blocking issues before training, such as missing target values or no usable feature columns.": "这里会在训练前检查阻断问题，例如目标列缺失值过多，或没有可用特征列。",
        "Resolved priority metric: {priority_metric}": "最终使用的优先指标：{priority_metric}",
        "Data preview": "数据预览",
        "First 50 rows are shown for a quick sanity check before training.": "这里展示前 50 行，方便你在训练前做一次快速确认。",
        "Manual cleaning rules are set to apply only before training. The data preview and EDA below still show the pre-cleaning analysis subset.": "手动清洗规则当前设定为只在训练前生效，所以这里的数据预览和下面的 EDA 仍然显示清洗前的分析子集。",
        "EDA summary": "EDA 摘要",
        "EDA means a quick health check for the dataset: shape, duplicates, missing values, correlations, and target behavior.": "EDA 可以理解为数据健康检查，主要看形状、重复值、缺失值、相关性和目标列表现。",
        "Rows": "行数",
        "Columns": "列数",
        "Duplicate rows": "重复行数",
        "Rows with missing": "存在缺失值的行数",
        "Column profile": "字段概览",
        "Missing rate": "缺失率",
        "Primary target profile": "主目标列概览",
        "Target profile": "目标列概览",
        "Target task types": "目标列任务类型",
        "Missingness": "缺失情况",
        "Correlations": "相关性",
        "Target relationships": "目标关系",
        "Quality warnings": "质量警告",
        "Correlated missingness pairs": "缺失情况相关的字段对",
        "Numeric correlations with target": "与目标列的数值相关性",
        "Strong numeric feature correlations": "高相关的数值特征对",
        "4. Start training": "4. 开始训练",
        "Launch the local baseline after you have reviewed the target, checks, and data summary.": "确认目标列、检查结果和数据摘要后，就可以启动本地 baseline 训练。",
        "Training is blocked by validation issues.": "训练被校验问题阻止。",
        "Resolve the flagged issues for: {failing_targets}.": "请先解决这些目标列的提示问题：{failing_targets}。",
        "The run is ready to start.": "当前运行已经可以开始。",
        "Click Run training to build the local baseline and unlock the results step.": "点击“开始训练”即可生成本地 baseline，并进入结果步骤。",
        "Run training": "开始训练",
        "Cleaning data and training model locally...": "正在本地清洗数据并训练模型……",
        "Resolve blocking preflight issues before training: {failing_targets}": "训练前请先解决这些阻断性检查问题：{failing_targets}",
        "No feature columns remain after excluding selected target and ignored columns.": "排除目标列和忽略列之后，已经没有可用特征列了。",
        "Run completed: {completed_ids}": "运行已完成：{completed_ids}",
        "Language / 语言": "Language / 语言",
        "auto": "自动",
        "classification": "分类",
        "regression": "回归",
        "drop_column": "删除列",
        "filter_row": "筛选行",
        "is_null": "为空",
        "not_null": "不为空",
        "equals": "等于",
        "not_equals": "不等于",
        "in": "属于",
        "not_in": "不属于",
        "contains": "包含",
        "not_contains": "不包含",
        "gt": "大于",
        "gte": "大于等于",
        "lt": "小于",
        "lte": "小于等于",
    }
}


def _ui_language() -> str:
    return str(st.session_state.get("ui_language", DEFAULT_UI_LANGUAGE))


def _t(text: str, **kwargs: Any) -> str:
    template = UI_TRANSLATIONS.get(_ui_language(), {}).get(text, text)
    return template.format(**kwargs) if kwargs else template


def _render_language_switcher() -> None:
    st.sidebar.selectbox(
        _t("Language / 语言"),
        UI_LANGUAGE_OPTIONS,
        key="ui_language",
        format_func=lambda code: UI_LANGUAGE_LABELS.get(code, code),
    )


def _local_llm_config_enabled() -> bool:
    return os.getenv("ML_PLATFORM_ALLOW_LOCAL_LLM_CONFIG", "1") != "0"


def _load_local_llm_config() -> dict[str, str]:
    if not _local_llm_config_enabled():
        return {}
    if not LOCAL_LLM_CONFIG_PATH.exists():
        return {}
    try:
        raw_config = json.loads(LOCAL_LLM_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Unable to read local LLM config: %s", exc)
        return {}
    if not isinstance(raw_config, dict):
        return {}
    normalized: dict[str, str] = {}
    for key in ("llm_api_key", "llm_base_url", "llm_model"):
        value = raw_config.get(key)
        if not isinstance(value, str):
            value = raw_config.get(_LEGACY_LOCAL_LLM_KEYS[key])
        if isinstance(value, str):
            normalized[key] = value
    return normalized


def _save_local_llm_config(config: dict[str, str]) -> None:
    if not _local_llm_config_enabled():
        return
    try:
        LOCAL_LLM_CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
        os.chmod(LOCAL_LLM_CONFIG_PATH, 0o600)
    except OSError as exc:
        logger.warning("Unable to write local LLM config: %s", exc)


def _delete_local_llm_config() -> None:
    if not _local_llm_config_enabled():
        return
    try:
        LOCAL_LLM_CONFIG_PATH.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Unable to delete local LLM config: %s", exc)


def _dataset_fingerprint(df: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes()
    schema = json.dumps(
        {
            "columns": [str(column) for column in df.columns],
            "dtypes": [str(dtype) for dtype in df.dtypes],
            "rows": len(df),
        },
        ensure_ascii=True,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(schema + payload).hexdigest()


def _load_hero_background() -> str:
    hero_image_path = next((path for path in HERO_IMAGE_CANDIDATES if path.exists()), None)
    if hero_image_path is None:
        return (
            "linear-gradient(100deg, rgba(40, 64, 88, 0.98) 0 38%, "
            "rgba(39, 70, 101, 0.86) 38% 62%, rgba(42, 53, 67, 0.92) 62%)"
        )
    encoded = base64.b64encode(hero_image_path.read_bytes()).decode("ascii")
    return (
        "linear-gradient(100deg, rgba(18, 31, 43, 0.78) 0 36%, "
        "rgba(18, 31, 43, 0.54) 36% 68%, rgba(18, 31, 43, 0.34) 68%), "
        f'url("data:image/jpeg;base64,{encoded}")'
    )


def _apply_design_system() -> None:
    hero_background = _load_hero_background()
    st.markdown(
        """
        <style>
            @import url("https://fonts.googleapis.com/css2?family=Exo:wght@400;500;600;700;800;900&display=swap");

            :root {
                --lab-ink: #171717;
                --lab-muted: #5f6b7a;
                --lab-faint: #8994a3;
                --lab-panel: #ffffff;
                --lab-panel-strong: #f4f6f8;
                --lab-line: #dfe4eb;
                --lab-accent: #5c6672;
                --lab-accent-soft: #eef1f4;
                --lab-cyan: #5c6672;
                --lab-warn: #d86b35;
                --lab-bg: #f2f5f8;
            }

            html,
            body,
            .stApp {
                font-family: Exo, "Helvetica Neue", Helvetica, Arial, sans-serif !important;
            }

            button,
            input,
            textarea,
            select,
            [data-baseweb="select"] > div {
                font-family: Exo, "Helvetica Neue", Helvetica, Arial, sans-serif !important;
            }

            .stApp {
                color: var(--lab-ink);
                background:
                    linear-gradient(180deg, #ffffff 0, #f7f8fb 34rem, var(--lab-bg) 100%),
                    var(--lab-bg);
                font-family: Exo, "Helvetica Neue", Helvetica, Arial, sans-serif;
            }

            .stApp::before {
                content: "";
                position: fixed;
                inset: 0;
                pointer-events: none;
                background-image:
                    linear-gradient(120deg, transparent 0 64%, rgba(47, 143, 199, 0.045) 64% 66%, transparent 66%),
                    linear-gradient(90deg, rgba(23, 23, 23, 0.025) 1px, transparent 1px);
                background-size: 280px 140px, 72px 44px;
                mask-image: linear-gradient(to bottom, black, transparent 56%);
            }

            .block-container {
                max-width: 1200px;
                padding-top: 1rem;
                padding-bottom: 4rem;
            }

            [data-testid="stHeader"] {
                display: none;
            }

            [data-testid="stToolbar"],
            [data-testid="stDecoration"],
            [data-testid="stStatusWidget"],
            .stDeployButton {
                display: none !important;
            }

            [data-testid="stSidebar"] {
                background: #f7f8fa;
                border-right: 1px solid var(--lab-line);
            }

            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3 {
                color: var(--lab-ink);
                font-family: Exo, "Helvetica Neue", Helvetica, Arial, sans-serif;
                font-weight: 800;
                letter-spacing: 0;
            }

            h1, h2, h3 {
                color: var(--lab-ink);
                font-family: Exo, "Helvetica Neue", Helvetica, Arial, sans-serif;
                font-weight: 800;
                letter-spacing: 0;
            }

            h1 {
                font-size: clamp(2.45rem, 5vw, 5.6rem) !important;
                line-height: 0.98 !important;
                max-width: 780px;
            }

            h2 {
                margin-top: 1.55rem;
                padding-top: 1rem;
                border-top: 1px solid var(--lab-line);
            }

            h3 {
                font-size: 1.45rem !important;
            }

            p, li, label, .stMarkdown, [data-testid="stCaptionContainer"] {
                color: var(--lab-muted);
                font-family: Exo, "Helvetica Neue", Helvetica, Arial, sans-serif;
            }

            .lab-hero {
                position: relative;
                overflow: hidden;
                min-height: 325px;
                padding: 2.7rem 2.9rem 2.25rem;
                margin-bottom: 1.55rem;
                border: 1px solid #cfd6df;
                background: __HERO_BACKGROUND__;
                background-position: center;
                background-size: cover;
                border-radius: 0;
                box-shadow: 0 10px 24px rgba(21, 37, 54, 0.14);
                animation: labRise 500ms ease-out both;
            }

            .lab-hero::after {
                content: "";
                position: absolute;
                right: 9%;
                top: 0;
                z-index: 0;
                width: 24rem;
                height: 100%;
                border: 0;
                border-radius: 0;
                transform: skewX(-11deg);
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.01));
            }

            .lab-kicker {
                display: inline-flex;
                align-items: center;
                gap: 0.55rem;
                margin-bottom: 1rem;
                color: #ffffff;
                font-size: 0.9rem;
                font-weight: 800;
                letter-spacing: 0.02em;
                text-transform: none;
                font-family: Exo, "Helvetica Neue", Helvetica, Arial, sans-serif;
            }

            .lab-kicker::before {
                content: none;
            }

            .lab-hero h1 {
                position: relative;
                z-index: 2;
                margin: 0 0 1rem;
                color: #ffffff;
                text-shadow: 0 2px 14px rgba(0, 0, 0, 0.18);
            }

            .lab-hero p {
                position: relative;
                z-index: 2;
                max-width: 700px;
                margin: 0;
                color: #edf4fb;
                font-size: 1.08rem;
                line-height: 1.62;
                font-weight: 400;
            }

            .lab-rail {
                display: flex;
                position: relative;
                z-index: 2;
                flex-wrap: wrap;
                gap: 0.65rem;
                margin-top: 1.6rem;
            }

            .lab-pill {
                border: 1px solid var(--lab-line);
                color: #ffffff;
                background: rgba(255, 255, 255, 0.18);
                border-radius: 7px;
                padding: 0.5rem 0.75rem;
                font-size: 0.76rem;
                font-family: Exo, "Helvetica Neue", Helvetica, Arial, sans-serif;
                text-transform: none;
                letter-spacing: 0;
                font-weight: 700;
                backdrop-filter: blur(4px);
            }

            .lab-empty {
                padding: 1.35rem 1.5rem;
                border: 1px dashed #c7ced8;
                background: #ffffff;
                color: #4d5562;
                border-radius: 9px;
                font-family: Exo, "Helvetica Neue", Helvetica, Arial, sans-serif;
            }

            .lab-caption {
                margin: -0.35rem 0 0.75rem;
                color: var(--lab-muted);
            }

            [data-testid="stMetric"] {
                padding: 1rem 1.1rem;
                border: 1px solid var(--lab-line);
                background: var(--lab-panel);
                border-radius: 9px;
                box-shadow: 0 6px 18px rgba(21, 37, 54, 0.06);
            }

            [data-testid="stMetricLabel"] {
                color: var(--lab-muted);
                font-family: Exo, "Helvetica Neue", Helvetica, Arial, sans-serif;
            }

            [data-testid="stMetricValue"] {
                color: var(--lab-accent);
                font-family: Exo, "Helvetica Neue", Helvetica, Arial, sans-serif;
                font-weight: 800;
            }

            [data-testid="stFileUploader"] section,
            [data-testid="stDataFrame"],
            [data-testid="stJson"],
            [data-testid="stExpander"],
            div[data-testid="stTabs"] [role="tabpanel"] {
                border: 1px solid var(--lab-line);
                background: var(--lab-panel);
                border-radius: 9px;
                box-shadow: 0 6px 18px rgba(21, 37, 54, 0.05);
            }

            [data-testid="stFileUploader"] section {
                padding: 1.1rem;
                border-radius: 9px;
            }

            [data-testid="stFileUploader"] section,
            [data-testid="stFileUploader"] small,
            [data-testid="stFileUploader"] p,
            [data-testid="stFileUploader"] label,
            [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] * {
                color: #4d5562 !important;
            }

            [data-testid="stFileUploader"] section svg {
                color: #5c6672 !important;
                width: 2.2rem !important;
                height: 2.2rem !important;
            }

            /* Icon sizing for all interactive components */
            svg {
                display: inline-block;
                flex-shrink: 0;
            }

            [data-testid="stExpander"] svg {
                width: 1rem !important;
                height: 1rem !important;
            }

            [data-baseweb="select"] svg,
            [data-baseweb="menu"] svg {
                width: 1.1rem !important;
                height: 1.1rem !important;
            }

            .stAlert svg {
                width: 1.1rem !important;
                height: 1.1rem !important;
            }

            [data-testid="stMetric"] svg {
                width: 1.3rem !important;
                height: 1.3rem !important;
            }

            [data-testid="stSpinner"] svg {
                width: 1.5rem !important;
                height: 1.5rem !important;
            }

            button [data-testid="stBaseButton-icon"] svg,
            .stDownloadButton button svg {
                width: 1rem !important;
                height: 1rem !important;
            }

            .stCheckbox svg,
            .stRadio svg {
                width: 1.05rem !important;
                height: 1.05rem !important;
            }

            [data-testid="stStatusWidget"] svg {
                width: 1.4rem !important;
                height: 1.4rem !important;
            }

            [data-testid="stFileUploader"] button {
                color: #ffffff !important;
                background: #5c6672 !important;
                border: 1px solid #5c6672 !important;
                border-radius: 6px !important;
            }

            input,
            textarea,
            [data-baseweb="select"] > div {
                color: var(--lab-ink) !important;
                background: #ffffff !important;
                border-color: #c7ced8 !important;
            }

            button[kind="primary"],
            .stDownloadButton button {
                border: 1px solid #4e5863 !important;
                background: #4e5863 !important;
                color: #ffffff !important;
                font-weight: 800 !important;
                box-shadow: 0 8px 18px rgba(21, 37, 54, 0.14);
                transition: transform 160ms ease, box-shadow 160ms ease;
            }

            button[kind="primary"] *,
            .stDownloadButton button * {
                color: #ffffff !important;
            }

            button[kind="primary"]:disabled,
            button[kind="primary"][disabled] {
                border-color: #c3cad3 !important;
                background: #e3e7ec !important;
                color: #7b8592 !important;
                box-shadow: none;
                opacity: 1 !important;
            }

            button[kind="primary"]:disabled *,
            button[kind="primary"][disabled] * {
                color: #7b8592 !important;
            }

            button[kind="primary"]:hover,
            .stDownloadButton button:hover {
                transform: translateY(-1px);
                box-shadow: 0 12px 24px rgba(21, 37, 54, 0.2);
            }

            button,
            input,
            textarea,
            [data-baseweb="select"] > div {
                border-radius: 6px !important;
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 0.35rem;
                border-bottom: 1px solid var(--lab-line);
            }

            .stTabs [data-baseweb="tab"] {
                color: var(--lab-muted);
                background: #f8f9fb;
                border: 1px solid var(--lab-line);
                border-radius: 6px;
                font-weight: 700;
            }

            .stTabs [aria-selected="true"] {
                color: #ffffff !important;
                background: var(--lab-accent);
                border-color: var(--lab-accent);
            }

            .stTabs [aria-selected="true"] *,
            .stTabs [aria-selected="true"] p,
            .stTabs [aria-selected="true"] span {
                color: #ffffff !important;
            }

            [data-testid="stAlert"] {
                border-radius: 9px;
                border: 1px solid #b9edf5;
                background: var(--lab-accent-soft);
            }

            @keyframes labRise {
                from { opacity: 0; transform: translateY(14px); }
                to { opacity: 1; transform: translateY(0); }
            }

            @media (max-width: 720px) {
                .block-container {
                    padding-left: 1rem;
                    padding-right: 1rem;
                }

                .lab-hero {
                    min-height: 260px;
                    padding: 1.5rem;
                }

                .lab-hero::after {
                    opacity: 0.42;
                }
            }
        </style>
        """.replace("__HERO_BACKGROUND__", hero_background),
        unsafe_allow_html=True,
    )


def _render_hero() -> None:
    hero_description = _t(
        "Upload a tabular dataset, inspect data quality, train a local baseline,\n                and export the artifacts from one compact experiment surface."
    )
    hero_pills = [_t("Dates"), _t("Submit"), _t("Experiments"), _t("Artifacts")]
    st.markdown(
        """
        <section class="lab-hero">
            <h1>ML Platform</h1>
            <p>
                __HERO_DESCRIPTION__
            </p>
            <div class="lab-rail">
                <span class="lab-pill">__PILL_1__</span>
                <span class="lab-pill">__PILL_2__</span>
                <span class="lab-pill">__PILL_3__</span>
                <span class="lab-pill">__PILL_4__</span>
            </div>
        </section>
        """
        .replace("__HERO_DESCRIPTION__", hero_description)
        .replace("__PILL_1__", hero_pills[0])
        .replace("__PILL_2__", hero_pills[1])
        .replace("__PILL_3__", hero_pills[2])
        .replace("__PILL_4__", hero_pills[3]),
        unsafe_allow_html=True,
    )


def _render_help_center() -> None:
    st.subheader(_t("Quick start"))
    _section_caption(_t("This page is organized as a guided first run. Advanced settings stay out of the way until you need them."))
    st.info(
        _t(
            "Upload a dataset, choose the column to predict, review the checks, then run training. Optional AI help and advanced adjustments can stay closed for a first pass."
        )
    )


def _section_caption(text: str) -> None:
    st.markdown(f'<p class="lab-caption">{text}</p>', unsafe_allow_html=True)


def _render_step_status(current_action: str, next_action: str, level: str = "info") -> None:
    message = _t("Now: {current_action} Next: {next_action}", current_action=current_action, next_action=next_action)
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    else:
        st.info(message)


def _configure_llm_settings(settings: Settings) -> Settings:
    allow_local_llm_config = _local_llm_config_enabled()
    saved_config = _load_local_llm_config()
    saved_api_key = saved_config.get("llm_api_key", "")
    saved_base_url = saved_config.get("llm_base_url", "")
    saved_model = saved_config.get("llm_model", "")
    _section_caption(_t("Optional AI help: you can finish the full local training flow without any API key. Add one only if you want AI-generated suggestions and a more natural-language report."))

    api_key = saved_api_key
    base_url = saved_base_url or settings.llm_base_url or ""
    model = saved_model or settings.llm_model

    with st.expander(_t("Optional AI help"), expanded=False):
        if allow_local_llm_config:
            st.caption(_t("This environment can remember settings locally. Hosted deployments can also use `LLM_API_KEY` or Streamlit Secrets."))
        else:
            st.caption(_t("Local persistence is disabled here, so enter a key per session or configure `LLM_API_KEY` / Streamlit Secrets."))

        config_cols = st.columns(3)
        with config_cols[0]:
            api_key = st.text_input(
                _t("API key"),
                value=saved_api_key,
                key="_llm_api_key",
                type="password",
                placeholder=_t("Uses LLM_API_KEY if empty"),
            )
        with config_cols[1]:
            base_url = st.text_input(
                _t("Base URL"),
                value=saved_base_url or settings.llm_base_url or "",
                key="_llm_base_url",
                placeholder=_t("LLM default or compatible API URL"),
            )
        with config_cols[2]:
            model = st.text_input(_t("Model"), value=saved_model or settings.llm_model, key="_llm_model")

        if allow_local_llm_config:
            remember_config = st.checkbox(_t("Remember LLM settings on this device"), value=bool(saved_api_key))
            if remember_config and api_key:
                _save_local_llm_config(
                    {
                        "llm_api_key": api_key,
                        "llm_base_url": base_url,
                        "llm_model": model,
                    }
                )
            elif not remember_config and saved_config:
                _delete_local_llm_config()
                saved_config = {}

            if saved_config and st.button(_t("Forget saved LLM settings")):
                _delete_local_llm_config()
                st.rerun()

    return Settings(
        runs_dir=settings.runs_dir,
        data_dir=settings.data_dir,
        llm_api_key=api_key or saved_api_key or settings.llm_api_key,
        llm_base_url=base_url or settings.llm_base_url or None,
        llm_model=model or settings.llm_model,
    )


def _infer_task_type(df: pd.DataFrame, target: str) -> str:
    series = df[target].dropna()
    if not pd.api.types.is_numeric_dtype(series):
        return "classification"
    unique_count = int(series.nunique())
    if unique_count <= max(20, int(len(series) * 0.05)):
        return "classification"
    return "regression"


def _target_task_types(df: pd.DataFrame, targets: list[str], task_type_choice: str) -> dict[str, str]:
    if task_type_choice in {"classification", "regression"}:
        return {target: task_type_choice for target in targets}
    return {target: _infer_task_type(df, target) for target in targets}


def _experiment_signature(
    *,
    dataset_fingerprint: str,
    target_columns: list[str],
    task_type_choice: str,
    time_budget: int,
    excluded_columns: list[str],
    test_size: float,
    high_missing_threshold: float,
    random_state: int,
    priority_metric_choice: str,
    planner_brief: str,
    feature_plan: Any,
    manual_cleaning_plan: Any,
) -> str:
    payload = {
        "dataset_fingerprint": dataset_fingerprint,
        "target_columns": target_columns,
        "task_type_choice": task_type_choice,
        "time_budget": time_budget,
        "excluded_columns": excluded_columns,
        "test_size": test_size,
        "high_missing_threshold": high_missing_threshold,
        "random_state": random_state,
        "priority_metric_choice": priority_metric_choice,
        "planner_brief": planner_brief.strip(),
        "feature_engineering_operations": _feature_plan_operations(feature_plan) if feature_plan else [],
        "manual_cleaning_plan": artifact_to_dict(manual_cleaning_plan) if manual_cleaning_plan else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def _initialize_experiment_state(dataset_signature: str, columns: list[str]) -> None:
    signature = (dataset_signature, tuple(columns))
    if st.session_state.get("_dataset_signature") == signature:
        return
    st.session_state["_dataset_signature"] = signature
    st.session_state["planner_brief"] = ""
    st.session_state["target_columns"] = []
    st.session_state["task_type_choice"] = "auto"
    st.session_state["time_budget"] = 30
    st.session_state["time_budget_text"] = "30"
    st.session_state["priority_metric_choice"] = "auto"
    st.session_state["apply_feature_engineering"] = False
    st.session_state["excluded_columns"] = []
    st.session_state["test_size"] = 0.2
    st.session_state["high_missing_threshold"] = 0.9
    st.session_state["random_state"] = 42
    st.session_state["random_state_text"] = "42"
    st.session_state["_planner_signature"] = None
    st.session_state["_planner_suggestion"] = None
    st.session_state["_feature_engineering_plan_signature"] = None
    st.session_state["_feature_engineering_plan"] = None
    st.session_state["_feature_engineering_override_plan"] = None
    st.session_state["manual_cleaning_brief"] = ""
    st.session_state["_manual_cleaning_plan_signature"] = None
    st.session_state["_manual_cleaning_suggested_plan"] = None
    st.session_state["_manual_cleaning_plan"] = None
    st.session_state["_manual_cleaning_override_plan"] = None
    st.session_state["_latest_results_signature"] = None
    st.session_state["_latest_results"] = []


def _render_run_outputs(results: list[dict[str, object]]) -> None:
    st.subheader(_t("5. Review results"))
    _section_caption(_t("Training has finished. Start with the short summary below, then open details or download files."))
    completed_targets = ", ".join(str(result["target"]) for result in results)
    _render_step_status(
        _t("Training finished for: {completed_targets}.", completed_targets=completed_targets),
        _t("Check the validation notes first, then download the report, model, or prediction sample you need."),
        level="success",
    )
    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.metric(_t("Completed runs"), len(results))
    with summary_cols[1]:
        st.metric(_t("Targets trained"), len({str(result["target"]) for result in results}))
    with summary_cols[2]:
        st.metric(_t("Result files per run"), 3)
    summary_rows = [
        {
            "target": str(result["target"]),
            "task_type": _t(str(result["task_type"])),
            "priority_metric": str(result["priority_metric"]),
            "trainer": str(result["trained"].trainer_name),
        }
        for result in results
    ]
    summary_frame = pd.DataFrame(summary_rows).rename(
        columns={
            "target": _t("Target"),
            "task_type": _t("Task type"),
            "priority_metric": _t("Priority metric"),
            "trainer": _t("Trainer"),
        }
    )
    st.write(_t("Run summary"))
    st.dataframe(summary_frame, hide_index=True, use_container_width=True)
    st.write(_t("Download files"))
    for result in results:
        run = result["run"]
        label = f"{result['target']} ({_t(str(result['task_type']))})"
        with st.expander(label, expanded=len(results) == 1):
            download_cols = st.columns(3)
            with download_cols[0]:
                st.download_button(
                    _t("Download model"),
                    data=result["model_path"].read_bytes(),
                    file_name=f"{run.run_id}_{result['target']}_model.joblib",
                    mime="application/octet-stream",
                )
            with download_cols[1]:
                st.download_button(
                    _t("Download report"),
                    data=result["report_path"].read_text(encoding="utf-8"),
                    file_name=f"{run.run_id}_{result['target']}_report.md",
                    mime="text/markdown",
                )
            with download_cols[2]:
                st.download_button(
                    _t("Download predictions"),
                    data=result["prediction_path"].read_text(encoding="utf-8"),
                    file_name=f"{run.run_id}_{result['target']}_prediction_sample.csv",
                    mime="text/csv",
                )

    st.write(_t("Detailed results"))
    for result in results:
        run = result["run"]
        with st.expander(_t("{target} results", target=result["target"]), expanded=len(results) == 1):
            st.write(_t("Metrics"))
            _render_metrics(result["metrics"], priority_metric=result["priority_metric"])
            st.write(_t("Validation checks"))
            _render_validation_summary(
                preflight=artifact_to_dict(result["preflight_validation"]),
                postrun=artifact_to_dict(result["postrun_validation"]),
                recommendations=artifact_to_dict(result["recommendations"]),
                priority_metric=result["priority_metric"],
            )
            st.write(_t("Data flow"))
            _render_data_flow(result["data_flow"])
            st.write(_t("Feature importance"))
            st.dataframe(pd.DataFrame(result["trained"].feature_importance), use_container_width=True)
            st.write(_t("Analysis report"))
            st.markdown(result["report"])
            st.caption(_t("Artifacts saved to {path}", path=Path(run.path).resolve()))


def _queue_plan_suggestion(plan_suggestion: dict[str, object], columns: list[str]) -> None:
    suggested_targets = [column for column in plan_suggestion.get("suggested_targets", []) if column in columns]
    suggested_task_type = plan_suggestion.get("suggested_task_type")
    pending_targets = suggested_targets or st.session_state.get("target_columns", [])
    current_targets = set(pending_targets)
    excluded = [
        column
        for column in plan_suggestion.get("suggested_excluded_columns", [])
        if column in columns and column not in current_targets
    ]
    metric = plan_suggestion.get("priority_metric")
    st.session_state["_pending_plan_suggestion"] = {
        "target_columns": suggested_targets,
        "task_type_choice": suggested_task_type,
        "excluded_columns": excluded,
        "priority_metric_choice": metric,
    }


def _consume_pending_plan_suggestion(columns: list[str]) -> None:
    pending = st.session_state.pop("_pending_plan_suggestion", None)
    if not pending:
        return

    pending_targets = [column for column in pending.get("target_columns", []) if column in columns]
    if pending_targets:
        st.session_state["target_columns"] = pending_targets

    suggested_task_type = pending.get("task_type_choice")
    if suggested_task_type in {"auto", "classification", "regression"}:
        st.session_state["task_type_choice"] = suggested_task_type

    excluded_columns = [
        column
        for column in pending.get("excluded_columns", [])
        if column in columns and column not in set(st.session_state.get("target_columns", []))
    ]
    st.session_state["excluded_columns"] = excluded_columns

    metric = pending.get("priority_metric_choice")
    if metric in {"auto", "accuracy", "f1_weighted", "precision_weighted", "recall_weighted", "roc_auc", "rmse", "mae", "r2"}:
        st.session_state["priority_metric_choice"] = metric


def _render_text_items(title: str, items: list[object], empty_text: str) -> None:
    st.write(_t(title))
    normalized_items = items if isinstance(items, list) else [items]
    clean_items = [str(item) for item in normalized_items if str(item).strip()]
    if not clean_items:
        st.caption(_t(empty_text))
        return
    for item in clean_items:
        st.write(f"- {item}")


def _normalize_item_list(value: Any) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _render_planner_suggestion(plan_data: dict[str, object]) -> None:
    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.metric(_t("Planner"), str(plan_data.get("planner_name") or "unknown"))
    with summary_cols[1]:
        st.metric(_t("Task type"), str(plan_data.get("suggested_task_type") or "auto"))
    with summary_cols[2]:
        st.metric(_t("Priority metric"), str(plan_data.get("priority_metric") or "auto"))

    target_items = [str(item) for item in plan_data.get("suggested_targets", []) if str(item).strip()]
    excluded_items = [str(item) for item in plan_data.get("suggested_excluded_columns", []) if str(item).strip()]
    detail_cols = st.columns(2)
    with detail_cols[0]:
        st.write(_t("Suggested targets"))
        if target_items:
            st.dataframe(pd.DataFrame({_t("Target"): target_items}), hide_index=True, use_container_width=True)
        else:
            st.caption(_t("No target suggestion."))
    with detail_cols[1]:
        st.write(_t("Suggested exclusions"))
        if excluded_items:
            st.dataframe(pd.DataFrame({_t("Column"): excluded_items}), hide_index=True, use_container_width=True)
        else:
            st.caption(_t("No excluded columns suggested."))

    notes_cols = st.columns(2)
    with notes_cols[0]:
        _render_text_items("Notes", _normalize_item_list(plan_data.get("notes")), "No notes.")
    with notes_cols[1]:
        _render_text_items("Risk flags", _normalize_item_list(plan_data.get("risk_flags")), "No risk flags.")

    with st.expander(_t("Raw planner JSON"), expanded=False):
        st.json(plan_data, expanded=True)


def _get_planner_suggestion(
    df: pd.DataFrame,
    eda_summary: dict[str, Any],
    settings: Settings,
    user_brief: str,
    dataset_fingerprint: str,
):
    signature = (
        PLANNER_CACHE_VERSION,
        dataset_fingerprint,
        tuple(str(column) for column in df.columns),
        tuple(str(dtype) for dtype in df.dtypes),
        len(df),
        user_brief.strip(),
        bool(settings.llm_api_key),
        settings.llm_base_url or "",
        settings.llm_model,
    )
    if st.session_state.get("_planner_signature") == signature:
        return st.session_state.get("_planner_suggestion")

    plan = suggest_plan(
        df=df,
        eda_summary=eda_summary,
        settings=settings,
        user_brief=user_brief,
    )
    st.session_state["_planner_signature"] = signature
    st.session_state["_planner_suggestion"] = plan
    return plan


def _render_feature_engineering_plan(plan_data: dict[str, object]) -> None:
    operations = _normalize_item_list(plan_data.get("operations"))
    rejected = _normalize_item_list(plan_data.get("rejected_operations"))
    notes = _normalize_item_list(plan_data.get("notes"))

    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.metric(_t("Planner"), str(plan_data.get("planner_name") or "local_whitelist"))
    with summary_cols[1]:
        st.metric(_t("Accepted ops"), len(operations))
    with summary_cols[2]:
        st.metric(_t("Rejected ops"), len(rejected))

    if operations:
        rows = []
        for operation in operations:
            if not isinstance(operation, dict):
                continue
            rows.append(
                {
                    _t("Operation"): operation.get("operation"),
                    _t("Source"): operation.get("source_column") or ", ".join(operation.get("columns", [])),
                    _t("Detail"): operation.get("operator") or ", ".join(operation.get("parts", [])) or operation.get("bins") or "",
                    _t("Rationale"): operation.get("rationale", ""),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.caption(_t("No feature engineering operations were accepted."))

    detail_cols = st.columns(2)
    with detail_cols[0]:
        _render_text_items("Notes", notes, "No notes.")
    with detail_cols[1]:
        _render_text_items("Rejected operations", rejected, "No rejected operations.")


def _feature_plan_operations(plan_data: Any) -> list[Any]:
    if isinstance(plan_data, dict):
        return list(plan_data.get("operations", []))
    return list(getattr(plan_data, "operations", []))


def _get_feature_engineering_plan(
    df: pd.DataFrame,
    target: str,
    settings: Settings,
    user_brief: str,
):
    signature = (
        FEATURE_PLAN_CACHE_VERSION,
        tuple(str(column) for column in df.columns),
        tuple(str(dtype) for dtype in df.dtypes),
        len(df),
        target,
        user_brief.strip(),
        bool(settings.llm_api_key),
        settings.llm_base_url or "",
        settings.llm_model,
    )
    if st.session_state.get("_feature_engineering_plan_signature") == signature:
        return st.session_state.get("_feature_engineering_plan")

    plan = suggest_feature_engineering_plan(
        df=df,
        target=target,
        eda_summary=generate_eda_summary(df, target=target),
        settings=settings,
        user_brief=user_brief,
    )
    st.session_state["_feature_engineering_plan_signature"] = signature
    st.session_state["_feature_engineering_plan"] = plan
    return plan


def _clone_json_data(value: Any) -> Any:
    return json.loads(json.dumps(artifact_to_dict(value), ensure_ascii=False))


def _new_manual_cleaning_rule(column: str = "", *, rule_type: str = "filter_row") -> dict[str, object]:
    return {
        "id": f"manual_rule_{os.urandom(4).hex()}",
        "enabled": True,
        "rule_type": rule_type,
        "column": column,
        "operator": "equals" if rule_type == "filter_row" else None,
        "value": None,
        "rationale": "",
    }


def _blank_manual_cleaning_plan(user_brief: str = "") -> dict[str, object]:
    return {
        "planner_name": "manual",
        "user_brief": user_brief,
        "effect_stage": DEFAULT_EFFECT_STAGE,
        "rules": [_new_manual_cleaning_rule()],
        "notes": [],
        "rejected_rules": [],
    }


def _manual_cleaning_rule_value_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if str(item).strip())
    return str(value)


def _manual_cleaning_effect_stage_label(effect_stage: str) -> str:
    if effect_stage == "pre_training":
        return _t("Apply only before training")
    return _t("Apply before EDA and training")


def _manual_cleaning_rule_rows(plan_data: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rule in plan_data.get("rules", []):
        if not isinstance(rule, dict):
            continue
        rows.append(
            {
                _t("Enabled"): bool(rule.get("enabled", True)),
                _t("Rule type"): _t(str(rule.get("rule_type"))),
                _t("Column"): rule.get("column"),
                _t("Operator"): _t(str(rule.get("operator") or "-")),
                _t("Value"): _manual_cleaning_rule_value_text(rule.get("value")) or "-",
                _t("Rationale"): rule.get("rationale") or "",
            }
        )
    return rows


def _get_manual_cleaning_plan(
    df: pd.DataFrame,
    target: str,
    settings: Settings,
    user_brief: str,
):
    signature = (
        MANUAL_CLEANING_PLAN_CACHE_VERSION,
        tuple(str(column) for column in df.columns),
        tuple(str(dtype) for dtype in df.dtypes),
        len(df),
        target,
        user_brief.strip(),
        bool(settings.llm_api_key),
        settings.llm_base_url or "",
        settings.llm_model,
    )
    if st.session_state.get("_manual_cleaning_plan_signature") == signature:
        return st.session_state.get("_manual_cleaning_suggested_plan")

    plan = suggest_manual_cleaning_plan(
        df=df,
        target=target,
        settings=settings,
        user_brief=user_brief,
    )
    st.session_state["_manual_cleaning_plan_signature"] = signature
    st.session_state["_manual_cleaning_suggested_plan"] = plan
    return plan


def _render_manual_cleaning_editor(
    draft_plan: dict[str, object],
    *,
    available_columns: list[str],
) -> dict[str, object]:
    current_rules = [item for item in draft_plan.get("rules", []) if isinstance(item, dict)]
    effect_stage_value = str(draft_plan.get("effect_stage") or DEFAULT_EFFECT_STAGE)
    if effect_stage_value not in VALID_EFFECT_STAGES:
        effect_stage_value = DEFAULT_EFFECT_STAGE
    effect_stage_options = ["pre_eda", "pre_training"]
    effect_stage = st.radio(
        _t("Rule effect stage"),
        effect_stage_options,
        index=effect_stage_options.index(effect_stage_value),
        format_func=_manual_cleaning_effect_stage_label,
        horizontal=True,
    )

    updated_rules: list[dict[str, object]] = []
    delete_rule_id: str | None = None
    for index, rule in enumerate(current_rules, start=1):
        rule_id = str(rule.get("id") or f"manual_rule_{index}")
        enabled_default = bool(rule.get("enabled", True))
        rule_type_default = str(rule.get("rule_type") or "filter_row")
        if rule_type_default not in VALID_RULE_TYPES:
            rule_type_default = "filter_row"
        column_default = str(rule.get("column") or "")
        operator_default = str(rule.get("operator") or "equals") if rule_type_default == "filter_row" else ""
        if operator_default not in FILTER_OPERATORS:
            operator_default = "equals"
        value_default = _manual_cleaning_rule_value_text(rule.get("value"))
        rationale_default = str(rule.get("rationale") or "")

        title = _t("Rule {index}: {column}", index=index, column=column_default or _t("Select column"))
        with st.expander(title, expanded=len(current_rules) == 1):
            top_cols = st.columns([0.8, 1.0, 1.3, 0.9])
            with top_cols[0]:
                enabled = st.checkbox(_t("Enabled"), value=enabled_default, key=f"manual_rule_enabled_{rule_id}")
            with top_cols[1]:
                rule_type = st.selectbox(
                    _t("Rule type"),
                    ["drop_column", "filter_row"],
                    index=["drop_column", "filter_row"].index(rule_type_default),
                    key=f"manual_rule_type_{rule_id}",
                    format_func=lambda value: _t("Drop column") if value == "drop_column" else _t("Filter rows"),
                )
            with top_cols[2]:
                column_options = ["", *available_columns]
                column_index = column_options.index(column_default) if column_default in column_options else 0
                column = st.selectbox(
                    _t("Column"),
                    column_options,
                    index=column_index,
                    key=f"manual_rule_column_{rule_id}",
                    format_func=lambda value: _t("Select column") if value == "" else value,
                )
            with top_cols[3]:
                delete_clicked = st.button(_t("Delete rule"), key=f"manual_rule_delete_{rule_id}")
                if delete_clicked:
                    delete_rule_id = rule_id

            operator: str | None = None
            parsed_value: str | list[str] | None = None
            if rule_type == "filter_row":
                operator_cols = st.columns([1.1, 1.9])
                with operator_cols[0]:
                    operator_options = [
                        "is_null",
                        "not_null",
                        "equals",
                        "not_equals",
                        "in",
                        "not_in",
                        "contains",
                        "not_contains",
                        "gt",
                        "gte",
                        "lt",
                        "lte",
                    ]
                    operator = st.selectbox(
                        _t("Operator"),
                        operator_options,
                        index=operator_options.index(operator_default),
                        key=f"manual_rule_operator_{rule_id}",
                        format_func=lambda value: _t(value),
                    )
                with operator_cols[1]:
                    if operator in {"is_null", "not_null"}:
                        st.caption(_t("No value is needed for this operator."))
                    else:
                        label = _t("Values (comma-separated)") if operator in {"in", "not_in"} else _t("Value")
                        raw_value = st.text_input(label, value=value_default, key=f"manual_rule_value_{rule_id}")
                        if operator in {"in", "not_in"}:
                            parsed_value = [item.strip() for item in raw_value.split(",") if item.strip()]
                        else:
                            parsed_value = raw_value.strip() or None
            else:
                st.caption(_t("This rule drops the selected column before downstream processing."))

            rationale = st.text_input(_t("Rationale"), value=rationale_default, key=f"manual_rule_rationale_{rule_id}")
            updated_rules.append(
                {
                    "id": rule_id,
                    "enabled": enabled,
                    "rule_type": rule_type,
                    "column": column,
                    "operator": operator,
                    "value": parsed_value,
                    "rationale": rationale,
                }
            )

    if delete_rule_id:
        updated_rules = [rule for rule in updated_rules if str(rule.get("id")) != delete_rule_id]

    controls = st.columns(2)
    add_blank_rule = controls[0].button(_t("Add blank rule"))
    reset_draft = controls[1].button(_t("Reset draft"))

    updated_plan = {
        "planner_name": str(draft_plan.get("planner_name") or "manual"),
        "user_brief": str(draft_plan.get("user_brief") or ""),
        "effect_stage": effect_stage,
        "rules": updated_rules,
        "notes": list(draft_plan.get("notes", [])) if isinstance(draft_plan.get("notes"), list) else [],
        "rejected_rules": [],
    }
    if add_blank_rule:
        updated_plan["rules"].append(_new_manual_cleaning_rule())
    if reset_draft:
        applied = st.session_state.get("_manual_cleaning_plan")
        st.session_state["_manual_cleaning_override_plan"] = _clone_json_data(applied) if applied else None
        st.rerun()

    st.session_state["_manual_cleaning_override_plan"] = updated_plan
    if delete_rule_id or add_blank_rule:
        st.rerun()
    return updated_plan


def _render_target_profile(target_data: dict[str, object]) -> None:
    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.metric(_t("Target"), str(target_data.get("name") or "-"))
    with summary_cols[1]:
        st.metric(_t("Missing rows"), int(target_data.get("missing_count") or 0))
    with summary_cols[2]:
        st.metric(_t("Unique values"), int(target_data.get("unique_count") or 0))

    stats = target_data.get("stats", {})
    if isinstance(stats, dict) and stats:
        stats_rows = [{_t("Stat"): key, _t("Value"): value} for key, value in stats.items()]
        st.dataframe(pd.DataFrame(stats_rows), hide_index=True, use_container_width=True)

    top_values = target_data.get("top_values", {})
    if isinstance(top_values, dict) and top_values:
        st.write(_t("Top target values"))
        st.dataframe(
            pd.DataFrame([{_t("Value"): key, _t("Count"): value} for key, value in top_values.items()]),
            hide_index=True,
            use_container_width=True,
        )


def _flatten_target_relationships(relationships: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric_correlation_rows: list[dict[str, object]] = []
    numeric_group_rows: list[dict[str, object]] = []
    categorical_rows: list[dict[str, object]] = []

    for item in relationships.get("numeric_features", []):
        if not isinstance(item, dict):
            continue
        feature = str(item.get("feature") or "")
        if "target_correlation" in item:
            numeric_correlation_rows.append(
                {"feature": feature, "target_correlation": item.get("target_correlation")}
            )
        by_target = item.get("by_target", {})
        if isinstance(by_target, dict):
            for target_value, stats in by_target.items():
                if not isinstance(stats, dict):
                    continue
                numeric_group_rows.append(
                    {
                        "feature": feature,
                        "target_value": str(target_value),
                        "mean": stats.get("mean"),
                        "median": stats.get("median"),
                        "count": stats.get("count"),
                    }
                )

    for item in relationships.get("categorical_features", []):
        if not isinstance(item, dict):
            continue
        feature = str(item.get("feature") or "")
        distribution = item.get("target_distribution_by_value", {})
        if not isinstance(distribution, dict):
            continue
        for feature_value, target_rates in distribution.items():
            if not isinstance(target_rates, dict):
                continue
            for target_value, rate in target_rates.items():
                categorical_rows.append(
                    {
                        "feature": feature,
                        "feature_value": str(feature_value),
                        "target_value": str(target_value),
                        "share": rate,
                    }
                )

    return (
        pd.DataFrame(numeric_correlation_rows),
        pd.DataFrame(numeric_group_rows),
        pd.DataFrame(categorical_rows),
    )


def _render_target_relationships(relationships: dict[str, object]) -> None:
    numeric_correlations, numeric_groups, categorical_distribution = _flatten_target_relationships(relationships)

    if not numeric_correlations.empty:
        st.write(_t("Numeric feature relationships"))
        st.dataframe(numeric_correlations, hide_index=True, use_container_width=True)

    if not numeric_groups.empty:
        st.write(_t("Numeric feature distribution by target"))
        st.dataframe(numeric_groups, hide_index=True, use_container_width=True)

    if not categorical_distribution.empty:
        st.write(_t("Categorical feature target distribution"))
        st.dataframe(categorical_distribution, hide_index=True, use_container_width=True)

    if numeric_correlations.empty and numeric_groups.empty and categorical_distribution.empty:
        st.caption(_t("No target relationship summary available."))


def _render_metrics(metrics: dict[str, object], priority_metric: str | None = None) -> None:
    test_rows: list[dict[str, object]] = []
    train_rows: list[dict[str, object]] = []
    for metric_name, value in metrics.items():
        phase = "train" if str(metric_name).startswith("train_") else "test"
        label = str(metric_name).removeprefix("train_")
        row = {"metric": label, "value": value}
        if phase == "train":
            train_rows.append(row)
        else:
            test_rows.append(row)

    if priority_metric:
        resolved_value = metrics.get(priority_metric)
        priority_cols = st.columns(2)
        with priority_cols[0]:
            st.metric(_t("Priority metric"), priority_metric)
        with priority_cols[1]:
            st.metric(_t("Priority value"), "-" if resolved_value is None else f"{float(resolved_value):.4f}")

    metric_cols = st.columns(2)
    with metric_cols[0]:
        st.write(_t("Test metrics"))
        if test_rows:
            st.dataframe(pd.DataFrame(test_rows), hide_index=True, use_container_width=True)
        else:
            st.caption(_t("No test metrics."))
    with metric_cols[1]:
        st.write(_t("Train metrics"))
        if train_rows:
            st.dataframe(pd.DataFrame(train_rows), hide_index=True, use_container_width=True)
        else:
            st.caption(_t("No train metrics."))


def _preview_to_frame(preview: dict[str, object]) -> pd.DataFrame:
    rows = preview.get("rows", [])
    columns = [str(column) for column in preview.get("columns", [])]
    if not isinstance(rows, list):
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    ordered_columns = [column for column in columns if column in frame.columns]
    remaining_columns = [column for column in frame.columns if column not in ordered_columns]
    return frame[ordered_columns + remaining_columns] if not frame.empty or ordered_columns else frame


def _shape_label(snapshot: dict[str, object]) -> str:
    rows = snapshot.get("rows")
    columns = snapshot.get("columns")
    if rows is None and columns is None:
        return "-"
    if rows is None:
        return f"? x {columns}"
    if columns is None:
        return f"{rows} x ?"
    return f"{rows} x {columns}"


def _render_data_flow(trace_data: dict[str, object]) -> None:
    snapshots = [item for item in trace_data.get("snapshots", []) if isinstance(item, dict)]
    if not snapshots:
        st.caption(_t("No data flow trace available."))
        return

    target_name = str(trace_data.get("target") or "run")
    for start in range(0, len(snapshots), 4):
        chunk = snapshots[start : start + 4]
        columns = st.columns(len(chunk))
        for index, snapshot in enumerate(chunk, start=start + 1):
            with columns[index - start - 1]:
                st.caption(_t("{index}. {label}", index=index, label=snapshot.get("label") or snapshot.get("step")))
                st.metric(_t("Shape"), _shape_label(snapshot))
                partition = str(snapshot.get("partition") or "full")
                stage = str(snapshot.get("stage") or "-")
                delta = snapshot.get("rows_delta")
                delta_text = "-" if delta is None else f"{int(delta):+d}"
                st.caption(_t("{stage} | {partition} | delta {delta_text}", stage=stage, partition=partition, delta_text=delta_text))

    summary_rows = []
    for index, snapshot in enumerate(snapshots, start=1):
        summary_rows.append(
            {
                "index": index,
                "step": snapshot.get("step"),
                "label": snapshot.get("label"),
                "stage": snapshot.get("stage"),
                "partition": snapshot.get("partition"),
                "kind": snapshot.get("data_kind"),
                "shape": _shape_label(snapshot),
                "rows_delta": snapshot.get("rows_delta"),
                "added": len(snapshot.get("columns_added", [])),
                "removed": len(snapshot.get("columns_removed", [])),
            }
        )
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

    options = [
        f"{index}. {snapshot.get('label') or snapshot.get('step')} [{snapshot.get('partition') or 'full'}]"
        for index, snapshot in enumerate(snapshots, start=1)
    ]
    selected = st.selectbox(
        _t("Inspect data flow step"),
        options,
        key=f"data_flow_step_{target_name}",
    )
    selected_index = options.index(selected)
    selected_snapshot = snapshots[selected_index]

    metric_cols = st.columns(5)
    with metric_cols[0]:
        st.metric(_t("Stage"), str(selected_snapshot.get("stage") or "-"))
    with metric_cols[1]:
        st.metric(_t("Partition"), str(selected_snapshot.get("partition") or "-"))
    with metric_cols[2]:
        st.metric(_t("Kind"), str(selected_snapshot.get("data_kind") or "-"))
    with metric_cols[3]:
        st.metric(_t("Shape"), _shape_label(selected_snapshot))
    with metric_cols[4]:
        memory = selected_snapshot.get("memory_mb")
        memory_text = "-" if memory is None else f"{float(memory):.4f} MB"
        st.metric(_t("Memory"), memory_text)

    delta_cols = st.columns(3)
    with delta_cols[0]:
        delta = selected_snapshot.get("rows_delta")
        st.metric(_t("Rows delta"), "-" if delta is None else f"{int(delta):+d}")
    with delta_cols[1]:
        st.metric(_t("Columns added"), len(selected_snapshot.get("columns_added", [])))
    with delta_cols[2]:
        st.metric(_t("Columns removed"), len(selected_snapshot.get("columns_removed", [])))

    metadata = selected_snapshot.get("metadata", {})
    if isinstance(metadata, dict) and metadata:
        st.write(_t("Metadata"))
        st.json(metadata, expanded=True)

    detail_cols = st.columns(2)
    with detail_cols[0]:
        added = [str(item) for item in selected_snapshot.get("columns_added", []) if str(item).strip()]
        _render_text_items("Columns added", added, "No columns added.")
    with detail_cols[1]:
        removed = [str(item) for item in selected_snapshot.get("columns_removed", []) if str(item).strip()]
        _render_text_items("Columns removed", removed, "No columns removed.")

    preview = selected_snapshot.get("preview")
    if isinstance(preview, dict):
        st.write(_t("Preview"))
        preview_frame = _preview_to_frame(preview)
        st.dataframe(preview_frame, use_container_width=True)
        if preview.get("truncated"):
            st.caption(_t("Preview truncated to the first rows."))


def _render_validation_summary(
    preflight: dict[str, object],
    postrun: dict[str, object],
    recommendations: dict[str, object],
    priority_metric: str,
) -> None:
    summary_cols = st.columns(4)
    with summary_cols[0]:
        st.metric(_t("Requested metric"), priority_metric)
    with summary_cols[1]:
        st.metric(_t("Preflight"), _t("OK") if preflight.get("ok_to_run") else _t("Blocked"))
    with summary_cols[2]:
        st.metric(_t("Postrun"), _t("OK") if postrun.get("ok") else _t("Check issues"))
    with summary_cols[3]:
        st.metric(_t("Trainer"), str(postrun.get("trainer_name") or "-"))

    preflight_detail_cols = st.columns(3)
    with preflight_detail_cols[0]:
        st.metric(_t("Feature count"), int(preflight.get("feature_count") or 0))
    with preflight_detail_cols[1]:
        st.metric(_t("Dropped target rows"), int(preflight.get("dropped_target_rows") or 0))
    with preflight_detail_cols[2]:
        st.metric(_t("Report mode"), str(postrun.get("report_mode") or "-"))

    generalization_gap = postrun.get("generalization_gap", {})
    if isinstance(generalization_gap, dict) and generalization_gap:
        gap_rows = [{"metric": key, "value": value} for key, value in generalization_gap.items()]
        st.write(_t("Generalization gap"))
        st.dataframe(pd.DataFrame(gap_rows), hide_index=True, use_container_width=True)

    class_balance = preflight.get("class_balance", {})
    if isinstance(class_balance, dict) and class_balance:
        st.write(_t("Class balance"))
        st.dataframe(
            pd.DataFrame([{"class": key, "share": value} for key, value in class_balance.items()]),
            hide_index=True,
            use_container_width=True,
        )

    recommended_exclusions = preflight.get("recommended_excluded_columns", [])
    detected_leakage = preflight.get("detected_leakage_columns", [])
    detail_cols = st.columns(2)
    with detail_cols[0]:
        _render_text_items("Recommended exclusions", list(recommended_exclusions), "No extra exclusions suggested.")
    with detail_cols[1]:
        _render_text_items("Potential leakage columns", list(detected_leakage), "No leakage columns detected.")

    issue_cols = st.columns(2)
    with issue_cols[0]:
        st.write(_t("Preflight issues"))
        _render_issue_table(list(preflight.get("issues", [])))
    with issue_cols[1]:
        st.write(_t("Postrun issues"))
        _render_issue_table(list(postrun.get("issues", [])))

    recommendation_cols = st.columns(2)
    with recommendation_cols[0]:
        _render_text_items("Recommendation summary", list(recommendations.get("summary", [])), "No summary available.")
    with recommendation_cols[1]:
        _render_text_items("Next steps", list(recommendations.get("next_steps", [])), "No next steps available.")


def _render_issue_table(issues: list[dict[str, object]]) -> None:
    if not issues:
        st.caption(_t("No issues surfaced."))
        return
    st.dataframe(pd.DataFrame(issues), use_container_width=True)


def _render_integer_input(label: str, state_key: str, min_value: int | None = None) -> int:
    text_key = f"{state_key}_text"
    raw_value = st.text_input(label, key=text_key)
    current_value = int(st.session_state.get(state_key, 0))
    candidate = raw_value.strip()
    try:
        parsed_value = int(candidate)
    except ValueError:
        st.caption(_t("Enter a whole number. Using {current_value} until corrected.", current_value=current_value))
        return current_value

    if min_value is not None and parsed_value < min_value:
        st.caption(_t("Enter a value greater than or equal to {min_value}. Using {current_value} until corrected.", min_value=min_value, current_value=current_value))
        return current_value

    st.session_state[state_key] = parsed_value
    return parsed_value


def main() -> None:
    _apply_design_system()
    _render_language_switcher()
    _render_hero()
    _render_help_center()
    settings = _configure_llm_settings(load_settings())
    storage = RunStorage(settings.runs_dir)

    st.subheader(_t("1. Upload data"))
    _section_caption(_t("Start with one CSV file or a demo dataset. The app saves each run under the configured runs directory."))

    demo_csvs = sorted(Path(p).name for p in PROJECT_ROOT.glob("data/*.csv") if p.is_file())

    upload_label = _t("Upload CSV")
    demo_label_to_name = {_t("Demo: {name}", name=name): name for name in demo_csvs}
    data_source = st.radio(
        _t("Data source"),
        [upload_label, *list(demo_label_to_name.keys())],
        horizontal=True,
        index=0,
    )

    df: pd.DataFrame
    if data_source in demo_label_to_name:
        demo_name = demo_label_to_name[data_source]
        demo_path = PROJECT_ROOT / "data" / demo_name
        df = read_csv(demo_path)
        _render_step_status(
            _t("Loaded demo dataset `{demo_name}` with {rows} rows and {columns} columns.", demo_name=demo_name, rows=len(df), columns=len(df.columns)),
            _t("Choose the column you want to predict."),
            level="success",
        )
    else:
        uploaded_file = st.file_uploader(_t("Upload CSV"), type=["csv"], label_visibility="collapsed")
        if uploaded_file is None:
            _render_step_status(
                _t("No dataset has been loaded yet."),
                _t("Upload a CSV or pick a demo dataset to unlock the next step."),
            )
            st.markdown(
                f"""
                <div class="lab-empty">
                    {_t("Drop a CSV here to unlock schema inspection, missingness checks, training controls, and exportable run artifacts.")}
                </div>
                """,
                unsafe_allow_html=True,
            )
            return
        df = read_csv(uploaded_file)
        _render_step_status(
            _t("Loaded `{file_name}` with {rows} rows and {columns} columns.", file_name=uploaded_file.name, rows=len(df), columns=len(df.columns)),
            _t("Choose the column you want to predict."),
            level="success",
        )
    current_dataset_fingerprint = _dataset_fingerprint(df)
    columns = list(df.columns)
    _initialize_experiment_state(current_dataset_fingerprint, columns)
    _consume_pending_plan_suggestion(columns)

    st.subheader(_t("2. Choose what to predict"))
    _section_caption(_t("Pick the column you want the app to predict. The app can infer the task type automatically."))
    setup_cols = st.columns([1.5, 1.0])
    with setup_cols[0]:
        target_columns = st.multiselect(
            _t("Target variables"),
            columns,
            key="target_columns",
            help=_t("Select one or more targets. Multi-target runs train one model per target."),
        )
    with setup_cols[1]:
        task_type_choice = st.radio(
            _t("Task type"),
            ["auto", "classification", "regression"],
            horizontal=True,
            key="task_type_choice",
            format_func=lambda value: _t(value),
        )

    if not target_columns:
        _render_step_status(
            _t("Your dataset is ready, but no prediction target has been selected yet."),
            _t("Select at least one target column to continue to the checks step."),
            level="warning",
        )
        return

    selected_targets = ", ".join(str(target) for target in target_columns)
    _render_step_status(
        _t("Selected target columns: {selected_targets}.", selected_targets=selected_targets)
        if len(target_columns) > 1
        else _t("Selected target column: {selected_targets}.", selected_targets=selected_targets),
        _t("Review the data checks before starting training."),
        level="success",
    )

    exclude_options = [column for column in columns if column not in target_columns]
    with st.expander(_t("Advanced experiment settings"), expanded=False):
        st.caption(_t("Most first runs can keep the defaults here. Open this only if you want more control."))
        excluded_columns = st.multiselect(
            _t("Exclude columns from EDA and training features"),
            exclude_options,
            key="excluded_columns",
            help=_t("Excluded columns are removed before EDA and are not used as model features."),
        )
        top_advanced_cols = st.columns(2)
        with top_advanced_cols[0]:
            time_budget = _render_integer_input(_t("Training time budget seconds"), "time_budget", min_value=5)
        with top_advanced_cols[1]:
            priority_metric_choice = st.selectbox(
                _t("Priority metric"),
                ["auto", "accuracy", "f1_weighted", "precision_weighted", "recall_weighted", "roc_auc", "rmse", "mae", "r2"],
                key="priority_metric_choice",
                help=_t("This is the score the trainer treats as most important when choosing the best baseline."),
            )

        config_cols = st.columns(3)
        with config_cols[0]:
            test_size = st.slider(_t("Test size"), min_value=0.1, max_value=0.5, step=0.05, key="test_size")
        with config_cols[1]:
            high_missing_threshold = st.slider(
                _t("Drop feature when missing rate is above"),
                min_value=0.5,
                max_value=1.0,
                step=0.05,
                key="high_missing_threshold",
            )
        with config_cols[2]:
            random_state = _render_integer_input(_t("Random state"), "random_state")

    analysis_columns = [column for column in columns if column not in excluded_columns]
    base_analysis_df = df[analysis_columns].copy()
    primary_target = target_columns[0]
    applied_manual_cleaning_payload = st.session_state.get("_manual_cleaning_plan")
    manual_cleaning_plan = None
    analysis_manual_cleaning_log: list[dict[str, object]] = []
    analysis_manual_cleaning_impact: dict[str, object] = {}
    analysis_df = base_analysis_df
    if isinstance(applied_manual_cleaning_payload, dict):
        manual_cleaning_plan = validate_manual_cleaning_plan(
            applied_manual_cleaning_payload,
            base_analysis_df,
            primary_target,
            protected_columns=target_columns,
        )
        if any(rule.enabled for rule in manual_cleaning_plan.rules) and manual_cleaning_plan.effect_stage == "pre_eda":
            analysis_df, analysis_manual_cleaning_log, analysis_manual_cleaning_impact = apply_manual_cleaning_plan(
                base_analysis_df,
                manual_cleaning_plan,
                primary_target,
                protected_columns=target_columns,
            )

    candidate_feature_columns = [column for column in analysis_df.columns if column not in target_columns]
    task_types = _target_task_types(analysis_df, target_columns, task_type_choice)
    if len(target_columns) > 1:
        st.caption(_t("Multi-target mode trains and stores one independent run per target. Other selected targets are excluded from each model's feature set."))

    eda_summary = generate_eda_summary(analysis_df, target=primary_target)
    st.subheader(_t("3. Check data before training"))
    _section_caption(_t("Use the brief, validation checks, and data summary to catch issues before you spend time training."))
    st.write(_t("Planning help"))
    _section_caption(_t("This optional brief lets you describe your goal in plain language so the app can suggest a sensible first setup."))
    planner_brief = st.text_area(
        _t("Planning brief"),
        key="planner_brief",
        placeholder=_t("Example: predict churn, treat customer_id as reference only, and keep this as a quick first pass."),
        help=_t("Optional natural-language brief used to suggest targets, task type, exclusions, and a priority metric."),
    )
    plan_suggestion = _get_planner_suggestion(
        df=analysis_df,
        eda_summary=eda_summary,
        settings=settings,
        user_brief=planner_brief,
        dataset_fingerprint=current_dataset_fingerprint,
    )
    plan_data = artifact_to_dict(plan_suggestion)
    with st.expander(_t("Planner suggestion"), expanded=bool(planner_brief.strip())):
        _render_planner_suggestion(plan_data)
        if st.button(_t("Apply planner suggestions")):
            _queue_plan_suggestion(plan_data, columns)
            st.rerun()

    feature_plan = None
    with st.expander(_t("Advanced adjustments"), expanded=False):
        st.caption(_t("Most first runs can skip this section. Open it only if you want to clean rows or columns manually, or add extra local feature transformations."))
        st.write(_t("Manual cleaning rules"))
        _section_caption(_t("If you already know some rows or columns should be filtered out, draft the rules here before training."))
        manual_cleaning_brief = st.text_area(
            _t("Cleaning rules brief"),
            key="manual_cleaning_brief",
            placeholder=_t("Example: drop customer_id and keep rows where monthly_spend > 20 and churn equals 1."),
            help=_t("Natural-language rules are converted into a structured draft. Nothing is applied until you confirm."),
        )
        manual_rule_controls = st.columns(3)
        with manual_rule_controls[0]:
            if st.button(_t("Generate cleaning rules")):
                suggested_manual_plan = _get_manual_cleaning_plan(
                    df=base_analysis_df,
                    target=primary_target,
                    settings=settings,
                    user_brief=manual_cleaning_brief,
                )
                st.session_state["_manual_cleaning_override_plan"] = _clone_json_data(suggested_manual_plan)
                st.rerun()
        with manual_rule_controls[1]:
            if st.button(_t("Start with blank rule")):
                st.session_state["_manual_cleaning_override_plan"] = _blank_manual_cleaning_plan(manual_cleaning_brief)
                st.rerun()
        with manual_rule_controls[2]:
            if st.session_state.get("_manual_cleaning_plan") and st.button(_t("Clear applied manual rules")):
                st.session_state["_manual_cleaning_plan"] = None
                st.session_state["_manual_cleaning_override_plan"] = None
                st.rerun()

        if st.session_state.get("_manual_cleaning_override_plan") is None and manual_cleaning_plan is not None:
            st.session_state["_manual_cleaning_override_plan"] = _clone_json_data(manual_cleaning_plan)

        draft_manual_plan = st.session_state.get("_manual_cleaning_override_plan")
        if isinstance(draft_manual_plan, dict):
            draft_manual_plan["user_brief"] = manual_cleaning_brief
            edited_manual_plan = _render_manual_cleaning_editor(
                draft_manual_plan,
                available_columns=list(base_analysis_df.columns),
            )
            validated_manual_preview = validate_manual_cleaning_plan(
                edited_manual_plan,
                base_analysis_df,
                primary_target,
                protected_columns=target_columns,
            )
            preview_df, preview_log, preview_impact = apply_manual_cleaning_plan(
                base_analysis_df,
                validated_manual_preview,
                primary_target,
                protected_columns=target_columns,
            )
            preview_plan_data = artifact_to_dict(validated_manual_preview)
            summary_cols = st.columns(5)
            with summary_cols[0]:
                st.metric(_t("Planner"), str(preview_plan_data.get("planner_name") or "manual"))
            with summary_cols[1]:
                st.metric(_t("Effect stage"), _manual_cleaning_effect_stage_label(validated_manual_preview.effect_stage))
            with summary_cols[2]:
                st.metric(_t("Accepted rules"), len(preview_plan_data.get("rules", [])))
            with summary_cols[3]:
                st.metric(_t("Rejected rules"), len(preview_plan_data.get("rejected_rules", [])))
            with summary_cols[4]:
                st.metric(_t("Rows removed"), int(preview_impact.get("rows_removed") or 0))

            rule_rows = _manual_cleaning_rule_rows(preview_plan_data)
            if rule_rows:
                st.dataframe(pd.DataFrame(rule_rows), hide_index=True, use_container_width=True)
            else:
                st.caption(_t("No manual cleaning rules are in the current draft."))

            impact_cols = st.columns(3)
            with impact_cols[0]:
                st.metric(_t("Columns removed"), len(preview_impact.get("columns_removed", [])))
            with impact_cols[1]:
                st.metric(_t("Rows after"), int(preview_impact.get("rows_after") or len(base_analysis_df)))
            with impact_cols[2]:
                st.metric(_t("Columns after"), int(preview_impact.get("columns_after") or len(base_analysis_df.columns)))

            detail_cols = st.columns(2)
            with detail_cols[0]:
                _render_text_items("Notes", preview_plan_data.get("notes", []), "No notes.")
            with detail_cols[1]:
                _render_text_items("Rejected rules", preview_plan_data.get("rejected_rules", []), "No rejected rules.")

            with st.expander(_t("Manual cleaning preview impact"), expanded=False):
                st.json(preview_impact, expanded=True)
                st.dataframe(preview_df.head(20), use_container_width=True)
                if preview_log:
                    st.write(_t("Planned cleaning log"))
                    st.json(preview_log, expanded=True)

            apply_cols = st.columns(2)
            with apply_cols[0]:
                if st.button(_t("Apply manual cleaning rules"), type="primary"):
                    st.session_state["_manual_cleaning_plan"] = preview_plan_data
                    st.session_state["_manual_cleaning_override_plan"] = _clone_json_data(preview_plan_data)
                    st.rerun()
            with apply_cols[1]:
                if st.session_state.get("_manual_cleaning_plan"):
                    st.caption(_t("Applied rules remain active until you clear them or apply a different draft."))
        else:
            st.caption(_t("Generate rules from a brief or start with a blank rule to configure manual cleaning."))

        apply_feature_engineering = st.checkbox(
            _t("Apply local whitelist feature engineering"),
            key="apply_feature_engineering",
            help=_t("LLM can propose a structured plan, but only local whitelisted transformations are executed inside the training pipeline."),
        )
        if apply_feature_engineering:
            feature_plan_override = st.session_state.get("_feature_engineering_override_plan")
            if isinstance(feature_plan_override, dict) and feature_plan_override.get("feature_engineering_operations"):
                feature_plan = feature_plan_override
            else:
                feature_plan_df = analysis_df[candidate_feature_columns + [primary_target]].copy()
                feature_plan = _get_feature_engineering_plan(
                    df=feature_plan_df,
                    target=primary_target,
                    settings=settings,
                    user_brief=planner_brief,
                )
            with st.expander(_t("Feature engineering plan"), expanded=True):
                _render_feature_engineering_plan(artifact_to_dict(feature_plan))

    current_experiment_signature = _experiment_signature(
        dataset_fingerprint=current_dataset_fingerprint,
        target_columns=target_columns,
        task_type_choice=task_type_choice,
        time_budget=int(time_budget),
        excluded_columns=excluded_columns,
        test_size=float(test_size),
        high_missing_threshold=float(high_missing_threshold),
        random_state=int(random_state),
        priority_metric_choice=priority_metric_choice,
        planner_brief=planner_brief,
        feature_plan=feature_plan,
        manual_cleaning_plan=manual_cleaning_plan,
    )

    priority_metrics = {
        target: resolve_priority_metric(_target_task_types(analysis_df, [target], task_type_choice)[target], priority_metric_choice)
        for target in target_columns
    }
    preflight_by_target: dict[str, object] = {}
    for target in target_columns:
        preflight_input = analysis_df[[column for column in analysis_df.columns if column not in target_columns or column == target]].copy()
        if manual_cleaning_plan is not None and any(rule.enabled for rule in manual_cleaning_plan.rules) and manual_cleaning_plan.effect_stage == "pre_training":
            preflight_input, _, _ = apply_manual_cleaning_plan(
                preflight_input,
                manual_cleaning_plan,
                target,
                protected_columns=target_columns,
            )
        preflight_by_target[target] = validate_preflight(
            df=preflight_input,
            target=target,
            task_type=task_types[target],
            excluded_columns=[],
            priority_metric=priority_metric_choice,
            high_missing_threshold=float(high_missing_threshold),
        )
    failing_targets = [target for target, validation in preflight_by_target.items() if not validation.ok_to_run]
    if failing_targets:
        _render_step_status(
            _t("The app found blocking issues in the current setup."),
            _t("Fix the checks for: {failing_targets} before starting training.", failing_targets=", ".join(failing_targets)),
            level="warning",
        )
    else:
        _render_step_status(
            _t("The dataset and target selection passed the current checks."),
            _t("You can start training after this review, or adjust the setup first."),
            level="success",
        )
    st.write(_t("Preflight validation"))
    _section_caption(_t("This check looks for blocking issues before training, such as missing target values or no usable feature columns."))
    with st.expander(_t("Preflight validation"), expanded=True):
        for target in target_columns:
            validation = artifact_to_dict(preflight_by_target[target])
            st.write(f"{target} ({_t(task_types[target])})")
            st.caption(_t("Resolved priority metric: {priority_metric}", priority_metric=priority_metrics[target]))
            _render_issue_table(validation.get("issues", []))

    st.write(_t("Data preview"))
    _section_caption(_t("First 50 rows are shown for a quick sanity check before training."))
    if manual_cleaning_plan is not None and any(rule.enabled for rule in manual_cleaning_plan.rules) and manual_cleaning_plan.effect_stage == "pre_training":
        st.info(_t("Manual cleaning rules are set to apply only before training. The data preview and EDA below still show the pre-cleaning analysis subset."))
    st.dataframe(analysis_df.head(50), use_container_width=True)

    st.write(_t("EDA summary"))
    _section_caption(_t("EDA means a quick health check for the dataset: shape, duplicates, missing values, correlations, and target behavior."))
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric(_t("Rows"), eda_summary["shape"]["rows"])
    with metric_cols[1]:
        st.metric(_t("Columns"), eda_summary["shape"]["columns"])
    with metric_cols[2]:
        st.metric(_t("Duplicate rows"), eda_summary["duplicate_rows"])
    with metric_cols[3]:
        st.metric(_t("Rows with missing"), eda_summary["missingness"]["rows_with_any_missing"])

    st.write(_t("Column profile"))
    st.dataframe(pd.DataFrame(eda_summary["columns"]).T, use_container_width=True)

    if primary_target in analysis_df.columns and eda_summary.get("target"):
        label = _t("Primary target profile") if len(target_columns) > 1 else _t("Target profile")
        st.write(label)
        _render_target_profile(eda_summary["target"])
    if len(target_columns) > 1:
        st.write(_t("Target task types"))
        st.dataframe(
            pd.DataFrame(
                [{_t("Target"): target, _t("Task type"): _t(task_type)} for target, task_type in task_types.items()]
            ),
            use_container_width=True,
        )

    eda_tabs = st.tabs([_t("Missingness"), _t("Correlations"), _t("Target relationships"), _t("Quality warnings")])
    with eda_tabs[0]:
        top_missing = eda_summary["missingness"]["top_missing_columns"]
        if top_missing:
            st.dataframe(
                pd.DataFrame(
                    [{_t("Column"): column, _t("Missing rate"): rate} for column, rate in top_missing.items()]
                ),
                use_container_width=True,
            )
        correlated_missing = eda_summary["missingness"]["correlated_missing_pairs"]
        if correlated_missing:
            st.write(_t("Correlated missingness pairs"))
            st.dataframe(pd.DataFrame(correlated_missing), use_container_width=True)
    with eda_tabs[1]:
        top_pairs = eda_summary["correlations"]["top_numeric_pairs"]
        target_corr = eda_summary["correlations"]["target_numeric_correlations"]
        if target_corr:
            st.write(_t("Numeric correlations with target"))
            st.dataframe(pd.DataFrame(target_corr), use_container_width=True)
        if top_pairs:
            st.write(_t("Strong numeric feature correlations"))
            st.dataframe(pd.DataFrame(top_pairs), use_container_width=True)
    with eda_tabs[2]:
        _render_target_relationships(eda_summary.get("target_relationships", {}))
    with eda_tabs[3]:
        for warning in eda_summary["quality_warnings"]:
            st.warning(warning)

    st.subheader(_t("4. Start training"))
    _section_caption(_t("Launch the local baseline after you have reviewed the target, checks, and data summary."))
    if failing_targets:
        _render_step_status(
            _t("Training is blocked by validation issues."),
            _t("Resolve the flagged issues for: {failing_targets}.", failing_targets=", ".join(failing_targets)),
            level="warning",
        )
    else:
        _render_step_status(
            _t("The run is ready to start."),
            _t("Click Run training to build the local baseline and unlock the results step."),
            level="success",
        )
    results: list[dict[str, object]] = []
    if st.session_state.get("_latest_results_signature") == current_experiment_signature:
        results = list(st.session_state.get("_latest_results", []))

    if st.button(_t("Run training"), type="primary"):
        results = []
        with st.spinner(_t("Cleaning data and training model locally...")):
            if failing_targets:
                logger.error("Blocking preflight issues targets=%s", failing_targets)
                st.error(_t("Resolve blocking preflight issues before training: {failing_targets}", failing_targets=", ".join(failing_targets)))
                return

            feature_columns = candidate_feature_columns
            if not feature_columns:
                logger.error("No feature columns remain after exclusions")
                st.error(_t("No feature columns remain after excluding selected target and ignored columns."))
                return

            logger.info("Starting training pipeline targets=%s task_types=%s", target_columns, task_types)
            for target in target_columns:
                logger.info("Training target=%s task_type=%s", target, task_types[target])
                task_type = task_types[target]
                priority_metric = priority_metrics[target]
                preflight_validation = preflight_by_target[target]
                tracker = DataFlowTracker(target=target)
                tracker.snapshot_dataframe(
                    "raw_dataset",
                    "Raw dataset",
                    "intake",
                    df,
                    preview=True,
                    metadata={"source_columns": list(df.columns)},
                )
                tracker.snapshot_dataframe(
                    "analysis_subset",
                    "Analysis subset",
                    "intake",
                    base_analysis_df,
                    metadata={"excluded_columns": excluded_columns},
                )
                if manual_cleaning_plan is not None and any(rule.enabled for rule in manual_cleaning_plan.rules) and manual_cleaning_plan.effect_stage == "pre_eda":
                    tracker.snapshot_dataframe(
                        "after_manual_cleaning_pre_eda",
                        "After manual cleaning (EDA + training)",
                        "intake",
                        analysis_df,
                        metadata=analysis_manual_cleaning_impact,
                    )

                target_df = analysis_df[feature_columns + [target]].copy()
                tracker.snapshot_dataframe(
                    "target_dataset",
                    "Target dataset",
                    "intake",
                    target_df,
                    preview=True,
                    metadata={"target": target, "feature_columns": feature_columns},
                )
                training_input_df = target_df
                manual_cleaning_log = list(analysis_manual_cleaning_log)
                manual_cleaning_impact = dict(analysis_manual_cleaning_impact)
                if manual_cleaning_plan is not None and any(rule.enabled for rule in manual_cleaning_plan.rules) and manual_cleaning_plan.effect_stage == "pre_training":
                    target_manual_plan = validate_manual_cleaning_plan(
                        manual_cleaning_plan,
                        target_df,
                        target,
                        protected_columns=target_columns,
                    )
                    training_input_df, manual_cleaning_log, manual_cleaning_impact = apply_manual_cleaning_plan(
                        target_df,
                        target_manual_plan,
                        target,
                        protected_columns=target_columns,
                    )
                    tracker.snapshot_dataframe(
                        "after_manual_cleaning_pre_training",
                        "After manual cleaning (training only)",
                        "intake",
                        training_input_df,
                        metadata=manual_cleaning_impact,
                    )

                target_eda_summary = generate_eda_summary(target_df, target=target)
                config = CleanConfig(
                    target=target,
                    task_type=task_type,
                    test_size=float(test_size),
                    random_state=int(random_state),
                    high_missing_threshold=float(high_missing_threshold),
                    feature_engineering_operations=_feature_plan_operations(feature_plan) if feature_plan else None,
                )
                cleaned = clean_and_split(training_input_df, config, tracker=tracker)
                if manual_cleaning_log:
                    cleaned.cleaning_log = manual_cleaning_log + cleaned.cleaning_log
                trained = train_model(cleaned, time_budget=int(time_budget), metric_preference=priority_metric, tracker=tracker)
                metrics, prediction_sample = evaluate_model(trained.model, cleaned, task_type=task_type, tracker=tracker)
                tracker.snapshot_artifact(
                    "metrics_summary",
                    "Metrics summary",
                    "evaluation",
                    metadata={"metrics": metrics, "priority_metric": priority_metric},
                )
                planned_report_mode = "llm" if settings.llm_enabled else "rule_based"
                postrun_validation = validate_postrun(
                    metrics=metrics,
                    prediction_sample=prediction_sample,
                    task_type=task_type,
                    priority_metric=priority_metric,
                    trainer_name=trained.trainer_name,
                    optimization_metric_used=trained.optimization_metric_used,
                    feature_importance=trained.feature_importance,
                    report_mode=planned_report_mode,
                )
                recommendations = build_recommendations(preflight_validation, postrun_validation)

                report, report_mode = generate_report_result(
                    eda_summary=target_eda_summary,
                    cleaning_log=cleaned.cleaning_log,
                    metrics=metrics,
                    feature_importance=trained.feature_importance,
                    settings=settings,
                    plan_suggestion=plan_suggestion,
                    preflight_validation=preflight_validation,
                    postrun_validation=postrun_validation,
                    recommendations=recommendations,
                )
                postrun_validation = validate_postrun(
                    metrics=metrics,
                    prediction_sample=prediction_sample,
                    task_type=task_type,
                    priority_metric=priority_metric,
                    trainer_name=trained.trainer_name,
                    optimization_metric_used=trained.optimization_metric_used,
                    feature_importance=trained.feature_importance,
                    report_mode=report_mode,
                )
                recommendations = build_recommendations(preflight_validation, postrun_validation)
                if report_mode != planned_report_mode:
                    report, report_mode = generate_report_result(
                        eda_summary=target_eda_summary,
                        cleaning_log=cleaned.cleaning_log,
                        metrics=metrics,
                        feature_importance=trained.feature_importance,
                        settings=settings,
                        plan_suggestion=plan_suggestion,
                        preflight_validation=preflight_validation,
                        postrun_validation=postrun_validation,
                        recommendations=recommendations,
                    )

                run = storage.create_run(
                    config={
                        "target": target,
                        "target_columns": target_columns,
                        "task_type": task_type,
                        "task_type_choice": task_type_choice,
                        "excluded_columns": excluded_columns,
                        "test_size": test_size,
                        "high_missing_threshold": high_missing_threshold,
                        "random_state": random_state,
                        "time_budget": time_budget,
                        "priority_metric": priority_metric,
                        "trainer": trained.trainer_name,
                        "planner_name": plan_suggestion.planner_name,
                        "feature_engineering_enabled": bool(feature_plan),
                        "manual_cleaning_enabled": bool(manual_cleaning_plan and any(rule.enabled for rule in manual_cleaning_plan.rules)),
                        "manual_cleaning_effect_stage": manual_cleaning_plan.effect_stage if manual_cleaning_plan else None,
                        "dataset_fingerprint": current_dataset_fingerprint,
                    }
                )
                storage.save_json(run, "plan.json", artifact_to_dict(plan_suggestion))
                if feature_plan:
                    storage.save_json(run, "feature_engineering_plan.json", artifact_to_dict(feature_plan))
                if manual_cleaning_plan:
                    storage.save_json(run, "manual_cleaning_plan.json", artifact_to_dict(manual_cleaning_plan))
                storage.save_json(run, "eda_summary.json", target_eda_summary)
                storage.save_json(run, "cleaning_log.json", cleaned.cleaning_log)
                storage.save_json(run, "metrics.json", metrics)
                data_flow_payload = artifact_to_dict(tracker.to_trace())
                storage.save_json(run, "data_flow.json", data_flow_payload)
                storage.save_json(run, "feature_importance.json", trained.feature_importance)
                storage.save_json(run, "validation_pre.json", artifact_to_dict(preflight_validation))
                storage.save_json(run, "validation_post.json", artifact_to_dict(postrun_validation))
                storage.save_json(run, "recommendations.json", artifact_to_dict(recommendations))
                storage.save_json(
                    run,
                    "training_summary.json",
                    {
                        "trainer_name": trained.trainer_name,
                        "optimization_metric_used": trained.optimization_metric_used,
                        "training_notes": trained.training_notes or [],
                        "manual_cleaning_effect_stage": manual_cleaning_plan.effect_stage if manual_cleaning_plan else None,
                        "manual_cleaning_applied_rules": 0
                        if not manual_cleaning_plan
                        else sum(1 for rule in manual_cleaning_plan.rules if rule.enabled),
                    },
                )
                report_path = storage.save_text(run, "report.md", report)
                prediction_path = storage.save_predictions(run, prediction_sample)
                model_path = storage.save_model(run, trained.model)
                storage.record_run(run, metrics=metrics, status="completed")
                results.append(
                    {
                        "target": target,
                        "task_type": task_type,
                        "run": run,
                        "trained": trained,
                        "metrics": metrics,
                        "report": report,
                        "preflight_validation": preflight_validation,
                        "postrun_validation": postrun_validation,
                        "recommendations": recommendations,
                        "priority_metric": priority_metric,
                        "data_flow": data_flow_payload,
                        "report_path": report_path,
                        "prediction_path": prediction_path,
                        "model_path": model_path,
                    }
                )

        st.session_state["_latest_results_signature"] = current_experiment_signature
        st.session_state["_latest_results"] = list(results)
        completed_ids = ", ".join(str(result["run"].run_id) for result in results)
        logger.info("Training pipeline complete run_ids=%s", completed_ids)
        st.success(_t("Run completed: {completed_ids}", completed_ids=completed_ids))

    if results:
        _render_run_outputs(results)


if __name__ == "__main__":
    main()
