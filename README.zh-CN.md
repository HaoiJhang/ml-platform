# ML Platform 中文使用说明

这是一个 Streamlit 表格机器学习原型工具，覆盖了从 CSV 导入、数据概览、基础清洗、模型训练、结果评估到报告导出的完整流程。它适合做快速 baseline、数据质量初筛和小规模实验记录，不适合直接当成生产系统。

当前版本的定位比较明确：强调快速试验、单用户工作流和快速闭环。它没有做认证、多用户协作、分布式训练和生产监控。

## 1. 你能用它做什么

这个应用主要面向表格型数据集。你可以上传一个 CSV，选择一个或多个目标列，查看 EDA 摘要，决定要排除哪些字段，运行本地训练，然后下载模型、预测样本和分析报告。

如果配置了 LLM，还可以得到更自然语言的计划建议和实验报告；如果没有配置，流程仍然会完成，只是回退到本地规则版报告。

## 2. 环境准备

项目使用 `uv` 管理 Python 环境，要求 Python 版本在 `>=3.11,<3.12`。

第一次进入项目后，建议执行：

```bash
UV_CACHE_DIR=.uv-cache uv sync --python 3.11
```

如果你已经拉过代码但很久没同步过依赖，尤其是在代理环境下，先再跑一次：

```bash
UV_CACHE_DIR=.uv-cache uv sync
```

## 3. 启动方式

先运行测试确认环境正常：

```bash
UV_CACHE_DIR=.uv-cache uv run pytest
```

然后启动应用：

```bash
UV_CACHE_DIR=.uv-cache uv run streamlit run app.py
```

默认访问地址是 `http://localhost:8501`。

## 3.1 Streamlit Community Cloud 部署

在 Streamlit Community Cloud 里从 GitHub 部署时使用这些参数：

- Repository：`HaoiJhang/ml-platform`
- Branch：你要部署的分支
- Main file path：`app.py`
- Python version：`3.11`

项目根目录已经包含 `uv.lock`，Streamlit Community Cloud 会把它作为依赖文件使用。当前项目声明的 Python 版本是 `>=3.11,<3.12`，所以部署时需要在 Advanced settings 里选择 Python 3.11，不要使用 Cloud 默认版本。

为了让公网界面和本地基准样式保持一致，项目把 `streamlit` 固定在 `1.50.0`，不跟随新版上传组件布局变化。

如果希望公网版本内置兼容 LLM API 的配置，请在 Streamlit Community Cloud 的 Secrets 里填写，不要写进仓库：

```toml
LLM_API_KEY = "..."
LLM_BASE_URL = "..."
LLM_MODEL = "gpt-4o-mini"
```

默认情况下，只要勾选页面里的 `Remember LLM settings on this device`，应用就会把 LLM 配置保存到当前项目目录下的本地文件 `.ml_platform.local.json`。如果是共享机器或公网部署，建议显式设置环境变量 `ML_PLATFORM_ALLOW_LOCAL_LLM_CONFIG=0` 关闭这个能力，改用宿主环境 secrets。

## 4. 页面使用流程

应用里的主要使用顺序如下。

### 4.1 Report engine

这里用于配置可选的 LLM 能力，包括：

- `API key`
- `Base URL`
- `Model`

如果你只想跑本地 baseline，这里可以不填。没有 API key 时，应用仍然可以完成训练和评估。

默认会显示 `Remember LLM settings on this device`。勾选后，配置会保存到当前项目目录下的本地文件 `.ml_platform.local.json`；这个文件已经被 `.gitignore` 忽略，不会默认提交到仓库。如果是共享机器或公网部署，建议设置 `ML_PLATFORM_ALLOW_LOCAL_LLM_CONFIG=0` 禁用本地持久化。

### 4.2 Dataset intake

上传一个 CSV 文件。当前主要针对表格型结构化数据，推荐一行一条样本，列名清晰、目标列明确。

### 4.3 Experiment setup

这里决定本次实验的训练方式，主要包括：

- 目标列 `Target columns`
- 任务类型 `Task type`
- 要排除的列 `Exclude columns`
- 测试集比例 `Test size`
- 高缺失阈值 `Drop feature when missing rate is above`
- 随机种子 `Random state`
- 优先指标 `Priority metric`
- 时间预算 `Time budget`

如果你一次选择多个目标列，系统会为每个目标列分别训练一套模型，并分别保存结果。

### 4.4 Planning brief

这是一个自然语言输入框，用来描述你的目标，例如：

```text
predict churn, optimize recall, ignore customer_id-like fields, keep this as a quick baseline.
```

系统会基于这段描述生成建议，包括推荐目标列、任务类型、应排除字段和优先指标。你可以查看 `Planner suggestion`，然后点击 `Apply planner suggestions` 自动把建议带入表单。

### 4.5 Feature engineering plan

如果启用 `Apply local whitelist feature engineering`，应用会根据当前 brief 和数据结构生成一份结构化特征工程计划。LLM 只负责提出 JSON 计划，不会写代码，也不会直接执行任意逻辑。

当前本地白名单支持这些变换：

- 日期拆分：从日期列生成 year、month、day、dayofweek、quarter 或 is_weekend。
- 类别频次编码：把高基数类别列转换成训练集内拟合的频率特征。
- 数值分箱：基于训练集分位点做数值 binning。
- 数值字段组合：支持 ratio、difference、sum 和 product。
- 类别规则映射：把指定类别值映射到本地分组标签，未命中值进入默认组。

为了降低泄漏风险，频次编码和数值分箱会作为 sklearn pipeline 的一部分只在训练集 `fit`，测试集只 `transform`。目标列不会被允许进入特征工程计划；多目标训练时，其他目标列也不会作为特征输入。

### 4.6 Execution plan / Preflight validation

在真正训练之前，页面会先展示一些自动检查结果，例如：

- 目标列是否存在
- 排除列是否误伤目标列
- 是否存在高缺失字段
- 是否存在常量列
- 是否有疑似 ID 列
- 是否有疑似目标泄漏
- 类别不平衡或样本过少

如果存在阻断级问题，训练不会继续。

### 4.7 Data preview / EDA summary

这里会展示数据前 50 行，以及面向 baseline 的紧凑 EDA，包括：

- 行列数
- 重复行数量
- 缺失情况
- 列画像
- 目标列画像
- 特征相关性
- 目标关系摘要
- 数据质量告警

### 4.8 Run training

点击 `Run training` 后，应用会在本地执行完整流程：

1. 数据清洗与切分
2. 模型训练
3. 指标评估
4. 生成预测样本
5. 生成报告
6. 保存 artifacts

### 4.9 Artifacts / Run results

训练完成后，你可以直接在页面中：

- 下载模型文件
- 下载 Markdown 报告
- 下载预测样本 CSV

页面也会展示：

- test/train 指标
- preflight 与 postrun 校验结果
- 推荐下一步动作
- 特征重要性
- 分析报告正文

## 5. 运行产物保存在哪里

每次完成训练后，输出会保存在 `runs/` 目录下。单次 run 通常会包含这些文件：

- `config.json`
- `plan.json`
- `eda_summary.json`
- `cleaning_log.json`
- `metrics.json`
- `feature_importance.json`
- `feature_engineering_plan.json`，仅在启用特征工程时生成
- `prediction_sample.csv`
- `validation_pre.json`
- `validation_post.json`
- `recommendations.json`
- `training_summary.json`
- `model.joblib`
- `report.md`

同时，运行元数据还会写入 `runs/runs.sqlite`。

## 6. LLM 配置说明

如果设置了 `LLM_API_KEY` 环境变量，应用会自动读取；如果页面中手动输入了 API key，则优先使用页面值。

`Base URL` 支持兼容接口。如果你接的是代理或兼容服务，可以在这里填写对应地址。

如果 LLM 调用失败，应用会自动回退到本地规则版 planner 或 report，不会因为 LLM 不可用而阻断整个训练流程。

## 7. 常见问题

### 7.1 页面刷新后 API key 丢失

如果本地持久化是开启状态，勾选 `Remember LLM settings on this device` 后，配置会保存到本地 `.ml_platform.local.json`，刷新后自动回填。如果你在共享机器或公网部署里显式设置了 `ML_PLATFORM_ALLOW_LOCAL_LLM_CONFIG=0`，那页面输入的 API key 只在当前会话有效；这类环境更稳妥的做法是在 Streamlit Community Cloud 的 Secrets 中配置 `LLM_API_KEY`。

### 7.2 看到 “LLM report generation failed” 但训练已经完成

这通常表示模型训练没问题，只是 LLM 生成自然语言报告失败。应用会自动回退到 rule-based 报告，所以实验结果和 artifacts 仍然是可用的。

### 7.3 代理环境下出现 SOCKS 相关报错

如果本机设置了 `ALL_PROXY`、`HTTPS_PROXY` 等代理变量，并且代理是 `socks5://...` 这类 SOCKS 协议，旧环境可能会报 `socksio` 缺失。当前项目依赖已经补上这个包，但你需要执行一次：

```bash
UV_CACHE_DIR=.uv-cache uv sync
```

这样本地虚拟环境才会真正安装新依赖。

### 7.4 没有 API key 能不能用

可以。没有 API key 时，LLM 辅助能力会关闭，但本地训练、评估、规则版报告和 artifacts 导出都仍然可用。

## 8. 开发与测试

日常开发建议先跑测试：

```bash
UV_CACHE_DIR=.uv-cache uv run pytest
```

如果需要查看当前依赖解析结果或补装新依赖，继续使用 `uv` 即可，不需要手动维护 `requirements.txt`。

## 9. 当前边界

这个项目更像“本地实验台”，不是通用 AutoML 平台。使用时需要明确几个边界：它依赖 CSV 输入，对复杂特征工程支持有限，对大规模数据和生产推理链路没有覆盖，结果更适合做 baseline 和方向判断，不适合直接拿去做上线决策。
