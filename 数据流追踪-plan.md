# 数据流追踪 — 基于现有代码的落地计划

## 现在已经确认的事实

这份计划不再按猜测写，而是按当前代码真实路径来定。

UI 侧现在的数据路径是：

`df`（CSV 原始读取结果）  
→ `analysis_df = df[analysis_columns].copy()`（排除用户手动忽略列之后）  
→ 对每个 target 构造 `target_df = analysis_df[feature_columns + [target]].copy()`  
→ `clean_and_split(target_df, config)`  
→ `train_model(cleaned, ...)`  
→ `evaluate_model(trained.model, cleaned, ...)`

`clean_and_split()` 内部已经明确存在这些数据变化：

1. 删除 target 缺失行：`dropna(subset=[target])`
2. 删除高缺失特征列：`drop_high_missing_features`
3. 删除常量列：`drop_constant_features`
4. 构造 `X / y`
5. `train_test_split`
6. 构造 `preprocessor`，如果启用特征工程，则为 `Pipeline([feature_engineering, columns])`

`train_model()` 内部现在有两条训练路径：

1. FLAML 路径：先 `preprocessor.fit_transform(X_train)`，再训练 AutoML。
2. sklearn fallback 路径：`Pipeline([preprocessor, estimator]).fit(X_train, y_train)`。

`evaluate_model()` 会基于 `cleaned.X_test` 生成 `prediction_sample`，这个样本是目前最适合直接放进 UI 预览的数据。

所以，原 plan 里“只在 app.py 插 5 个快照点”是不够的。那样只能看到大阶段，无法看到真正发生行列变化的清洗细节，也看不到预处理后特征维度扩张。

## 目标重新收敛

目标不是做逐行 lineage，也不是保存每一步完整 DataFrame，而是做“阶段级可解释追踪”：

用户能看到每一步数据的行数、列数、内存、列增减、所属分支，以及少量中间预览；同时这些信息可以落盘为 artifact，在 UI 和 `runs/<run_id>/data_flow.json` 中复用。

## 明确不做的事

这次不追踪“某一行从输入到输出的逐条映射关系”。当前 pipeline 里没有稳定 row id，也有 one-hot 和特征工程扩展，这类逐行血缘会把实现复杂度抬得很高。

这次也不保存完整中间表。只保存统计信息和最多前 20 行预览，否则运行内存和 artifact 体积都会失控。

## 原 plan 里需要修正的点

第一个问题是 `DataFlowSnapshot.preview_df: pd.DataFrame | None` 这个设计不能直接落盘。现在 `storage.save_json()` 依赖 `json.dumps()`，而 `artifact_to_dict()` 也不会把 DataFrame 自动转成 JSON-safe 结构，这会直接失败。

第二个问题是把“清洗”当成一个单独快照过于粗糙。真正的行列变化发生在 `clean_and_split()` 内部，如果不在那里插桩，UI 只能看到“前后变了”，但看不到“为什么变了”。

第三个问题是“训练后”只记模型信息还不够。对于数据流追踪，训练阶段最有价值的是“预处理后的特征空间变成了多少列”，尤其在 one-hot 和特征工程打开时，这一步是用户最关心的数据形态变化之一。

第四个问题是目前流程并不是严格单线。`train_test_split` 之后会出现 train/test 分叉，所以追踪模型必须带上 partition 概念，而不是假设所有步骤都只有一个 DataFrame。

## 推荐的数据结构

建议新建 `src/ml_platform/data_flow.py`，但 dataclass 最好继续与现有 artifact 风格兼容，字段全部保持 JSON-safe。

```python
@dataclass(frozen=True)
class DataPreview:
    columns: list[str]
    rows: list[dict[str, object]]
    truncated: bool = False


@dataclass(frozen=True)
class DataFlowSnapshot:
    step: str                    # 例如 raw_dataset / after_target_drop / train_split
    label: str                   # UI 展示名
    stage: str                   # intake / cleaning / training / evaluation
    partition: str = "full"      # full / train / test
    data_kind: str = "dataframe" # dataframe / matrix / artifact
    rows: int | None = None
    columns: int | None = None
    memory_mb: float | None = None
    rows_delta: int | None = None
    columns_added: list[str] = field(default_factory=list)
    columns_removed: list[str] = field(default_factory=list)
    preview: DataPreview | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DataFlowTrace:
    target: str
    snapshots: list[DataFlowSnapshot] = field(default_factory=list)
```

`preview` 只存 records，不存 DataFrame 对象。`metadata` 里也只放 JSON-safe 内容，例如阈值、trainer 名称、metric、feature importance 数量、sparse density、feature engineering 操作名列表。

## Tracker 该怎么接入

最稳的方案不是让 app.py 自己猜内部状态，而是把 tracker 作为可选依赖传进各层函数：

```python
def clean_and_split(df, config, tracker: DataFlowTracker | None = None) -> CleanedData
def train_model(cleaned, time_budget=30, metric_preference="auto", tracker: DataFlowTracker | None = None) -> TrainedModel
def evaluate_model(model, cleaned, task_type, tracker: DataFlowTracker | None = None) -> tuple[dict[str, float | None], pd.DataFrame]
```

这样做的好处是：

1. 对现有调用方改动很小，默认 `None`，不会破坏测试和外部接口。
2. 真正发生数据变化的模块自己记录快照，不需要 app.py 复刻内部逻辑。
3. 文档、UI、artifact 三者共享同一份 trace，不会出现口径不一致。

## 快照点设计

### app.py 负责的外层快照

这些步骤发生在 pipeline 入口和结果出口，适合在 `app.py` 记录：

1. `raw_dataset`
   记录 `df`。这是 CSV 刚读进来的原始表，`preview` 打开。

2. `analysis_subset`
   记录 `analysis_df`。这是去掉用户手动排除列之后、EDA 和训练真正使用的输入。

3. `target_dataset`
   记录 `target_df`。多 target 模式下，每个 target 都要各自记录这一份，因为其他 target 会从特征中排除。

4. `post_training_artifact`
   记录训练结果摘要，不是 DataFrame。`metadata` 包括 `trainer_name`、`optimization_metric_used`、`training_notes`、`feature_importance_count`。

5. `prediction_sample`
   记录 `evaluate_model()` 返回的 `prediction_sample`，这个步骤应该带预览。

6. `metrics_summary`
   记录评估指标摘要，不是 DataFrame。这样 UI 不必从别处拼状态。

### clean_and_split() 内部快照

这是本次追踪里最关键的一组：

1. `cleaning_input`
   进入清洗前的 `target_df`。

2. `after_target_drop`
   删除 target 缺失行之后。这里一定要记录 `rows_delta`，并把 `rows_removed` 也写进 `metadata`。

3. `after_high_missing_drop`
   删除高缺失特征列之后。这里重点是 `columns_removed`，以及阈值 `high_missing_threshold`。

4. `after_constant_drop`
   删除常量列之后。这里同样记录 `columns_removed`。

5. `train_split`
   `X_train`，`partition="train"`。

6. `test_split`
   `X_test`，`partition="test"`。

7. `preprocessor_plan`
   这一步不是数据表，而是 artifact 型快照。记录 `numeric_features`、`categorical_features`、`standardize_numeric`、`feature_engineering_operations`。

这里不建议再单独记录 `X` 和 `y` 的中间态，因为 UI 价值不高，而且 `target_dataset` 与 `train/test split` 之间已经足够解释数据从哪里开始分叉。

### train_model() 内部快照

训练阶段最重要的是把“原始特征表”变成“模型真正看到的特征矩阵”。

1. `train_matrix`
   记录预处理后的训练矩阵形状。`data_kind="matrix"`。至少要有 `rows`、`columns`，如果是 sparse，再补 `density`。

2. `test_matrix_probe`
   不需要对整份 test 做昂贵转换，但最好记录一个轻量 probe。可以对 `cleaned.X_test.head(20)` 做 transform，只记录 shape 和 preview 是否可渲染。

3. `trained_model`
   artifact 型快照，记录 trainer 与优化指标。

实现上要注意两条训练路径：

- FLAML 路径已经显式拿到了 `preprocessor.fit_transform(cleaned.X_train)`，这里直接记录。
- sklearn fallback 路径不要为了追踪再额外 `fit` 一次。正确做法是在 `model.fit(...)` 完成后，用已经 fitted 的 `model.named_steps["preprocessor"]` 做一次 `transform(cleaned.X_train.head(20))` 或直接对整份 `cleaned.X_train` 取 shape；如果担心成本，可以只对 `head(20)` 取预览，对整份用 `len(cleaned.X_train)` + `get_feature_names_out()` 推导列数。

## UI 展示方式

不建议只做一条非常长的横条，因为现在真实步骤会超过 10 个，而且存在 train/test 分支。更稳的展示方式是：

第一层是“时间线卡片”，按顺序展示所有快照，支持横向滚动。卡片上只放：步骤名、partition、rows x cols、rows delta、列增减数量。

第二层是“步骤详情面板”。用户选中某个步骤后，下方展示：

1. 结构指标和 metadata；
2. 若有 `preview`，则 `st.dataframe(...)`；
3. 若 `data_kind == "matrix"`，只展示 shape、density、特征列数，不强行渲染矩阵。

这样既能快速扫全局，也不会因为矩阵或 train/test 分叉把 UI 做乱。

## 落盘和结果复用

每个 target run 都单独保存一份 `data_flow.json`，路径与 `plan.json`、`metrics.json` 并列。

推荐在 `results.append(...)` 里把 `data_flow` 一并带出来，然后在 “Run results” expander 中新增一个 `Data flow` 区块做展示。这样不依赖 `st.session_state` 也能工作；只有在后续需要跨 rerun 保留上一次追踪时，再考虑放进 session state。

## 建议的实现顺序

第一步先做 `src/ml_platform/data_flow.py`，把 snapshot、preview、序列化和 diff 逻辑定死。这一步完成后，后面所有模块只负责调用，不再各自拼 JSON。

第二步改 `clean_and_split()`，把最关键的清洗阶段追踪接进去。只要这一步完成，用户已经能看到当前 pipeline 里最核心的数据变化。

第三步改 `train_model()`，补上预处理后矩阵维度和 trainer 摘要。否则 feature engineering 打开时，数据流还是缺最重要的一段。

第四步改 `evaluate_model()` 和 `app.py`，把 `prediction_sample`、metrics 和最终 UI 串起来。

第五步再补 artifact 落盘和 smoke test，确保每次 run 都会产生 `data_flow.json`。

## 测试要求

至少补下面三类测试：

1. `tests/test_data_flow.py`
   覆盖 snapshot diff、preview 截断、JSON 序列化。

2. `tests/test_cleaning.py`
   增加 tracker 集成断言，例如：
   - `after_target_drop.rows_delta < 0`
   - `after_high_missing_drop.columns_removed` 与真实被删列一致
   - `train_split` / `test_split` 行数之和等于清洗后总行数

3. `tests/test_smoke_pipeline.py`
   断言 run 目录里存在 `data_flow.json`，并且至少包含 `target_dataset`、`after_target_drop`、`train_split`、`prediction_sample` 这些关键步骤。

如果只做 UI 不补这些测试，后面一旦清洗逻辑调整，追踪就很容易和真实行为脱节。

## 完成标准

做到下面这些，才算这份 plan 真正落地：

1. 跑一次 demo 数据时，用户能在 UI 里看到从原始表到预测样本的完整步骤序列。
2. 至少能解释清楚三类变化：删行、删列、特征维度扩张。
3. feature engineering 打开和关闭两种情况下，训练阶段的矩阵列数明显不同，并能在追踪里看出来。
4. 每个 run 目录下都有可读的 `data_flow.json`。
5. `uv run pytest` 全绿。

## 现在最稳的结论

如果目标是“观察 pipeline 中每一步的数据变化”，那追踪逻辑必须下沉到 `clean_and_split()` 和 `train_model()`，而不能只靠 `app.py` 外层打点。

如果目标是“UI 可视化 + artifact 落盘”，那 snapshot 结构必须完全 JSON-safe，不能把 DataFrame 直接塞进 dataclass。

这两点不改，后面的实现都会停留在“看起来像有 plan”，但无法稳定落地。
