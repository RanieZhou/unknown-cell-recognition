# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 适用范围

- 日常工程开发应优先面向 `src/scgrn` 中的重构后流程，以及仓库根目录下的 CLI 入口脚本。
- 较早实验阶段留下的材料只应作为低优先级参考。如果历史脚本或说明与 `src/scgrn` 冲突，应优先以重构后的实现为准。
- 当前维护中的主线最终方法是 `lineage_selective_rescue_globalT`，默认 backbone 是 `scanvi`。

## 环境与常用命令

### 环境准备

```bash
conda env create -f environment.yml
conda activate scgrn
```

备选安装方式：

```bash
pip install -r requirements.txt
```

### 主要入口脚本

训练 backbone：

```bash
python train.py --config configs/immune_dc.yaml
```

在单一环境中运行完整推理：

```bash
python infer.py --config configs/immune_dc.yaml
```

如果 backbone 与 GRN 工具链位于不同环境，请改用拆分路径：

```bash
python infer_backbone.py --config configs/immune_dc.yaml
python infer_grn.py --config configs/immune_dc.yaml
```

运行 rescue 阶段：

```bash
python rescue.py --config configs/immune_dc.yaml
```

生成评估输出：

```bash
python evaluate.py --config configs/immune_dc.yaml
```

运行完整流程：

```bash
python run_pipeline.py --config configs/immune_dc.yaml
```

当前仓库中可用的数据集配置示例：

```bash
python train.py --config configs/immune_dc.yaml
python train.py --config configs/pancreas.yaml
```

### 通用 CLI 参数

所有主要入口都支持：

```bash
--seed 42
--output_dir outputs/runs
```

`--use_cached` 目前由 `train.py`、`infer.py`、`infer_grn.py` 和 `run_pipeline.py` 支持。

`infer_backbone.py` 与 `rescue.py` 不暴露 `--use_cached`，因为这两个阶段没有独立缓存策略。

`evaluate.py` 只支持 `--config`、`--seed` 和 `--output_dir`。

### 重要流程说明

- `infer.py` 会同时运行 backbone inference 与 GRN inference。
- `train.py` 会写出 `intermediate/split_assignments.csv`。
- `infer_backbone.py` 会写出 `intermediate/expression_scores.csv`，并在已有 `split_assignments.csv` 时复用它，从而保证训练与推理共享同一份 split 契约。
- expression combiner 仅使用 `val_known` 上的误分类监督进行学习，特征为 `entropy_norm`、`distance_norm` 和 `expr_margin_unknown_score`；如果 `val_known` 为空，或者监督目标退化为单一类别，backbone inference 必须报错，而不是回退到固定权重。
- `infer_grn.py` 会读取 `intermediate/expression_scores.csv`，并写出 GRN 侧产物。
- `rescue.py` 只能在所需的 inference 产物已经存在之后再运行。

## 测试、lint 与构建状态

当前仓库中没有发现统一定义好的测试、lint 或构建命令。

仓库现在已经包含位于 `tests/` 下的一组聚焦型 `unittest` 回归文件，但仍然没有 `pytest` 配置、`pyproject.toml` 或 `Makefile` 这类能定义全仓统一标准命令的配置。因此，在这些工具真正加入之前，不要自行假定统一的 lint、typecheck 或单测命令。

## 整体架构

### 1. 仓库根目录下的轻量 CLI 包装层

根目录脚本都是轻量包装层，基本都遵循同一模式：

1. 解析 CLI 参数
2. 调用 `scgrn.config.load_config(...)`
3. 调用 `scgrn.paths.prepare_run_paths(...)`
4. 分发到 `src/scgrn/pipeline` 下的某个流程阶段

当前映射关系如下：

- `train.py` → `scgrn.pipeline.run_train.run_train`
- `infer.py` → `scgrn.pipeline.run_infer.run_infer`
- `infer_backbone.py` → `scgrn.pipeline.run_infer.run_infer_backbone`
- `infer_grn.py` → `scgrn.pipeline.run_infer.run_infer_grn`
- `rescue.py` → `scgrn.pipeline.run_rescue.run_rescue`
- `evaluate.py` → `scgrn.pipeline.run_evaluate.run_evaluate`
- `run_pipeline.py` → `scgrn.pipeline.run_full_pipeline.run_full_pipeline`

如果要改行为，真正的实现工作通常应落在 `src/scgrn` 中，而不是根目录脚本里。

### 2. 配置模型

配置加载集中在 `src/scgrn/config.py`。

关键行为：

- 数据集配置会叠加在 `configs/default.yaml` 之上。
- CLI 参数会在合并后覆盖部分配置字段。
- 校验逻辑要求以下 section 全部存在：`project`、`data`、`backbone`、`training`、`expression`、`grn`、`rescue`、`threshold` 和 `runtime`。
- 一些 legacy 时代的别名结构会被归一化到当前重构后的配置结构中。

实际含义：如果某个行为看起来与数据集有关，改代码前要同时检查 `configs/default.yaml` 与当前数据集配置。

### 3. 标准化 run 目录结构

run 目录由 `src/scgrn/paths.py` 创建，结构如下：

```text
<output_root>/<backbone_name>/<run_name>/
```

每个 run 下都会包含：

```text
logs/
checkpoints/
intermediate/
predictions/
metrics/
plots/
reports/
config_snapshot.yaml
```

阶段之间的大部分通信都通过 `intermediate/` 下的文件完成，最终分析结果主要位于 `metrics/` 和 `reports/`。

### 4. 流程阶段

重构后的流程是分阶段、基于文件衔接的：

1. **Train**

   - 实现在 `src/scgrn/pipeline/run_train.py`
   - 负责加载数据集、创建 split、选择 backbone adapter 并训练 backbone。
   - 会写出 `intermediate/split_summary.csv`、`intermediate/split_assignments.csv` 等 split 相关产物。
2. **Backbone inference**

   - 实现在 `src/scgrn/pipeline/run_infer.py`
   - 运行 `adapter.predict(...)` 与 `adapter.build_expression_artifacts(...)`。
   - 对下游最关键的契约文件是 `intermediate/expression_scores.csv`。
3. **GRN inference**

   - 同样实现在 `src/scgrn/pipeline/run_infer.py`
   - 读取 `expression_scores.csv`，导出 pySCENIC 输入，运行 GRN 特征构建，并写出：
     - `intermediate/grn_scores.csv`
     - `intermediate/aucell_all_cells.csv`
     - `intermediate/regulon_centroids.csv`
     - `intermediate/dual_fusion_scores.csv`
4. **Rescue**

   - 实现在 `src/scgrn/pipeline/run_rescue.py`
   - 读取 expression 与 GRN 产物，先运行 selective fusion，再运行 lineage-aware rescue。
   - 当前主线最终方法是 `lineage_selective_rescue_globalT`。
5. **Evaluate**

   - 实现在 `src/scgrn/pipeline/run_evaluate.py`
   - 写出 markdown 摘要文件与 legacy consistency 相关输出。
6. **Full pipeline**

   - 实现在 `src/scgrn/pipeline/run_full_pipeline.py`
   - 依次执行 `train -> infer -> rescue -> evaluate`。

### 5. `src/scgrn` 内部包结构

重构后的代码按职责划分：

- `data/`：数据集加载、预处理与 split 生成
- `backbone/`：backbone 抽象层、不同 backbone 的 adapter、expression artifact 提取
- `grn/`：pySCENIC 导出/加载与 GRN 空间打分
- `rescue/`：selective fusion、lineage-aware rescue、threshold 与 corridor 逻辑
- `evaluation/`：指标、图表、报告与 legacy consistency 检查
- `pipeline/`：顶层阶段编排
- `utils/`：IO、日志、随机种子等辅助工具

## 重要契约与不变量

### Expression artifact schema 是最重要的下游接口

`src/scgrn/backbone/base.py` 定义了标准化的 expression artifact schema。下游的 rescue 与 evaluation 代码默认该 schema 存在且兼容。

仓库中最重要的阶段交接文件是：

```text
intermediate/expression_scores.csv
```

如果你要修改 backbone 行为，或者新增 backbone，请保持这个文件契约稳定，除非你明确打算同步更新所有下游消费者。

### Backbone 扩展点是显式的

backbone 抽象位于 `src/scgrn/backbone/base.py`，adapter 选择逻辑位于 `src/scgrn/backbone/registry.py`。

当前状态：

- `scanvi` 已实现
- `scnym` 已实现
- `scgpt` 不在当前主线配置校验支持范围内；它应被视为未来扩展点，而不是当前可运行 backbone

如果新增 backbone，预期路径是：

1. 在 `src/scgrn/backbone/` 下新增 adapter
2. 在 `src/scgrn/backbone/registry.py` 中注册
3. 保持标准化 expression artifact 输出与下游阶段兼容

### Split 是核心实验状态

数据切分逻辑位于 `src/scgrn/data/make_split.py`。整个流程依赖以下标准 split 标签：

- `train_known`
- `val_known`
- `test_known`
- `test_unknown`

这些 split 标签会贯穿训练、校准、rescue 与评估。只要改动 split 语义，就会影响全仓流程。

## 当前配置中的数据集默认值

`configs/default.yaml` 给出了当前日常开发的默认前提：

- 数据集名称：`immune_dc`
- 默认 backbone：`scanvi`
- 方法名：`lineage_selective_rescue_globalT`
- 输出根目录：`outputs/runs`
- 默认随机种子：`42`

`configs/pancreas.yaml` 是当前最主要的数据集特化示例：

- 将 backbone 切换为 `scnym`
- 使用 `counts_source: x`
- 使用 `model_layer: model_input`

当你在排查数据集特定行为时，应先对照 `configs/default.yaml` 与对应数据集配置的差异。

## Legacy 背景

部分文档仍然总结了历史实验阶段的信息，这些内容在解释结果时可能有用，但它们不是默认实现路径。修改代码时，应优先依赖 `src/scgrn` 下的重构后模块，而不是回到早期实验结构。