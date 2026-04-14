# SCGRN

当前仓库维护中的主线最终方法是 `lineage_selective_rescue_globalT`。

当前维护中的主线实现位于 [src/scgrn](./src/scgrn)。历史实验脚本仅作为参考材料保留，不应视为默认实现路径。

## Backbone

默认 backbone 是 `scANVI`，通过以下配置指定：

```yaml
backbone:
  name: scanvi
```

当前已验证的主线组合为：

- `scanvi + lineage_selective_rescue_globalT`

目前仍受支持的维护中 backbone 有：

- `scanvi`
- `scnym`

允许 backbone 生成额外的中间产物，但下游稳定业务接口应保持为 expression artifact 表。

expression 侧融合不再使用手工固定权重，而是基于 `val_known` 上的误分类监督学习得到。backbone 阶段会先在 `val_known` 上校准 `entropy_norm`、`distance_norm` 和 `expr_margin_unknown_score`，再拟合一个确定性的逻辑回归组合器，其输出列为 `expr_fused`。

## 主线命令

训练 backbone：

```bash
python train.py --config configs/immune_dc.yaml
```

运行推理与上游特征构建：

```bash
python infer.py --config configs/immune_dc.yaml
```

如果 `scanvi` 和 `pyscenic` 位于不同环境，请改用拆分命令：

```bash
# scanvi 环境
python infer_backbone.py --config configs/immune_dc.yaml

# pyscenic 环境
python infer_grn.py --config configs/immune_dc.yaml
```

运行 selective rescue 与 lineage rescue：

```bash
python rescue.py --config configs/immune_dc.yaml
```

生成评估摘要：

```bash
python evaluate.py --config configs/immune_dc.yaml
```

该步骤还会在当前 run 的 `reports/` 和 `metrics/` 目录下写出与 legacy 对齐的基线报告和 CSV 表。

端到端运行完整流程：

```bash
python run_pipeline.py --config configs/immune_dc.yaml
```

`infer.py` 和 `run_pipeline.py` 默认假设单一环境同时具备 `scanvi` 与 `pyscenic` 的运行条件。如果 backbone 与 GRN 工具链分属不同环境，请使用 `infer_backbone.py` 和 `infer_grn.py`，不要直接使用 `infer.py`。

常用参数：

- `--seed 42`：覆盖配置中的随机种子
- `--output_dir outputs/runs`：覆盖默认输出根目录
- `--use_cached`：在存在现有 checkpoint 或中间产物时复用缓存；`train.py`、`infer.py`、`infer_grn.py` 和 `run_pipeline.py` 支持该参数

对于拆分推理路径：

- `train.py` 会写出 `intermediate/split_assignments.csv`
- `infer_backbone.py` 会写出 `expression_scores.csv` 等 backbone 侧产物，并在已有 `split_assignments.csv` 时复用它，从而保证训练与推理使用同一份 split 契约
- `infer_grn.py` 会读取 `expression_scores.csv`，并写出 `grn_scores.csv` 等 GRN 侧产物
- `rescue.py` 应在上述两个阶段都完成后再运行

## 项目结构

- [configs](./configs)：重构后主线流程使用的 YAML 配置文件
- [src/scgrn](./src/scgrn)：当前维护中的主线实现
- [docs](./docs)：实验总结、数据集对比与方法说明
- [outputs/runs](./outputs/runs)：重构后主线流程的标准化运行输出
- [outputs/scripts/legacy](./outputs/scripts/legacy)：仅用于追溯的历史实验脚本归档
- [outputs](./outputs)：包含标准化运行结果与保留作参考的历史实验输出

日常工程开发应优先关注 `src/scgrn`、仓库根目录入口脚本以及 `configs/`。`outputs/scripts/legacy` 和 `outputs/` 下的历史实验目录应视为历史材料。

## 输出结构

标准化 run 目录结构如下：

```text
outputs/
  runs/
    scanvi/
      immune_dc_lsr_globalT/
        config_snapshot.yaml
        logs/
        checkpoints/
        intermediate/
        predictions/
        metrics/
        plots/
        reports/
```

最终主线结果是 `lineage_selective_rescue_globalT`，保存在当前 run 的 `metrics/lineage_method_comparison.csv` 和 `reports/lineage_summary.md` 中。
