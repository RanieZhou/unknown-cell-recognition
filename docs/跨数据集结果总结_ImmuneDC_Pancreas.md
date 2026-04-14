# 跨数据集结果总结：Immune DC 与 Pancreas 最小迁移验证

## 1. 文档目的

> 说明：文中的 `E008`、`E009`、`E010` 是历史实验阶段编号，用于对应当时的结果目录与分析批次，不代表当前主线的默认命名规范。当前维护主线仍以 `src/scgrn/`、根目录入口脚本和 `configs/` 为准。

本文件用于统一总结当前仓库在两个数据集上的实验结果：

1. `Immune DC` 主数据集  
   数据文件：`data/human_immune_health_atlas_dc.patched.h5ad`
2. `Pancreas` 第二数据集的最小迁移验证  
   数据文件：`data/human_pancreas_norm_complexBatch.patched.h5ad`

目标不是继续扩展新方法，而是回答两个问题：

1. 当前主方法 `lineage_selective_rescue_globalT` 是否在主数据集上有稳定、可解释的收益？
2. 该方法在第二个数据集上是否仍然是最优或最稳，从而支持“global fusion 无效、local rescue 更合理”的总体结论？

---

## 2. 最终结论先看

截至当前实验，最推荐作为主结果汇报的方法仍然是：

- `lineage_selective_rescue_globalT`

原因不是它在所有场景下都大幅领先，而是它同时满足了下面 4 点：

1. 在 `Immune DC` 主数据集上，相对 `expr_fused` 和 `selective_fused_score` 有稳定增益。
2. 它的收益来源可以被定位到特定 lineage corridor，而不是黑箱式全局融合。
3. 在第二个 `Pancreas` 数据集上，它仍然是 5 个候选方法中最优或最稳的方法。
4. 进一步针对 `cDC1-like corridor` 的扩展尝试没有带来实质收益，因此当前主方法不应再被更复杂的变体替代。

可以压缩成一句话：

> expression-only 仍是强 baseline；GRN 不适合作为全局增强信号；但在 expression 不确定且生物学 corridor 可解释的局部区域中，GRN 作为 rescue 证据是合理且可复现的。

---

## 3. 数据集与任务设置

### 3.1 Immune DC 主数据集

- 已知类：`cDC2`, `pDC`, `cDC1`
- 未知类：`ASDC`
- 主要结论来自：
  - `outputs/E008/`
  - `outputs/E009/`
  - `outputs/E010/`

### 3.2 Pancreas 最小迁移验证数据集

- 已知类：`alpha`, `beta`, `delta`
- 未知类：`gamma`
- 结果位于：
  - `outputs/NEWDATA_EVAL/`

### 3.3 两个数据集上保持不变的核心方法学约束

- 固定使用 `global threshold`
- 不采用 `bucket-specific threshold` 作为最终判定规则
- 保持 `distance` 分支定义为“到所有 known 类中心的最小距离”
- 主对比逻辑保持为：
  - `expr_fused`
  - `grn_distance`
  - `dual_fused_v1` 或其旧版全局融合对照
  - `selective_fused_score`
  - `lineage_selective_rescue_globalT`

---

## 4. Immune DC：主数据集结果总结

### 4.1 多 seed 稳定性

多 seed 验证使用 `seed = 42, 43, 44`，关键汇总见：

- `outputs/E008/E008_multiseed_method_summary_mean_std.csv`

关键结果如下：

| 方法 | AUROC mean | AUPR mean | Macro-F1 mean | 备注 |
|---|---:|---:|---:|---|
| `expr_fused` | 0.9735 | 0.8651 | 0.8797 | 强 baseline |
| `grn_distance` | 0.7385 | 0.2581 | 0.5337 | 明显较弱 |
| `dual_fused_v2_rankavg` | 0.9410 | 0.6382 | 0.8038 | 全局融合失败 |
| `selective_fused_score` | 0.9723 | 0.8579 | 0.8733 | 接近但不如 `expr_fused` |
| `lineage_selective_rescue_globalT` | 0.9758 | 0.8777 | 0.8897 | 当前最优 |

这组多 seed 结果给出的结论是：

1. `lineage_selective_rescue_globalT` 在 3 个 seed 中满足稳定优于或不低于 `selective_fused_score` 的条件。
2. `bucketT` 仍然不优于 `globalT`。
3. `cDC2_like` FP 没有明显膨胀。

因此，主数据集上当前主方法的收益不是偶然的单 seed 现象。

#### 多 seed 可视化

下图展示了 `expr_fused`、`selective_fused_score`、`lineage_selective_rescue_globalT`、`lineage_selective_rescue_bucketT` 在 3 个 seed 上的 `AUPR / Macro-F1 / cDC1_like recall` 变化趋势。可以直接看到：

1. `globalT` 整体稳定优于或不低于 `selective_fused_score`
2. `bucketT` 没有形成更稳的提升

![E008 多 seed 稳定性图](../outputs/E008/E008_seed_summary_plot.png)

### 4.2 生物学解释性

关键解释性结果位于：

- `outputs/E009/E009_summary.md`
- `outputs/E009/E009_regulon_rescued_vs_stillFN.csv`
- `outputs/E009/E009_regulon_stillFN_vs_cDC1.csv`
- `outputs/E009/E009_stable_regulon_panel.csv`

这组解释性分析的核心结论不是“ASDC 被完全分成两个纯类”，而是：

1. `rescued_ASDC` 更偏 `pDC-like corridor`
2. `still_FN_ASDC` 更偏 `cDC1-like corridor`
3. 这种差异在 regulon / UMAP 层面比在 marker gene 层面更清楚

当前可用于后续方法构建的稳定 regulon panel 被压缩为 15 个 regulon：

- `ASDC-high`: `SPIB(+)`, `IRF8(+)`, `IRF7(+)`, `TCF4(+)`, `BCL11A(+)`, `ETS1(+)`, `BPTF(+)`, `NCOA1(+)`, `TFEC(+)`, `IKZF1(+)`
- `cDC1-reference`: `HMGA1(+)`, `SPI1(+)`, `ZNF711(+)`, `BCL6(+)`, `YBX1(+)`

这支持一个更谨慎但更可靠的解释：

> `lineage_selective_rescue_globalT` 的收益具有明确的生物学 corridor 结构，而不是单纯依赖黑箱打分提升。

#### 解释性可视化

下面几张图分别从 marker、regulon、UMAP 投影三个角度支持这一点。

`rescued_ASDC` / `still_FN_ASDC` 的 marker 热图：

![E009 marker heatmap](../outputs/E009/E009_marker_heatmap.png)

ASDC subgroup 的 regulon 热图：

![E009 regulon heatmap](../outputs/E009/E009_regulon_heatmap.png)

expression 空间下的 subgroup 投影：

![E009 expression UMAP](../outputs/E009/E009_umap_expression_subgroups.png)

regulon 空间下的 subgroup 投影：

![E009 regulon UMAP](../outputs/E009/E009_umap_regulon_subgroups.png)

### 4.3 cDC1-like 扩展失败

关键结果位于：

- `outputs/E010/E010_method_comparison_mean_std.csv`
- `outputs/E010/E010_summary.md`

对比 `lineage_selective_rescue_globalT` 与 `dual_corridor_rescue`：

| 方法 | AUPR mean | Macro-F1 mean | cDC1_like recall mean | pDC_like recall mean | cDC2_like FP mean |
|---|---:|---:|---:|---:|---:|
| `lineage_selective_rescue_globalT` | 0.8777 | 0.8897 | 0.8265 | 0.8968 | 0.0265 |
| `dual_corridor_rescue` | 0.8801 | 0.8895 | 0.8265 | 0.8977 | 0.0277 |

这次 cDC1-like 扩展尝试的结论很明确：

1. `AUPR` 略升，但 `Macro-F1` 没有提升。
2. `cDC1_like recall` 没有改善。
3. `cDC2_like FP` 反而略高。
4. 候选 `A/B` 本身非常弱，说明当前 cDC1-specific 证据还不足以形成稳定增益。

因此：

- 这次扩展尝试应作为负结果或附录保留
- 不应替代 `lineage_selective_rescue_globalT`

---

## 5. Pancreas：第二数据集最小迁移验证

关键结果位于：

- `outputs/NEWDATA_EVAL/NEWDATA_method_comparison.csv`
- `outputs/NEWDATA_EVAL/NEWDATA_summary.md`

注意：该数据集没有 `raw counts`，因此此次迁移验证在实现上做了最小必要调整：

- backbone 使用归一化 `X`
- `SCVI gene_likelihood = normal`
- pySCENIC 也从归一化表达矩阵导出

这意味着它是“最小迁移验证”，不是和主数据集完全同条件的原样复制。

### 5.1 新数据集的 5 方法对比

| 方法 | AUROC | AUPR | FPR95 | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| `expr_fused` | 0.9141 | 0.7902 | 0.3296 | 0.8145 | 0.7423 |
| `grn_distance` | 0.6039 | 0.3869 | 0.8843 | 0.6935 | 0.4916 |
| `dual_fused_v1` | 0.9134 | 0.7824 | 0.3315 | 0.8080 | 0.7310 |
| `selective_fused_score` | 0.9133 | 0.7855 | 0.3296 | 0.8101 | 0.7356 |
| `lineage_selective_rescue_globalT` | 0.9172 | 0.8029 | 0.3296 | 0.8184 | 0.7453 |

对应判断如下：

1. `expr_fused` 仍然是强 baseline。
2. `grn_distance` 仍然明显偏弱。
3. `dual_fused_v1` 仍然不如 `expression-only`。
4. `selective_fused_score` 这次只达到“接近”，没有超过 `expr_fused`。
5. `lineage_selective_rescue_globalT` 是新数据集上最优或最稳的方法。

#### 新数据集可视化

ROC / PR 图直接展示了 5 个方法在 pancreas 数据集上的相对排序：

![NEWDATA ROC](../outputs/NEWDATA_EVAL/plots/NEWDATA_roc.png)

![NEWDATA PR](../outputs/NEWDATA_EVAL/plots/NEWDATA_pr.png)

### 5.2 新数据集上主方法为何更稳

从 confusion matrix 看：

- `expr_fused`: TP = 329, FP = 58
- `selective_fused_score`: TP = 322, FP = 61
- `lineage_selective_rescue_globalT`: TP = 326, FP = 46

这说明 `lineage_selective_rescue_globalT` 在新数据集上的优势主要不是靠更激进地提高 recall，而是：

1. 基本保住了 TP
2. 同时显著减少了 FP

因此它表现出的是“更稳”的改进，而不是冒进式 gain。

主方法在新数据集上的 confusion matrix 如下，直观看到它主要通过减少 FP 获益：

![NEWDATA confusion matrix](../outputs/NEWDATA_EVAL/plots/NEWDATA_confusion_matrix_lineage_selective_rescue_globalT.png)

### 5.3 新数据集上的结构性现象

从 per-cell 结果看，主方法的收益主要集中在一个特定 lineage corridor 上，而不是所有 bucket 都均匀改善。

这再次支持了同一套方法学观点：

> 全局融合仍然不合理；当 GRN 只在 expression 不确定、且 corridor 上有生物学解释时作为 rescue 证据，结果更稳定。

---

## 6. 两个数据集放在一起看

### 6.1 一致复现的模式

在 `Immune DC` 和 `Pancreas` 两个数据集上，一致复现了以下格局：

1. `expr_fused` 是强 baseline  
   expression-only 已经足够强，说明 backbone 提供了扎实的主信息。

2. `grn_distance` 作为全局主信号偏弱  
   单独依赖 GRN 距离无法替代 expression。

3. `dual_fused_v1` 这类全局融合不如 expression-only  
   说明 GRN 不是稳定的全局增益信号。

4. `selective_fused_score` 只能在局部接近或小幅改善  
   说明“只在 expression 不确定时引入 GRN”是对的方向，但还不够充分。

5. `lineage_selective_rescue_globalT` 在两个数据集上都最优或最稳  
   这是当前最强的跨数据集结论。

### 6.2 不一致或需要谨慎解读的部分

1. `Immune DC` 上的解释性更强  
   因为有明确的 `ASDC -> pDC-like / cDC1-like corridor` 结构，并且解释性分析给出了 subgroup 与 regulon 支撑。

2. `Pancreas` 只是最小迁移验证  
   它证明方法学格局是可迁移的，但没有提供和 `Immune DC` 同等深度的 subgroup 生物学解释。

3. `cDC1-like` 的进一步扩展没有成功  
   E010 说明当前可用 regulon panel 还不足以把 `cDC1-like residual` 单独救起来，因此主方法应停在 `lineage_selective_rescue_globalT`。

---

## 7. 对“生物学可解释性”的最终判断

如果目标是“做一个不仅有分数，而且有生物学解释路径的方法”，那么当前最合适的结论是：

- `lineage_selective_rescue_globalT` 已经达到了“机制可分解、带有生物学启发、结果可复现”的可解释性要求。

它可解释的地方在于：

1. 先由 expression backbone 定义“不确定样本”
2. 再由 lineage corridor 决定 GRN 是否允许介入
3. 通过 `positive-only rescue` 限制 GRN 只能作为增量证据
4. 最终仍保持单一 `global threshold`，避免 bucket-specific threshold 的人为技巧性增益

因此更准确的表述不是：

- “该方法恢复了真实调控机制”

而是：

- “该方法将 unknown detection 的收益定位到了具有生物学意义的局部 corridor，并且这种局部 rescue 结构在第二个数据集上也保持成立”

---

## 8. 最终推荐

当前推荐的最终结论与方法选择如下：

1. 主方法：
   - `lineage_selective_rescue_globalT`

2. 主数据集结论：
   - 方法增益在多 seed 上稳定
   - 收益主要来自 biologically meaningful corridor
   - `bucket-specific threshold` 是错误方向

3. 第二数据集结论：
   - `expr_fused` 仍强
   - `grn_distance` 仍弱
   - `dual_fused_v1` 仍不如 expression-only
   - `lineage_selective_rescue_globalT` 仍最优或最稳

4. 对后续扩展的判断：
   - 暂不建议继续扩展更复杂的 fusion 或 cDC1-specific rescue 变体
   - 当前应以 `lineage_selective_rescue_globalT` 作为最终主方法
   - cDC1-like 扩展尝试保留为负结果 / 附录分析

---

## 9. 关键结果文件索引

### Immune DC

- `outputs/E008/E008_multiseed_method_summary_mean_std.csv`
- `outputs/E009/E009_summary.md`
- `outputs/E009/E009_stable_regulon_panel.csv`
- `outputs/E010/E010_summary.md`
- `outputs/E010/E010_method_comparison_mean_std.csv`

### Pancreas

- `outputs/NEWDATA_EVAL/NEWDATA_method_comparison.csv`
- `outputs/NEWDATA_EVAL/NEWDATA_summary.md`
- `outputs/NEWDATA_EVAL/NEWDATA_confusion_matrix_lineage_selective_rescue_globalT.csv`
