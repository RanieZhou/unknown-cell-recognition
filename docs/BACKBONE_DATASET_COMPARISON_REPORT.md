# Backbone × Dataset 对比实验汇报文档

## 1. 实验目的

本轮实验的目标不是单纯比较分类精度，而是系统评估以下问题：

1. 在统一的开放集识别主线下，不同 backbone 是否会改变 expression-side 统计特性。
2. `lineage_selective_rescue_globalT` 这条主线在不同 backbone、不同数据集上是否仍然成立。
3. 引入 GRN / regulon 特征后，是否能够在不显著损伤识别性能的前提下提升未知细胞识别的生物学可解释性。
4. 哪一组组合最适合作为后续研究和优化的主干版本。

本次对比覆盖 2 个 backbone 和 2 个数据集，共 4 组完整运行：

- `scanvi × immune_dc`
- `scnym × immune_dc`
- `scanvi × pancreas`
- `scnym × pancreas`

## 2. 实验设计

### 2.1 主线流程

四组实验均采用同一条标准流程：

1. backbone 训练
2. backbone 推断并生成统一 expression artifacts
3. pySCENIC / AUCell / regulon centroid / GRN distance 构建
4. selective fusion
5. lineage-selective rescue
6. evaluation、summary、plot 输出

### 2.2 统一方法结构

本轮对比并未更换后段主流程，而是保持以下结构一致：

- expression uncertainty / distance
- GRN distance / auxiliary score
- `selective_fused_score`
- `lineage_selective_rescue_globalT`
- `lineage_selective_rescue_bucketT`

这样做的目的，是把变化尽量收敛到 backbone 本身。

### 2.3 数据集设置

#### immune_dc

- 数据文件：`data/human_immune_health_atlas_dc.patched.h5ad`
- 标签列：`AIFI_L2`
- known classes：`cDC2`, `pDC`, `cDC1`
- unknown class：`ASDC`
- split：
  - `train_known = 15935`
  - `val_known = 3415`
  - `test_known = 3415`
  - `test_unknown = 522`

#### pancreas

- 数据文件：`data/human_pancreas_norm_complexBatch.patched.h5ad`
- 标签列：`celltype`
- known classes：`alpha`, `beta`, `delta`
- unknown class：`gamma`
- split：
  - `train_known = 7501`
  - `val_known = 1608`
  - `test_known = 1608`
  - `test_unknown = 699`

### 2.4 Backbone 设置

#### scANVI

- 作为当前原始主干 backbone
- 更适合直接输出 uncertainty 相关信号
- 当前 selective fusion / lineage rescue 的参数和逻辑最初主要是在该 backbone 行为上形成的

#### scNym

- 作为可替换 backbone 接入
- 使用官方 semi-supervised 设置：
  - unlabeled target
  - MixMatch
  - DANN
  - pseudolabel thresholding

### 2.5 输出目录

不同 backbone 已按目录隔离：

- `outputs/runs/scanvi/immune_dc_lsr_globalT/`
- `outputs/runs/scnym/immune_dc_lsr_globalT/`
- `outputs/runs/scanvi/pancreas_lsr_globalT/`
- `outputs/runs/scnym/pancreas_lsr_globalT/`

## 3. 评估指标

本轮主要使用以下指标：

- `AUROC`
- `AUPR`
- `FPR95`
- `Accuracy`
- `Macro_F1`
- `Precision`
- `Recall`

对于 lineage rescue 额外关注：

- `rescued_count`
- `hurt_count`
- `net_gain`

对 bucket 级分析额外关注：

- `Recall_on_unknown`
- `FP_rate_on_known`
- 各 lineage bucket 的净收益变化

## 4. 关键结果总表

### 4.1 各组合最优方法概览

| 数据集 | Backbone | selective 阶段最优 | selective Macro-F1 | selective AUPR | lineage 阶段最优 | lineage Macro-F1 | lineage AUPR |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| immune_dc | scanvi | entropy | 0.9031 | 0.8730 | lineage_selective_rescue_globalT | 0.8806 | 0.8402 |
| immune_dc | scnym | expr_distance | 0.8801 | 0.8590 | dual_fused_v2_rankavg | 0.7947 | 0.6116 |
| pancreas | scanvi | entropy | 0.8815 | 0.9187 | expr_fused | 0.8531 | 0.8880 |
| pancreas | scnym | expr_distance | 0.7405 | 0.8265 | lineage_selective_rescue_globalT | 0.7426 | 0.6770 |

这张表已经给出一个核心事实：

- `scanvi` 仍是整体更强、更稳的 backbone。
- `immune_dc + scanvi` 是最接近原始主线结论的一组。
- `scnym` 改变了 expression-side 信号结构，导致原有 fusion / rescue 不再天然成立。

## 5. immune_dc 结果分析

### 5.1 scanvi × immune_dc

结果位置：

- `outputs/runs/scanvi/immune_dc_lsr_globalT/metrics/`
- `outputs/runs/scanvi/immune_dc_lsr_globalT/plots/`
- `outputs/runs/scanvi/immune_dc_lsr_globalT/reports/`

#### selective 阶段结果

| 方法 | AUROC | AUPR | Macro-F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| entropy | 0.9814 | 0.8730 | 0.9031 | 0.7690 | 0.9119 |
| expr_distance | 0.9668 | 0.7947 | 0.8717 | 0.7540 | 0.8046 |
| expr_fused | 0.9685 | 0.8203 | 0.8772 | 0.7589 | 0.8199 |
| grn_distance | 0.8505 | 0.5013 | 0.7560 | 0.6140 | 0.5364 |
| dual_fused_v2_rankavg | 0.9587 | 0.7735 | 0.8712 | 0.7465 | 0.8123 |
| selective_fused_score | 0.9704 | 0.8384 | 0.8775 | 0.7570 | 0.8238 |

结论：

- `entropy` 是单项最强指标。
- `selective_fused_score` 相比 `expr_fused` 有小幅正增益。
- GRN 分支单独不如 expression，但仍然具备中等区分能力。

#### lineage 阶段结果

| 方法 | AUROC | AUPR | Macro-F1 | Precision | Recall | net_gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| expr_fused | 0.9685 | 0.8203 | 0.8772 | 0.7589 | 0.8199 | - |
| selective_fused_score | 0.9704 | 0.8384 | 0.8775 | 0.7570 | 0.8238 | - |
| lineage_selective_rescue_globalT | 0.9711 | 0.8402 | 0.8806 | 0.7665 | 0.8238 | +7 |
| lineage_selective_rescue_bucketT | 0.9711 | 0.8402 | 0.8505 | 0.7400 | 0.7414 | -41 |

结论：

- `globalT` 仍然是最终最优方法。
- `bucketT` 明显过度修正，尤其伤害了部分 bucket 的 recall。
- 这组结果与原主线结论一致，说明 `scanvi + immune_dc` 仍可作为当前最稳定主干。

#### 图表解读

1. `selective_roc.png`
   - `selective_fused_score` 曲线略优于 `expr_fused`
   - `grn_distance` 明显低于 expression 主线，但并非完全失效

2. `lineage_roc.png`
   - `globalT` 与 `selective_fused_score` 相比有轻微上移
   - `bucketT` 的 ROC 不差，但最终阈值判别效果变差，说明问题主要出在 threshold / bucket correction

3. `lineage_bucket_barplot.png`
   - `cDC1_like` 桶是 `bucketT` 的主要失分来源
   - `globalT` 更稳，没有明显破坏 bucket 内部平衡

#### 生物学解释层

这一组是最能支撑“引入 GRN 是为了增加解释性而不是纯粹刷指标”的：

- GRN 提供了 `nearest_grn_class`
- lineage bucket 提供了 `pDC_like / cDC1_like / cDC2_like` 语境
- regulon heatmap 与 top regulon barplot 可以支撑 ASDC 的调控程序解释

### 5.2 scnym × immune_dc

结果位置：

- `outputs/runs/scnym/immune_dc_lsr_globalT/metrics/`
- `outputs/runs/scnym/immune_dc_lsr_globalT/plots/`
- `outputs/runs/scnym/immune_dc_lsr_globalT/reports/`

#### selective 阶段结果

| 方法 | AUROC | AUPR | Macro-F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| entropy | 0.5480 | 0.2829 | 0.5843 | 0.3464 | 0.2031 |
| expr_distance | 0.9763 | 0.8590 | 0.8801 | 0.6973 | 0.9310 |
| expr_fused | 0.8885 | 0.5853 | 0.7231 | 0.5460 | 0.4885 |
| grn_distance | 0.7734 | 0.3417 | 0.6342 | 0.4673 | 0.2739 |
| dual_fused_v2_rankavg | 0.9186 | 0.6116 | 0.7947 | 0.6259 | 0.6667 |
| selective_fused_score | 0.8893 | 0.5905 | 0.7322 | 0.5647 | 0.5019 |

结论：

- `expr_distance` 成为绝对主导信号。
- `entropy` 基本失效，这与 `scanvi` 的行为完全不同。
- 原有 `expr_fused = 0.5 * entropy + 0.5 * distance` 在该 backbone 上明显不合适。
- selective fusion 只比 `expr_fused` 略好，但远不如 `expr_distance`。

#### lineage 阶段结果

| 方法 | AUROC | AUPR | Macro-F1 | Precision | Recall | net_gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| expr_fused | 0.8885 | 0.5853 | 0.7231 | 0.5460 | 0.4885 | - |
| selective_fused_score | 0.8893 | 0.5905 | 0.7322 | 0.5647 | 0.5019 | - |
| lineage_selective_rescue_globalT | 0.8899 | 0.5978 | 0.7165 | 0.5381 | 0.4732 | -8 |
| lineage_selective_rescue_bucketT | 0.8899 | 0.5978 | 0.7008 | 0.5283 | 0.4291 | -19 |
| dual_fused_v2_rankavg | 0.9186 | 0.6116 | 0.7947 | 0.6259 | 0.6667 | - |

结论：

- 当前 lineage rescue 不再成立。
- `globalT` 和 `bucketT` 都是负增益。
- 这不是“模型坏了”，而是 backbone 改变后，后段参数与信号假设未重新校准。

#### 图表解读

1. `selective_roc.png`
   - 红线 `selective_fused_score` 只比蓝线 `expr_fused` 略高
   - 但真正最强的信号其实是 `expr_distance`，该信息主要体现在指标表中

2. `lineage_roc.png`
   - lineage rescue 曲线与 selective 几乎贴合
   - 说明 lineage rescue 没有创造新的区分能力

3. `lineage_bucket_barplot.png`
   - `pDC_like` 有局部收益
   - `cDC1_like` 和 `cDC2_like` 的误伤抵消了整体增益

#### 解释

这组结果给出的关键信息是：

- `scnym` 并不天然兼容现有 selective fusion / lineage rescue 设计。
- 如果继续用 `scnym`，优先要重配 expression 融合层，而不是直接继续调 rescue。

## 6. pancreas 结果分析

### 6.1 scanvi × pancreas

结果位置：

- `outputs/runs/scanvi/pancreas_lsr_globalT/metrics/`
- `outputs/runs/scanvi/pancreas_lsr_globalT/plots/`
- `outputs/runs/scanvi/pancreas_lsr_globalT/reports/`

#### selective 阶段结果

| 方法 | AUROC | AUPR | Macro-F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| entropy | 0.9738 | 0.9187 | 0.8815 | 0.9066 | 0.7639 |
| expr_distance | 0.8027 | 0.5641 | 0.4220 | 0.3030 | 0.0143 |
| expr_fused | 0.9469 | 0.8880 | 0.8531 | 0.9159 | 0.6853 |
| grn_distance | 0.6060 | 0.3619 | 0.4582 | 0.3659 | 0.0644 |
| dual_fused_v2_rankavg | 0.9237 | 0.7863 | 0.6347 | 0.8049 | 0.2833 |
| selective_fused_score | 0.9468 | 0.8870 | 0.8510 | 0.9105 | 0.6838 |

结论：

- `entropy` 依然最强，说明 `scanvi` 在 pancreas 上也保留了 uncertainty 优势。
- `expr_distance` 在 pancreas 上很弱，几乎不能直接作为 backbone 主信号。
- selective fusion 没有创造增益，`selective_fused_score` 略低于 `expr_fused`。

#### lineage 阶段结果

| 方法 | AUROC | AUPR | Macro-F1 | Precision | Recall | net_gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| expr_fused | 0.9469 | 0.8880 | 0.8531 | 0.9159 | 0.6853 | - |
| selective_fused_score | 0.9468 | 0.8870 | 0.8510 | 0.9105 | 0.6838 | - |
| lineage_selective_rescue_globalT | 0.9453 | 0.8826 | 0.8384 | 0.8988 | 0.6609 | -25 |
| lineage_selective_rescue_bucketT | 0.9453 | 0.8826 | 0.8223 | 0.8971 | 0.6237 | -49 |

结论：

- 在 pancreas 上，`scanvi` backbone 本身是有效的。
- 但 lineage rescue 规则不适合直接迁移到 pancreas。
- 也就是说，问题不在 backbone，而在后段规则跨数据集泛化不足。

#### 图表解读

1. `selective_roc.png`
   - 蓝线和红线几乎重合
   - 说明 selective fusion selective gating 对 pancreas 几乎没有净收益

2. `lineage_bucket_barplot.png`
   - `beta_like` recall 有提升
   - `delta_like` recall 被显著拉低，是总体负收益的核心原因

### 6.2 scnym × pancreas

结果位置：

- `outputs/runs/scnym/pancreas_lsr_globalT/metrics/`
- `outputs/runs/scnym/pancreas_lsr_globalT/plots/`
- `outputs/runs/scnym/pancreas_lsr_globalT/reports/`

#### selective 阶段结果

| 方法 | AUROC | AUPR | Macro-F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| entropy | 0.6728 | 0.6357 | 0.7185 | 0.8172 | 0.4349 |
| expr_distance | 0.9215 | 0.8265 | 0.7405 | 0.8553 | 0.4649 |
| expr_fused | 0.7264 | 0.6746 | 0.7380 | 0.8460 | 0.4635 |
| grn_distance | 0.5959 | 0.3532 | 0.4493 | 0.3171 | 0.0558 |
| dual_fused_v2_rankavg | 0.7209 | 0.5742 | 0.6044 | 0.7068 | 0.2518 |
| selective_fused_score | 0.7263 | 0.6741 | 0.7365 | 0.8316 | 0.4664 |

结论：

- selective 阶段最强的仍然是 `expr_distance`。
- 但与 `immune_dc` 不同，这里 `scnym` 的整体绝对性能明显下降。
- selective fusion 没有帮助，`selective_fused_score` 甚至略低于 `expr_fused`。

#### lineage 阶段结果

| 方法 | AUROC | AUPR | Macro-F1 | Precision | Recall | net_gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| expr_fused | 0.7264 | 0.6746 | 0.7380 | 0.8460 | 0.4635 | - |
| selective_fused_score | 0.7263 | 0.6741 | 0.7365 | 0.8316 | 0.4664 | - |
| lineage_selective_rescue_globalT | 0.7268 | 0.6770 | 0.7426 | 0.8483 | 0.4721 | +6 |
| lineage_selective_rescue_bucketT | 0.7268 | 0.6770 | 0.7306 | 0.8400 | 0.4506 | -10 |

结论：

- `globalT` 有轻微正增益。
- 但这组的整体指标仍明显低于 `scanvi + pancreas`。
- 因此这不是一个值得替代 `scanvi` 的结果，只能说明该组在当前参数下可以跑通，且保留了局部 rescue 能力。

#### 图表解读

1. `selective_roc.png`
   - expression fused 与 selective fused 几乎完全重合
   - GRN 仍然较弱

2. `lineage_bucket_barplot.png`
   - `alpha_like` 基本无显著变化
   - `beta_like` 和 `delta_like` 也只有局部细微调整
   - 这解释了为什么 `globalT` 只有非常有限的正收益

## 7. 跨组合综合结论

### 7.1 backbone 维度

#### scanvi

优点：

- 在两个数据集上都能稳定输出高质量 uncertainty 信号
- 与现有 `expr_fused` 及后段 rescue 逻辑兼容性更好
- 仍然是当前最适合作为主干的 backbone

不足：

- pancreas 上 lineage rescue 泛化不足
- GRN 分支在 pancreas 上贡献有限

#### scnym

优点：

- 可作为可插拔 backbone 正常接入完整流程
- 在 `immune_dc` 与 `pancreas` 上都能形成有效的 `expr_distance`
- 在某些情况下距离分支很强，说明潜在表示空间具备开放集区分能力

不足：

- 与现有 `expr_fused`、selective fusion、lineage rescue 参数体系不兼容
- `entropy` 信号与 `scanvi` 的行为差异很大
- 直接替换 backbone 后，原主线结论不再自动成立

### 7.2 数据集维度

#### immune_dc

- 更适合验证 `lineage_selective_rescue_globalT`
- lineage 结构较清晰
- GRN 解释层更容易形成稳定故事

#### pancreas

- backbone 本身可以工作
- 但现有 lineage rescue 规则跨数据集泛化不足
- 更像是在测试规则迁移能力，而不是验证原主线最优性

### 7.3 GRN / interpretability 维度

本轮结果支持一个较稳的结论：

- GRN 的主要价值更偏向解释层，而不是保证绝对指标持续上涨。
- 在最成功的组合 `scanvi + immune_dc` 上，GRN 能为未知细胞提供 lineage 语境和 regulon-level 解释。
- 在其他组合上，GRN 并未稳定带来指标收益，但仍然可以作为解释模块存在。

## 8. 最适合汇报的主结论

如果用于汇报 PPT，建议把主结论收敛到以下几条：

1. 我们已经在统一项目结构下完成了 `scanvi` 和 `scnym` 在两个数据集上的完整主线对比。
2. `scanvi + immune_dc` 仍然是当前最强、最稳定、最符合原始方法假设的组合。
3. `scanvi + pancreas` 说明 backbone 没问题，但 lineage rescue 规则不应直接跨数据集照搬。
4. `scnym` 成功接入主流程，但它改变了 expression-side 统计结构，不能直接套用 `scanvi` 时代的 fusion / rescue 参数。
5. GRN 的稳定价值在于增强未知细胞识别的生物学可解释性，而不是在所有组合中都保证数值增益。

## 9. PPT 建议结构

建议汇报按下面顺序组织：

1. 研究问题
   - 为什么要比较 backbone
   - 为什么需要同时看 immune_dc 和 pancreas

2. 方法总览
   - backbone -> expression artifacts -> GRN -> selective fusion -> lineage rescue

3. 实验设计
   - 2 个 backbone
   - 2 个数据集
   - 统一 split 与统一评估指标

4. immune_dc 结果
   - 主表：scanvi vs scnym
   - 图：`selective_roc`, `lineage_roc`, `lineage_bucket_barplot`
   - 结论：`scanvi + immune_dc` 仍是最佳主线

5. pancreas 结果
   - 主表：scanvi vs scnym
   - 图：`selective_roc`, `lineage_bucket_barplot`
   - 结论：后段 rescue 规则泛化不足

6. 综合比较
   - backbone 是否可替换
   - 哪些模块对 backbone 敏感
   - 哪些模块可以保持稳定

7. 后续计划
   - 保持 `scanvi + immune_dc` 为主干
   - 若继续推进 `scnym`，需先重配 expression fusion
   - 若继续推进 pancreas，需重新设计 lineage rescue 规则

## 10. 后续建议

### 10.1 作为当前主干

优先推荐：

- `scanvi + immune_dc + lineage_selective_rescue_globalT`

这是当前最完整、最稳定、也最适合讲完整方法故事的一组。

### 10.2 如果继续做 scnym

优先级建议：

1. 先重配 expression fusion
   - 不要默认保留 `0.5 * entropy + 0.5 * distance`
2. 重新评估 selective gate 规则
3. 再考虑是否保留现有 lineage rescue 结构

### 10.3 如果继续做 pancreas

优先级建议：

1. 不要直接套用 immune_dc 的 lineage rescue 规则
2. 先做 pancreas-specific corridor / alpha / threshold 校准
3. 再判断 lineage rescue 在 pancreas 上是否值得保留

## 11. 相关结果文件

### 11.1 immune_dc

- `outputs/runs/scanvi/immune_dc_lsr_globalT/`
- `outputs/runs/scnym/immune_dc_lsr_globalT/`

### 11.2 pancreas

- `outputs/runs/scanvi/pancreas_lsr_globalT/`
- `outputs/runs/scnym/pancreas_lsr_globalT/`

### 11.3 重点图表

immune_dc：

- `outputs/runs/scanvi/immune_dc_lsr_globalT/plots/selective_roc.png`
- `outputs/runs/scanvi/immune_dc_lsr_globalT/plots/lineage_roc.png`
- `outputs/runs/scanvi/immune_dc_lsr_globalT/plots/lineage_bucket_barplot.png`
- `outputs/runs/scnym/immune_dc_lsr_globalT/plots/selective_roc.png`
- `outputs/runs/scnym/immune_dc_lsr_globalT/plots/lineage_roc.png`
- `outputs/runs/scnym/immune_dc_lsr_globalT/plots/lineage_bucket_barplot.png`

pancreas：

- `outputs/runs/scanvi/pancreas_lsr_globalT/plots/selective_roc.png`
- `outputs/runs/scanvi/pancreas_lsr_globalT/plots/lineage_roc.png`
- `outputs/runs/scanvi/pancreas_lsr_globalT/plots/lineage_bucket_barplot.png`
- `outputs/runs/scnym/pancreas_lsr_globalT/plots/selective_roc.png`
- `outputs/runs/scnym/pancreas_lsr_globalT/plots/lineage_roc.png`
- `outputs/runs/scnym/pancreas_lsr_globalT/plots/lineage_bucket_barplot.png`
