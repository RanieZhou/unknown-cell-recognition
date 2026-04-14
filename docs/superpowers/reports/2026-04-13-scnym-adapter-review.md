# scNym adapter 结构审查报告

日期：2026-04-13

## 审查范围

本次仅审查以下主线代码中的 backbone adapter 结构与契约，不重写算法实现：

- `/d/Desktop/研一/code/scGRN/src/scgrn/backbone/scnym_adapter.py`
- `/d/Desktop/研一/code/scGRN/src/scgrn/backbone/scanvi_adapter.py`
- `/d/Desktop/研一/code/scGRN/src/scgrn/backbone/base.py`

同时新增回归测试，用于把本次结构结论固化为可执行契约：

- `/d/Desktop/研一/code/scGRN/tests/test_scnym_adapter_contracts.py`

## 结构结论

### 1. scNym 不是 scanvi adapter 的直接套壳

从代码结构看，`ScnymBackboneAdapter` 没有复用 scanvi 专属上游 helper：

- 没有导入 `train_backbone.py` 中的 `create_scanvi_labels`、`load_trained_backbone`、`train_backbone`
- 没有导入 `infer_backbone.py` 中的 `extract_predictions`
- 自己走独立上游路径：
  - 通过 `scanpy.pp.normalize_total` 与 `scanpy.pp.log1p` 做预处理
  - 通过 `sc.pp.highly_variable_genes` 选 HVG
  - 通过 `scnym.api.scnym_api` 分别执行 train / predict
  - 自己维护 `scnym_label`、`scNym_split`、`scNym_probabilities`、`X_scnym` 等键

相对地，`ScanviBackboneAdapter` 的上游明确依赖 scanvi 专属 helper：

- `create_scanvi_labels(...)`
- `train_backbone(...)`
- `load_trained_backbone(...)`
- `extract_predictions(...)`

因此，“scnym 只是 scanvi adapter 换皮”这个怀疑，从当前代码结构上看不成立。

### 2. 两者共享的是下游 expression artifact 契约

`scanvi` 与 `scnym` 在 `build_expression_artifacts(...)` 里共享相同的下游表达空间接口：

- `compute_expression_scores(...)`
- `ensure_expression_schema(...)`
- `ExpressionArtifactsResult`

也就是说，二者共享的是“如何把各自 backbone 输出整理成统一 expression artifact”的下游协议，而不是共享上游训练/预测内部实现。

## 对“效果差是否因为 scanvi 套壳”的判断

基于这次结构审查，不能把 scnym 效果问题归因于“它只是 scanvi 套壳”。更准确的判断是：

- 上游训练、预处理、特征选择、预测调用路径，scnym 是独立的
- 下游表达分数构建与 schema 标准化，scnym 与 scanvi 故意复用同一契约

因此，如果 scnym 效果较差，更应优先检查 scnym 自身路径上的设计与配置，而不是怀疑它直接复用了 scanvi 的训练/推理逻辑。

## 从当前结构能看出的、真正可能影响 scNym 效果的点

以下几点是从代码结构中可以直接观察到、且更可能影响 scnym 表现的地方：

1. 预处理路径与 scanvi 不同
   - scnym 强制把 `adata.X` 准备成 log1p(CPM)
   - 这条预处理链是否与数据来源、layer 选择、下游假设一致，会直接影响效果

2. HVG 选择只基于 train-known 细胞
   - `highly_variable_genes` 在 train-known 子集上计算
   - 这会决定 scnym 实际看到的特征空间，可能影响泛化到 unknown 的能力

3. unknown 细胞被显式标为 `Unlabeled`
   - 这符合半监督路径，但效果会受到 split 组成与 unlabeled 分布影响

4. scNym 自身配置覆盖项
   - `config_name`
   - `ssl_kwargs`
   - `model_kwargs`
   - `use_domain_labels`
   这些都是 scnym 特有的行为入口，比“scanvi 套壳”更值得排查

5. 下游统一 expression contract 也可能放大或掩盖 backbone 差异
   - 两个 adapter 最终都会进入同一个 `compute_expression_scores(...)`
   - 所以最终效果既受 scnym 上游影响，也受共享下游打分逻辑影响
   - 但这不等于 scnym 上游是 scanvi 的复制品

## 本次落地的契约固化

新增测试文件覆盖了两个核心事实：

1. `scnym_adapter.py` 不依赖 scanvi-only helper
2. `scnym` 与 `scanvi` 通过真实导入与返回类型共享同一个 expression artifact 下游契约
3. `ensure_expression_schema(...)` 能把 backbone 特定输出标准化到统一 schema

为避免测试退化成脆弱的字符串扫描，本次使用 AST 级 import 结构断言，并补充少量 schema 行为断言；没有把测试建立在额外的生产元数据字段之上。

## 非目标说明

本任务没有：

- 重写 scnym 算法
- 调整 scanvi/scnym 训练逻辑
- 改动主线实验配置
- 对效果问题做实验性归因

本任务只完成了结构审查与契约固化。
