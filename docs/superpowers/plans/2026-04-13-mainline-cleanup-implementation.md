# Mainline Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 收口 `src/scgrn` 主线，去掉历史实验残留和无需求分支，改成默认 fail-fast、只保留 `globalT`、并把 expression 主线权重从手工固定改成校准集学习式组合。

**Architecture:** 先做主线公共 schema 与命名清理，再删除 `scgpt` 并同步文档，随后收紧 split 与数值方法的 fail-fast 契约。方法层面只保留 `globalT` 作为最终阈值策略，并把 expression 主线组合改成基于 `val_known` 误分类目标的浅层逻辑回归，保留 `scanvi` / `scnym` 统一下游 schema，但允许 `scnym` 保持自己的内部准备逻辑。

**Tech Stack:** Python, pandas, numpy, scikit-learn, AnnData/scanpy, unittest, pySCENIC, scVI-tools/scNym adapters

---

## File Structure

### Core code to modify

- `src/scgrn/constants.py`
  - 定义主线方法名、支持的 backbone、split 常量、方法排序。
- `src/scgrn/config.py`
  - 删除 `scgpt` 相关占位与旧默认 run 命名，保持配置校验与主线语义一致。
- `src/scgrn/data/make_split.py`
  - 统一 split 列名到主线语义；读取旧 split 文件时只在边缘层做一次映射。
- `src/scgrn/backbone/base.py`
  - 更新标准 expression schema，去掉历史字段名作为主线必需列。
- `src/scgrn/backbone/extract_expression_scores.py`
  - 去掉 fixed 0.5/0.5 组合；增加基于 `val_known` 的 expression learning helper；对 Mahalanobis 失败改为报错。
- `src/scgrn/backbone/registry.py`
  - 物理删除 `scgpt` 分支。
- `src/scgrn/backbone/scnym_adapter.py`
  - 改成依赖 canonical split 列；保留 scNym 自己的准备逻辑；为审查报告提供事实依据。
- `src/scgrn/grn/build_grn_features.py`
  - 改成使用 canonical split 列筛选 `train_known`。
- `src/scgrn/grn/score_grn_distance.py`
  - 改成使用 canonical split 列筛选 `train_known`。
- `src/scgrn/pipeline/run_train.py`
  - 写出 canonical split 字段，不再生成历史列名。
- `src/scgrn/pipeline/run_infer.py`
  - backbone inference 阶段缺少 `split_assignments.csv` 时直接失败；写出 canonical split 字段。
- `src/scgrn/pipeline/run_rescue.py`
  - 去掉 `E006/E007` 文案与 `bucketT` 输出；把主线报告聚焦到 `globalT`。
- `src/scgrn/rescue/rescue_score.py`
  - 如果 expression 学习式组合落在 rescue 层，这里放 helper；否则只保留与主线 score 一致的字段使用。
- `src/scgrn/rescue/threshold.py`
  - 删除仅服务 `bucketT` 最终阈值的 helper，保留 gate/bucket 分析真正还需要的函数。
- `src/scgrn/rescue/lineage_selective_rescue.py`
  - 删除 `bucketT` 作为最终方法的整条路径，只保留 `lineage_selective_rescue_globalT`。
- `src/scgrn/evaluation/reports.py`
  - summary 文案去掉 `E006/E007` 与 `bucketT` 主结果描述，改成当前主线语义。

### Docs to modify

- `README.md`
  - 更新 split 字段、主线输出、`globalT` 唯一结果、expression learning 说明。
- `CLAUDE.md`
  - 同步主线说明；修掉仍写着 `prepare_run_paths(...)` 和 `scgpt` 扩展点的过时描述。

### New docs to create

- `docs/superpowers/reports/2026-04-13-scnym-adapter-review.md`
  - 记录 `scnym` adapter 审查结论：哪些是合理的统一接口，哪些仍受旧 split/schema 假设影响，是否存在 scanvi-only helper 耦合。

### Tests to modify or create

- Create: `tests/test_mainline_cleanup_contracts.py`
  - 静态约束：`src/scgrn` 中不再出现 `E00X`、`scgpt`、`lineage_selective_rescue_bucketT`。
- Modify: `tests/test_split_contracts.py`
  - 验证 legacy split 列读取只在边缘层映射为 canonical `split`；新产物只写 `split`。
- Modify: `tests/test_pipeline_guards.py`
  - 增加 Mahalanobis 失败直接报错；增加 expression learning target 退化时直接报错。
- Modify: `tests/test_cli_contracts.py`
  - 不再测试“`scgpt` 在 validation 阶段拒绝”，改成“只要 backbone 不在支持集合里就直接拒绝”。
- Create: `tests/test_expression_learning_contracts.py`
  - 验证 expression 学习式组合使用 `val_known` 误分类目标，并产出确定性的权重。
- Create: `tests/test_scnym_adapter_contracts.py`
  - 验证 `scnym_adapter.py` 不再依赖历史 split 字段名，并记录 review 中确认的接口边界。

---

### Task 1: Canonical split schema and historical naming cleanup

**Files:**
- Modify: `src/scgrn/constants.py`
- Modify: `src/scgrn/data/make_split.py`
- Modify: `src/scgrn/backbone/base.py`
- Modify: `src/scgrn/pipeline/run_train.py`
- Modify: `src/scgrn/pipeline/run_infer.py`
- Modify: `src/scgrn/grn/build_grn_features.py`
- Modify: `src/scgrn/grn/score_grn_distance.py`
- Modify: `src/scgrn/rescue/lineage_selective_rescue.py`
- Modify: `src/scgrn/rescue/rescue_score.py`
- Modify: `src/scgrn/evaluation/reports.py`
- Create: `tests/test_mainline_cleanup_contracts.py`
- Modify: `tests/test_split_contracts.py`

- [ ] **Step 1: Write the failing tests**

Add a new static contract test file that fails on any remaining `E00X` marker inside `src/scgrn`, and extend split tests so legacy CSV input is normalized to `split` immediately.

```python
# tests/test_mainline_cleanup_contracts.py
from __future__ import annotations

from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "scgrn"


class MainlineCleanupContractsTest(unittest.TestCase):
    def test_src_scgrn_no_longer_contains_e00x_markers(self) -> None:
        offenders: list[str] = []
        pattern = re.compile(r"E00[0-9]")
        for path in SRC.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if pattern.search(text):
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [], f"Found historical experiment markers: {offenders}")
```

```python
# tests/test_split_contracts.py
    def test_load_saved_split_assignments_normalizes_legacy_column_to_split(self) -> None:
        adata = self._make_adata()
        assignments = pd.DataFrame(
            {
                "cell_id": ["cell_a", "cell_b", "cell_c", "cell_d"],
                "E005_split": ["train_known", "val_known", "test_known", "test_unknown"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            split_path = Path(tmpdir) / "split_assignments.csv"
            assignments.to_csv(split_path, index=False)
            load_saved_split_assignments(adata, split_path, label_key="label")

        self.assertIn("split", adata.obs.columns)
        self.assertNotIn("E005_split", adata.obs.columns)
        self.assertEqual(adata.obs.loc["cell_d", "split"], "test_unknown")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python tests/test_mainline_cleanup_contracts.py
python tests/test_split_contracts.py
```

Expected:
- `test_src_scgrn_no_longer_contains_e00x_markers` fails and lists files such as `src/scgrn/data/make_split.py` and `src/scgrn/rescue/lineage_selective_rescue.py`
- `test_load_saved_split_assignments_normalizes_legacy_column_to_split` fails because current code writes/reads `E005_split`

- [ ] **Step 3: Write the minimal implementation**

Introduce canonical split naming in one place and use it everywhere.

```python
# src/scgrn/constants.py
METHOD_NAME = "lineage_selective_rescue_globalT"
DEFAULT_BACKBONE_NAME = "scanvi"
SUPPORTED_BACKBONES = ("scanvi", "scnym")
EXPRESSION_SCHEMA_VERSION = "backbone_expression_v1"

SPLIT_COLUMN = "split"
LEGACY_SPLIT_COLUMNS = ("E005_split",)
SPLIT_TRAIN = "train_known"
SPLIT_VAL = "val_known"
SPLIT_TEST_KNOWN = "test_known"
SPLIT_TEST_UNKNOWN = "test_unknown"
```

```python
# src/scgrn/data/make_split.py
from ..constants import (
    LEGACY_SPLIT_COLUMNS,
    SPLIT_COLUMN,
    SPLIT_TEST_KNOWN,
    SPLIT_TEST_UNKNOWN,
    SPLIT_TRAIN,
    SPLIT_VAL,
)


def _normalize_split_column(split_df: pd.DataFrame) -> pd.DataFrame:
    if SPLIT_COLUMN in split_df.columns:
        return split_df.copy()
    for legacy_name in LEGACY_SPLIT_COLUMNS:
        if legacy_name in split_df.columns:
            renamed = split_df.copy()
            renamed[SPLIT_COLUMN] = renamed.pop(legacy_name)
            return renamed
    raise ValueError(
        f"Split assignment file must contain '{SPLIT_COLUMN}' or one of {list(LEGACY_SPLIT_COLUMNS)}"
    )


def _indices_from_split_column(adata):
    train_idx = adata.obs_names[adata.obs[SPLIT_COLUMN] == SPLIT_TRAIN].tolist()
    val_idx = adata.obs_names[adata.obs[SPLIT_COLUMN] == SPLIT_VAL].tolist()
    test_known_idx = adata.obs_names[adata.obs[SPLIT_COLUMN] == SPLIT_TEST_KNOWN].tolist()
    unknown_idx = adata.obs_names[adata.obs[SPLIT_COLUMN] == SPLIT_TEST_UNKNOWN].tolist()
    return train_idx, val_idx, test_known_idx, unknown_idx
```

```python
# src/scgrn/pipeline/run_train.py and src/scgrn/pipeline/run_infer.py
from ..constants import SPLIT_COLUMN

split_index_df = pd.DataFrame(
    {
        "cell_id": adata.obs_names.astype(str),
        SPLIT_COLUMN: adata.obs[SPLIT_COLUMN].values,
    }
)
```

```python
# src/scgrn/grn/build_grn_features.py, score_grn_distance.py, rescue/*.py, evaluation/reports.py
from ..constants import SPLIT_COLUMN

train_known_cells = expression_scores.loc[
    expression_scores[SPLIT_COLUMN] == "train_known", "cell_id"
].astype(str).tolist()
```

Update `src/scgrn/backbone/base.py` so `STANDARD_EXPRESSION_COLUMNS` uses `split`, and only `ensure_expression_schema()` may map legacy `E005_split` into `split` when reading old artifacts.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python tests/test_mainline_cleanup_contracts.py
python tests/test_split_contracts.py
```

Expected:
- Both files end with `OK`
- No file under `src/scgrn` still contains `E00X`
- Legacy split CSV still loads, but mainline in-memory and newly written outputs use `split`

- [ ] **Step 5: Commit**

```bash
git add tests/test_mainline_cleanup_contracts.py tests/test_split_contracts.py src/scgrn/constants.py src/scgrn/data/make_split.py src/scgrn/backbone/base.py src/scgrn/pipeline/run_train.py src/scgrn/pipeline/run_infer.py src/scgrn/grn/build_grn_features.py src/scgrn/grn/score_grn_distance.py src/scgrn/rescue/lineage_selective_rescue.py src/scgrn/rescue/rescue_score.py src/scgrn/evaluation/reports.py
git commit -m "refactor: replace legacy split naming in mainline"
```

---

### Task 2: Remove `scgpt` from the maintained mainline and sync docs

**Files:**
- Modify: `src/scgrn/config.py`
- Modify: `src/scgrn/backbone/registry.py`
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `tests/test_cli_contracts.py`
- Modify: `tests/test_mainline_cleanup_contracts.py`

- [ ] **Step 1: Write the failing tests**

Replace the `scgpt`-specific validation test with a generic unsupported-backbone test, and extend the static cleanup contract to fail if `scgpt` still appears anywhere under `src/scgrn`.

```python
# tests/test_cli_contracts.py
    def test_unknown_backbone_is_rejected_during_config_validation(self) -> None:
        config = load_config(ROOT / "configs" / "default.yaml")
        config["backbone"]["name"] = "made_up_backbone"

        with self.assertRaisesRegex(ValueError, "Unsupported backbone.name"):
            validate_config(config)
```

```python
# tests/test_mainline_cleanup_contracts.py
    def test_src_scgrn_no_longer_mentions_scgpt(self) -> None:
        offenders: list[str] = []
        for path in SRC.rglob("*.py"):
            text = path.read_text(encoding="utf-8").lower()
            if "scgpt" in text:
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [], f"Found removed backbone references: {offenders}")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python tests/test_cli_contracts.py
python tests/test_mainline_cleanup_contracts.py
```

Expected:
- `test_src_scgrn_no_longer_mentions_scgpt` fails on `src/scgrn/config.py` and `src/scgrn/backbone/registry.py`
- `test_unknown_backbone_is_rejected_during_config_validation` passes only after the old `scgpt`-special-case branch is gone

- [ ] **Step 3: Write the minimal implementation**

Delete the placeholder branch instead of keeping a future-extension stub.

```python
# src/scgrn/config.py
from .constants import DEFAULT_BACKBONE_NAME, METHOD_NAME, SUPPORTED_BACKBONES

# delete UNIMPLEMENTED_BACKBONES

backbone_cfg["name"] = str(backbone_cfg["name"]).strip().lower()
if backbone_cfg["name"] not in SUPPORTED_BACKBONES:
    raise ValueError(f"Unsupported backbone.name: {backbone_cfg['name']}")
```

```python
# src/scgrn/backbone/registry.py
def get_backbone_adapter(config: dict):
    backbone_name = get_backbone_name(config)
    if backbone_name == "scanvi":
        return ScanviBackboneAdapter()
    if backbone_name == "scnym":
        return ScnymBackboneAdapter()
    raise ValueError(f"Unsupported backbone '{backbone_name}'")
```

```markdown
# README.md / CLAUDE.md
- 删除任何 “scgpt 未来扩展点” 描述
- 明确当前只维护 `scanvi` 与 `scnym`
- 在 `CLAUDE.md` 中把 “call scgrn.paths.prepare_run_paths(...)” 改回当前真实行为：
  1. `resolve_run_paths(...)`
  2. `materialize_run_paths(...)`
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python tests/test_cli_contracts.py
python tests/test_mainline_cleanup_contracts.py
```

Expected:
- `OK`
- `src/scgrn` 中不再有 `scgpt` 文本残留
- `README.md` 与 `CLAUDE.md` 的 backbone 支持范围与代码一致

- [ ] **Step 5: Commit**

```bash
git add tests/test_cli_contracts.py tests/test_mainline_cleanup_contracts.py src/scgrn/config.py src/scgrn/backbone/registry.py README.md CLAUDE.md
git commit -m "refactor: drop scgpt placeholder from mainline"
```

---

### Task 3: Enforce fail-fast contracts for missing split artifacts and Mahalanobis fitting

**Files:**
- Modify: `src/scgrn/pipeline/run_infer.py`
- Modify: `src/scgrn/backbone/extract_expression_scores.py`
- Modify: `tests/test_split_contracts.py`
- Modify: `tests/test_pipeline_guards.py`

- [ ] **Step 1: Write the failing tests**

Add one test that requires `run_infer_backbone()` to fail when `split_assignments.csv` is missing, and one test that requires expression scoring to fail when class-level precision matrices cannot be fit reliably.

```python
# tests/test_split_contracts.py
from types import SimpleNamespace
from unittest import mock
from scgrn.pipeline.run_infer import run_infer_backbone

    def test_run_infer_backbone_requires_saved_split_assignments(self) -> None:
        paths = SimpleNamespace(
            intermediate=Path("missing_intermediate"),
            logs=Path("logs"),
            checkpoints=Path("checkpoints"),
            metrics=Path("metrics"),
        )
        config = {
            "training": {"seed": 42},
            "data": {"label_column": "label", "known_classes": ["A", "B"]},
        }
        with mock.patch("scgrn.pipeline.run_infer.load_dataset") as load_dataset:
            load_dataset.return_value = self._make_adata()
            with self.assertRaisesRegex(FileNotFoundError, "split_assignments.csv"):
                run_infer_backbone(config, paths, logger=mock.Mock())
```

```python
# tests/test_pipeline_guards.py
    def test_compute_expression_scores_rejects_unstable_mahalanobis_fit(self) -> None:
        obs = pd.DataFrame(
            {
                "label": ["A", "B", "A"],
                "split": ["train_known", "train_known", "val_known"],
            },
            index=["cell_0", "cell_1", "cell_2"],
        )
        adata = ad.AnnData(X=np.ones((3, 2)), obs=obs)
        latent_all = np.array([[0.0, 0.0], [1.0, 1.0], [0.2, 0.1]], dtype=float)
        soft_pred = pd.DataFrame({"A": [0.8, 0.2, 0.7], "B": [0.2, 0.8, 0.3]}, index=adata.obs_names)
        config = {"data": {"known_classes": ["A", "B"], "label_column": "label"}, "expression": {}}

        with self.assertRaisesRegex(ValueError, "Mahalanobis"):
            compute_expression_scores(
                adata,
                latent_all,
                soft_pred,
                train_idx=["cell_0", "cell_1"],
                val_idx=["cell_2"],
                test_known_idx=[],
                unknown_idx=[],
                config=config,
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python tests/test_split_contracts.py
python tests/test_pipeline_guards.py
```

Expected:
- missing-split test fails because current `run_infer_backbone()` silently regenerates the split
- Mahalanobis test fails because current code falls back to Euclidean

- [ ] **Step 3: Write the minimal implementation**

Make the backbone inference stage require the saved split artifact, and extract Mahalanobis fitting into a helper that raises instead of downgrading the metric.

```python
# src/scgrn/pipeline/run_infer.py
split_path = paths.intermediate / "split_assignments.csv"
if not split_path.exists():
    raise FileNotFoundError(
        f"Missing required split artifact: {split_path}. "
        "Run train.py first so backbone inference reuses the exact saved split contract."
    )
logger.info("Reusing saved split assignments from %s", split_path)
adata, train_idx, val_idx, test_known_idx, unknown_idx, split_summary = load_saved_split_assignments(
    adata,
    split_path,
    label_key=config["data"]["label_column"],
)
```

```python
# src/scgrn/backbone/extract_expression_scores.py
def _fit_precision_matrices(train_latent, train_labels, known_classes):
    centroids: dict[str, np.ndarray] = {}
    cov_invs: dict[str, np.ndarray] = {}
    for cls in known_classes:
        cls_latent = train_latent[train_labels == cls]
        if cls_latent.shape[0] < 2:
            raise ValueError(f"Mahalanobis fit requires at least 2 train-known cells for class '{cls}'")
        cov = np.cov(cls_latent, rowvar=False)
        if not np.isfinite(cov).all():
            raise ValueError(f"Mahalanobis covariance contains non-finite values for class '{cls}'")
        rank = np.linalg.matrix_rank(cov)
        if rank < cov.shape[0]:
            raise ValueError(f"Mahalanobis covariance is singular for class '{cls}'")
        centroids[cls] = cls_latent.mean(axis=0)
        cov_invs[cls] = np.linalg.inv(cov)
    return centroids, cov_invs
```

Use `_fit_precision_matrices(...)` inside `compute_expression_scores()` and remove the `use_mahalanobis` / Euclidean fallback branch completely.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python tests/test_split_contracts.py
python tests/test_pipeline_guards.py
```

Expected:
- `OK`
- backbone inference without saved split fails immediately
- unstable Mahalanobis fitting fails immediately with a clear `ValueError`

- [ ] **Step 5: Commit**

```bash
git add tests/test_split_contracts.py tests/test_pipeline_guards.py src/scgrn/pipeline/run_infer.py src/scgrn/backbone/extract_expression_scores.py
git commit -m "fix: make split and distance contracts fail fast"
```

---

### Task 4: Review `scnym` adapter and record the findings

**Files:**
- Modify: `src/scgrn/backbone/scnym_adapter.py`
- Create: `docs/superpowers/reports/2026-04-13-scnym-adapter-review.md`
- Create: `tests/test_scnym_adapter_contracts.py`

- [ ] **Step 1: Write the failing test**

Create a contract test that fails if `scnym_adapter.py` still references the legacy split field directly.

```python
# tests/test_scnym_adapter_contracts.py
from __future__ import annotations

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCNYM_PATH = ROOT / "src" / "scgrn" / "backbone" / "scnym_adapter.py"


class ScnymAdapterContractsTest(unittest.TestCase):
    def test_scnym_adapter_uses_canonical_split_column(self) -> None:
        source = SCNYM_PATH.read_text(encoding="utf-8")
        self.assertNotIn("E005_split", source)
        self.assertIn("SPLIT_COLUMN", source)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python tests/test_scnym_adapter_contracts.py
```

Expected:
- test fails because current adapter still reads `prepared.obs["E005_split"]`

- [ ] **Step 3: Write the minimal implementation and review report**

Update the adapter to use the canonical split constant, then write a short review report with concrete conclusions.

```python
# src/scgrn/backbone/scnym_adapter.py
from ..constants import SPLIT_COLUMN, SPLIT_TEST_UNKNOWN

prepared.obs.loc[
    prepared.obs[SPLIT_COLUMN] == SPLIT_TEST_UNKNOWN,
    self.label_key,
] = self.unlabeled_category

split_map = {
    "train_known": "train",
    "val_known": "val",
    "test_known": "test",
    "test_unknown": "target",
}
prepared.obs[self.split_key] = prepared.obs[SPLIT_COLUMN].map(split_map).fillna("target").astype(str)
```

```markdown
# docs/superpowers/reports/2026-04-13-scnym-adapter-review.md
# scNym Adapter Review

## Findings

1. `scnym_adapter.py` does not call any scanvi-only helper such as `create_scanvi_labels()` or `train_backbone()`.
2. The adapter had one real mainline leak: it still depended on the historical split field name instead of the canonical split schema.
3. The adapter shares `compute_expression_scores(...)` with `scanvi`, which is acceptable as a downstream artifact builder so long as the schema remains shared and the backbone-specific preparation stays separate.

## Conclusion

The current scNym path is not a thin copy of the scanvi training stack. The real cleanup needed here is schema decoupling and explicit documentation of what is shared: downstream artifact construction, not upstream training internals.
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python tests/test_scnym_adapter_contracts.py
```

Expected:
- `OK`
- review report exists at `docs/superpowers/reports/2026-04-13-scnym-adapter-review.md`

- [ ] **Step 5: Commit**

```bash
git add tests/test_scnym_adapter_contracts.py src/scgrn/backbone/scnym_adapter.py docs/superpowers/reports/2026-04-13-scnym-adapter-review.md
git commit -m "docs: record scnym adapter review findings"
```

---

### Task 5: Replace fixed expression weights with validation-learned combination

**Files:**
- Modify: `src/scgrn/backbone/extract_expression_scores.py`
- Modify: `src/scgrn/constants.py`
- Modify: `configs/default.yaml`
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Create: `tests/test_expression_learning_contracts.py`
- Modify: `tests/test_pipeline_guards.py`

- [ ] **Step 1: Write the failing tests**

Add a focused test file for the new learning helper and a guard for degenerate validation targets.

```python
# tests/test_expression_learning_contracts.py
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scgrn.backbone.extract_expression_scores import fit_expression_combiner, apply_expression_combiner


class ExpressionLearningContractsTest(unittest.TestCase):
    def test_fit_expression_combiner_learns_deterministic_weights_from_val_known_errors(self) -> None:
        df = pd.DataFrame(
            {
                "split": ["val_known", "val_known", "val_known", "val_known"],
                "true_label": ["A", "A", "B", "B"],
                "predicted_label": ["A", "B", "B", "A"],
                "entropy_norm": [0.05, 0.90, 0.10, 0.85],
                "distance_norm": [0.10, 0.80, 0.15, 0.75],
                "expr_margin_unknown_score": [0.10, 0.70, 0.20, 0.90],
            }
        )

        model = fit_expression_combiner(df)
        scores = apply_expression_combiner(df, model)

        self.assertGreater(scores.iloc[1], scores.iloc[0])
        self.assertGreater(scores.iloc[3], scores.iloc[2])
        self.assertEqual(sorted(model.keys()), ["feature_names", "intercept", "weights"])
```

```python
# tests/test_pipeline_guards.py
    def test_fit_expression_combiner_rejects_degenerate_validation_target(self) -> None:
        df = pd.DataFrame(
            {
                "split": ["val_known", "val_known"],
                "true_label": ["A", "B"],
                "predicted_label": ["A", "B"],
                "entropy_norm": [0.1, 0.2],
                "distance_norm": [0.2, 0.3],
                "expr_margin_unknown_score": [0.1, 0.2],
            }
        )

        with self.assertRaisesRegex(ValueError, "val_known"):
            fit_expression_combiner(df)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python tests/test_expression_learning_contracts.py
python tests/test_pipeline_guards.py
```

Expected:
- import error or missing-function failure because the learning helpers do not exist yet
- degenerate-target test fails because no validation-target check exists

- [ ] **Step 3: Write the minimal implementation**

Fit expression uncertainty on `val_known` only, using misclassification on known validation cells as the supervised target.

```python
# src/scgrn/backbone/extract_expression_scores.py
from sklearn.linear_model import LogisticRegression
from ..constants import SPLIT_COLUMN, SPLIT_VAL

EXPR_COMBINER_FEATURES = [
    "entropy_norm",
    "distance_norm",
    "expr_margin_unknown_score",
]


def fit_expression_combiner(df: pd.DataFrame) -> dict[str, object]:
    val_df = df[df[SPLIT_COLUMN] == SPLIT_VAL].copy()
    if val_df.empty:
        raise ValueError("No val_known cells available for expression combiner fitting")
    y = (val_df["predicted_label"].astype(str) != val_df["true_label"].astype(str)).astype(int)
    if y.nunique() < 2:
        raise ValueError("val_known misclassification target is degenerate; cannot fit expression combiner")
    X = val_df[EXPR_COMBINER_FEATURES].to_numpy(dtype=float)
    model = LogisticRegression(solver="liblinear", class_weight="balanced", random_state=0)
    model.fit(X, y.to_numpy())
    return {
        "feature_names": list(EXPR_COMBINER_FEATURES),
        "weights": model.coef_[0].astype(float).tolist(),
        "intercept": float(model.intercept_[0]),
    }


def apply_expression_combiner(df: pd.DataFrame, model: dict[str, object]) -> pd.Series:
    X = df[list(model["feature_names"])].to_numpy(dtype=float)
    weights = np.asarray(model["weights"], dtype=float)
    logits = X @ weights + float(model["intercept"])
    return pd.Series(1.0 / (1.0 + np.exp(-logits)), index=df.index, name="expr_fused")
```

```python
# same file, inside compute_expression_scores(...)
expr_margin = np.sort(probs, axis=1)[:, -1] - np.sort(probs, axis=1)[:, -2]
expr_margin_unknown_score = 1.0 - expr_margin

scores_df = pd.DataFrame(
    {
        "cell_id": adata.obs_names.astype(str),
        "true_label": adata.obs[label_key].values,
        "split": adata.obs[SPLIT_COLUMN].values,
        "predicted_label": soft_pred[known_classes].idxmax(axis=1).values,
        "nearest_known_class": nearest_class,
        "entropy": entropy,
        "entropy_norm": entropy_norm,
        "latent_distance": distances,
        "distance_norm": distance_norm,
        "expr_margin": expr_margin,
        "expr_margin_unknown_score": expr_margin_unknown_score,
    }
)
combiner = fit_expression_combiner(scores_df)
scores_df["expr_fused"] = apply_expression_combiner(scores_df, combiner)
```

```yaml
# configs/default.yaml
expression:
  combiner: logistic_val_error
  distance_metric: mahalanobis_min_known
```

Delete `entropy_weight` and `distance_weight` from `configs/default.yaml`, `README.md`, and `CLAUDE.md`.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python tests/test_expression_learning_contracts.py
python tests/test_pipeline_guards.py
```

Expected:
- `OK`
- learned expression scores are deterministic for the same inputs
- degenerate validation targets fail explicitly instead of silently reverting to fixed weights

- [ ] **Step 5: Commit**

```bash
git add tests/test_expression_learning_contracts.py tests/test_pipeline_guards.py src/scgrn/backbone/extract_expression_scores.py src/scgrn/constants.py configs/default.yaml README.md CLAUDE.md
git commit -m "feat: learn expression fusion weights from val-known errors"
```

---

### Task 6: Remove `bucketT` as a final mainline result and keep only `globalT`

**Files:**
- Modify: `src/scgrn/constants.py`
- Modify: `src/scgrn/rescue/threshold.py`
- Modify: `src/scgrn/rescue/lineage_selective_rescue.py`
- Modify: `src/scgrn/pipeline/run_rescue.py`
- Modify: `src/scgrn/evaluation/reports.py`
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `tests/test_mainline_cleanup_contracts.py`

- [ ] **Step 1: Write the failing test**

Extend the static cleanup contract so the task cannot finish while `bucketT` still exists in the maintained mainline.

```python
# tests/test_mainline_cleanup_contracts.py
    def test_src_scgrn_no_longer_contains_buckett_mainline(self) -> None:
        offenders: list[str] = []
        for path in SRC.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "lineage_selective_rescue_bucketT" in text or "bucketT" in text:
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [], f"Found removed bucket-threshold path: {offenders}")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python tests/test_mainline_cleanup_contracts.py
```

Expected:
- test fails on `src/scgrn/constants.py`, `src/scgrn/rescue/lineage_selective_rescue.py`, `src/scgrn/pipeline/run_rescue.py`, and `src/scgrn/evaluation/reports.py`

- [ ] **Step 3: Write the minimal implementation**

Delete `bucketT` as a final-method path, but keep bucket-based gate analysis only where it still supports lineage gating and per-bucket metrics.

```python
# src/scgrn/constants.py
DEFAULT_METHOD_ORDER = [
    "expr_fused",
    "selective_fused_score",
    "lineage_selective_rescue_globalT",
    "grn_distance",
    "grn_aux_score",
    "dual_fused_v2_rankavg",
    "v3_ml_anomaly",
]
```

```python
# src/scgrn/rescue/lineage_selective_rescue.py
# delete the duplicated bucketT score column
# df["lineage_selective_rescue_bucketT"] = df["lineage_selective_rescue_score"]

global_methods = [
    ("expr_fused", "expr_fused"),
    ("selective_fused_score", "selective_fused_score"),
    ("grn_distance", "grn_distance"),
    ("dual_fused_v2_rankavg", "dual_fused_v2_rankavg"),
    ("lineage_selective_rescue_globalT", "lineage_selective_rescue_score"),
]

for method_name, score_col in global_methods:
    threshold = compute_threshold_on_val(df.loc[df[SPLIT_COLUMN] == SPLIT_VAL, score_col].to_numpy(), 95)
    pred_col = f"{method_name}_pred"
    df = predict_with_global_threshold(df, score_col, threshold, pred_col)
```

```python
# src/scgrn/pipeline/run_rescue.py
plot_roc_curves(
    lineage["df"],
    [
        ("expr_fused", "expr_fused", lineage["metrics_by_method"].get("expr_fused", {}).get("AUROC", float("nan"))),
        ("selective_fused_score", "selective_fused_score", lineage["metrics_by_method"].get("selective_fused_score", {}).get("AUROC", float("nan"))),
        ("lineage_selective_rescue_globalT", "lineage_selective_rescue_globalT", lineage["metrics_by_method"].get("lineage_selective_rescue_globalT", {}).get("AUROC", float("nan"))),
    ],
    paths.plots / "lineage_roc.png",
)
```

```markdown
# src/scgrn/evaluation/reports.py / README.md / CLAUDE.md
- 删除 “bucketT vs globalT” 作为最终主结果的并列描述
- E007 summary 改成只总结 `lineage_selective_rescue_globalT`
- bucket 只作为分析维度，不再作为最终阈值策略名称
```

If `predict_with_bucket_threshold()` becomes unused after this cleanup, delete it from `src/scgrn/rescue/threshold.py` in the same commit.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python tests/test_mainline_cleanup_contracts.py
```

Expected:
- `OK`
- no maintained `src/scgrn` file still mentions `bucketT`
- docs only describe `lineage_selective_rescue_globalT` as the final result

- [ ] **Step 5: Commit**

```bash
git add tests/test_mainline_cleanup_contracts.py src/scgrn/constants.py src/scgrn/rescue/threshold.py src/scgrn/rescue/lineage_selective_rescue.py src/scgrn/pipeline/run_rescue.py src/scgrn/evaluation/reports.py README.md CLAUDE.md
git commit -m "refactor: keep only global threshold mainline"
```

---

## Final verification checklist

After Task 6, run all touched regression files individually:

```bash
python tests/test_mainline_cleanup_contracts.py
python tests/test_cli_contracts.py
python tests/test_split_contracts.py
python tests/test_pipeline_guards.py
python tests/test_expression_learning_contracts.py
python tests/test_scnym_adapter_contracts.py
python tests/test_paths_contracts.py
python tests/test_entrypoint_paths_usage.py
```

Expected:
- every file ends with `OK`
- `src/scgrn` contains no `E00X`, `scgpt`, or `bucketT`
- backbone inference fails immediately when the saved split artifact is missing
- expression scoring fails immediately when Mahalanobis fitting or validation-target learning is invalid
- docs and code both describe a single maintained final result: `lineage_selective_rescue_globalT`

## Spec coverage self-check

- **模块 A：命名与主线清理** → Task 1
- **模块 B：删除 `scgpt`** → Task 2
- **模块 C：fail-fast 收紧** → Task 3
- **模块 D1：`scnym` 审查** → Task 4
- **模块 D2：expression 学习式组合** → Task 5
- **模块 D3：只保留 `globalT`** → Task 6

No spec section is left without a task.
