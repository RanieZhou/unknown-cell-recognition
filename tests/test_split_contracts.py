from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scgrn.backbone.base import STANDARD_EXPRESSION_COLUMNS, ensure_expression_schema
from scgrn.constants import SPLIT_COLUMN
from scgrn.data.make_split import load_saved_split_assignments


class SplitContractsTest(unittest.TestCase):
    def _make_adata(self):
        obs = pd.DataFrame(
            {
                "label": ["cDC2", "pDC", "cDC1", "ASDC"],
            },
            index=["cell_a", "cell_b", "cell_c", "cell_d"],
        )
        return ad.AnnData(X=np.ones((4, 2)), obs=obs)

    def test_load_saved_split_assignments_restores_saved_membership_from_canonical_split(self) -> None:
        adata = self._make_adata()
        assignments = pd.DataFrame(
            {
                "cell_id": ["cell_a", "cell_b", "cell_c", "cell_d"],
                SPLIT_COLUMN: ["train_known", "val_known", "test_known", "test_unknown"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            split_path = Path(tmpdir) / "split_assignments.csv"
            assignments.to_csv(split_path, index=False)
            _, train_idx, val_idx, test_known_idx, unknown_idx, summary = load_saved_split_assignments(
                adata,
                split_path,
                label_key="label",
            )

        self.assertEqual(train_idx, ["cell_a"])
        self.assertEqual(val_idx, ["cell_b"])
        self.assertEqual(test_known_idx, ["cell_c"])
        self.assertEqual(unknown_idx, ["cell_d"])
        self.assertEqual(adata.obs.loc["cell_d", SPLIT_COLUMN], "test_unknown")
        self.assertEqual(int(summary["count"].sum()), 4)

    def test_load_saved_split_assignments_maps_legacy_e005_split_only_at_boundary(self) -> None:
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
            _, train_idx, val_idx, test_known_idx, unknown_idx, summary = load_saved_split_assignments(
                adata,
                split_path,
                label_key="label",
            )

        self.assertEqual(train_idx, ["cell_a"])
        self.assertEqual(val_idx, ["cell_b"])
        self.assertEqual(test_known_idx, ["cell_c"])
        self.assertEqual(unknown_idx, ["cell_d"])
        self.assertIn(SPLIT_COLUMN, adata.obs.columns)
        self.assertNotIn("E005_split", adata.obs.columns)
        self.assertEqual(list(summary.columns), [SPLIT_COLUMN, "label", "count"])

    def test_expression_schema_promotes_canonical_split_column(self) -> None:
        df = pd.DataFrame(
            {
                "cell_id": ["cell_a"],
                "true_label": ["cDC2"],
                "E005_split": ["train_known"],
                "pred_label": ["cDC2"],
                "predicted_label": ["cDC2"],
                "max_prob": [0.9],
                "entropy": [0.1],
                "entropy_norm": [0.1],
                "expr_distance": [0.2],
                "latent_distance": [0.2],
                "distance_norm": [0.2],
                "expr_fused_score": [0.3],
                "expr_fused": [0.3],
                "prob_cDC2": [0.9],
            }
        )

        normalized = ensure_expression_schema(df, ["cDC2"])

        self.assertIn(SPLIT_COLUMN, normalized.columns)
        self.assertNotIn("E005_split", normalized.columns)
        self.assertEqual(normalized.loc[0, SPLIT_COLUMN], "train_known")
        self.assertEqual(STANDARD_EXPRESSION_COLUMNS[2], SPLIT_COLUMN)

    def test_load_saved_split_assignments_rejects_missing_cells(self) -> None:
        adata = self._make_adata()
        assignments = pd.DataFrame(
            {
                "cell_id": ["cell_a", "cell_b", "cell_c"],
                SPLIT_COLUMN: ["train_known", "val_known", "test_known"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            split_path = Path(tmpdir) / "split_assignments.csv"
            assignments.to_csv(split_path, index=False)
            with self.assertRaisesRegex(ValueError, "missing cells"):
                load_saved_split_assignments(adata, split_path, label_key="label")

    def test_load_saved_split_assignments_rejects_duplicate_cell_ids(self) -> None:
        adata = self._make_adata()
        assignments = pd.DataFrame(
            {
                "cell_id": ["cell_a", "cell_a", "cell_c", "cell_d"],
                SPLIT_COLUMN: ["train_known", "val_known", "test_known", "test_unknown"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            split_path = Path(tmpdir) / "split_assignments.csv"
            assignments.to_csv(split_path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_saved_split_assignments(adata, split_path, label_key="label")

    def test_load_saved_split_assignments_rejects_invalid_split_labels(self) -> None:
        adata = self._make_adata()
        assignments = pd.DataFrame(
            {
                "cell_id": ["cell_a", "cell_b", "cell_c", "cell_d"],
                SPLIT_COLUMN: ["train_known", "val_known", "oops", "test_unknown"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            split_path = Path(tmpdir) / "split_assignments.csv"
            assignments.to_csv(split_path, index=False)
            with self.assertRaisesRegex(ValueError, "Invalid split labels"):
                load_saved_split_assignments(adata, split_path, label_key="label")


if __name__ == "__main__":
    unittest.main()
