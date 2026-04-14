from __future__ import annotations

from pathlib import Path
import logging
import sys
import tempfile
import unittest
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scgrn.backbone.extract_expression_scores import compute_expression_scores, fit_expression_combiner
from scgrn.constants import SPLIT_COLUMN
from scgrn.grn.build_grn_features import export_pyscenic_input
from scgrn.pipeline.run_infer import run_infer_backbone
from scgrn.rescue.corridor import assign_lineage_buckets


class PipelineGuardsTest(unittest.TestCase):
    def test_compute_expression_scores_rejects_non_finite_or_zero_sum_probabilities(self) -> None:
        obs = pd.DataFrame(
            {
                "label": ["A", "A", "B", "B", "A", "B", "U"],
                SPLIT_COLUMN: [
                    "train_known",
                    "train_known",
                    "train_known",
                    "train_known",
                    "val_known",
                    "test_known",
                    "test_unknown",
                ],
            },
            index=[f"cell_{i}" for i in range(7)],
        )
        adata = ad.AnnData(X=np.ones((7, 2)), obs=obs)
        latent_all = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [1.0, 1.0],
                [1.1, 1.0],
                [0.2, 0.1],
                [1.2, 1.1],
                [0.5, 0.5],
            ],
            dtype=float,
        )
        soft_pred = pd.DataFrame(
            {
                "A": [0.9, 0.8, 0.2, 0.1, 0.0, 0.2, 0.5],
                "B": [0.1, 0.2, 0.8, 0.9, 0.0, 0.8, 0.5],
            },
            index=adata.obs_names,
        )
        config = {
            "data": {"known_classes": ["A", "B"], "label_column": "label"},
            "expression": {"combiner": "val_known_misclassification_logreg"},
        }

        with self.assertRaisesRegex(ValueError, "probability"):
            compute_expression_scores(
                adata,
                latent_all,
                soft_pred,
                train_idx=["cell_0", "cell_1", "cell_2", "cell_3"],
                val_idx=["cell_4"],
                test_known_idx=["cell_5"],
                unknown_idx=["cell_6"],
                config=config,
            )

    def test_compute_expression_scores_rejects_mahalanobis_fit_failure(self) -> None:
        obs = pd.DataFrame(
            {
                "label": ["A", "A", "B", "B", "A", "B", "U"],
                SPLIT_COLUMN: [
                    "train_known",
                    "train_known",
                    "train_known",
                    "train_known",
                    "val_known",
                    "test_known",
                    "test_unknown",
                ],
            },
            index=[f"cell_{i}" for i in range(7)],
        )
        adata = ad.AnnData(X=np.ones((7, 2)), obs=obs)
        latent_all = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [0.1, 0.1],
                [1.1, 1.1],
                [0.5, 0.5],
            ],
            dtype=float,
        )
        soft_pred = pd.DataFrame(
            {
                "A": [0.9, 0.8, 0.2, 0.1, 0.7, 0.2, 0.5],
                "B": [0.1, 0.2, 0.8, 0.9, 0.3, 0.8, 0.5],
            },
            index=adata.obs_names,
        )
        config = {
            "data": {"known_classes": ["A", "B"], "label_column": "label"},
            "expression": {"combiner": "val_known_misclassification_logreg"},
        }

        with patch("scgrn.backbone.extract_expression_scores.np.linalg.inv", side_effect=np.linalg.LinAlgError("singular")):
            with self.assertRaisesRegex(ValueError, "Mahalanobis"):
                compute_expression_scores(
                    adata,
                    latent_all,
                    soft_pred,
                    train_idx=["cell_0", "cell_1", "cell_2", "cell_3"],
                    val_idx=["cell_4"],
                    test_known_idx=["cell_5"],
                    unknown_idx=["cell_6"],
                    config=config,
                )

    def test_fit_expression_combiner_rejects_degenerate_val_known_targets(self) -> None:
        df = pd.DataFrame(
            {
                "split": ["train_known", "val_known", "val_known"],
                "true_label": ["A", "A", "B"],
                "predicted_label": ["A", "A", "B"],
                "entropy_norm": [0.1, 0.2, 0.3],
                "distance_norm": [0.1, 0.2, 0.3],
                "expr_margin_unknown_score": [0.1, 0.2, 0.3],
            }
        )

        with self.assertRaisesRegex(ValueError, "val_known"):
            fit_expression_combiner(df)

    def test_fit_expression_combiner_rejects_missing_val_known_rows(self) -> None:
        df = pd.DataFrame(
            {
                "split": ["train_known", "test_known"],
                "true_label": ["A", "B"],
                "predicted_label": ["A", "A"],
                "entropy_norm": [0.1, 0.2],
                "distance_norm": [0.1, 0.2],
                "expr_margin_unknown_score": [0.1, 0.2],
            }
        )

        with self.assertRaisesRegex(ValueError, "val_known"):
            fit_expression_combiner(df)

    def test_run_infer_backbone_requires_saved_split_assignments(self) -> None:
        class Paths:
            def __init__(self, root: Path):
                self.root = root
                self.logs = root / "logs"
                self.checkpoints = root / "checkpoints"
                self.intermediate = root / "intermediate"
                self.metrics = root / "metrics"
                for path in [self.logs, self.checkpoints, self.intermediate, self.metrics]:
                    path.mkdir(parents=True, exist_ok=True)

        config = {
            "training": {"seed": 42},
            "data": {
                "input_h5ad": "unused.h5ad",
                "label_column": "label",
                "batch_column": "batch",
                "counts_source": "X",
                "model_layer": "counts",
                "use_classes": ["A", "B", "U"],
                "known_classes": ["A", "B"],
                "unknown_classes": ["U"],
            },
            "backbone": {"name": "scanvi"},
            "runtime": {"use_cached": False},
        }
        obs = pd.DataFrame(
            {"label": ["A", "B", "U"], "batch": ["b1", "b1", "b1"]},
            index=["cell_0", "cell_1", "cell_2"],
        )
        adata = ad.AnnData(X=np.ones((3, 2)), obs=obs, var=pd.DataFrame(index=["gene_a", "gene_b"]))
        adata.layers["counts"] = adata.X.copy()
        logger = logging.getLogger("pipeline-guards")
        logger.handlers = [logging.NullHandler()]

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = Paths(Path(tmpdir))
            with patch("scgrn.pipeline.run_infer.load_dataset", return_value=adata), patch(
                "scgrn.pipeline.run_infer.make_split", side_effect=AssertionError("make_split should not be called")
            ):
                with self.assertRaisesRegex(FileNotFoundError, "split_assignments.csv"):
                    run_infer_backbone(config, paths, logger=logger)

    def test_assign_lineage_buckets_rejects_unknown_lineages(self) -> None:
        df = pd.DataFrame(
            {
                "predicted_label": ["A", "mystery"],
                "nearest_grn_class": ["B", "ghost"],
            }
        )

        with self.assertRaisesRegex(ValueError, "Unexpected lineage"):
            assign_lineage_buckets(df, ["A", "B"])

    def test_export_pyscenic_input_uses_current_feature_axis_not_raw_axis(self) -> None:
        obs = pd.DataFrame(index=["cell_1", "cell_2", "cell_3"])
        var = pd.DataFrame(index=["gene_a", "gene_b", "gene_c"])
        adata = ad.AnnData(X=np.ones((3, 3), dtype=np.float32), obs=obs, var=var)
        adata.layers["model_input"] = adata.X.copy()

        raw_var = pd.DataFrame(index=["raw_1", "raw_2", "raw_3", "raw_4"])
        raw_adata = ad.AnnData(X=np.ones((3, 4), dtype=np.float32), obs=obs.copy(), var=raw_var)
        adata.raw = raw_adata

        expression_scores = pd.DataFrame(
            {
                "cell_id": ["cell_1", "cell_2", "cell_3"],
                SPLIT_COLUMN: ["train_known", "train_known", "train_known"],
            }
        )
        config = {
            "data": {"model_layer": "model_input"},
            "grn": {"pyscenic": {"gene_filter_pct": 0.0}},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = export_pyscenic_input(adata, expression_scores, config, Path(tmpdir))
            exported = pd.read_csv(outputs["tsv"], sep="\t", index_col=0)

        self.assertEqual(list(exported.columns), ["gene_a", "gene_b", "gene_c"])


if __name__ == "__main__":
    unittest.main()
