from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scgrn.backbone.extract_expression_scores import apply_expression_combiner, fit_expression_combiner


class ExpressionLearningContractsTest(unittest.TestCase):
    def test_fit_expression_combiner_learns_deterministic_weights_from_val_known_errors(self) -> None:
        df = pd.DataFrame(
            {
                "split": ["train_known", "val_known", "val_known", "val_known", "val_known", "test_unknown"],
                "true_label": ["A", "A", "A", "B", "B", "U"],
                "predicted_label": ["A", "A", "B", "B", "A", "B"],
                "entropy_norm": [0.10, 0.10, 0.85, 0.20, 0.90, 0.95],
                "distance_norm": [0.10, 0.15, 0.80, 0.30, 0.88, 0.92],
                "expr_margin_unknown_score": [0.05, 0.10, 0.92, 0.15, 0.95, 0.97],
            }
        )

        model_a = fit_expression_combiner(df)
        model_b = fit_expression_combiner(df)

        self.assertEqual(
            model_a["feature_names"],
            ["entropy_norm", "distance_norm", "expr_margin_unknown_score"],
        )
        np.testing.assert_allclose(model_a["weights"], model_b["weights"])
        self.assertAlmostEqual(model_a["intercept"], model_b["intercept"])

        scored = apply_expression_combiner(df, model_a)
        self.assertIn("expr_fused", scored.columns)
        self.assertGreater(scored.loc[2, "expr_fused"], scored.loc[1, "expr_fused"])
        self.assertGreater(scored.loc[4, "expr_fused"], scored.loc[3, "expr_fused"])

    def test_apply_expression_combiner_requires_declared_features(self) -> None:
        df = pd.DataFrame({"entropy_norm": [0.1], "distance_norm": [0.2]})
        model = {
            "feature_names": ["entropy_norm", "distance_norm", "expr_margin_unknown_score"],
            "weights": [1.0, 1.0, 1.0],
            "intercept": 0.0,
        }

        with self.assertRaisesRegex(ValueError, "expr_margin_unknown_score"):
            apply_expression_combiner(df, model)


if __name__ == "__main__":
    unittest.main()
