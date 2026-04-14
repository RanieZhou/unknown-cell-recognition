from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scgrn.constants import SPLIT_COLUMN


TARGET_FILES = [
    ROOT / "src" / "scgrn" / "constants.py",
    ROOT / "src" / "scgrn" / "data" / "make_split.py",
    ROOT / "src" / "scgrn" / "backbone" / "base.py",
    ROOT / "src" / "scgrn" / "backbone" / "extract_expression_scores.py",
    ROOT / "src" / "scgrn" / "backbone" / "scnym_adapter.py",
    ROOT / "src" / "scgrn" / "pipeline" / "run_train.py",
    ROOT / "src" / "scgrn" / "pipeline" / "run_infer.py",
    ROOT / "src" / "scgrn" / "pipeline" / "run_rescue.py",
    ROOT / "src" / "scgrn" / "grn" / "build_grn_features.py",
    ROOT / "src" / "scgrn" / "grn" / "score_grn_distance.py",
    ROOT / "src" / "scgrn" / "rescue" / "threshold.py",
    ROOT / "src" / "scgrn" / "rescue" / "lineage_selective_rescue.py",
    ROOT / "src" / "scgrn" / "rescue" / "rescue_score.py",
    ROOT / "src" / "scgrn" / "evaluation" / "metrics.py",
    ROOT / "src" / "scgrn" / "evaluation" / "plots.py",
    ROOT / "src" / "scgrn" / "evaluation" / "reports.py",
]

TARGET_FILES_WITH_NO_EXPERIMENT_LABELS = [
    ROOT / "src" / "scgrn" / "pipeline" / "run_rescue.py",
    ROOT / "src" / "scgrn" / "rescue" / "lineage_selective_rescue.py",
    ROOT / "src" / "scgrn" / "evaluation" / "reports.py",
]

TARGET_FILES_WITH_NO_BUCKETT_MAINLINE = [
    ROOT / "README.md",
    ROOT / "CLAUDE.md",
    ROOT / "src" / "scgrn" / "constants.py",
    ROOT / "src" / "scgrn" / "pipeline" / "run_rescue.py",
    ROOT / "src" / "scgrn" / "rescue" / "lineage_selective_rescue.py",
    ROOT / "src" / "scgrn" / "evaluation" / "plots.py",
    ROOT / "src" / "scgrn" / "evaluation" / "reports.py",
    ROOT / "src" / "scgrn" / "evaluation" / "consistency.py",
]


class MainlineCleanupContractsTest(unittest.TestCase):
    def test_mainline_target_files_use_canonical_split_column(self) -> None:
        offenders = []
        for path in TARGET_FILES:
            text = path.read_text(encoding="utf-8")
            if "E005_split" in text:
                offenders.append(str(path))
        self.assertEqual(offenders, [], f"Found legacy E005_split references: {offenders}")

    def test_task1_target_files_do_not_reference_legacy_experiment_labels(self) -> None:
        offenders = []
        for path in TARGET_FILES_WITH_NO_EXPERIMENT_LABELS:
            text = path.read_text(encoding="utf-8")
            if "E006" in text or "E007" in text:
                offenders.append(str(path))
        self.assertEqual(offenders, [], f"Found legacy E006/E007 references: {offenders}")

    def test_split_constant_is_canonical_name(self) -> None:
        self.assertEqual(SPLIT_COLUMN, "split")

    def test_task6_mainline_target_files_do_not_reference_buckett_final_method(self) -> None:
        offenders = []
        for path in TARGET_FILES_WITH_NO_BUCKETT_MAINLINE:
            text = path.read_text(encoding="utf-8")
            if "bucketT" in text or "lineage_selective_rescue_bucketT" in text:
                offenders.append(str(path))
        self.assertEqual(offenders, [], f"Found bucketT mainline references: {offenders}")


if __name__ == "__main__":
    unittest.main()
