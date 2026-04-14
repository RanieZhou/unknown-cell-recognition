from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scgrn.config import load_config, validate_config

SRC_SCGRN = ROOT / "src" / "scgrn"


ROOT_SCRIPTS_WITHOUT_RESUME = [
    ROOT / "train.py",
    ROOT / "infer.py",
    ROOT / "infer_backbone.py",
    ROOT / "infer_grn.py",
    ROOT / "rescue.py",
    ROOT / "run_pipeline.py",
]

ROOT_SCRIPTS_WITH_USE_CACHED = [
    ROOT / "train.py",
    ROOT / "infer.py",
    ROOT / "infer_grn.py",
    ROOT / "run_pipeline.py",
]


def parse_declared_flags(script_path: Path) -> set[str]:
    tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
    captured: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                captured.add(arg.value)
    return captured


class CliContractsTest(unittest.TestCase):
    def test_unknown_backbone_is_rejected_during_config_validation(self) -> None:
        config = load_config(ROOT / "configs" / "default.yaml")
        config["backbone"]["name"] = "unknown_backbone"

        with self.assertRaisesRegex(ValueError, "unknown_backbone"):
            validate_config(config)

    def test_src_scgrn_no_longer_mentions_scgpt(self) -> None:
        offenders = []
        for path in SRC_SCGRN.rglob("*.py"):
            if "scgpt" in path.read_text(encoding="utf-8").lower():
                offenders.append(str(path))
        self.assertEqual(offenders, [], f"Found scgpt references under src/scgrn: {offenders}")

    def test_infer_backbone_cli_does_not_expose_resume_or_use_cached(self) -> None:
        flags = parse_declared_flags(ROOT / "infer_backbone.py")
        self.assertNotIn("--resume", flags)
        self.assertNotIn("--use_cached", flags)

    def test_rescue_cli_does_not_expose_resume_or_use_cached(self) -> None:
        flags = parse_declared_flags(ROOT / "rescue.py")
        self.assertNotIn("--resume", flags)
        self.assertNotIn("--use_cached", flags)

    def test_all_runtime_clis_drop_resume_until_real_resume_behavior_exists(self) -> None:
        for script_path in ROOT_SCRIPTS_WITHOUT_RESUME:
            flags = parse_declared_flags(script_path)
            self.assertNotIn("--resume", flags, f"Did not expect --resume in {script_path.name}")

    def test_supported_clis_keep_use_cached_only_where_it_has_real_behavior(self) -> None:
        for script_path in ROOT_SCRIPTS_WITH_USE_CACHED:
            flags = parse_declared_flags(script_path)
            self.assertIn("--use_cached", flags, f"Expected --use_cached in {script_path.name}")


if __name__ == "__main__":
    unittest.main()
