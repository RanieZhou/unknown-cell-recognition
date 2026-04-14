from __future__ import annotations

import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = [
    ROOT / "train.py",
    ROOT / "infer.py",
    ROOT / "infer_backbone.py",
    ROOT / "infer_grn.py",
    ROOT / "rescue.py",
    ROOT / "evaluate.py",
    ROOT / "run_pipeline.py",
]


def imported_names(script_path: Path) -> set[str]:
    tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "scgrn.paths":
            names.update(alias.name for alias in node.names)
    return names


def function_calls(script_path: Path) -> list[str]:
    tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                calls.append(func.id)
    return calls


class EntrypointPathsUsageTest(unittest.TestCase):
    def test_entrypoints_use_explicit_resolve_and_materialize_steps(self) -> None:
        for script_path in ENTRYPOINTS:
            names = imported_names(script_path)
            calls = function_calls(script_path)
            self.assertIn("resolve_run_paths", names, script_path.name)
            self.assertIn("materialize_run_paths", names, script_path.name)
            self.assertIn("resolve_run_paths", calls, script_path.name)
            self.assertIn("materialize_run_paths", calls, script_path.name)
            self.assertNotIn("prepare_run_paths", names, script_path.name)
            self.assertNotIn("prepare_run_paths", calls, script_path.name)


if __name__ == "__main__":
    unittest.main()
