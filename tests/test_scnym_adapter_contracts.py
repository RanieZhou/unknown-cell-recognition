from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scgrn.backbone.base import ensure_expression_schema, validate_expression_artifacts_frame

SCNYM_ADAPTER = ROOT / "src" / "scgrn" / "backbone" / "scnym_adapter.py"
SCANVI_ADAPTER = ROOT / "src" / "scgrn" / "backbone" / "scanvi_adapter.py"
BACKBONE_REGISTRY = ROOT / "src" / "scgrn" / "backbone" / "registry.py"


class AdapterModule:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def imported_names_from(self, module_name: str, *, level: int = 1) -> set[str]:
        imported: set[str] = set()
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != module_name or node.level != level:
                continue
            for alias in node.names:
                imported.add(alias.asname or alias.name)
        return imported

    def top_level_imported_names_from(self, module_name: str, *, level: int = 1) -> set[str]:
        imported: set[str] = set()
        for node in self.tree.body:
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != module_name or node.level != level:
                continue
            for alias in node.names:
                imported.add(alias.asname or alias.name)
        return imported

    def has_class_attr(self, class_name: str, attr_name: str) -> bool:
        for node in self.tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for stmt in node.body:
                    if not isinstance(stmt, ast.Assign):
                        continue
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and target.id == attr_name:
                            return True
        return False


class ScnymAdapterContractsTest(unittest.TestCase):
    def test_registry_uses_lazy_backbone_imports(self) -> None:
        module = AdapterModule(BACKBONE_REGISTRY)
        self.assertEqual(module.top_level_imported_names_from("scanvi_adapter"), set())
        self.assertEqual(module.top_level_imported_names_from("scnym_adapter"), set())
        self.assertEqual(module.imported_names_from("scanvi_adapter"), {"ScanviBackboneAdapter"})
        self.assertEqual(module.imported_names_from("scnym_adapter"), {"ScnymBackboneAdapter"})

    def test_scnym_does_not_import_scanvi_only_helpers(self) -> None:
        module = AdapterModule(SCNYM_ADAPTER)

        self.assertEqual(module.imported_names_from("train_backbone"), set())
        self.assertEqual(module.imported_names_from("infer_backbone"), set())
        self.assertEqual(
            module.imported_names_from("extract_expression_scores"),
            {"compute_expression_scores"},
        )
        self.assertIn("ensure_expression_schema", module.imported_names_from("base"))
        self.assertFalse(module.has_class_attr("ScnymBackboneAdapter", "upstream_dependencies"))
        self.assertFalse(module.has_class_attr("ScnymBackboneAdapter", "shared_expression_contract"))

    def test_scanvi_and_scnym_share_the_same_real_downstream_contract_imports(self) -> None:
        scanvi_module = AdapterModule(SCANVI_ADAPTER)
        scnym_module = AdapterModule(SCNYM_ADAPTER)

        self.assertEqual(
            scanvi_module.imported_names_from("extract_expression_scores"),
            {"compute_expression_scores"},
        )
        self.assertEqual(
            scnym_module.imported_names_from("extract_expression_scores"),
            {"compute_expression_scores"},
        )
        self.assertIn("ensure_expression_schema", scanvi_module.imported_names_from("base"))
        self.assertIn("ensure_expression_schema", scnym_module.imported_names_from("base"))
        self.assertIn("ExpressionArtifactsResult", scanvi_module.imported_names_from("base"))
        self.assertIn("ExpressionArtifactsResult", scnym_module.imported_names_from("base"))
        self.assertFalse(module_has_class_attr := scanvi_module.has_class_attr("ScanviBackboneAdapter", "shared_expression_contract"))
        self.assertFalse(scnym_module.has_class_attr("ScnymBackboneAdapter", "shared_expression_contract"))

    def test_shared_expression_schema_normalizes_backbone_specific_outputs(self) -> None:
        normalized = ensure_expression_schema(
            df=__import__("pandas").DataFrame(
                {
                    "cell_id": ["c1"],
                    "true_label": ["A"],
                    "split": ["test_unknown"],
                    "predicted_label": ["A"],
                    "entropy": [0.2],
                    "latent_distance": [0.5],
                    "expr_fused": [0.7],
                    "prob_A": [0.8],
                    "prob_B": [0.2],
                }
            ),
            known_classes=["A", "B"],
        )

        self.assertEqual(normalized.loc[0, "pred_label"], "A")
        self.assertEqual(normalized.loc[0, "expr_distance"], 0.5)
        self.assertEqual(normalized.loc[0, "expr_fused_score"], 0.7)
        self.assertEqual(normalized.loc[0, "max_prob"], 0.8)
        validate_expression_artifacts_frame(normalized, ["A", "B"], require_compat=False)


if __name__ == "__main__":
    unittest.main()
