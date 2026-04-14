from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scgrn.paths import materialize_run_paths, resolve_run_paths


class PathsContractsTest(unittest.TestCase):
    def _write_config(self, root: Path) -> Path:
        config_dir = root / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "toy.yaml"
        config_path.write_text("project: {}\n", encoding="utf-8")
        return config_path

    def _config(self, config_path: Path) -> dict:
        return {
            "config_path": str(config_path),
            "project": {"output_root": "outputs/runs", "run_name": "demo_run"},
            "backbone": {"name": "scanvi"},
        }

    def test_resolve_run_paths_is_pure_and_does_not_touch_filesystem(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = self._write_config(root)
            config = self._config(config_path)

            paths = resolve_run_paths(config)

            self.assertFalse(paths.run_root.exists())
            self.assertFalse((paths.run_root / "config_snapshot.yaml").exists())
            self.assertEqual(paths.run_root, root / "outputs" / "runs" / "scanvi" / "demo_run")

    def test_materialize_run_paths_creates_directories_and_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = self._write_config(root)
            config = self._config(config_path)
            paths = resolve_run_paths(config)

            materialize_run_paths(paths, config)

            self.assertTrue(paths.logs.exists())
            self.assertTrue(paths.metrics.exists())
            self.assertTrue((paths.run_root / "config_snapshot.yaml").exists())


if __name__ == "__main__":
    unittest.main()
