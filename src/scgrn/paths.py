"""Path helpers for standardized run outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RunPaths:
    project_root: Path
    config_path: Path
    output_root: Path
    backbone_root: Path
    run_root: Path
    logs: Path
    checkpoints: Path
    intermediate: Path
    predictions: Path
    metrics: Path
    plots: Path
    reports: Path


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def infer_project_root(config_path: str | Path) -> Path:
    config_path = Path(config_path).resolve()
    return config_path.parent.parent


def resolve_run_paths(config: dict) -> RunPaths:
    config_path = Path(config["config_path"]).resolve()
    project_root = infer_project_root(config_path)
    output_root = (project_root / config["project"]["output_root"]).resolve()
    backbone_name = str(config["backbone"]["name"]).strip().lower()
    backbone_root = output_root / backbone_name
    run_root = backbone_root / config["project"]["run_name"]

    return RunPaths(
        project_root=project_root,
        config_path=config_path,
        output_root=output_root,
        backbone_root=backbone_root,
        run_root=run_root,
        logs=run_root / "logs",
        checkpoints=run_root / "checkpoints",
        intermediate=run_root / "intermediate",
        predictions=run_root / "predictions",
        metrics=run_root / "metrics",
        plots=run_root / "plots",
        reports=run_root / "reports",
    )


def materialize_run_paths(paths: RunPaths, config: dict) -> RunPaths:
    materialized = RunPaths(
        project_root=paths.project_root,
        config_path=paths.config_path,
        output_root=_ensure(paths.output_root),
        backbone_root=_ensure(paths.backbone_root),
        run_root=_ensure(paths.run_root),
        logs=_ensure(paths.logs),
        checkpoints=_ensure(paths.checkpoints),
        intermediate=_ensure(paths.intermediate),
        predictions=_ensure(paths.predictions),
        metrics=_ensure(paths.metrics),
        plots=_ensure(paths.plots),
        reports=_ensure(paths.reports),
    )
    snapshot_path = materialized.run_root / "config_snapshot.yaml"
    with snapshot_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    return materialized


def prepare_run_paths(config: dict) -> RunPaths:
    return materialize_run_paths(resolve_run_paths(config), config)
