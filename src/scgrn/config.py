"""Configuration loading and validation for the refactored project."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .constants import DEFAULT_BACKBONE_NAME, METHOD_NAME, SUPPORTED_BACKBONES


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = config
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _merge_missing(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        target.setdefault(key, deepcopy(value))


def _normalize_alias_sections(config: dict[str, Any]) -> None:
    project_cfg = config.setdefault("project", {})
    data_cfg = config.setdefault("data", {})
    training_cfg = config.setdefault("training", {})
    rescue_cfg = config.setdefault("rescue", {})

    dataset_cfg = config.get("dataset", {})
    if isinstance(dataset_cfg, dict) and dataset_cfg.get("name") and not project_cfg.get("dataset_name"):
        project_cfg["dataset_name"] = dataset_cfg["name"]

    train_cfg = config.get("train", {})
    if isinstance(train_cfg, dict):
        hyper_cfg = train_cfg.get("hyperparameters", {})
        if isinstance(hyper_cfg, dict):
            _merge_missing(training_cfg, hyper_cfg)
        _merge_missing(training_cfg, {k: v for k, v in train_cfg.items() if k != "hyperparameters"})

    output_cfg = config.get("output", {})
    if isinstance(output_cfg, dict) and output_cfg.get("dir") and not project_cfg.get("output_root"):
        project_cfg["output_root"] = output_cfg["dir"]

    if data_cfg.get("input_path") and not data_cfg.get("input_h5ad"):
        data_cfg["input_h5ad"] = data_cfg["input_path"]

    backbone_cfg = config.setdefault("backbone", {})
    if not backbone_cfg.get("name"):
        backbone_cfg["name"] = training_cfg.get("backbone_type", DEFAULT_BACKBONE_NAME)

    rescue_cfg.setdefault("corridor_rule", "priority_union")
    rescue_cfg.setdefault(
        "threshold_rule",
        f"global_percentile_{config.get('threshold', {}).get('percentile', 95)}",
    )


def _normalize_classes(config: dict[str, Any]) -> None:
    data_cfg = config["data"]
    known = [str(item) for item in data_cfg.get("known_classes", [])]
    unknown = [str(item) for item in data_cfg.get("unknown_classes", [])]
    if not known:
        raise ValueError("data.known_classes must not be empty")
    if not unknown:
        raise ValueError("data.unknown_classes must not be empty")
    data_cfg["known_classes"] = known
    data_cfg["unknown_classes"] = unknown
    if data_cfg.get("use_classes") in [None, []]:
        data_cfg["use_classes"] = known + unknown
    else:
        data_cfg["use_classes"] = [str(item) for item in data_cfg["use_classes"]]


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    _normalize_alias_sections(config)
    required_sections = [
        "project",
        "data",
        "backbone",
        "training",
        "expression",
        "grn",
        "rescue",
        "threshold",
        "runtime",
    ]
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise ValueError(f"Missing config sections: {missing_sections}")

    project_cfg = config["project"]
    data_cfg = config["data"]
    backbone_cfg = config["backbone"]
    train_cfg = config["training"]

    if not project_cfg.get("dataset_name"):
        raise ValueError("project.dataset_name is required")
    if not data_cfg.get("input_h5ad"):
        raise ValueError("data.input_h5ad is required")
    if not data_cfg.get("label_column"):
        raise ValueError("data.label_column is required")
    if not data_cfg.get("batch_column"):
        raise ValueError("data.batch_column is required")
    if not backbone_cfg.get("name"):
        raise ValueError("backbone.name is required")
    if not isinstance(train_cfg.get("seed"), int):
        raise ValueError("training.seed must be an integer")

    _normalize_classes(config)

    backbone_cfg["name"] = str(backbone_cfg["name"]).strip().lower()
    if backbone_cfg["name"] not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported backbone.name: {backbone_cfg['name']}")

    rescue_cfg = config["rescue"]
    alpha_map = rescue_cfg.get("lineage_alpha_map", {})
    if not isinstance(alpha_map, dict) or not alpha_map:
        raise ValueError("rescue.lineage_alpha_map must be a non-empty mapping")
    rescue_cfg.setdefault("corridor_priority", data_cfg["known_classes"])

    runtime_cfg = config["runtime"]
    runtime_cfg.setdefault("use_cached", False)
    runtime_cfg.setdefault("resume", False)
    runtime_cfg.setdefault("save_plots", True)
    runtime_cfg.setdefault("save_reports", True)

    project_cfg.setdefault("method_name", METHOD_NAME)
    project_cfg.setdefault("run_name", f"{project_cfg['dataset_name']}_lsr_globalT")
    project_cfg.setdefault("output_root", "outputs/runs")
    train_cfg.setdefault("backbone_type", backbone_cfg["name"])

    data_cfg.setdefault("counts_source", "raw")
    data_cfg.setdefault("model_layer", "counts")
    data_cfg.setdefault("train_known_ratio", 0.70)
    data_cfg.setdefault("val_known_ratio", 0.15)
    data_cfg.setdefault("test_known_ratio", 0.15)

    config["config_path"] = str(config.get("config_path", ""))
    return config


def load_config(
    config_path: str | Path,
    *,
    seed: int | None = None,
    output_dir: str | None = None,
    use_cached: bool | None = None,
    resume: bool | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    default_path = config_path.parent / "default.yaml"

    base_cfg = _read_yaml(default_path) if config_path.name != "default.yaml" else {}
    current_cfg = _read_yaml(config_path)
    config = _deep_merge(base_cfg, current_cfg)
    config["config_path"] = str(config_path)

    if seed is not None:
        _set_nested(config, "training.seed", seed)
    if output_dir is not None:
        _set_nested(config, "project.output_root", output_dir)
    if use_cached is not None:
        _set_nested(config, "runtime.use_cached", use_cached)
    if resume is not None:
        _set_nested(config, "runtime.resume", resume)

    return validate_config(config)
