"""Backbone adapter registry."""

from __future__ import annotations

from ..constants import DEFAULT_BACKBONE_NAME



def get_backbone_name(config: dict) -> str:
    backbone_cfg = config.get("backbone", {})
    training_cfg = config.get("training", {})
    name = backbone_cfg.get("name") or training_cfg.get("backbone_type") or DEFAULT_BACKBONE_NAME
    return str(name).strip().lower()


def get_backbone_adapter(config: dict):
    backbone_name = get_backbone_name(config)
    if backbone_name == "scanvi":
        from .scanvi_adapter import ScanviBackboneAdapter

        return ScanviBackboneAdapter()
    if backbone_name == "scnym":
        from .scnym_adapter import ScnymBackboneAdapter

        return ScnymBackboneAdapter()
    raise ValueError(f"Unsupported backbone '{backbone_name}'")
