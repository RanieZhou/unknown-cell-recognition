"""scGRN mainline package centered on lineage_selective_rescue_globalT."""

from .config import load_config
from .paths import prepare_run_paths

__all__ = ["load_config", "prepare_run_paths"]
