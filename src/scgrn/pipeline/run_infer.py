"""Inference-stage entrypoint."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..backbone.base import ensure_expression_schema, validate_expression_artifacts_frame
from ..backbone.registry import get_backbone_adapter
from ..constants import EXPRESSION_SCHEMA_VERSION, SPLIT_COLUMN
from ..data.load_data import load_dataset
from ..data.make_split import load_saved_split_assignments, make_split
from ..grn.build_grn_features import export_pyscenic_input, run_pyscenic_python
from ..grn.score_grn_distance import score_grn_space
from ..rescue.lineage_selective_rescue import build_dual_fused_v2
from ..utils.io import write_dataframe, write_json
from ..utils.logger import setup_logger
from ..utils.seed import seed_everything


def _write_backbone_extra_artifacts(output_dir, artifacts: dict[str, object]) -> None:
    for filename, payload in artifacts.items():
        path = output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, pd.DataFrame):
            payload.to_csv(path, index=False)
        elif isinstance(payload, pd.Series):
            payload.to_frame().to_csv(path, index=False)
        elif isinstance(payload, np.ndarray):
            if path.suffix.lower() == ".npy":
                np.save(path, payload)
            else:
                pd.DataFrame(payload).to_csv(path, index=False)
        else:
            raise TypeError(f"Unsupported backbone artifact payload for {filename}: {type(payload)!r}")


def _load_expression_scores(paths, known_classes: list[str]) -> pd.DataFrame:
    expression_scores_path = paths.intermediate / "expression_scores.csv"
    if not expression_scores_path.exists():
        raise FileNotFoundError(
            f"Missing backbone output: {expression_scores_path}. "
            f"Run infer_backbone.py first, or use infer.py in a single environment."
        )
    expression_scores = pd.read_csv(expression_scores_path)
    expression_scores = ensure_expression_schema(expression_scores, known_classes)
    validate_expression_artifacts_frame(expression_scores, known_classes)
    return expression_scores


def run_infer_backbone(config: dict, paths, *, logger=None):
    logger = logger or setup_logger(paths.logs, "infer_backbone")
    seed_everything(int(config["training"]["seed"]))
    logger.info("Loading dataset for backbone inference stage")
    adata = load_dataset(config)
    split_path = paths.intermediate / "split_assignments.csv"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Missing split assignments: {split_path}. "
            "Run train.py first so backbone inference reuses the saved split contract."
        )
    logger.info("Reusing saved split assignments from %s", split_path)
    adata, train_idx, val_idx, test_known_idx, unknown_idx, split_summary = load_saved_split_assignments(
        adata,
        split_path,
        label_key=config["data"]["label_column"],
    )
    adapter = get_backbone_adapter(config)
    logger.info("Running backbone adapter: %s", adapter.name)
    predict_result = adapter.predict(adata, train_idx, val_idx, config, paths.checkpoints)
    expression_result = adapter.build_expression_artifacts(
        adata,
        predict_result,
        {
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_known_idx": test_known_idx,
            "unknown_idx": unknown_idx,
        },
        config,
    )
    validate_expression_artifacts_frame(expression_result.expression_scores, config["data"]["known_classes"])
    write_dataframe(paths.intermediate / "expression_scores.csv", expression_result.expression_scores)
    write_dataframe(paths.intermediate / "split_summary.csv", split_summary)
    split_index_df = pd.DataFrame({"cell_id": adata.obs_names.astype(str), SPLIT_COLUMN: adata.obs[SPLIT_COLUMN].values})
    write_dataframe(paths.intermediate / "split_assignments.csv", split_index_df)
    _write_backbone_extra_artifacts(paths.intermediate, predict_result.extra_artifacts)
    _write_backbone_extra_artifacts(paths.intermediate, expression_result.extra_artifacts)
    write_json(
        paths.metrics / "infer_backbone_stage_metadata.json",
        {
            "backbone_name": predict_result.backbone_name,
            "expression_schema_version": expression_result.schema_version,
            "expression_scores": str(paths.intermediate / "expression_scores.csv"),
            "split_summary": str(paths.intermediate / "split_summary.csv"),
        },
    )
    return {
        "adata": predict_result.prepared_adata,
        "adata_train": predict_result.training_adata,
        "backbone_name": predict_result.backbone_name,
        "expression_scores": expression_result.expression_scores,
    }


def run_infer_grn(config: dict, paths, *, logger=None):
    logger = logger or setup_logger(paths.logs, "infer_grn")
    logger.info("Loading dataset for GRN inference stage")
    adata = load_dataset(config)
    expression_scores = _load_expression_scores(paths, config["data"]["known_classes"])

    logger.info("Exporting pySCENIC input")
    export_paths = export_pyscenic_input(adata, expression_scores, config, paths.intermediate)
    logger.info("Running pySCENIC feature construction")
    pyscenic_artifacts = run_pyscenic_python(config, paths.intermediate, use_cached=bool(config["runtime"]["use_cached"]))
    logger.info("Scoring GRN space")
    grn_scores, auc_all, centroid_df = score_grn_space(
        adata,
        expression_scores,
        config,
        paths.intermediate,
        pyscenic_artifacts,
    )
    dual_fusion_df = build_dual_fused_v2(expression_scores.merge(grn_scores, on="cell_id", how="left"))

    write_dataframe(paths.intermediate / "grn_scores.csv", grn_scores)
    auc_all.to_csv(paths.intermediate / "aucell_all_cells.csv")
    centroid_df.to_csv(paths.intermediate / "regulon_centroids.csv")
    write_dataframe(paths.intermediate / "dual_fusion_scores.csv", dual_fusion_df)
    write_json(
        paths.metrics / "infer_stage_metadata.json",
        {
            "backbone_name": config["backbone"]["name"],
            "expression_schema_version": EXPRESSION_SCHEMA_VERSION,
            "expression_scores": str(paths.intermediate / "expression_scores.csv"),
            "grn_scores": str(paths.intermediate / "grn_scores.csv"),
            "aucell_all_cells": str(paths.intermediate / "aucell_all_cells.csv"),
            "regulon_centroids": str(paths.intermediate / "regulon_centroids.csv"),
            "train_known_expr_tsv": str(export_paths["tsv"]),
            "train_known_expr_loom": str(export_paths["loom"]),
        },
    )
    write_json(
        paths.metrics / "infer_grn_stage_metadata.json",
        {
            "backbone_name": config["backbone"]["name"],
            "expression_scores": str(paths.intermediate / "expression_scores.csv"),
            "grn_scores": str(paths.intermediate / "grn_scores.csv"),
            "aucell_all_cells": str(paths.intermediate / "aucell_all_cells.csv"),
            "regulon_centroids": str(paths.intermediate / "regulon_centroids.csv"),
            "train_known_expr_tsv": str(export_paths["tsv"]),
            "train_known_expr_loom": str(export_paths["loom"]),
        },
    )
    return {
        "adata": adata,
        "backbone_name": config["backbone"]["name"],
        "expression_scores": expression_scores,
        "grn_scores": grn_scores,
        "aucell_all": auc_all,
        "regulon_centroids": centroid_df,
        "dual_fusion_df": dual_fusion_df,
    }


def run_infer(config: dict, paths, *, logger=None):
    logger = logger or setup_logger(paths.logs, "infer")
    backbone_outputs = run_infer_backbone(config, paths, logger=logger)
    grn_outputs = run_infer_grn(config, paths, logger=logger)
    combined = dict(backbone_outputs)
    combined.update({k: v for k, v in grn_outputs.items() if k not in combined})
    return combined
