"""scANVI adapter for the standardized backbone interface."""

from __future__ import annotations

import pandas as pd

from .base import (
    BackboneAdapter,
    BackbonePredictResult,
    BackboneTrainResult,
    ExpressionArtifactsResult,
    ensure_expression_schema,
)
from .extract_expression_scores import compute_expression_scores
from .infer_backbone import extract_predictions
from .train_backbone import create_scanvi_labels, load_trained_backbone, train_backbone


class ScanviBackboneAdapter(BackboneAdapter):
    """Adapter that preserves the current validated scANVI mainline."""

    name = "scanvi"
    label_key = "scanvi_label"

    def _prepare_adata(self, adata, train_idx, config: dict):
        return create_scanvi_labels(
            adata,
            train_idx,
            label_key=config["data"]["label_column"],
            seed=int(config["training"]["seed"]),
            unlabel_frac=float(config["training"]["unlabel_frac"]),
        )

    def train(self, adata, train_idx, val_idx, config: dict, checkpoints_dir, *, use_cached: bool = False) -> BackboneTrainResult:
        prepared_adata = self._prepare_adata(adata, train_idx, config)
        checkpoint_path = checkpoints_dir / "scanvi_model"
        if use_cached and checkpoint_path.exists():
            model, adata_train = load_trained_backbone(prepared_adata, train_idx, val_idx, config, checkpoints_dir)
        else:
            model, adata_train = train_backbone(prepared_adata, train_idx, val_idx, config, checkpoints_dir)
        return BackboneTrainResult(
            backbone_name=self.name,
            model=model,
            prepared_adata=prepared_adata,
            training_adata=adata_train,
            metadata={"label_key": self.label_key},
        )

    def predict(self, adata, train_idx, val_idx, config: dict, checkpoints_dir) -> BackbonePredictResult:
        prepared_adata = self._prepare_adata(adata, train_idx, config)
        model, adata_train = load_trained_backbone(prepared_adata, train_idx, val_idx, config, checkpoints_dir)
        latent_all, soft_pred, hard_pred = extract_predictions(model, prepared_adata)
        return BackbonePredictResult(
            backbone_name=self.name,
            model=model,
            prepared_adata=prepared_adata,
            training_adata=adata_train,
            latent_representation=latent_all,
            soft_predictions=soft_pred,
            hard_predictions=hard_pred,
            extra_artifacts={
                "scanvi_latent_all.npy": latent_all,
                "scanvi_soft_predictions.csv": pd.DataFrame(soft_pred).reset_index(drop=True),
            },
            metadata={"label_key": self.label_key},
        )

    def build_expression_artifacts(self, adata, predict_output: BackbonePredictResult, split_context: dict, config: dict) -> ExpressionArtifactsResult:
        expression_scores, _, centroids = compute_expression_scores(
            predict_output.prepared_adata,
            predict_output.latent_representation,
            predict_output.soft_predictions,
            split_context["train_idx"],
            split_context["val_idx"],
            split_context["test_known_idx"],
            split_context["unknown_idx"],
            config,
        )
        expression_scores = ensure_expression_schema(expression_scores, config["data"]["known_classes"])
        return ExpressionArtifactsResult(
            backbone_name=self.name,
            expression_scores=expression_scores,
            centroids=centroids,
            extra_artifacts={
                "latent_cell_order.csv": pd.DataFrame({"cell_id": predict_output.prepared_adata.obs_names.astype(str)})
            },
            metadata={"label_key": self.label_key},
        )
