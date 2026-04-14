"""scNym adapter following the official semi-supervised API path."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from ..constants import SPLIT_COLUMN, SPLIT_TEST_UNKNOWN
from .base import (
    BackboneAdapter,
    BackbonePredictResult,
    BackboneTrainResult,
    ExpressionArtifactsResult,
    ensure_expression_schema,
)
from .extract_expression_scores import compute_expression_scores


class ScnymBackboneAdapter(BackboneAdapter):
    """Adapter that uses official scNym semi-supervised training and prediction."""

    name = "scnym"
    label_key = "scnym_label"
    split_key = "scNym_split"
    prediction_key = "scNym"
    probability_key = "scNym_probabilities"
    embedding_key = "X_scnym"
    confidence_key = "scNym_confidence"
    domain_key = "scnym_domain"
    unlabeled_category = "Unlabeled"

    def _require_scnym(self):
        try:
            from scnym.api import CONFIGS, scnym_api
        except Exception as exc:
            raise ImportError(
                "scNym is not installed in the current environment. "
                "Run scNym backbone stages inside the `scnym310` environment."
            ) from exc
        return CONFIGS, scnym_api

    def _checkpoint_dir(self, checkpoints_dir: Path) -> Path:
        return checkpoints_dir / "scnym_model"

    def _has_cached_model(self, model_dir: Path) -> bool:
        return (model_dir / "00_best_model_weights.pkl").exists() and (
            model_dir / "scnym_train_results.pkl"
        ).exists()

    def _scnym_cfg(self, config: dict) -> dict:
        overrides = deepcopy(config.get("backbone", {}).get("scnym", {}))
        config_name = str(overrides.pop("config_name", "new_identity_discovery"))
        CONFIGS, _ = self._require_scnym()
        if config_name not in CONFIGS:
            raise ValueError(
                f"Unsupported scNym config_name='{config_name}'. "
                f"Available predefined configs: {sorted(CONFIGS.keys())}"
            )
        cfg = deepcopy(CONFIGS[config_name])

        ssl_overrides = deepcopy(overrides.pop("ssl_kwargs", {}))
        model_overrides = deepcopy(overrides.pop("model_kwargs", {}))
        if model_overrides:
            cfg.setdefault("model_kwargs", {}).update(model_overrides)
        if ssl_overrides:
            cfg.setdefault("ssl_kwargs", {}).update(ssl_overrides)
        for key, value in overrides.items():
            cfg[key] = value

        if "seed" not in cfg:
            cfg["seed"] = int(config["training"]["seed"])
        return cfg

    def _prepare_adata(self, adata, train_idx, config: dict):
        prepared = adata.copy()
        prepared.var_names_make_unique()

        model_layer = str(config["data"]["model_layer"])
        if model_layer in prepared.layers:
            prepared.X = prepared.layers[model_layer].copy()
        else:
            prepared.X = prepared.X.copy()

        # scNym API expects log1p(CPM) in adata.X.
        sc.pp.normalize_total(prepared, target_sum=1e6)
        sc.pp.log1p(prepared)

        n_top_genes = int(
            config.get("backbone", {}).get("scnym", {}).get("n_top_genes", 2500)
        )
        train_mask = prepared.obs_names.isin(train_idx)
        train_adata = prepared[train_mask].copy()
        sc.pp.highly_variable_genes(
            train_adata,
            n_top_genes=min(n_top_genes, prepared.n_vars),
        )
        hvg_mask = np.array(train_adata.var["highly_variable"]).astype(bool)
        selected_genes = np.array(train_adata.var_names)[hvg_mask]
        if len(selected_genes) == 0:
            raise RuntimeError("No HVGs were selected from train-known cells for scNym.")
        prepared = prepared[:, selected_genes].copy()

        label_col = str(config["data"]["label_column"])
        prepared.obs[self.label_key] = prepared.obs[label_col].astype(str)
        prepared.obs.loc[
            prepared.obs[SPLIT_COLUMN] == SPLIT_TEST_UNKNOWN,
            self.label_key,
        ] = self.unlabeled_category

        split_map = {
            "train_known": "train",
            "val_known": "val",
            "test_known": "test",
            "test_unknown": "target",
        }
        prepared.obs[self.split_key] = (
            prepared.obs[SPLIT_COLUMN].map(split_map).fillna("target").astype(str)
        )

        use_domain_labels = bool(
            config.get("backbone", {}).get("scnym", {}).get("use_domain_labels", False)
        )
        batch_col = config["data"].get("batch_column")
        if use_domain_labels and batch_col and batch_col in prepared.obs:
            prepared.obs[self.domain_key] = (
                prepared.obs[batch_col].astype("category").cat.codes.astype(int)
            )

        return prepared

    def _labeled_training_adata(self, prepared_adata):
        mask = prepared_adata.obs[self.label_key] != self.unlabeled_category
        return prepared_adata[mask].copy()

    def train(
        self,
        adata,
        train_idx,
        val_idx,
        config: dict,
        checkpoints_dir: Path,
        *,
        use_cached: bool = False,
    ) -> BackboneTrainResult:
        _, scnym_api = self._require_scnym()

        prepared_adata = self._prepare_adata(adata, train_idx, config)
        training_adata = self._labeled_training_adata(prepared_adata)
        model_dir = self._checkpoint_dir(checkpoints_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        if not (use_cached and self._has_cached_model(model_dir)):
            scnym_api(
                adata=prepared_adata,
                task="train",
                groupby=self.label_key,
                domain_groupby=self.domain_key if self.domain_key in prepared_adata.obs else None,
                out_path=str(model_dir),
                config=self._scnym_cfg(config),
            )

        return BackboneTrainResult(
            backbone_name=self.name,
            model=str(model_dir),
            prepared_adata=prepared_adata,
            training_adata=training_adata,
            metadata={"label_key": self.label_key, "split_key": self.split_key},
        )

    def predict(
        self,
        adata,
        train_idx,
        val_idx,
        config: dict,
        checkpoints_dir: Path,
    ) -> BackbonePredictResult:
        _, scnym_api = self._require_scnym()

        prepared_adata = self._prepare_adata(adata, train_idx, config)
        training_adata = self._labeled_training_adata(prepared_adata)
        model_dir = self._checkpoint_dir(checkpoints_dir)
        if not self._has_cached_model(model_dir):
            raise FileNotFoundError(
                f"Missing scNym checkpoint under {model_dir}. Run train.py first."
            )

        scnym_api(
            adata=prepared_adata,
            task="predict",
            trained_model=str(model_dir),
            out_path=str(model_dir),
            config=self._scnym_cfg(config),
            key_added=self.prediction_key,
        )

        known_classes = list(config["data"]["known_classes"])
        soft_pred = self._extract_soft_predictions(prepared_adata, known_classes)
        hard_pred = prepared_adata.obs[self.prediction_key].astype(str).to_numpy()
        latent = self._extract_latent_representation(prepared_adata)

        extra_soft = soft_pred.copy()
        extra_soft.insert(0, "cell_id", prepared_adata.obs_names.astype(str))
        extra_soft["predicted_label"] = hard_pred
        if self.confidence_key in prepared_adata.obs:
            extra_soft["confidence"] = prepared_adata.obs[self.confidence_key].to_numpy()

        return BackbonePredictResult(
            backbone_name=self.name,
            model=str(model_dir),
            prepared_adata=prepared_adata,
            training_adata=training_adata,
            latent_representation=latent,
            soft_predictions=soft_pred,
            hard_predictions=hard_pred,
            extra_artifacts={
                "scnym_embedding.npy": latent,
                "scnym_soft_predictions.csv": extra_soft,
            },
            metadata={"label_key": self.label_key, "split_key": self.split_key},
        )

    def _extract_soft_predictions(self, prepared_adata, known_classes: list[str]) -> pd.DataFrame:
        probabilities = prepared_adata.uns.get(self.probability_key)
        if probabilities is None:
            raise ValueError(
                f"scNym predict output is missing `adata.uns['{self.probability_key}']`."
            )

        if isinstance(probabilities, pd.DataFrame):
            soft_pred = probabilities.copy()
        else:
            soft_pred = pd.DataFrame(probabilities)

        if soft_pred.shape[0] != prepared_adata.n_obs:
            raise ValueError(
                "scNym probability matrix row count does not match the number of cells."
            )

        if soft_pred.index is None or len(soft_pred.index) != prepared_adata.n_obs:
            soft_pred.index = prepared_adata.obs_names.astype(str)

        rename_map = {}
        for col in soft_pred.columns:
            col_str = str(col)
            rename_map[col] = col_str.replace("prob_", "", 1) if col_str.startswith("prob_") else col_str
        soft_pred = soft_pred.rename(columns=rename_map)

        for cls in known_classes:
            if cls not in soft_pred.columns:
                soft_pred[cls] = 0.0
        return soft_pred.loc[:, known_classes].astype(float)

    def _extract_latent_representation(self, prepared_adata):
        if self.embedding_key not in prepared_adata.obsm:
            raise ValueError(
                f"scNym predict output is missing `adata.obsm['{self.embedding_key}']`."
            )
        return np.asarray(prepared_adata.obsm[self.embedding_key], dtype=float)

    def build_expression_artifacts(
        self,
        adata,
        predict_output: BackbonePredictResult,
        split_context: dict,
        config: dict,
    ) -> ExpressionArtifactsResult:
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
        expression_scores = ensure_expression_schema(
            expression_scores,
            config["data"]["known_classes"],
        )
        return ExpressionArtifactsResult(
            backbone_name=self.name,
            expression_scores=expression_scores,
            centroids=centroids,
            extra_artifacts={
                "latent_cell_order.csv": pd.DataFrame(
                    {"cell_id": predict_output.prepared_adata.obs_names.astype(str)}
                )
            },
            metadata={"label_key": self.label_key, "split_key": self.split_key},
        )
