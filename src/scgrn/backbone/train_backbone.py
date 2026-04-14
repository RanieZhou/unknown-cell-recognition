"""scANVI-specific training utilities used by the scanvi adapter."""

from __future__ import annotations

import numpy as np


def create_scanvi_labels(adata, train_idx, *, label_key: str, seed: int, unlabel_frac: float):
    adata = adata.copy()
    adata.obs["scanvi_label"] = "unlabeled"
    adata.obs.loc[train_idx, "scanvi_label"] = adata.obs.loc[train_idx, label_key].astype(str)

    n_mask = int(len(train_idx) * unlabel_frac)
    if n_mask > 0:
        rng = np.random.RandomState(seed)
        mask_idx = rng.choice(train_idx, size=n_mask, replace=False)
        adata.obs.loc[mask_idx, "scanvi_label"] = "unlabeled"
    return adata


def train_backbone(adata, train_idx, val_idx, config: dict, checkpoints_dir):
    import scvi
    from scvi.model import SCANVI, SCVI

    training_cfg = config["training"]
    data_cfg = config["data"]

    train_val_idx = train_idx + val_idx
    adata_train = adata[train_val_idx].copy()
    adata_train.obs.loc[val_idx, "scanvi_label"] = "unlabeled"

    SCVI.setup_anndata(
        adata_train,
        layer=data_cfg["model_layer"],
        batch_key=data_cfg["batch_column"],
        labels_key="scanvi_label",
    )

    scvi.settings.seed = int(training_cfg["seed"])
    scvi_model = SCVI(
        adata_train,
        n_latent=int(training_cfg["n_latent"]),
        n_layers=int(training_cfg["n_layers"]),
        n_hidden=int(training_cfg["n_hidden"]),
        dropout_rate=float(training_cfg["dropout_rate"]),
        gene_likelihood=str(training_cfg["gene_likelihood"]),
    )
    scvi_model.train(
        max_epochs=int(training_cfg["scvi_max_epochs"]),
        early_stopping=True,
        early_stopping_patience=10,
        early_stopping_monitor="elbo_validation",
        check_val_every_n_epoch=1,
    )

    scanvi_model = SCANVI.from_scvi_model(
        scvi_model,
        unlabeled_category="unlabeled",
        labels_key="scanvi_label",
    )
    scanvi_model.train(
        max_epochs=int(training_cfg["scanvi_max_epochs"]),
        early_stopping=True,
        early_stopping_patience=10,
        check_val_every_n_epoch=1,
    )

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    scvi_model.save(str(checkpoints_dir / "scvi_model"), overwrite=True)
    scanvi_model.save(str(checkpoints_dir / "scanvi_model"), overwrite=True)
    return scanvi_model, adata_train


def load_trained_backbone(adata, train_idx, val_idx, config: dict, checkpoints_dir):
    from scvi.model import SCANVI, SCVI

    data_cfg = config["data"]
    train_val_idx = train_idx + val_idx
    adata_train = adata[train_val_idx].copy()
    adata_train.obs.loc[val_idx, "scanvi_label"] = "unlabeled"

    SCVI.setup_anndata(
        adata_train,
        layer=data_cfg["model_layer"],
        batch_key=data_cfg["batch_column"],
        labels_key="scanvi_label",
    )
    scanvi_model = SCANVI.load(str(checkpoints_dir / "scanvi_model"), adata=adata_train)
    return scanvi_model, adata_train
