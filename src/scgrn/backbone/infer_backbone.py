"""scANVI-specific inference helpers used by the scanvi adapter."""

from __future__ import annotations


def extract_predictions(scanvi_model, adata):
    latent_all = scanvi_model.get_latent_representation(adata)
    soft_pred = scanvi_model.predict(adata, soft=True)
    hard_pred = scanvi_model.predict(adata, soft=False)
    return latent_all, soft_pred, hard_pred
