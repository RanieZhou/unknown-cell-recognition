"""
E005 Step 1: scANVI expression backbone
========================================
1. Load data, create counts layer from adata.raw.X
2. Split: train_known(70%) / val_known(15%) / test_known(15%) + test_unknown(all ASDC)
3. Semi-supervised masking: 10% of train_known labels -> "unlabeled"
4. SCVI pretrain -> scANVI fine-tune
5. Extract latent, classification prob, entropy, distance score
6. Compute expr_fused score

Usage:
    python scripts/run_e005_scanvi_backbone.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse
from scipy.spatial.distance import mahalanobis
from sklearn.model_selection import train_test_split

import torch
import scvi
from scvi.model import SCVI, SCANVI

torch.set_float32_matmul_precision("medium")

warnings.filterwarnings("ignore")

# ============================================================
# Config
# ============================================================
SEED = 42
DATA_PATH = r"D:\Desktop\研一\code\scGRN\data\human_immune_health_atlas_dc.patched.h5ad"
OUTPUT_DIR = r"D:\Desktop\研一\code\scGRN\outputs\E005"
LOG_DIR = r"D:\Desktop\研一\code\scGRN\logs\E005"

KNOWN_CLASSES = ["cDC2", "pDC", "cDC1"]
UNKNOWN_CLASS = "ASDC"
LABEL_KEY = "AIFI_L2"
BATCH_KEY = "batch_id"

# scVI / scANVI hyperparams
N_LATENT = 10
N_LAYERS = 2
N_HIDDEN = 128
DROPOUT_RATE = 0.1
GENE_LIKELIHOOD = "nb"
SCVI_MAX_EPOCHS = 200
SCANVI_MAX_EPOCHS = 100

# Semi-supervised masking
UNLABEL_FRAC = 0.10

# Scoring
EXPR_ENTROPY_WEIGHT = 0.5
EXPR_DISTANCE_WEIGHT = 0.5

np.random.seed(SEED)
scvi.settings.seed = SEED


def setup_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def log(msg):
    print(f"[E005] {msg}", flush=True)


# ============================================================
# 1. Load & prepare data
# ============================================================
def load_data():
    log("Loading data...")
    adata = ad.read_h5ad(DATA_PATH)
    log(f"  shape: {adata.shape}")

    # Create counts layer from raw
    log("  Creating counts layer from adata.raw.X...")
    adata.layers["counts"] = adata.raw.X.copy()

    # Verify
    sample = adata.layers["counts"][:100, :100]
    if sparse.issparse(sample):
        sample = sample.toarray()
    assert np.all(sample == sample.astype(int)), "counts layer is not integer!"
    log("  Counts layer verified (integer values).")

    return adata


# ============================================================
# 2. Data splitting
# ============================================================
def split_data(adata):
    log("Splitting data...")

    # Separate known and unknown
    known_mask = adata.obs[LABEL_KEY].isin(KNOWN_CLASSES)
    unknown_mask = adata.obs[LABEL_KEY] == UNKNOWN_CLASS

    known_idx = adata.obs_names[known_mask].tolist()
    unknown_idx = adata.obs_names[unknown_mask].tolist()

    log(f"  Known cells: {len(known_idx)}")
    log(f"  Unknown cells ({UNKNOWN_CLASS}): {len(unknown_idx)}")

    # Stratified split on known cells
    known_labels = adata.obs.loc[known_idx, LABEL_KEY].values

    # First split: 70% train, 30% rest
    train_idx, rest_idx, train_labels, rest_labels = train_test_split(
        known_idx, known_labels, test_size=0.30,
        stratify=known_labels, random_state=SEED
    )
    # Second split: rest -> 50/50 => 15% val, 15% test
    val_idx, test_known_idx, _, _ = train_test_split(
        rest_idx, rest_labels, test_size=0.50,
        stratify=rest_labels, random_state=SEED
    )

    # Create split column
    adata.obs["E005_split"] = "excluded"
    adata.obs.loc[train_idx, "E005_split"] = "train_known"
    adata.obs.loc[val_idx, "E005_split"] = "val_known"
    adata.obs.loc[test_known_idx, "E005_split"] = "test_known"
    adata.obs.loc[unknown_idx, "E005_split"] = "test_unknown"

    # Summary
    split_summary = adata.obs.groupby(["E005_split", LABEL_KEY]).size().reset_index(name="count")
    log("  Split summary:")
    print(split_summary.to_string(index=False))

    split_summary.to_csv(os.path.join(OUTPUT_DIR, "E005_split_summary.csv"), index=False)

    return adata, train_idx, val_idx, test_known_idx, unknown_idx


# ============================================================
# 3. Semi-supervised label masking
# ============================================================
def create_scanvi_labels(adata, train_idx):
    log("Creating scANVI labels with semi-supervised masking...")

    # Start with all cells as "unlabeled"
    # Only train_known cells get their true labels (minus the 10% masked)
    adata.obs["scanvi_label"] = "unlabeled"

    # Assign true labels to train_known cells
    adata.obs.loc[train_idx, "scanvi_label"] = adata.obs.loc[train_idx, LABEL_KEY].astype(str)

    # Mask 10% of train_known
    n_mask = int(len(train_idx) * UNLABEL_FRAC)
    rng = np.random.RandomState(SEED)
    mask_idx = rng.choice(train_idx, size=n_mask, replace=False)
    adata.obs.loc[mask_idx, "scanvi_label"] = "unlabeled"

    log(f"  Masked {n_mask}/{len(train_idx)} train cells as 'unlabeled'")
    log(f"  Label distribution (full adata):")
    print(adata.obs["scanvi_label"].value_counts().to_string())

    return adata


# ============================================================
# 4. Train SCVI + scANVI
# ============================================================
def train_models(adata, train_idx, val_idx):
    log("Preparing training AnnData...")

    # scANVI trains on train+val? No — train only
    # But we need val for early stopping
    # scvi-tools uses train_size internally; we provide the full dataset 
    # and let the split column guide

    # Create combined train+val subset for model training
    train_val_idx = train_idx + val_idx
    adata_train = adata[train_val_idx].copy()

    # Mark val cells as unlabeled for scANVI training (they should not contribute labels)
    adata_train.obs.loc[val_idx, "scanvi_label"] = "unlabeled"

    log(f"  Training AnnData shape: {adata_train.shape}")
    log(f"  Label distribution in training data:")
    print(adata_train.obs["scanvi_label"].value_counts().to_string())

    # Setup anndata
    SCVI.setup_anndata(
        adata_train,
        layer="counts",
        batch_key=BATCH_KEY,
        labels_key="scanvi_label",
    )

    # --- SCVI pretrain ---
    log("Training SCVI...")
    scvi_model = SCVI(
        adata_train,
        n_latent=N_LATENT,
        n_layers=N_LAYERS,
        n_hidden=N_HIDDEN,
        dropout_rate=DROPOUT_RATE,
        gene_likelihood=GENE_LIKELIHOOD,
    )

    scvi_model.train(
        max_epochs=SCVI_MAX_EPOCHS,
        early_stopping=True,
        early_stopping_patience=10,
        early_stopping_monitor="elbo_validation",
        check_val_every_n_epoch=1,
    )
    log(f"  SCVI training done. Best epoch history length: {len(scvi_model.history['elbo_validation'])}")

    # Save SCVI model
    scvi_model_path = os.path.join(OUTPUT_DIR, "scvi_model")
    scvi_model.save(scvi_model_path, overwrite=True)
    log(f"  SCVI model saved to {scvi_model_path}")

    # --- scANVI fine-tune ---
    log("Training scANVI from SCVI...")
    scanvi_model = SCANVI.from_scvi_model(
        scvi_model,
        unlabeled_category="unlabeled",
        labels_key="scanvi_label",
    )

    scanvi_model.train(
        max_epochs=SCANVI_MAX_EPOCHS,
        early_stopping=True,
        early_stopping_patience=10,
        check_val_every_n_epoch=1,
    )
    log(f"  scANVI training done.")

    # Save scANVI model
    scanvi_model_path = os.path.join(OUTPUT_DIR, "scanvi_model")
    scanvi_model.save(scanvi_model_path, overwrite=True)
    log(f"  scANVI model saved to {scanvi_model_path}")

    return scvi_model, scanvi_model, adata_train


# ============================================================
# 5. Extract predictions on full dataset
# ============================================================
def extract_predictions(scanvi_model, adata, adata_train):
    log("Extracting predictions on full dataset...")

    # The full adata already has scanvi_label set to "unlabeled" for all
    # non-train cells (including ASDC), so the model registry will accept it.
    # We just need to validate/register the full adata with the model.

    # Get latent representations for all cells
    latent_all = scanvi_model.get_latent_representation(adata)
    log(f"  Latent shape (all): {latent_all.shape}")

    # Get soft predictions (classification probabilities)
    soft_pred = scanvi_model.predict(adata, soft=True)
    log(f"  Soft predictions shape: {soft_pred.shape}")
    log(f"  Prediction columns: {list(soft_pred.columns)}")

    # Hard predictions
    hard_pred = scanvi_model.predict(adata, soft=False)

    return latent_all, soft_pred, hard_pred


# ============================================================
# 6. Compute expression scores
# ============================================================
def compute_expression_scores(adata, latent_all, soft_pred,
                              train_idx, val_idx, test_known_idx, unknown_idx):
    log("Computing expression scores...")

    # --- 6.1 Entropy ---
    probs = soft_pred[KNOWN_CLASSES].values  # Only known class probabilities
    # Re-normalize over known classes (in case there's an "unlabeled" column)
    probs = probs / probs.sum(axis=1, keepdims=True)
    probs = np.clip(probs, 1e-10, 1.0)  # avoid log(0)

    entropy = -np.sum(probs * np.log(probs), axis=1)
    log(f"  Entropy range: [{entropy.min():.4f}, {entropy.max():.4f}]")

    # --- 6.2 Latent distance to known class centroids ---
    # Build cell index mapping
    cell_to_pos = {cell: i for i, cell in enumerate(adata.obs_names)}

    train_positions = [cell_to_pos[c] for c in train_idx]
    train_latent = latent_all[train_positions]
    train_labels = adata.obs.loc[train_idx, LABEL_KEY].values

    # Compute centroids and covariance for each known class
    centroids = {}
    cov_invs = {}
    use_mahalanobis = {}

    for cls in KNOWN_CLASSES:
        cls_mask = train_labels == cls
        cls_latent = train_latent[cls_mask]
        centroids[cls] = cls_latent.mean(axis=0)

        try:
            cov = np.cov(cls_latent, rowvar=False)
            # Regularize covariance
            cov += np.eye(cov.shape[0]) * 1e-6
            cov_inv = np.linalg.inv(cov)
            cov_invs[cls] = cov_inv
            use_mahalanobis[cls] = True
            log(f"  {cls}: Mahalanobis distance (n={cls_mask.sum()})")
        except np.linalg.LinAlgError:
            use_mahalanobis[cls] = False
            log(f"  {cls}: Falling back to Euclidean distance (n={cls_mask.sum()})")

    # Compute min distance for each cell
    distances = np.zeros(len(latent_all))
    nearest_class = []

    for i in range(len(latent_all)):
        min_dist = np.inf
        min_cls = None
        for cls in KNOWN_CLASSES:
            if use_mahalanobis[cls]:
                d = mahalanobis(latent_all[i], centroids[cls], cov_invs[cls])
            else:
                d = np.linalg.norm(latent_all[i] - centroids[cls])
            if d < min_dist:
                min_dist = d
                min_cls = cls
        distances[i] = min_dist
        nearest_class.append(min_cls)

    log(f"  Distance range: [{distances.min():.4f}, {distances.max():.4f}]")

    # --- 6.3 Normalize on val_known (min-max) ---
    val_positions = [cell_to_pos[c] for c in val_idx]

    val_entropy = entropy[val_positions]
    val_distance = distances[val_positions]

    entropy_min, entropy_max = val_entropy.min(), val_entropy.max()
    distance_min, distance_max = val_distance.min(), val_distance.max()

    # Avoid division by zero
    entropy_range = max(entropy_max - entropy_min, 1e-10)
    distance_range = max(distance_max - distance_min, 1e-10)

    entropy_norm = (entropy - entropy_min) / entropy_range
    distance_norm = (distances - distance_min) / distance_range

    # Clip to [0, beyond 1 is allowed for test data]
    # Don't clip — values > 1 for test cells are informative

    # --- 6.4 Fused expression score ---
    expr_fused = EXPR_ENTROPY_WEIGHT * entropy_norm + EXPR_DISTANCE_WEIGHT * distance_norm

    log(f"  expr_fused range on val: [{expr_fused[val_positions].min():.4f}, {expr_fused[val_positions].max():.4f}]")

    # --- Build output DataFrame ---
    scores_df = pd.DataFrame({
        "cell_id": adata.obs_names,
        "true_label": adata.obs[LABEL_KEY].values,
        "E005_split": adata.obs["E005_split"].values,
        "predicted_label": soft_pred[KNOWN_CLASSES].idxmax(axis=1).values,
        "nearest_known_class": nearest_class,
        "entropy": entropy,
        "entropy_norm": entropy_norm,
        "latent_distance": distances,
        "distance_norm": distance_norm,
        "expr_fused": expr_fused,
    })

    # Add per-class probabilities
    for cls in KNOWN_CLASSES:
        if cls in soft_pred.columns:
            scores_df[f"prob_{cls}"] = soft_pred[cls].values
        else:
            scores_df[f"prob_{cls}"] = 0.0

    return scores_df, latent_all, centroids


# ============================================================
# 7. Save outputs
# ============================================================
def save_outputs(scores_df, latent_all, adata, train_idx, val_idx):
    log("Saving outputs...")

    # Save expression scores
    scores_df.to_csv(os.path.join(OUTPUT_DIR, "E005_expression_scores.csv"), index=False)
    log("  Saved E005_expression_scores.csv")

    # Save obs subsets
    train_obs = adata.obs.loc[train_idx]
    val_obs = adata.obs.loc[val_idx]
    train_obs.to_csv(os.path.join(OUTPUT_DIR, "E005_train_known_obs.csv"))
    val_obs.to_csv(os.path.join(OUTPUT_DIR, "E005_val_known_obs.csv"))
    log("  Saved train/val obs CSVs")

    # Save latent representations
    cell_to_pos = {cell: i for i, cell in enumerate(adata.obs_names)}

    train_positions = [cell_to_pos[c] for c in train_idx]
    np.save(os.path.join(OUTPUT_DIR, "E005_scanvi_latent_train.npy"), latent_all[train_positions])

    test_mask = adata.obs["E005_split"].isin(["test_known", "test_unknown"])
    test_positions = [cell_to_pos[c] for c in adata.obs_names[test_mask]]
    np.save(os.path.join(OUTPUT_DIR, "E005_scanvi_latent_test.npy"), latent_all[test_positions])
    log("  Saved latent .npy files")

    # Quick metrics preview
    test_scores = scores_df[scores_df["E005_split"].isin(["test_known", "test_unknown"])]
    n_test_known = (test_scores["E005_split"] == "test_known").sum()
    n_test_unknown = (test_scores["E005_split"] == "test_unknown").sum()
    log(f"  Test set: {n_test_known} known + {n_test_unknown} unknown")

    # Expression-only threshold (95th on val_known)
    val_scores = scores_df[scores_df["E005_split"] == "val_known"]
    T_expr = np.percentile(val_scores["expr_fused"], 95)
    log(f"  Expression threshold (95th on val): {T_expr:.4f}")

    # Quick AUROC preview
    from sklearn.metrics import roc_auc_score
    y_true = (test_scores["E005_split"] == "test_unknown").astype(int).values
    auroc = roc_auc_score(y_true, test_scores["expr_fused"].values)
    log(f"  Expression-only AUROC (expr_fused): {auroc:.4f}")

    return T_expr


# ============================================================
# Main
# ============================================================
def main():
    setup_dirs()

    # 1. Load
    adata = load_data()

    # 2. Split
    adata, train_idx, val_idx, test_known_idx, unknown_idx = split_data(adata)

    # 3. Create scANVI labels
    adata = create_scanvi_labels(adata, train_idx)

    # 4. Train or load saved models
    scanvi_model_path = os.path.join(OUTPUT_DIR, "scanvi_model")
    scvi_model_path = os.path.join(OUTPUT_DIR, "scvi_model")

    if os.path.exists(scanvi_model_path):
        log("Found saved scANVI model, loading...")
        # Need to setup anndata first for loading
        train_val_idx = train_idx + val_idx
        adata_train = adata[train_val_idx].copy()
        adata_train.obs.loc[val_idx, "scanvi_label"] = "unlabeled"
        SCVI.setup_anndata(
            adata_train,
            layer="counts",
            batch_key=BATCH_KEY,
            labels_key="scanvi_label",
        )
        scanvi_model = SCANVI.load(scanvi_model_path, adata=adata_train)
        log("  scANVI model loaded.")
    else:
        scvi_model, scanvi_model, adata_train = train_models(adata, train_idx, val_idx)

    # 5. Extract predictions
    latent_all, soft_pred, hard_pred = extract_predictions(scanvi_model, adata, adata_train)

    # 6. Compute scores
    scores_df, latent_all, centroids = compute_expression_scores(
        adata, latent_all, soft_pred,
        train_idx, val_idx, test_known_idx, unknown_idx
    )

    # 7. Save
    T_expr = save_outputs(scores_df, latent_all, adata, train_idx, val_idx)

    log("=" * 60)
    log("Step 1 DONE. Next: run_e005_export_pyscenic_input.py")
    log("=" * 60)


if __name__ == "__main__":
    main()
