"""Expression-score extraction for the refactored mainline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from sklearn.linear_model import LogisticRegression

from ..constants import SPLIT_COLUMN, SPLIT_VAL
from .base import ensure_expression_schema

EXPRESSION_COMBINER_FEATURES = ["entropy_norm", "distance_norm", "expr_margin_unknown_score"]


def _minmax_scale(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref_min = float(np.min(reference))
    ref_max = float(np.max(reference))
    return (values - ref_min) / max(ref_max - ref_min, 1e-10)


def fit_expression_combiner(df: pd.DataFrame) -> dict:
    val_df = df.loc[df[SPLIT_COLUMN] == SPLIT_VAL].copy()
    if val_df.empty:
        raise ValueError("Expression combiner requires non-empty val_known supervision targets.")

    missing = [col for col in ["true_label", "predicted_label", *EXPRESSION_COMBINER_FEATURES] if col not in val_df.columns]
    if missing:
        raise ValueError(f"Expression combiner cannot fit from val_known rows; missing columns: {missing}")

    target = (val_df["predicted_label"].astype(str) != val_df["true_label"].astype(str)).astype(int)
    if target.nunique() < 2:
        raise ValueError("Expression combiner requires val_known misclassification targets with both classes present.")

    features = val_df.loc[:, EXPRESSION_COMBINER_FEATURES].to_numpy(dtype=float)
    model = LogisticRegression(random_state=0, solver="liblinear", max_iter=1000)
    model.fit(features, target.to_numpy(dtype=int))
    return {
        "feature_names": list(EXPRESSION_COMBINER_FEATURES),
        "weights": model.coef_[0].astype(float).tolist(),
        "intercept": float(model.intercept_[0]),
    }


def apply_expression_combiner(df: pd.DataFrame, model: dict) -> pd.DataFrame:
    feature_names = list(model.get("feature_names", []))
    missing = [col for col in feature_names if col not in df.columns]
    if missing:
        raise ValueError(f"Expression combiner cannot score rows; missing columns: {missing}")

    weights = np.asarray(model["weights"], dtype=float)
    intercept = float(model["intercept"])
    features = df.loc[:, feature_names].to_numpy(dtype=float)
    logits = features @ weights + intercept
    probabilities = 1.0 / (1.0 + np.exp(-logits))

    scored = df.copy()
    scored["expr_fused"] = probabilities
    return scored


def compute_expression_scores(
    adata,
    latent_all,
    soft_pred,
    train_idx,
    val_idx,
    test_known_idx,
    unknown_idx,
    config: dict,
):
    data_cfg = config["data"]

    known_classes = data_cfg["known_classes"]
    label_key = data_cfg["label_column"]

    probs = soft_pred[known_classes].to_numpy(dtype=float)
    if not np.isfinite(probs).all():
        raise ValueError("Expression probability matrix contains non-finite values.")
    row_sums = probs.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Expression probability matrix contains rows with non-positive probability sums.")
    probs = probs / row_sums
    probs = np.clip(probs, 1e-10, 1.0)
    entropy = -np.sum(probs * np.log(probs), axis=1)

    sorted_probs = np.sort(probs, axis=1)
    expr_margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    expr_margin_unknown_score = 1.0 - expr_margin

    cell_to_pos = {cell: i for i, cell in enumerate(adata.obs_names)}
    train_positions = [cell_to_pos[c] for c in train_idx]
    val_positions = [cell_to_pos[c] for c in val_idx]
    if len(val_positions) == 0:
        raise ValueError("Expression score extraction requires non-empty val_known rows for calibration and supervision.")

    train_latent = latent_all[train_positions]
    train_labels = adata.obs.loc[train_idx, label_key].values

    centroids = {}
    cov_invs = {}
    for cls in known_classes:
        cls_mask = train_labels == cls
        cls_latent = train_latent[cls_mask]
        centroids[cls] = cls_latent.mean(axis=0)
        try:
            cov = np.cov(cls_latent, rowvar=False) + np.eye(cls_latent.shape[1]) * 1e-6
            cov_invs[cls] = np.linalg.inv(cov)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"Mahalanobis covariance fit failed for class '{cls}'. "
                "Aborting instead of falling back to Euclidean distance."
            ) from exc

    distances = np.zeros(len(latent_all), dtype=float)
    nearest_class = []
    for i in range(len(latent_all)):
        min_dist = np.inf
        min_cls = None
        for cls in known_classes:
            distance = mahalanobis(latent_all[i], centroids[cls], cov_invs[cls])
            if distance < min_dist:
                min_dist = distance
                min_cls = cls
        distances[i] = min_dist
        nearest_class.append(min_cls)

    val_entropy = entropy[val_positions]
    val_distance = distances[val_positions]
    val_margin_unknown = expr_margin_unknown_score[val_positions]
    entropy_norm = _minmax_scale(entropy, val_entropy)
    distance_norm = _minmax_scale(distances, val_distance)
    expr_margin_unknown_score = _minmax_scale(expr_margin_unknown_score, val_margin_unknown)

    scores_df = pd.DataFrame(
        {
            "cell_id": adata.obs_names.astype(str),
            "true_label": adata.obs[label_key].values,
            SPLIT_COLUMN: adata.obs[SPLIT_COLUMN].values,
            "predicted_label": soft_pred[known_classes].idxmax(axis=1).values,
            "nearest_known_class": nearest_class,
            "entropy": entropy,
            "entropy_norm": entropy_norm,
            "latent_distance": distances,
            "distance_norm": distance_norm,
            "expr_margin": expr_margin,
            "expr_margin_unknown_score": expr_margin_unknown_score,
        }
    )

    for cls in known_classes:
        scores_df[f"prob_{cls}"] = soft_pred[cls].values

    combiner_model = fit_expression_combiner(scores_df)
    scores_df = apply_expression_combiner(scores_df, combiner_model)
    scores_df = ensure_expression_schema(scores_df, known_classes)
    return scores_df, latent_all, centroids
