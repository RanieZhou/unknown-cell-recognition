"""Selective fusion and lineage rescue score builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from ..constants import SPLIT_COLUMN, SPLIT_TEST_KNOWN, SPLIT_TEST_UNKNOWN, SPLIT_VAL, TEST_SPLITS
from ..evaluation.metrics import minmax_on_ref


def compute_expr_margin(df, known_classes):
    prob_cols = [f"prob_{cls}" for cls in known_classes]
    probs = df[prob_cols].to_numpy(dtype=float)
    sorted_probs = np.sort(probs, axis=1)
    df = df.copy()
    df["expr_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
    df["expr_margin_unknown_score_raw"] = 1.0 - df["expr_margin"]
    return df


def compute_grn_margin(df, auc_df, centroid_df, known_classes):
    missing_rows = [cls for cls in known_classes if cls not in centroid_df.index]
    if missing_rows:
        raise ValueError(f"Missing known-class centroids in centroid file: {missing_rows}")

    common_regulons = [col for col in centroid_df.columns if col in auc_df.columns]
    if not common_regulons:
        raise ValueError("No shared regulon columns between AUCell matrix and centroid file")

    centroid_mat = centroid_df.loc[known_classes, common_regulons].to_numpy(dtype=float)
    distances = np.full((len(df), len(known_classes)), np.nan, dtype=float)

    available_mask = df["cell_id"].astype(str).isin(auc_df.index)
    available_cells = df.loc[available_mask, "cell_id"].astype(str).tolist()
    if available_cells:
        cell_mat = auc_df.loc[available_cells, common_regulons].to_numpy(dtype=float)
        dist_mat = cdist(cell_mat, centroid_mat, metric="cosine")
        dist_mat = np.nan_to_num(dist_mat, nan=1.0, posinf=1.0, neginf=0.0)
        distances[np.where(available_mask)[0], :] = dist_mat

    nearest_labels = []
    for row in distances:
        nearest_labels.append(known_classes[int(np.nanargmin(row))] if np.isfinite(row).any() else np.nan)
    sorted_dist = np.sort(distances, axis=1)

    df = df.copy()
    df["nearest_grn_class"] = nearest_labels
    df["grn_distance_recomputed"] = sorted_dist[:, 0]
    df["grn_margin"] = sorted_dist[:, 1] - sorted_dist[:, 0]
    if "grn_distance" in df.columns and df["grn_distance"].isna().any():
        df["grn_distance"] = df["grn_distance"].fillna(df["grn_distance_recomputed"])
    elif "grn_distance" not in df.columns:
        df["grn_distance"] = df["grn_distance_recomputed"]
    return df


def add_selective_fusion_scores(df, *, gate_quantile: float, alpha: float):
    df = df.copy()
    val_mask = df[SPLIT_COLUMN] == SPLIT_VAL
    if not val_mask.any():
        raise ValueError("No val_known cells available for calibration")

    df["entropy_minmax"], _ = minmax_on_ref(df["entropy"].to_numpy(), df.loc[val_mask, "entropy"].to_numpy())
    df["expr_distance_minmax"], _ = minmax_on_ref(df["expr_distance"].to_numpy(), df.loc[val_mask, "expr_distance"].to_numpy())
    df["expr_margin_unknown_score"], _ = minmax_on_ref(
        df["expr_margin_unknown_score_raw"].to_numpy(),
        df.loc[val_mask, "expr_margin_unknown_score_raw"].to_numpy(),
    )
    df["grn_distance_minmax"], _ = minmax_on_ref(df["grn_distance"].to_numpy(), df.loc[val_mask, "grn_distance"].to_numpy())
    df["grn_margin_minmax"], _ = minmax_on_ref(df["grn_margin"].to_numpy(), df.loc[val_mask, "grn_margin"].to_numpy())
    df["grn_margin_unknown_score"] = 1.0 - df["grn_margin_minmax"]
    df["grn_aux_score"] = 0.5 * df["grn_distance_minmax"] + 0.5 * df["grn_margin_unknown_score"]
    df["expr_uncertainty_score"] = (
        df["entropy_minmax"] + df["expr_distance_minmax"] + df["expr_margin_unknown_score"]
    ) / 3.0
    gate_threshold = float(np.percentile(df.loc[val_mask, "expr_uncertainty_score"], gate_quantile * 100))
    df["is_gated"] = df["expr_uncertainty_score"] >= gate_threshold
    df["selective_fused_score"] = np.where(
        df["is_gated"],
        (1.0 - alpha) * df["expr_fused"] + alpha * df["grn_aux_score"],
        df["expr_fused"],
    )
    return df, gate_threshold


def build_variant(
    df,
    prefix,
    gate_threshold_info,
    alpha_map,
    positive_only=True,
    allow_cdc2=False,
):
    df = df.copy()
    bucket_gate_threshold = df["lineage_bucket"].map(gate_threshold_info["by_bucket"]).fillna(gate_threshold_info["global"])
    df[f"{prefix}_gate_threshold"] = bucket_gate_threshold
    df[f"{prefix}_is_bucket_gated"] = df["expr_uncertainty_score"] >= bucket_gate_threshold
    allowed_keys = list(alpha_map.keys()) if allow_cdc2 else [key for key, value in alpha_map.items() if value > 0]
    df[f"{prefix}_lineage_allowed"] = df["lineage_bucket"].isin(allowed_keys)
    if positive_only:
        df[f"{prefix}_is_positive_grn_rescue_candidate"] = df["grn_aux_score"] > df["expr_fused"]
    else:
        df[f"{prefix}_is_positive_grn_rescue_candidate"] = True
    df[f"{prefix}_is_rescue_eligible"] = (
        df[f"{prefix}_is_bucket_gated"]
        & df[f"{prefix}_lineage_allowed"]
        & df[f"{prefix}_is_positive_grn_rescue_candidate"]
    )
    df[f"{prefix}_alpha"] = df["lineage_bucket"].map(alpha_map).fillna(0.0).astype(float)
    if not allow_cdc2:
        blocked_bucket = [bucket for bucket, value in alpha_map.items() if value == 0.0]
        if blocked_bucket:
            df.loc[df["lineage_bucket"].isin(blocked_bucket), f"{prefix}_alpha"] = 0.0
    delta = df["grn_aux_score"] - df["expr_fused"]
    df[f"{prefix}_delta"] = np.where(df[f"{prefix}_is_rescue_eligible"], df[f"{prefix}_alpha"] * delta, 0.0)
    df[f"{prefix}_score"] = df["expr_fused"] + df[f"{prefix}_delta"]
    return df


def compute_grn_proximity(df, unknown_class):
    asdc_df = df[df["true_label"] == unknown_class].copy()
    if asdc_df.empty or "nearest_grn_class" not in asdc_df.columns:
        return pd.DataFrame(columns=["nearest_class", "count", "proportion"])
    counts = asdc_df["nearest_grn_class"].dropna().value_counts()
    if counts.empty:
        return pd.DataFrame(columns=["nearest_class", "count", "proportion"])
    return pd.DataFrame(
        {
            "nearest_class": counts.index,
            "count": counts.values,
            "proportion": [round(float(value), 4) for value in counts.values / counts.sum()],
        }
    )


def error_analysis(df, score_col="selective_fused_score", threshold=None, feature_cols=None, nearest_col="nearest_grn_class"):
    test_df = df[df[SPLIT_COLUMN].isin([SPLIT_TEST_KNOWN, SPLIT_TEST_UNKNOWN])].copy()
    if test_df.empty or threshold is None or score_col not in test_df.columns:
        return pd.DataFrame(), "selective_fused_score threshold unavailable."
    y_true = (test_df[SPLIT_COLUMN] == SPLIT_TEST_UNKNOWN).astype(int).to_numpy()
    y_pred = (test_df[score_col].to_numpy() > threshold).astype(int)

    labels = []
    for truth, pred in zip(y_true, y_pred):
        if truth == 1 and pred == 1:
            labels.append("TP")
        elif truth == 0 and pred == 1:
            labels.append("FP")
        elif truth == 0 and pred == 0:
            labels.append("TN")
        else:
            labels.append("FN")
    test_df["error_type"] = labels

    feature_cols = feature_cols or [
        "entropy",
        "expr_distance",
        "expr_margin",
        "grn_distance",
        "grn_margin",
        "expr_uncertainty_score",
    ]
    save_cols = [
        "cell_id",
        SPLIT_COLUMN,
        "true_label",
        "predicted_label",
        score_col,
        "error_type",
    ]
    save_cols.extend([col for col in feature_cols if col in test_df.columns])
    if "is_gated" in test_df.columns:
        save_cols.append("is_gated")
    if nearest_col in test_df.columns:
        save_cols.append(nearest_col)
    save_cols = [col for col in dict.fromkeys(save_cols) if col in test_df.columns]

    counts = test_df["error_type"].value_counts()
    explanation_parts = [
        f"TP={int(counts.get('TP', 0))}, FP={int(counts.get('FP', 0))}, TN={int(counts.get('TN', 0))}, FN={int(counts.get('FN', 0))}"
    ]
    if "is_gated" in test_df.columns:
        gate_by_type = test_df.groupby("error_type")["is_gated"].mean().round(4).to_dict()
        explanation_parts.append(f"gate rate by error type={gate_by_type}")
    fn_df = test_df[test_df["error_type"] == "FN"].copy()
    if not fn_df.empty and nearest_col in fn_df.columns:
        fn_nearest = fn_df[nearest_col].value_counts(normalize=True).round(4).to_dict()
        explanation_parts.append(f"FN nearest GRN class={fn_nearest}")
    return test_df[save_cols], "; ".join(explanation_parts)
