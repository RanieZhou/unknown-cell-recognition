"""Threshold helpers for selective and lineage rescue."""

from __future__ import annotations

import numpy as np


def compute_threshold_on_val(scores, percentile=95):
    scores = np.asarray(scores, dtype=float)
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return np.nan
    return float(np.percentile(finite, percentile))


def fit_thresholds_by_bucket(df, known_buckets, value_col, quantile=0.95, split_col="split", split_value="val_known", min_cells=100):
    val_df = df[df[split_col] == split_value].copy()
    if val_df.empty:
        raise ValueError(f"No {split_value} cells available for threshold fitting")
    global_threshold = float(np.percentile(val_df[value_col].dropna(), quantile * 100))
    bucket_thresholds = {}
    bucket_sizes = {}
    for bucket in known_buckets:
        sub = val_df[val_df["lineage_bucket"] == bucket][value_col].dropna()
        bucket_sizes[bucket] = int(len(sub))
        bucket_thresholds[bucket] = float(np.percentile(sub, quantile * 100)) if len(sub) >= min_cells else global_threshold
    return {
        "global": global_threshold,
        "by_bucket": bucket_thresholds,
        "bucket_sizes": bucket_sizes,
        "quantile": quantile,
        "value_col": value_col,
    }


def map_bucket_thresholds(df, threshold_info, bucket_col="lineage_bucket"):
    return df[bucket_col].map(threshold_info["by_bucket"]).fillna(threshold_info["global"]).astype(float)


def predict_with_global_threshold(df, score_col, threshold, pred_col):
    df[pred_col] = (df[score_col].to_numpy(dtype=float) > float(threshold)).astype(int)
    return df


def predict_with_bucket_threshold(df, score_col, threshold_info, pred_col, threshold_col):
    thresholds = map_bucket_thresholds(df, threshold_info)
    df[threshold_col] = thresholds
    df[pred_col] = (df[score_col].to_numpy(dtype=float) > thresholds.to_numpy(dtype=float)).astype(int)
    return df
