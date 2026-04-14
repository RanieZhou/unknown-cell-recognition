"""Metrics reused across inference, rescue, and evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ..constants import SPLIT_COLUMN, SPLIT_TEST_KNOWN, SPLIT_TEST_UNKNOWN, TEST_SPLITS


class MinMaxFitter:
    """Fit min/max on a reference set and reuse on other splits."""

    def __init__(self):
        self.min_val = 0.0
        self.max_val = 1.0

    def fit(self, ref_values):
        ref_values = np.asarray(ref_values, dtype=float)
        finite = ref_values[np.isfinite(ref_values)]
        if finite.size == 0:
            self.min_val = 0.0
            self.max_val = 1.0
            return self
        self.min_val = float(np.min(finite))
        self.max_val = float(np.max(finite))
        if self.max_val <= self.min_val:
            self.max_val = self.min_val + 1e-8
        return self

    def transform(self, values):
        values = np.asarray(values, dtype=float)
        scaled = (values - self.min_val) / (self.max_val - self.min_val)
        return np.clip(scaled, 0.0, 1.0)


def minmax_on_ref(all_values, ref_values):
    fitter = MinMaxFitter().fit(ref_values)
    return fitter.transform(all_values), fitter


def sanitize_scores(y_score):
    scores = np.asarray(y_score, dtype=float)
    scores = np.where(np.isfinite(scores), scores, np.nan)
    if np.isnan(scores).all():
        return scores
    finite_max = np.nanmax(scores)
    return np.where(np.isnan(scores), finite_max, scores)


def compute_fpr_at_tpr(y_true, y_score, tpr_target=0.95):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if np.unique(y_true).size < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.where(tpr >= tpr_target)[0]
    if idx.size == 0:
        return 1.0
    return float(fpr[idx[0]])


def compute_all_metrics(y_true_binary, y_score, threshold=None):
    y_true_binary = np.asarray(y_true_binary, dtype=int)
    scores = sanitize_scores(y_score)
    metrics = {
        "AUROC": np.nan,
        "AUPR": np.nan,
        "FPR95": np.nan,
        "threshold_95": np.nan,
        "Accuracy": np.nan,
        "Macro_F1": np.nan,
        "Precision": np.nan,
        "Recall": np.nan,
    }
    if scores.size == 0 or np.isnan(scores).all() or np.unique(y_true_binary).size < 2:
        return metrics

    metrics["AUROC"] = round(float(roc_auc_score(y_true_binary, scores)), 4)
    metrics["AUPR"] = round(float(average_precision_score(y_true_binary, scores)), 4)
    metrics["FPR95"] = round(float(compute_fpr_at_tpr(y_true_binary, scores, 0.95)), 4)

    if threshold is None or not np.isfinite(threshold):
        return metrics
    y_pred = (scores > threshold).astype(int)
    metrics["threshold_95"] = round(float(threshold), 6)
    metrics["Accuracy"] = round(float(accuracy_score(y_true_binary, y_pred)), 4)
    metrics["Macro_F1"] = round(float(f1_score(y_true_binary, y_pred, average="macro", zero_division=0)), 4)
    metrics["Precision"] = round(float(precision_score(y_true_binary, y_pred, zero_division=0)), 4)
    metrics["Recall"] = round(float(recall_score(y_true_binary, y_pred, zero_division=0)), 4)
    return metrics


def compute_gate_stats(df, gate_col="is_gated", split_col=SPLIT_COLUMN):
    test_df = df[df[split_col].isin(TEST_SPLITS)].copy()
    if gate_col not in df.columns or test_df.empty:
        return {
            "total_test": int(len(test_df)),
            "gated_total": 0,
            "gated_fraction": np.nan,
            "gated_on_unknown": 0,
            "gated_fraction_on_unknown": np.nan,
            "gated_on_known": 0,
            "gated_fraction_on_known": np.nan,
            "n_unknown": int((test_df[split_col] == SPLIT_TEST_UNKNOWN).sum()) if not test_df.empty else 0,
            "n_known": int((test_df[split_col] == SPLIT_TEST_KNOWN).sum()) if not test_df.empty else 0,
        }

    test_df[gate_col] = test_df[gate_col].fillna(False).astype(bool)
    unknown_mask = test_df[split_col] == SPLIT_TEST_UNKNOWN
    known_mask = test_df[split_col] == SPLIT_TEST_KNOWN
    return {
        "total_test": int(len(test_df)),
        "gated_total": int(test_df[gate_col].sum()),
        "gated_fraction": round(float(test_df[gate_col].mean()), 4),
        "gated_on_unknown": int(test_df.loc[unknown_mask, gate_col].sum()),
        "gated_fraction_on_unknown": round(float(test_df.loc[unknown_mask, gate_col].mean()), 4),
        "gated_on_known": int(test_df.loc[known_mask, gate_col].sum()),
        "gated_fraction_on_known": round(float(test_df.loc[known_mask, gate_col].mean()), 4),
        "n_unknown": int(unknown_mask.sum()),
        "n_known": int(known_mask.sum()),
    }


def metrics_to_frame(metrics_by_method):
    rows = []
    for method, metrics in metrics_by_method.items():
        row = {"method": method}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_metrics_from_predictions(y_true, scores, y_pred, threshold_value):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    scores = sanitize_scores(scores)
    metrics = {
        "AUROC": np.nan,
        "AUPR": np.nan,
        "FPR95": np.nan,
        "threshold_95": threshold_value,
        "Accuracy": np.nan,
        "Macro_F1": np.nan,
        "Precision": np.nan,
        "Recall": np.nan,
    }
    if scores.size == 0 or np.isnan(scores).all() or np.unique(y_true).size < 2:
        return metrics
    metrics["AUROC"] = round(float(roc_auc_score(y_true, scores)), 4)
    metrics["AUPR"] = round(float(average_precision_score(y_true, scores)), 4)
    metrics["FPR95"] = round(float(compute_fpr_at_tpr(y_true, scores, 0.95)), 4)
    metrics["Accuracy"] = round(float(accuracy_score(y_true, y_pred)), 4)
    metrics["Macro_F1"] = round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)
    metrics["Precision"] = round(float(precision_score(y_true, y_pred, zero_division=0)), 4)
    metrics["Recall"] = round(float(recall_score(y_true, y_pred, zero_division=0)), 4)
    if np.isscalar(threshold_value) and pd.notna(threshold_value):
        try:
            metrics["threshold_95"] = round(float(threshold_value), 6)
        except (TypeError, ValueError):
            metrics["threshold_95"] = threshold_value
    return metrics


def evaluate_method(df, method_name, score_col, pred_col, threshold_value=None, extra_metrics=None):
    test_df = df[df[SPLIT_COLUMN].isin(TEST_SPLITS)].copy()
    y_true = (test_df[SPLIT_COLUMN] == SPLIT_TEST_UNKNOWN).astype(int).to_numpy()
    y_pred = test_df[pred_col].to_numpy(dtype=int)
    scores = test_df[score_col].to_numpy(dtype=float)
    metrics = compute_metrics_from_predictions(y_true, scores, y_pred, threshold_value)
    if extra_metrics:
        metrics.update(extra_metrics)
    return method_name, metrics


def compute_rescue_stats(df, baseline_pred_col, target_pred_col):
    test_df = df[df[SPLIT_COLUMN].isin(TEST_SPLITS)].copy()
    y_true = (test_df[SPLIT_COLUMN] == SPLIT_TEST_UNKNOWN).astype(int)
    baseline_correct = test_df[baseline_pred_col].astype(int) == y_true
    target_correct = test_df[target_pred_col].astype(int) == y_true
    rescued = (~baseline_correct) & target_correct
    hurt = baseline_correct & (~target_correct)
    return {
        "rescued_count": int(rescued.sum()),
        "hurt_count": int(hurt.sum()),
        "net_gain": int(rescued.sum() - hurt.sum()),
    }


def compute_bucket_metrics(df, known_buckets, method_pred_cols, gate_cols=None, baseline_pred_col="expr_fused_pred"):
    test_df = df[df[SPLIT_COLUMN].isin(TEST_SPLITS)].copy()
    gate_cols = gate_cols or {}
    rows = []

    y_true_all = (test_df[SPLIT_COLUMN] == SPLIT_TEST_UNKNOWN).astype(int)
    baseline_correct = test_df[baseline_pred_col].astype(int) == y_true_all

    for method_name, pred_col in method_pred_cols.items():
        for bucket in known_buckets:
            sub = test_df[test_df["lineage_bucket"] == bucket].copy()
            if sub.empty:
                continue
            y_true = (sub[SPLIT_COLUMN] == SPLIT_TEST_UNKNOWN).astype(int)
            y_pred = sub[pred_col].astype(int)
            known_mask = sub[SPLIT_COLUMN] == SPLIT_TEST_KNOWN
            unknown_mask = sub[SPLIT_COLUMN] == SPLIT_TEST_UNKNOWN
            fp_count = int(((y_pred == 1) & known_mask).sum())
            tp_count = int(((y_pred == 1) & unknown_mask).sum())
            gate_fraction = np.nan
            if method_name in gate_cols and gate_cols[method_name] in sub.columns:
                gate_fraction = round(float(sub[gate_cols[method_name]].mean()), 4)

            baseline_correct_sub = baseline_correct.loc[sub.index]
            method_correct_sub = y_pred == y_true
            rescued = (~baseline_correct_sub) & method_correct_sub
            hurt = baseline_correct_sub & (~method_correct_sub)

            rows.append(
                {
                    "method": method_name,
                    "lineage_bucket": bucket,
                    "n_cells": int(len(sub)),
                    "unknown_proportion": round(float(unknown_mask.mean()), 4),
                    "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
                    "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
                    "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
                    "Macro_F1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
                    "FP_count_on_known": fp_count,
                    "FP_rate_on_known": round(float((fp_count / known_mask.sum()) if known_mask.sum() else np.nan), 4),
                    "TP_count_on_unknown": tp_count,
                    "Recall_on_unknown": round(float((tp_count / unknown_mask.sum()) if unknown_mask.sum() else np.nan), 4),
                    "gated_fraction": gate_fraction,
                    "rescued_count": int(rescued.sum()),
                    "hurt_count": int(hurt.sum()),
                    "net_gain": int(rescued.sum() - hurt.sum()),
                }
            )
    return pd.DataFrame(rows)


def build_rescue_delta_analysis(df, pred_cols, baseline_pred_col="expr_fused_pred"):
    test_df = df[df[SPLIT_COLUMN].isin(TEST_SPLITS)].copy()
    y_true = (test_df[SPLIT_COLUMN] == SPLIT_TEST_UNKNOWN).astype(int)
    baseline_correct = test_df[baseline_pred_col].astype(int) == y_true
    rows = []
    for method_name, pred_col in pred_cols.items():
        method_correct = test_df[pred_col].astype(int) == y_true
        for idx, row in test_df.iterrows():
            rescued = (not baseline_correct.loc[idx]) and bool(method_correct.loc[idx])
            hurt = bool(baseline_correct.loc[idx]) and (not method_correct.loc[idx])
            rows.append(
                {
                    "cell_id": row["cell_id"],
                    SPLIT_COLUMN: row[SPLIT_COLUMN],
                    "true_label": row["true_label"],
                    "lineage_bucket": row["lineage_bucket"],
                    "nearest_grn_class": row.get("nearest_grn_class"),
                    "method": method_name,
                    "baseline_pred": int(test_df.loc[idx, baseline_pred_col]),
                    "method_pred": int(test_df.loc[idx, pred_col]),
                    "rescued": bool(rescued),
                    "hurt": bool(hurt),
                }
            )
    return pd.DataFrame(rows)


def summarize_gate_activity(df, prefix):
    test_df = df[df[SPLIT_COLUMN].isin(TEST_SPLITS)].copy()
    gate_col = f"{prefix}_is_bucket_gated"
    eligible_col = f"{prefix}_is_rescue_eligible"
    if gate_col not in test_df.columns or eligible_col not in test_df.columns:
        return {}
    unknown_mask = test_df[SPLIT_COLUMN] == SPLIT_TEST_UNKNOWN
    known_mask = test_df[SPLIT_COLUMN] == SPLIT_TEST_KNOWN
    return {
        "gated_fraction": round(float(test_df[gate_col].mean()), 4),
        "gated_fraction_on_unknown": round(float(test_df.loc[unknown_mask, gate_col].mean()), 4),
        "gated_fraction_on_known": round(float(test_df.loc[known_mask, gate_col].mean()), 4),
        "rescue_eligible_fraction": round(float(test_df[eligible_col].mean()), 4),
        "rescue_eligible_fraction_on_unknown": round(float(test_df.loc[unknown_mask, eligible_col].mean()), 4),
        "rescue_eligible_fraction_on_known": round(float(test_df.loc[known_mask, eligible_col].mean()), 4),
    }
