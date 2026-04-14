"""Plotting helpers for the standardized pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

from ..constants import SPLIT_COLUMN, SPLIT_TEST_KNOWN, SPLIT_TEST_UNKNOWN
from .metrics import sanitize_scores


def plot_roc_curves(df, method_specs, output_path):
    test_df = df[df[SPLIT_COLUMN].isin([SPLIT_TEST_KNOWN, SPLIT_TEST_UNKNOWN])].copy()
    y_true = (test_df[SPLIT_COLUMN] == SPLIT_TEST_UNKNOWN).astype(int).to_numpy()
    if np.unique(y_true).size < 2:
        return
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2"]
    plt.figure(figsize=(8.5, 6.2))
    for idx, (method_name, score_col, auroc) in enumerate(method_specs):
        if score_col not in test_df.columns or pd.isna(auroc):
            continue
        scores = sanitize_scores(test_df[score_col].to_numpy(dtype=float))
        if np.isnan(scores).all():
            continue
        fpr, tpr, _ = roc_curve(y_true, scores)
        plt.plot(fpr, tpr, linewidth=2, color=palette[idx % len(palette)], label=f"{method_name} (AUROC={auroc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.35)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_ratio_ablation(summary_df, output_path):
    if summary_df.empty:
        return
    metrics = ["AUROC", "AUPR", "FPR95"]
    methods = summary_df["method"].dropna().unique().tolist()
    palette = {"expr_fused": "#1f77b4", "selective_fused_score": "#d62728", "grn_distance": "#2ca02c"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    for ax, metric in zip(axes, metrics):
        for method in methods:
            sub = summary_df[summary_df["method"] == method].sort_values("ratio")
            if sub.empty:
                continue
            ax.errorbar(
                sub["ratio"],
                sub[f"{metric}_mean"],
                yerr=sub[f"{metric}_std"],
                marker="o",
                linewidth=2,
                capsize=3,
                color=palette.get(method),
                label=method,
            )
        ax.set_title(metric)
        ax.set_xlabel("Unknown Ratio")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Metric")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)), frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_regulon_heatmap(centroid_df, output_path, title, n_top=30):
    if centroid_df.empty:
        return
    regulon_var = centroid_df.var(axis=0).sort_values(ascending=False)
    top_regulons = regulon_var.head(min(n_top, len(regulon_var))).index.tolist()
    if not top_regulons:
        return
    plt.figure(figsize=(16, 5))
    sns.heatmap(centroid_df.loc[:, top_regulons], cmap="YlOrRd", linewidths=0.5, xticklabels=True, yticklabels=True)
    plt.title(title)
    plt.xlabel("Regulon")
    plt.ylabel("Group")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_top_regulon_barplot(diff_series, comparison_name, output_path, n_top=10):
    diff_series = diff_series.dropna()
    if diff_series.empty:
        return
    top_up = diff_series.nlargest(min(n_top, len(diff_series)))
    top_down = diff_series.nsmallest(min(n_top, len(diff_series)))
    plot_df = pd.concat([top_up, top_down]).sort_values()
    colors = ["#d62728" if value > 0 else "#1f77b4" for value in plot_df.values]
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(plot_df)), plot_df.values, color=colors, edgecolor="white")
    plt.yticks(range(len(plot_df)), plot_df.index, fontsize=9)
    plt.axvline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Mean Regulon Activity Difference")
    plt.title(comparison_name)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_bucket_barplot(bucket_metrics_df, output_path):
    if bucket_metrics_df.empty:
        return
    methods = ["expr_fused", "selective_fused_score", "lineage_selective_rescue_globalT"]
    buckets = bucket_metrics_df["lineage_bucket"].dropna().unique().tolist()
    colors = {"expr_fused": "#1f77b4", "selective_fused_score": "#d62728", "lineage_selective_rescue_globalT": "#2ca02c"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    width = 0.22
    x = np.arange(len(buckets))
    for offset_idx, method in enumerate(methods):
        sub = bucket_metrics_df[bucket_metrics_df["method"] == method].set_index("lineage_bucket")
        recall_vals = [sub.loc[bucket, "Recall_on_unknown"] if bucket in sub.index else np.nan for bucket in buckets]
        fp_vals = [sub.loc[bucket, "FP_rate_on_known"] if bucket in sub.index else np.nan for bucket in buckets]
        offset = (offset_idx - 1) * width
        axes[0].bar(x + offset, recall_vals, width=width, color=colors[method], label=method)
        axes[1].bar(x + offset, fp_vals, width=width, color=colors[method], label=method)
    axes[0].set_title("Unknown Recall by Lineage Bucket")
    axes[1].set_title("Known FP Rate by Lineage Bucket")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(buckets)
        ax.grid(axis="y", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_asdc_subgroup_heatmap(asdc_df, auc_df, output_path, n_top=25):
    group_means = {}
    for subgroup in ["always_TP_ASDC", "rescued_ASDC", "still_FN_ASDC"]:
        cells = asdc_df.loc[asdc_df["asdc_subgroup"] == subgroup, "cell_id"].astype(str)
        available = [cell for cell in cells if cell in auc_df.index]
        if available:
            group_means[subgroup] = auc_df.loc[available].mean(axis=0)
    if len(group_means) < 2:
        return
    heatmap_df = pd.DataFrame(group_means).T
    top_regulons = heatmap_df.var(axis=0).sort_values(ascending=False).head(min(n_top, heatmap_df.shape[1])).index
    if len(top_regulons) == 0:
        return
    plt.figure(figsize=(14, 4.5))
    sns.heatmap(heatmap_df.loc[:, top_regulons], cmap="YlOrRd", linewidths=0.5)
    plt.title("ASDC Subgroup Regulon Heatmap")
    plt.xlabel("Regulon")
    plt.ylabel("ASDC Subgroup")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
