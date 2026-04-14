"""
E005 Step 5: Dual fusion, evaluation, and biological interpretation
=====================================================================
1. Load expr scores + GRN scores
2. ECDF normalization on val_known
3. Dual fusion: expr_pct + λ * max(0, grn_pct - τg)
4. Threshold determination (95th percentile on val_known)
5. Evaluation: AUROC, AUPR, FPR95, accuracy, macro-F1
6. Biological interpretation: case studies + plots

Usage:
    python scripts/run_e005_dual_fusion.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    accuracy_score, f1_score, confusion_matrix
)
from scipy.stats import rankdata
import anndata as ad

# ============================================================
# Config
# ============================================================
OUTPUT_DIR = r"D:\Desktop\研一\code\scGRN\outputs\E005\v2"   # v2
OUTPUT_DIR_V1 = r"D:\Desktop\研一\code\scGRN\outputs\E005\v1"  # expression scores from v1
DATA_PATH = r"D:\Desktop\研一\code\scGRN\data\human_immune_health_atlas_dc.patched.h5ad"
os.makedirs(OUTPUT_DIR, exist_ok=True)

KNOWN_CLASSES = ["cDC2", "pDC", "cDC1"]
UNKNOWN_CLASS = "ASDC"
LABEL_KEY = "AIFI_L2"

# v2 改进：rank 平均融合策略（替代 v1 的线性 expr_pct + lambda*grn_bonus）
GRN_WEIGHT = 0.3      # GRN rank 的权重（expr rank 权重 = 1 - GRN_WEIGHT）
PERCENTILE_THRESHOLD = 95


def log(msg):
    print(f"[E005-fusion] {msg}", flush=True)


# ============================================================
# 1. Load scores
# ============================================================
def load_scores():
    log("Loading scores...")
    expr_df = pd.read_csv(os.path.join(OUTPUT_DIR_V1, "E005_expression_scores.csv"))
    grn_df = pd.read_csv(os.path.join(OUTPUT_DIR, "E005_grn_scores.csv"))

    # Merge
    df = expr_df.merge(grn_df, on="cell_id", how="left")
    log(f"  Merged DataFrame shape: {df.shape}")
    log(f"  Columns: {list(df.columns)}")

    # Check for NaN/inf in GRN scores
    df["grn_distance_score"] = df["grn_distance_score"].replace([np.inf, -np.inf], np.nan)
    n_nan = df["grn_distance_score"].isna().sum()
    if n_nan > 0:
        finite_vals = df["grn_distance_score"].dropna()
        fill_val = finite_vals.max() if len(finite_vals) > 0 else 1.0
        log(f"  WARNING: {n_nan} cells have NaN/inf GRN scores. Filling with {fill_val:.4f}.")
        df["grn_distance_score"] = df["grn_distance_score"].fillna(fill_val)

    return df


# ============================================================
# 2. ECDF normalization
# ============================================================
def ecdf_normalize(scores, reference_scores):
    """Convert scores to empirical CDF values based on reference distribution."""
    n_ref = len(reference_scores)
    sorted_ref = np.sort(reference_scores)

    result = np.searchsorted(sorted_ref, scores, side="right") / n_ref
    return result


def normalize_scores(df):
    """v2: rank-based normalization on val_known reference."""
    log("Normalizing scores with ECDF on val_known...")

    val_mask = df["E005_split"] == "val_known"

    val_expr = df.loc[val_mask, "expr_fused"].values
    df["expr_pct"] = ecdf_normalize(df["expr_fused"].values, val_expr)

    val_grn = df.loc[val_mask, "grn_distance_score"].replace([np.inf, -np.inf], np.nan).dropna().values
    grn_vals = df["grn_distance_score"].replace([np.inf, -np.inf], np.nan)
    if len(val_grn) > 0 and grn_vals.notna().any():
        fill_val = np.nanmax(grn_vals)
        grn_vals_filled = grn_vals.fillna(fill_val)
        df["grn_pct"] = ecdf_normalize(grn_vals_filled.values, val_grn)
    else:
        df["grn_pct"] = 0.5  # neutral when GRN all invalid

    log(f"  expr_pct range on val: [{df.loc[val_mask, 'expr_pct'].min():.4f}, {df.loc[val_mask, 'expr_pct'].max():.4f}]")
    log(f"  grn_pct range on val: [{df.loc[val_mask, 'grn_pct'].min():.4f}, {df.loc[val_mask, 'grn_pct'].max():.4f}]")

    return df


# ============================================================
# 3. Dual fusion — v2: rank-average
# ============================================================
def compute_dual_fusion(df):
    """v2: weighted rank average.
    dual_fused = (1 - GRN_WEIGHT) * expr_pct + GRN_WEIGHT * grn_pct
    Compared to v1 linear bonus, this is more conservative:
    GRN can help but cannot push score above expr_pct alone by more than GRN_WEIGHT.
    """
    log("Computing dual fusion score (v2: rank-average)...")
    log(f"  GRN_WEIGHT={GRN_WEIGHT}, expr_weight={1-GRN_WEIGHT}")

    df["dual_fused"] = (1 - GRN_WEIGHT) * df["expr_pct"].values + GRN_WEIGHT * df["grn_pct"].values

    log(f"  dual_fused range: [{df['dual_fused'].min():.4f}, {df['dual_fused'].max():.4f}]")

    return df, GRN_WEIGHT  # return weight instead of tau_g


# ============================================================
# 4. Threshold determination
# ============================================================
def compute_thresholds(df):
    log("Computing thresholds on val_known...")

    val_mask = df["E005_split"] == "val_known"

    thresholds = {}

    # For each score type
    for score_name in ["entropy_norm", "distance_norm", "expr_fused", "expr_pct", "dual_fused", "grn_distance_score", "grn_pct"]:
        if score_name in df.columns:
            vals = df.loc[val_mask, score_name].values
            finite_vals = vals[np.isfinite(vals)]
            if len(finite_vals) > 0:
                t = np.percentile(finite_vals, PERCENTILE_THRESHOLD)
                thresholds[score_name] = t
                log(f"  T({score_name}): {t:.4f}")
            else:
                log(f"  T({score_name}): SKIPPED (no finite values)")

    return thresholds


# ============================================================
# 5. Evaluation
# ============================================================
def compute_fpr_at_tpr(y_true, y_score, tpr_target=0.95):
    """Compute FPR when TPR >= tpr_target."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # Find the first point where TPR >= target
    idx = np.where(tpr >= tpr_target)[0]
    if len(idx) == 0:
        return 1.0
    return fpr[idx[0]]


def evaluate(df, thresholds):
    log("Evaluating...")

    test_mask = df["E005_split"].isin(["test_known", "test_unknown"])
    test_df = df[test_mask].copy()

    y_true_binary = (test_df["E005_split"] == "test_unknown").astype(int).values
    y_true_label = test_df["true_label"].values

    metrics = {}

    # Score columns to evaluate
    score_columns = {
        "entropy": "entropy_norm",
        "expr_distance": "distance_norm",
        "expr_fused": "expr_fused",
        "dual_fused": "dual_fused",
        "grn_distance": "grn_distance_score",
    }

    for name, col in score_columns.items():
        if col not in test_df.columns:
            continue

        scores = test_df[col].values.copy()

        # Handle NaN/inf in scores
        scores = np.where(np.isfinite(scores), scores, np.nan)
        nan_mask = np.isnan(scores)
        if nan_mask.all():
            log(f"  {name}: SKIPPED (all NaN)")
            continue
        if nan_mask.any():
            finite_max = np.nanmax(scores)
            scores = np.where(nan_mask, finite_max, scores)

        # Unknown detection metrics (unknown = positive class)
        auroc = roc_auc_score(y_true_binary, scores)
        aupr = average_precision_score(y_true_binary, scores)
        fpr95 = compute_fpr_at_tpr(y_true_binary, scores, 0.95)

        metrics[name] = {
            "AUROC": round(auroc, 4),
            "AUPR": round(aupr, 4),
            "FPR95": round(fpr95, 4),
        }

        log(f"  {name}: AUROC={auroc:.4f}, AUPR={aupr:.4f}, FPR95={fpr95:.4f}")

    # Known-class classification (using expr_fused threshold)
    T_expr = thresholds.get("expr_fused", thresholds.get("expr_pct", 0.95))

    # For cells predicted as known (below threshold), check classification accuracy
    test_known_df = test_df[test_df["E005_split"] == "test_known"]
    pred_labels = test_known_df["predicted_label"].values
    true_labels = test_known_df["true_label"].values

    acc = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro")
    metrics["known_classification"] = {
        "accuracy": round(acc, 4),
        "macro_F1": round(macro_f1, 4),
    }
    log(f"  Known classification: accuracy={acc:.4f}, macro-F1={macro_f1:.4f}")

    return metrics, test_df


# ============================================================
# 6. Plotting
# ============================================================
def plot_roc_curves(test_df, metrics):
    log("Plotting ROC curves...")

    y_true = (test_df["E005_split"] == "test_unknown").astype(int).values

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    score_columns = {
        "entropy": ("entropy_norm", "#FF6B6B"),
        "expr_distance": ("distance_norm", "#4ECDC4"),
        "expr_fused": ("expr_fused", "#45B7D1"),
        "dual_fused": ("dual_fused", "#F7DC6F"),
        "grn_distance": ("grn_distance_score", "#BB8FCE"),
    }

    for name, (col, color) in score_columns.items():
        if col not in test_df.columns or name not in metrics:
            continue
        scores = test_df[col].values.copy()
        scores = np.where(np.isfinite(scores), scores, np.nanmax(scores[np.isfinite(scores)]))
        fpr, tpr, _ = roc_curve(y_true, scores)
        auroc = metrics[name]["AUROC"]
        ax.plot(fpr, tpr, label=f"{name} (AUROC={auroc:.4f})", color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("E005: Unknown Detection ROC Curves", fontsize=14)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "E005_roc.png"), dpi=150)
    plt.close()
    log("  Saved E005_roc.png")


def plot_umap_dual_fused(df):
    log("Plotting UMAP colored by dual_fused...")

    adata = ad.read_h5ad(DATA_PATH)
    umap_coords = adata.obsm["X_umap"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: True labels
    ax = axes[0]
    label_colors = {"cDC2": "#E74C3C", "pDC": "#3498DB", "cDC1": "#2ECC71", "ASDC": "#F39C12"}
    for label, color in label_colors.items():
        mask = adata.obs[LABEL_KEY] == label
        ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                   c=color, label=label, s=1, alpha=0.5)
    ax.set_title("True Labels", fontsize=13)
    ax.legend(markerscale=5, fontsize=9)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

    # Panel 2: expr_fused score
    ax = axes[1]
    cell_order = list(adata.obs_names)
    cell_to_idx = {c: i for i, c in enumerate(cell_order)}
    expr_scores = np.full(len(cell_order), np.nan)
    for _, row in df.iterrows():
        if row["cell_id"] in cell_to_idx:
            expr_scores[cell_to_idx[row["cell_id"]]] = row["expr_fused"]

    sc = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                    c=expr_scores, cmap="YlOrRd", s=1, alpha=0.5)
    plt.colorbar(sc, ax=ax, shrink=0.8)
    ax.set_title("expr_fused Score", fontsize=13)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

    # Panel 3: dual_fused score
    ax = axes[2]
    dual_scores = np.full(len(cell_order), np.nan)
    for _, row in df.iterrows():
        if row["cell_id"] in cell_to_idx:
            dual_scores[cell_to_idx[row["cell_id"]]] = row["dual_fused"]

    sc = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                    c=dual_scores, cmap="YlOrRd", s=1, alpha=0.5)
    plt.colorbar(sc, ax=ax, shrink=0.8)
    ax.set_title("dual_fused Score", fontsize=13)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

    plt.suptitle("E005: UMAP Visualization", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "E005_umap_dual_fused.png"), dpi=150)
    plt.close()
    log("  Saved E005_umap_dual_fused.png")


def plot_regulon_heatmap(df):
    log("Plotting regulon activity heatmap...")

    auc_path = os.path.join(OUTPUT_DIR, "E005_aucell_all_cells.csv")
    if not os.path.exists(auc_path):
        log("  WARNING: AUCell all cells CSV not found. Skipping heatmap.")
        return

    auc_all = pd.read_csv(auc_path, index_col=0)
    log(f"  AUCell matrix shape: {auc_all.shape}")

    # Compute centroids for each class + unknown
    centroids = {}
    for cls in KNOWN_CLASSES + [UNKNOWN_CLASS]:
        cls_cells = df[df["true_label"] == cls]["cell_id"].values
        cls_cells_avail = [c for c in cls_cells if c in auc_all.index]
        if len(cls_cells_avail) > 0:
            centroids[cls] = auc_all.loc[cls_cells_avail].mean(axis=0)

    if len(centroids) < 2:
        log("  WARNING: Not enough centroids for heatmap.")
        return

    centroid_df = pd.DataFrame(centroids).T

    # Select top variable regulons
    regulon_var = centroid_df.var(axis=0)
    top_regulons = regulon_var.nlargest(30).index.tolist()

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        centroid_df[top_regulons],
        ax=ax,
        cmap="YlOrRd",
        xticklabels=True,
        yticklabels=True,
        linewidths=0.5,
    )
    ax.set_title("E005: Top 30 Variable Regulons (Class Centroids)", fontsize=14)
    ax.set_xlabel("Regulon", fontsize=11)
    ax.set_ylabel("Cell Type", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "E005_regulon_heatmap.png"), dpi=150)
    plt.close()
    log("  Saved E005_regulon_heatmap.png")


def plot_top_regulon_barplot(df):
    log("Plotting top regulon barplot (ASDC vs nearest known)...")

    auc_path = os.path.join(OUTPUT_DIR, "E005_aucell_all_cells.csv")
    if not os.path.exists(auc_path):
        log("  WARNING: AUCell CSV not found. Skipping barplot.")
        return

    auc_all = pd.read_csv(auc_path, index_col=0)

    # ASDC centroid
    asdc_cells = df[df["true_label"] == UNKNOWN_CLASS]["cell_id"].values
    asdc_avail = [c for c in asdc_cells if c in auc_all.index]

    # Find nearest known class (most common nearest_grn_class for ASDC)
    asdc_df = df[df["true_label"] == UNKNOWN_CLASS]
    nearest_cls = None
    if "nearest_grn_class" in asdc_df.columns:
        grn_mode = asdc_df["nearest_grn_class"].dropna().mode()
        if len(grn_mode) > 0:
            nearest_cls = grn_mode.iloc[0]
    if nearest_cls is None and "nearest_known_class" in asdc_df.columns:
        nk_mode = asdc_df["nearest_known_class"].dropna().mode()
        if len(nk_mode) > 0:
            nearest_cls = nk_mode.iloc[0]
    if nearest_cls is None:
        nearest_cls = KNOWN_CLASSES[0]
        log(f"  WARNING: Could not determine nearest class, defaulting to {nearest_cls}")

    nearest_cells = df[(df["true_label"] == nearest_cls) & 
                       (df["E005_split"] == "train_known")]["cell_id"].values
    nearest_avail = [c for c in nearest_cells if c in auc_all.index]

    if len(asdc_avail) == 0 or len(nearest_avail) == 0:
        log("  WARNING: Not enough cells for barplot.")
        return

    asdc_mean = auc_all.loc[asdc_avail].mean(axis=0)
    nearest_mean = auc_all.loc[nearest_avail].mean(axis=0)

    # Compute difference
    diff = asdc_mean - nearest_mean
    top_up = diff.nlargest(10)
    top_down = diff.nsmallest(10)
    top_diff = pd.concat([top_up, top_down])

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#E74C3C" if v > 0 else "#3498DB" for v in top_diff.values]
    bars = ax.barh(range(len(top_diff)), top_diff.values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(top_diff)))
    ax.set_yticklabels(top_diff.index, fontsize=9)
    ax.set_xlabel("Mean AUC Difference (ASDC - Nearest Known)", fontsize=11)
    ax.set_title(f"E005: Top Differentially Active Regulons\n(ASDC vs {nearest_cls})", fontsize=13)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "E005_top_regulon_barplot.png"), dpi=150)
    plt.close()
    log("  Saved E005_top_regulon_barplot.png")


# ============================================================
# 7. Case study
# ============================================================
def case_study(df, thresholds):
    log("Running case study analysis...")

    T_expr = thresholds.get("expr_fused", 0)
    T_dual = thresholds.get("dual_fused", 0)

    test_mask = df["E005_split"].isin(["test_known", "test_unknown"])
    test_df = df[test_mask].copy()

    # 1. True positive unknown: ASDC with high dual_fused
    tp_unknown = test_df[
        (test_df["true_label"] == UNKNOWN_CLASS) &
        (test_df["dual_fused"] > T_dual)
    ].copy()
    tp_unknown["case_type"] = "TP_unknown"

    # 2. False positive known: known cell rejected by dual_fused
    fp_known = test_df[
        (test_df["true_label"].isin(KNOWN_CLASSES)) &
        (test_df["dual_fused"] > T_dual)
    ].copy()
    fp_known["case_type"] = "FP_known"

    # 3. Expression-only miss but dual rescue
    expr_miss_dual_rescue = test_df[
        (test_df["true_label"] == UNKNOWN_CLASS) &
        (test_df["expr_fused"] <= T_expr) &
        (test_df["dual_fused"] > T_dual)
    ].copy()
    expr_miss_dual_rescue["case_type"] = "dual_rescue"

    case_df = pd.concat([tp_unknown, fp_known, expr_miss_dual_rescue], ignore_index=True)
    case_df = case_df.drop_duplicates(subset=["cell_id"])

    log(f"  TP unknown: {len(tp_unknown)}")
    log(f"  FP known: {len(fp_known)}")
    log(f"  Dual rescue: {len(expr_miss_dual_rescue)}")

    # Save case study CSV
    cols_to_save = [
        "cell_id", "case_type", "true_label", "predicted_label",
        "nearest_known_class", "entropy", "latent_distance",
        "expr_fused", "grn_distance_score", "dual_fused",
        "expr_pct", "grn_pct"
    ]
    cols_available = [c for c in cols_to_save if c in case_df.columns]
    case_df[cols_available].to_csv(os.path.join(OUTPUT_DIR, "E005_case_study.csv"), index=False)
    log("  Saved E005_case_study.csv")

    return case_df


# ============================================================
# 8. Summary report
# ============================================================
def generate_summary(metrics, thresholds, df):
    log("Generating summary report...")

    summary_lines = [
        "# E005 Experiment Summary",
        "",
        "## Dataset",
        f"- Data: human_immune_health_atlas_dc.patched.h5ad",
        f"- Known classes: {', '.join(KNOWN_CLASSES)}",
        f"- Unknown class: {UNKNOWN_CLASS}",
        "",
        "## Split",
    ]

    for split in ["train_known", "val_known", "test_known", "test_unknown"]:
        n = (df["E005_split"] == split).sum()
        summary_lines.append(f"- {split}: {n} cells")

    summary_lines.extend([
        "",
        "## Thresholds (95th on val_known)",
    ])
    for name, val in thresholds.items():
        summary_lines.append(f"- {name}: {val:.4f}")

    summary_lines.extend([
        "",
        "## Unknown Detection Metrics",
        "",
        "| Method | AUROC | AUPR | FPR95 |",
        "|--------|-------|------|-------|",
    ])

    for name in ["entropy", "expr_distance", "expr_fused", "grn_distance", "dual_fused"]:
        if name in metrics:
            m = metrics[name]
            summary_lines.append(
                f"| {name} | {m['AUROC']:.4f} | {m['AUPR']:.4f} | {m['FPR95']:.4f} |"
            )

    if "known_classification" in metrics:
        kc = metrics["known_classification"]
        summary_lines.extend([
            "",
            "## Known Classification (test_known)",
            f"- Accuracy: {kc['accuracy']:.4f}",
            f"- Macro-F1: {kc['macro_F1']:.4f}",
        ])

    summary_lines.extend([
        "",
        f"## Fusion Parameters (v2: rank-average)",
        f"- GRN_WEIGHT = {GRN_WEIGHT}",
        f"- expr_weight = {1 - GRN_WEIGHT}",
        f"- Fusion: dual = (1-w)*expr_pct + w*grn_pct",
        f"- v2 changes: gene_filter=0.5%, NES_threshold=2.5, rank-average fusion",
    ])

    summary_text = "\n".join(summary_lines)

    with open(os.path.join(OUTPUT_DIR, "E005_summary.md"), "w", encoding="utf-8") as f:
        f.write(summary_text)
    log("  Saved E005_summary.md")

    # Also save metrics as JSON
    with open(os.path.join(OUTPUT_DIR, "E005_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    log("  Saved E005_metrics.json")


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load
    df = load_scores()

    # 2. Normalize
    df = normalize_scores(df)

    # 3. Dual fusion
    df, tau_g_value = compute_dual_fusion(df)

    # 4. Thresholds
    thresholds = compute_thresholds(df)

    # 5. Evaluate
    metrics, test_df = evaluate(df, thresholds)

    # 6. Save fusion scores
    fusion_cols = ["cell_id", "E005_split", "true_label", "expr_fused", "expr_pct",
                   "grn_distance_score", "grn_pct", "dual_fused"]
    fusion_cols_avail = [c for c in fusion_cols if c in df.columns]
    df[fusion_cols_avail].to_csv(os.path.join(OUTPUT_DIR, "E005_fusion_scores.csv"), index=False)
    log("Saved E005_fusion_scores.csv")

    # 7. Plots
    plot_roc_curves(test_df, metrics)
    plot_umap_dual_fused(df)
    plot_regulon_heatmap(df)
    plot_top_regulon_barplot(df)

    # 8. Case study
    case_df = case_study(df, thresholds)

    # 9. Summary
    generate_summary(metrics, thresholds, df)

    # Save full test eval
    test_eval = df[df["E005_split"].isin(["test_known", "test_unknown"])]
    test_eval.to_csv(os.path.join(OUTPUT_DIR, "E005_test_eval_obs.csv"), index=False)
    log("Saved E005_test_eval_obs.csv")

    log("=" * 60)
    log("E005 COMPLETE!")
    log("=" * 60)


if __name__ == "__main__":
    main()
