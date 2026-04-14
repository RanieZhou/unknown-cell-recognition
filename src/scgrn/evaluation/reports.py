"""Markdown summaries and subgroup analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import SPLIT_COLUMN, SPLIT_TEST_UNKNOWN


def compute_class_regulon_means(df, auc_df, known_classes, unknown_class, label_col="true_label"):
    centroids = {}
    for cell_type in known_classes + [unknown_class]:
        cell_ids = df.loc[df[label_col] == cell_type, "cell_id"].astype(str)
        available = [cell_id for cell_id in cell_ids if cell_id in auc_df.index]
        if available:
            centroids[cell_type] = auc_df.loc[available].mean(axis=0)
    if not centroids:
        return pd.DataFrame()
    return pd.DataFrame(centroids).T


def compute_pairwise_diff(centroid_df, known_classes, target_class):
    if centroid_df.empty or target_class not in centroid_df.index:
        return pd.DataFrame()
    diffs = {}
    for known_class in known_classes:
        if known_class in centroid_df.index:
            diffs[f"{target_class}_vs_{known_class}"] = centroid_df.loc[target_class] - centroid_df.loc[known_class]
    return pd.DataFrame(diffs)


def build_interpretability_table(centroid_df, known_classes, unknown_class):
    if centroid_df.empty or unknown_class not in centroid_df.index:
        return pd.DataFrame()
    rows = []
    for regulon in centroid_df.columns:
        row = {"regulon": regulon}
        abs_diffs = []
        for cell_type in known_classes + [unknown_class]:
            if cell_type in centroid_df.index:
                row[f"{cell_type}_mean"] = round(float(centroid_df.loc[cell_type, regulon]), 6)
        for known_class in known_classes:
            if known_class in centroid_df.index:
                diff_value = float(centroid_df.loc[unknown_class, regulon] - centroid_df.loc[known_class, regulon])
                row[f"diff_vs_{known_class}"] = round(diff_value, 6)
                abs_diffs.append(abs(diff_value))
        row["abs_max_diff"] = round(float(max(abs_diffs)), 6) if abs_diffs else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("abs_max_diff", ascending=False).reset_index(drop=True)


def get_top_diff_regulons(diff_series, n_top=10):
    diff_series = diff_series.dropna()
    return diff_series.nlargest(n_top), diff_series.nsmallest(n_top)


def assign_asdc_subgroups(df, baseline_pred_col, target_pred_col, unknown_class):
    df = df.copy()
    test_unknown = df[(df[SPLIT_COLUMN] == SPLIT_TEST_UNKNOWN) & (df["true_label"] == unknown_class)].copy()
    baseline_pred = test_unknown[baseline_pred_col].astype(int).to_numpy()
    target_pred = test_unknown[target_pred_col].astype(int).to_numpy()
    subgroup = np.where(
        (baseline_pred == 1) & (target_pred == 1),
        "always_TP_ASDC",
        np.where((baseline_pred == 0) & (target_pred == 1), "rescued_ASDC", "still_FN_ASDC"),
    )
    test_unknown["asdc_subgroup"] = subgroup
    return test_unknown


def summarize_asdc_subgroups(asdc_df, feature_cols):
    rows = []
    for subgroup in ["always_TP_ASDC", "rescued_ASDC", "still_FN_ASDC"]:
        sub = asdc_df[asdc_df["asdc_subgroup"] == subgroup].copy()
        if sub.empty:
            continue
        row = {"asdc_subgroup": subgroup, "n_cells": int(len(sub))}
        if "nearest_grn_class" in sub.columns:
            row["nearest_grn_class_distribution"] = str(sub["nearest_grn_class"].value_counts(normalize=True).round(4).to_dict())
        if "lineage_bucket" in sub.columns:
            row["lineage_bucket_distribution"] = str(sub["lineage_bucket"].value_counts(normalize=True).round(4).to_dict())
        for col in feature_cols:
            if col in sub.columns:
                row[f"{col}_mean"] = round(float(sub[col].mean()), 6)
                row[f"{col}_std"] = round(float(sub[col].std(ddof=0)), 6)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_asdc_regulon_diff(asdc_df, auc_df):
    rows = []
    for group_a, group_b in [("rescued_ASDC", "still_FN_ASDC"), ("always_TP_ASDC", "still_FN_ASDC")]:
        cells_a = asdc_df.loc[asdc_df["asdc_subgroup"] == group_a, "cell_id"].astype(str)
        cells_b = asdc_df.loc[asdc_df["asdc_subgroup"] == group_b, "cell_id"].astype(str)
        avail_a = [cell for cell in cells_a if cell in auc_df.index]
        avail_b = [cell for cell in cells_b if cell in auc_df.index]
        if not avail_a or not avail_b:
            continue
        mean_a = auc_df.loc[avail_a].mean(axis=0)
        mean_b = auc_df.loc[avail_b].mean(axis=0)
        diff = (mean_a - mean_b).sort_values(key=lambda s: s.abs(), ascending=False)
        for rank, regulon in enumerate(diff.index, start=1):
            rows.append(
                {
                    "comparison": f"{group_a}_vs_{group_b}",
                    "regulon": regulon,
                    f"{group_a}_mean": round(float(mean_a[regulon]), 6),
                    f"{group_b}_mean": round(float(mean_b[regulon]), 6),
                    "diff": round(float(diff[regulon]), 6),
                    "abs_diff": round(float(abs(diff[regulon])), 6),
                    "rank_abs_diff": rank,
                }
            )
    return pd.DataFrame(rows)


def build_selective_fusion_summary(config, metrics_by_method, thresholds, gate_threshold, gate_stats, proximity_df, interpret_table, error_explanation, ratio_summary_df):
    data_cfg = config["data"]
    expr_metrics = metrics_by_method.get("expr_fused", {})
    selective_metrics = metrics_by_method.get("selective_fused_score", {})
    nearest_text = "N/A"
    if not proximity_df.empty:
        top_row = proximity_df.iloc[0]
        nearest_text = f"{top_row['nearest_class']} ({top_row['proportion']:.4f})"
    ratio_lines = []
    if not ratio_summary_df.empty:
        for ratio in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:
            sub = ratio_summary_df[(ratio_summary_df["ratio"].round(2) == round(ratio, 2)) & (ratio_summary_df["method"] == "selective_fused_score")]
            if not sub.empty:
                row = sub.iloc[0]
                ratio_lines.append(
                    f"- ratio {ratio:.2f}: AUROC {row['AUROC_mean']:.4f}+/-{row['AUROC_std']:.4f}, AUPR {row['AUPR_mean']:.4f}+/-{row['AUPR_std']:.4f}, FPR95 {row['FPR95_mean']:.4f}+/-{row['FPR95_std']:.4f}"
                )
    lines = [
        "# Selective Fusion Summary",
        "",
        "## Setup",
        f"- Dataset: {config['project']['dataset_name']}",
        f"- Label column: {data_cfg['label_column']}",
        f"- Known classes: {', '.join(data_cfg['known_classes'])}",
        f"- Unknown class: {', '.join(data_cfg['unknown_classes'])}",
        f"- Seed: {config['training']['seed']}",
        f"- Gate quantile: {config['rescue']['gate_quantile']}",
        f"- Alpha: {config['rescue']['selective_alpha']}",
        "",
        "## Main Findings",
        f"1. selective fusion vs expr_fused: expr_fused AUROC/AUPR/FPR95 = {expr_metrics.get('AUROC', np.nan)} / {expr_metrics.get('AUPR', np.nan)} / {expr_metrics.get('FPR95', np.nan)}; selective_fused_score = {selective_metrics.get('AUROC', np.nan)} / {selective_metrics.get('AUPR', np.nan)} / {selective_metrics.get('FPR95', np.nan)}.",
        f"2. gated cells fraction: total={gate_stats.get('gated_fraction', np.nan)}, unknown={gate_stats.get('gated_fraction_on_unknown', np.nan)}, known={gate_stats.get('gated_fraction_on_known', np.nan)}.",
        f"3. ASDC nearest GRN class: {nearest_text}",
        "4. unknown ratio trend:",
    ]
    lines.extend(ratio_lines if ratio_lines else ["- ratio ablation summary unavailable."])
    lines.extend(
        [
            "",
            "## Thresholds",
            f"- expr_uncertainty gate threshold: {gate_threshold:.6f}",
        ]
    )
    for method_name, threshold in thresholds.items():
        lines.append(f"- {method_name}: {threshold:.6f}")
    top_abs = interpret_table.head(10)["regulon"].tolist() if not interpret_table.empty else []
    lines.extend(
        [
            "",
            "## GRN Interpretability",
            f"- Top absolute-difference regulons: {', '.join(top_abs[:10]) if top_abs else 'N/A'}",
            "",
            "## Error Analysis",
            f"- {error_explanation}",
        ]
    )
    return "\n".join(lines)


def build_lineage_rescue_summary(metrics_by_method, bucket_metrics_df, component_ablation_df, asdc_summary_df, threshold_exports):
    def metric(method, key):
        return metrics_by_method.get(method, {}).get(key, np.nan)

    def bucket_row(method, bucket):
        sub = bucket_metrics_df[(bucket_metrics_df["method"] == method) & (bucket_metrics_df["lineage_bucket"] == bucket)]
        return sub.iloc[0] if not sub.empty else None

    pdc_expr = bucket_row("expr_fused", "pDC_like")
    pdc_selective = bucket_row("selective_fused_score", "pDC_like")
    pdc_rescue = bucket_row("lineage_selective_rescue_globalT", "pDC_like")
    cdc2_selective = bucket_row("selective_fused_score", "cDC2_like")
    cdc2_rescue = bucket_row("lineage_selective_rescue_globalT", "cDC2_like")
    full_row = component_ablation_df[component_ablation_df["variant"] == "bucket_threshold_full"]
    no_bucket_row = component_ablation_df[component_ablation_df["variant"] == "global_threshold_only"]
    no_cdc2_row = component_ablation_df[component_ablation_df["variant"] == "allow_cdc2_block_relaxation"]
    no_positive_row = component_ablation_df[component_ablation_df["variant"] == "disable_positive_only_gate"]
    full_row = full_row.iloc[0] if not full_row.empty else None
    no_bucket_row = no_bucket_row.iloc[0] if not no_bucket_row.empty else None
    no_cdc2_row = no_cdc2_row.iloc[0] if not no_cdc2_row.empty else None
    no_positive_row = no_positive_row.iloc[0] if not no_positive_row.empty else None
    subgroup_counts = asdc_summary_df.set_index("asdc_subgroup")["n_cells"].to_dict() if not asdc_summary_df.empty else {}
    lines = [
        "# Lineage Rescue Summary",
        "",
        "## Main Findings",
        f"1. lineage_selective_rescue_globalT vs selective_fused_score: AUROC {metric('selective_fused_score', 'AUROC')} -> {metric('lineage_selective_rescue_globalT', 'AUROC')}, AUPR {metric('selective_fused_score', 'AUPR')} -> {metric('lineage_selective_rescue_globalT', 'AUPR')}, Macro-F1 {metric('selective_fused_score', 'Macro_F1')} -> {metric('lineage_selective_rescue_globalT', 'Macro_F1')}.",
        f"2. pDC_like bucket contribution: unknown recall expr_fused={pdc_expr['Recall_on_unknown'] if pdc_expr is not None else 'NA'}, selective_fused_score={pdc_selective['Recall_on_unknown'] if pdc_selective is not None else 'NA'}, lineage_rescue={pdc_rescue['Recall_on_unknown'] if pdc_rescue is not None else 'NA'}.",
        f"3. cDC2_like FP control: FP rate selective_fused_score={cdc2_selective['FP_rate_on_known'] if cdc2_selective is not None else 'NA'}, lineage_rescue={cdc2_rescue['FP_rate_on_known'] if cdc2_rescue is not None else 'NA'}; no_cDC2_block net_gain={no_cdc2_row['net_gain'] if no_cdc2_row is not None else 'NA'}.",
        f"4. positive-only rescue stability: full_variant net_gain={full_row['net_gain'] if full_row is not None else 'NA'}, no_positive_only net_gain={no_positive_row['net_gain'] if no_positive_row is not None else 'NA'}.",
        f"5. Bucket-aware gating remains available through the component ablation outputs; maintained final reporting keeps only globalT. Current globalT Macro-F1={metric('lineage_selective_rescue_globalT', 'Macro_F1')}; component full net_gain={full_row['net_gain'] if full_row is not None else 'NA'}, no_bucket_threshold net_gain={no_bucket_row['net_gain'] if no_bucket_row is not None else 'NA'}.",
        f"6. ASDC subgroup counts: always_TP={subgroup_counts.get('always_TP_ASDC', 'NA')}, rescued={subgroup_counts.get('rescued_ASDC', 'NA')}, still_FN={subgroup_counts.get('still_FN_ASDC', 'NA')}.",
        "",
        "## Threshold Notes",
        "- Gate thresholds fitted on val_known 90th percentile of expr_uncertainty_score with bucket fallback at n < min_bucket_val_cells.",
        f"- Global score threshold: {threshold_exports.get('lineage_selective_rescue_globalT', {}).get('global', 'NA')}",
    ]
    return "\n".join(lines)
