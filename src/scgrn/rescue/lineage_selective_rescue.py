"""Mainline rescue orchestration for selective fusion and lineage rescue behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import (
    DEFAULT_METHOD_ORDER,
    SPLIT_COLUMN,
    SPLIT_TEST_KNOWN,
    SPLIT_TEST_UNKNOWN,
    SPLIT_VAL,
    TEST_SPLITS,
)
from ..evaluation.metrics import (
    build_rescue_delta_analysis,
    compute_all_metrics,
    compute_bucket_metrics,
    compute_gate_stats,
    compute_rescue_stats,
    evaluate_method,
    metrics_to_frame,
    summarize_gate_activity,
)
from ..evaluation.reports import (
    assign_asdc_subgroups,
    build_interpretability_table,
    build_lineage_rescue_summary,
    build_selective_fusion_summary,
    compute_asdc_regulon_diff,
    compute_class_regulon_means,
    compute_pairwise_diff,
    summarize_asdc_subgroups,
)
from .corridor import assign_lineage_buckets, known_buckets
from .rescue_score import (
    add_selective_fusion_scores,
    build_variant,
    compute_expr_margin,
    compute_grn_margin,
    compute_grn_proximity,
    error_analysis,
)
from .threshold import (
    compute_threshold_on_val,
    fit_thresholds_by_bucket,
    predict_with_global_threshold,
)


def build_dual_fused_v2(df: pd.DataFrame) -> pd.DataFrame:
    val_mask = df[SPLIT_COLUMN] == SPLIT_VAL
    scores = df["expr_fused"].to_numpy(dtype=float)
    ref = df.loc[val_mask, "expr_fused"].to_numpy(dtype=float)
    sorted_ref = np.sort(ref)
    df = df.copy()
    df["expr_pct"] = np.searchsorted(sorted_ref, scores, side="right") / len(sorted_ref)

    grn_ref = df.loc[val_mask, "grn_distance_score"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    grn_vals = df["grn_distance_score"].replace([np.inf, -np.inf], np.nan)
    fill_val = float(np.nanmax(grn_vals.to_numpy(dtype=float))) if grn_vals.notna().any() else 1.0
    grn_sorted_ref = np.sort(grn_ref) if len(grn_ref) else np.array([0.5])
    df["grn_pct"] = (
        np.searchsorted(grn_sorted_ref, grn_vals.fillna(fill_val).to_numpy(dtype=float), side="right") / len(grn_sorted_ref)
        if len(grn_ref)
        else 0.5
    )
    df["dual_fused_v2_rankavg"] = 0.7 * df["expr_pct"] + 0.3 * df["grn_pct"]
    return df


def run_ratio_ablation(df, methods=None, n_repeats=20, base_seed=42):
    methods = methods or [("expr_fused", "expr_fused"), ("selective_fused_score", "selective_fused_score"), ("grn_distance", "grn_distance")]
    target_ratios = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

    def compute_subset_sizes(n_known, n_unknown, ratio):
        if ratio <= 0 or ratio >= 1:
            return 0, 0
        n_unknown_use = n_unknown
        n_known_use = int(round(n_unknown_use * (1.0 - ratio) / ratio))
        if n_known_use > n_known:
            n_known_use = n_known
            n_unknown_use = int(round(n_known_use * ratio / (1.0 - ratio)))
            n_unknown_use = min(n_unknown_use, n_unknown)
        return int(n_known_use), int(n_unknown_use)

    test_known = df[df[SPLIT_COLUMN] == "test_known"].copy()
    test_unknown = df[df[SPLIT_COLUMN] == "test_unknown"].copy()
    results = []
    for ratio in target_ratios:
        n_known_use, n_unknown_use = compute_subset_sizes(len(test_known), len(test_unknown), ratio)
        if n_known_use <= 0 or n_unknown_use <= 0:
            continue
        for rep in range(n_repeats):
            rng = np.random.default_rng(base_seed + rep)
            known_idx = rng.choice(len(test_known), size=n_known_use, replace=False)
            unknown_idx = rng.choice(len(test_unknown), size=n_unknown_use, replace=False)
            subset = pd.concat([test_known.iloc[known_idx].copy(), test_unknown.iloc[unknown_idx].copy()], ignore_index=True)
            y_true = (subset[SPLIT_COLUMN] == "test_unknown").astype(int).to_numpy()
            actual_ratio = float(y_true.mean()) if y_true.size else np.nan
            for method_name, score_col in methods:
                threshold = compute_threshold_on_val(df.loc[df[SPLIT_COLUMN] == SPLIT_VAL, score_col].to_numpy(), 95)
                metrics = compute_all_metrics(y_true, subset[score_col].to_numpy(), threshold)
                metrics.update(
                    {
                        "ratio_target": ratio,
                        "ratio_actual": round(actual_ratio, 4) if np.isfinite(actual_ratio) else np.nan,
                        "seed": base_seed + rep,
                        "method": method_name,
                        "score_col": score_col,
                        "n_known_sub": n_known_use,
                        "n_unknown_sub": n_unknown_use,
                    }
                )
                results.append(metrics)
    raw_df = pd.DataFrame(results)
    if raw_df.empty:
        return raw_df, pd.DataFrame()
    metric_cols = ["AUROC", "AUPR", "FPR95", "Precision", "Recall", "Macro_F1"]
    rows = []
    for (ratio, method_name), group in raw_df.groupby(["ratio_target", "method"], dropna=False):
        row = {
            "ratio": round(float(ratio), 4),
            "method": method_name,
            "n_repeats": int(len(group)),
            "ratio_actual_mean": round(float(group["ratio_actual"].mean()), 4),
            "ratio_actual_std": round(float(group["ratio_actual"].std(ddof=0)), 4),
            "n_known_sub_mean": round(float(group["n_known_sub"].mean()), 2),
            "n_unknown_sub_mean": round(float(group["n_unknown_sub"].mean()), 2),
        }
        for col in metric_cols:
            values = group[col].dropna()
            row[f"{col}_mean"] = round(float(values.mean()), 4) if not values.empty else np.nan
            row[f"{col}_std"] = round(float(values.std(ddof=0)), 4) if not values.empty else np.nan
        rows.append(row)
    return raw_df, pd.DataFrame(rows).sort_values(["ratio", "method"]).reset_index(drop=True)


def run_selective_fusion(expression_scores, grn_scores, auc_df, centroid_df, config: dict):
    known_classes = config["data"]["known_classes"]
    unknown_class = config["data"]["unknown_classes"][0]
    df = expression_scores.merge(grn_scores, on="cell_id", how="left")
    df = build_dual_fused_v2(df)
    df["entropy_raw"] = df["entropy"]
    df["latent_distance_raw"] = df["latent_distance"]
    df["entropy"] = df["entropy_norm"]
    df["expr_distance_norm"] = df["distance_norm"]
    df["grn_distance"] = df["grn_distance_score"]

    df = compute_expr_margin(df, known_classes)
    df = compute_grn_margin(df, auc_df, centroid_df, known_classes)
    df, gate_threshold = add_selective_fusion_scores(
        df,
        gate_quantile=float(config["rescue"]["gate_quantile"]),
        alpha=float(config["rescue"]["selective_alpha"]),
    )

    method_specs = [
        ("entropy", "entropy"),
        ("expr_distance", "expr_distance"),
        ("expr_fused", "expr_fused"),
        ("grn_distance", "grn_distance"),
        ("grn_aux_score", "grn_aux_score"),
        ("dual_fused_v2_rankavg", "dual_fused_v2_rankavg"),
        ("selective_fused_score", "selective_fused_score"),
    ]
    test_df = df[df[SPLIT_COLUMN].isin(TEST_SPLITS)].copy()
    y_true = (test_df[SPLIT_COLUMN] == SPLIT_TEST_UNKNOWN).astype(int).to_numpy()
    metrics_by_method = {}
    thresholds = {}
    for method_name, score_col in method_specs:
        threshold = compute_threshold_on_val(df.loc[df[SPLIT_COLUMN] == SPLIT_VAL, score_col].to_numpy(), 95)
        thresholds[method_name] = threshold
        metrics = compute_all_metrics(y_true, test_df[score_col].to_numpy(), threshold)
        if method_name == "selective_fused_score":
            metrics.update(compute_gate_stats(df))
        metrics_by_method[method_name] = metrics

    ratio_raw_df, ratio_summary_df = run_ratio_ablation(df)
    gate_stats = compute_gate_stats(df)
    proximity_df = compute_grn_proximity(df, unknown_class)
    centroid_means_df = compute_class_regulon_means(df, auc_df, known_classes, unknown_class)
    interpret_table = build_interpretability_table(centroid_means_df, known_classes, unknown_class)
    pairwise_df = compute_pairwise_diff(centroid_means_df, known_classes, unknown_class)
    error_df, error_explanation = error_analysis(
        df,
        score_col="selective_fused_score",
        threshold=thresholds["selective_fused_score"],
        feature_cols=["entropy", "expr_distance", "expr_margin", "grn_distance", "grn_margin", "expr_uncertainty_score"],
        nearest_col="nearest_grn_class",
    )
    summary_text = build_selective_fusion_summary(
        config,
        metrics_by_method,
        thresholds,
        gate_threshold,
        gate_stats,
        proximity_df,
        interpret_table,
        error_explanation,
        ratio_summary_df,
    )
    return {
        "df": df,
        "metrics_by_method": metrics_by_method,
        "thresholds": thresholds,
        "metrics_df": metrics_to_frame(metrics_by_method),
        "gate_threshold": gate_threshold,
        "gate_stats": gate_stats,
        "ratio_raw_df": ratio_raw_df,
        "ratio_summary_df": ratio_summary_df,
        "proximity_df": proximity_df,
        "centroid_means_df": centroid_means_df,
        "interpret_table": interpret_table,
        "pairwise_df": pairwise_df,
        "error_df": error_df,
        "error_explanation": error_explanation,
        "summary_text": summary_text,
    }


def run_component_ablation(df, known_bucket_list, gate_threshold_info, alpha_main):
    variant_specs = [
        {"name": "bucket_threshold_full", "prefix": "bucket_threshold_full", "positive_only": True, "allow_cdc2": False, "alpha_map": alpha_main, "threshold_mode": "bucket"},
        {"name": "global_threshold_only", "prefix": "global_threshold_only", "positive_only": True, "allow_cdc2": False, "alpha_map": alpha_main, "threshold_mode": "global"},
        {"name": "allow_cdc2_block_relaxation", "prefix": "allow_cdc2_block_relaxation", "positive_only": True, "allow_cdc2": True, "alpha_map": {**alpha_main, known_bucket_list[-1]: 0.15}, "threshold_mode": "bucket"},
        {"name": "disable_positive_only_gate", "prefix": "disable_positive_only_gate", "positive_only": False, "allow_cdc2": False, "alpha_map": alpha_main, "threshold_mode": "bucket"},
    ]

    def predict_component_variant(frame, score_col, threshold_info, pred_col, threshold_col, threshold_mode):
        frame = frame.copy()
        if threshold_mode == "global":
            return predict_with_global_threshold(frame, score_col, threshold_info["global"], pred_col)
        frame[threshold_col] = frame["lineage_bucket"].map(threshold_info["by_bucket"]).fillna(threshold_info["global"]).astype(float)
        frame[pred_col] = (frame[score_col].to_numpy(dtype=float) > frame[threshold_col].to_numpy(dtype=float)).astype(int)
        return frame
    rows = []
    threshold_info_by_variant = {}
    for spec in variant_specs:
        df = build_variant(df, prefix=spec["prefix"], gate_threshold_info=gate_threshold_info, alpha_map=spec["alpha_map"], positive_only=spec["positive_only"], allow_cdc2=spec["allow_cdc2"])
        score_col = f"{spec['prefix']}_score"
        pred_col = f"{spec['prefix']}_pred"
        threshold_info = fit_thresholds_by_bucket(df, known_bucket_list, score_col, quantile=0.95, min_cells=100)
        threshold_info_by_variant[spec["name"]] = threshold_info
        df = predict_component_variant(
            df,
            score_col,
            threshold_info,
            pred_col,
            f"{spec['prefix']}_threshold",
            spec["threshold_mode"],
        )
        threshold_value = "bucket_specific" if spec["threshold_mode"] == "bucket" else threshold_info["global"]
        _, metrics = evaluate_method(df, spec["name"], score_col, pred_col, threshold_value=threshold_value)
        metrics.update(compute_rescue_stats(df, "expr_fused_pred", pred_col))
        metrics.update(summarize_gate_activity(df, spec["prefix"]))
        metrics["threshold_mode"] = spec["threshold_mode"]
        metrics["score_col"] = score_col
        rows.append({"variant": spec["name"], **metrics})
    return df, pd.DataFrame(rows), threshold_info_by_variant


def run_lineage_selective_rescue(selective_df, auc_df, config: dict):
    known_classes = config["data"]["known_classes"]
    unknown_class = config["data"]["unknown_classes"][0]
    corridor_priority = config["rescue"]["corridor_priority"]
    known_bucket_list = known_buckets(corridor_priority)
    threshold_cfg = config["threshold"]
    alpha_map = config["rescue"]["lineage_alpha_map"]

    df = assign_lineage_buckets(selective_df, corridor_priority)
    gate_threshold_info = fit_thresholds_by_bucket(
        df,
        known_bucket_list,
        "expr_uncertainty_score",
        quantile=float(config["rescue"]["gate_quantile"]),
        min_cells=int(threshold_cfg["min_bucket_val_cells"]),
    )
    df = build_variant(
        df,
        prefix="lineage_selective_rescue",
        gate_threshold_info=gate_threshold_info,
        alpha_map=alpha_map,
        positive_only=bool(config["rescue"]["positive_only"]),
        allow_cdc2=bool(config["rescue"]["allow_cdc2"]),
    )
    df["lineage_selective_rescue_globalT"] = df["lineage_selective_rescue_score"]

    threshold_configs = {}
    global_methods = [
        ("expr_fused", "expr_fused"),
        ("selective_fused_score", "selective_fused_score"),
        ("grn_distance", "grn_distance"),
        ("dual_fused_v2_rankavg", "dual_fused_v2_rankavg"),
        ("lineage_selective_rescue_globalT", "lineage_selective_rescue_globalT"),
    ]
    for method_name, score_col in global_methods:
        threshold_info = fit_thresholds_by_bucket(df, known_bucket_list, score_col, quantile=0.95, min_cells=int(threshold_cfg["min_bucket_val_cells"]))
        threshold_configs[method_name] = {"score_col": score_col, "pred_col": f"{method_name}_pred", "mode": "global", "threshold": threshold_info["global"], "threshold_info": threshold_info}

    for _, config_item in threshold_configs.items():
        df = predict_with_global_threshold(df, config_item["score_col"], config_item["threshold"], config_item["pred_col"])

    metrics_by_method = {}
    threshold_exports = {}
    gate_metrics_main = summarize_gate_activity(df, "lineage_selective_rescue")
    selective_gate_metrics = {
        "gated_fraction": round(float(df.loc[df[SPLIT_COLUMN].isin(TEST_SPLITS), "is_gated"].mean()), 4),
        "gated_fraction_on_unknown": round(float(df.loc[df[SPLIT_COLUMN] == SPLIT_TEST_UNKNOWN, "is_gated"].mean()), 4),
        "gated_fraction_on_known": round(float(df.loc[df[SPLIT_COLUMN] == "test_known", "is_gated"].mean()), 4),
    }
    for method_name, config_item in threshold_configs.items():
        pred_col = config_item["pred_col"]
        threshold_value = config_item["threshold"]
        extra = {}
        if method_name == "lineage_selective_rescue_globalT":
            extra.update(gate_metrics_main)
        if method_name == "selective_fused_score":
            extra.update(selective_gate_metrics)
        _, metrics = evaluate_method(df, method_name, config_item["score_col"], pred_col, threshold_value=threshold_value, extra_metrics=extra)
        if method_name == "lineage_selective_rescue_globalT":
            metrics.update(compute_rescue_stats(df, "expr_fused_pred", pred_col))
        metrics_by_method[method_name] = metrics
        threshold_exports[method_name] = {
            "mode": config_item["mode"],
            "global": round(float(config_item["threshold_info"]["global"]), 6),
            "by_bucket": {k: round(float(v), 6) for k, v in config_item["threshold_info"]["by_bucket"].items()},
            "bucket_sizes": config_item["threshold_info"]["bucket_sizes"],
        }

    bucket_metrics_df = compute_bucket_metrics(
        df,
        known_bucket_list,
        {
            "expr_fused": "expr_fused_pred",
            "selective_fused_score": "selective_fused_score_pred",
            "lineage_selective_rescue_globalT": "lineage_selective_rescue_globalT_pred",
        },
        gate_cols={"selective_fused_score": "is_gated", "lineage_selective_rescue_globalT": "lineage_selective_rescue_is_bucket_gated"},
        baseline_pred_col="expr_fused_pred",
    )
    rescue_delta_df = build_rescue_delta_analysis(
        df,
        {"selective_fused_score": "selective_fused_score_pred", "lineage_selective_rescue_globalT": "lineage_selective_rescue_globalT_pred"},
        baseline_pred_col="expr_fused_pred",
    )
    df, component_ablation_df, component_thresholds = run_component_ablation(df, known_bucket_list, gate_threshold_info, alpha_map)
    asdc_df = assign_asdc_subgroups(df, "expr_fused_pred", "lineage_selective_rescue_globalT_pred", unknown_class)
    asdc_summary_df = summarize_asdc_subgroups(asdc_df, ["entropy", "expr_distance", "expr_margin", "grn_distance", "grn_margin", "expr_uncertainty_score"])
    asdc_regulon_diff_df = compute_asdc_regulon_diff(asdc_df, auc_df)
    summary_text = build_lineage_rescue_summary(metrics_by_method, bucket_metrics_df, component_ablation_df, asdc_summary_df, threshold_exports)

    metrics_df = metrics_to_frame(metrics_by_method)
    if not metrics_df.empty:
        metrics_df["method"] = pd.Categorical(metrics_df["method"], categories=DEFAULT_METHOD_ORDER, ordered=True)
        metrics_df = metrics_df.sort_values("method").reset_index(drop=True)
    return {
        "df": df,
        "metrics_by_method": metrics_by_method,
        "metrics_df": metrics_df,
        "threshold_exports": threshold_exports,
        "bucket_metrics_df": bucket_metrics_df,
        "rescue_delta_df": rescue_delta_df,
        "component_ablation_df": component_ablation_df,
        "component_thresholds": component_thresholds,
        "asdc_df": asdc_df,
        "asdc_summary_df": asdc_summary_df,
        "asdc_regulon_diff_df": asdc_regulon_diff_df,
        "summary_text": summary_text,
        "gate_threshold_info": gate_threshold_info,
    }
