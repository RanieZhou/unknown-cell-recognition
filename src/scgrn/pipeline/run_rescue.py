"""Rescue-stage entrypoint."""

from __future__ import annotations

import pandas as pd

from ..backbone.base import ensure_expression_schema, validate_expression_artifacts_frame
from ..evaluation.plots import (
    plot_asdc_subgroup_heatmap,
    plot_bucket_barplot,
    plot_ratio_ablation,
    plot_regulon_heatmap,
    plot_roc_curves,
    plot_top_regulon_barplot,
)
from ..rescue.lineage_selective_rescue import run_lineage_selective_rescue, run_selective_fusion
from ..utils.io import write_dataframe, write_json
from ..utils.logger import setup_logger


def run_rescue(config: dict, paths, *, logger=None):
    logger = logger or setup_logger(paths.logs, "rescue")
    expression_scores = pd.read_csv(paths.intermediate / "expression_scores.csv")
    expression_scores = ensure_expression_schema(expression_scores, config["data"]["known_classes"])
    validate_expression_artifacts_frame(expression_scores, config["data"]["known_classes"])
    grn_scores = pd.read_csv(paths.intermediate / "grn_scores.csv")
    auc_all = pd.read_csv(paths.intermediate / "aucell_all_cells.csv", index_col=0)
    auc_all.index = auc_all.index.astype(str)
    centroid_df = pd.read_csv(paths.intermediate / "regulon_centroids.csv", index_col=0)
    centroid_df.index = centroid_df.index.astype(str)

    logger.info("Running selective fusion mainline")
    selective = run_selective_fusion(expression_scores, grn_scores, auc_all, centroid_df, config)
    write_dataframe(paths.predictions / "selective_scores_per_cell.csv", selective["df"])
    write_dataframe(paths.metrics / "selective_method_comparison.csv", selective["metrics_df"])
    write_dataframe(paths.metrics / "selective_ratio_ablation.csv", selective["ratio_summary_df"])
    write_dataframe(paths.metrics / "selective_gate_stats.csv", pd.DataFrame([selective["gate_stats"]]))
    write_dataframe(paths.metrics / "selective_error_analysis.csv", selective["error_df"])
    write_dataframe(paths.metrics / "selective_regulon_class_proximity.csv", selective["proximity_df"])
    write_dataframe(paths.metrics / "selective_regulon_interpretability.csv", selective["interpret_table"])
    write_json(
        paths.metrics / "selective_thresholds.json",
        {
            "thresholds": selective["thresholds"],
            "gate_threshold": selective["gate_threshold"],
        },
    )
    if config["runtime"]["save_reports"]:
        (paths.reports / "selective_summary.md").write_text(selective["summary_text"], encoding="utf-8")

    logger.info("Running lineage_selective_rescue_globalT mainline")
    lineage = run_lineage_selective_rescue(selective["df"], auc_all, config)
    write_dataframe(paths.predictions / "lineage_scores_per_cell.csv", lineage["df"])
    write_dataframe(paths.metrics / "lineage_method_comparison.csv", lineage["metrics_df"])
    write_dataframe(paths.metrics / "lineage_bucket_metrics.csv", lineage["bucket_metrics_df"])
    write_dataframe(paths.metrics / "lineage_rescue_delta_analysis.csv", lineage["rescue_delta_df"])
    write_dataframe(paths.metrics / "lineage_component_ablation.csv", lineage["component_ablation_df"])
    write_dataframe(paths.metrics / "lineage_asdc_subgroup_analysis.csv", lineage["asdc_summary_df"])
    write_dataframe(paths.metrics / "lineage_asdc_regulon_diff.csv", lineage["asdc_regulon_diff_df"])
    write_json(
        paths.metrics / "lineage_thresholds.json",
        {
            "gate_thresholds": lineage["gate_threshold_info"],
            "score_thresholds": lineage["threshold_exports"],
            "component_thresholds": lineage["component_thresholds"],
        },
    )
    if config["runtime"]["save_reports"]:
        (paths.reports / "lineage_summary.md").write_text(lineage["summary_text"], encoding="utf-8")

    if config["runtime"]["save_plots"]:
        plot_roc_curves(
            selective["df"],
            [(name, name, selective["metrics_by_method"].get(name, {}).get("AUROC", float("nan"))) for name in ["expr_fused", "selective_fused_score", "grn_distance"]],
            paths.plots / "selective_roc.png",
        )
        plot_ratio_ablation(selective["ratio_summary_df"], paths.plots / "selective_ratio_ablation.png")
        plot_regulon_heatmap(selective["centroid_means_df"], paths.plots / "selective_regulon_heatmap.png", "Selective Fusion Regulon Heatmap")
        if "ASDC_vs_pDC" in selective["pairwise_df"].columns:
            plot_top_regulon_barplot(selective["pairwise_df"]["ASDC_vs_pDC"], "ASDC vs pDC", paths.plots / "selective_top_regulon_barplot.png")

        plot_roc_curves(
            lineage["df"],
            [
                ("expr_fused", "expr_fused", lineage["metrics_by_method"].get("expr_fused", {}).get("AUROC", float("nan"))),
                ("selective_fused_score", "selective_fused_score", lineage["metrics_by_method"].get("selective_fused_score", {}).get("AUROC", float("nan"))),
                ("lineage_selective_rescue_globalT", "lineage_selective_rescue_globalT", lineage["metrics_by_method"].get("lineage_selective_rescue_globalT", {}).get("AUROC", float("nan"))),
            ],
            paths.plots / "lineage_roc.png",
        )
        plot_bucket_barplot(lineage["bucket_metrics_df"], paths.plots / "lineage_bucket_barplot.png")
        plot_asdc_subgroup_heatmap(lineage["asdc_df"], auc_all, paths.plots / "lineage_asdc_subgroup_heatmap.png")

    return {"selective": selective, "lineage": lineage}
