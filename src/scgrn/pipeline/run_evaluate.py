"""Evaluation-stage entrypoint."""

from __future__ import annotations

import pandas as pd

from ..backbone.base import ensure_expression_schema, validate_expression_artifacts_frame
from ..evaluation.consistency import (
    build_legacy_artifact_alignment,
    build_legacy_bucket_tradeoffs,
    build_legacy_component_alignment,
    build_legacy_consistency_report,
    build_legacy_method_baseline,
    build_legacy_stage_continuity,
    build_new_vs_legacy_metrics,
)
from ..utils.io import write_dataframe, write_json
from ..utils.logger import setup_logger


def run_evaluate(config: dict, paths, *, logger=None):
    logger = logger or setup_logger(paths.logs, "evaluate")
    selective_summary = paths.reports / "selective_summary.md"
    lineage_summary = paths.reports / "lineage_summary.md"
    consistency_report = paths.reports / "consistency_check.md"
    expression_scores_path = paths.intermediate / "expression_scores.csv"
    if expression_scores_path.exists():
        expression_scores = pd.read_csv(expression_scores_path)
        expression_scores = ensure_expression_schema(expression_scores, config["data"]["known_classes"])
        validate_expression_artifacts_frame(expression_scores, config["data"]["known_classes"])
    selective_metrics_path = paths.metrics / "selective_method_comparison.csv"
    lineage_metrics_path = paths.metrics / "lineage_method_comparison.csv"
    selective_metrics = pd.read_csv(selective_metrics_path) if selective_metrics_path.exists() else pd.DataFrame()
    lineage_metrics = pd.read_csv(lineage_metrics_path) if lineage_metrics_path.exists() else pd.DataFrame()

    selective_report = [
        "# selective_fused_score",
        "",
        "## Key Metrics" if not selective_metrics.empty else "Selective metrics are not available yet. Run `python rescue.py --config <config>` first.",
    ]
    if not selective_metrics.empty:
        for _, row in selective_metrics.iterrows():
            selective_report.append(
                f"- {row['method']}: AUROC={row.get('AUROC', float('nan'))}, AUPR={row.get('AUPR', float('nan'))}, Macro_F1={row.get('Macro_F1', float('nan'))}"
            )
    selective_summary.write_text("\n".join(selective_report), encoding="utf-8")

    lineage_report = [
        "# lineage_selective_rescue_globalT",
        "",
        "## Key Metrics" if not lineage_metrics.empty else "Lineage rescue metrics are not available yet. Run `python rescue.py --config <config>` first.",
    ]
    if not lineage_metrics.empty:
        for _, row in lineage_metrics.iterrows():
            lineage_report.append(
                f"- {row['method']}: AUROC={row.get('AUROC', float('nan'))}, AUPR={row.get('AUPR', float('nan'))}, Macro_F1={row.get('Macro_F1', float('nan'))}"
            )
    lineage_summary.write_text("\n".join(lineage_report), encoding="utf-8")

    artifact_df = build_legacy_artifact_alignment(paths)
    stage_df, stage_details = build_legacy_stage_continuity(paths)
    baseline_df, delta_df = build_legacy_method_baseline(paths)
    component_df = build_legacy_component_alignment(paths)
    bucket_df = build_legacy_bucket_tradeoffs(paths)
    new_vs_legacy_df = build_new_vs_legacy_metrics(lineage_metrics, paths)

    write_dataframe(paths.metrics / "legacy_artifact_alignment.csv", artifact_df)
    write_dataframe(paths.metrics / "legacy_stage_continuity.csv", stage_df)
    write_json(paths.metrics / "legacy_stage_continuity_details.json", stage_details)
    write_dataframe(paths.metrics / "legacy_method_baseline.csv", baseline_df)
    write_dataframe(paths.metrics / "legacy_method_deltas.csv", delta_df)
    write_dataframe(paths.metrics / "legacy_component_alignment.csv", component_df)
    write_dataframe(paths.metrics / "legacy_bucket_tradeoffs.csv", bucket_df)
    write_dataframe(paths.metrics / "new_vs_legacy_metrics.csv", new_vs_legacy_df)

    consistency_text = build_legacy_consistency_report(
        config,
        paths,
        artifact_df,
        stage_df,
        stage_details,
        baseline_df,
        delta_df,
        component_df,
        bucket_df,
        new_vs_legacy_df,
    )
    consistency_report.write_text(consistency_text, encoding="utf-8")
    logger.info("Wrote evaluation summaries to %s", paths.reports)
    return {"selective_summary": selective_summary, "lineage_summary": lineage_summary, "consistency_check": consistency_report}
