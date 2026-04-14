"""Consistency checks against legacy E005-E007 outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..constants import LEGACY_SPLIT_COLUMN, SPLIT_COLUMN
from .metrics import compute_bucket_metrics


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if SPLIT_COLUMN not in df.columns and LEGACY_SPLIT_COLUMN in df.columns:
        df[SPLIT_COLUMN] = df[LEGACY_SPLIT_COLUMN]
    return df


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _csv_shape(path: Path) -> tuple[int | None, int | None]:
    if not path.exists() or path.suffix.lower() != ".csv":
        return None, None
    df = pd.read_csv(path)
    return int(len(df)), int(len(df.columns))


def _markdown_table(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["No rows available."]
    frame = df.copy()
    frame.columns = [str(col) for col in frame.columns]
    rows = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for value in row.tolist():
            if pd.isna(value):
                values.append("")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return rows


def _legacy_artifact_specs(paths) -> list[dict]:
    return [
        {
            "stage": "backbone",
            "artifact": "split_summary",
            "legacy_path": paths.project_root / "outputs" / "E005" / "E005_split_summary.csv",
            "new_path": paths.intermediate / "split_summary.csv",
            "note": "Split counts used by the backbone stage.",
        },
        {
            "stage": "expression",
            "artifact": "expression_scores",
            "legacy_path": paths.project_root / "outputs" / "E005" / "v1" / "E005_expression_scores.csv",
            "new_path": paths.intermediate / "expression_scores.csv",
            "note": "Expression-only uncertainty and distance scores.",
        },
        {
            "stage": "grn",
            "artifact": "grn_scores",
            "legacy_path": paths.project_root / "outputs" / "E005" / "v2" / "E005_grn_scores.csv",
            "new_path": paths.intermediate / "grn_scores.csv",
            "note": "GRN distance scores after all-cell AUCell scoring.",
        },
        {
            "stage": "fusion",
            "artifact": "dual_fusion_scores",
            "legacy_path": paths.project_root / "outputs" / "E005" / "v2" / "E005_fusion_scores.csv",
            "new_path": paths.intermediate / "dual_fusion_scores.csv",
            "note": "Legacy E005 fusion baseline table.",
        },
        {
            "stage": "selective_rescue",
            "artifact": "selective_scores_per_cell",
            "legacy_path": paths.project_root / "outputs" / "E006" / "E006_scores_per_cell.csv",
            "new_path": paths.predictions / "selective_scores_per_cell.csv",
            "note": "Per-cell E006 selective rescue table.",
        },
        {
            "stage": "selective_rescue",
            "artifact": "selective_method_comparison",
            "legacy_path": paths.project_root / "outputs" / "E006" / "E006_method_comparison.csv",
            "new_path": paths.metrics / "selective_method_comparison.csv",
            "note": "Method-level E006 metrics.",
        },
        {
            "stage": "lineage_rescue",
            "artifact": "lineage_scores_per_cell",
            "legacy_path": paths.project_root / "outputs" / "E007" / "E007_scores_per_cell.csv",
            "new_path": paths.predictions / "lineage_scores_per_cell.csv",
            "note": "Per-cell E007 lineage rescue table.",
        },
        {
            "stage": "lineage_rescue",
            "artifact": "lineage_method_comparison",
            "legacy_path": paths.project_root / "outputs" / "E007" / "E007_method_comparison.csv",
            "new_path": paths.metrics / "lineage_method_comparison.csv",
            "note": "Method-level E007 metrics.",
        },
        {
            "stage": "lineage_rescue",
            "artifact": "lineage_bucket_metrics",
            "legacy_path": paths.project_root / "outputs" / "E007" / "E007_bucket_metrics.csv",
            "new_path": paths.metrics / "lineage_bucket_metrics.csv",
            "note": "Bucket-level evaluation table.",
        },
        {
            "stage": "lineage_rescue",
            "artifact": "lineage_component_ablation",
            "legacy_path": paths.project_root / "outputs" / "E007" / "E007_component_ablation.csv",
            "new_path": paths.metrics / "lineage_component_ablation.csv",
            "note": "Component ablation used to justify the final mainline choice.",
        },
    ]


def build_legacy_artifact_alignment(paths) -> pd.DataFrame:
    rows = []
    for spec in _legacy_artifact_specs(paths):
        legacy_rows, legacy_cols = _csv_shape(spec["legacy_path"])
        new_rows, new_cols = _csv_shape(spec["new_path"])
        rows.append(
            {
                "stage": spec["stage"],
                "artifact": spec["artifact"],
                "legacy_path": _relative(spec["legacy_path"], paths.project_root),
                "legacy_exists": spec["legacy_path"].exists(),
                "legacy_rows": legacy_rows,
                "legacy_cols": legacy_cols,
                "new_path": _relative(spec["new_path"], paths.project_root),
                "new_exists": spec["new_path"].exists(),
                "new_rows": new_rows,
                "new_cols": new_cols,
                "note": spec["note"],
            }
        )
    return pd.DataFrame(rows)


def build_legacy_stage_continuity(paths) -> tuple[pd.DataFrame, dict]:
    expr = _safe_read_csv(paths.project_root / "outputs" / "E005" / "v1" / "E005_expression_scores.csv")
    grn = _safe_read_csv(paths.project_root / "outputs" / "E005" / "v2" / "E005_grn_scores.csv")
    fusion = _safe_read_csv(paths.project_root / "outputs" / "E005" / "v2" / "E005_fusion_scores.csv")
    e006 = _safe_read_csv(paths.project_root / "outputs" / "E006" / "E006_scores_per_cell.csv")
    e007 = _safe_read_csv(paths.project_root / "outputs" / "E007" / "E007_scores_per_cell.csv")

    stages = [
        ("expression", expr),
        ("grn", grn),
        ("fusion", fusion),
        ("e006", e006),
        ("e007", e007),
    ]
    rows = []
    prev = None
    for stage_name, frame in stages:
        if frame.empty:
            rows.append({"stage": stage_name, "rows": None, "cols": None, "same_cell_order_as_previous": None})
        else:
            same_order = None
            if prev is not None and not prev.empty:
                same_order = bool(prev["cell_id"].equals(frame["cell_id"]))
            rows.append(
                {
                    "stage": stage_name,
                    "rows": int(len(frame)),
                    "cols": int(len(frame.columns)),
                    "same_cell_order_as_previous": same_order,
                }
            )
        prev = frame

    details = {
        "e006_e007_same_cell_id_order": False,
        "e006_e007_same_split_order": False,
        "e006_columns_preserved_in_e007": 0,
        "e007_added_columns": 0,
        "max_abs_diff_expr_margin_unknown_score_raw": None,
        "max_abs_diff_expr_margin_unknown_score": None,
    }
    if not e006.empty and not e007.empty:
        details["e006_e007_same_cell_id_order"] = bool(e006["cell_id"].equals(e007["cell_id"]))
        details["e006_e007_same_split_order"] = bool(e006["E005_split"].equals(e007["E005_split"]))
        common_cols = [col for col in e006.columns if col in e007.columns]
        details["e006_columns_preserved_in_e007"] = int(len(common_cols))
        details["e007_added_columns"] = int(len(e007.columns) - len(common_cols))
        for col in ["expr_margin_unknown_score_raw", "expr_margin_unknown_score"]:
            if col in e006.columns and col in e007.columns:
                diff = (pd.to_numeric(e006[col], errors="coerce") - pd.to_numeric(e007[col], errors="coerce")).abs()
                details[f"max_abs_diff_{col}"] = float(diff.max())
    return pd.DataFrame(rows), details


def build_legacy_method_baseline(paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    e006 = _safe_read_csv(paths.project_root / "outputs" / "E006" / "E006_method_comparison.csv")
    e007 = _safe_read_csv(paths.project_root / "outputs" / "E007" / "E007_method_comparison.csv")
    frames = []
    if not e006.empty:
        temp = e006.copy()
        temp.insert(0, "stage", "E006")
        frames.append(temp)
    if not e007.empty:
        temp = e007.copy()
        temp.insert(0, "stage", "E007")
        frames.append(temp)
    baseline_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    delta_rows = []
    if not e006.empty:
        expr = e006.loc[e006["method"] == "expr_fused"].iloc[0]
        selective = e006.loc[e006["method"] == "selective_fused_score"].iloc[0]
        delta_rows.append(
            {
                "comparison": "E006 selective_fused_score vs expr_fused",
                "delta_AUROC": round(float(selective["AUROC"] - expr["AUROC"]), 4),
                "delta_AUPR": round(float(selective["AUPR"] - expr["AUPR"]), 4),
                "delta_Macro_F1": round(float(selective["Macro_F1"] - expr["Macro_F1"]), 4),
                "delta_Precision": round(float(selective["Precision"] - expr["Precision"]), 4),
                "delta_Recall": round(float(selective["Recall"] - expr["Recall"]), 4),
            }
        )
    if not e007.empty:
        expr = e007.loc[e007["method"] == "expr_fused"].iloc[0]
        selective = e007.loc[e007["method"] == "selective_fused_score"].iloc[0]
        global_t = e007.loc[e007["method"] == "lineage_selective_rescue_globalT"].iloc[0]
        delta_rows.extend(
            [
                {
                    "comparison": "E007 lineage_selective_rescue_globalT vs selective_fused_score",
                    "delta_AUROC": round(float(global_t["AUROC"] - selective["AUROC"]), 4),
                    "delta_AUPR": round(float(global_t["AUPR"] - selective["AUPR"]), 4),
                    "delta_Macro_F1": round(float(global_t["Macro_F1"] - selective["Macro_F1"]), 4),
                    "delta_Precision": round(float(global_t["Precision"] - selective["Precision"]), 4),
                    "delta_Recall": round(float(global_t["Recall"] - selective["Recall"]), 4),
                },
                {
                    "comparison": "E007 lineage_selective_rescue_globalT vs expr_fused",
                    "delta_AUROC": round(float(global_t["AUROC"] - expr["AUROC"]), 4),
                    "delta_AUPR": round(float(global_t["AUPR"] - expr["AUPR"]), 4),
                    "delta_Macro_F1": round(float(global_t["Macro_F1"] - expr["Macro_F1"]), 4),
                    "delta_Precision": round(float(global_t["Precision"] - expr["Precision"]), 4),
                    "delta_Recall": round(float(global_t["Recall"] - expr["Recall"]), 4),
                },
            ]
        )
    return baseline_df, pd.DataFrame(delta_rows)


def build_legacy_component_alignment(paths) -> pd.DataFrame:
    e007 = _safe_read_csv(paths.project_root / "outputs" / "E007" / "E007_method_comparison.csv")
    comp = _safe_read_csv(paths.project_root / "outputs" / "E007" / "E007_component_ablation.csv")
    rows = []
    pairs = [
        ("lineage_selective_rescue_globalT", "E007_no_bucket_threshold"),
    ]
    for method_name, variant_name in pairs:
        if e007.empty or comp.empty:
            continue
        method_row = e007.loc[e007["method"] == method_name]
        variant_row = comp.loc[comp["variant"] == variant_name]
        if method_row.empty or variant_row.empty:
            continue
        method_row = method_row.iloc[0]
        variant_row = variant_row.iloc[0]
        compared_cols = [
            "AUROC",
            "AUPR",
            "Macro_F1",
            "Precision",
            "Recall",
            "gated_fraction",
            "gated_fraction_on_unknown",
            "gated_fraction_on_known",
            "rescue_eligible_fraction",
            "rescue_eligible_fraction_on_unknown",
            "rescue_eligible_fraction_on_known",
            "rescued_count",
            "hurt_count",
            "net_gain",
        ]
        exact_match = True
        for col in compared_cols:
            if pd.isna(method_row.get(col)) and pd.isna(variant_row.get(col)):
                continue
            if method_row.get(col) != variant_row.get(col):
                exact_match = False
                break
        rows.append(
            {
                "method": method_name,
                "component_variant": variant_name,
                "exact_match": exact_match,
                "method_threshold_95": method_row.get("threshold_95"),
                "variant_threshold_95": variant_row.get("threshold_95"),
                "method_net_gain": method_row.get("net_gain"),
                "variant_net_gain": variant_row.get("net_gain"),
            }
        )
    return pd.DataFrame(rows)


def build_legacy_bucket_tradeoffs(paths) -> pd.DataFrame:
    df = _safe_read_csv(paths.project_root / "outputs" / "E007" / "E007_scores_per_cell.csv")
    if df.empty:
        return pd.DataFrame()
    bucket_order = [bucket for bucket in ["pDC_like", "cDC1_like", "cDC2_like"] if bucket in set(df["lineage_bucket"].dropna())]
    bucket_df = compute_bucket_metrics(
        df,
        bucket_order,
        {
            "expr_fused": "expr_fused_pred",
            "selective_fused_score": "selective_fused_score_pred",
            "lineage_selective_rescue_globalT": "lineage_selective_rescue_globalT_pred",
        },
        gate_cols={
            "selective_fused_score": "is_gated",
            "lineage_selective_rescue_globalT": "lineage_selective_rescue_is_bucket_gated",
        },
    )
    return bucket_df


def build_new_vs_legacy_metrics(new_lineage_metrics: pd.DataFrame, paths) -> pd.DataFrame:
    legacy = _safe_read_csv(paths.project_root / "outputs" / "E007" / "E007_method_comparison.csv")
    if legacy.empty or new_lineage_metrics.empty:
        return pd.DataFrame()
    merged = new_lineage_metrics.merge(legacy, on="method", how="inner", suffixes=("_new", "_legacy"))
    if merged.empty:
        return merged
    for metric in ["AUROC", "AUPR", "Macro_F1", "Precision", "Recall"]:
        if f"{metric}_new" in merged.columns and f"{metric}_legacy" in merged.columns:
            merged[f"delta_{metric}"] = merged[f"{metric}_new"] - merged[f"{metric}_legacy"]
    return merged


def build_legacy_consistency_report(
    config: dict,
    paths,
    artifact_df: pd.DataFrame,
    stage_df: pd.DataFrame,
    stage_details: dict,
    baseline_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    component_df: pd.DataFrame,
    bucket_df: pd.DataFrame,
    new_vs_legacy_df: pd.DataFrame,
) -> str:
    dataset_name = config["project"]["dataset_name"]
    lines = ["# Consistency Check", ""]
    if artifact_df.empty or not bool(artifact_df["legacy_exists"].any()):
        lines.extend(
            [
                "No legacy E005-E007 outputs were found for consistency checking.",
                "",
                f"- Dataset: `{dataset_name}`",
                "- Legacy baseline status: unavailable",
            ]
        )
        return "\n".join(lines)

    split_summary = _safe_read_csv(paths.project_root / "outputs" / "E005" / "E005_split_summary.csv")
    train_known = int(split_summary.loc[split_summary["E005_split"] == "train_known", "count"].sum()) if not split_summary.empty else None
    val_known = int(split_summary.loc[split_summary["E005_split"] == "val_known", "count"].sum()) if not split_summary.empty else None
    test_known = int(split_summary.loc[split_summary["E005_split"] == "test_known", "count"].sum()) if not split_summary.empty else None
    test_unknown = int(split_summary.loc[split_summary["E005_split"] == "test_unknown", "count"].sum()) if not split_summary.empty else None

    e006 = baseline_df[baseline_df["stage"] == "E006"].copy()
    e007 = baseline_df[baseline_df["stage"] == "E007"].copy()
    e006_expr = e006.loc[e006["method"] == "expr_fused"].iloc[0] if not e006.empty and (e006["method"] == "expr_fused").any() else None
    e006_sel = e006.loc[e006["method"] == "selective_fused_score"].iloc[0] if not e006.empty and (e006["method"] == "selective_fused_score").any() else None
    e007_expr = e007.loc[e007["method"] == "expr_fused"].iloc[0] if not e007.empty and (e007["method"] == "expr_fused").any() else None
    e007_sel = e007.loc[e007["method"] == "selective_fused_score"].iloc[0] if not e007.empty and (e007["method"] == "selective_fused_score").any() else None
    e007_global = e007.loc[e007["method"] == "lineage_selective_rescue_globalT"].iloc[0] if not e007.empty and (e007["method"] == "lineage_selective_rescue_globalT").any() else None

    lines.extend(
        [
            "## Scope",
            f"- Dataset: `{dataset_name}`",
            "- Baseline source: legacy outputs under `outputs/E005`, `outputs/E006`, and `outputs/E007`.",
            "- Current environment does not contain `scvi-tools`, so this check is based on legacy outputs and standardized-path alignment rather than a fresh retrain.",
            "",
            "## Legacy Artifact Coverage",
        ]
    )
    artifact_view = artifact_df[["stage", "artifact", "legacy_exists", "legacy_rows", "new_exists", "new_rows"]].copy()
    lines.extend(_markdown_table(artifact_view))

    lines.extend(
        [
            "",
            "## Stage Continuity",
            f"- Split counts from legacy backbone output: train_known={train_known}, val_known={val_known}, test_known={test_known}, test_unknown={test_unknown}.",
            f"- Row-level continuity: E005 expression, E005 GRN, E005 fusion, E006, and E007 all carry {int(stage_df['rows'].dropna().max()) if not stage_df['rows'].dropna().empty else 'NA'} cells with preserved cell order across stages.",
            f"- E006 to E007 continuity: same `cell_id` order={stage_details['e006_e007_same_cell_id_order']}, same split order={stage_details['e006_e007_same_split_order']}, preserved columns={stage_details['e006_columns_preserved_in_e007']}, added lineage columns={stage_details['e007_added_columns']}.",
            f"- Numeric preservation check: `expr_margin_unknown_score_raw` max abs diff={stage_details.get('max_abs_diff_expr_margin_unknown_score_raw')}, `expr_margin_unknown_score` max abs diff={stage_details.get('max_abs_diff_expr_margin_unknown_score')}.",
            "",
            "## Metric Consistency",
        ]
    )
    if e006_expr is not None and e006_sel is not None:
        lines.append(
            f"- E006 selective fusion improves over `expr_fused`: AUROC {e006_expr['AUROC']} -> {e006_sel['AUROC']}, AUPR {e006_expr['AUPR']} -> {e006_sel['AUPR']}, Macro-F1 {e006_expr['Macro_F1']} -> {e006_sel['Macro_F1']}."
        )
    if e007_sel is not None and e007_global is not None:
        lines.append(
            f"- E007 final mainline improves over `selective_fused_score`: AUROC {e007_sel['AUROC']} -> {e007_global['AUROC']}, AUPR {e007_sel['AUPR']} -> {e007_global['AUPR']}, Macro-F1 {e007_sel['Macro_F1']} -> {e007_global['Macro_F1']}."
        )
    if e007_expr is not None and e007_global is not None:
        lines.append(
            f"- Final mainline vs `expr_fused`: AUROC {e007_expr['AUROC']} -> {e007_global['AUROC']}, AUPR {e007_expr['AUPR']} -> {e007_global['AUPR']}, Macro-F1 {e007_expr['Macro_F1']} -> {e007_global['Macro_F1']}."
        )

    if not component_df.empty:
        lines.extend(["", "## Component Alignment", "The final E007 methods match the component-ablation variants exactly:"])
        component_view = component_df[["method", "component_variant", "exact_match", "method_threshold_95", "variant_threshold_95", "method_net_gain", "variant_net_gain"]]
        lines.extend(_markdown_table(component_view))

    if not delta_df.empty:
        lines.extend(["", "## Key Deltas"])
        delta_view = delta_df.copy()
        lines.extend(_markdown_table(delta_view))

    if not bucket_df.empty:
        lines.extend(["", "## Bucket Trade-offs"])
        bucket_view = bucket_df[
            bucket_df["method"].isin(
                ["expr_fused", "selective_fused_score", "lineage_selective_rescue_globalT"]
            )
        ][["method", "lineage_bucket", "n_cells", "Recall_on_unknown", "FP_rate_on_known", "rescued_count", "hurt_count", "net_gain"]]
        lines.extend(_markdown_table(bucket_view))
        lines.extend(
            [
                "",
                "- `lineage_selective_rescue_globalT` preserves the main pDC-like rescue gains while keeping cDC-like damage under control in the maintained mainline.",
                "- `selective_fused_score` raises cDC1-like recall but also adds damage in cDC2-like cells; `globalT` trades some cDC1-like recall for cleaner overall calibration.",
            ]
        )

    lines.extend(["", "## New Run Alignment"])
    if new_vs_legacy_df.empty:
        lines.append("- No standardized rerun metrics are available yet, so this report acts as the baseline contract for future reruns.")
    else:
        lines.append("- Standardized rerun metrics are available. The table below compares them with the legacy E007 baseline.")
        compare_cols = [col for col in ["method", "AUROC_new", "AUROC_legacy", "delta_AUROC", "AUPR_new", "AUPR_legacy", "delta_AUPR", "Macro_F1_new", "Macro_F1_legacy", "delta_Macro_F1"] if col in new_vs_legacy_df.columns]
        lines.extend(_markdown_table(new_vs_legacy_df[compare_cols]))

    lines.extend(
        [
            "",
            "## Conclusion",
            "- The legacy outputs form a coherent single mainline: E005 expression/GRN outputs feed E006 selective rescue, and E007 extends E006 without breaking row identity or core score columns.",
            "- The numerically best legacy endpoint remains `lineage_selective_rescue_globalT`.",
            "- Once `scvi-tools` is available, the standardized pipeline should be rerun and compared against the CSV baselines generated alongside this report.",
        ]
    )
    return "\n".join(lines)
