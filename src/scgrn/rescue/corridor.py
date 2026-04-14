"""Lineage bucket assignment."""

from __future__ import annotations

import numpy as np

from ..constants import DEFAULT_LINEAGE_BUCKET_SUFFIX


def known_buckets(priority_order: list[str]) -> list[str]:
    return [f"{cls}{DEFAULT_LINEAGE_BUCKET_SUFFIX}" for cls in priority_order]


def assign_lineage_buckets(df, priority_order: list[str]):
    df = df.copy()
    expected = {str(cls) for cls in priority_order}
    df["expr_lineage"] = df["predicted_label"].astype(str)
    df["grn_lineage"] = df["nearest_grn_class"].astype(str)

    unexpected_expr = sorted({value for value in df["expr_lineage"].unique() if value not in expected})
    unexpected_grn = sorted({value for value in df["grn_lineage"].unique() if value not in expected})
    unexpected = sorted(set(unexpected_expr + unexpected_grn))
    if unexpected:
        raise ValueError(f"Unexpected lineage labels encountered during bucket assignment: {unexpected}")

    bucket = np.full(len(df), f"{priority_order[-1]}{DEFAULT_LINEAGE_BUCKET_SUFFIX}", dtype=object)
    for cls in reversed(priority_order):
        cls_bucket = f"{cls}{DEFAULT_LINEAGE_BUCKET_SUFFIX}"
        match = (df["expr_lineage"] == cls) | (df["grn_lineage"] == cls)
        bucket = np.where(match, cls_bucket, bucket)
    df["lineage_bucket"] = bucket
    return df
