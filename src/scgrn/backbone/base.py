"""Backbone adapter interface and expression artifact schema helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..constants import EXPRESSION_SCHEMA_VERSION, LEGACY_SPLIT_COLUMN, SPLIT_COLUMN

STANDARD_EXPRESSION_COLUMNS = [
    "cell_id",
    "true_label",
    SPLIT_COLUMN,
    "pred_label",
    "max_prob",
    "entropy",
    "expr_distance",
    "expr_fused_score",
]

COMPAT_EXPRESSION_COLUMNS = [
    "predicted_label",
    "entropy_norm",
    "latent_distance",
    "distance_norm",
    "expr_fused",
]


def probability_columns(known_classes: list[str]) -> list[str]:
    return [f"prob_{cls}" for cls in known_classes]


def ensure_expression_schema(df: pd.DataFrame, known_classes: list[str]) -> pd.DataFrame:
    df = df.copy()
    prob_cols = [col for col in probability_columns(known_classes) if col in df.columns]

    if SPLIT_COLUMN not in df.columns and LEGACY_SPLIT_COLUMN in df.columns:
        df[SPLIT_COLUMN] = df[LEGACY_SPLIT_COLUMN]
    if LEGACY_SPLIT_COLUMN in df.columns:
        df = df.drop(columns=[LEGACY_SPLIT_COLUMN])

    if "pred_label" not in df.columns and "predicted_label" in df.columns:
        df["pred_label"] = df["predicted_label"]
    if "predicted_label" not in df.columns and "pred_label" in df.columns:
        df["predicted_label"] = df["pred_label"]

    if "expr_distance" not in df.columns and "latent_distance" in df.columns:
        df["expr_distance"] = df["latent_distance"]
    if "latent_distance" not in df.columns and "expr_distance" in df.columns:
        df["latent_distance"] = df["expr_distance"]

    if "expr_distance_norm" not in df.columns and "distance_norm" in df.columns:
        df["expr_distance_norm"] = df["distance_norm"]

    if "expr_fused_score" not in df.columns and "expr_fused" in df.columns:
        df["expr_fused_score"] = df["expr_fused"]
    if "expr_fused" not in df.columns and "expr_fused_score" in df.columns:
        df["expr_fused"] = df["expr_fused_score"]

    if "max_prob" not in df.columns:
        if prob_cols:
            df["max_prob"] = df[prob_cols].max(axis=1)
        else:
            df["max_prob"] = np.nan

    if "nearest_known_class" not in df.columns and "pred_label" in df.columns:
        df["nearest_known_class"] = df["pred_label"]

    return df


def validate_expression_artifacts_frame(df: pd.DataFrame, known_classes: list[str], *, require_compat: bool = True) -> pd.DataFrame:
    required = list(STANDARD_EXPRESSION_COLUMNS)
    if require_compat:
        required.extend(COMPAT_EXPRESSION_COLUMNS)
    required.extend(probability_columns(known_classes))
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Expression artifact schema is incomplete; missing columns: {missing}")
    return df


@dataclass
class BackboneTrainResult:
    backbone_name: str
    model: Any
    prepared_adata: Any
    training_adata: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackbonePredictResult:
    backbone_name: str
    model: Any
    prepared_adata: Any
    training_adata: Any | None = None
    latent_representation: Any | None = None
    soft_predictions: pd.DataFrame | None = None
    hard_predictions: Any | None = None
    extra_artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpressionArtifactsResult:
    backbone_name: str
    expression_scores: pd.DataFrame
    centroids: dict[str, Any] | None = None
    schema_version: str = EXPRESSION_SCHEMA_VERSION
    extra_artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class BackboneAdapter(ABC):
    """Common interface for backbone-specific training and inference."""

    name: str

    @abstractmethod
    def train(
        self,
        adata,
        train_idx,
        val_idx,
        config: dict,
        checkpoints_dir: Path,
        *,
        use_cached: bool = False,
    ) -> BackboneTrainResult:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        adata,
        train_idx,
        val_idx,
        config: dict,
        checkpoints_dir: Path,
    ) -> BackbonePredictResult:
        raise NotImplementedError

    @abstractmethod
    def build_expression_artifacts(
        self,
        adata,
        predict_output: BackbonePredictResult,
        split_context: dict[str, Any],
        config: dict,
    ) -> ExpressionArtifactsResult:
        raise NotImplementedError
