"""GRN scoring and distance computation."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cosine

from ..constants import SPLIT_COLUMN, SPLIT_TRAIN


def score_grn_space(
    adata,
    expression_scores: pd.DataFrame,
    config: dict,
    grn_dir: Path,
    pyscenic_artifacts: dict[str, Path],
):
    from pyscenic.aucell import aucell

    known_classes = config["data"]["known_classes"]
    label_key = config["data"]["label_column"]

    train_known_cells = expression_scores.loc[
        expression_scores[SPLIT_COLUMN] == SPLIT_TRAIN, "cell_id"
    ].astype(str).tolist()
    all_cells = expression_scores["cell_id"].astype(str).tolist()

    auc_train = pd.read_csv(pyscenic_artifacts["aucell_train_csv"], index_col=0)
    train_labels = expression_scores.loc[
        expression_scores[SPLIT_COLUMN] == SPLIT_TRAIN, ["cell_id", "true_label"]
    ].copy()
    train_labels["cell_id"] = train_labels["cell_id"].astype(str)
    train_labels = train_labels.set_index("cell_id")["true_label"]

    expr_all = adata.layers[config["data"]["model_layer"]]
    if sparse.issparse(expr_all):
        expr_all = expr_all.toarray()
    expr_all_df = pd.DataFrame(expr_all, index=adata.obs_names.astype(str), columns=adata.var_names)

    with Path(pyscenic_artifacts["regulons_pkl"]).open("rb") as handle:
        regulons = pickle.load(handle)
    auc_all = aucell(
        expr_all_df,
        regulons,
        num_workers=int(config["grn"]["pyscenic"].get("num_workers", 4)),
    )

    centroids = {}
    for cls in known_classes:
        cls_cells = train_labels[train_labels == cls].index.tolist()
        cls_cells_available = [cell for cell in cls_cells if cell in auc_all.index]
        centroids[cls] = auc_all.loc[cls_cells_available].mean(axis=0).values

    grn_distances = []
    nearest_grn_class = []
    for cell in all_cells:
        if cell not in auc_all.index:
            grn_distances.append(np.nan)
            nearest_grn_class.append(np.nan)
            continue
        cell_auc = auc_all.loc[cell].values
        min_dist = np.inf
        min_cls = None
        for cls in known_classes:
            dist = cosine(cell_auc, centroids[cls])
            if dist < min_dist:
                min_dist = dist
                min_cls = cls
        grn_distances.append(min_dist)
        nearest_grn_class.append(min_cls)

    grn_df = pd.DataFrame(
        {
            "cell_id": all_cells,
            "grn_distance_score": grn_distances,
            "nearest_grn_class": nearest_grn_class,
        }
    )
    centroids_df = pd.DataFrame(centroids, index=auc_all.columns).T
    return grn_df, auc_all, centroids_df
