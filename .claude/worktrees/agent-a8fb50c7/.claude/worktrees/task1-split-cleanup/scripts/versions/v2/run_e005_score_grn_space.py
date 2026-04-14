"""
E005 Step 4: Score GRN space
==============================
1. Load regulon AUC matrix from pySCENIC output (aucell loom)
2. Score ALL cells (train/val/test) in regulon space using the model built from train_known
3. Compute grn_distance_score (cosine distance to regulon centroids)

Usage:
    python scripts/run_e005_score_grn_space.py
"""

import os
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.spatial.distance import cosine
import loompy
import json

# ============================================================
# Config
# ============================================================
DATA_PATH = r"D:\Desktop\研一\code\scGRN\data\human_immune_health_atlas_dc.patched.h5ad"
OUTPUT_DIR = r"D:\Desktop\研一\code\scGRN\outputs\E005\v2"        # v2 outputs
OUTPUT_DIR_V1 = r"D:\Desktop\研一\code\scGRN\outputs\E005\v1"      # reuse v1 expression scores
os.makedirs(OUTPUT_DIR, exist_ok=True)

AUCELL_LOOM = os.path.join(OUTPUT_DIR, "E005_aucell.loom")
SCORES_PATH = os.path.join(OUTPUT_DIR_V1, "E005_expression_scores.csv")  # step1 unchanged

KNOWN_CLASSES = ["cDC2", "pDC", "cDC1"]
UNKNOWN_CLASS = "ASDC"
LABEL_KEY = "AIFI_L2"


def log(msg):
    print(f"[E005-grn] {msg}", flush=True)


def main():
    # 1. Load expression scores (for split info and cell IDs)
    log("Loading expression scores...")
    scores_df = pd.read_csv(SCORES_PATH)

    train_known_cells = scores_df[scores_df["E005_split"] == "train_known"]["cell_id"].values
    all_cells = scores_df["cell_id"].values
    log(f"  Total cells: {len(all_cells)}, train_known: {len(train_known_cells)}")

    # 2. Load regulon AUC matrix from loom (train_known only)
    # First try loading from CSV (saved by run_e005_pyscenic_python.py)
    aucell_csv = os.path.join(OUTPUT_DIR, "E005_aucell_mtx.csv")
    if os.path.exists(aucell_csv):
        log(f"Loading regulon AUC matrix from CSV: {aucell_csv}")
        auc_df_train = pd.read_csv(aucell_csv, index_col=0)
        regulon_names_final = auc_df_train.columns.tolist()
        cell_ids_loom = auc_df_train.index.values
        auc_matrix_train = auc_df_train.values
        log(f"  AUC matrix shape: {auc_matrix_train.shape}")
        log(f"  Regulons: {len(regulon_names_final)}, first 5: {regulon_names_final[:5]}")
    else:
        log("Loading regulon AUC matrix from loom...")
        with loompy.connect(AUCELL_LOOM, mode="r") as ds:
            auc_matrix_train = ds[:, :]
            # Our script saves CellID in ra (row_attrs) and Regulons in ca (col_attrs)
            # Try both orientations
            ra_keys = list(ds.ra.keys())
            ca_keys = list(ds.ca.keys())
            log(f"  Loom row_attrs: {ra_keys}, col_attrs: {ca_keys}")
            log(f"  Loom matrix shape: {auc_matrix_train.shape}")

            # Detect regulon names and cell IDs from available attributes
            regulon_names = None
            cell_ids_train = None

            if "Regulons" in ca_keys:
                regulon_names = ds.ca["Regulons"]
            elif "Gene" in ca_keys:
                regulon_names = ds.ca["Gene"]
            elif "Regulons" in ra_keys:
                regulon_names = ds.ra["Regulons"]
            elif "Gene" in ra_keys:
                regulon_names = ds.ra["Gene"]

            if "CellID" in ra_keys:
                cell_ids_train = ds.ra["CellID"]
            elif "CellID" in ca_keys:
                cell_ids_train = ds.ca["CellID"]

        # Determine orientation based on shape vs known cell count
        if auc_matrix_train.shape[0] == len(train_known_cells):
            log("  Shape is (cells, regulons)")
            if regulon_names is not None and len(regulon_names) == auc_matrix_train.shape[1]:
                regulon_names_final = list(regulon_names)
            else:
                regulon_names_final = [f"R{i}" for i in range(auc_matrix_train.shape[1])]
            cell_ids_loom = cell_ids_train if cell_ids_train is not None and len(cell_ids_train) == auc_matrix_train.shape[0] else None
        elif auc_matrix_train.shape[1] == len(train_known_cells):
            log("  Transposing: (regulons, cells) -> (cells, regulons)")
            auc_matrix_train = auc_matrix_train.T
            if regulon_names is not None and len(regulon_names) == auc_matrix_train.shape[1]:
                regulon_names_final = list(regulon_names)
            else:
                regulon_names_final = [f"R{i}" for i in range(auc_matrix_train.shape[1])]
            cell_ids_loom = cell_ids_train if cell_ids_train is not None and len(cell_ids_train) == auc_matrix_train.shape[0] else None
        else:
            log(f"  WARNING: Matrix shape {auc_matrix_train.shape} doesn't match train_known count {len(train_known_cells)}")
            regulon_names_final = [f"R{i}" for i in range(auc_matrix_train.shape[1])]
            cell_ids_loom = None

    log(f"  Final AUC matrix: {auc_matrix_train.shape}")
    log(f"  Number of regulons: {len(regulon_names_final)}")

    # 3. Build regulon AUC DataFrame for train_known
    auc_df_train = pd.DataFrame(
        auc_matrix_train,
        columns=regulon_names_final[:auc_matrix_train.shape[1]]
    )
    if cell_ids_loom is not None and len(cell_ids_loom) == len(auc_df_train):
        auc_df_train.index = cell_ids_loom
    else:
        auc_df_train.index = train_known_cells[:len(auc_df_train)]

    log(f"  AUC DataFrame shape: {auc_df_train.shape}")

    # 4. Compute regulon centroids for each known class
    log("Computing regulon centroids on train_known...")
    train_labels = scores_df[scores_df["E005_split"] == "train_known"].set_index("cell_id")["true_label"]

    centroids = {}
    for cls in KNOWN_CLASSES:
        cls_cells = train_labels[train_labels == cls].index
        # Match with available cells in auc_df
        cls_cells_available = [c for c in cls_cells if c in auc_df_train.index]
        if len(cls_cells_available) == 0:
            log(f"  WARNING: No cells found for class {cls}")
            continue
        centroids[cls] = auc_df_train.loc[cls_cells_available].mean(axis=0).values
        log(f"  {cls}: centroid from {len(cls_cells_available)} cells")

    # 5. Score ALL cells in regulon space
    # For cells not in train_known, we need to run AUCell on them too
    # BUT pySCENIC aucell was only run on train_known
    # We need to score val/test cells using the regulon definitions

    log("Scoring all cells in regulon space...")
    log("  Loading full AnnData for projection...")
    adata = ad.read_h5ad(DATA_PATH)

    # We need the regulon definitions from the ctx step
    # Load regulons from the CSV output of pyscenic ctx
    ctx_reg_path = os.path.join(OUTPUT_DIR, "E005_regulons.csv")

    from pyscenic.aucell import aucell
    from ctxcore.genesig import GeneSignature
    import pickle
    import ast

    log("  Loading regulon definitions...")
    regulon_pkl = os.path.join(OUTPUT_DIR, "E005_regulons.pkl")

    # Delete stale pickle from previous bad parsing (if it has wrong regulons)
    if os.path.exists(regulon_pkl):
        with open(regulon_pkl, "rb") as f:
            old_regulons = pickle.load(f)
        # Check if it looks like a bad parse (regulon names are numbers)
        if len(old_regulons) > 0:
            first_name = old_regulons[0].name if hasattr(old_regulons[0], 'name') else str(old_regulons[0])
            try:
                float(first_name)
                log(f"  WARNING: Stale pickle with bad regulon names detected. Regenerating...")
                os.remove(regulon_pkl)
            except ValueError:
                pass

    if os.path.exists(regulon_pkl):
        with open(regulon_pkl, "rb") as f:
            regulons = pickle.load(f)
        log(f"  Loaded {len(regulons)} regulons from pickle")
    else:
        # Parse from ctx CSV output
        # The CSV was saved by modules2df with index=False, so TF names are LOST.
        # CSV structure:
        #   Row 0: "Enrichment,Enrichment,..." (MultiIndex level)
        #   Row 1: "AUC,NES,...,TargetGenes,RankAtMax" (column names)
        #   Row 2+: data
        # TargetGenes column contains: "[('GENE1', weight1), ('GENE2', weight2), ...]"
        # The TF itself has weight == 1.0 in the target genes list.

        log("  Parsing regulons from ctx CSV (recovering TF names from TargetGenes)...")
        ctx_df = pd.read_csv(ctx_reg_path, header=1)  # skip "Enrichment" row, use row 1 as header
        log(f"  ctx CSV shape: {ctx_df.shape}, columns: {list(ctx_df.columns)}")

        # Parse TargetGenes and recover TF names
        regulon_dict = {}  # TF_name -> {gene: max_weight}

        for idx, row in ctx_df.iterrows():
            tg_str = str(row.get("TargetGenes", ""))
            if not tg_str or tg_str == "nan":
                continue
            try:
                gene_list = ast.literal_eval(tg_str)
            except Exception:
                continue

            if not isinstance(gene_list, list) or len(gene_list) == 0:
                continue

            # Find TF: the gene with weight exactly 1.0
            tf_name = None
            for gene, weight in gene_list:
                if abs(float(weight) - 1.0) < 1e-10:
                    tf_name = gene
                    break

            if tf_name is None:
                # Fallback: gene with highest weight
                tf_name = max(gene_list, key=lambda x: float(x[1]))[0]
                log(f"    WARNING: No gene with weight=1.0 in row {idx}, using {tf_name}")

            if tf_name not in regulon_dict:
                regulon_dict[tf_name] = {}

            # Merge target genes (keep max weight across motif hits)
            for gene, weight in gene_list:
                w = float(weight)
                if gene not in regulon_dict[tf_name] or w > regulon_dict[tf_name][gene]:
                    regulon_dict[tf_name][gene] = w

        # Create GeneSignature regulons with (+) suffix (activating)
        regulons = []
        for tf_name, gene_weights in sorted(regulon_dict.items()):
            regulons.append(GeneSignature(
                name=f"{tf_name}(+)",
                gene2weight=gene_weights,
            ))

        log(f"  Recovered {len(regulons)} regulons from {len(ctx_df)} CSV rows")
        if len(regulons) > 0:
            log(f"  First 5 regulon names: {[r.name for r in regulons[:5]]}")

        # Save for future use
        if len(regulons) > 0:
            with open(regulon_pkl, "wb") as f:
                pickle.dump(regulons, f)
            log(f"  Saved regulons to {regulon_pkl}")

    # 6. Run AUCell on all cells
    log("  Running AUCell on all cells...")

    # Get raw count matrix for all cells
    expr_all = adata.raw.X
    if sparse.issparse(expr_all):
        expr_all_dense = expr_all.toarray()
    else:
        expr_all_dense = np.array(expr_all)

    expr_all_df = pd.DataFrame(
        expr_all_dense,
        index=adata.obs_names,
        columns=adata.raw.var_names
    )

    # Run AUCell
    auc_all = aucell(expr_all_df, regulons, num_workers=4)
    log(f"  AUCell result shape: {auc_all.shape}")

    # 7. Compute GRN distance scores for all cells
    log("Computing GRN distance scores...")

    # Recompute centroids using the full AUCell matrix
    # (use only train_known cells)
    regulon_cols = auc_all.columns.tolist()

    centroids_full = {}
    for cls in KNOWN_CLASSES:
        cls_cells = train_labels[train_labels == cls].index
        cls_cells_available = [c for c in cls_cells if c in auc_all.index]
        centroids_full[cls] = auc_all.loc[cls_cells_available].mean(axis=0).values
        log(f"  {cls}: centroid from {len(cls_cells_available)} cells")

    # Compute cosine distance to nearest centroid for each cell
    grn_distances = []
    nearest_grn_class = []

    for cell in all_cells:
        if cell not in auc_all.index:
            grn_distances.append(np.nan)
            nearest_grn_class.append("unknown")
            continue

        cell_auc = auc_all.loc[cell].values
        min_dist = np.inf
        min_cls = None

        for cls in KNOWN_CLASSES:
            d = cosine(cell_auc, centroids_full[cls])
            if d < min_dist:
                min_dist = d
                min_cls = cls

        grn_distances.append(min_dist)
        nearest_grn_class.append(min_cls)

    # 8. Save GRN scores
    grn_df = pd.DataFrame({
        "cell_id": all_cells,
        "grn_distance_score": grn_distances,
        "nearest_grn_class": nearest_grn_class,
    })

    # Add per-regulon AUC for cells that have it
    # (Save separately due to size)

    grn_df.to_csv(os.path.join(OUTPUT_DIR, "E005_grn_scores.csv"), index=False)
    log(f"  Saved E005_grn_scores.csv")

    # Save full AUCell matrix
    auc_all.to_csv(os.path.join(OUTPUT_DIR, "E005_aucell_all_cells.csv"))
    log(f"  Saved E005_aucell_all_cells.csv")

    # Save regulon centroids
    centroids_df = pd.DataFrame(centroids_full, index=regulon_cols).T
    centroids_df.to_csv(os.path.join(OUTPUT_DIR, "E005_regulon_centroids.csv"))
    log(f"  Saved E005_regulon_centroids.csv")

    log("=" * 60)
    log("Step 4 DONE. Next: python scripts/run_e005_dual_fusion.py")
    log("=" * 60)


if __name__ == "__main__":
    main()
