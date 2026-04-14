"""
E005 Step 2: Export pySCENIC input from train_known
=====================================================
Exports:
  - Expression matrix (genes x cells) as a loom file
  - Only train_known cells (no unknown leakage)

Usage:
    python scripts/run_e005_export_pyscenic_input.py
"""

import os
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
import loompy

# ============================================================
# Config
# ============================================================
DATA_PATH = r"D:\Desktop\研一\code\scGRN\data\human_immune_health_atlas_dc.patched.h5ad"
OUTPUT_DIR_V1 = r"D:\Desktop\研一\code\scGRN\outputs\E005\v1"  # step1 scores
OUTPUT_DIR = r"D:\Desktop\研一\code\scGRN\outputs\E005\v2"     # v2 outputs
SCORES_PATH = os.path.join(OUTPUT_DIR_V1, "E005_expression_scores.csv")  # reuse v1 split

# v2 注意：基因过滤保持与 v1 相同（1%），避免内存溢出；v2 改进重点为 NES 阈值
GENE_FILTER_PCT = 0.01  # 与 v1 相同，不扩大基因集

# pySCENIC output paths
LOOM_OUT = os.path.join(OUTPUT_DIR, "E005_train_known_expr.loom")



def log(msg):
    print(f"[E005-export] {msg}", flush=True)


def main():
    # 1. Load expression scores to identify train_known cells
    log("Loading expression scores to get train_known cell IDs...")
    scores_df = pd.read_csv(SCORES_PATH)
    train_known_cells = scores_df[scores_df["E005_split"] == "train_known"]["cell_id"].values
    log(f"  train_known cells: {len(train_known_cells)}")

    # 2. Load full adata
    log("Loading AnnData...")
    adata = ad.read_h5ad(DATA_PATH)

    # 3. Subset to train_known
    adata_train = adata[train_known_cells].copy()
    log(f"  Subset shape: {adata_train.shape}")

    # 4. Use raw counts for pySCENIC
    # pySCENIC works best with raw counts
    expr_matrix = adata_train.raw.X
    if sparse.issparse(expr_matrix):
        expr_matrix = expr_matrix.toarray()
    expr_matrix = expr_matrix.astype(np.float32)

    gene_names = np.array(adata_train.raw.var_names)
    cell_names = np.array(adata_train.obs_names)

    log(f"  Expression matrix shape: {expr_matrix.shape}")
    log(f"  Gene names: {len(gene_names)}, first 5: {list(gene_names[:5])}")

    # 5. 过滤基因：v2 改为 0.5% 阈值（v1 为 1%），保留更多低丰度基因
    min_cells = int(GENE_FILTER_PCT * expr_matrix.shape[0])
    gene_mask = (expr_matrix > 0).sum(axis=0) >= min_cells
    expr_filtered = expr_matrix[:, gene_mask]
    genes_filtered = gene_names[gene_mask]
    log(f"  After gene filter (>={min_cells} cells, {GENE_FILTER_PCT*100:.1f}%): {expr_filtered.shape[1]} genes (from {len(gene_names)})")

    # 6. Export as loom file
    log(f"  Writing loom file: {LOOM_OUT}")

    # loompy expects (genes x cells) for default, but the convention used by
    # pySCENIC is (cells x genes) stored as row_attrs=CellID, col_attrs=Gene
    # Actually pySCENIC expects a loom with rows=cells, cols=genes
    # But loompy create() expects matrix shape (rows, cols) with corresponding attrs

    row_attrs = {"CellID": cell_names}
    col_attrs = {"Gene": genes_filtered}

    loompy.create(LOOM_OUT, expr_filtered, row_attrs=row_attrs, col_attrs=col_attrs)
    log(f"  Loom file written: {LOOM_OUT}")

    # 7. Also export as TSV for alternative pySCENIC input
    tsv_out = os.path.join(OUTPUT_DIR, "E005_train_known_expr.tsv")
    log(f"  Writing TSV file: {tsv_out}")
    expr_df = pd.DataFrame(expr_filtered, index=cell_names, columns=genes_filtered)
    expr_df.index.name = "Cell"
    expr_df.to_csv(tsv_out, sep="\t")
    log(f"  TSV file written: {tsv_out}")


    log("=" * 60)
    log(f"Step 2 DONE (v2: gene_filter={GENE_FILTER_PCT*100:.1f}%). Next: run_e005_pyscenic_python.py")
    log("=" * 60)


if __name__ == "__main__":
    main()
