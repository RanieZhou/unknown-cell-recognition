"""
E005 Step 3: pySCENIC via Python API (Windows-compatible)
=========================================================
Bypasses CLI multiprocessing issues on Windows by running
pySCENIC ctx/aucell in the MAIN process (no subprocess spawning).

Usage (in pyscenic env):
    python scripts/run_e005_pyscenic_python.py
"""

import os
import sys
import time
import pickle
import logging
import tempfile
from functools import partial
from math import ceil

import numpy as np
import pandas as pd

# ============================================================
# Config
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "E005", "v2")  # v2
DATA_DIR = os.path.join(PROJECT_DIR, "data", "pyscenic_dbs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPR_TSV = os.path.join(OUTPUT_DIR, "E005_train_known_expr.tsv")
TF_LIST = os.path.join(DATA_DIR, "allTFs_hg38.txt")
RANKING_DB_PATH = os.path.join(
    DATA_DIR,
    "hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather",
)
MOTIF_TBL = os.path.join(DATA_DIR, "motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl")

GRN_ADJ = os.path.join(OUTPUT_DIR, "E005_grn_adjacencies.tsv")
CTX_REG = os.path.join(OUTPUT_DIR, "E005_regulons.csv")
AUCELL_LOOM = os.path.join(OUTPUT_DIR, "E005_aucell.loom")

# v2 改进：降低 NES 阈值以获取更多 regulon
NES_THRESHOLD = 2.5  # v1 为 3.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOG = logging.getLogger("E005-pyscenic")


def main():
    # ==============================================================
    # 0. Load expression matrix
    # ==============================================================
    LOG.info("Loading expression matrix from TSV ...")
    t0 = time.time()
    ex_mtx = pd.read_csv(EXPR_TSV, sep="\t", index_col=0)
    LOG.info(f"  Shape: {ex_mtx.shape}  ({time.time()-t0:.1f}s)")

    # ==============================================================
    # 1. GRN — skip if adjacencies already exist
    # ==============================================================
    if os.path.exists(GRN_ADJ) and os.path.getsize(GRN_ADJ) > 0:
        LOG.info(f"GRN adjacencies already exist ({GRN_ADJ}), skipping.")
        adjacencies = pd.read_csv(GRN_ADJ, sep="\t")
    else:
        LOG.info("Running GRN (GRNBoost2) ...")
        with open(TF_LIST, "r") as f:
            tf_names = [line.strip() for line in f if line.strip()]
        from arboreto.algo import grnboost2

        t1 = time.time()
        adjacencies = grnboost2(
            expression_data=ex_mtx, tf_names=tf_names, verbose=True
        )
        adjacencies.to_csv(GRN_ADJ, sep="\t", index=False)
        LOG.info(f"  GRN done. Time: {(time.time()-t1)/60:.1f} min")

    LOG.info(f"  Adjacencies shape: {adjacencies.shape}")

    # ==============================================================
    # 2. CTX — motif enrichment / pruning  (SINGLE PROCESS)
    # ==============================================================
    LOG.info("=" * 60)
    LOG.info("Running CTX (motif enrichment) — single-process mode ...")
    t2 = time.time()

    from pyscenic.utils import modules_from_adjacencies
    from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase, MemoryDecorator
    from pyscenic.prune import df2regulons
    from pyscenic.transform import modules2df, module2features_auc1st_impl
    from pyscenic.utils import load_motif_annotations

    # 2a. Create modules from adjacencies
    LOG.info("  Creating modules from adjacencies ...")
    modules = list(
        modules_from_adjacencies(adjacencies, ex_mtx, rho_mask_dropouts=True)
    )
    LOG.info(f"  Modules created: {len(modules)}")

    # 2b. Load ranking database into memory
    LOG.info("  Loading ranking database ...")
    rnkdb_raw = RankingDatabase(fname=RANKING_DB_PATH, name=os.path.basename(RANKING_DB_PATH))
    rnkdb = MemoryDecorator(rnkdb_raw)
    LOG.info(f"  Database loaded: {rnkdb.name}")

    # 2c. Load motif annotations
    LOG.info("  Loading motif annotations ...")
    motif_annotations = load_motif_annotations(
        MOTIF_TBL,
        motif_similarity_fdr=0.001,
        orthologous_identity_threshold=0.0,
    )
    LOG.info(f"  Motif annotations loaded: {motif_annotations.shape}")

    # 2d. Run enrichment — directly in main process (no subprocess!)
    LOG.info("  Running motif enrichment (this may take a while) ...")
    module2features_func = partial(
        module2features_auc1st_impl,
        rank_threshold=1500,
        auc_threshold=0.05,
        nes_threshold=NES_THRESHOLD,  # v2: 2.5 (v1 was 3.0)
        filter_for_annotation=True,
    )

    df_motifs = modules2df(
        rnkdb,
        modules,
        motif_annotations=motif_annotations,
        module2features_func=module2features_func,
        weighted_recovery=False,
    )

    elapsed_ctx = (time.time() - t2) / 60
    LOG.info(f"  CTX done. Time: {elapsed_ctx:.1f} min")
    LOG.info(f"  Motif enrichment table shape: {df_motifs.shape}")

    # 2e. Save (with index=True to preserve TF names for downstream)
    df_motifs.to_csv(CTX_REG)
    LOG.info(f"  Regulons CSV saved to: {CTX_REG}")

    # 2f. Convert to regulon objects
    regulons = df2regulons(df_motifs)
    LOG.info(f"  Regulons: {len(regulons)}")

    # 2g. Save regulons as pickle for Step 4
    regulon_pkl = os.path.join(OUTPUT_DIR, "E005_regulons.pkl")
    with open(regulon_pkl, "wb") as f:
        pickle.dump(regulons, f)
    LOG.info(f"  Regulons pickle saved to: {regulon_pkl}")

    # ==============================================================
    # 3. AUCell — regulon activity scoring
    # ==============================================================
    LOG.info("=" * 60)
    LOG.info("Running AUCell ...")
    t3 = time.time()

    from pyscenic.aucell import aucell

    auc_mtx = aucell(ex_mtx, regulons, num_workers=1)
    LOG.info(f"  AUCell matrix shape: {auc_mtx.shape}")
    LOG.info(f"  AUCell done. Time: {(time.time()-t3)/60:.1f} min")

    # Save as loom
    import loompy

    if os.path.exists(AUCELL_LOOM):
        os.remove(AUCELL_LOOM)
    row_attrs = {"CellID": np.array(auc_mtx.index)}
    col_attrs = {"Regulons": np.array(auc_mtx.columns)}
    loompy.create(AUCELL_LOOM, auc_mtx.values, row_attrs=row_attrs, col_attrs=col_attrs)
    LOG.info(f"  Output loom: {AUCELL_LOOM}")

    # Also save as CSV for easier downstream use
    auc_csv = os.path.join(OUTPUT_DIR, "E005_aucell_mtx.csv")
    auc_mtx.to_csv(auc_csv)
    LOG.info(f"  Output CSV:  {auc_csv}")

    # ==============================================================
    # Done
    # ==============================================================
    total = (time.time() - t0) / 60
    LOG.info("=" * 60)
    LOG.info(f"pySCENIC pipeline DONE. Total time: {total:.1f} min")
    LOG.info(f"  Next: python scripts/run_e005_score_grn_space.py")
    LOG.info("=" * 60)


if __name__ == "__main__":
    main()
