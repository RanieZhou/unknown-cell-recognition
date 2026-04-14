"""pySCENIC input export and regulon construction."""

from __future__ import annotations

from pathlib import Path

import loompy
import numpy as np
import pandas as pd
from scipy import sparse

from ..constants import SPLIT_COLUMN, SPLIT_TRAIN


def export_pyscenic_input(adata, expression_scores: pd.DataFrame, config: dict, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gene_filter_pct = float(config["grn"]["pyscenic"]["gene_filter_pct"])

    train_known_cells = expression_scores.loc[
        expression_scores[SPLIT_COLUMN] == SPLIT_TRAIN, "cell_id"
    ].astype(str).tolist()
    adata_train = adata[train_known_cells].copy()

    expr_matrix = adata_train.layers[config["data"]["model_layer"]]
    if sparse.issparse(expr_matrix):
        expr_matrix = expr_matrix.toarray()
    expr_matrix = expr_matrix.astype(np.float32)

    gene_names = np.array(adata_train.var_names)
    cell_names = np.array(adata_train.obs_names)

    if expr_matrix.shape[1] != len(gene_names):
        raise ValueError(
            "Expression matrix columns do not match the current feature axis. "
            "Use the active model layer and var_names from the same AnnData view."
        )

    min_cells = int(gene_filter_pct * expr_matrix.shape[0])
    gene_mask = (expr_matrix > 0).sum(axis=0) >= min_cells
    expr_filtered = expr_matrix[:, gene_mask]
    genes_filtered = gene_names[gene_mask]

    loom_out = output_dir / "train_known_expr.loom"
    tsv_out = output_dir / "train_known_expr.tsv"
    loompy.create(
        str(loom_out),
        expr_filtered,
        row_attrs={"CellID": cell_names},
        col_attrs={"Gene": genes_filtered},
    )

    expr_df = pd.DataFrame(expr_filtered, index=cell_names, columns=genes_filtered)
    expr_df.index.name = "Cell"
    expr_df.to_csv(tsv_out, sep="\t")
    return {"loom": loom_out, "tsv": tsv_out}


def run_pyscenic_python(config: dict, pyscenic_dir: Path, *, use_cached: bool = False) -> dict[str, Path]:
    import logging
    import time
    from functools import partial

    import pandas as pd
    from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase, MemoryDecorator
    from pyscenic.aucell import aucell
    from pyscenic.prune import df2regulons
    from pyscenic.transform import module2features_auc1st_impl, modules2df
    from pyscenic.utils import load_motif_annotations, modules_from_adjacencies

    pyscenic_cfg = config["grn"]["pyscenic"]
    pyscenic_dir.mkdir(parents=True, exist_ok=True)

    expr_tsv = pyscenic_dir / "train_known_expr.tsv"
    grn_adj = pyscenic_dir / "grn_adjacencies.tsv"
    regulons_csv = pyscenic_dir / "regulons.csv"
    regulons_pkl = pyscenic_dir / "regulons.pkl"
    aucell_loom = pyscenic_dir / "aucell_train.loom"
    aucell_csv = pyscenic_dir / "aucell_train.csv"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("scgrn-pyscenic")

    ex_mtx = pd.read_csv(expr_tsv, sep="\t", index_col=0)
    tf_list = Path(pyscenic_cfg["db_dir"]) / pyscenic_cfg["tf_list"]
    ranking_db = Path(pyscenic_cfg["db_dir"]) / pyscenic_cfg["ranking_db"]
    motif_tbl = Path(pyscenic_cfg["db_dir"]) / pyscenic_cfg["motif_table"]

    if use_cached and grn_adj.exists() and grn_adj.stat().st_size > 0:
        adjacencies = pd.read_csv(grn_adj, sep="\t")
    else:
        with tf_list.open("r", encoding="utf-8") as handle:
            tf_names = [line.strip() for line in handle if line.strip()]
        from arboreto.algo import grnboost2

        t0 = time.time()
        adjacencies = grnboost2(expression_data=ex_mtx, tf_names=tf_names, verbose=True)
        adjacencies.to_csv(grn_adj, sep="\t", index=False)
        logger.info("GRNBoost2 done in %.1f min", (time.time() - t0) / 60)

    if use_cached and regulons_csv.exists() and regulons_pkl.exists():
        pass
    else:
        modules = list(modules_from_adjacencies(adjacencies, ex_mtx, rho_mask_dropouts=True))
        rnkdb = MemoryDecorator(RankingDatabase(fname=str(ranking_db), name=ranking_db.name))
        motif_annotations = load_motif_annotations(
            str(motif_tbl),
            motif_similarity_fdr=0.001,
            orthologous_identity_threshold=0.0,
        )
        module2features_func = partial(
            module2features_auc1st_impl,
            rank_threshold=1500,
            auc_threshold=0.05,
            nes_threshold=float(pyscenic_cfg["nes_threshold"]),
            filter_for_annotation=True,
        )
        df_motifs = modules2df(
            rnkdb,
            modules,
            motif_annotations=motif_annotations,
            module2features_func=module2features_func,
            weighted_recovery=False,
        )
        df_motifs.to_csv(regulons_csv)
        regulons = df2regulons(df_motifs)
        from .load_regulon import save_regulons

        save_regulons(regulons, regulons_pkl)

    from .load_regulon import load_regulons

    regulons = load_regulons(regulons_pkl)
    auc_mtx = aucell(ex_mtx, regulons, num_workers=int(pyscenic_cfg.get("aucell_workers", 1)))
    if aucell_loom.exists():
        aucell_loom.unlink()
    loompy.create(
        str(aucell_loom),
        auc_mtx.values,
        row_attrs={"CellID": np.array(auc_mtx.index)},
        col_attrs={"Regulons": np.array(auc_mtx.columns)},
    )
    auc_mtx.to_csv(aucell_csv)

    return {
        "expr_tsv": expr_tsv,
        "grn_adjacencies": grn_adj,
        "regulons_csv": regulons_csv,
        "regulons_pkl": regulons_pkl,
        "aucell_train_loom": aucell_loom,
        "aucell_train_csv": aucell_csv,
    }
