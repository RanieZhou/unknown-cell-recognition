"""Regulon loading and parsing helpers."""

from __future__ import annotations

import ast
import pickle
from pathlib import Path


def load_regulons(regulon_pkl: Path):
    with regulon_pkl.open("rb") as handle:
        return pickle.load(handle)


def save_regulons(regulons, regulon_pkl: Path) -> None:
    regulon_pkl.parent.mkdir(parents=True, exist_ok=True)
    with regulon_pkl.open("wb") as handle:
        pickle.dump(regulons, handle)


def parse_regulons_from_ctx_csv(ctx_csv: Path):
    import pandas as pd
    from ctxcore.genesig import GeneSignature

    ctx_df = pd.read_csv(ctx_csv, header=1)
    regulon_dict: dict[str, dict[str, float]] = {}
    for idx, row in ctx_df.iterrows():
        tg_str = str(row.get("TargetGenes", ""))
        if not tg_str or tg_str == "nan":
            continue
        try:
            gene_list = ast.literal_eval(tg_str)
        except Exception:
            continue
        if not isinstance(gene_list, list) or not gene_list:
            continue

        tf_name = None
        for gene, weight in gene_list:
            if abs(float(weight) - 1.0) < 1e-10:
                tf_name = gene
                break
        if tf_name is None:
            tf_name = max(gene_list, key=lambda item: float(item[1]))[0]

        regulon_dict.setdefault(tf_name, {})
        for gene, weight in gene_list:
            weight = float(weight)
            if gene not in regulon_dict[tf_name] or weight > regulon_dict[tf_name][gene]:
                regulon_dict[tf_name][gene] = weight

    return [
        GeneSignature(name=f"{tf_name}(+)", gene2weight=gene_weights)
        for tf_name, gene_weights in sorted(regulon_dict.items())
    ]
