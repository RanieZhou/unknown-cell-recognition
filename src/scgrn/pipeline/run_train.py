"""Train-stage entrypoint."""

from __future__ import annotations

import pandas as pd

from ..backbone.registry import get_backbone_adapter
from ..constants import SPLIT_COLUMN
from ..data.load_data import load_dataset
from ..data.make_split import make_split
from ..utils.io import write_dataframe, write_json
from ..utils.logger import setup_logger
from ..utils.seed import seed_everything


def run_train(config: dict, paths, *, logger=None):
    logger = logger or setup_logger(paths.logs, "train")
    seed_everything(int(config["training"]["seed"]))
    logger.info("Loading dataset for training stage")
    adata = load_dataset(config)
    adata, train_idx, val_idx, test_known_idx, unknown_idx, split_summary = make_split(adata, config)
    adapter = get_backbone_adapter(config)
    logger.info("Running backbone adapter: %s", adapter.name)
    train_result = adapter.train(
        adata,
        train_idx,
        val_idx,
        config,
        paths.checkpoints,
        use_cached=bool(config["runtime"]["use_cached"]),
    )

    write_dataframe(paths.intermediate / "split_summary.csv", split_summary)
    split_index_df = pd.DataFrame({"cell_id": adata.obs_names.astype(str), SPLIT_COLUMN: adata.obs[SPLIT_COLUMN].values})
    label_key = train_result.metadata.get("label_key")
    if label_key and label_key in train_result.prepared_adata.obs:
        split_index_df[label_key] = train_result.prepared_adata.obs[label_key].values
    write_dataframe(paths.intermediate / "split_assignments.csv", split_index_df)
    write_json(
        paths.metrics / "train_stage_metadata.json",
        {
            "backbone_name": train_result.backbone_name,
            "seed": int(config["training"]["seed"]),
            "n_train_known": len(train_idx),
            "n_val_known": len(val_idx),
            "n_test_known": len(test_known_idx),
            "n_test_unknown": len(unknown_idx),
            "known_classes": config["data"]["known_classes"],
            "unknown_classes": config["data"]["unknown_classes"],
        },
    )
    return {
        "adata": train_result.prepared_adata,
        "adata_train": train_result.training_adata,
        "backbone_model": train_result.model,
        "backbone_name": train_result.backbone_name,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_known_idx": test_known_idx,
        "unknown_idx": unknown_idx,
        "split_summary": split_summary,
    }
