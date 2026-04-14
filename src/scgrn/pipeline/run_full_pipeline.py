"""Full pipeline orchestration."""

from __future__ import annotations

from ..utils.logger import setup_logger
from .run_evaluate import run_evaluate
from .run_infer import run_infer
from .run_rescue import run_rescue
from .run_train import run_train


def run_full_pipeline(config: dict, paths):
    logger = setup_logger(paths.logs, "pipeline")
    run_train(config, paths, logger=logger)
    run_infer(config, paths, logger=logger)
    run_rescue(config, paths, logger=logger)
    return run_evaluate(config, paths, logger=logger)
