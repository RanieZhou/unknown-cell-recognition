from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from scgrn.config import load_config
from scgrn.paths import materialize_run_paths, resolve_run_paths
from scgrn.pipeline.run_infer import run_infer_backbone


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output_dir")
    args = parser.parse_args()

    config = load_config(
        args.config,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    paths = resolve_run_paths(config)
    paths = materialize_run_paths(paths, config)
    run_infer_backbone(config, paths)


if __name__ == "__main__":
    main()
