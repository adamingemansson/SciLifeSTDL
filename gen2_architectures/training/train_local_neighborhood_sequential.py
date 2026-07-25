"""Run several train_local_neighborhood.py configs back-to-back as ONE
unsupervised job on a single GPU -- e.g. Architecture 1 followed by its
1b/1c ablations, so there's no manual "wait for Arch1, then launch 1b,
then launch 1c" babysitting.

Each config keeps its own checkpoint_dir (so they never collide) and, by
default, its own max_wall_clock_hours (36h each for arch1/1b/1c --
running all three back-to-back at full budget takes ~4.5 days). Pass
--max_wall_clock_hours_override to cap EVERY config in the chain to a
shorter budget instead -- sensible for the ablations specifically, since
they exist to interpret the primary comparison, not to be trained as
deeply as it.

Usage:
    python3 -m gen2_architectures.training.train_local_neighborhood_sequential \\
        --configs gen2_architectures/configs/arch1_gpt_baseline.yaml \\
                  gen2_architectures/configs/arch1b_image_only_baseline.yaml \\
                  gen2_architectures/configs/arch1c_gene_only_baseline.yaml
"""
from __future__ import annotations

import argparse

from gen2_architectures.training import train_local_neighborhood


def main(
    config_paths: list[str], smoke_steps: int | None = None,
    max_wall_clock_hours_override: float | None = None,
) -> None:
    for i, config_path in enumerate(config_paths):
        print(f"=== [{i + 1}/{len(config_paths)}] starting {config_path} ===")
        train_local_neighborhood.main(
            config_path, smoke_steps=smoke_steps,
            max_wall_clock_hours_override=max_wall_clock_hours_override,
        )
        print(f"=== [{i + 1}/{len(config_paths)}] done: {config_path} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, nargs="+", required=True,
                         help="Config paths, run in the given order.")
    parser.add_argument("--smoke_steps", type=int, default=None,
                         help="Passed through to every config in the chain.")
    parser.add_argument("--max_wall_clock_hours_override", type=float, default=None,
                         help="If set, overrides EVERY config's own max_wall_clock_hours "
                              "for this run only (the YAML files are never modified).")
    args = parser.parse_args()
    main(args.configs, args.smoke_steps, args.max_wall_clock_hours_override)
