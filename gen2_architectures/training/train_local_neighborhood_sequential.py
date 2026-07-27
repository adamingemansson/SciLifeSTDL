"""Run several train_local_neighborhood.py configs back-to-back as ONE
unsupervised job on a single GPU -- e.g. Architecture 1 followed by its
1b/1c ablations, so there's no manual "wait for Arch1, then launch 1b,
then launch 1c" babysitting.

Each config keeps its own checkpoint_dir (so they never collide). Budget
control (2026-07-25, revised): by default every config uses its own
YAML's max_wall_clock_hours, which is how the whole CHAIN ends up much
longer than any single GPU's individual run (e.g. 36h + 12h + 12h = 60h
for Arch1+1b+1c, well past the ~36h the other 3 GPUs' jobs take) -- if
you want the whole chain to fit in roughly the SAME wall-clock window as
every other GPU's job, use --max_wall_clock_hours_overrides to give each
config its own explicit share of one shared total instead (one value per
--configs entry, in order); --max_wall_clock_hours_override (singular)
is also still available for the simpler "cap every config in the chain
to the same budget" case, but do NOT use it when the configs need
different depths (e.g. it would cap Architecture 1 down to the ablations'
shorter budget too, not just the ablations).

Usage:
    # give Arch1/1b/1c explicit, DIFFERENT shares of one ~36h window
    # (e.g. 26h + 5h + 5h) instead of each using its own full YAML budget:
    python3 -m gen2_architectures.training.train_local_neighborhood_sequential \\
        --configs gen2_architectures/configs/arch1_gpt_baseline.yaml \\
                  gen2_architectures/configs/arch1b_image_only_baseline.yaml \\
                  gen2_architectures/configs/arch1c_gene_only_baseline.yaml \\
        --max_wall_clock_hours_overrides 26 5 5
"""
from __future__ import annotations

import argparse

from gen2_architectures.training import train_local_neighborhood


def main(
    config_paths: list[str], smoke_steps: int | None = None,
    max_wall_clock_hours_override: float | None = None,
    max_wall_clock_hours_overrides: list[float] | None = None,
    skip_final_eval: bool = False,
) -> None:
    if max_wall_clock_hours_override is not None and max_wall_clock_hours_overrides is not None:
        raise ValueError(
            "pass at most one of max_wall_clock_hours_override (uniform) or "
            "max_wall_clock_hours_overrides (per-config list), not both"
        )
    if max_wall_clock_hours_overrides is not None and len(max_wall_clock_hours_overrides) != len(config_paths):
        raise ValueError(
            f"max_wall_clock_hours_overrides has {len(max_wall_clock_hours_overrides)} entries "
            f"but config_paths has {len(config_paths)} -- need exactly one hours value per config"
        )

    for i, config_path in enumerate(config_paths):
        hours_override = (
            max_wall_clock_hours_overrides[i] if max_wall_clock_hours_overrides is not None
            else max_wall_clock_hours_override
        )
        print(f"=== [{i + 1}/{len(config_paths)}] starting {config_path}"
              f"{f' (max_wall_clock_hours={hours_override})' if hours_override is not None else ''} ===")
        train_local_neighborhood.main(
            config_path, smoke_steps=smoke_steps, max_wall_clock_hours_override=hours_override,
            skip_final_eval=skip_final_eval,
        )
        print(f"=== [{i + 1}/{len(config_paths)}] done: {config_path} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, nargs="+", required=True,
                         help="Config paths, run in the given order.")
    parser.add_argument("--smoke_steps", type=int, default=None,
                         help="Passed through to every config in the chain.")
    parser.add_argument("--max_wall_clock_hours_override", type=float, default=None,
                         help="If set, overrides EVERY config's own max_wall_clock_hours to "
                              "this SAME value for this run only (the YAML files are never "
                              "modified). Mutually exclusive with --max_wall_clock_hours_overrides.")
    parser.add_argument("--max_wall_clock_hours_overrides", type=float, nargs="+", default=None,
                         help="Per-config hours, one value per --configs entry in order "
                              "(e.g. --max_wall_clock_hours_overrides 26 5 5 for a 3-config "
                              "chain) -- use this when the configs need different depths within "
                              "one shared total wall-clock window. Mutually exclusive with "
                              "--max_wall_clock_hours_override.")
    parser.add_argument("--skip_final_eval", action="store_true",
                         help="Passed through to every config in the chain -- skip the final "
                              "held-out test evaluation entirely (the checkpoint is still saved).")
    args = parser.parse_args()
    main(
        args.configs, args.smoke_steps,
        args.max_wall_clock_hours_override, args.max_wall_clock_hours_overrides,
        skip_final_eval=args.skip_final_eval,
    )
