#!/usr/bin/env python3
"""Promote selected screening configs to matched 40k, three-seed configs."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def promote(path: Path, output_dir: Path, seeds: list[int], steps: int,
            unique_masks: int, novae_unique_masks: int, eval_samples: int) -> list[Path]:
    cfg = yaml.safe_load(path.read_text())
    base_name = str(cfg["experiment_name"])
    created = []
    uses_novae = str(cfg.get("data", {}).get("novae_mode", "disabled")) == "context_only"
    for seed in seeds:
        out_cfg = yaml.safe_load(yaml.safe_dump(cfg))
        exp = f"confirm_{base_name.removeprefix('screen_')}_seed{seed}"
        out_cfg["experiment_name"] = exp
        out_cfg["training"]["epochs"] = int(steps)
        out_cfg["training"]["seed"] = int(seed)
        selected_unique_masks = novae_unique_masks if uses_novae else (unique_masks or steps)
        out_cfg["training"]["unique_mask_count"] = int(selected_unique_masks)
        out_cfg["training"]["checkpoint_every_n_steps"] = 5000
        out_cfg["training"]["checkpoint_dir"] = f"results/checkpoints/complexity_ladder/{exp}"
        out_cfg["validation"]["every_n_steps"] = 1000
        out_cfg["validation"]["patience_checks"] = 8
        out_cfg["evaluation"]["n_samples"] = int(eval_samples)
        mask_kind = "novae" if uses_novae else "standard"
        out_cfg["evaluation"]["training_mask_bank_path"] = (
            f"results/mask_banks/training/complexity_ladder/confirm_{mask_kind}_"
            f"seed{seed}_{steps}_u{selected_unique_masks}.json"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{exp}.yaml"
        out_path.write_text(yaml.safe_dump(out_cfg, sort_keys=False, width=1000))
        created.append(out_path)
    return created


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("configs/complexity_ladder/confirm"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument(
        "--unique-masks", type=int, default=0,
        help="unique masks for non-Novae confirmation; 0 means one fresh mask per step",
    )
    parser.add_argument("--novae-unique-masks", type=int, default=512)
    parser.add_argument("--eval-samples", type=int, default=20)
    args = parser.parse_args()

    created = []
    for path in args.configs:
        created.extend(promote(
            path, args.output_dir, args.seeds, args.steps,
            args.unique_masks, args.novae_unique_masks, args.eval_samples,
        ))
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
