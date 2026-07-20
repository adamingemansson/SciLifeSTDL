"""Build a persistent validation/test mask bank for one experiment config."""
from __future__ import annotations
import argparse
from pathlib import Path
from omegaconf import OmegaConf

from src.training.train import load_adata, _load_images
from src.data import loaders
from src.data.mask_bank import ensure_mask_bank


def main(config: str, output: str | None = None):
    cfg = OmegaConf.load(config)
    adata = load_adata(cfg)
    adata, _ = _load_images(cfg, adata)
    coords = loaders.get_coords_3d(adata)
    slice_ids = adata.obs["slice_id"].to_numpy()
    evaluation = cfg.get("evaluation", {})
    output = output or evaluation.get(
        "mask_bank_path", f"results/mask_banks/{cfg.experiment_name}.json"
    )
    counts = {
        "validation": int(evaluation.get("n_validation_masks", 4)),
        "test": int(evaluation.get("n_test_masks", 8)),
    }
    seeds = {
        "validation": int(evaluation.get("validation_seed", 700_000)),
        "test": int(evaluation.get("test_seed", 900_000)),
    }
    ensure_mask_bank(output, coords, slice_ids, adata.obs_names, cfg.masking, counts, seeds)
    print(Path(output).resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    main(args.config, args.output)
