#!/usr/bin/env python3
"""Populate frozen GigaPath patch caches without loading any GEX matrix.

Multi-process held-out runs otherwise discover a missing cache concurrently
and each process may instantiate the large frozen image encoder. This helper
warms each declared sample exactly once before the smoke/full batches start.
It reads only H&E patches and barcodes; validation/test expression remains
unloaded until the training entry point reaches the appropriate split.
"""
from __future__ import annotations

import argparse

from omegaconf import OmegaConf

from src.data import loaders
from src.training.train import get_gigapath_features


def main(config_path: str, sample_ids: list[str] | None = None) -> None:
    cfg = OmegaConf.load(config_path)
    ids = list(sample_ids or cfg.data.get("sample_ids", []))
    if not ids:
        ids = [str(cfg.data.sample_id)]
    for index, sample_id in enumerate(ids, start=1):
        print(f"GigaPath cache {index}/{len(ids)}: {sample_id}", flush=True)
        patches, barcodes = loaders.load_hest_patches(cfg.data.hest_data_dir, sample_id)
        features = get_gigapath_features(cfg, patches, barcodes, sample_id=sample_id)
        print(
            f"GigaPath cache ready: {sample_id} -> {tuple(features.shape)}",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-id", action="append", dest="sample_ids")
    args = parser.parse_args()
    main(args.config, args.sample_ids)
