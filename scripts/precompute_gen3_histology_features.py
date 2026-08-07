#!/usr/bin/env python3
"""Build Gen3's own manifest-driven histology-morphology spot-feature
cache.

Deterministic, CPU-only, no model or checkpoint dependency -- purely
`conditional_wae/histology_features.py`'s fixed formulas (RGB/stain
color statistics, GLCM texture, multiscale spatial pooling) over each
sample's real H&E patches and coordinates.

    python scripts/precompute_gen3_histology_features.py \\
        --config <any resolved config with data.hest_data_dir/hest_cache_dir set> \\
        --manifest /path/to/dataset_manifest.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from omegaconf import OmegaConf

from gen3_multiscale.conditional_wae.histology_cache import build_histology_feature_cache, cfg_cache_root
from gen3_multiscale.data import example_builder
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--manifest", required=True,
        help="Path to a real, already-built Gen3 dataset manifest "
             "(dataset_manifest.build_dataset_manifest's saved JSON output).",
    )
    parser.add_argument("--sample-id", action="append", dest="sample_ids",
                         help="Restrict to these manifest sample_ids (default: every sample in "
                              "the manifest).")
    parser.add_argument("--neighbor-k", type=int, default=6)
    parser.add_argument("--regional-radius-multiplier", type=float, default=5.0)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    manifest = load_dataset_manifest(args.manifest)
    ids = list(args.sample_ids or manifest["samples"].keys())
    cache_root = cfg_cache_root(cfg)
    for index, sample_id in enumerate(ids, start=1):
        print(f"Gen3 histology-feature cache {index}/{len(ids)}: {sample_id}", flush=True)
        adata, patches, image_source_available = example_builder.load_sample_for_examples(manifest, sample_id)
        coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
        build_histology_feature_cache(
            cache_root, sample_id, np.asarray(adata.obs_names), coords, patches, image_source_available,
            neighbor_k=args.neighbor_k, regional_radius_multiplier=args.regional_radius_multiplier,
        )


if __name__ == "__main__":
    main()
