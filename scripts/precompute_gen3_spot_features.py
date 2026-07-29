#!/usr/bin/env python3
"""Build Gen3's own manifest-driven GigaPath spot-feature cache.

20th Codex re-audit (Step 5/6 boundary #3, response to the confirmed-
closed dense-WSI launch blocker): the legacy
``scripts/precompute_gigapath_samples.py`` cache is NOT a valid Gen3
input -- it uses an unpinned tile-encoder revision and carries no
provenance at all. This script builds
``gen3_multiscale/data/spot_feature_cache.py``'s own cache instead: one
real encode pass per manifest sample, with a MANDATORY immutable
``--tile-encoder-revision`` -- exactly the same discipline
``scripts/precompute_gigapath_wsi_tiles.py`` already enforces for the
dense WSI cache. Use the SAME revision for both, so
``tile_encoder_preflight.require_consistent_tile_encoder_provenance``
can later confirm every cache in an experiment agrees.

Driven by a real, already-built Gen3 dataset manifest
(``dataset_manifest.build_dataset_manifest``'s saved output) -- every
sample this script encodes is one this experiment's manifest actually
declared, never an independently-specified sample_id list that could
silently drift from what the manifest/masks were built against.
"""
from __future__ import annotations

import argparse

import numpy as np
from omegaconf import OmegaConf

from gen3_multiscale.data import example_builder
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.data.spot_feature_cache import (
    encode_gen3_spot_feature_cache, load_gigapath_tile_encoder_for_gen3,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--manifest", required=True,
        help="Path to a real, already-built Gen3 dataset manifest "
             "(dataset_manifest.build_dataset_manifest's saved JSON output).",
    )
    parser.add_argument("--sample-id", action="append", dest="sample_ids",
                         help="Restrict to these manifest sample_ids (default: every sample in "
                              "the manifest).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--tile-encoder-revision", required=True,
        help="MANDATORY: an already-resolved, immutable Hugging Face commit SHA (40 lowercase "
             "hex characters) for prov-gigapath/prov-gigapath's tile encoder -- see "
             "scripts/precompute_gigapath_wsi_tiles.py's own --tile-encoder-revision for the "
             "identical requirement applied to the dense WSI cache. Use the SAME revision for "
             "both so every cache in one experiment shares one tile-encoder identity.",
    )
    args = parser.parse_args()
    from src.models.conditioning import _validate_immutable_hf_revision
    _validate_immutable_hf_revision(args.tile_encoder_revision)

    cfg = OmegaConf.load(args.config)
    manifest = load_dataset_manifest(args.manifest)
    ids = list(args.sample_ids or manifest["samples"].keys())
    # 21st Codex re-audit hardening: "prefer loading the tile encoder
    # once per worker and reusing it across samples; the current CLI
    # reloads the large model for every sample." Load once here, reuse
    # the same encoder/provenance for every sample in this run.
    print(f"Loading GigaPath tile encoder (revision {args.tile_encoder_revision}) once for "
          f"{len(ids)} sample(s)...", flush=True)
    encoder, provenance = load_gigapath_tile_encoder_for_gen3(args.tile_encoder_revision, device=args.device)
    for index, sample_id in enumerate(ids, start=1):
        print(f"Gen3 spot-feature cache {index}/{len(ids)}: {sample_id}", flush=True)
        adata, patches, image_source_available = example_builder.load_sample_for_examples(
            manifest, sample_id,
        )
        encode_gen3_spot_feature_cache(
            cfg, sample_id, np.asarray(adata.obs_names), patches, image_source_available,
            encoder, provenance, device=args.device, batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
