#!/usr/bin/env python3
"""Build Gen3's own manifest-driven OmiCLIP spot-feature cache.

Mirrors `scripts/precompute_gen3_uni2_spot_features.py` exactly, for the
OmiCLIP (CoCa ViT-L/14) tile encoder instead of UNI2: one real encode pass
per manifest sample, a MANDATORY immutable `--omiclip-pinned-revision` and
a real, local `--omiclip-checkpoint-path`
(`gen4.omiclip_encoder.FrozenOmiCLIPTileEncoder` never downloads anything
or falls back to a randomly-initialized model), full provenance recorded
once and reused across every sample, written to
`gen3_multiscale.gen4.omiclip_spot_cache`'s own cache directory
(`omiclip_gen3_spot_cache/`, resolved from `--config` exactly the way
`load_gen3_omiclip_spot_features` resolves it at load time -- see
`omiclip_spot_cache.cfg_cache_root`).

Driven by a real, already-built Gen3 dataset manifest
(`dataset_manifest.build_dataset_manifest`'s saved output) -- every
sample this script encodes is one this experiment's manifest actually
declared, never an independently-specified sample_id list.
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

from gen3_multiscale.data import example_builder
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.gen4.omiclip_encoder import FrozenOmiCLIPTileEncoder
from gen3_multiscale.gen4.omiclip_spot_cache import build_omiclip_spot_feature_cache, cfg_cache_root


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
        "--omiclip-checkpoint-path", required=True,
        help="MANDATORY: path to a real, already-downloaded WangGuangyuLab/Loki OmiCLIP "
             "checkpoint.pt file on local disk. FrozenOmiCLIPTileEncoder refuses to download "
             "anything or fall back to a randomly-initialized model.",
    )
    parser.add_argument(
        "--omiclip-pinned-revision", required=True,
        help="MANDATORY: an already-resolved, immutable Hugging Face commit SHA (40 lowercase "
             "hex characters) for WangGuangyuLab/Loki -- the exact commit "
             "--omiclip-checkpoint-path was downloaded from.",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")

    cfg = OmegaConf.load(args.config)
    cache_root = cfg_cache_root(cfg)
    manifest = load_dataset_manifest(args.manifest)
    ids = list(args.sample_ids or manifest["samples"].keys())
    print(f"Loading OmiCLIP (coca_ViT-L-14) tile encoder (revision "
          f"{args.omiclip_pinned_revision}) once for {len(ids)} sample(s)...", flush=True)
    encoder = FrozenOmiCLIPTileEncoder(
        args.omiclip_checkpoint_path, args.omiclip_pinned_revision, device=args.device,
    )
    for index, sample_id in enumerate(ids, start=1):
        print(f"Gen3 OmiCLIP spot-feature cache {index}/{len(ids)}: {sample_id}", flush=True)
        adata, patches, image_source_available = example_builder.load_sample_for_examples(
            manifest, sample_id,
        )
        path = build_omiclip_spot_feature_cache(
            cache_root, sample_id, np.asarray(adata.obs_names), patches, image_source_available,
            encoder, batch_size=args.batch_size,
        )
        print(f"{sample_id}: wrote {path}", flush=True)


if __name__ == "__main__":
    main()
