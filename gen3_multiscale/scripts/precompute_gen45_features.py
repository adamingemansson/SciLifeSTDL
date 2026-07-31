#!/usr/bin/env python3
"""Precompute the UNI2/scFoundation caches consumed by Gen4 and Gen5."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from gen3_multiscale.data.dataset_manifest import gene_panel_hash, load_dataset_manifest
from gen3_multiscale.data.example_builder import (
    load_expression_for_model_target_space,
    load_sample_for_examples,
)
from gen3_multiscale.gen4.scfoundation_cache import build_scfoundation_spot_feature_cache
from gen3_multiscale.gen4.scfoundation_encoder import FrozenSCFoundationEncoder
from gen3_multiscale.gen4.uni2_dense_wsi_cache import build_uni2_dense_wsi_cache
from gen3_multiscale.gen4.uni2_encoder import FrozenUNI2TileEncoder
from gen3_multiscale.gen4.uni2_spot_cache import build_uni2_spot_feature_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument(
        "--modalities", choices=("uni2", "scfoundation", "both"), default="both",
    )
    parser.add_argument("--uni2-checkpoint")
    parser.add_argument("--uni2-revision")
    parser.add_argument("--scfoundation-checkpoint")
    parser.add_argument("--scfoundation-vocab")
    parser.add_argument("--scfoundation-repo")
    parser.add_argument("--scfoundation-revision")
    parser.add_argument(
        "--report",
        help="Optional JSON path for the completed shard and exact encoder identities.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--spot-batch-size", type=int, default=32)
    parser.add_argument("--wsi-batch-size", type=int, default=32)
    parser.add_argument("--scfoundation-batch-size", type=int, default=256)
    parser.add_argument("--scfoundation-inference-batch-size", type=int, default=4)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    args = parser.parse_args()
    if args.n_shards <= 0 or not 0 <= args.shard_index < args.n_shards:
        raise ValueError("require n_shards > 0 and 0 <= shard_index < n_shards")
    if min(
        args.spot_batch_size,
        args.wsi_batch_size,
        args.scfoundation_batch_size,
        args.scfoundation_inference_batch_size,
    ) <= 0:
        raise ValueError("all batch sizes must be positive")

    manifest = load_dataset_manifest(args.manifest)
    sample_ids = sorted(manifest["samples"])[args.shard_index::args.n_shards]
    cache_root = args.cache_root or manifest.get("hest_cache_dir")
    if not cache_root:
        raise ValueError("--cache-root is required when the manifest has no hest_cache_dir")
    cfg = OmegaConf.create({
        "data": {
            "hest_data_dir": manifest["hest_data_dir"],
            "hest_cache_dir": cache_root,
            "uni2_slide_context_cache_dir": None,
        },
    })
    use_uni2 = args.modalities in {"uni2", "both"}
    use_scf = args.modalities in {"scfoundation", "both"}
    if use_uni2 and (not args.uni2_checkpoint or not args.uni2_revision):
        raise ValueError("UNI2 caching requires --uni2-checkpoint and --uni2-revision")
    if use_scf and (
        not args.scfoundation_checkpoint
        or not args.scfoundation_vocab
        or not args.scfoundation_repo
        or not args.scfoundation_revision
    ):
        raise ValueError(
            "scFoundation caching requires --scfoundation-checkpoint, "
            "--scfoundation-vocab, --scfoundation-repo, and "
            "--scfoundation-revision"
        )

    uni2 = (
        FrozenUNI2TileEncoder(
            args.uni2_checkpoint, args.uni2_revision, device=args.device,
        )
        if use_uni2 else None
    )
    scfoundation = (
        FrozenSCFoundationEncoder(
            args.scfoundation_checkpoint,
            args.scfoundation_vocab,
            list(manifest["gene_panel"]),
            repo_path=args.scfoundation_repo,
            repo_revision=args.scfoundation_revision,
            device=args.device,
            inference_microbatch_size=args.scfoundation_inference_batch_size,
        )
        if use_scf else None
    )
    completed = []
    for sample_id in sample_ids:
        # scFoundation consumes expression only.  The old shared path loaded
        # and aligned every sample's full H&E patch tensor even for
        # ``--modalities scfoundation``; those multi-gigabyte allocations
        # were scientifically irrelevant and could leave the process at a
        # very high RSS watermark across a cohort.  Only UNI2 needs pixels.
        if uni2 is not None:
            adata, patches, image_available = load_sample_for_examples(
                manifest, sample_id,
            )
        else:
            adata = load_expression_for_model_target_space(manifest, sample_id)
            patches = None
            image_available = None
        barcodes = np.asarray(adata.obs_names, dtype=str)
        if uni2 is not None:
            assert patches is not None and image_available is not None
            build_uni2_spot_feature_cache(
                cache_root,
                sample_id,
                barcodes,
                patches,
                image_available,
                uni2,
                batch_size=args.spot_batch_size,
            )
            build_uni2_dense_wsi_cache(
                cfg,
                sample_id,
                uni2,
                batch_size=args.wsi_batch_size,
                device=args.device,
            )
        if scfoundation is not None:
            if "_scilifestdl_raw_library_size" not in adata.obs:
                raise ValueError(
                    f"{sample_id}: raw library sizes are missing; cannot build scFoundation cache"
                )
            build_scfoundation_spot_feature_cache(
                cache_root,
                sample_id,
                barcodes,
                adata.X,
                gene_panel_hash(list(manifest["gene_panel"])),
                scfoundation,
                batch_size=args.scfoundation_batch_size,
                raw_library_size=np.asarray(
                    adata.obs["_scilifestdl_raw_library_size"], dtype=np.float32,
                ),
            )
        completed.append(sample_id)
        print(f"PASS: {sample_id}", flush=True)

    report = {
        "completed_sample_ids": completed,
        "n_completed": len(completed),
        "shard_index": args.shard_index,
        "n_shards": args.n_shards,
        "uni2_identity": uni2.identity.as_dict() if uni2 is not None else None,
        "scfoundation_identity": (
            scfoundation.identity.as_dict() if scfoundation is not None else None
        ),
    }
    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
        tmp_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        os.replace(tmp_path, report_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
