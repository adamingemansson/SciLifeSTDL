#!/usr/bin/env python3
"""Prepare one verified, shared initialization bundle for manual Gen3 runs.

Architecture 4's numeric residual basis is deliberately a placeholder here:
the shared-initialization loader excludes that external buffer, so the real
training-only basis fitted after Architecture 3 is selected is never replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.models.gene_basis import GeneResidualBasis
from gen3_multiscale.models.model_factory import build_architecture, persist_four_architecture_initializations
from gen3_multiscale.models.slide_encoder import FrozenGigaPathSlideEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gigapath-slide-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists; initialization bundles are immutable")
    manifest = load_dataset_manifest(args.manifest)
    gene_names = [str(gene) for gene in manifest["gene_panel"]]
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    configs = {
        f"architecture{idx}": OmegaConf.to_container(
            OmegaConf.load(config_dir / f"architecture{idx}.yaml"), resolve=True,
        )
        for idx in range(1, 5)
    }
    rank = int(configs["architecture4"]["model"]["params"]["gene_basis_rank"])
    if rank > len(gene_names):
        raise ValueError(f"Architecture 4 basis rank {rank} exceeds gene count {len(gene_names)}")
    placeholder_matrix = torch.zeros((rank, len(gene_names)), dtype=torch.float32)
    placeholder_matrix[torch.arange(rank), torch.arange(rank)] = 1.0
    placeholder_basis = GeneResidualBasis(
        basis=placeholder_matrix,
        gene_names=tuple(gene_names),
        gene_names_hash=hashlib.sha256("\0".join(gene_names).encode()).hexdigest(),
    )

    slide_encoder = FrozenGigaPathSlideEncoder(args.gigapath_slide_checkpoint)
    models = {}
    for name, config in configs.items():
        kwargs = {}
        if name in {"architecture3", "architecture4"}:
            kwargs.update(
                slide_encoder=slide_encoder,
                gigapath_checkpoint_sha256=slide_encoder.checkpoint_sha256,
            )
        if name == "architecture4":
            kwargs.update(gene_basis=placeholder_basis, gene_names=gene_names)
        models[name] = build_architecture(
            config, n_genes=len(gene_names),
            gex_feature_dim=int(config["data"]["gex_feature_dim"]), **kwargs,
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging.{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"stale staging path exists: {staging}")
    try:
        init_manifest = persist_four_architecture_initializations(models, staging)
        # Keep human-facing recorded paths correct after the atomic directory rename.
        for architecture_name, entry in init_manifest["architectures"].items():
            if entry.get("weights_path"):
                entry["weights_path"] = str(output_dir / architecture_name / "trainable_weights.pt")
        (staging / "initialization_manifest.json").write_text(
            json.dumps(init_manifest, indent=2, sort_keys=True) + "\n"
        )
        os.rename(staging, output_dir)
    except BaseException:
        if staging.exists():
            import shutil

            shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"verified synchronized initialization bundle written to {output_dir}")
    print("Architecture 4 external basis buffer: excluded (real fitted basis will be preserved)")


if __name__ == "__main__":
    main()
