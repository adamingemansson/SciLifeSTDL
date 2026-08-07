#!/usr/bin/env python3
"""Fit and save the MK conditional-WAE gene-coexpression basis.

Pure gene-expression structure -- pools real, normalized full-gene
expression (`example_builder.load_sample_for_examples`) across every
TRAINING-split sample the given dataset manifest declares (never
validation/test), and fits a low-rank orthonormal basis via
`models.gene_basis.fit_gene_residual_basis` capturing which genes
co-vary together. Deliberately has NO dependency on any image
tile-encoder cache (GigaPath/UNI2) existing -- this artifact is fit
before, and independently of, any conditional-WAE training run.

    python -m gen3_multiscale.scripts.fit_conditional_wae_gene_coexpression_basis \\
        --manifest /path/to/dataset_manifest.json \\
        --output-basis-path /path/to/gene_coexpression_basis.pt \\
        --rank 64 --svd-device cuda:0

At real production scale (tens of thousands of pooled training spots x
a ~17k-gene panel), the CPU-only randomized SVD path is prohibitively
slow -- pass --svd-device cuda:N to use the GPU path this codebase
already relies on for the equivalent Architecture 4 residual-basis fit
at this exact matrix scale.
"""
from __future__ import annotations

import argparse
import json

from gen3_multiscale.conditional_wae.coexpression import (
    fit_conditional_wae_gene_coexpression_basis, save_conditional_wae_gene_coexpression_basis,
)
from gen3_multiscale.data import example_builder
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="A real, already-built Gen3 dataset manifest")
    parser.add_argument("--output-basis-path", required=True)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--svd-device", default="cpu",
        help="'cpu' (default) or a CUDA device ('cuda' / 'cuda:N'). At real production scale "
             "(tens of thousands of pooled training spots x a ~17k-gene panel), the CPU path is "
             "prohibitively slow -- pass a CUDA device to use models.gene_basis.fit_gene_residual_"
             "basis's GPU path instead.",
    )
    args = parser.parse_args()

    manifest = load_dataset_manifest(args.manifest)
    train_sample_ids = list(manifest["train_sample_ids"])
    if not train_sample_ids:
        raise ValueError("dataset manifest has zero train_sample_ids -- nothing to fit a coexpression basis on")
    gene_names = list(manifest["gene_panel"])

    expression_by_sample = {}
    for index, sample_id in enumerate(sorted(train_sample_ids), start=1):
        print(f"loading training expression {index}/{len(train_sample_ids)}: {sample_id}", flush=True)
        adata, _patches, _image_source_available = example_builder.load_sample_for_examples(manifest, sample_id)
        expression_by_sample[sample_id] = adata.X

    basis, metadata = fit_conditional_wae_gene_coexpression_basis(
        expression_by_sample, train_sample_ids, gene_names, rank=args.rank, seed=args.seed,
        svd_device=args.svd_device,
    )
    path = save_conditional_wae_gene_coexpression_basis(basis, metadata, args.output_basis_path)
    print(f"gene coexpression basis fit and saved to {path}: {json.dumps(metadata, indent=2)}")


if __name__ == "__main__":
    main()
