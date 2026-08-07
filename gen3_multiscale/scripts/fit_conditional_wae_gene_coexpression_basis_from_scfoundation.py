#!/usr/bin/env python3
"""Fit the MK conditional-WAE gene-coexpression basis from scFoundation's
pretrained per-gene embeddings, instead of a from-scratch SVD fit on this
project's own training expression (see
scripts/fit_conditional_wae_gene_coexpression_basis.py, the from-scratch
sibling of this script).

Loads a real, checkpoint-verified `FrozenSCFoundationEncoder` (the SAME
class Gen4/Gen5 already use, same fail-closed checkpoint/repo/vocabulary
discipline) purely to read its `pos_emb` gene-embedding table -- no
forward pass over any real expression/patch data is performed, so this
never touches per-sample H&E patches or GEX at all.

    python -m gen3_multiscale.scripts.fit_conditional_wae_gene_coexpression_basis_from_scfoundation \\
        --manifest /path/to/dataset_manifest.json \\
        --output-basis-path /path/to/gene_coexpression_basis_scfoundation.pt \\
        --scfoundation-checkpoint /path/to/models.ckpt \\
        --scfoundation-vocab /path/to/OS_scRNA_gene_index.19264.tsv \\
        --scfoundation-repo /path/to/scFoundation \\
        --scfoundation-revision <40-hex-commit> \\
        --rank 64 --svd-device cuda:0
"""
from __future__ import annotations

import argparse
import json

from gen3_multiscale.conditional_wae.coexpression import save_conditional_wae_gene_coexpression_basis
from gen3_multiscale.conditional_wae.coexpression_scfoundation import (
    _REQUIRED_METADATA_FIELDS_SCFOUNDATION,
    extract_scfoundation_gene_embedding_table,
    fit_conditional_wae_gene_coexpression_basis_from_scfoundation,
)
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.gen4.scfoundation_encoder import FrozenSCFoundationEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="A real, already-built Gen3 dataset manifest")
    parser.add_argument("--output-basis-path", required=True)
    parser.add_argument("--scfoundation-checkpoint", required=True)
    parser.add_argument("--scfoundation-vocab", required=True)
    parser.add_argument("--scfoundation-repo", required=True)
    parser.add_argument("--scfoundation-revision", required=True)
    parser.add_argument("--device", default="cuda", help="Device to load the scFoundation checkpoint onto")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--svd-device", default="cpu",
        help="'cpu' (default) or a CUDA device ('cuda' / 'cuda:N') for the basis-fitting SVD step "
             "itself (independent of --device, which only places the scFoundation checkpoint).",
    )
    args = parser.parse_args()

    manifest = load_dataset_manifest(args.manifest)
    gene_names = list(manifest["gene_panel"])

    encoder = FrozenSCFoundationEncoder(
        args.scfoundation_checkpoint, args.scfoundation_vocab, gene_names,
        repo_path=args.scfoundation_repo, repo_revision=args.scfoundation_revision, device=args.device,
    )
    embedding_table, report = extract_scfoundation_gene_embedding_table(encoder)
    print(
        f"scFoundation gene coverage: {report['n_found']}/{report['n_genes']} panel genes found in "
        f"scFoundation's vocabulary (embedding_dim={report['embedding_dim']})", flush=True,
    )

    basis, metadata = fit_conditional_wae_gene_coexpression_basis_from_scfoundation(
        embedding_table, gene_names, encoder.identity, report,
        rank=args.rank, seed=args.seed, svd_device=args.svd_device,
    )
    path = save_conditional_wae_gene_coexpression_basis(
        basis, metadata, args.output_basis_path, required_fields=_REQUIRED_METADATA_FIELDS_SCFOUNDATION,
    )
    printable_metadata = {key: value for key, value in metadata.items() if key != "missing_genes"}
    printable_metadata["n_missing_genes_omitted_from_this_summary"] = len(metadata["missing_genes"])
    print(f"scFoundation gene coexpression basis fit and saved to {path}: {json.dumps(printable_metadata, indent=2)}")


if __name__ == "__main__":
    main()
