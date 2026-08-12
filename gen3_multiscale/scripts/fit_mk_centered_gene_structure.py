#!/usr/bin/env python3
"""Fit the shared, training-only gene structure for the MK field screen."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json

import numpy as np

from gen3_multiscale.conditional_wae.structured_field import (
    fit_centered_organ_balanced_gene_structure,
    save_centered_gene_structure_artifact,
)
from gen3_multiscale.data import example_builder
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest


def _select_spot_rows(
    expression, sample_id: str, *, max_spots_per_slide: int, seed: int,
):
    """Return a deterministic, bounded subset without densifying the slide."""
    n_rows = int(expression.shape[0])
    if max_spots_per_slide < 1:
        raise ValueError("max_spots_per_slide must be positive")
    if n_rows <= max_spots_per_slide:
        indices = np.arange(n_rows, dtype=np.int64)
    else:
        digest = hashlib.sha256(f"{int(seed)}\0{sample_id}".encode()).digest()
        sample_seed = int.from_bytes(digest[:8], "little", signed=False)
        rng = np.random.default_rng(sample_seed)
        indices = np.sort(
            rng.choice(n_rows, size=max_spots_per_slide, replace=False)
        ).astype(np.int64, copy=False)
    # Advanced indexing makes an independent bounded object for both dense
    # arrays and scipy sparse matrices, so retaining this result cannot keep
    # the complete AnnData expression matrix alive.
    return expression[indices].copy(), indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--svd-device", default="cpu")
    parser.add_argument(
        "--max-spots-per-slide", type=int, default=512,
        help=(
            "Deterministic uniform training-spot cap per slide (default: 512). "
            "This bounds the pooled SVD matrix while preserving equal-slide/"
            "equal-organ weighting."
        ),
    )
    args = parser.parse_args()

    manifest = load_dataset_manifest(args.manifest)
    train_ids = [str(value) for value in manifest.get("train_sample_ids", [])]
    held_out = set(manifest.get("validation_sample_ids", [])) | set(
        manifest.get("test_sample_ids", [])
    )
    overlap = sorted(set(train_ids) & held_out)
    if overlap:
        raise ValueError(f"manifest split leakage: training ids also held out: {overlap}")
    records = manifest.get("samples") or {}
    organ_by_sample = {}
    expression_by_sample = {}
    source_row_counts = {}
    selected_row_counts = {}
    selection_digest = hashlib.sha256()
    for index, sample_id in enumerate(sorted(train_ids), start=1):
        record = records.get(sample_id) or {}
        if "organ" not in record:
            raise ValueError(f"{sample_id}: manifest record has no organ")
        organ_by_sample[sample_id] = str(record["organ"])
        print(f"loading centered-basis training slide {index}/{len(train_ids)}: {sample_id}", flush=True)
        adata, _patches, _available = example_builder.load_sample_for_examples(
            manifest, sample_id,
        )
        source_row_counts[sample_id] = int(adata.X.shape[0])
        selected, selected_indices = _select_spot_rows(
            adata.X, sample_id,
            max_spots_per_slide=args.max_spots_per_slide,
            seed=args.seed,
        )
        expression_by_sample[sample_id] = selected
        selected_row_counts[sample_id] = int(selected.shape[0])
        selection_digest.update(sample_id.encode())
        selection_digest.update(b"\0")
        selection_digest.update(np.ascontiguousarray(selected_indices).tobytes())
        # Patches are required only while load_sample_for_examples verifies
        # the source/cache contract. Never retain them across training slides.
        del _patches, _available, adata

    artifact = fit_centered_organ_balanced_gene_structure(
        expression_by_sample,
        train_ids,
        organ_by_sample,
        list(manifest["gene_panel"]),
        rank=args.rank,
        seed=args.seed,
        svd_device=args.svd_device,
    )
    metadata = dict(artifact.metadata)
    metadata.update({
        "spot_sampling": "deterministic_uniform_without_replacement_per_slide",
        "max_spots_per_slide": int(args.max_spots_per_slide),
        "source_row_counts": source_row_counts,
        "selected_row_counts": selected_row_counts,
        "spot_selection_sha256": selection_digest.hexdigest(),
    })
    artifact = replace(artifact, metadata=metadata)
    path = save_centered_gene_structure_artifact(artifact, args.output)
    print(f"saved centered MK gene structure to {path}")
    print(json.dumps(artifact.metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
