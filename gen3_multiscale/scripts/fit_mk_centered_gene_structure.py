#!/usr/bin/env python3
"""Fit the shared, training-only gene structure for the MK field screen."""
from __future__ import annotations

import argparse
import json

from gen3_multiscale.conditional_wae.structured_field import (
    fit_centered_organ_balanced_gene_structure,
    save_centered_gene_structure_artifact,
)
from gen3_multiscale.data import example_builder
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--svd-device", default="cpu")
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
    for index, sample_id in enumerate(sorted(train_ids), start=1):
        record = records.get(sample_id) or {}
        if "organ" not in record:
            raise ValueError(f"{sample_id}: manifest record has no organ")
        organ_by_sample[sample_id] = str(record["organ"])
        print(f"loading centered-basis training slide {index}/{len(train_ids)}: {sample_id}", flush=True)
        adata, _patches, _available = example_builder.load_sample_for_examples(
            manifest, sample_id,
        )
        expression_by_sample[sample_id] = adata.X

    artifact = fit_centered_organ_balanced_gene_structure(
        expression_by_sample,
        train_ids,
        organ_by_sample,
        list(manifest["gene_panel"]),
        rank=args.rank,
        seed=args.seed,
        svd_device=args.svd_device,
    )
    path = save_centered_gene_structure_artifact(artifact, args.output)
    print(f"saved centered MK gene structure to {path}")
    print(json.dumps(artifact.metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
