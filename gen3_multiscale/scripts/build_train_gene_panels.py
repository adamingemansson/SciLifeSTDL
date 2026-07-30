#!/usr/bin/env python3
"""Build immutable top-50/top-200 variance panels from training samples only."""
from __future__ import annotations

import argparse

from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.evaluation.train_gene_panels import (
    build_train_derived_gene_panels, save_train_derived_gene_panels,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = load_dataset_manifest(args.manifest)
    artifact = build_train_derived_gene_panels(manifest)
    save_train_derived_gene_panels(artifact, args.output)
    print(
        f"wrote {args.output}: {artifact['n_training_spots']} training spots, "
        + ", ".join(f"{name}={len(genes)}" for name, genes in artifact["panels"].items())
    )


if __name__ == "__main__":
    main()
