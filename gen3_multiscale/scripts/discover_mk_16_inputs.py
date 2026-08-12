#!/usr/bin/env python3
"""Find mutually compatible expanded-cohort inputs for the MK 16-arm screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gen3_multiscale.conditional_wae.structured_field import (
    load_centered_gene_structure_artifact,
)
from gen3_multiscale.evaluation.train_gene_panels import (
    load_train_derived_gene_panels,
)


DEFAULT_ROOTS = (
    "/data/adam.ingemansson/SciLifeSTDL-MK/gen3_multiscale/results",
    "/data/adam.ingemansson/SciLifeSTDL/gen3_multiscale/results",
    "/data/adam.ingemansson/SciLifeSTDL/data/cache/hest1k",
)
DEFAULT_COMPARISON = (
    "/data/adam.ingemansson/SciLifeSTDL/gen3_multiscale/results/"
    "gen3_full_harmonicfix_20260730T150307Z/config_arch1/config.yaml"
)


def _files(roots: list[Path], pattern: str) -> list[Path]:
    found = {
        path.resolve()
        for root in roots if root.is_dir()
        for path in root.rglob(pattern) if path.is_file()
    }
    return sorted(found, key=lambda path: path.stat().st_mtime, reverse=True)


def discover_inputs(
    roots: list[Path], *, validation_samples: int = 14,
    comparison: Path = Path(DEFAULT_COMPARISON),
) -> dict:
    panel_paths = _files(roots, "*gene_panels*.json")
    structure_paths = _files(roots, "*centered_gene_structure*.pt")
    records = []
    for manifest_path in _files(roots, "*manifest*.json"):
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            continue
        if not isinstance(manifest, dict) or not manifest.get("gene_panel"):
            continue
        if len(manifest.get("validation_sample_ids", [])) != validation_samples:
            continue
        panels = []
        for path in panel_paths:
            try:
                load_train_derived_gene_panels(path, manifest)
            except Exception:
                continue
            panels.append(str(path))
        structures = []
        expected_train = sorted(str(value) for value in manifest.get("train_sample_ids", []))
        for path in structure_paths:
            try:
                artifact = load_centered_gene_structure_artifact(
                    path, [str(value) for value in manifest["gene_panel"]],
                )
            except Exception:
                continue
            if artifact.metadata["train_sample_ids"] == expected_train:
                structures.append(str(path))
        records.append({
            "manifest": str(manifest_path),
            "n_train_samples": len(manifest.get("train_sample_ids", [])),
            "n_validation_samples": len(manifest.get("validation_sample_ids", [])),
            "n_genes": len(manifest["gene_panel"]),
            "compatible_panels": panels,
            "compatible_centered_gene_structures": structures,
        })
    return {
        "comparison": str(comparison.resolve()),
        "comparison_exists": comparison.is_file(),
        "validation_samples_required": validation_samples,
        "candidates": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", dest="roots")
    parser.add_argument("--validation-samples", type=int, default=14)
    parser.add_argument("--comparison", default=DEFAULT_COMPARISON)
    args = parser.parse_args()
    roots = [Path(value).expanduser().resolve() for value in (args.roots or DEFAULT_ROOTS)]
    print(json.dumps(discover_inputs(
        roots, validation_samples=args.validation_samples,
        comparison=Path(args.comparison).expanduser(),
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
