"""Inventory HEST-1k Visium coverage: what's already downloaded locally vs.
what's available in the full catalog, broken down by organ.

Answers the concrete question behind the gen2_architectures sample-scope
decision (README section 11 / 9): once we commit to Visium-only for this
round (see README's own reasoning -- mixing technologies collapses the
shared gene panel to whatever the narrowest targeted panel covers), which
organs do we already have real data for, and which organs would need a
fresh download to reach genuine multi-organ coverage?

Run this ON THE SERVER, where data/raw/hest1k/ actually lives and
huggingface-cli is already authenticated (see docs/dataset_notes.md's own
download instructions -- this script reads the same public metadata CSV
that section already uses).

Usage:
    python3 gen2_architectures/scripts/inventory_hest1k.py \
        --hest-data-dir data/raw/hest1k

Prints:
  1. What's downloaded locally right now (organ x sample-count, Visium only,
     cross-checked against BOTH expression (.h5ad under st/) and image
     patches (.h5 under patches/) actually being present -- a sample
     missing either half can't be used by any gen2 architecture that
     touches images, which is all of them).
  2. The full HEST-1k Visium catalog's organ breakdown, so you can see
     what's available to expand into.
  3. Organs with real local coverage vs. organs with zero local coverage.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gen2_architectures.data.hest1k_catalog import load_visium_metadata, locally_downloaded_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hest-data-dir", type=str, default="data/raw/hest1k")
    parser.add_argument(
        "--metadata-csv", type=str, default="hf://datasets/MahmoodLab/hest/HEST_v1_3_0.csv",
        help="Public HEST-1k metadata CSV (docs/dataset_notes.md's own source). "
             "Pass a local path instead if you've already downloaded it.",
    )
    parser.add_argument(
        "--species", type=str, default="Homo sapiens",
        help="Filter to this species (real HEST-1k values confirmed 2026-07-25: "
             "'Homo sapiens' (421 Visium samples), 'Mus musculus' (181) -- matches "
             "hest1k_catalog.py::resolve_sample_selection's own default and the real "
             "zero-gene-intersection bug that default guards against). Pass 'all' to "
             "keep every species (e.g. to inspect mouse coverage separately).",
    )
    args = parser.parse_args()
    species = None if args.species == "all" else args.species

    hest_data_dir = Path(args.hest_data_dir)
    print(f"Reading HEST-1k metadata from {args.metadata_csv} (species={args.species}) ...")
    visium = load_visium_metadata(args.metadata_csv, species=species)
    print(f"Full catalog Visium samples: {len(visium)} across {visium['organ'].nunique()} organs.\n")

    st_ids, patch_ids = locally_downloaded_ids(hest_data_dir)
    usable_ids = st_ids & patch_ids  # has BOTH expression and image patches
    expr_only_ids = st_ids - patch_ids
    patch_only_ids = patch_ids - st_ids

    local_visium = visium[visium["id"].isin(usable_ids)]
    local_by_organ = Counter(local_visium["organ"])
    catalog_by_organ = Counter(visium["organ"])

    print("=" * 78)
    print("LOCALLY DOWNLOADED (expression + patches both present -> actually usable)")
    print("=" * 78)
    if not local_by_organ:
        print("  (nothing found -- check --hest-data-dir)")
    for organ, count in sorted(local_by_organ.items(), key=lambda kv: -kv[1]):
        sample_ids = sorted(local_visium[local_visium["organ"] == organ]["id"])
        print(f"  {organ:<25} {count:>3} samples   {', '.join(sample_ids)}")

    if expr_only_ids:
        print(f"\n  WARNING: {len(expr_only_ids)} sample(s) have expression data but no "
              f"patches (unusable by any gen2 architecture, all of which need images): "
              f"{sorted(expr_only_ids)}")
    if patch_only_ids:
        print(f"\n  WARNING: {len(patch_only_ids)} sample(s) have patches but no expression "
              f"data: {sorted(patch_only_ids)}")

    print()
    print("=" * 78)
    print("FULL HEST-1k VISIUM CATALOG (organ -> sample count, for planning what to add)")
    print("=" * 78)
    for organ, count in sorted(catalog_by_organ.items(), key=lambda kv: -kv[1]):
        have = local_by_organ.get(organ, 0)
        flag = "" if have else "  <-- ZERO local coverage"
        print(f"  {organ:<25} {count:>3} in catalog, {have:>3} downloaded{flag}")

    print()
    zero_coverage = [o for o in catalog_by_organ if o not in local_by_organ]
    print(f"Organs with real Visium data available but ZERO local coverage: {len(zero_coverage)}")
    if zero_coverage:
        print(f"  {sorted(zero_coverage)}")
    print()
    print("To download a specific organ's samples, see docs/dataset_notes.md's "
          "'How to actually download a HEST-1k sample' section -- filter meta_df by "
          "organ + technology=='Visium', then snapshot_download with allow_patterns "
          "built from the resulting ids.")


if __name__ == "__main__":
    main()
