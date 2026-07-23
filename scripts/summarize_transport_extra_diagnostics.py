#!/usr/bin/env python3
"""Summarize the smallhole (207-210), local_k sweep (211-214), and
global-candidate (215) diagnostics.

All are direct held-out runs (no O0X-style capacity gates), so this reuses
summarize_transport_suite_v2.py's own _heldout_row parsing of
heldout_sample_summary.json, scoped to just these experiment_names -- none
of them are in that script's own hardcoded HELDOUT_RUNS list, so it can't
see them. The two "reference" rows (194/206) are the original v2 suite's
C05/harmonic runs that 215 shares its evaluation.mask_bank_dir with -- same
held-out test masks, so they're the direct comparison point for whether the
global candidate actually helped.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from summarize_transport_suite_v2 import FIELDNAMES, _heldout_row  # type: ignore

RUNS = (
    ("smallhole", "transport_c01_raw_gex_only_v2_20k_smallhole"),
    ("smallhole", "transport_c07_geometry_only_scoring_v2_20k_smallhole"),
    ("smallhole", "transport_c05_all_modalities_concat_k128_v2_20k_smallhole"),
    ("smallhole", "transport_harmonic_k128_v2_smallhole"),
    ("local_k_sweep", "transport_c05_all_modalities_concat_k256_v2_20k"),
    ("local_k_sweep", "transport_harmonic_k256_v2"),
    ("local_k_sweep", "transport_c05_all_modalities_concat_k512_v2_20k"),
    ("local_k_sweep", "transport_harmonic_k512_v2"),
    ("global_candidate", "transport_c05_global_candidate_v2_20k"),
    ("reference", "transport_c05_all_modalities_concat_k128_v2_20k"),
    ("reference", "transport_harmonic_k128_v2"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", default="results/checkpoints/recovery_suite")
    parser.add_argument("--output", default="reports/recovery_suite/transport_extra_diagnostics.csv")
    args = parser.parse_args()
    checkpoint_root = Path(args.checkpoint_root)

    rows = []
    for axis, run in RUNS:
        row = _heldout_row(checkpoint_root, run)
        row["phase"] = axis
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {output}")
    for row in rows:
        print(f"  [{row['phase']:>13}] {row['experiment_name']:<55} status={row['status']:<11} "
              f"pcc={row['pcc']}")


if __name__ == "__main__":
    main()
