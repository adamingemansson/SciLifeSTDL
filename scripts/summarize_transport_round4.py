#!/usr/bin/env python3
"""Summarize the round-4 gene-encoder x candidate-mechanism matrix
(220-235 -- see experimental/PLAN.md). Direct held-out runs (no O0X-style
capacity gates), reuses summarize_transport_suite_v2.py's own
_heldout_row parsing of heldout_sample_summary.json, scoped to
experiment_names that aren't in that script's own hardcoded HELDOUT_RUNS
list. Includes 194/206/215's own rows as reference points since every
config in this round shares one of their evaluation.mask_bank_dir values.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from summarize_transport_suite_v2 import FIELDNAMES, _heldout_row  # type: ignore

RUNS = (
    ("gene_encoder", "transport_c05_gene_mlp_v2_10k"),
    ("gene_encoder", "transport_c05_gene_tokenized_v2_10k"),
    ("retrieval", "transport_c05_retrieval_v2_10k"),
    ("retrieval_combo", "transport_c05_retrieval_geometry_v2_10k"),
    ("retrieval_combo", "transport_c05_retrieval_smallhole_v2_10k"),
    ("retrieval_combo", "transport_c05_retrieval_k256_v2_10k"),
    ("gene_x_retrieval", "transport_c05_gene_mlp_retrieval_v2_10k"),
    ("gene_x_retrieval", "transport_c05_gene_tokenized_retrieval_v2_10k"),
    ("gene_x_global", "transport_c05_gene_mlp_global_v2_10k"),
    ("gene_x_global", "transport_c05_gene_tokenized_global_v2_10k"),
    # 230-235 (BANKSY niche candidate) intentionally NOT listed yet --
    # add once those configs exist and banksy_py is confirmed installed.
    ("reference", "transport_c05_all_modalities_concat_k128_v2_20k"),
    ("reference", "transport_harmonic_k128_v2"),
    ("reference", "transport_c05_all_modalities_concat_k128_v2_20k_smallhole"),
    ("reference", "transport_harmonic_k128_v2_smallhole"),
    ("reference", "transport_c05_all_modalities_concat_k256_v2_20k"),
    ("reference", "transport_harmonic_k256_v2"),
    ("reference", "transport_c05_global_candidate_v2_20k"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", default="results/checkpoints/recovery_suite")
    parser.add_argument("--output", default="reports/recovery_suite/transport_round4.csv")
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
        print(f"  [{row['phase']:>16}] {row['experiment_name']:<55} status={row['status']:<11} "
              f"pcc={row['pcc']}")


if __name__ == "__main__":
    main()
