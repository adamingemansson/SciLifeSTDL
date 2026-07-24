#!/usr/bin/env python3
"""Pull PCC/RMSE (full panel + lung_hest_bench_50) for the lung_round gigapathfix suite."""
import json
from pathlib import Path

CHECKPOINT_ROOT = Path("results/checkpoints/lung_round")

EXPERIMENTS = [
    "transport_lung_stpath_scratch_v1_10k_holefix_gigapathfix",
    "transport_lung_simple_fusion_v1_10k_holefix_gigapathfix",
    "transport_lung_simple_cross_attn_v1_10k_holefix_gigapathfix",
    "transport_lung_harmonic_v1_holefix",
    "transport_lung_stpath_pretrained_eval_v1_holefix_gigapathfix",
    "transport_lung_simple_stpath_transformer_v2_10k_holefix_gigapathfix",
    "transport_lung_simple_cross_attn_decoder_v1_10k_holefix_gigapathfix",
    "transport_lung_cross_attn_universal_gene_v1_10k_holefix_gigapathfix",
    "transport_lung_spatial_transformer_universal_gene_v1_10k_holefix_gigapathfix",
    "transport_lung_spatial_transformer_stpath_gene_v1_10k_holefix_gigapathfix",
    "transport_lung_cross_attn_stpath_gene_v1_10k_holefix_gigapathfix",
]

PANEL = "lung_hest_bench_50"


def main() -> None:
    rows = []
    for name in EXPERIMENTS:
        path = CHECKPOINT_ROOT / name / "heldout_sample_summary.json"
        if not path.is_file():
            rows.append((name, None))
            continue
        summary = json.loads(path.read_text())
        primary_mode = summary["primary_image_mode"]
        primary = summary["image_modes"][primary_mode]
        rows.append((name, primary))

    header = (
        f"{'experiment':<65} {'full_pcc':>9} {'full_rmse':>10} "
        f"{'hest50_pcc':>11} {'hest50_rmse':>12}"
    )
    print(header)
    print("-" * len(header))
    for name, primary in rows:
        if primary is None:
            print(f"{name:<65} {'MISSING':>9}")
            continue
        full_pcc = primary.get("pcc_mean", float("nan"))
        full_rmse = primary.get("rmse_mean", float("nan"))
        hest_pcc = primary.get(f"pcc_{PANEL}_mean", float("nan"))
        hest_rmse = primary.get(f"rmse_{PANEL}_mean", float("nan"))
        print(
            f"{name:<65} {full_pcc:>9.4f} {full_rmse:>10.4f} "
            f"{hest_pcc:>11.4f} {hest_rmse:>12.4f}"
        )


if __name__ == "__main__":
    main()
