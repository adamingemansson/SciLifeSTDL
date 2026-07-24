"""Fast, no-retrain diagnostic: does the notebook-vs-ours PCC gap shrink once
STPath's own evaluation convention (raw-count log1p, no library-size
normalization) is used instead of our library-size-normalized-log1p space?

Reloads an ALREADY-TRAINED checkpoint (no training happens here) and re-runs
the audit evaluation with the new raw_counts/raw_library_size arguments
wired into evaluate_model_on_mask_bank (2026-07-24). Writes to a fresh
output path per sample so it never touches/resumes the checkpoint's real
audit_test_metrics_{sid}.json. Prints our normalized-space PCC next to the
notebook-comparable raw-log1p PCC, per test sample and averaged, for both
the full gene panel and any fixed evaluation panels (e.g. lung_hest_bench_50)
declared in the config.

Usage:
    python3 scripts/lung_stpath_raw_log1p_check.py configs/lung_round/305_lung_stpath_pretrained_eval.yaml
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.mask_bank import ensure_mask_bank
from src.evaluation.audit_evaluation import evaluate_model_on_mask_bank
from src.training.train import (
    _validated_sample_groups,
    load_multi_sample_data,
    load_trained_model,
)


def _fixed_gene_panels(cfg) -> dict[str, list[str]]:
    panels: dict[str, list[str]] = {}
    for panel_name, raw_path in cfg.get("evaluation", {}).get("fixed_gene_panel_paths", {}).items():
        payload = json.loads(Path(str(raw_path)).read_text())
        genes = payload.get("genes") if isinstance(payload, dict) else payload
        panels[str(panel_name)] = list(dict.fromkeys(str(g) for g in genes))
    return panels


def main(config_path: str) -> None:
    cfg = OmegaConf.load(config_path)
    checkpoint_dir = Path(cfg.training.get("checkpoint_dir", f"results/checkpoints/{cfg.experiment_name}"))
    print(f"loading trained checkpoint from {checkpoint_dir} (no training)")
    model, gene_names = load_trained_model(str(checkpoint_dir))

    _train_ids, _validation_ids, test_ids = _validated_sample_groups(cfg)
    if not test_ids:
        raise ValueError(f"{config_path} has no data.test_sample_ids to evaluate on")

    test_samples, test_adatas = load_multi_sample_data(cfg, sample_ids=test_ids, reference_gene_names=gene_names)
    gene_panels = _fixed_gene_panels(cfg)

    evaluation_cfg = cfg.get("evaluation", {})
    mask_bank_dir = Path(evaluation_cfg.get("mask_bank_dir", "results/mask_banks"))
    out_dir = checkpoint_dir / "raw_log1p_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sid, sample, adata in zip(test_ids, test_samples, test_adatas):
        coords3d, expr, slice_ids, images, organ, tech = sample[:6]
        counts = {
            "validation": int(evaluation_cfg.get("n_validation_masks", 4)),
            "test": int(evaluation_cfg.get("n_test_masks", 8)),
        }
        seeds = {
            "validation": int(evaluation_cfg.get("validation_seed", 700_000)),
            "test": int(evaluation_cfg.get("test_seed", 900_000)),
        }
        bank, _bank_path = ensure_mask_bank(
            mask_bank_dir / f"{sid}.json", coords3d, slice_ids, adata.obs_names, cfg.masking, counts, seeds
        )
        novae_dict = {
            "context_gene_features": sample[6],
            "context_novae_features": sample[7],
            "context_gene_feature_provider": sample[8],
            "context_novae_feature_provider": sample[9],
            "context_niche_feature_provider": sample[11],
        }
        print(f"evaluating {sid} ({coords3d.shape[0]} spots)...")
        result = evaluate_model_on_mask_bank(
            model, cfg, adata, coords3d, expr, slice_ids, images, bank,
            novae_dict,
            output_path=out_dir / f"audit_test_metrics_{sid}.json",
            organ=organ, tech=tech,
            slide_context=sample[10],
            gene_panels=gene_panels,
            raw_counts=adata.layers["raw_counts"] if "raw_counts" in adata.layers else None,
            raw_library_size=(
                adata.obs["_scilifestdl_raw_library_size"].to_numpy()
                if "_scilifestdl_raw_library_size" in adata.obs else None
            ),
            expression_target_sum=float(cfg.data.get("expression_target_sum", 1e4)),
        )
        primary_mode = result["primary_image_mode"]
        summary = result["image_modes"][primary_mode]["summary"]
        row = {"sample_id": str(sid)}
        for metric in ("pcc", "pcc_raw_log1p", *(f"pcc_{p}" for p in gene_panels)):
            value = summary.get(metric, {})
            row[metric] = value.get("mean") if isinstance(value, dict) else None
        rows.append(row)

    if not rows:
        print("no test samples evaluated")
        return

    header = f"{'sample_id':<12}" + "".join(f"{k:>18}" for k in rows[0] if k != "sample_id")
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        line = f"{row['sample_id']:<12}"
        for k, v in row.items():
            if k == "sample_id":
                continue
            line += f"{v:>18.4f}" if v is not None else f"{'n/a':>18}"
        print(line)

    print("\nmeans across test samples:")
    for k in rows[0]:
        if k == "sample_id":
            continue
        values = [row[k] for row in rows if row[k] is not None]
        mean = float(np.mean(values)) if values else float("nan")
        print(f"  {k}: {mean:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 scripts/lung_stpath_raw_log1p_check.py <config.yaml>")
    main(sys.argv[1])
