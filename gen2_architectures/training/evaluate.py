"""Held-out evaluation wrapper shared by every gen2 training script.

Thin glue around the copied audit_evaluation.py harness (fixed mask banks,
predictive-mean/PCC/RMSE/ST-FID/ST-MMD, the notebook-comparable
pcc_raw_log1p metric) — no evaluation LOGIC lives here, only the plumbing
to call it consistently across architectures 1/2/3B/4.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gen2_architectures.data import loaders
from gen2_architectures.data.mask_bank import ensure_mask_bank
from gen2_architectures.evaluation.audit_evaluation import evaluate_model_on_mask_bank


def resolve_fixed_gene_panels(cfg) -> dict[str, list[str]]:
    """Read evaluation.fixed_gene_panel_paths (e.g. HEST-50-style panels)
    from disk. Deliberately only the FIXED-panel half of the original
    codebase's _resolved_evaluation_gene_panels (src/training/train.py) —
    that function's other half (train_variance_gene_panel_sizes, ranking
    genes by variance across TRAINING samples only) needs the full training
    set loaded in memory, which none of gen2's training scripts keep
    resident after the training-sample gene panel is fixed. Fixed panels
    are the ones actually needed for notebook-comparable reporting
    (lung_hest_bench_50-style JSON files); variance-ranked panels can be
    added later if a specific run needs them."""
    panels: dict[str, list[str]] = {}
    for panel_name, raw_path in cfg.get("evaluation", {}).get("fixed_gene_panel_paths", {}).items():
        payload = json.loads(Path(str(raw_path)).read_text())
        genes = payload.get("genes") if isinstance(payload, dict) else payload
        panels[str(panel_name)] = list(dict.fromkeys(str(g) for g in genes))
    return panels


def evaluate_sample(
    model, cfg, adata, images, gene_inputs: dict, sample_id: str, split: str, output_dir: Path,
) -> dict:
    """Evaluate one held-out sample's test (or validation) mask bank.

    split: "test" or "validation" — controls which records ensure_mask_bank
    exposes (split_records inside evaluate_model_on_mask_bank always reads
    the "test" split; validation-time monitoring during training should use
    a SEPARATE, smaller check via the same mechanism if needed — this
    function evaluates the immutable test bank, matching the original
    codebase's own "never train against the actual test split" discipline).
    """
    coords3d = loaders.get_coords_3d(adata)
    evaluation_cfg = cfg.get("evaluation", {})
    mask_bank_dir = Path(evaluation_cfg.get("mask_bank_dir", "results/mask_banks"))
    counts = {
        "validation": int(evaluation_cfg.get("n_validation_masks", 4)),
        "test": int(evaluation_cfg.get("n_test_masks", 8)),
    }
    seeds = {
        "validation": int(evaluation_cfg.get("validation_seed", 700_000)),
        "test": int(evaluation_cfg.get("test_seed", 900_000)),
    }
    bank = ensure_mask_bank(
        mask_bank_dir / f"{sample_id}.json", coords3d, adata.obs["slice_id"].to_numpy(),
        adata.obs_names, cfg.masking, counts, seeds,
    )
    expr = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    organ = str(adata.obs["organ"].iloc[0]) if "organ" in adata.obs else None
    tech = str(adata.obs["tech"].iloc[0]) if "tech" in adata.obs else None
    return evaluate_model_on_mask_bank(
        model, cfg, adata, coords3d, expr, adata.obs["slice_id"].to_numpy(), images, bank,
        gene_inputs, output_path=Path(output_dir) / f"audit_test_metrics_{sample_id}.json",
        organ=organ, tech=tech,
        gene_panels=resolve_fixed_gene_panels(cfg),
        raw_counts=adata.layers["raw_counts"] if "raw_counts" in adata.layers else None,
        raw_library_size=(
            adata.obs["_scilifestdl_raw_library_size"].to_numpy()
            if "_scilifestdl_raw_library_size" in adata.obs else None
        ),
        expression_target_sum=float(cfg.data.get("expression_target_sum", 1e4)),
    )
