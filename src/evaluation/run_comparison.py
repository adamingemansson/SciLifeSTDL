"""Compare experiment configs through the immutable audit evaluation path.

Every config is trained independently by :mod:`src.training.train`, which now
uses fixed validation/test mask banks, best-validation checkpoint selection,
predictive means across stochastic samples and explicit image-availability
ablations. This module only orchestrates those runs and prints one compact
comparison table; it deliberately contains no separate masking or Novae path.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf




def _any_config_uses_novae(
    model_config_paths: list[str], overrides: list[str] | None = None
) -> bool:
    """Compatibility helper used by callers that preflight a config set."""
    from src.data.context_features import model_uses_novae

    for path in model_config_paths:
        cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        if model_uses_novae(dict(cfg.model.get("params", {}))):
            return True
    return False


def _fid_n_components(pca_components: int, context_n: int, context_d: int, query_n: int) -> int:
    """Return a covariance-safe PCA width for ST-FID/ST-MMD.

    The PCA basis is fitted on context-derived neighborhoods, while FID/MMD
    covariance is estimated on query-derived embeddings.  Therefore the
    component count must respect *both* sample counts as well as the feature
    width.  The one-component floor is retained for backward-compatible
    diagnostics; callers may still choose to skip FID when fewer than two
    statistically meaningful components are available.
    """
    return max(1, min(
        int(pca_components),
        max(1, int(context_n) - 1),
        max(1, int(context_d)),
        max(1, int(query_n) - 1),
    ))

def _release_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _audit_metrics_for_config(path: str, overrides: list[str] | None, skip_training: bool) -> dict:
    """Run or re-evaluate one single-sample config through the audit path."""
    from src.training.train import (
        main as train_main,
        load_trained_model,
        _load_data,
        prepare_novae_inputs,
        _mask_bank_for_config,
    )
    from src.evaluation.audit_evaluation import evaluate_model_on_mask_bank

    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    if cfg.data.get("train_sample_ids") is not None or cfg.data.get("sample_ids") is not None:
        raise ValueError(
            "run_comparison currently compares single-sample configs. Multi-sample "
            "experiments write heldout_sample_summary.json from train.py."
        )

    checkpoint_dir = Path(cfg.training.get(
        "checkpoint_dir", f"results/checkpoints/{cfg.experiment_name}"
    ))
    metrics_path = checkpoint_dir / "audit_test_metrics.json"

    if not skip_training:
        train_main(path, overrides or [])
    elif not metrics_path.exists():
        model, _gene_names = load_trained_model(str(checkpoint_dir))
        adata, coords3d, expr, slice_ids, images = _load_data(cfg)
        model_params = cfg.model.get("params", {})
        novae_inputs = prepare_novae_inputs(
            cfg, adata, model_params, coords3d, slice_ids,
            sample_id=cfg.data.get("sample_id"),
        )
        bank, _ = _mask_bank_for_config(cfg, adata, coords3d, slice_ids)
        organ = str(adata.obs["organ"].iloc[0]) if "organ" in adata.obs else None
        tech = str(adata.obs["tech"].iloc[0]) if "tech" in adata.obs else None
        evaluate_model_on_mask_bank(
            model, cfg, adata, coords3d, expr, slice_ids, images, bank,
            novae_inputs, metrics_path, organ=organ, tech=tech,
        )

    if not metrics_path.exists():
        raise FileNotFoundError(f"audit metrics were not produced at {metrics_path}")
    return json.loads(metrics_path.read_text())


def main(model_config_paths: list[str], k_neighborhood: int = 8, pca_components: int = 10,
         overrides: list[str] | None = None, skip_training: bool = False,
         shuffle_diagnostic: bool = False):
    """Train/evaluate configs and print mean ± mask-bank variability."""
    overrides = list(overrides or [])
    if not any(x.startswith("evaluation.k_neighborhood=") for x in overrides):
        overrides.append(f"evaluation.k_neighborhood={int(k_neighborhood)}")
    if not any(x.startswith("evaluation.pca_n_components=") for x in overrides):
        overrides.append(f"evaluation.pca_n_components={int(pca_components)}")
    if shuffle_diagnostic and not any(x.startswith("evaluation.image_modes=") for x in overrides):
        overrides.append("evaluation.image_modes=[full,shuffled]")

    rows = []
    for path in model_config_paths:
        result = _audit_metrics_for_config(path, overrides, skip_training)
        full = result["image_modes"].get("full", next(iter(result["image_modes"].values())))
        summary = full["summary"]
        rows.append((
            result["experiment_name"],
            summary["pcc"]["mean"], summary["pcc"]["std"],
            summary["rmse"]["mean"], summary["rmse"]["std"],
            summary["st_fid"]["mean"], summary["st_mmd"]["mean"],
            summary["spatial_domain_plausibility"]["mean"],
            result["effective_pca_components"], result["n_samples_per_mask"],
        ))
        _release_torch_memory()

    header = (
        f"{'experiment':<42}{'PCC':>10}{'PCC sd':>10}{'RMSE':>10}{'RMSE sd':>10}"
        f"{'ST-FID':>11}{'ST-MMD':>11}{'domain':>10}{'PCA d':>8}{'S':>5}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        name, pcc, pcc_sd, rmse, rmse_sd, fid, mmd, domain, pca_d, samples = row
        print(
            f"{name:<42}{pcc:>10.4f}{pcc_sd:>10.4f}{rmse:>10.4f}{rmse_sd:>10.4f}"
            f"{fid:>11.4f}{mmd:>11.4f}{domain:>10.4f}{pca_d:>8d}{samples:>5d}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", type=str)
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--shuffle-diagnostic", action="store_true")
    args = parser.parse_args()
    main(args.configs, overrides=args.override, skip_training=args.skip_training,
         shuffle_diagnostic=args.shuffle_diagnostic)
