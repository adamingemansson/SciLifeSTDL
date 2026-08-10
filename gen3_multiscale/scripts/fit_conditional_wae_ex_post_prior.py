#!/usr/bin/env python3
"""Fit an ex-post inference-time sampling density for a trained conditional WAE.

Standard, no-retrain fix for the WAE "prior hole" problem (Tolstikhin et al.,
ICLR 2018; Ghosh et al., "From Variational to Deterministic Autoencoders",
ICLR 2020): a WAE-MMD/WAE-GAN regularizer only matches the AGGREGATE encoded
posterior to the prior, not each individual sample, so the true encoded
distribution of real training targets can be narrower (or wider, or shifted)
than N(0,I) even after training converges. Sampling raw z~N(0,I) at inference
-- which is what sample_predictive_distribution has always done -- then draws
from the wrong distribution relative to what the decoder actually learned to
reconstruct from.

This script encodes every real training-target realization through the
model's OWN frozen expression encoder (encode_posterior; never used for loss
or reported metrics, exists precisely for offline diagnostics like this one)
and fits a single diagonal Gaussian (per-latent-dim mean + std) to the result.
Only the training split is used -- fitting on validation/test targets would
leak held-out labels into a component of the inference procedure, which this
project's leakage discipline forbids.

The fitted z_mean/z_std can then be passed to
evaluation/conditional_wae_evaluator.py via --ex-post-prior to resample
inference-time z from the fitted density instead of the raw prior, with NO
retraining and NO change to the checkpoint itself.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from gen3_multiscale.conditional_wae import ConditionalWAEMaskedGEXDataset
from gen3_multiscale.conditional_wae.contract import static_audit_conditional_wae_config
from gen3_multiscale.config_identity import resolved_config
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.training import checkpoint as checkpoint_module
from gen3_multiscale.training.gen3_dataset import Gen3SpatialFieldDataset, build_gen3_mask_schedule
from gen3_multiscale.training.gen3_preflight import load_and_preflight_samples
from gen3_multiscale.training.train import expected_tile_encoder_provenance
from gen3_multiscale.training.train_conditional_wae import _build_model, _manifest, _verify_resume


def fit_conditional_wae_ex_post_prior(
    config_path: str,
    checkpoint_dir: str,
    output: str,
    *,
    n_masks_per_sample: int = 16,
    device_name: str = "cuda",
    use_best: bool = True,
    allow_code_drift: bool = False,
) -> dict:
    if n_masks_per_sample < 1:
        raise ValueError("n_masks_per_sample must be positive")
    config = resolved_config(config_path)
    static_audit_conditional_wae_config(config)
    dataset_manifest = load_dataset_manifest(config["data"]["gen3_manifest_path"])
    train_ids = list(dataset_manifest["train_sample_ids"])
    cfg_om = OmegaConf.create(config)
    samples, preflight_report = load_and_preflight_samples(
        cfg_om,
        dataset_manifest,
        train_ids,
        expected_tile_encoder_provenance(config),
    )
    schedule = build_gen3_mask_schedule(
        dataset_manifest,
        samples,
        config["masking"]["strata"],
        "train",
        n_training_masks_per_sample=int(n_masks_per_sample),
    )
    base_dataset = Gen3SpatialFieldDataset(
        dataset_manifest, samples, schedule, config["masking"]["strata"],
        novae_enabled=False,
    )
    dataset = ConditionalWAEMaskedGEXDataset(
        base_dataset,
        include_observed_gex=bool(config["model"]["include_observed_gex"]),
    )
    requested_checkpoint = Path(checkpoint_dir) / "best" if use_best else Path(checkpoint_dir)
    old_manifest = checkpoint_module.load_checkpoint_run_manifest(requested_checkpoint)
    if old_manifest is None:
        raise ValueError(f"{requested_checkpoint}: no bundle-bound run manifest")
    current_manifest = _manifest(config, dataset_manifest, preflight_report)
    old_cache = old_manifest.get("cache_content_by_sample") or {}
    for sample_id, identity in preflight_report["cache_content_by_sample"].items():
        if sample_id in old_cache and old_cache[sample_id] != identity:
            raise ValueError(
                f"{sample_id}: fitting cache content differs from the checkpoint's training cache"
            )
    current_manifest["cache_content_fingerprint"] = old_manifest["cache_content_fingerprint"]
    current_manifest["cache_content_by_sample"] = old_cache
    _verify_resume(old_manifest, current_manifest, allow_code_drift=allow_code_drift)
    gene_names = list(dataset_manifest["gene_panel"])
    checkpoint_module.verify_gene_names(requested_checkpoint, gene_names)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model = _build_model(config, len(gene_names), gene_names=gene_names).to(device)
    checkpoint_module.load_trainable_state(model, requested_checkpoint)
    model.eval()

    latent_sum = torch.zeros(model.latent_dim, dtype=torch.float64, device=device)
    latent_sum_squared = torch.zeros(model.latent_dim, dtype=torch.float64, device=device)
    n_rows = 0
    with torch.no_grad():
        for index in range(len(dataset)):
            inputs, target, identity = dataset[index]
            target_tensor = torch.as_tensor(np.asarray(target, dtype=np.float32), device=device)
            z = model.encode_posterior(target_tensor, inputs).to(torch.float64)
            latent_sum += z.sum(dim=0)
            latent_sum_squared += (z * z).sum(dim=0)
            n_rows += z.shape[0]
            print(
                f"ex-post prior fitting progress: {index + 1}/{len(dataset)} "
                f"sample={identity['sample_id']} stratum={identity['stratum']} rows_so_far={n_rows}",
                flush=True,
            )
    if n_rows == 0:
        raise ValueError("no training rows were encoded -- empty train split or mask schedule")
    mean = (latent_sum / n_rows).cpu().numpy()
    variance = (latent_sum_squared / n_rows).cpu().numpy() - mean * mean
    std = np.sqrt(np.clip(variance, 0.0, None))

    report = {
        "version": 1,
        "kind": "conditional_wae_ex_post_prior",
        "config_path": str(config_path),
        "checkpoint_dir": str(requested_checkpoint),
        "n_items": len(dataset),
        "n_rows": int(n_rows),
        "latent_dim": int(model.latent_dim),
        "z_mean": mean.tolist(),
        "z_std": std.tolist(),
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    os.replace(temporary, output_path)
    print(f"ex-post prior saved to {output_path}", flush=True)
    print(
        f"z_std across {model.latent_dim} latent dims: "
        f"min={std.min():.4f} mean={std.mean():.4f} max={std.max():.4f} "
        f"(1.0 = matches the N(0,I) prior with no collapse)",
        flush=True,
    )
    print(
        f"||z_mean||_2={float(np.linalg.norm(mean)):.4f} "
        f"(0.0 = aggregate posterior centered on the prior's mean)",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-masks-per-sample", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-use-best", action="store_true")
    parser.add_argument("--allow-code-drift", action="store_true")
    args = parser.parse_args()
    fit_conditional_wae_ex_post_prior(
        args.config,
        args.checkpoint_dir,
        args.output,
        n_masks_per_sample=args.n_masks_per_sample,
        device_name=args.device,
        use_best=not args.no_use_best,
        allow_code_drift=args.allow_code_drift,
    )


if __name__ == "__main__":
    main()
