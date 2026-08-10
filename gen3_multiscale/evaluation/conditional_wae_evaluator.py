#!/usr/bin/env python3
"""Held-out evaluator for conditional WAE supervisor tasks."""
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
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.evaluation.gen3_evaluator import (
    _GaussianCalibrationAccumulator,
    load_configured_gene_panels,
    per_item_reconstruction_metrics,
)
from gen3_multiscale.evaluation.metrics import (
    aggregate_patient_metrics,
    gene_panel_metrics,
    resolve_gene_panels,
)
from gen3_multiscale.training import checkpoint as checkpoint_module
from gen3_multiscale.training.gen3_dataset import Gen3SpatialFieldDataset, build_gen3_mask_schedule
from gen3_multiscale.training.gen3_preflight import load_and_preflight_samples
from gen3_multiscale.training.train import expected_tile_encoder_provenance
from gen3_multiscale.training.train_conditional_wae import (
    _build_model,
    _manifest,
    _stable_seed,
    _verify_resume,
)
from gen3_multiscale.config_identity import resolved_config


def evaluate_conditional_wae(
    config_path: str,
    checkpoint_dir: str,
    output: str,
    *,
    split: str = "validation",
    n_masks_per_sample: int = 8,
    device_name: str = "cuda",
    use_best: bool = True,
    allow_code_drift: bool = False,
    n_samples: int | None = None,
    ex_post_prior_path: str | None = None,
) -> dict:
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    if n_masks_per_sample < 1:
        raise ValueError("n_masks_per_sample must be positive")
    ex_post_prior = None
    if ex_post_prior_path is not None:
        ex_post_prior = json.loads(Path(ex_post_prior_path).read_text())
    config = resolved_config(config_path)
    static_audit_conditional_wae_config(config)
    dataset_manifest = load_dataset_manifest(config["data"]["gen3_manifest_path"])
    split_ids = list(dataset_manifest[f"{split}_sample_ids"])
    cfg_om = OmegaConf.create(config)
    samples, preflight_report = load_and_preflight_samples(
        cfg_om,
        dataset_manifest,
        split_ids,
        expected_tile_encoder_provenance(config),
    )
    schedule = build_gen3_mask_schedule(
        dataset_manifest,
        samples,
        config["masking"]["strata"],
        split,
        split_counts={split: int(n_masks_per_sample)},
        split_seeds={split: 700_000 if split == "validation" else 900_000},
    )
    base_dataset = Gen3SpatialFieldDataset(
        dataset_manifest, samples, schedule, config["masking"]["strata"],
        novae_enabled=False,
    )
    base_dataset.validate_boundary_schedule()
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
                f"{sample_id}: evaluation cache content differs from the checkpoint's training cache"
            )
    # A held-out test sample was not part of training cache preflight. Identity
    # fields shared with training remain exact; the split-local cache mapping is
    # checked by the loader itself and recorded in this report.
    current_manifest["cache_content_fingerprint"] = old_manifest["cache_content_fingerprint"]
    current_manifest["cache_content_by_sample"] = old_cache
    _verify_resume(old_manifest, current_manifest, allow_code_drift=allow_code_drift)
    gene_names = list(dataset_manifest["gene_panel"])
    checkpoint_module.verify_gene_names(requested_checkpoint, gene_names)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model = _build_model(config, len(gene_names), gene_names=gene_names).to(device)
    checkpoint_module.load_trainable_state(model, requested_checkpoint)
    model.eval()
    ex_post_z_mean = ex_post_z_std = None
    if ex_post_prior is not None:
        if int(ex_post_prior["latent_dim"]) != model.latent_dim:
            raise ValueError(
                f"ex-post prior latent_dim={ex_post_prior['latent_dim']} != model latent_dim={model.latent_dim}"
            )
        ex_post_z_mean = torch.tensor(ex_post_prior["z_mean"], dtype=torch.float32, device=device)
        ex_post_z_std = torch.tensor(ex_post_prior["z_std"], dtype=torch.float32, device=device)

    panels = load_configured_gene_panels(config, dataset_manifest)
    _indices, panel_metadata = resolve_gene_panels(gene_names, panels) if panels else ({}, {})
    arm_names = ("model", "conditional_mean")
    items = {arm: [] for arm in arm_names}
    panel_items = {arm: {panel: [] for panel in panels} for arm in arm_names}
    patient_ids = []
    records = []
    calibration = _GaussianCalibrationAccumulator()
    inference_samples = int(n_samples or config["model"]["params"]["n_inference_samples"])
    evaluation_seed = int(config["training"].get("seed", 0))
    with torch.no_grad():
        for index in range(len(dataset)):
            inputs, target, identity = dataset[index]
            patient_id = str(samples[identity["sample_id"]].patient_id)
            patient_ids.append(patient_id)
            generator = torch.Generator(device=device).manual_seed(
                _stable_seed(evaluation_seed, identity),
            )
            prediction = model.sample_predictive_distribution(
                inputs, n_samples=inference_samples, generator=generator,
                z_mean=ex_post_z_mean, z_std=ex_post_z_std,
            )
            true = np.asarray(target, dtype=np.float32)
            model_pred = prediction["predictive_mean"].detach().cpu().numpy().astype(np.float32)
            mean_pred = prediction["conditional_mean_expression"].detach().cpu().numpy().astype(np.float32)
            model_metrics = per_item_reconstruction_metrics(model_pred, true)
            mean_metrics = per_item_reconstruction_metrics(mean_pred, true)
            items["model"].append(model_metrics)
            items["conditional_mean"].append(mean_metrics)
            record = {
                **identity,
                "patient_id": patient_id,
                "model": model_metrics,
                "conditional_mean": mean_metrics,
            }
            if panels:
                record["gene_panels"] = {}
                for arm, prediction_array in (("model", model_pred), ("conditional_mean", mean_pred)):
                    metrics = gene_panel_metrics(prediction_array, true, gene_names, panels)
                    record["gene_panels"][arm] = metrics
                    for panel, panel_metric in metrics.items():
                        panel_items[arm][panel].append(panel_metric)
            std = prediction["predictive_std"].detach().cpu().numpy().astype(np.float64)
            safe_std = np.where(std > 1e-8, std, np.nan)
            calibration.add_item((true.astype(np.float64) - model_pred) / safe_std)
            records.append(record)
            print(
                f"conditional WAE evaluation progress: {index + 1}/{len(dataset)} "
                f"sample={identity['sample_id']} stratum={identity['stratum']}",
                flush=True,
            )
    aggregated = {
        arm: aggregate_patient_metrics(arm_items, patient_ids)
        for arm, arm_items in items.items()
    }
    panel_aggregated = {
        panel: {
            arm: aggregate_patient_metrics(panel_items[arm][panel], patient_ids)
            for arm in arm_names
        }
        for panel in panels
    }
    evaluated_training_state = checkpoint_module.load_training_state(requested_checkpoint)
    report = {
        "version": 1,
        "kind": "conditional_wae_supervisor_evaluation",
        "config_path": str(config_path),
        "checkpoint_dir": str(requested_checkpoint),
        "checkpoint_step": evaluated_training_state.get("step"),
        "checkpoint_masks_seen": evaluated_training_state.get("masks_seen"),
        "checkpoint_validation_total": evaluated_training_state.get("total"),
        "split": split,
        "n_samples": len(split_ids),
        "n_items": len(dataset),
        "n_latent_samples_per_item": inference_samples,
        "ex_post_prior_path": str(ex_post_prior_path) if ex_post_prior_path else None,
        "task": config["model"]["task"],
        "regularizer": config["model"]["regularizer"],
        "query_he_visible": True,
        "query_gex_visible": False,
        "cache_preflight_report": preflight_report,
        "per_arm_patient_aggregated_metrics": aggregated,
        "gene_panel_metadata": panel_metadata,
        "per_panel_patient_aggregated_metrics": panel_aggregated,
        "conditional_wae_calibration": calibration.summary(),
        "per_item_records": records,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    os.replace(temporary, output_path)
    print(f"conditional WAE evaluation report saved to {output_path}", flush=True)
    for arm in arm_names:
        metrics = aggregated[arm]
        print(
            f"{arm} pcc={metrics['pcc']['patient_mean']:.4f} "
            f"rmse={metrics['rmse']['patient_mean']:.4f}",
            flush=True,
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--n-masks-per-sample", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-use-best", action="store_true")
    parser.add_argument("--allow-code-drift", action="store_true")
    parser.add_argument("--n-samples", type=int)
    parser.add_argument(
        "--ex-post-prior",
        help="Path to a JSON fit by scripts/fit_conditional_wae_ex_post_prior.py. When given, "
             "inference-time z is drawn from this fitted per-latent-dim Gaussian instead of the "
             "raw N(0,I) prior -- the standard no-retrain fix for WAE aggregate-posterior/prior "
             "mismatch. Omit to reproduce the exact pre-existing N(0,I) sampling behavior.",
    )
    args = parser.parse_args()
    evaluate_conditional_wae(
        args.config,
        args.checkpoint_dir,
        args.output,
        split=args.split,
        n_masks_per_sample=args.n_masks_per_sample,
        device_name=args.device,
        use_best=not args.no_use_best,
        allow_code_drift=args.allow_code_drift,
        n_samples=args.n_samples,
        ex_post_prior_path=args.ex_post_prior,
    )


if __name__ == "__main__":
    main()
