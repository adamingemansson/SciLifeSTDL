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


@torch.no_grad()
def _latent_path_predictions(
    model,
    inputs,
    target_expression,
    *,
    n_prior_samples: int,
    generator: torch.Generator,
    z_mean: torch.Tensor | None = None,
    z_std: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Evaluate the distinct deterministic, posterior and prior WAE paths.

    This is diagnostic-only: ``posterior_reconstruction`` is allowed to encode
    the real target GEX and must never be presented as an inference result.  It
    answers whether the trained decoder can use target-aligned latent
    information at all.  The zero and shuffled paths then separate "decoder
    ignores z" from "decoder uses z, but prior samples are misaligned".
    """
    if n_prior_samples < 1:
        raise ValueError("n_prior_samples must be positive")
    if model.distributional_head is not None:
        raise ValueError(
            "latent-path diagnostics require likelihood='gaussian_mse'; the "
            "distributional head bypasses the WAE latent decoder at inference"
        )

    context = model.image_conditioner(inputs)
    target = model._target(
        inputs,
        target_expression,
        device=context.device,
        dtype=context.dtype,
        n_rows=context.shape[0],
    )
    posterior_z = model._encode_target(target, context)

    posterior = model.decode_latent_from_context(posterior_z, context, inputs)

    zero_z = torch.zeros_like(posterior_z)
    zero_prediction = model.decode_latent_from_context(zero_z, context, inputs)

    if posterior_z.shape[0] > 1:
        # A deterministic cyclic permutation breaks spot-to-latent alignment
        # without consuming the prior-sampling RNG stream.  Consequently the
        # diagnostic's ``model`` arm remains bit-identical to ordinary rho=0
        # prior sampling for the same seed and number of draws.
        shuffled_z = torch.roll(posterior_z, shifts=1, dims=0)
    else:
        shuffled_z = posterior_z
    shuffled_prediction = model.decode_latent_from_context(shuffled_z, context, inputs)

    if z_mean is None:
        z_mean = torch.zeros(model.latent_dim, dtype=context.dtype, device=context.device)
    else:
        z_mean = z_mean.to(dtype=context.dtype, device=context.device)
    if z_std is None:
        z_std = torch.ones(model.latent_dim, dtype=context.dtype, device=context.device)
    else:
        z_std = z_std.to(dtype=context.dtype, device=context.device)
    if z_mean.shape != (model.latent_dim,) or z_std.shape != (model.latent_dim,):
        raise ValueError("z_mean and z_std must match the model latent dimension")
    if bool((z_std < 0).any()):
        raise ValueError("z_std must be non-negative")

    prior_predictions = []
    for _ in range(n_prior_samples):
        noise = torch.randn(
            posterior_z.shape,
            dtype=context.dtype,
            device=context.device,
            generator=generator,
        )
        prior_z = z_mean.unsqueeze(0) + noise * z_std.unsqueeze(0)
        prior_predictions.append(model.decode_latent_from_context(prior_z, context, inputs))
    stacked = torch.stack(prior_predictions)
    conditional_mean = model.predict_point_from_context(context, inputs)
    return {
        "model": stacked.mean(0),
        "predictive_std": stacked.std(0, unbiased=False),
        "conditional_mean": conditional_mean,
        "posterior_reconstruction": posterior,
        "zero_latent": zero_prediction,
        "shuffled_posterior": shuffled_prediction,
        "posterior_z": posterior_z,
    }


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
    latent_spatial_correlation: float = 0.0,
    diagnose_latent: bool = False,
) -> dict:
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    if n_masks_per_sample < 1:
        raise ValueError("n_masks_per_sample must be positive")
    if diagnose_latent and latent_spatial_correlation != 0.0:
        raise ValueError(
            "--diagnose-latent currently requires --latent-spatial-correlation 0; "
            "the diagnostic isolates marginal prior/posterior alignment, not joint-field correlation"
        )
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
    arm_names = ["model", "conditional_mean"]
    if diagnose_latent:
        arm_names += ["posterior_reconstruction", "zero_latent", "shuffled_posterior"]
    items = {arm: [] for arm in arm_names}
    panel_items = {arm: {panel: [] for panel in panels} for arm in arm_names}
    patient_ids = []
    records = []
    calibration = _GaussianCalibrationAccumulator()
    inference_samples = int(n_samples or config["model"]["params"]["n_inference_samples"])
    evaluation_seed = int(config["training"].get("seed", 0))
    latent_sum = torch.zeros(model.latent_dim, dtype=torch.float64, device=device)
    latent_sum_squared = torch.zeros_like(latent_sum)
    latent_rows = 0
    latent_output_sensitivity = []
    with torch.no_grad():
        for index in range(len(dataset)):
            inputs, target, identity = dataset[index]
            patient_id = str(samples[identity["sample_id"]].patient_id)
            patient_ids.append(patient_id)
            generator = torch.Generator(device=device).manual_seed(
                _stable_seed(evaluation_seed, identity),
            )
            true = np.asarray(target, dtype=np.float32)
            if diagnose_latent:
                prediction = _latent_path_predictions(
                    model,
                    inputs,
                    target,
                    n_prior_samples=inference_samples,
                    generator=generator,
                    z_mean=ex_post_z_mean,
                    z_std=ex_post_z_std,
                )
                prediction_tensors = {arm: prediction[arm] for arm in arm_names}
                posterior_z = prediction["posterior_z"].to(torch.float64)
                latent_sum += posterior_z.sum(dim=0)
                latent_sum_squared += posterior_z.square().sum(dim=0)
                latent_rows += int(posterior_z.shape[0])
                latent_output_sensitivity.append({
                    "posterior_vs_zero_rmse": float(torch.sqrt(torch.mean(
                        (prediction["posterior_reconstruction"] - prediction["zero_latent"]).square()
                    ))),
                    "posterior_vs_shuffled_rmse": float(torch.sqrt(torch.mean(
                        (prediction["posterior_reconstruction"] - prediction["shuffled_posterior"]).square()
                    ))),
                })
                predictive_std = prediction["predictive_std"]
            else:
                prediction = model.sample_predictive_distribution(
                    inputs, n_samples=inference_samples, generator=generator,
                    z_mean=ex_post_z_mean, z_std=ex_post_z_std,
                    latent_spatial_correlation=latent_spatial_correlation,
                )
                prediction_tensors = {
                    "model": prediction["predictive_mean"],
                    "conditional_mean": prediction["conditional_mean_expression"],
                }
                predictive_std = prediction["predictive_std"]
            prediction_arrays = {
                arm: tensor.detach().cpu().numpy().astype(np.float32)
                for arm, tensor in prediction_tensors.items()
            }
            arm_metrics = {
                arm: per_item_reconstruction_metrics(prediction_arrays[arm], true)
                for arm in arm_names
            }
            for arm in arm_names:
                items[arm].append(arm_metrics[arm])
            record = {
                **identity,
                "patient_id": patient_id,
                **arm_metrics,
            }
            if diagnose_latent:
                record["latent_output_sensitivity"] = latent_output_sensitivity[-1]
            if panels:
                record["gene_panels"] = {}
                for arm, prediction_array in prediction_arrays.items():
                    metrics = gene_panel_metrics(prediction_array, true, gene_names, panels)
                    record["gene_panels"][arm] = metrics
                    for panel, panel_metric in metrics.items():
                        panel_items[arm][panel].append(panel_metric)
            model_pred = prediction_arrays["model"]
            std = predictive_std.detach().cpu().numpy().astype(np.float64)
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
    latent_summary = None
    if diagnose_latent:
        if latent_rows < 1:
            raise ValueError("latent diagnostic encoded no posterior rows")
        z_mean_observed = latent_sum / latent_rows
        z_variance_observed = latent_sum_squared / latent_rows - z_mean_observed.square()
        z_std_observed = torch.sqrt(torch.clamp(z_variance_observed, min=0.0))
        latent_summary = {
            "n_rows": int(latent_rows),
            "posterior_mean_l2": float(torch.linalg.vector_norm(z_mean_observed)),
            "posterior_std_min": float(z_std_observed.min()),
            "posterior_std_mean": float(z_std_observed.mean()),
            "posterior_std_max": float(z_std_observed.max()),
            "mean_posterior_vs_zero_output_rmse": float(np.mean([
                row["posterior_vs_zero_rmse"] for row in latent_output_sensitivity
            ])),
            "mean_posterior_vs_shuffled_output_rmse": float(np.mean([
                row["posterior_vs_shuffled_rmse"] for row in latent_output_sensitivity
            ])),
        }
    report = {
        "version": 2,
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
        "latent_spatial_correlation": float(latent_spatial_correlation),
        "diagnose_latent": bool(diagnose_latent),
        "latent_diagnostic_summary": latent_summary,
        "prediction_roles": {
            "primary_point_prediction": "conditional_mean",
            "deterministic_h_and_e_prediction": "conditional_mean",
            "wae_prior_predictive_mean": "model",
            "calibration_prediction": "model",
            "posterior_reconstruction_uses_target_gex": bool(diagnose_latent),
        },
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
        label = {
            "model": "wae_prior_predictive_mean",
            "conditional_mean": "deterministic_h_and_e_point_prediction",
        }.get(arm, arm)
        print(
            f"{label} pcc={metrics['pcc']['patient_mean']:.4f} "
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
        "--diagnose-latent", action="store_true",
        help="Add target-posterior, zero-latent and within-mask shuffled-posterior arms. "
             "The posterior arm reads held-out target GEX and is diagnostic only, never inference.",
    )
    parser.add_argument(
        "--ex-post-prior",
        help="Path to a JSON fit by scripts/fit_conditional_wae_ex_post_prior.py. When given, "
             "inference-time z is drawn from this fitted per-latent-dim Gaussian instead of the "
             "raw N(0,I) prior -- the standard no-retrain fix for WAE aggregate-posterior/prior "
             "mismatch. Omit to reproduce the exact pre-existing N(0,I) sampling behavior.",
    )
    parser.add_argument(
        "--latent-spatial-correlation", type=float, default=0.0,
        help="Correlation rho in [0,1] between spots' inference-time latent draws. "
             "z_i = sqrt(rho)*z_shared + sqrt(1-rho)*z_i keeps each z_i exactly marginally "
             "N(0,I) -- what the MMD/GAN regularizer trained for -- while making a drawn "
             "field spatially coherent instead of i.i.d. per spot. Pure inference-time "
             "change, valid on already-trained checkpoints; 0.0 (default) reproduces the "
             "historical sampler exactly.",
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
        latent_spatial_correlation=args.latent_spatial_correlation,
        diagnose_latent=args.diagnose_latent,
    )


if __name__ == "__main__":
    main()
