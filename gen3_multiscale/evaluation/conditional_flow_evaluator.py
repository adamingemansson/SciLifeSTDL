#!/usr/bin/env python3
"""Held-out evaluator for the matched MK conditional-flow tasks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from gen3_multiscale.conditional_flow import ConditionalWAEMaskedGEXDataset
from gen3_multiscale.conditional_flow.contract import static_audit_conditional_flow_config
from gen3_multiscale.config_identity import resolved_config
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
from gen3_multiscale.evaluation.schedule_contract import fixed_mask_evaluation_metadata
from gen3_multiscale.training import checkpoint as checkpoint_module
from gen3_multiscale.training.gen3_dataset import Gen3SpatialFieldDataset, build_gen3_mask_schedule
from gen3_multiscale.training.gen3_preflight import load_and_preflight_samples
from gen3_multiscale.training.train import expected_tile_encoder_provenance
from gen3_multiscale.training.train_conditional_flow import (
    _build_model,
    _manifest,
    _verify_resume,
)
from gen3_multiscale.training.train_conditional_wae import _stable_seed


def evaluate_conditional_flow(
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
    n_steps: int | None = None,
) -> dict:
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    if n_masks_per_sample < 1:
        raise ValueError("n_masks_per_sample must be positive")
    config = resolved_config(config_path)
    static_audit_conditional_flow_config(config)
    dataset_manifest = load_dataset_manifest(config["data"]["gen3_manifest_path"])
    split_ids = list(dataset_manifest[f"{split}_sample_ids"])
    samples, preflight_report = load_and_preflight_samples(
        OmegaConf.create(config), dataset_manifest, split_ids,
        expected_tile_encoder_provenance(config),
    )
    schedule = build_gen3_mask_schedule(
        dataset_manifest, samples, config["masking"]["strata"], split,
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
                f"{sample_id}: evaluation cache content differs from training"
            )
    current_manifest["cache_content_fingerprint"] = old_manifest["cache_content_fingerprint"]
    current_manifest["cache_content_by_sample"] = old_cache
    _verify_resume(old_manifest, current_manifest, allow_code_drift=allow_code_drift)
    gene_names = list(dataset_manifest["gene_panel"])
    checkpoint_module.verify_gene_names(requested_checkpoint, gene_names)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model = _build_model(config, len(gene_names)).to(device)
    checkpoint_module.load_trainable_state(model, requested_checkpoint)
    model.eval()

    panels = load_configured_gene_panels(config, dataset_manifest)
    _indices, panel_metadata = resolve_gene_panels(gene_names, panels) if panels else ({}, {})
    arm_names = ("model", "conditional_mean")
    items = {arm: [] for arm in arm_names}
    panel_items = {arm: {panel: [] for panel in panels} for arm in arm_names}
    patient_ids, records = [], []
    calibration = _GaussianCalibrationAccumulator()
    inference_samples = int(n_samples or config["model"]["params"]["n_inference_samples"])
    inference_steps = int(n_steps or config["model"]["params"]["n_ode_steps"])
    seed = int(config["training"].get("seed", 0))
    with torch.no_grad():
        for index in range(len(dataset)):
            inputs, target, identity = dataset[index]
            patient_id = str(samples[identity["sample_id"]].patient_id)
            patient_ids.append(patient_id)
            generator = torch.Generator(device=device).manual_seed(
                _stable_seed(seed, identity),
            )
            prediction = model.sample_predictive_distribution(
                inputs, n_samples=inference_samples, n_steps=inference_steps,
                generator=generator,
            )
            true = np.asarray(target, dtype=np.float32)
            model_pred = prediction["predictive_mean"].cpu().numpy().astype(np.float32)
            mean_pred = prediction["conditional_mean_expression"].cpu().numpy().astype(np.float32)
            model_metrics = per_item_reconstruction_metrics(model_pred, true)
            mean_metrics = per_item_reconstruction_metrics(mean_pred, true)
            items["model"].append(model_metrics)
            items["conditional_mean"].append(mean_metrics)
            record = {
                **identity, "patient_id": patient_id,
                "model": model_metrics, "conditional_mean": mean_metrics,
            }
            if panels:
                record["gene_panels"] = {}
                for arm, values in (("model", model_pred), ("conditional_mean", mean_pred)):
                    metrics = gene_panel_metrics(values, true, gene_names, panels)
                    record["gene_panels"][arm] = metrics
                    for panel, metric in metrics.items():
                        panel_items[arm][panel].append(metric)
            std = prediction["predictive_std"].cpu().numpy().astype(np.float64)
            calibration.add_item(
                (true.astype(np.float64) - model_pred)
                / np.where(std > 1e-8, std, np.nan)
            )
            records.append(record)
            print(
                f"conditional flow evaluation progress: {index + 1}/{len(dataset)} "
                f"sample={identity['sample_id']} stratum={identity['stratum']}",
                flush=True,
            )
    aggregated = {
        arm: aggregate_patient_metrics(values, patient_ids)
        for arm, values in items.items()
    }
    panel_aggregated = {
        panel: {
            arm: aggregate_patient_metrics(panel_items[arm][panel], patient_ids)
            for arm in arm_names
        }
        for panel in panels
    }
    report = {
        "version": 1,
        "kind": "conditional_flow_supervisor_evaluation",
        "config_path": str(config_path),
        "checkpoint_dir": str(requested_checkpoint),
        **fixed_mask_evaluation_metadata(
            split=split, sample_ids=split_ids, strata=config["masking"]["strata"],
            masks_per_stratum_per_sample=n_masks_per_sample,
            actual_n_items=len(dataset),
        ),
        "n_latent_samples_per_item": inference_samples,
        "n_ode_steps": inference_steps,
        "task": config["model"]["task"],
        "coupling": config["model"]["coupling"],
        "query_he_visible": True,
        "query_gex_visible": False,
        "cache_preflight_report": preflight_report,
        "per_arm_patient_aggregated_metrics": aggregated,
        "gene_panel_metadata": panel_metadata,
        "per_panel_patient_aggregated_metrics": panel_aggregated,
        "conditional_flow_calibration": calibration.summary(),
        "per_item_records": records,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    os.replace(temporary, output_path)
    print(f"conditional flow evaluation report saved to {output_path}", flush=True)
    for arm in arm_names:
        metrics = aggregated[arm]
        print(
            f"{arm} pcc={metrics['pcc']['patient_mean']:.4f} "
            f"rmse={metrics['rmse']['patient_mean']:.4f}", flush=True,
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--n-masks-per-stratum-per-sample", "--n-masks-per-sample",
        dest="n_masks_per_sample", type=int, default=8,
        help="Masks for each configured stratum of each sample; legacy spelling retained.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-use-best", action="store_true")
    parser.add_argument("--allow-code-drift", action="store_true")
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--n-steps", type=int)
    args = parser.parse_args()
    evaluate_conditional_flow(
        args.config, args.checkpoint_dir, args.output,
        split=args.split, n_masks_per_sample=args.n_masks_per_sample,
        device_name=args.device, use_best=not args.no_use_best,
        allow_code_drift=args.allow_code_drift,
        n_samples=args.n_samples, n_steps=args.n_steps,
    )


if __name__ == "__main__":
    main()
