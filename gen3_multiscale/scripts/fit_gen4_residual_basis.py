#!/usr/bin/env python3
"""Fit a Gen4 flow residual basis from one selected Gen4 conditioner.

Only manifest-declared training samples are used. The resulting basis is
bound to the exact conditioner bundle, dataset, gene panel, masks, and
consumed cache content in ``<basis>.provenance.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from gen3_multiscale.data.dataset_manifest import gene_panel_hash, load_dataset_manifest
from gen3_multiscale.gen4.basis_fit import fit_gen4_residual_basis
from gen3_multiscale.gen4.dataset_adapter import (
    Gen4SpatialFieldDataset,
    load_and_preflight_gen4_samples,
)
from gen3_multiscale.gen4.trainer_adapter import _resolve_gen4_arm
from gen3_multiscale.training import checkpoint as checkpoint_module
from gen3_multiscale.training.gen3_dataset import build_gen3_mask_schedule
from gen3_multiscale.training.train import (
    build_model_for_inference,
    config_identity_fingerprint,
    dataset_manifest_fingerprint,
    resolved_config,
    verify_full_checkpoint_identity,
)


def fit_and_save_gen4_basis(
    conditioner_config_path: str,
    conditioner_checkpoint_dir: str,
    output_basis_path: str,
    *,
    n_masks_per_sample: int = 20,
    rank: int = 64,
    device_str: str = "cpu",
    allow_code_drift: bool = False,
    svd_device: str = "cpu",
    svd_n_iter: int = 4,
    svd_oversamples: int = 16,
) -> dict:
    config = resolved_config(conditioner_config_path)
    model_cfg = config.get("model") or {}
    if model_cfg.get("arm") is None or str(model_cfg.get("kind")) != "conditioner":
        raise ValueError("basis fitting requires a Gen4 conditioner config")
    if n_masks_per_sample < len(config["masking"]["strata"]):
        raise ValueError("n_masks_per_sample must cover every configured mask stratum")

    manifest_path = (config.get("data") or {}).get("gen3_manifest_path")
    if not manifest_path:
        raise ValueError("data.gen3_manifest_path is required")
    manifest = load_dataset_manifest(manifest_path)
    train_ids = list(manifest["train_sample_ids"])
    if not train_ids:
        raise ValueError("dataset manifest has no training samples")
    gene_names = list(manifest["gene_panel"])
    cfg_om = OmegaConf.create(config)
    samples, preflight = load_and_preflight_gen4_samples(
        cfg_om, manifest, train_ids, config, require_resolved_artifacts=True,
    )

    pinned = checkpoint_module.resolve_checkpoint_identity(conditioner_checkpoint_dir)
    verify_full_checkpoint_identity(
        pinned.resolved_dir,
        config=config,
        dataset_manifest=manifest,
        gene_names=gene_names,
        cache_content_by_sample=preflight["cache_content_by_sample"],
        allow_code_drift=allow_code_drift,
    )

    strata = config["masking"]["strata"]
    schedule = build_gen3_mask_schedule(
        manifest,
        samples,
        strata,
        role="train",
        n_training_masks_per_sample=n_masks_per_sample,
    )
    dataset = Gen4SpatialFieldDataset(
        manifest, samples, schedule, strata, cfg_om, gene_names,
    )
    device = torch.device(device_str)
    conditioner, info = build_model_for_inference(
        config,
        gene_names=gene_names,
        device=device,
        checkpoint_dir=pinned.resolved_dir,
        smoke=False,
        dataset_manifest=manifest,
        cache_content_by_sample=preflight["cache_content_by_sample"],
        allow_code_drift=allow_code_drift,
    )
    if info["kind"] != "conditioner":
        raise RuntimeError("resolved model is not a deterministic conditioner")

    basis = fit_gen4_residual_basis(
        conditioner,
        dataset,
        gene_names,
        rank=rank,
        device=device,
        output_basis_path=output_basis_path,
        svd_device=svd_device,
        svd_n_iter=svd_n_iter,
        svd_oversamples=svd_oversamples,
    )
    basis_sha256 = hashlib.sha256(
        np.ascontiguousarray(basis.basis.detach().cpu().numpy()).tobytes()
    ).hexdigest()
    training_mask_schedule_fingerprint = hashlib.sha256(
        json.dumps(schedule.reports, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    provenance = {
        "version": 1,
        "kind": "gen4_residual_basis_provenance",
        "conditioner_arm": _resolve_gen4_arm(config),
        "conditioner_config_identity_fingerprint": config_identity_fingerprint(config),
        "conditioner_checkpoint_sha256": pinned.weights_sha256,
        "conditioner_checkpoint_step": pinned.step,
        "conditioner_checkpoint_bundle_id": pinned.bundle_dir,
        "conditioner_checkpoint_manifest_sha256": pinned.manifest_sha256,
        "dataset_manifest_fingerprint": dataset_manifest_fingerprint(manifest),
        "gene_panel_hash": gene_panel_hash(gene_names),
        "gene_residual_basis_sha256": basis_sha256,
        "rank": int(basis.rank),
        "svd_device_type": torch.device(svd_device).type,
        "svd_n_iter": int(svd_n_iter),
        "svd_oversamples": int(svd_oversamples),
        "n_masks_per_sample": int(n_masks_per_sample),
        "train_sample_ids": sorted(train_ids),
        "cache_content_by_sample": preflight["cache_content_by_sample"],
        "mask_schedule_reports": schedule.reports,
        "training_mask_schedule_fingerprint": training_mask_schedule_fingerprint,
    }
    provenance_path = Path(f"{output_basis_path}.provenance.json")
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = provenance_path.with_name(f"{provenance_path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(provenance, indent=2, sort_keys=True, default=str))
    os.replace(tmp, provenance_path)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--conditioner-checkpoint-dir", required=True)
    parser.add_argument("--output-basis-path", required=True)
    parser.add_argument("--n-masks-per-sample", type=int, default=20)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--svd-device", default="cpu")
    parser.add_argument("--svd-n-iter", type=int, default=4)
    parser.add_argument("--svd-oversamples", type=int, default=16)
    parser.add_argument("--allow-code-drift", action="store_true")
    args = parser.parse_args()
    report = fit_and_save_gen4_basis(
        args.config,
        args.conditioner_checkpoint_dir,
        args.output_basis_path,
        n_masks_per_sample=args.n_masks_per_sample,
        rank=args.rank,
        device_str=args.device,
        allow_code_drift=args.allow_code_drift,
        svd_device=args.svd_device,
        svd_n_iter=args.svd_n_iter,
        svd_oversamples=args.svd_oversamples,
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
