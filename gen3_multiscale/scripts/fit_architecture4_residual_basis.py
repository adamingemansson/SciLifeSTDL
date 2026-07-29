#!/usr/bin/env python3
"""Adam's Step 6 audit #11 deliverable (of commit 27e1232): the real
Architecture 4 residual-basis fitting pipeline. "There is no real
pipeline for fitting its residual basis" before this script existed --
`required_fingerprints.gene_residual_basis` (train.py::maybe_load_gene_basis)
had to point at SOMETHING, but nothing in this codebase ever produced a
real one.

Adam's own recommended sequence, implemented literally:
    1. Train Architecture 3 (elsewhere -- `train.py --config
       architecture3.yaml`, not this script's job).
    2. Load its selected, already-trained checkpoint.
    3. Generate residuals on TRAINING samples only.
    4. Fit and persist the basis, with dataset/gene/checkpoint/mask
       provenance recorded alongside it.
    5. (Not this script -- see `train.py::
       maybe_load_pretrained_conditioner_for_architecture4`) initialize
       Architecture 4 from that EXACT Architecture 3 conditioner and
       freeze it initially while training the flow apparatus.

"Architecture 4 therefore should not yet run concurrently from random
initialization with Architectures 1-3" -- this script's whole existence
is the fix: Architecture 4 cannot be trained meaningfully until this has
been run once against a real, trained Architecture 3 checkpoint.

    python -m gen3_multiscale.scripts.fit_architecture4_residual_basis \\
        --config gen3_multiscale/configs/architecture3.yaml \\
        --architecture3-checkpoint-dir /path/to/trained/architecture3 \\
        --output-basis-path /path/to/gene_residual_basis.pt \\
        --n-masks-per-sample 20 --rank 64
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from gen3_multiscale.data.dataset_manifest import gene_panel_hash, load_dataset_manifest
from gen3_multiscale.models import model_factory
from gen3_multiscale.models.gene_basis import fit_gene_residual_basis, save_gene_residual_basis
from gen3_multiscale.training import checkpoint as checkpoint_module
from gen3_multiscale.training.gen3_dataset import Gen3SpatialFieldDataset, build_gen3_mask_schedule
from gen3_multiscale.training.gen3_preflight import load_and_preflight_samples
from gen3_multiscale.training.train import (
    config_fingerprint, dataset_manifest_fingerprint, expected_tile_encoder_provenance, file_sha256, resolved_config,
)


def compute_training_residuals(architecture3_model: torch.nn.Module, train_dataset, device: torch.device) -> np.ndarray:
    """Real residuals -- `target_expression - Architecture3's own
    deterministic conditioner mean` -- over EVERY item currently in
    `train_dataset`. `train_dataset` is a real `Gen3SpatialFieldDataset`
    built with `role="train"`, so this never touches validation/test data
    by construction (Adam's "generate residuals on training samples
    only")."""
    architecture3_model.eval()
    residuals = []
    with torch.no_grad():
        for idx in range(len(train_dataset)):
            inputs, targets = train_dataset[idx]
            target_expression = torch.as_tensor(targets.query_expression, dtype=torch.float32, device=device)
            out = architecture3_model(inputs)
            residual = target_expression - torch.as_tensor(out["expression"], dtype=torch.float32, device=device)
            residuals.append(residual.detach().cpu().numpy())
    if not residuals:
        raise ValueError("compute_training_residuals: train_dataset produced zero items")
    return np.concatenate(residuals, axis=0).astype(np.float32)


def fit_and_save_architecture4_basis(
    architecture3_config_path: str, architecture3_checkpoint_dir: str, output_basis_path: str,
    *, n_masks_per_sample: int = 20, rank: int = 64, device_str: str = "cpu",
) -> dict:
    """The full pipeline. Returns (and persists alongside the basis file,
    as `<output_basis_path>.provenance.json`) a provenance record binding
    the fitted basis to the exact config, dataset manifest, gene panel,
    Architecture 3 checkpoint, and mask schedule it was produced from --
    Adam's "fit and persist the basis with dataset/gene/checkpoint/mask
    provenance"."""
    config = resolved_config(architecture3_config_path)
    architecture_id = str(config["model"]["architecture"])
    if architecture_id != "3":
        raise ValueError(
            f"fit_and_save_architecture4_basis requires an Architecture 3 config (got model.architecture="
            f"{architecture_id!r}) -- the basis must be fit against Architecture 3's own trained "
            "conditioner, per Adam's Step 6 audit #11, never a different architecture's"
        )
    data_cfg = config["data"]
    manifest_path = data_cfg.get("gen3_manifest_path")
    if not manifest_path:
        raise ValueError("data.gen3_manifest_path must be set to a real, already-built dataset manifest")
    dataset_manifest = load_dataset_manifest(manifest_path)
    train_ids = list(dataset_manifest["train_sample_ids"])
    if not train_ids:
        raise ValueError("dataset manifest has zero train_sample_ids -- nothing to fit a residual basis on")

    cfg_om = OmegaConf.create(config)
    expected_provenance = expected_tile_encoder_provenance(config)
    samples, _preflight_report = load_and_preflight_samples(cfg_om, dataset_manifest, train_ids, expected_provenance)

    strata = config["masking"]["strata"]
    train_schedule = build_gen3_mask_schedule(
        dataset_manifest, samples, strata, role="train", n_training_masks_per_sample=n_masks_per_sample,
    )
    train_dataset = Gen3SpatialFieldDataset(dataset_manifest, samples, train_schedule, strata)

    gene_names = list(dataset_manifest["gene_panel"])
    n_genes = len(gene_names)
    gex_feature_dim = int(data_cfg.get("gex_feature_dim", 128))
    device = torch.device(device_str)

    architecture3_model = model_factory.build_architecture(
        config, n_genes=n_genes, gex_feature_dim=gex_feature_dim, seed=int(config["training"].get("seed", 0)),
    ).to(device)
    checkpoint_module.verify_gene_names(architecture3_checkpoint_dir, gene_names)
    checkpoint_module.load_trainable_state(architecture3_model, architecture3_checkpoint_dir)

    residuals = compute_training_residuals(architecture3_model, train_dataset, device)
    basis = fit_gene_residual_basis(residuals, gene_names, rank=rank)
    saved_basis_path = save_gene_residual_basis(basis, output_basis_path)

    weights_path = Path(architecture3_checkpoint_dir) / "trainable_weights.pt"
    checkpoint_sha256 = file_sha256(weights_path) if weights_path.is_file() else None
    provenance = {
        "version": 1,
        "kind": "gen3_architecture4_residual_basis_provenance",
        "architecture3_config_path": str(architecture3_config_path),
        "architecture3_config_fingerprint": config_fingerprint(config),
        "architecture3_checkpoint_dir": str(architecture3_checkpoint_dir),
        "architecture3_checkpoint_trainable_weights_sha256": checkpoint_sha256,
        "dataset_manifest_fingerprint": dataset_manifest_fingerprint(dataset_manifest),
        "gene_panel_hash": gene_panel_hash(gene_names),
        "n_genes": n_genes,
        "train_sample_ids": sorted(train_ids),
        "n_masks_per_sample": int(n_masks_per_sample),
        "n_residual_rows": int(residuals.shape[0]),
        "rank": basis.rank,
        "mask_schedule_reports": train_schedule.reports,
        "output_basis_path": str(saved_basis_path),
    }
    provenance_path = Path(f"{saved_basis_path}.provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True, default=str))
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="A real architecture3.yaml (data.gen3_manifest_path must be set)")
    parser.add_argument("--architecture3-checkpoint-dir", required=True, help="A real, already-trained Architecture 3 checkpoint_dir")
    parser.add_argument("--output-basis-path", required=True)
    parser.add_argument("--n-masks-per-sample", type=int, default=20)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    provenance = fit_and_save_architecture4_basis(
        args.config, args.architecture3_checkpoint_dir, args.output_basis_path,
        n_masks_per_sample=args.n_masks_per_sample, rank=args.rank, device_str=args.device,
    )
    print(f"gene residual basis fit and saved: {json.dumps(provenance, indent=2, default=str)}")


if __name__ == "__main__":
    main()
