#!/usr/bin/env python3
"""Step 6 of the real Gen3 data builder/trainer: the real training
entrypoint. `python -m gen3_multiscale.training.train --config <path>
[--smoke]` -- this exact module path and CLI contract is what
`training/launch_four_gpu_suite.py::default_command_builder` has assumed
since Phase 8, before this module existed (that module's own docstring:
"gen3_multiscale.training.train, a module that DOES NOT EXIST YET...
this launcher's own mechanics are fully implemented and tested against a
stub command; the actual four 24-hour jobs cannot be started today,
with or without this launcher, until that entrypoint and a real data
builder exist." Both now exist.)

Order of operations, deliberately fixed (Adam's Step 6 mandatory
requirements #3/#4/#7): load the immutable dataset manifest -> run the
mandatory cache-coverage + tile-encoder-provenance preflight -> build the
mask schedule/dataset -> construct the model -> load its verified
synchronized initialization (fails closed) -> construct the optimizer ->
train. Nothing before "construct the model" ever imports torch's
autograd machinery for a real forward pass, and nothing after preflight
can proceed if preflight raised.

`--smoke` runs exactly ONE training step (and one validation step, if a
validation split exists) then exits -- the "one-step smoke test"
deliverable literally IS `train.py --config <cfg> --smoke` run once per
architecture config. This module never starts a real multi-hour run on
its own: nothing below `if __name__ == "__main__":` executes on import,
and a real >1-step run only ever happens via an explicit CLI invocation
with `--smoke` omitted -- the same "does not automatically launch"
structural guarantee `launch_four_gpu_suite.py` already documents for
itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from gen3_multiscale.data.dataset_manifest import gene_panel_hash, load_dataset_manifest
from gen3_multiscale.models import model_factory
from gen3_multiscale.models.losses import combined_reconstruction_loss
from gen3_multiscale.training import checkpoint as checkpoint_module
from gen3_multiscale.training.gen3_dataset import (
    Gen3SpatialFieldDataset, build_gen3_mask_schedule, gen3_identity_collate,
)
from gen3_multiscale.training.gen3_preflight import load_and_preflight_samples, save_gen3_preflight_report


def resolved_config(config_path: str | Path) -> dict:
    return OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)


def expected_tile_encoder_provenance(config: dict) -> dict:
    """The experiment-declared expected provenance
    `tile_encoder_preflight.require_consistent_tile_encoder_provenance`
    requires -- sourced from `data.tile_encoder_revision`, never
    inferred from whichever cache happens to load first."""
    data_cfg = config.get("data") or {}
    revision = data_cfg.get("tile_encoder_revision")
    if not revision:
        raise ValueError(
            "data.tile_encoder_revision must be set to the experiment's pinned, immutable "
            "Hugging Face commit SHA -- see scripts/precompute_gigapath_wsi_tiles.py's own "
            "--tile-encoder-revision for how it is resolved"
        )
    return {"hf_repo_id": "prov-gigapath/prov-gigapath", "hf_revision": str(revision), "schema_version": 1}


def maybe_build_slide_encoder(config: dict):
    """Only Architecture 3/4's `use_global_slide` branch needs a real,
    live `FrozenGigaPathSlideEncoder` (regional H&E pooling needs only
    the already-cached tile FEATURES, no separate model). Returns
    `(None, None)` when not needed, or when no real checkpoint is
    configured -- `model_factory.resolve_model_kwargs`/the architecture
    constructor itself then fails closed if `use_global_slide=true` was
    requested without one, exactly as designed (17th Codex re-audit)."""
    model_params = (config.get("model") or {}).get("params") or {}
    if not model_params.get("use_global_slide"):
        return None, None
    checkpoint_path = (config.get("required_fingerprints") or {}).get("gigapath_checkpoint")
    if not checkpoint_path:
        return None, None
    from gen3_multiscale.models.slide_encoder import FrozenGigaPathSlideEncoder

    encoder = FrozenGigaPathSlideEncoder(str(checkpoint_path))
    return encoder, encoder.checkpoint_sha256


def maybe_load_gene_basis(config: dict, gene_names: list[str]):
    """Architecture 4 only: `required_fingerprints.gene_basis` must point
    at an already-fit, saved `GeneResidualBasis`
    (`models/gene_basis.py::save_gene_residual_basis`) -- fit OFFLINE, on
    TRAINING-split residuals only, per Architecture4's own docstring.
    This trainer never fits one itself (it would need a trained
    conditioner's own residuals to fit against in the first place)."""
    architecture_id = str((config.get("model") or {}).get("architecture", ""))
    if architecture_id != "4":
        return None, None
    from gen3_multiscale.models.gene_basis import load_gene_residual_basis, verify_gene_residual_basis

    path = (config.get("required_fingerprints") or {}).get("gene_residual_basis")
    if not path:
        raise ValueError(
            "Architecture 4 requires required_fingerprints.gene_residual_basis -- fit one offline via "
            "models.gene_basis.fit_gene_residual_basis on TRAINING-split residuals, then "
            "models.gene_basis.save_gene_residual_basis it, before training Architecture 4"
        )
    basis = load_gene_residual_basis(path)
    verify_gene_residual_basis(basis, gene_names)
    return basis, gene_names


def compute_step_losses(architecture_id: str, model, inputs, target_expression: torch.Tensor,
                         query_coords: torch.Tensor, gradient_weight: float, k_neighbors: int,
                         flow_weight: float = 1.0) -> dict:
    """Architecture-generic where possible: Architectures 1/2/3 share one
    deterministic reconstruction objective (models/losses.py); Architecture
    4 additionally adds its stopped-gradient flow-matching loss, computed
    from a SINGLE conditioner pass (model.compute_losses) so the
    reconstruction and flow losses agree on the same dropout mask (3rd
    Codex re-audit finding, CONTRACT.md)."""
    if architecture_id == "4":
        out = model.compute_losses(inputs, target_expression)
        recon = combined_reconstruction_loss(
            out["expression"], target_expression, query_coords,
            gradient_weight=gradient_weight, k_neighbors=k_neighbors,
        )
        total = recon["total"] + flow_weight * out["flow_loss"]
        return {"total": total, "primary": recon["primary"], "gradient": recon["gradient"], "flow_loss": out["flow_loss"]}
    out = model(inputs)
    return combined_reconstruction_loss(
        out["expression"], target_expression, query_coords,
        gradient_weight=gradient_weight, k_neighbors=k_neighbors,
    )


def _dataset_manifest_fingerprint(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _config_fingerprint(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def build_run_manifest(
    *, config: dict, config_path: str, dataset_manifest: dict, gene_names: list[str], seed: int,
    preflight_report: dict, train_schedule, val_schedule, architecture_id: str,
    checkpoint_dir: Path, synchronized_init_manifest_path: Path | None,
) -> dict:
    """Requirement #8: one artifact binding dataset, gene panel, split,
    mask, cache, model/checkpoint, configuration, and seed fingerprints
    together -- so a later reader can tell EXACTLY what this run trained
    on without re-deriving it from scattered files."""
    return {
        "version": 1,
        "kind": "gen3_step6_run_manifest",
        "config_path": str(config_path),
        "config_fingerprint": _config_fingerprint(config),
        "seed": int(seed),
        "dataset_manifest_fingerprint": _dataset_manifest_fingerprint(dataset_manifest),
        "gene_panel_hash": gene_panel_hash(gene_names),
        "n_genes": len(gene_names),
        "split": {
            "train_sample_ids": list(dataset_manifest["train_sample_ids"]),
            "validation_sample_ids": list(dataset_manifest["validation_sample_ids"]),
            "test_sample_ids": list(dataset_manifest["test_sample_ids"]),
        },
        "cache_preflight_report": preflight_report,
        "mask_schedule_reports": {
            "train": train_schedule.reports,
            "validation": val_schedule.reports if val_schedule is not None else {},
        },
        "model_architecture": architecture_id,
        "checkpoint_dir": str(checkpoint_dir),
        "synchronized_init_manifest_path": (
            str(synchronized_init_manifest_path) if synchronized_init_manifest_path is not None else None
        ),
    }


def save_run_manifest(run_manifest: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(run_manifest, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)
    return path


def _log_step(step: int, split: str, losses: dict, extra: str = "") -> None:
    parts = ", ".join(f"{k}={float(v.detach()) if torch.is_tensor(v) else float(v):.6f}" for k, v in losses.items())
    print(f"[step {step}] {split}: {parts}{extra}", flush=True)


def run_training(config_path: str, smoke: bool = False) -> dict:
    """The real Step 6 training entrypoint. Returns a small summary dict
    (never a live model/optimizer -- those are process-local); a caller
    that wants the trained model runs this in-process and reads
    `checkpoint_dir` afterward, matching every other artifact-based
    hand-off in this package."""
    config = resolved_config(config_path)
    data_cfg = config["data"]
    training_cfg = config["training"]
    architecture_id = str(config["model"]["architecture"])

    manifest_path = data_cfg.get("gen3_manifest_path")
    if not manifest_path:
        raise ValueError("data.gen3_manifest_path must be set to a real, already-built dataset manifest")
    if not Path(manifest_path).is_file():
        raise FileNotFoundError(f"dataset manifest not found at {manifest_path} -- build it first")
    dataset_manifest = load_dataset_manifest(manifest_path)

    # Requirement #1: sample selection and the train/validation/test
    # split are read EXCLUSIVELY from the manifest -- never re-derived.
    train_ids = list(dataset_manifest["train_sample_ids"])
    validation_ids = list(dataset_manifest["validation_sample_ids"])
    if not train_ids:
        raise ValueError("dataset manifest has zero train_sample_ids -- nothing to train on")

    checkpoint_dir = Path(training_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Requirements #3/#4: cache coverage + tile-encoder provenance
    # consistency, BEFORE any model/optimizer/DataLoader is constructed.
    # Preflighted over train+validation samples -- the only manifest
    # roles this trainer touches (test-split evaluation is Step 7's job).
    cfg_om = OmegaConf.create(config)
    expected_provenance = expected_tile_encoder_provenance(config)
    samples, preflight_report = load_and_preflight_samples(
        cfg_om, dataset_manifest, train_ids + validation_ids, expected_provenance,
    )
    save_gen3_preflight_report(preflight_report, checkpoint_dir / "preflight_report.json")

    strata = config["masking"]["strata"]
    train_samples = {sid: s for sid, s in samples.items() if sid in train_ids}
    val_samples = {sid: s for sid, s in samples.items() if sid in validation_ids}

    n_training_masks = 2 if smoke else int(data_cfg.get("n_training_masks_per_sample", 500))
    n_validation_masks = 1 if smoke else int(data_cfg.get("n_validation_masks", 4))

    train_schedule = build_gen3_mask_schedule(
        dataset_manifest, train_samples, strata, role="train", n_training_masks_per_sample=n_training_masks,
    )
    val_schedule = None
    val_loader = None
    if val_samples:
        val_schedule = build_gen3_mask_schedule(
            dataset_manifest, val_samples, strata, role="validation",
            split_counts={"validation": n_validation_masks}, split_seeds={"validation": 700_000},
            mask_bank_dir=str(checkpoint_dir),
        )

    novae_enabled = bool((data_cfg.get("novae") or {}).get("enabled", False))
    train_dataset = Gen3SpatialFieldDataset(
        dataset_manifest, train_samples, train_schedule, strata, novae_enabled=novae_enabled,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=not smoke, collate_fn=gen3_identity_collate,
    )
    if val_schedule is not None:
        val_dataset = Gen3SpatialFieldDataset(
            dataset_manifest, val_samples, val_schedule, strata, novae_enabled=novae_enabled,
        )
        # requirement #9: deterministic FIXED-mask validation -- never shuffled.
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False, collate_fn=gen3_identity_collate,
        )

    seed = int(training_cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    sample_rng = random.Random(seed)

    gene_names = list(dataset_manifest["gene_panel"])
    n_genes = len(gene_names)
    gex_feature_dim = int(data_cfg.get("gex_feature_dim", 128))
    slide_encoder, gigapath_checkpoint_sha256 = maybe_build_slide_encoder(config)
    gene_basis, resolved_gene_names = maybe_load_gene_basis(config, gene_names)

    device = torch.device(training_cfg.get("device", "cpu") if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)  # re-seed immediately before construction -- shared init discipline (model_factory.build_architecture mirrors this)
    model = model_factory.build_architecture(
        config, n_genes=n_genes, gex_feature_dim=gex_feature_dim, gene_basis=gene_basis,
        gene_names=resolved_gene_names, slide_encoder=slide_encoder,
        gigapath_checkpoint_sha256=gigapath_checkpoint_sha256, seed=seed,
    ).to(device)

    # Requirement #7: mandatory, verified synchronized initialization --
    # fails closed if it cannot be verified.
    synchronized_init_manifest_path = None
    synchronized_init_dir = training_cfg.get("synchronized_init_dir")
    if synchronized_init_dir:
        sync_dir = Path(synchronized_init_dir)
        synchronized_init_manifest_path = sync_dir / "initialization_manifest.json"
        sync_manifest = json.loads(synchronized_init_manifest_path.read_text())
        architecture_name = f"architecture{architecture_id}"
        model_factory.load_synchronized_initialization(
            model, sync_dir / architecture_name, sync_manifest, architecture_name,
        )
    elif not smoke:
        raise ValueError(
            "training.synchronized_init_dir must be set for a real (non-smoke) run -- Step 6 "
            "requires a verified, persisted synchronized initialization for every architecture "
            "(persist_four_architecture_initializations); refusing to train four architectures "
            "from four independently-random starting points"
        )

    resume_step = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training_cfg.get("lr", 1e-4)))
    training_state = checkpoint_module.load_training_state(checkpoint_dir)
    if training_state.get("step", 0) > 0:
        checkpoint_module.verify_gene_names(checkpoint_dir, gene_names)
        checkpoint_module.load_trainable_state(model, checkpoint_dir)
        checkpoint_module.load_optimizer_and_rng_state(optimizer, checkpoint_dir, rng=sample_rng)
        resume_step = int(training_state["step"])
        print(f"resumed from checkpoint at step {resume_step}", flush=True)

    run_manifest = build_run_manifest(
        config=config, config_path=str(config_path), dataset_manifest=dataset_manifest, gene_names=gene_names,
        seed=seed, preflight_report=preflight_report, train_schedule=train_schedule, val_schedule=val_schedule,
        architecture_id=architecture_id, checkpoint_dir=checkpoint_dir,
        synchronized_init_manifest_path=synchronized_init_manifest_path,
    )
    save_run_manifest(run_manifest, checkpoint_dir / "run_manifest.json")

    loss_cfg = config.get("loss") or {}
    gradient_weight = float(loss_cfg.get("gradient_weight", 0.05))
    k_neighbors = int(loss_cfg.get("k_neighbors", 6))
    gradient_clip_val = float(training_cfg.get("gradient_clip_val", 1.0))
    total_steps = 1 if smoke else int(training_cfg.get("total_steps", 1))
    log_every_n_steps = max(1, int(training_cfg.get("log_every_n_steps", 50)))
    checkpoint_every_n_steps = max(1, int(training_cfg.get("checkpoint_every_n_steps", 2000)))
    checkpoint_keep_last = int(training_cfg.get("checkpoint_keep_last", 2))

    model.train()
    train_iter = iter(train_loader)
    n_skipped_nonfinite = 0
    step = resume_step
    start_time = time.time()
    while step < resume_step + total_steps:
        try:
            inputs, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            inputs, targets = next(train_iter)

        target_expression = torch.as_tensor(targets.query_expression, dtype=torch.float32, device=device)
        query_coords = torch.as_tensor(inputs.query_coords, dtype=torch.float32, device=device)

        optimizer.zero_grad(set_to_none=True)
        losses = compute_step_losses(
            architecture_id, model, inputs, target_expression, query_coords,
            gradient_weight=gradient_weight, k_neighbors=k_neighbors,
        )
        total_loss = losses["total"]
        # Requirement #9: finite-loss check -- never step the optimizer
        # on a NaN/Inf loss (a real, plausible failure mode: a masking
        # draw with a degenerate query/context geometry, or numerical
        # instability deep in the transport head's softmax).
        if not torch.isfinite(total_loss):
            n_skipped_nonfinite += 1
            print(f"[step {step}] SKIPPED: non-finite total loss ({float(total_loss.detach())!r})", flush=True)
            step += 1
            continue
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
        if not torch.isfinite(grad_norm):
            n_skipped_nonfinite += 1
            print(f"[step {step}] SKIPPED: non-finite gradient norm ({float(grad_norm)!r})", flush=True)
            optimizer.zero_grad(set_to_none=True)
            step += 1
            continue
        optimizer.step()

        if step % log_every_n_steps == 0 or smoke:
            _log_step(step, "train", losses, extra=f", grad_norm={float(grad_norm):.4f}")

        if val_loader is not None and (smoke or (step > resume_step and step % int(training_cfg.get("eval_every_n_steps", 2000)) == 0)):
            model.eval()
            with torch.no_grad():
                val_totals = []
                for val_inputs, val_targets in val_loader:
                    val_target_expression = torch.as_tensor(val_targets.query_expression, dtype=torch.float32, device=device)
                    val_query_coords = torch.as_tensor(val_inputs.query_coords, dtype=torch.float32, device=device)
                    val_losses = compute_step_losses(
                        architecture_id, model, val_inputs, val_target_expression, val_query_coords,
                        gradient_weight=gradient_weight, k_neighbors=k_neighbors,
                    )
                    val_totals.append(float(val_losses["total"]))
                    if smoke:
                        break
                mean_val_loss = float(np.mean(val_totals)) if val_totals else float("nan")
                _log_step(step, "validation", {"total": mean_val_loss})
            model.train()

        step += 1
        if (not smoke) and step % checkpoint_every_n_steps == 0:
            checkpoint_module.save_checkpoint(
                model, config, gene_names, checkpoint_dir, step,
                extra_metadata={"n_skipped_nonfinite": n_skipped_nonfinite},
                keep_last=checkpoint_keep_last, optimizer=optimizer, rng=sample_rng,
            )

    if not smoke:
        checkpoint_module.save_checkpoint(
            model, config, gene_names, checkpoint_dir, step,
            extra_metadata={"n_skipped_nonfinite": n_skipped_nonfinite},
            keep_last=checkpoint_keep_last, optimizer=optimizer, rng=sample_rng,
        )

    elapsed = time.time() - start_time
    summary = {
        "ok": True, "smoke": smoke, "architecture": architecture_id, "final_step": step,
        "n_skipped_nonfinite": n_skipped_nonfinite, "elapsed_seconds": elapsed,
        "checkpoint_dir": str(checkpoint_dir),
    }
    print(f"training run finished: {summary}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true", help="Run exactly one training (and one validation) step, then exit.")
    args = parser.parse_args()
    run_training(args.config, smoke=args.smoke)


if __name__ == "__main__":
    main()
