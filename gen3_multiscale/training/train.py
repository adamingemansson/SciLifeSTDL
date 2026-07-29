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


def maybe_load_pretrained_conditioner_for_architecture4(
    model, config: dict, architecture_id: str, gene_names: list[str], smoke: bool,
) -> bool:
    """Adam's Step 6 audit #11 (confirmed real gap): "There is no real
    pipeline for fitting [Architecture 4's] residual basis... initialize
    Architecture 4 from that exact Architecture 3 conditioner and
    initially freeze the conditioner while training flow. Architecture 4
    therefore should not yet run concurrently from random initialization
    with Architectures 1-3." Loads a REAL, already-trained Architecture 3
    checkpoint directly onto `model.conditioner` (a full Architecture3
    instance by construction -- see architectures.py::Architecture4) --
    called AFTER synchronized-init loading, so it deliberately OVERWRITES
    whatever synchronized/shared initial weights the conditioner started
    with; `gene_basis.py::fit_gene_residual_basis` itself already needs
    THIS exact trained conditioner's own residuals to have been fit
    against in the first place (see `scripts/
    fit_architecture4_residual_basis.py`), so an untrained conditioner
    would silently mismatch the very basis Architecture 4 is required to
    load. Freezes every conditioner parameter afterward unless
    `model.params.freeze_conditioner_initially` is explicitly set false
    -- "initially stop gradients from the flow loss into the
    deterministic conditioner" already holds for the LOSS computation
    (architectures.py's own `.detach()` discipline); this additionally
    stops the OPTIMIZER from updating the conditioner's own weights at
    all while `total_steps` is spent on the flow apparatus, matching
    Adam's "initially freeze" instruction literally, not merely via
    gradient-flow detachment. Returns True if a pretrained conditioner
    was loaded (smoke runs are exempt, matching every other "not required
    for --smoke" gate in this trainer); False for any non-Architecture-4
    run."""
    if architecture_id != "4":
        return False
    checkpoint_dir = (config.get("required_fingerprints") or {}).get("architecture3_conditioner_checkpoint")
    if not checkpoint_dir:
        if smoke:
            return False
        raise ValueError(
            "Architecture 4 requires required_fingerprints.architecture3_conditioner_checkpoint -- "
            "a real, already-trained Architecture 3 checkpoint_dir. Train Architecture 3 to "
            "completion first (scripts/fit_architecture4_residual_basis.py then needs that exact "
            "checkpoint to fit a real gene-residual basis); Architecture 4 must never start "
            "training from a random or merely synchronized-init conditioner"
        )
    checkpoint_module.verify_gene_names(checkpoint_dir, gene_names)
    checkpoint_module.load_trainable_state(model.conditioner, checkpoint_dir)
    freeze = bool(((config.get("model") or {}).get("params") or {}).get("freeze_conditioner_initially", True))
    if freeze:
        for p in model.conditioner.parameters():
            p.requires_grad = False
    return True


def compute_training_gene_scale(train_samples: dict) -> np.ndarray:
    """Real, TRAINING-ONLY per-gene standardization scale for
    `spatial_gradient_loss`'s `per_gene_scale` -- Adam's Step 6 audit #9
    (confirmed real gap): "Compute spatial-gradient gene scales from
    training samples only and persist them; do not normalize
    independently using each query target." Without this, `losses.py`'s
    `spatial_gradient_loss` falls back to standardizing by EACH QUERY
    SET's own std (a documented simplification in that module, never
    meant to be the production path) -- a noisy, target-dependent scale
    that differs draw to draw and can leak information about the
    specific held-out queries into the loss's own normalization.

    Pooled per-gene std across every TRAINING sample's full, real
    expression matrix (`adata.X`) -- never touches validation/test data.
    A pure function of the training samples already loaded by preflight,
    so it is naturally reproducible across a resume as long as the same
    training data passed content-provenance verification (see
    `dataset_manifest.verify_content_provenance`)."""
    if not train_samples:
        raise ValueError("compute_training_gene_scale: no training samples given")
    pooled = []
    for sample in train_samples.values():
        X = sample.adata.X
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        pooled.append(np.asarray(X, dtype=np.float64))
    pooled_matrix = np.concatenate(pooled, axis=0)
    scale = pooled_matrix.std(axis=0)
    return np.clip(scale, 1e-6, None).astype(np.float32)


def save_gene_scale(scale: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.npy")
    np.save(tmp, np.asarray(scale, dtype=np.float32))
    os.replace(tmp, path)
    return path


def load_gene_scale(path: str | Path) -> np.ndarray:
    return np.load(path)


def deterministic_train_index_for_step(step: int, dataset_len: int, seed: int) -> int:
    """Maps a GLOBAL training step counter to a dataset index via a
    per-epoch deterministic permutation -- resume-EXACT: given the same
    (seed, step, dataset_len), this always returns the identical index,
    computed fresh or after a resume, with no separate cursor/permutation
    file to persist or go stale. Adam's Step 6 audit #5 (confirmed real
    gap): "Replace ordinary shuffled DataLoader resume with a global-
    step-deterministic sampler or persist epoch/permutation/cursor, so
    resume continues the exact sample sequence" -- a plain shuffled
    `torch.utils.data.DataLoader` reshuffles a FRESH, unsaved permutation
    every time its iterator is (re)created, so a "resumed" run previously
    continued from an entirely different point in an entirely different
    random ordering than the original run would have reached by the same
    step, not a genuine continuation of the same training trajectory.

    `epoch = step // dataset_len`; the permutation for that epoch is
    derived from a `torch.Generator` seeded by a value that depends on
    BOTH the run's own `seed` and `epoch` (not on process/wall-clock
    state), so every epoch gets its own distinct, but always
    reproducible, shuffle -- exactly what `shuffle=True` was providing
    before, minus the irreproducibility."""
    if dataset_len < 1:
        raise ValueError("deterministic_train_index_for_step: dataset_len must be positive")
    epoch, index_in_epoch = divmod(int(step), int(dataset_len))
    generator = torch.Generator().manual_seed((int(seed) * 1_000_003 + epoch) % (2**63))
    permutation = torch.randperm(dataset_len, generator=generator)
    return int(permutation[index_in_epoch].item())


def _validate_numeric_config(training_cfg: dict, loss_cfg: dict) -> None:
    """Adam's Step 6 audit #9's last bullet: "Validate all numeric
    configuration values." Fails closed (ValueError) on a config that
    would otherwise silently misbehave (a negative/zero learning rate
    that never learns, a negative weight decay, etc.) rather than
    discovering it hours into a real run."""
    checks = {
        "training.lr": (float(training_cfg.get("lr", 1e-4)), lambda v: v > 0, "must be positive"),
        "training.gradient_clip_val": (float(training_cfg.get("gradient_clip_val", 1.0)), lambda v: v > 0, "must be positive"),
        "training.total_steps": (int(training_cfg.get("total_steps", 1)), lambda v: v > 0, "must be positive"),
        "training.max_wall_clock_hours": (float(training_cfg.get("max_wall_clock_hours", 24.0)), lambda v: v > 0, "must be positive"),
        "training.log_every_n_steps": (int(training_cfg.get("log_every_n_steps", 50)), lambda v: v > 0, "must be positive"),
        "training.checkpoint_every_n_steps": (int(training_cfg.get("checkpoint_every_n_steps", 2000)), lambda v: v > 0, "must be positive"),
        "training.eval_every_n_steps": (int(training_cfg.get("eval_every_n_steps", 2000)), lambda v: v > 0, "must be positive"),
        "training.optimizer.weight_decay": (float((training_cfg.get("optimizer") or {}).get("weight_decay", 0.01)), lambda v: v >= 0, "must be non-negative"),
        "training.optimizer.eps": (float((training_cfg.get("optimizer") or {}).get("eps", 1e-8)), lambda v: v > 0, "must be positive"),
        "loss.gradient_weight": (float(loss_cfg.get("gradient_weight", 0.05)), lambda v: v >= 0, "must be non-negative"),
        "loss.k_neighbors": (int(loss_cfg.get("k_neighbors", 6)), lambda v: v > 0, "must be positive"),
        "loss.flow_weight": (float(loss_cfg.get("flow_weight", 1.0)), lambda v: v >= 0, "must be non-negative"),
    }
    for name, (value, predicate, message) in checks.items():
        if not predicate(value):
            raise ValueError(f"config field {name}={value!r} is invalid: {message}")
    betas = (training_cfg.get("optimizer") or {}).get("betas", [0.9, 0.999])
    if len(betas) != 2 or not all(0.0 <= float(b) < 1.0 for b in betas):
        raise ValueError(f"config field training.optimizer.betas={betas!r} must be exactly two values in [0, 1)")


def compute_step_losses(architecture_id: str, model, inputs, target_expression: torch.Tensor,
                         query_coords: torch.Tensor, gradient_weight: float, k_neighbors: int,
                         flow_weight: float = 1.0, per_gene_scale: torch.Tensor | None = None,
                         flow_generator: torch.Generator | None = None) -> dict:
    """Architecture-generic where possible: Architectures 1/2/3 share one
    deterministic reconstruction objective (models/losses.py); Architecture
    4 additionally adds its stopped-gradient flow-matching loss, computed
    from a SINGLE conditioner pass (model.compute_losses) so the
    reconstruction and flow losses agree on the same dropout mask (3rd
    Codex re-audit finding, CONTRACT.md). `flow_weight` was previously
    hardcoded at every call site (Adam's Step 6 audit #9, confirmed) --
    now always threaded through from `loss.flow_weight` in the resolved
    config. `flow_generator`, when given, makes the flow loss's own
    random t/x0 draw reproducible (used for validation logging only --
    see `compute_deterministic_reconstruction_losses` for the metric
    actually used for model selection, audit #8)."""
    if architecture_id == "4":
        out = model.compute_losses(inputs, target_expression, generator=flow_generator)
        recon = combined_reconstruction_loss(
            out["expression"], target_expression, query_coords,
            gradient_weight=gradient_weight, k_neighbors=k_neighbors, per_gene_scale=per_gene_scale,
        )
        total = recon["total"] + flow_weight * out["flow_loss"]
        return {"total": total, "primary": recon["primary"], "gradient": recon["gradient"], "flow_loss": out["flow_loss"]}
    out = model(inputs)
    return combined_reconstruction_loss(
        out["expression"], target_expression, query_coords,
        gradient_weight=gradient_weight, k_neighbors=k_neighbors, per_gene_scale=per_gene_scale,
    )


def compute_deterministic_reconstruction_losses(
    model, inputs, target_expression: torch.Tensor, query_coords: torch.Tensor,
    gradient_weight: float, k_neighbors: int, per_gene_scale: torch.Tensor | None = None,
) -> dict:
    """Architecture-GENERIC deterministic reconstruction objective, used
    for VALIDATION/model-selection (Adam's Step 6 audit #8: "Use the
    common deterministic reconstruction metrics for model selection; log
    flow loss separately with a fixed generator if desired"). `model(inputs)`
    works identically for all four architectures -- Architecture4.forward()
    is defined to return exactly its conditioner's own output (the same
    shape Architectures 1-3 return), and never touches the stochastic flow
    apparatus at all. This is the ONLY loss function whose value ever
    drives a validation-based decision (best-checkpoint selection); the
    flow loss, for Architecture 4, is logged separately (see
    `compute_step_losses`' `flow_generator`) and never mixed into it."""
    out = model(inputs)
    return combined_reconstruction_loss(
        out["expression"], target_expression, query_coords,
        gradient_weight=gradient_weight, k_neighbors=k_neighbors, per_gene_scale=per_gene_scale,
    )


def dataset_manifest_fingerprint(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def config_fingerprint(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode("utf-8")).hexdigest()


# Purely operational/scheduling training fields that a legitimate resume
# workflow ("bump total_steps and keep training", "raise checkpoint
# cadence", ...) must be free to change without verify_resume_consistency
# refusing to continue -- everything else in `training` (seed, lr,
# optimizer hyperparameters, synchronized_init_dir, device, ...) still
# counts toward the run's SCIENTIFIC identity and must stay fixed.
_RESUME_EXCLUDED_TRAINING_FIELDS = frozenset({
    "total_steps", "checkpoint_every_n_steps", "log_every_n_steps", "eval_every_n_steps",
    "max_wall_clock_hours", "checkpoint_dir", "checkpoint_keep_last",
})


def _config_identity_fingerprint(config: dict) -> str:
    """Same content as `config_fingerprint`, minus
    `_RESUME_EXCLUDED_TRAINING_FIELDS` -- the fingerprint
    `verify_resume_consistency` actually compares. Adam's Step 6 audit #5:
    "Verify the existing run manifest/config/dataset/cache/LongNet/init/
    basis fingerprints before loading anything. Refuse changed configs
    or artifacts" -- but a resumed run legitimately needs to be able to
    ask for MORE steps, a different checkpoint cadence, or a raised
    wall-clock budget without that being treated as "a changed config"
    in the sense this check is meant to catch."""
    identity_config = json.loads(json.dumps(config, default=str))  # deep copy, same serialization the hash itself uses
    training_section = dict(identity_config.get("training") or {})
    for field in _RESUME_EXCLUDED_TRAINING_FIELDS:
        training_section.pop(field, None)
    identity_config["training"] = training_section
    return hashlib.sha256(json.dumps(identity_config, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_run_manifest(
    *, config: dict, config_path: str, dataset_manifest: dict, gene_names: list[str], seed: int,
    preflight_report: dict, train_schedule, val_schedule, architecture_id: str,
    checkpoint_dir: Path, synchronized_init_manifest_path: Path | None,
    gigapath_checkpoint_sha256: str | None = None, gene_residual_basis_gene_names_hash: str | None = None,
    gene_scale_sha256: str | None = None,
) -> dict:
    """Requirement #8: one artifact binding dataset, gene panel, split,
    mask, cache, model/checkpoint, configuration, and seed fingerprints
    together -- so a later reader can tell EXACTLY what this run trained
    on without re-deriving it from scattered files.

    `synchronized_init_manifest_sha256`/`gigapath_checkpoint_sha256`/
    `gene_residual_basis_gene_names_hash`/`gene_scale_sha256` were added
    for Adam's Step 6 audit #5 ("Verify the existing run manifest/config/
    dataset/cache/LongNet/init/basis fingerprints before loading
    anything... Refuse changed configs or artifacts") -- the synchronized-
    init MANIFEST's own file content is hashed (not just its path, which
    could stay the same while the file underneath it changes), so
    `_verify_resume_consistency` can actually detect a swapped-out
    initialization/checkpoint/basis, not merely a swapped-out path
    string."""
    synchronized_init_manifest_sha256 = (
        file_sha256(synchronized_init_manifest_path)
        if synchronized_init_manifest_path is not None and Path(synchronized_init_manifest_path).is_file()
        else None
    )
    return {
        "version": 2,
        "kind": "gen3_step6_run_manifest",
        "config_path": str(config_path),
        "config_fingerprint": config_fingerprint(config),
        "config_identity_fingerprint": _config_identity_fingerprint(config),
        "seed": int(seed),
        "dataset_manifest_fingerprint": dataset_manifest_fingerprint(dataset_manifest),
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
        "synchronized_init_manifest_sha256": synchronized_init_manifest_sha256,
        "gigapath_checkpoint_sha256": gigapath_checkpoint_sha256,
        "gene_residual_basis_gene_names_hash": gene_residual_basis_gene_names_hash,
        "gene_scale_sha256": gene_scale_sha256,
    }


_RESUME_CONSISTENCY_FIELDS = (
    "config_identity_fingerprint", "dataset_manifest_fingerprint", "gene_panel_hash", "model_architecture",
    "synchronized_init_manifest_sha256", "gigapath_checkpoint_sha256",
    "gene_residual_basis_gene_names_hash", "gene_scale_sha256",
)


def verify_resume_consistency(old_run_manifest: dict, new_run_manifest: dict) -> None:
    """Adam's Step 6 audit #5: "Verify the existing run manifest/config/
    dataset/cache/LongNet/init/basis fingerprints before loading anything.
    Refuse changed configs or artifacts." Compares every identity-bearing
    field a resumed run must NOT silently change; raises (fail-closed) on
    the first mismatch, naming the field, rather than silently continuing
    training under a changed config, dataset, gene panel, architecture,
    synchronized initialization, GigaPath checkpoint, or gene-residual
    basis. Called BEFORE any checkpoint state is loaded and BEFORE the new
    run_manifest.json is written over the old one -- a caller must not
    overwrite the evidence of a real config-drift bug before it has been
    checked."""
    for field in _RESUME_CONSISTENCY_FIELDS:
        old_value = old_run_manifest.get(field)
        new_value = new_run_manifest.get(field)
        if old_value != new_value:
            raise ValueError(
                f"resume refused: {field} changed since the last checkpoint at {old_run_manifest.get('checkpoint_dir')} "
                f"(was {old_value!r}, now {new_value!r}) -- this checkpoint_dir's prior run used different "
                "config/dataset/gene-panel/architecture/synchronized-init/checkpoint/basis identity than the "
                "current invocation; resume from a fresh checkpoint_dir if this is a deliberate change"
            )


def _save_json_atomic(obj, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)
    return path


def save_run_manifest(run_manifest: dict, path: str | Path) -> Path:
    return _save_json_atomic(run_manifest, path)


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
    # Requirement #5: `train_dataset` is indexed DIRECTLY by a
    # deterministic per-step index (deterministic_train_index_for_step,
    # below) rather than iterated through a shuffled DataLoader --  a
    # plain `shuffle=True` DataLoader reshuffles a fresh, UNSAVED
    # permutation every epoch, so a resumed run previously continued from
    # a different point in a different random ordering than the original
    # run would have reached by the same step.
    train_dataset = Gen3SpatialFieldDataset(
        dataset_manifest, train_samples, train_schedule, strata, novae_enabled=novae_enabled,
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

    # Audit #11: Architecture 4 must never train from a random or merely
    # synchronized-init conditioner -- overwrite it with a REAL, already-
    # trained Architecture 3 checkpoint (and freeze it) before anything
    # else touches the model.
    maybe_load_pretrained_conditioner_for_architecture4(model, config, architecture_id, gene_names, smoke)

    # Requirement #9: real, TRAINING-ONLY per-gene standardization scale
    # for the spatial-gradient loss -- computed once here (train_samples
    # are already loaded by preflight) and persisted, never re-derived
    # from each query target at loss-computation time.
    gene_scale = compute_training_gene_scale(train_samples)
    gene_scale_path = save_gene_scale(gene_scale, checkpoint_dir / "gene_scale.npy")
    gene_scale_sha256 = hashlib.sha256(np.ascontiguousarray(gene_scale).tobytes()).hexdigest()
    gene_scale_tensor = torch.as_tensor(gene_scale, dtype=torch.float32, device=device)

    gene_residual_basis_gene_names_hash = gene_basis.gene_names_hash if gene_basis is not None else None

    # Requirement #5: build the NEW run manifest in memory (never yet
    # written) and, if a PRIOR run_manifest.json already exists in this
    # checkpoint_dir, verify every identity-bearing fingerprint agrees
    # with it BEFORE loading any checkpoint state and BEFORE overwriting
    # that prior manifest -- "Refuse changed configs or artifacts... Do
    # not overwrite the old run manifest before verifying it."
    run_manifest = build_run_manifest(
        config=config, config_path=str(config_path), dataset_manifest=dataset_manifest, gene_names=gene_names,
        seed=seed, preflight_report=preflight_report, train_schedule=train_schedule, val_schedule=val_schedule,
        architecture_id=architecture_id, checkpoint_dir=checkpoint_dir,
        synchronized_init_manifest_path=synchronized_init_manifest_path,
        gigapath_checkpoint_sha256=gigapath_checkpoint_sha256,
        gene_residual_basis_gene_names_hash=gene_residual_basis_gene_names_hash,
        gene_scale_sha256=gene_scale_sha256,
    )
    existing_run_manifest_path = checkpoint_dir / "run_manifest.json"
    if existing_run_manifest_path.is_file():
        old_run_manifest = json.loads(existing_run_manifest_path.read_text())
        verify_resume_consistency(old_run_manifest, run_manifest)

    resume_step = 0
    optimizer_cfg = training_cfg.get("optimizer") or {}
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training_cfg.get("lr", 1e-4)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.01)),
        betas=tuple(float(b) for b in optimizer_cfg.get("betas", [0.9, 0.999])),
        eps=float(optimizer_cfg.get("eps", 1e-8)),
    )
    training_state = checkpoint_module.load_training_state(checkpoint_dir)
    if training_state.get("step", 0) > 0:
        checkpoint_module.verify_gene_names(checkpoint_dir, gene_names)
        checkpoint_module.load_trainable_state(model, checkpoint_dir)
        # Requirement #5: optimizer/RNG state is now REQUIRED to resume a
        # Gen3 checkpoint, not a silent best-effort no-op -- a "resumed"
        # run without it would keep the trained weights but reset
        # optimizer momentum/variance to zero and every RNG stream from
        # scratch, which is a warm restart with different optimization
        # dynamics, not a genuine continuation.
        if not checkpoint_module.load_optimizer_and_rng_state(optimizer, checkpoint_dir, rng=sample_rng):
            raise ValueError(
                f"checkpoint at {checkpoint_dir} has training_state.json (step "
                f"{training_state['step']}) but no optimizer_rng_state.pt -- refusing to resume "
                "with reset optimizer momentum/RNG state; this checkpoint looks incomplete"
            )
        resume_step = int(training_state["step"])
        print(f"resumed from checkpoint at step {resume_step}", flush=True)

    save_run_manifest(run_manifest, checkpoint_dir / "run_manifest.json")

    loss_cfg = config.get("loss") or {}
    _validate_numeric_config(training_cfg, loss_cfg)
    gradient_weight = float(loss_cfg.get("gradient_weight", 0.05))
    k_neighbors = int(loss_cfg.get("k_neighbors", 6))
    flow_weight = float(loss_cfg.get("flow_weight", 1.0))
    gradient_clip_val = float(training_cfg.get("gradient_clip_val", 1.0))
    # Requirement #2 (confirmed real gap): total_steps is the run's
    # ABSOLUTE target step count, not "additional steps after every
    # resume" -- a checkpoint saved at step 20_000 with total_steps=
    # 100_000 stops at 100_000, never 120_000. Smoke mode is the one
    # exception: it always runs exactly ONE further step regardless of
    # resume_step, since it is a one-step sanity check, not a real run.
    total_steps = int(training_cfg.get("total_steps", 1))
    step_target = resume_step + 1 if smoke else total_steps
    max_wall_clock_hours = float(training_cfg.get("max_wall_clock_hours", 24.0))
    log_every_n_steps = max(1, int(training_cfg.get("log_every_n_steps", 50)))
    checkpoint_every_n_steps = max(1, int(training_cfg.get("checkpoint_every_n_steps", 2000)))
    checkpoint_keep_last = int(training_cfg.get("checkpoint_keep_last", 2))
    eval_every_n_steps = int(training_cfg.get("eval_every_n_steps", 2000))

    validation_history_path = checkpoint_dir / "validation_history.json"
    validation_history = (
        json.loads(validation_history_path.read_text()) if validation_history_path.is_file() else []
    )
    best_val_loss = min((entry["total"] for entry in validation_history), default=float("inf"))
    # Fixed, run-stable generator for Architecture 4's validation-only
    # flow-loss LOGGING (requirement #8: "log flow loss separately with a
    # fixed generator") -- never used for the selection metric itself.
    flow_val_generator = torch.Generator(device=device).manual_seed(seed)

    def _run_validation(current_step: int) -> dict | None:
        if val_loader is None:
            return None
        model.eval()
        with torch.no_grad():
            val_totals: list[float] = []
            val_flow_losses: list[float] = []
            for val_inputs, val_targets in val_loader:
                val_target_expression = torch.as_tensor(val_targets.query_expression, dtype=torch.float32, device=device)
                val_query_coords = torch.as_tensor(val_inputs.query_coords, dtype=torch.float32, device=device)
                # Requirement #8: model selection ALWAYS uses the common,
                # deterministic reconstruction objective -- never
                # Architecture 4's stochastic flow loss, which used to be
                # silently mixed into "total" during validation too.
                det_losses = compute_deterministic_reconstruction_losses(
                    model, val_inputs, val_target_expression, val_query_coords,
                    gradient_weight=gradient_weight, k_neighbors=k_neighbors, per_gene_scale=gene_scale_tensor,
                )
                val_totals.append(float(det_losses["total"]))
                if architecture_id == "4":
                    flow_loss = model.compute_flow_matching_loss(
                        val_inputs, val_target_expression, generator=flow_val_generator,
                    )
                    val_flow_losses.append(float(flow_loss))
                if smoke:
                    break
            mean_val_loss = float(np.mean(val_totals)) if val_totals else float("nan")
            # Requirement #8: reject a non-finite validation result rather
            # than silently logging "nan" and moving on -- a NaN/Inf
            # validation loss means the model (or the held-out data) is
            # broken in a way that must stop the run, exactly like a
            # non-finite TRAINING loss now does (requirement #3).
            if not np.isfinite(mean_val_loss):
                raise RuntimeError(
                    f"[step {current_step}] validation produced a non-finite mean total loss "
                    f"({mean_val_loss!r}) -- refusing to continue or select a checkpoint on this result"
                )
            entry = {"step": int(current_step), "total": mean_val_loss}
            if val_flow_losses:
                entry["flow_loss_mean_fixed_generator"] = float(np.mean(val_flow_losses))
            _log_step(current_step, "validation", {"total": mean_val_loss})
        model.train()
        return entry

    model.train()
    n_skipped_nonfinite = 0
    step = resume_step
    # Default completion_reason: reaching step_target without stopping
    # early for any other reason. Only ever overridden below (to
    # "wall_clock_limit_reached") when the loop actually breaks early --
    # deliberately NOT relying on a while/else construct here, since
    # Python's while-else also fires on ZERO iterations (step already >=
    # step_target at entry, e.g. resuming a checkpoint that had already
    # reached total_steps in a prior run), which this label already
    # correctly describes without any special-casing.
    completion_reason = "completed_smoke_step" if smoke else "completed_total_steps"
    start_time = time.time()
    dataset_len = len(train_dataset)
    while step < step_target:
        elapsed_hours = (time.time() - start_time) / 3600.0
        # Requirement #1: honor the configured wall-clock budget -- a
        # real run must stop, save a final checkpoint, and record WHY it
        # stopped, rather than running to total_steps regardless of how
        # long that actually takes.
        if elapsed_hours >= max_wall_clock_hours:
            completion_reason = "wall_clock_limit_reached"
            break

        idx = deterministic_train_index_for_step(step, dataset_len, seed)
        inputs, targets = train_dataset[idx]

        target_expression = torch.as_tensor(targets.query_expression, dtype=torch.float32, device=device)
        query_coords = torch.as_tensor(inputs.query_coords, dtype=torch.float32, device=device)

        optimizer.zero_grad(set_to_none=True)
        losses = compute_step_losses(
            architecture_id, model, inputs, target_expression, query_coords,
            gradient_weight=gradient_weight, k_neighbors=k_neighbors, flow_weight=flow_weight,
            per_gene_scale=gene_scale_tensor,
        )
        total_loss = losses["total"]
        # Requirement #3 (confirmed real gap): a NaN/Inf loss or gradient
        # must FAIL the run, not silently skip the step and eventually
        # return ok: true -- a run that "completes" while quietly
        # skipping every unstable step is a false success, not a real
        # one.
        if not torch.isfinite(total_loss):
            raise RuntimeError(
                f"[step {step}] non-finite total loss ({float(total_loss.detach())!r}) -- "
                "failing the run rather than skipping this step"
            )
        pre_step_params = (
            {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad} if smoke else None
        )
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(
                f"[step {step}] non-finite gradient norm ({float(grad_norm)!r}) -- "
                "failing the run rather than skipping this step"
            )
        optimizer.step()

        # Requirement #4: a real learning gate for smoke -- a finite
        # gradient norm alone does not prove the model is actually
        # trainable (a disconnected graph, a frozen backbone with zero
        # trainable parameters, or lr=0 could all still produce a finite,
        # zero grad_norm and a spuriously "passing" smoke run). Smoke
        # must additionally see a strictly POSITIVE gradient norm and at
        # least one trainable parameter that ACTUALLY changed value.
        if smoke:
            if not (grad_norm > 0):
                raise RuntimeError(
                    f"[step {step}] smoke learning gate failed: gradient norm is {float(grad_norm)!r}, "
                    "not strictly positive -- this model is not receiving a real training signal"
                )
            changed = any(
                not torch.equal(p.detach(), pre_step_params[name])
                for name, p in model.named_parameters() if p.requires_grad
            )
            if not changed:
                raise RuntimeError(
                    f"[step {step}] smoke learning gate failed: optimizer.step() produced zero "
                    "measurable change in any trainable parameter -- this model is not learning"
                )

        if step % log_every_n_steps == 0 or smoke:
            _log_step(step, "train", losses, extra=f", grad_norm={float(grad_norm):.4f}")

        if val_loader is not None and (smoke or (step > resume_step and step % eval_every_n_steps == 0)):
            entry = _run_validation(step)
            if entry is not None:
                validation_history.append(entry)
                _save_json_atomic(validation_history, validation_history_path)
                if entry["total"] < best_val_loss:
                    best_val_loss = entry["total"]
                    if not smoke:
                        best_dir = checkpoint_dir / "best"
                        checkpoint_module.save_trainable_state(model, best_dir)
                        _save_json_atomic(
                            {"step": int(step), "total": entry["total"]}, best_dir / "best_info.json",
                        )

        step += 1
        if (not smoke) and step % checkpoint_every_n_steps == 0:
            checkpoint_module.save_checkpoint(
                model, config, gene_names, checkpoint_dir, step,
                extra_metadata={"n_skipped_nonfinite": n_skipped_nonfinite, "completion_reason": "in_progress"},
                keep_last=checkpoint_keep_last, optimizer=optimizer, rng=sample_rng,
            )

    if not smoke and step > resume_step:
        checkpoint_module.save_checkpoint(
            model, config, gene_names, checkpoint_dir, step,
            extra_metadata={"n_skipped_nonfinite": n_skipped_nonfinite, "completion_reason": completion_reason},
            keep_last=checkpoint_keep_last, optimizer=optimizer, rng=sample_rng,
        )

    elapsed = time.time() - start_time
    summary = {
        "ok": True, "smoke": smoke, "architecture": architecture_id, "final_step": step,
        "n_skipped_nonfinite": n_skipped_nonfinite, "elapsed_seconds": elapsed,
        "completion_reason": completion_reason, "checkpoint_dir": str(checkpoint_dir),
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
