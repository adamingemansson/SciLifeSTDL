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
import platform
import random
import shutil
import subprocess
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


def maybe_load_gene_basis(
    config: dict, gene_names: list[str], dataset_manifest: dict | None = None, *, smoke: bool = False,
):
    """Architecture 4 only: `required_fingerprints.gene_basis` must point
    at an already-fit, saved `GeneResidualBasis`
    (`models/gene_basis.py::save_gene_residual_basis`) -- fit OFFLINE, on
    TRAINING-split residuals only, per Architecture4's own docstring.
    This trainer never fits one itself (it would need a trained
    conditioner's own residuals to fit against in the first place).

    Codex's re-audit of commit 90f853e, launch blocker #3: "Require the
    Architecture 4 basis provenance sidecar for every non-smoke run.
    Require and exactly validate dataset fingerprint, gene-panel hash,
    canonical Architecture 3 checkpoint identity, config identity and
    mask-schedule report. Missing fields must fail." The PRIOR version
    (Adam's Step 6 audit #4 of commit a32051b) made the sidecar OPTIONAL
    and skipped any field it happened to be missing -- a numerically
    valid basis with the correct gene order but completely unknown
    origin (no sidecar, or a sidecar missing every field) passed every
    check. Fixed here: the sidecar is now MANDATORY whenever `smoke` is
    false (a plain `--smoke` stays construction-only and exempt, matching
    every other "not required for --smoke" gate in this trainer), and
    every field below is REQUIRED to be present -- a missing field now
    raises exactly like a mismatched one. `dataset_manifest_fingerprint`
    and `gene_panel_hash` are compared for EXACT equality against the
    current run. The conditioner checkpoint's identity is now resolved
    via `checkpoint.py::resolve_checkpoint_identity` (never a raw hash of
    `checkpoint_dir`'s root convenience-mirror file -- the same
    root-vs-resolved-bundle bug this re-audit flagged in
    `maybe_load_pretrained_conditioner_for_architecture4`). `config
    identity` (the sidecar's own `architecture3_config_fingerprint`) and
    `mask-schedule report` (`mask_schedule_reports`) have no single
    "current" value to compare against from HERE (they describe the
    Architecture 3 run the basis was fit from, not this Architecture 4
    run) -- both are required to be PRESENT and non-empty (fail-closed on
    a basis whose provenance never actually recorded them), which is the
    literal "missing fields must fail" requirement; their CONTENT is not
    asserted equal to anything, since there is nothing in an Architecture
    4 run for them to equal. Gene panel is additionally checked via
    `verify_gene_residual_basis`'s own gene-name-hash comparison, always
    (smoke included).

    Mask-schedule identity is deliberately NOT bound to an exact,
    reproducible fingerprint: the basis is fit against POOLED residuals
    from many (`n_masks_per_sample`) independent mask draws, not one
    specific realized schedule, so pinning it to an exact mask-schedule
    fingerprint would incorrectly reject a legitimate basis whenever the
    training mask schedule is (deliberately) re-diversified -- this
    matches the same reasoning documented for `train.py`'s own
    `_RESUME_CONSISTENCY_FIELDS`, which also excludes mask-schedule
    fingerprints from its own identity comparisons for scheduling
    fields."""
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

    provenance_path = Path(f"{path}.provenance.json")
    if not provenance_path.is_file():
        if smoke:
            return basis, gene_names
        raise ValueError(
            f"gene residual basis at {path} has no provenance sidecar at {provenance_path} -- Codex's "
            "re-audit of commit 90f853e, launch blocker #3: 'Require the Architecture 4 basis "
            "provenance sidecar for every non-smoke run.' Fit the basis via "
            "scripts/fit_architecture4_residual_basis.py (which always writes one) before training "
            "Architecture 4 for real; a basis with no recorded origin must never be used"
        )
    provenance = json.loads(provenance_path.read_text())

    def _require_field(name: str):
        value = provenance.get(name)
        if value in (None, "", []):
            raise ValueError(
                f"gene residual basis provenance sidecar {provenance_path} is missing required field "
                f"{name!r} -- Codex's re-audit of commit 90f853e, launch blocker #3: 'missing fields "
                "must fail.' Refusing to use a basis whose provenance is incomplete"
            )
        return value

    if smoke:
        return basis, gene_names

    if dataset_manifest is None:
        raise ValueError(
            "maybe_load_gene_basis: dataset_manifest is required to validate the basis provenance "
            "sidecar for a non-smoke Architecture 4 run"
        )
    expected_fp = dataset_manifest_fingerprint(dataset_manifest)
    recorded_fp = _require_field("dataset_manifest_fingerprint")
    if recorded_fp != expected_fp:
        raise ValueError(
            f"gene residual basis at {path} was fit against a dataset manifest (fingerprint "
            f"{recorded_fp!r}) that does not match this run's dataset manifest (fingerprint "
            f"{expected_fp!r}) -- refusing to use a basis fit on different data"
        )

    expected_gene_panel_hash = gene_panel_hash(gene_names)
    recorded_gene_panel_hash = _require_field("gene_panel_hash")
    if recorded_gene_panel_hash != expected_gene_panel_hash:
        raise ValueError(
            f"gene residual basis at {path} was fit against a gene panel (hash {recorded_gene_panel_hash!r}) "
            f"that does not match this run's gene panel (hash {expected_gene_panel_hash!r}) -- refusing to "
            "use a basis fit on a different gene panel"
        )

    _require_field("architecture3_config_fingerprint")
    _require_field("mask_schedule_reports")

    conditioner_checkpoint_dir = (config.get("required_fingerprints") or {}).get(
        "architecture3_conditioner_checkpoint",
    )
    if not conditioner_checkpoint_dir:
        raise ValueError(
            f"gene residual basis at {path} has a provenance sidecar recording its Architecture 3 "
            "conditioner checkpoint identity, but this run's required_fingerprints."
            "architecture3_conditioner_checkpoint is not set -- cannot validate the basis was fit "
            "against the SAME conditioner this run is about to load"
        )
    recorded_checkpoint_sha256 = _require_field("architecture3_checkpoint_trainable_weights_sha256")
    actual_identity = checkpoint_module.resolve_checkpoint_identity(conditioner_checkpoint_dir)
    if actual_identity.weights_sha256 != recorded_checkpoint_sha256:
        raise ValueError(
            f"gene residual basis at {path} was fit against a different Architecture 3 checkpoint "
            f"(trainable_weights.pt sha256 {recorded_checkpoint_sha256!r}) than the one configured for "
            f"this run (sha256 {actual_identity.weights_sha256!r} at {conditioner_checkpoint_dir}) -- "
            "refusing to use a basis fit against a different conditioner's residuals"
        )
    return basis, gene_names


def maybe_load_pretrained_conditioner_for_architecture4(
    model, config: dict, architecture_id: str, gene_names: list[str], smoke: bool,
    *, require_for_smoke: bool = False,
) -> dict:
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
    gradient-flow detachment.

    Returns a dict identifying exactly which checkpoint was loaded (or
    that none was) -- Adam's Step 6 audit #4 of commit a32051b ("bind
    resume/evaluation to... exact Architecture 3 conditioner weights/
    step"): `run_training` records `checkpoint_sha256`/`checkpoint_step`
    into the run manifest so a resume or evaluation can be verified
    against the EXACT trained conditioner a run started from, not merely
    "some checkpoint was present at this path."

    `require_for_smoke` (Adam's Step 6 audit #8: "Separate a clearly
    named construction-only Architecture 4 smoke from the real staged
    smoke using the trained conditioner"): a plain `--smoke` run is
    CONSTRUCTION-ONLY by default (exempt from needing a real conditioner
    checkpoint, matching every other "not required for --smoke" gate in
    this trainer) unless a caller explicitly asks for the real staged
    smoke (`run_training(..., smoke=True, staged_smoke=True)`), in which
    case this behaves exactly like a non-smoke run and raises if no real
    checkpoint is configured."""
    if architecture_id != "4":
        return {"loaded": False, "checkpoint_dir": None, "checkpoint_sha256": None, "checkpoint_step": None}
    checkpoint_dir = (config.get("required_fingerprints") or {}).get("architecture3_conditioner_checkpoint")
    if not checkpoint_dir:
        if smoke and not require_for_smoke:
            return {"loaded": False, "checkpoint_dir": None, "checkpoint_sha256": None, "checkpoint_step": None}
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
    # Codex re-audit of commit 90f853e, launch blocker #2: weights are
    # loaded from the VERIFIED, resolved bundle (via load_trainable_state
    # above, which itself resolves through checkpoint.py's transactional
    # pointer) but the identity previously recorded here was hashed from
    # `checkpoint_dir`'s ROOT convenience mirror directly -- a crash
    # between the mirror refresh and the real bundle write could make
    # those represent DIFFERENT model states, so the recorded identity
    # would not actually describe what was just loaded onto the model.
    # `resolve_checkpoint_identity` computes `weights_sha256` from the
    # SAME resolved/verified path `load_trainable_state` just read from.
    identity = checkpoint_module.resolve_checkpoint_identity(checkpoint_dir)
    return {
        "loaded": True, "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_sha256": identity.weights_sha256, "checkpoint_step": identity.step,
    }


def build_model_for_inference(
    config: dict, *, gene_names: list[str], device: torch.device,
    checkpoint_dir: str | Path | None = None, smoke: bool = False, staged_smoke: bool = False,
    dataset_manifest: dict | None = None,
) -> tuple[torch.nn.Module, dict]:
    """The ONE real model-reconstruction pipeline -- construct the
    architecture (with a real `FrozenGigaPathSlideEncoder` when
    `use_global_slide` needs one), load its verified synchronized
    initialization when configured, load+freeze Architecture 4's exact
    Architecture 3 conditioner, then optionally load `checkpoint_dir`'s
    trainable weights on top.

    Adam's Step 6 audit #2 of commit a32051b: "Create one shared
    inference/model-reconstruction function used by trainer, evaluator,
    overfit gate, and residual-basis fitter." Before this function
    existed, `train.py::run_training`, `gen3_evaluator.py::
    _load_model_for_evaluation`, `step6_overfit_test.py::run_overfit_gate`,
    and `fit_architecture4_residual_basis.py::fit_and_save_architecture4_basis`
    each independently duplicated this construction -- and three of the
    four (evaluator, overfit gate, basis fitter) silently OMITTED the
    Architecture-4-conditioner step, meaning Architecture 4 evaluation/
    overfit-testing loaded trainable weights onto a conditioner that was
    never correctly loaded/frozen from the real Architecture 3 checkpoint
    the way training did. All four now call this one function.

    `checkpoint_dir=None` returns a freshly (synchronized-init +
    Architecture-4-conditioner) initialized model with no trainable-
    weight checkpoint loaded on top -- used for the overfit gate's
    "before training" baseline. A real `checkpoint_dir` additionally
    verifies gene names and loads that checkpoint's trainable weights,
    failing closed on a mismatched gene panel or incomplete checkpoint,
    exactly as every other checkpoint load in this package."""
    architecture_id = str((config.get("model") or {}).get("architecture", ""))
    training_cfg = config.get("training") or {}
    data_cfg = config.get("data") or {}
    n_genes = len(gene_names)
    gex_feature_dim = int(data_cfg.get("gex_feature_dim", 128))
    seed = int(training_cfg.get("seed", 0))

    slide_encoder, gigapath_checkpoint_sha256 = maybe_build_slide_encoder(config)
    # staged_smoke behaves like a non-smoke run for provenance purposes --
    # its whole point is to validate the real, provenanced artifacts, not
    # bypass that validation the way a construction-only smoke may.
    gene_basis, resolved_gene_names = maybe_load_gene_basis(
        config, gene_names, dataset_manifest=dataset_manifest, smoke=smoke and not staged_smoke,
    )

    torch.manual_seed(seed)  # re-seed immediately before construction -- shared init discipline
    model = model_factory.build_architecture(
        config, n_genes=n_genes, gex_feature_dim=gex_feature_dim, gene_basis=gene_basis,
        gene_names=resolved_gene_names, slide_encoder=slide_encoder,
        gigapath_checkpoint_sha256=gigapath_checkpoint_sha256, seed=seed,
    ).to(device)

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
            "(persist_four_architecture_initializations); refusing to build a model from four "
            "independently-random starting points"
        )

    conditioner_info = maybe_load_pretrained_conditioner_for_architecture4(
        model, config, architecture_id, gene_names, smoke, require_for_smoke=staged_smoke,
    )

    if checkpoint_dir is not None:
        checkpoint_module.verify_gene_names(checkpoint_dir, gene_names)
        checkpoint_module.load_trainable_state(model, checkpoint_dir)

    info = {
        "architecture_id": architecture_id,
        "gene_basis": gene_basis,
        "gigapath_checkpoint_sha256": gigapath_checkpoint_sha256,
        "synchronized_init_manifest_path": synchronized_init_manifest_path,
        "architecture3_conditioner": conditioner_info,
    }
    return model, info


def predict_for_metrics(
    architecture_id: str, model, inputs, *, generator: torch.Generator | None = None,
) -> dict:
    """The metric-basis prediction for `inputs` -- Adam's Step 6 audit #1
    of commit a32051b: "Architecture 4 validation/evaluation/overfit must
    evaluate a deterministic fixed-seed predictive mean from
    `sample_predictive_distribution`, not `forward()`'s frozen
    conditioner. Log conditioner-only metrics separately." Architectures
    1-3's `forward()` already IS the real model, so their `expression`
    output is used directly and there is no separate "conditioner-only"
    number to report.

    Returns `{"expression": <the metric-basis prediction>}` for every
    architecture, plus (Architecture 4 only) `"conditioner_only_expression"`
    (the frozen conditioner's own, secondary, mean -- never the reported
    headline metric) and `"predictive_std"` (per-query sampled-residual
    uncertainty). `generator`, when given, makes Architecture 4's
    stochastic sampling reproducible -- required for exact resume/
    evaluation consistency (same audit item)."""
    if architecture_id == "4":
        conditioner_out = model(inputs)
        predictive = model.sample_predictive_distribution(inputs, generator=generator)
        return {
            "expression": predictive["predictive_mean"],
            "conditioner_only_expression": conditioner_out["expression"],
            "predictive_std": predictive["predictive_std"],
        }
    out = model(inputs)
    return {"expression": out["expression"]}


def compute_training_gene_scale(train_samples: dict, *, chunk_size: int = 512) -> np.ndarray:
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
    `dataset_manifest.verify_content_provenance`).

    Adam's Step 6 audit #6 of commit a32051b: "sparse-slice expression
    before densifying; streaming train-gene mean/variance." The prior
    version densified EVERY training sample's FULL `adata.X` at once,
    then concatenated ALL of them into one pooled matrix before calling
    `.std(axis=0)` -- for a real ~17,000-gene panel with many training
    samples, that materializes the entire training expression dataset,
    densified, in memory simultaneously. This streams instead: each
    sample's (typically sparse) matrix is SLICED into row chunks first
    (sparse slicing is cheap; only the chunk itself is ever densified)
    and folded into a running per-gene sum/sum-of-squares/count, so at
    most one chunk's dense array is ever resident at a time -- no
    sample's full matrix, let alone the pooled matrix across every
    sample, is ever materialized. Mathematically identical result to the
    prior pooled `.std(axis=0)` (population std, ddof=0): Var[X] =
    E[X^2] - E[X]^2, accumulated in float64."""
    if not train_samples:
        raise ValueError("compute_training_gene_scale: no training samples given")
    n_genes = None
    total_sum: np.ndarray | None = None
    total_sumsq: np.ndarray | None = None
    total_count = 0
    for sample in train_samples.values():
        X = sample.adata.X
        n_rows = X.shape[0]
        if n_genes is None:
            n_genes = X.shape[1]
            total_sum = np.zeros(n_genes, dtype=np.float64)
            total_sumsq = np.zeros(n_genes, dtype=np.float64)
        elif X.shape[1] != n_genes:
            raise ValueError(f"compute_training_gene_scale: gene-dimension mismatch ({X.shape[1]} vs {n_genes})")
        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            chunk = X[start:end]  # sparse-slice FIRST -- cheap, no densification yet
            chunk = chunk.toarray() if hasattr(chunk, "toarray") else np.asarray(chunk)
            chunk = np.asarray(chunk, dtype=np.float64)
            total_sum += chunk.sum(axis=0)
            total_sumsq += (chunk ** 2).sum(axis=0)
            total_count += chunk.shape[0]
    if total_count == 0:
        raise ValueError("compute_training_gene_scale: zero rows across all training samples")
    mean = total_sum / total_count
    variance = np.clip(total_sumsq / total_count - mean ** 2, 0.0, None)
    scale = np.sqrt(variance)
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


def common_random_validation_seed(seed: int, item_index: int) -> int:
    """Codex re-audit of commit 90f853e, launch blocker #6: "Use common
    random numbers for Architecture 4 validation: seed by fixed
    evaluation seed plus stable mask identity, independent of training
    step and evaluation order." Extracted into its own named function
    (previously inline inside `_run_validation`'s closure) specifically
    so this determinism/step-independence property is directly unit-
    testable, not just observable end-to-end. `item_index` is `val_
    loader`'s own enumeration index -- stable because `val_loader` is
    always built with `shuffle=False` -- never the current training
    step, which is the property this function exists to guarantee: two
    calls with the SAME `(seed, item_index)` return the SAME value
    regardless of when (which training step) they are called."""
    return (int(seed) * 7_919 + int(item_index)) % (2**63)


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
    architecture_id: str, model, inputs, target_expression: torch.Tensor, query_coords: torch.Tensor,
    gradient_weight: float, k_neighbors: int, per_gene_scale: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> dict:
    """The reconstruction objective used for VALIDATION/model-selection
    (Adam's Step 6 audit #8, then corrected by audit #1 of commit
    a32051b). For Architectures 1-3, `model(inputs)` (their real forward
    pass) is used directly. For Architecture 4, audit #1 is explicit:
    "Architecture 4 validation/evaluation/overfit must evaluate a
    deterministic fixed-seed predictive mean from
    `sample_predictive_distribution`, not `forward()`'s frozen
    conditioner." `forward()` never touches the trained flow apparatus at
    all -- selecting on it would let Architecture 4's flow weights train
    for hours while the ONLY metric ever checked is blind to whether they
    learned anything. `predict_for_metrics` (with a caller-supplied,
    fixed-per-step `generator` for resume-exact reproducibility) is now
    used for every architecture; the result additionally carries
    `conditioner_only_total` for Architecture 4 -- a SECONDARY diagnostic
    logged alongside the real metric, never used for selection."""
    prediction = predict_for_metrics(architecture_id, model, inputs, generator=generator)
    result = combined_reconstruction_loss(
        prediction["expression"], target_expression, query_coords,
        gradient_weight=gradient_weight, k_neighbors=k_neighbors, per_gene_scale=per_gene_scale,
    )
    if "conditioner_only_expression" in prediction:
        conditioner_only = combined_reconstruction_loss(
            prediction["conditioner_only_expression"], target_expression, query_coords,
            gradient_weight=gradient_weight, k_neighbors=k_neighbors, per_gene_scale=per_gene_scale,
        )
        result = {**result, "conditioner_only_total": conditioner_only["total"]}
    return result


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


def config_identity_fingerprint(config: dict) -> str:
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


def _environment_versions() -> dict:
    """Best-effort, informational-only record of the software environment
    a run actually executed under -- Adam's Step 6 audit #9 of commit
    a32051b: "Record environment versions." Never raises: an environment
    lookup failing (e.g. `torch.version.cuda` on a CPU-only build) must
    never fail a real training run over a diagnostics field."""
    try:
        cuda_version = torch.version.cuda
    except Exception:
        cuda_version = None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": cuda_version,
        "cuda_available": bool(torch.cuda.is_available()),
        "platform": platform.platform(),
    }


def _worktree_diff_hash() -> str | None:
    """Best-effort sha256 over `git diff HEAD` (every tracked-file
    change, staged or not, relative to the current commit) plus `git
    status --porcelain` (so a new UNTRACKED file also changes the hash)
    -- Codex re-audit of commit 90f853e, launch blocker #10: "Bind code
    state on resume: exact commit plus clean-worktree status/diff hash."
    None (not an error) outside a git checkout or if git itself is
    unavailable, exactly like `_code_commit_hash`; an all-clean worktree
    still returns a real, stable hash (of two empty strings), never
    None, so "clean" is distinguishable from "unknown"."""
    try:
        diff = subprocess.run(
            ["git", "diff", "HEAD"], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=10,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=10,
        )
        if diff.returncode != 0 or status.returncode != 0:
            return None
        return hashlib.sha256(f"{diff.stdout}\x00{status.stdout}".encode("utf-8")).hexdigest()
    except Exception:
        return None


def _code_commit_hash() -> str | None:
    """Best-effort git commit SHA of the code that produced this run --
    Adam's Step 6 audit #9: "Record... code commit." None (not an error)
    outside a git checkout or if git itself is unavailable; a run
    manifest missing this field is a real, honestly-reported limitation,
    never a reason to fail the run."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def build_run_manifest(
    *, config: dict, config_path: str, dataset_manifest: dict, gene_names: list[str], seed: int,
    preflight_report: dict, train_schedule, val_schedule, architecture_id: str,
    checkpoint_dir: Path, synchronized_init_manifest_path: Path | None,
    gigapath_checkpoint_sha256: str | None = None, gene_residual_basis_gene_names_hash: str | None = None,
    gene_residual_basis_sha256: str | None = None, gene_scale_sha256: str | None = None,
    architecture3_conditioner_info: dict | None = None,
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
        "config_identity_fingerprint": config_identity_fingerprint(config),
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
        # Launch blocker #5: lifted to the top level (not merely buried
        # inside cache_preflight_report, which is NOT part of
        # _RESUME_CONSISTENCY_FIELDS) specifically so a resume can
        # compare it -- real per-sample cache content identity, not just
        # tile-encoder provenance.
        "cache_content_fingerprint": preflight_report.get("cache_content_fingerprint"),
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
        "gene_residual_basis_sha256": gene_residual_basis_sha256,
        "gene_scale_sha256": gene_scale_sha256,
        # Audit #4: Architecture 4's conditioner must be bound to the
        # EXACT Architecture 3 checkpoint (weights sha256 + step) it was
        # loaded from, not merely "some path was configured."
        "architecture3_conditioner_checkpoint_dir": (architecture3_conditioner_info or {}).get("checkpoint_dir"),
        "architecture3_conditioner_checkpoint_sha256": (architecture3_conditioner_info or {}).get("checkpoint_sha256"),
        "architecture3_conditioner_checkpoint_step": (architecture3_conditioner_info or {}).get("checkpoint_step"),
        # Audit #9: informational-only (never part of resume-consistency
        # verification -- a different torch/CUDA patch version resuming
        # the same run is not itself a scientific-identity change).
        "environment_versions": _environment_versions(),
        # Launch blocker #10: UNLIKE environment_versions, these two ARE
        # part of resume-consistency verification by default (see
        # `verify_resume_consistency`'s dedicated code-state check) --
        # the code that produced the run IS a scientific-identity field,
        # not merely informational context.
        "code_commit_hash": _code_commit_hash(),
        "code_worktree_diff_hash": _worktree_diff_hash(),
    }


_RESUME_CONSISTENCY_FIELDS = (
    "config_identity_fingerprint", "dataset_manifest_fingerprint", "gene_panel_hash", "model_architecture",
    "synchronized_init_manifest_sha256", "gigapath_checkpoint_sha256",
    "gene_residual_basis_gene_names_hash", "gene_residual_basis_sha256", "gene_scale_sha256",
    # Launch blocker #5: real per-sample cache CONTENT identity -- catches
    # a validly-regenerated cache with different numeric content but
    # identical tile-encoder provenance, which nothing else here would.
    "cache_content_fingerprint",
    # Audit #4: a resume must refuse to continue if Architecture 4's
    # conditioner checkpoint was swapped for a DIFFERENT Architecture 3
    # checkpoint (same or different path) between runs -- exact weight
    # identity, not merely "some checkpoint is configured."
    "architecture3_conditioner_checkpoint_sha256", "architecture3_conditioner_checkpoint_step",
)


def verify_resume_consistency(
    old_run_manifest: dict, new_run_manifest: dict, *, allow_code_drift: bool = False,
) -> None:
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
    checked.

    Scope caveat (Adam's Step 6 audit #9 of commit a32051b): this
    verifies IDENTITY -- the same config/dataset/gene-panel/architecture/
    synchronized-init/conditioner-checkpoint/basis/gene-scale a resumed
    run started from -- never BIT-EXACT FLOATING-POINT reproducibility
    of training on a GPU. Nothing in this trainer calls
    `torch.use_deterministic_algorithms` or otherwise enforces
    deterministic CUDA kernels/cuDNN algorithm selection; a resumed run
    on GPU can therefore diverge numerically step-by-step from an
    unbroken run even with every field this function checks unchanged --
    `environment_versions` is recorded in the run manifest for this
    reason, as informational context, not as a determinism guarantee.

    Codex re-audit of commit 90f853e, launch blocker #10: "Bind code
    state on resume: exact commit plus clean-worktree status/diff hash,
    or require an explicit scientifically-visible override." Checked
    separately from the generic `_RESUME_CONSISTENCY_FIELDS` loop below
    because it needs its own override, not because it is any less real:
    a resumed run whose code changed (a different commit, or a dirty
    worktree with different uncommitted edits) may have produced
    different losses/gradients/masking behavior than the run being
    resumed, for reasons none of the other fields here can see. Skipped
    when the OLD manifest recorded no commit at all (a checkpoint from
    outside a git checkout, or predating this field -- nothing to bind
    against). `allow_code_drift=True` (threaded from
    `run_training(..., allow_code_drift=True)` / `--allow-code-drift`)
    is the explicit override: it does not silence the check, it is
    RECORDED by the caller into the new run manifest's
    `code_drift_acknowledged` field so the override is scientifically
    visible in the artifact itself, never a silent bypass."""
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

    old_commit = old_run_manifest.get("code_commit_hash")
    new_commit = new_run_manifest.get("code_commit_hash")
    old_diff_hash = old_run_manifest.get("code_worktree_diff_hash")
    new_diff_hash = new_run_manifest.get("code_worktree_diff_hash")
    code_drifted = old_commit is not None and (old_commit != new_commit or old_diff_hash != new_diff_hash)
    if code_drifted and not allow_code_drift:
        raise ValueError(
            f"resume refused: code state changed since the last checkpoint at "
            f"{old_run_manifest.get('checkpoint_dir')} (commit {old_commit!r} -> {new_commit!r}, "
            f"worktree_diff_hash {old_diff_hash!r} -> {new_diff_hash!r}) -- pass allow_code_drift=True "
            "(run_training(..., allow_code_drift=True) / --allow-code-drift) if resuming under different "
            "code is genuinely intended; the override is recorded in the new run_manifest.json, never silent"
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


def save_best_checkpoint_bundle(
    model, gene_names: list[str], best_dir: str | Path, *, step: int, val_loss: float, run_manifest: dict,
) -> Path:
    """Atomically (re)write `best/` as a COMPLETE, independently
    verifiable inference bundle -- Adam's Step 6 audit #3 of commit
    a32051b: "Make best/ a complete, verifiable inference bundle... or an
    immutable pointer to one... including gene names, config/run-manifest
    identity, external artifact hashes, selected step, and weights."
    Before this function existed, `best/` held only `trainable_weights.pt`
    plus a 2-field `best_info.json` -- not independently loadable/
    verifiable as a standalone artifact separate from the live
    `checkpoint_dir` state.

    Staged in a temp directory and swapped in with a single `os.replace`
    so `best/` is always either the complete PREVIOUS bundle or the
    complete NEW one, never a partially-written mix of the two (mirrors
    `checkpoint.py`'s existing atomic-write discipline, extended here to
    the whole directory rather than one file at a time)."""
    best_dir = Path(best_dir)
    tmp_dir = best_dir.parent / f".{best_dir.name}.tmp{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_module.save_trainable_state(model, tmp_dir)
    (tmp_dir / "gene_names.json").write_text(json.dumps(list(gene_names), indent=2))

    file_hashes = {
        name: file_sha256(tmp_dir / name)
        for name in ("trainable_weights.pt", "gene_names.json") if (tmp_dir / name).is_file()
    }
    bundle_info = {
        "step": int(step), "total": float(val_loss), "files": file_hashes,
        "config_path": run_manifest.get("config_path"),
        "config_identity_fingerprint": run_manifest.get("config_identity_fingerprint"),
        "dataset_manifest_fingerprint": run_manifest.get("dataset_manifest_fingerprint"),
        "gene_panel_hash": run_manifest.get("gene_panel_hash"),
        "model_architecture": run_manifest.get("model_architecture"),
        "synchronized_init_manifest_sha256": run_manifest.get("synchronized_init_manifest_sha256"),
        "gigapath_checkpoint_sha256": run_manifest.get("gigapath_checkpoint_sha256"),
        "gene_residual_basis_gene_names_hash": run_manifest.get("gene_residual_basis_gene_names_hash"),
        "gene_residual_basis_sha256": run_manifest.get("gene_residual_basis_sha256"),
        "gene_scale_sha256": run_manifest.get("gene_scale_sha256"),
        "architecture3_conditioner_checkpoint_sha256": run_manifest.get("architecture3_conditioner_checkpoint_sha256"),
        "architecture3_conditioner_checkpoint_step": run_manifest.get("architecture3_conditioner_checkpoint_step"),
        "code_commit_hash": run_manifest.get("code_commit_hash"),
    }
    # Written LAST -- a reader can treat best_info.json's presence (and
    # its own per-file hashes matching) as proof the bundle is complete.
    (tmp_dir / "best_info.json").write_text(json.dumps(bundle_info, indent=2, sort_keys=True, default=str))

    if best_dir.exists():
        shutil.rmtree(best_dir)
    os.replace(tmp_dir, best_dir)
    return best_dir


def verify_checkpoint_bundle_identity(
    bundle_dir: str | Path, *, dataset_manifest: dict, gene_names: list[str], architecture_id: str | None = None,
) -> dict:
    """Fail-closed pre-load check for a `save_best_checkpoint_bundle`
    bundle -- Adam's Step 6 audit #7: "Verify the checkpoint run manifest
    against evaluation inputs before loading." Confirms every per-file
    hash the bundle's own `best_info.json` recorded still matches the
    file on disk right now (catches altered/corrupted bundle contents --
    audit #4's adversarial scenario) and that the bundle's
    `dataset_manifest_fingerprint`/`gene_panel_hash`/`model_architecture`
    agree with what the CALLER is about to evaluate against, before any
    weights are loaded. Returns the bundle's own info dict on success.

    Codex re-audit of commit 90f853e, launch blocker #4: "Missing
    dataset/gene metadata is accepted as valid." The prior version
    treated a MISSING recorded field (`info.get(...) is None`) as
    passing -- a bundle whose `best_info.json` never actually recorded
    its own dataset/gene-panel identity (e.g. a hand-assembled or
    corrupted `best_info.json`) passed this check by simply omitting the
    field it would have failed on. Every identity field checked here is
    now REQUIRED to be present; a missing field now fails exactly like a
    mismatched one."""
    bundle_dir = Path(bundle_dir)
    info_path = bundle_dir / "best_info.json"
    if not info_path.is_file():
        raise ValueError(f"{bundle_dir} has no best_info.json -- not a complete, verifiable inference bundle")
    info = json.loads(info_path.read_text())
    for name, expected_hash in (info.get("files") or {}).items():
        file_path = bundle_dir / name
        if not file_path.is_file():
            raise ValueError(f"{bundle_dir}: best_info.json references {name} but it is missing")
        if file_sha256(file_path) != expected_hash:
            raise ValueError(
                f"{bundle_dir}: {name} does not match the sha256 recorded in best_info.json -- "
                "the bundle's contents were altered after being written, refusing to load"
            )
    expected_dataset_fp = dataset_manifest_fingerprint(dataset_manifest)
    recorded_dataset_fp = info.get("dataset_manifest_fingerprint")
    if recorded_dataset_fp is None or recorded_dataset_fp != expected_dataset_fp:
        raise ValueError(
            f"{bundle_dir}: was selected under dataset_manifest_fingerprint={recorded_dataset_fp!r} but "
            f"this evaluation's dataset manifest fingerprints to {expected_dataset_fp!r} -- refusing to "
            "evaluate a checkpoint against different (or unrecorded) data than it was trained/selected on"
        )
    expected_gene_hash = gene_panel_hash(gene_names)
    recorded_gene_hash = info.get("gene_panel_hash")
    if recorded_gene_hash is None or recorded_gene_hash != expected_gene_hash:
        raise ValueError(
            f"{bundle_dir}: was selected under gene_panel_hash={recorded_gene_hash!r} but this "
            f"evaluation's gene panel hashes to {expected_gene_hash!r} -- refusing to load"
        )
    if architecture_id is not None:
        recorded_architecture = info.get("model_architecture")
        if recorded_architecture is None or str(recorded_architecture) != str(architecture_id):
            raise ValueError(
                f"{bundle_dir}: was selected under model_architecture={recorded_architecture!r} but this "
                f"evaluation is loading architecture {architecture_id!r} -- refusing to load"
            )
    return info


def verify_checkpoint_run_manifest_against_evaluation(
    checkpoint_dir: str | Path, *, config: dict, dataset_manifest: dict, gene_names: list[str],
) -> dict:
    """Fail-closed pre-load check for evaluating a checkpoint_dir's
    LATEST state directly (not a `best/` bundle, which is verified
    separately by `verify_checkpoint_bundle_identity`). Codex re-audit of
    commit 90f853e, launch blocker #4: "Latest-checkpoint evaluation does
    not compare the checkpoint's run manifest with the evaluation
    dataset ... must require complete metadata and verify configuration,
    architecture, dataset, gene panel, basis, conditioner, LongNet,
    synchronized initialization ... and exact weights before loading."
    ("Exact weights" is covered separately -- `checkpoint.py`'s own
    transactional `_resolve_checkpoint_source` already fail-closed
    verifies every file's hash before `load_trainable_state` uses it.)

    Compares `checkpoint_dir/run_manifest.json`'s own recorded identity
    fields against values freshly computed from THIS evaluation's
    config/dataset/gene-panel -- every field checked is REQUIRED to be
    present in the checkpoint's manifest; a missing field fails exactly
    like a mismatched one."""
    checkpoint_dir = Path(checkpoint_dir)
    manifest_path = checkpoint_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            f"{checkpoint_dir} has no run_manifest.json -- cannot verify this checkpoint's config/"
            "dataset/gene-panel/architecture/synchronized-init/conditioner/basis identity against the "
            "evaluation inputs before loading. Refusing to evaluate an unverifiable checkpoint (a "
            "best/ bundle, verified independently, is the alternative self-verifying path)"
        )
    checkpoint_run_manifest = json.loads(manifest_path.read_text())

    def _require_match(field: str, expected) -> None:
        recorded = checkpoint_run_manifest.get(field)
        if recorded is None or recorded != expected:
            raise ValueError(
                f"{checkpoint_dir}: run_manifest.json field {field}={recorded!r} does not match this "
                f"evaluation's own {field}={expected!r} -- refusing to load a checkpoint whose recorded "
                "identity does not match (or never recorded) what is being evaluated against"
            )

    architecture_id = str((config.get("model") or {}).get("architecture", ""))
    _require_match("model_architecture", architecture_id)
    _require_match("dataset_manifest_fingerprint", dataset_manifest_fingerprint(dataset_manifest))
    _require_match("gene_panel_hash", gene_panel_hash(gene_names))
    _require_match("config_identity_fingerprint", config_identity_fingerprint(config))

    training_cfg = config.get("training") or {}
    synchronized_init_dir = training_cfg.get("synchronized_init_dir")
    if synchronized_init_dir:
        sync_manifest_path = Path(synchronized_init_dir) / "initialization_manifest.json"
        expected_sync_sha256 = file_sha256(sync_manifest_path) if sync_manifest_path.is_file() else None
        _require_match("synchronized_init_manifest_sha256", expected_sync_sha256)

    model_params = (config.get("model") or {}).get("params") or {}
    required_fingerprints = config.get("required_fingerprints") or {}
    if model_params.get("use_global_slide"):
        gigapath_checkpoint_path = required_fingerprints.get("gigapath_checkpoint")
        expected_gigapath_sha256 = (
            file_sha256(gigapath_checkpoint_path)
            if gigapath_checkpoint_path and Path(gigapath_checkpoint_path).is_file() else None
        )
        _require_match("gigapath_checkpoint_sha256", expected_gigapath_sha256)

    if architecture_id == "4":
        conditioner_checkpoint_dir = required_fingerprints.get("architecture3_conditioner_checkpoint")
        if conditioner_checkpoint_dir:
            expected_conditioner_sha256 = checkpoint_module.resolve_checkpoint_identity(
                conditioner_checkpoint_dir,
            ).weights_sha256
            _require_match("architecture3_conditioner_checkpoint_sha256", expected_conditioner_sha256)
        basis_path = required_fingerprints.get("gene_residual_basis")
        if basis_path:
            from gen3_multiscale.models.gene_basis import load_gene_residual_basis

            basis = load_gene_residual_basis(basis_path)
            expected_basis_sha256 = hashlib.sha256(
                np.ascontiguousarray(basis.basis.detach().cpu().numpy()).tobytes()
            ).hexdigest()
            _require_match("gene_residual_basis_sha256", expected_basis_sha256)

    return checkpoint_run_manifest


def _log_step(step: int, split: str, losses: dict, extra: str = "") -> None:
    parts = ", ".join(f"{k}={float(v.detach()) if torch.is_tensor(v) else float(v):.6f}" for k, v in losses.items())
    print(f"[step {step}] {split}: {parts}{extra}", flush=True)


def run_training(
    config_path: str, smoke: bool = False, staged_smoke: bool = False, allow_code_drift: bool = False,
) -> dict:
    """The real Step 6 training entrypoint. Returns a small summary dict
    (never a live model/optimizer -- those are process-local); a caller
    that wants the trained model runs this in-process and reads
    `checkpoint_dir` afterward, matching every other artifact-based
    hand-off in this package.

    `staged_smoke` (Adam's Step 6 audit #8 of commit a32051b): only
    meaningful together with `smoke=True`. A plain `--smoke` is
    CONSTRUCTION-ONLY -- it never requires Architecture 4's real
    Architecture 3 conditioner checkpoint. `staged_smoke=True` runs the
    real, one-step "staged" smoke: Architecture 4 must have a real
    conditioner checkpoint configured, exactly like a non-smoke run.

    `allow_code_drift` (Codex re-audit of commit 90f853e, launch blocker
    #10): the explicit, scientifically-visible override for
    `verify_resume_consistency`'s code-state check -- see that
    function's own docstring. False (the default) means a resume whose
    code state changed since the checkpoint's last save is refused."""
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
    device = torch.device(training_cfg.get("device", "cpu") if torch.cuda.is_available() else "cpu")

    # Audit #2: the ONE shared model-reconstruction pipeline -- builds the
    # architecture, loads verified synchronized initialization (fails
    # closed for a real run), and loads+freezes Architecture 4's exact
    # Architecture 3 conditioner. `checkpoint_dir=None` here: whether to
    # resume THIS run's own checkpoint is decided further down, gated on
    # `training_state.get("step", 0) > 0`.
    model, model_info = build_model_for_inference(
        config, gene_names=gene_names, device=device, checkpoint_dir=None,
        smoke=smoke, staged_smoke=staged_smoke, dataset_manifest=dataset_manifest,
    )
    gene_basis = model_info["gene_basis"]
    gigapath_checkpoint_sha256 = model_info["gigapath_checkpoint_sha256"]
    synchronized_init_manifest_path = model_info["synchronized_init_manifest_path"]
    architecture3_conditioner_info = model_info["architecture3_conditioner"]

    # Requirement #9: real, TRAINING-ONLY per-gene standardization scale
    # for the spatial-gradient loss -- computed here (train_samples are
    # already loaded by preflight) but NOT YET WRITTEN to disk. Audit #4
    # of commit a32051b ("Do not overwrite gene_scale.npy or other
    # checkpoint artifacts before resume verification passes"): the
    # actual `save_gene_scale` write now happens further below, AFTER
    # `verify_resume_consistency` has passed -- a resume that gets
    # refused must never have already clobbered the prior run's
    # gene_scale.npy on its way to being refused.
    gene_scale = compute_training_gene_scale(train_samples)
    gene_scale_sha256 = hashlib.sha256(np.ascontiguousarray(gene_scale).tobytes()).hexdigest()
    gene_scale_tensor = torch.as_tensor(gene_scale, dtype=torch.float32, device=device)

    gene_residual_basis_gene_names_hash = gene_basis.gene_names_hash if gene_basis is not None else None
    # Audit #4: "bind resume/evaluation to exact numeric basis SHA256" --
    # gene_residual_basis_gene_names_hash alone only proves the GENE
    # ORDERING matches; it says nothing about the basis's own numeric
    # CONTENT, so a basis file re-fit (or hand-edited) to different
    # numeric values with the identical gene ordering would silently
    # pass every check that existed before this hash.
    gene_residual_basis_sha256 = (
        hashlib.sha256(np.ascontiguousarray(gene_basis.basis.detach().cpu().numpy()).tobytes()).hexdigest()
        if gene_basis is not None else None
    )

    # Requirement #5: build the NEW run manifest in memory (never yet
    # written) and, if a PRIOR run_manifest.json already exists in this
    # checkpoint_dir, verify every identity-bearing fingerprint agrees
    # with it BEFORE loading any checkpoint state, BEFORE writing
    # gene_scale.npy, and BEFORE overwriting that prior manifest --
    # "Refuse changed configs or artifacts... Do not overwrite the old
    # run manifest -- or gene_scale.npy, or any other checkpoint artifact
    # -- before verifying it" (audit #4).
    run_manifest = build_run_manifest(
        config=config, config_path=str(config_path), dataset_manifest=dataset_manifest, gene_names=gene_names,
        seed=seed, preflight_report=preflight_report, train_schedule=train_schedule, val_schedule=val_schedule,
        architecture_id=architecture_id, checkpoint_dir=checkpoint_dir,
        synchronized_init_manifest_path=synchronized_init_manifest_path,
        gigapath_checkpoint_sha256=gigapath_checkpoint_sha256,
        gene_residual_basis_gene_names_hash=gene_residual_basis_gene_names_hash,
        gene_residual_basis_sha256=gene_residual_basis_sha256,
        gene_scale_sha256=gene_scale_sha256,
        architecture3_conditioner_info=architecture3_conditioner_info,
    )
    existing_run_manifest_path = checkpoint_dir / "run_manifest.json"
    code_drift_acknowledged = False
    if existing_run_manifest_path.is_file():
        old_run_manifest = json.loads(existing_run_manifest_path.read_text())
        verify_resume_consistency(old_run_manifest, run_manifest, allow_code_drift=allow_code_drift)
        # Launch blocker #10: record whether the override was actually
        # NEEDED (not merely passed) -- a caller passing
        # allow_code_drift=True against a checkpoint whose code state
        # did NOT drift leaves this False, so the manifest only ever
        # claims an override happened when one genuinely did.
        old_commit = old_run_manifest.get("code_commit_hash")
        code_drift_acknowledged = allow_code_drift and old_commit is not None and (
            old_commit != run_manifest.get("code_commit_hash")
            or old_run_manifest.get("code_worktree_diff_hash") != run_manifest.get("code_worktree_diff_hash")
        )
    run_manifest["code_drift_acknowledged"] = code_drift_acknowledged

    # Only now, having passed resume-consistency verification (or there
    # being no prior run to verify against), is it safe to overwrite this
    # checkpoint_dir's gene_scale.npy -- audit #4.
    gene_scale_path = save_gene_scale(gene_scale, checkpoint_dir / "gene_scale.npy")

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
            val_conditioner_only_totals: list[float] = []
            val_flow_losses: list[float] = []
            for item_index, (val_inputs, val_targets) in enumerate(val_loader):
                # Codex re-audit of commit 90f853e, launch blocker #6:
                # "Use common random numbers for Architecture 4
                # validation: seed by fixed evaluation seed plus stable
                # mask identity, independent of training step and
                # evaluation order." The PRIOR generator was reseeded
                # from (seed, current_step) -- every checkpoint therefore
                # sampled Architecture 4's flow apparatus with DIFFERENT
                # noise on the SAME held-out item, so best-checkpoint
                # selection could partly reflect Monte Carlo luck in the
                # noise draw rather than a genuine difference between
                # checkpoints. Reseeding per FIXED `item_index` instead
                # (`val_loader` is `shuffle=False`, so this index names
                # the SAME held-out item at every call, regardless of
                # `current_step`) gives every checkpoint the IDENTICAL
                # noise draw on the IDENTICAL item -- a real common-
                # random-numbers comparison across checkpoints. Still
                # exactly reproducible across a resume, since it depends
                # only on the run's own fixed `seed` and the item's fixed
                # position in the deterministic validation dataset.
                predictive_val_generator = torch.Generator(device=device).manual_seed(
                    common_random_validation_seed(seed, item_index)
                )
                val_target_expression = torch.as_tensor(val_targets.query_expression, dtype=torch.float32, device=device)
                val_query_coords = torch.as_tensor(val_inputs.query_coords, dtype=torch.float32, device=device)
                # Requirement #8/audit #1: model selection ALWAYS uses the
                # real predictive distribution (Architecture 4) or real
                # forward() (Architectures 1-3) -- never Architecture 4's
                # frozen-conditioner-only forward(), which used to be
                # silently used for its selection metric.
                det_losses = compute_deterministic_reconstruction_losses(
                    architecture_id, model, val_inputs, val_target_expression, val_query_coords,
                    gradient_weight=gradient_weight, k_neighbors=k_neighbors, per_gene_scale=gene_scale_tensor,
                    generator=predictive_val_generator,
                )
                val_totals.append(float(det_losses["total"]))
                if "conditioner_only_total" in det_losses:
                    val_conditioner_only_totals.append(float(det_losses["conditioner_only_total"]))
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
            if val_conditioner_only_totals:
                # Audit #1: the frozen-conditioner-only reconstruction
                # loss, reported ONLY as a secondary diagnostic -- never
                # read for selection ("total" above always comes from the
                # real predictive mean for Architecture 4).
                entry["conditioner_only_total_mean_fixed_generator"] = float(np.mean(val_conditioner_only_totals))
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

        # Codex re-audit of commit 90f853e, launch blocker #7: "Fix
        # completed-step semantics. After optimizer.step(), increment
        # completed_steps, then log/validate/checkpoint using that
        # value." The prior code logged/validated using `step`'s PRE-
        # increment value (0-indexed: the step whose DATA this iteration
        # just trained on) but labeled periodic/final checkpoints with
        # `step`'s POST-increment value (1-indexed: the count of
        # completed updates) -- "step 2000" therefore meant two DIFFERENT
        # actual model states depending on whether it came from
        # validation/best-checkpoint-selection or from a periodic
        # checkpoint save. `completed_steps` (the count of optimizer
        # updates actually applied so far, including this one) is now
        # THE single value used everywhere a step gets logged, validated
        # against, or used to label a saved checkpoint.
        completed_steps = step + 1

        if completed_steps % log_every_n_steps == 0 or smoke:
            _log_step(completed_steps, "train", losses, extra=f", grad_norm={float(grad_norm):.4f}")

        if val_loader is not None and (smoke or (completed_steps % eval_every_n_steps == 0)):
            entry = _run_validation(completed_steps)
            if entry is not None:
                validation_history.append(entry)
                _save_json_atomic(validation_history, validation_history_path)
                if entry["total"] < best_val_loss:
                    best_val_loss = entry["total"]
                    if not smoke:
                        save_best_checkpoint_bundle(
                            model, gene_names, checkpoint_dir / "best",
                            step=completed_steps, val_loss=entry["total"], run_manifest=run_manifest,
                        )

        # Codex re-audit of commit 90f853e, launch blocker #1/#7: the
        # trainer used to ALSO save an unconditional final checkpoint
        # after the loop, regardless of whether the last in-loop
        # iteration had already just saved one at the exact same step --
        # whenever the natural end of training coincided with
        # `checkpoint_every_n_steps`, the SAME step got saved twice.
        # `completed_steps < step_target` skips the in-loop save exactly
        # when this iteration is ALSO the run's natural final step (that
        # save is the post-loop one below, which additionally carries the
        # real, final `completion_reason` -- the periodic save here only
        # ever records "in_progress").
        if (not smoke) and completed_steps % checkpoint_every_n_steps == 0 and completed_steps < step_target:
            checkpoint_module.save_checkpoint(
                model, config, gene_names, checkpoint_dir, completed_steps,
                extra_metadata={"n_skipped_nonfinite": n_skipped_nonfinite, "completion_reason": "in_progress"},
                keep_last=checkpoint_keep_last, optimizer=optimizer, rng=sample_rng,
            )
        step = completed_steps

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
    parser.add_argument(
        "--staged-smoke", action="store_true",
        help="Only meaningful with --smoke. Architecture 4's real, already-trained "
             "architecture3_conditioner_checkpoint is required (exactly like a non-smoke run), unlike a plain "
             "--smoke, which is construction-only and exempt. Adam's Step 6 audit #8 of commit a32051b.",
    )
    parser.add_argument(
        "--allow-code-drift", action="store_true",
        help="Explicit override to resume a checkpoint under a different git commit or dirty worktree "
             "than the one it was last saved under. Codex re-audit of commit 90f853e, launch blocker #10. "
             "Recorded in the new run_manifest.json's code_drift_acknowledged field -- never a silent bypass.",
    )
    args = parser.parse_args()
    run_training(args.config, smoke=args.smoke, staged_smoke=args.staged_smoke, allow_code_drift=args.allow_code_drift)


if __name__ == "__main__":
    main()
