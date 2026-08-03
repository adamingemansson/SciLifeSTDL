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

# Codex re-audit of commit 57f0e3c: "Prefer moving config fingerprint/
# loading utilities into a neutral module to avoid a circular import
# between the trainer and resolver." `resolved_config`/`config_
# fingerprint`/`config_identity_fingerprint` now live in
# `gen3_multiscale/config_identity.py` (no dependency on anything in
# this module) and are re-exported here so every existing `from
# gen3_multiscale.training.train import config_fingerprint, ...` caller
# across this codebase keeps working unchanged.
from gen3_multiscale.config_identity import config_fingerprint, config_identity_fingerprint, resolved_config
from gen3_multiscale.data.dataset_manifest import gene_panel_hash, load_dataset_manifest
from gen3_multiscale.models import model_factory
from gen3_multiscale.models.losses import combined_reconstruction_loss
from gen3_multiscale.training import checkpoint as checkpoint_module
from gen3_multiscale.training.gen3_dataset import (
    Gen3SpatialFieldDataset, build_gen3_mask_schedule, gen3_identity_collate,
)
from gen3_multiscale.training.gen3_preflight import load_and_preflight_samples, save_gen3_preflight_report


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
    cache_content_by_sample: dict[str, dict] | None = None,
    conditioner_identity: "checkpoint_module.CheckpointIdentity | None" = None,
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
    fields. `training_mask_schedule_fingerprint` is still REQUIRED to be
    present (bound, per Codex's re-audit of commit f7bb8a1's launch
    blocker #5), just not equality-checked, for the same reason.

    Codex re-audit of commit f7bb8a1, launch blocker #5: "Bind the basis
    sidecar to canonical bundle identity/step... and cache content.
    Validate all of those when Architecture 4 loads the basis." Two
    fields added since the prior round: `architecture3_checkpoint_step`
    (the resolved bundle's own step, cross-checked for EXACT equality
    against the currently configured conditioner checkpoint's resolved
    step -- weights_sha256 alone already implies the same step in
    practice, but a caller relying on this function to catch a
    swapped/rolled-back checkpoint should not have to reason about that
    implication); and `cache_content_by_sample`, compared PER-SAMPLE
    (never the single combined preflight fingerprint -- see
    `verify_full_checkpoint_identity`'s own docstring for why a combined
    hash cannot be validly compared across two calls that may cover
    different sample sets) against `cache_content_by_sample`, when the
    caller supplies one, for every sample_id present in BOTH sides;
    samples the basis-fitting run never touched are skipped, not failed."""
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
    _require_field("architecture3_config_identity_fingerprint")
    _require_field("mask_schedule_reports")
    _require_field("training_mask_schedule_fingerprint")

    # Codex re-audit of commit 2162ff4, finding #5: "recompute
    # training_mask_schedule_fingerprint from the recorded reports." A
    # required-present field can still be WRONG relative to the reports
    # it is supposed to describe (the "value was wrong when originally
    # saved" bug class); re-derive it fresh from the sidecar's OWN
    # recorded `mask_schedule_reports` and compare, catching a
    # self-inconsistent sidecar even though there is still no external
    # "current" value for either field to be checked against.
    recomputed_schedule_fingerprint = hashlib.sha256(
        json.dumps(provenance["mask_schedule_reports"], sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    if recomputed_schedule_fingerprint != provenance["training_mask_schedule_fingerprint"]:
        raise ValueError(
            f"gene residual basis provenance sidecar {provenance_path} records a "
            f"training_mask_schedule_fingerprint that does not match a fresh hash of its own recorded "
            "mask_schedule_reports -- the sidecar's two identity records disagree with each other, "
            "refusing to trust either (Codex re-audit of commit 2162ff4, finding #5)"
        )

    # Codex re-audit of commit 2162ff4, finding #5: "require exact
    # train_sample_ids." Logically implied by `dataset_manifest_
    # fingerprint` already matching (train_sample_ids is part of that
    # same manifest), but checked explicitly and independently anyway --
    # real defense-in-depth against a hypothetical bug in the fingerprint
    # computation itself, and matches Codex's literal ask.
    recorded_train_sample_ids = _require_field("train_sample_ids")
    expected_train_sample_ids = sorted(dataset_manifest.get("train_sample_ids") or [])
    if sorted(recorded_train_sample_ids) != expected_train_sample_ids:
        raise ValueError(
            f"gene residual basis at {path} was fit on train_sample_ids {sorted(recorded_train_sample_ids)!r} "
            f"but this run's dataset manifest declares train_sample_ids {expected_train_sample_ids!r} -- "
            "refusing to use a basis fit on a different training sample set"
        )

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
    recorded_checkpoint_step = _require_field("architecture3_checkpoint_step")
    # Codex re-audit of commit 7a2d819, finding #3: "Apply the same
    # resolve-once principle to Architecture 4's conditioner/basis
    # orchestration." Confirmed real: `conditioner_checkpoint_dir` (a
    # MUTABLE path -- Architecture 3 may still be training/checkpointing
    # concurrently) was independently resolved TWICE within this
    # function alone (`resolve_checkpoint_identity` here, then
    # `load_checkpoint_run_manifest` further below), and a THIRD time by
    # `maybe_load_pretrained_conditioner_for_architecture4` when it
    # actually loads weights onto the model -- three separate
    # resolutions of the same mutable path, any of which could observe a
    # DIFFERENT bundle if Architecture 3 saved a new checkpoint in
    # between. `build_model_for_inference` now resolves this path
    # EXACTLY ONCE and passes the result down as `conditioner_identity`
    # to both this function and `maybe_load_pretrained_conditioner_for_
    # architecture4` -- when given, it is used here INSTEAD of
    # re-resolving, so basis validation and the eventual weight load are
    # structurally guaranteed to describe the same immutable bundle.
    # `conditioner_identity` stays optional (falls back to resolving here
    # directly) so this function remains independently callable/testable
    # without requiring a caller to pin one first.
    actual_identity = conditioner_identity or checkpoint_module.resolve_checkpoint_identity(conditioner_checkpoint_dir)
    if actual_identity.weights_sha256 != recorded_checkpoint_sha256:
        raise ValueError(
            f"gene residual basis at {path} was fit against a different Architecture 3 checkpoint "
            f"(trainable_weights.pt sha256 {recorded_checkpoint_sha256!r}) than the one configured for "
            f"this run (sha256 {actual_identity.weights_sha256!r} at {conditioner_checkpoint_dir}) -- "
            "refusing to use a basis fit against a different conditioner's residuals"
        )
    if int(actual_identity.step) != int(recorded_checkpoint_step):
        raise ValueError(
            f"gene residual basis at {path} was fit against Architecture 3 checkpoint step "
            f"{recorded_checkpoint_step!r}, but the checkpoint configured for this run is at step "
            f"{actual_identity.step!r} -- refusing to use a basis fit against a different training step "
            "of the conditioner, even though its weights hash matches (Codex re-audit of commit "
            "f7bb8a1, launch blocker #5: bind the basis sidecar to canonical bundle identity/step)"
        )
    # Codex re-audit of commit 2162ff4, finding #5: "record and validate
    # canonical conditioner bundle_id/manifest SHA/step" -- weights sha256
    # + step alone cannot distinguish two DIFFERENT bundles that happen
    # to save byte-identical weights at the same step.
    recorded_checkpoint_bundle_id = _require_field("architecture3_checkpoint_bundle_id")
    recorded_checkpoint_manifest_sha256 = _require_field("architecture3_checkpoint_manifest_sha256")
    if actual_identity.bundle_dir != recorded_checkpoint_bundle_id:
        raise ValueError(
            f"gene residual basis at {path} was fit against Architecture 3 checkpoint bundle "
            f"{recorded_checkpoint_bundle_id!r}, but the checkpoint configured for this run resolves to "
            f"bundle {actual_identity.bundle_dir!r} -- refusing to use a basis fit against a different bundle"
        )
    if actual_identity.manifest_sha256 != recorded_checkpoint_manifest_sha256:
        raise ValueError(
            f"gene residual basis at {path} was fit against Architecture 3 checkpoint bundle manifest "
            f"sha256 {recorded_checkpoint_manifest_sha256!r}, but the checkpoint configured for this run "
            f"resolves to manifest sha256 {actual_identity.manifest_sha256!r} -- refusing to use a basis "
            "fit against a different bundle manifest"
        )

    # Codex re-audit of commit 2162ff4, finding #5: "compare recorded
    # Architecture 3 config identity with the conditioner bundle's run_
    # manifest." The sidecar's own `architecture3_config_identity_
    # fingerprint` previously had NO current value to compare against
    # from an Architecture 4 run's perspective -- but the CONDITIONER
    # BUNDLE's own bound run_manifest.json (now directly reachable, since
    # `conditioner_checkpoint_dir` is already resolved above) records the
    # EXACT config Architecture 3 was actually trained under, which is a
    # real, valid value to compare the sidecar's recorded fingerprint
    # against -- closing the gap where the recorded fingerprint could be
    # wrong (or from a stale/different Architecture 3 run) even while
    # every other conditioner-identity field matches. Reads from
    # `actual_identity.resolved_dir` (the SAME already-pinned, immutable
    # bundle directory used above), never `conditioner_checkpoint_dir`
    # directly -- a second raw resolution of the mutable path here is
    # exactly the TOCTOU finding #3 (Codex re-audit of commit 7a2d819)
    # closes.
    conditioner_run_manifest = checkpoint_module.load_checkpoint_run_manifest(actual_identity.resolved_dir)
    if conditioner_run_manifest is None:
        raise ValueError(
            f"gene residual basis at {path}: the configured Architecture 3 conditioner checkpoint at "
            f"{conditioner_checkpoint_dir} has no run_manifest.json bound inside its resolved bundle -- "
            "cannot validate the basis sidecar's recorded architecture3_config_identity_fingerprint "
            "against it"
        )
    recorded_architecture3_config_identity_fingerprint = provenance["architecture3_config_identity_fingerprint"]
    conditioner_config_identity_fingerprint = conditioner_run_manifest.get("config_identity_fingerprint")
    if recorded_architecture3_config_identity_fingerprint != conditioner_config_identity_fingerprint:
        raise ValueError(
            f"gene residual basis at {path} records architecture3_config_identity_fingerprint="
            f"{recorded_architecture3_config_identity_fingerprint!r}, but the configured Architecture 3 "
            f"conditioner checkpoint's own run_manifest.json records config_identity_fingerprint="
            f"{conditioner_config_identity_fingerprint!r} -- refusing to use a basis whose recorded "
            "Architecture 3 config identity disagrees with the conditioner it claims to have been fit from"
        )

    # Codex re-audit of commit 2162ff4, finding #5: "record the numerical
    # basis SHA256 and verify it when loading" -- `verify_gene_residual_
    # basis` above only checks the GENE NAMES hash; nothing previously
    # bound the sidecar to the basis's own NUMERICAL content, so a valid
    # sidecar could sit beside a different, same-shape, re-fit basis file
    # without detection.
    recorded_basis_sha256 = _require_field("gene_residual_basis_sha256")
    actual_basis_sha256 = hashlib.sha256(np.ascontiguousarray(basis.basis.detach().cpu().numpy()).tobytes()).hexdigest()
    if actual_basis_sha256 != recorded_basis_sha256:
        raise ValueError(
            f"gene residual basis at {path}: the basis file's own numerical content (sha256 "
            f"{actual_basis_sha256!r}) does not match its provenance sidecar's recorded "
            f"gene_residual_basis_sha256 ({recorded_basis_sha256!r}) -- refusing to use a basis file that "
            "does not match the one its own provenance describes"
        )

    # Launch blocker #5 (refined by Codex re-audit of commit 2162ff4,
    # finding #5: "cache_content_by_sample not required (missing content
    # silently disables comparison); missing individual training samples
    # in that mapping skipped" -- the SIDECAR's own recorded mapping is
    # now mandatory and must cover every training sample it claims to
    # have been fit on (below); a basis whose own provenance never
    # actually recorded a training sample's cache identity must never be
    # trusted, regardless of whether any particular caller later asks to
    # compare it.
    #
    # The CALLER's currently-loaded `cache_content_by_sample`, in
    # contrast, is legitimately partial: `evaluate_gen3_checkpoint` and
    # `fit_and_save_architecture4_basis` only preflight the sample_ids
    # relevant to what THEY are doing (e.g. evaluation preflights only
    # the split being evaluated, never the training samples this basis
    # was fit on) -- this function is also called during a real
    # Architecture 4 evaluation, which never re-loads training samples at
    # all. Requiring the caller's mapping to cover every training sample
    # unconditionally would wrongly reject every such legitimate,
    # narrower-scoped caller. Comparison therefore stays per-sample,
    # over the INTERSECTION of the two mappings -- a training sample the
    # caller did not touch this call is skipped (nothing to compare
    # against), never failed; a training sample the caller DID touch is
    # compared for real and any mismatch still fails closed. Per-sample
    # binding -- never the single combined preflight fingerprint, which
    # this Architecture 4 run's own sample set need not exactly match the
    # Architecture 3 basis-fitting run's (see
    # `verify_full_checkpoint_identity`'s docstring for the identical
    # reasoning).
    recorded_cache_content_by_sample = _require_field("cache_content_by_sample")
    missing_recorded = sorted(set(expected_train_sample_ids) - set(recorded_cache_content_by_sample))
    if missing_recorded:
        raise ValueError(
            f"gene residual basis provenance sidecar {provenance_path} is missing recorded cache content "
            f"identity for training sample(s) {missing_recorded} -- refusing to use a basis whose recorded "
            "cache identity does not cover every training sample it was fit on"
        )
    # Codex re-audit of commit 66d65f2, finding #4: "the basis sidecar's
    # training-sample cache identities are required, but they are not
    # compared directly against the canonical Architecture 3 run
    # manifest. That comparison is available and should be exact."
    # Confirmed real: the ONLY comparison for the sidecar's recorded
    # cache_content_by_sample was against the CALLER's own, currently-
    # loaded mapping (below) -- for an Architecture 4 training run this
    # happens to be that run's own preflight over train_ids+validation_ids,
    # which coincidentally overlaps the basis-fitting scope but is never
    # DEFINITIONALLY bound to what Architecture 3 was actually trained on.
    # `conditioner_run_manifest` (already loaded above, from the SAME
    # canonical, bundle-verified path `checkpoint_module.load_checkpoint_
    # run_manifest` uses everywhere else in this codebase) carries the
    # conditioner's OWN recorded `cache_preflight_report.cache_content_by_sample`
    # -- the actual ground truth for what Architecture 3 was trained
    # against -- and is now compared directly, exactly, for every training
    # sample this basis was fit on.
    conditioner_cache_content_by_sample = (
        (conditioner_run_manifest.get("cache_preflight_report") or {}).get("cache_content_by_sample") or {}
    )
    missing_in_conditioner_manifest = sorted(
        set(expected_train_sample_ids) - set(conditioner_cache_content_by_sample)
    )
    if missing_in_conditioner_manifest:
        raise ValueError(
            f"the configured Architecture 3 conditioner checkpoint's own canonical run_manifest.json is "
            f"missing recorded cache content identity for training sample(s) {missing_in_conditioner_manifest} "
            f"-- cannot verify the basis at {path} was fit against cache content that genuinely matches what "
            "Architecture 3 was trained on"
        )
    for sample_id in expected_train_sample_ids:
        recorded_content = recorded_cache_content_by_sample[sample_id]
        conditioner_content = conditioner_cache_content_by_sample[sample_id]
        if recorded_content != conditioner_content:
            raise ValueError(
                f"gene residual basis at {path} was fit using cache content for sample {sample_id!r} that "
                "does not match the configured Architecture 3 conditioner checkpoint's own canonical "
                "run_manifest.json recorded cache content for that same sample -- refusing to use a basis "
                "whose recorded cache-content provenance disagrees with the conditioner it claims to have "
                "been fit from (Codex re-audit of commit 66d65f2, finding #4)"
            )
    if cache_content_by_sample:
        for sample_id in expected_train_sample_ids:
            current_content = cache_content_by_sample.get(sample_id)
            if current_content is None:
                continue  # this caller's own preflight never touched this training sample -- nothing to compare
            recorded_content = recorded_cache_content_by_sample[sample_id]
            if recorded_content != current_content:
                raise ValueError(
                    f"gene residual basis at {path} was fit using cache content for sample {sample_id!r} "
                    f"that does not match this run's currently loaded cache content for that sample -- "
                    "refusing to use a basis fit against a stale or regenerated cache (Codex re-audit of "
                    "commit f7bb8a1, launch blocker #5)"
                )
    return basis, gene_names


def maybe_load_pretrained_conditioner_for_architecture4(
    model, config: dict, architecture_id: str, gene_names: list[str], smoke: bool,
    *, require_for_smoke: bool = False,
    conditioner_identity: "checkpoint_module.CheckpointIdentity | None" = None,
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
    _unloaded = {
        "loaded": False, "checkpoint_dir": None, "checkpoint_sha256": None, "checkpoint_step": None,
        "checkpoint_bundle_id": None, "checkpoint_manifest_sha256": None,
    }
    if architecture_id != "4":
        return dict(_unloaded)
    checkpoint_dir = (config.get("required_fingerprints") or {}).get("architecture3_conditioner_checkpoint")
    if not checkpoint_dir:
        if smoke and not require_for_smoke:
            return dict(_unloaded)
        raise ValueError(
            "Architecture 4 requires required_fingerprints.architecture3_conditioner_checkpoint -- "
            "a real, already-trained Architecture 3 checkpoint_dir. Train Architecture 3 to "
            "completion first (scripts/fit_architecture4_residual_basis.py then needs that exact "
            "checkpoint to fit a real gene-residual basis); Architecture 4 must never start "
            "training from a random or merely synchronized-init conditioner"
        )
    # Codex re-audit of commit 7a2d819, finding #3: "Apply the same
    # resolve-once principle to Architecture 4's conditioner/basis
    # orchestration." Confirmed real: `checkpoint_dir` here (the SAME
    # mutable `required_fingerprints.architecture3_conditioner_checkpoint`
    # path `maybe_load_gene_basis` also resolves, to validate the basis
    # against) was independently resolved a THIRD time by this function's
    # OWN `resolve_checkpoint_identity` call below, on top of whatever
    # `verify_gene_names`/`load_trainable_state` each resolve internally
    # -- across the two functions together, up to five separate
    # resolutions of one mutable path. `build_model_for_inference` now
    # resolves this path EXACTLY ONCE and passes the result down as
    # `conditioner_identity`; when given, `identity.resolved_dir` (an
    # immutable, already-verified bundle directory) is used for
    # `verify_gene_names`/`load_trainable_state` INSTEAD of the mutable
    # `checkpoint_dir`, and the identity itself is reused directly rather
    # than re-resolved a final time for the returned info dict --
    # structurally guaranteeing the SAME bundle is what gets gene-name-
    # checked, loaded onto the model, AND recorded, never three
    # potentially-different ones. Stays optional (falls back to resolving
    # `checkpoint_dir` directly) so this function remains independently
    # callable/testable without a caller pinning one first.
    identity = conditioner_identity or checkpoint_module.resolve_checkpoint_identity(checkpoint_dir)
    load_source = identity.resolved_dir if conditioner_identity is not None else checkpoint_dir
    checkpoint_module.verify_gene_names(load_source, gene_names)
    checkpoint_module.load_trainable_state(model.conditioner, load_source)
    freeze = bool(((config.get("model") or {}).get("params") or {}).get("freeze_conditioner_initially", True))
    if freeze:
        for p in model.conditioner.parameters():
            p.requires_grad = False
    return {
        "loaded": True, "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_sha256": identity.weights_sha256, "checkpoint_step": identity.step,
        # Codex re-audit of commit 2162ff4, finding #4: "bind conditioner
        # bundle_id + manifest_sha256 as well as weights/step" -- weights
        # sha256 + step alone cannot distinguish two DIFFERENT bundles
        # that happen to save byte-identical weights at the same step
        # (e.g. a resumed run re-saving with an unchanged optimizer
        # state); the bundle's own identity closes that gap.
        "checkpoint_bundle_id": identity.bundle_dir, "checkpoint_manifest_sha256": identity.manifest_sha256,
    }


def build_model_for_inference(
    config: dict, *, gene_names: list[str], device: torch.device,
    checkpoint_dir: str | Path | None = None, smoke: bool = False, staged_smoke: bool = False,
    dataset_manifest: dict | None = None, cache_content_by_sample: dict[str, dict] | None = None,
    allow_code_drift: bool = False,
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
    exactly as every other checkpoint load in this package.

    Item 5 (six-launch-blocker audit) + Integration audit follow-up: a
    Gen4/Gen5 config (`model.arm` present, never set by a Gen3 config)
    dispatches to `gen4.trainer_adapter.build_gen4_or_gen5_model_for_
    inference` instead of everything below -- the ONE minimal adapter
    point that lets every existing caller of THIS function (the trainer
    loop, the evaluator, the overfit gate) construct a real Gen4/Gen5
    model with zero changes of their own."""
    if (config.get("model") or {}).get("arm") is not None:
        from gen3_multiscale.gen4.trainer_adapter import build_gen4_or_gen5_model_for_inference

        return build_gen4_or_gen5_model_for_inference(
            config, gene_names=gene_names, device=device, checkpoint_dir=checkpoint_dir,
            smoke=smoke, staged_smoke=staged_smoke, dataset_manifest=dataset_manifest,
            cache_content_by_sample=cache_content_by_sample, allow_code_drift=allow_code_drift,
        )
    architecture_id = str((config.get("model") or {}).get("architecture", ""))
    training_cfg = config.get("training") or {}
    data_cfg = config.get("data") or {}
    n_genes = len(gene_names)
    gex_feature_dim = int(data_cfg.get("gex_feature_dim", 128))
    seed = int(training_cfg.get("seed", 0))

    slide_encoder, gigapath_checkpoint_sha256 = maybe_build_slide_encoder(config)
    # Codex re-audit of commit 7a2d819, finding #3: "Apply the same
    # resolve-once principle to Architecture 4's conditioner/basis
    # orchestration." The mutable `required_fingerprints.architecture3_
    # conditioner_checkpoint` path is resolved to an immutable bundle
    # identity EXACTLY ONCE here -- before `maybe_load_gene_basis`
    # validates the basis's provenance against it, and before
    # `maybe_load_pretrained_conditioner_for_architecture4` loads weights
    # from it -- and the SAME `pinned_conditioner_identity` is threaded
    # through both calls below, so a concurrent Architecture 3 checkpoint
    # save between them can never make basis validation describe a
    # DIFFERENT bundle than what actually gets loaded onto the model.
    # `needs_conditioner` mirrors exactly the condition under which BOTH
    # downstream functions actually consult the conditioner checkpoint
    # (skipped for a plain, construction-only `--smoke` run, required
    # otherwise -- including the real staged smoke).
    conditioner_checkpoint_dir = (config.get("required_fingerprints") or {}).get("architecture3_conditioner_checkpoint")
    needs_conditioner = architecture_id == "4" and not (smoke and not staged_smoke)
    pinned_conditioner_identity = None
    if needs_conditioner and conditioner_checkpoint_dir:
        pinned_conditioner_identity = checkpoint_module.resolve_checkpoint_identity(conditioner_checkpoint_dir)
    # staged_smoke behaves like a non-smoke run for provenance purposes --
    # its whole point is to validate the real, provenanced artifacts, not
    # bypass that validation the way a construction-only smoke may.
    gene_basis, resolved_gene_names = maybe_load_gene_basis(
        config, gene_names, dataset_manifest=dataset_manifest, smoke=smoke and not staged_smoke,
        cache_content_by_sample=cache_content_by_sample, conditioner_identity=pinned_conditioner_identity,
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
        conditioner_identity=pinned_conditioner_identity,
    )

    # Codex re-audit of commit 57f0e3c: "harden build_model_for_inference()
    # itself: whenever checkpoint_dir is supplied, resolve it once
    # internally and use the resulting immutable directory for both
    # gene-name verification and weight loading. Return that identity in
    # its info object." Confirmed real: `verify_gene_names`/`load_
    # trainable_state` each independently resolved `checkpoint_dir` (a
    # caller-supplied path that may itself still be mutable, e.g. a raw
    # `checkpoint_dir` rather than an already-pinned bundle) through
    # `checkpoint.py::_resolve_checkpoint_source` -- two SEPARATE
    # resolutions of the same path within this one function call, each
    # of which could in principle observe a different bundle if the
    # checkpoint were rewritten in between. Resolving once here and
    # reusing `.resolved_dir` for both closes that gap regardless of
    # whether the caller passed a raw mutable path or an already-pinned
    # one (resolving an already-pinned bundle directory is a safe,
    # idempotent no-op -- `checkpoint.py::_resolve_checkpoint_source`'s
    # own documented "caller directly resolves a HISTORY BUNDLE'S OWN
    # path" exception).
    resolved_checkpoint_identity = None
    if checkpoint_dir is not None:
        resolved_checkpoint_identity = checkpoint_module.resolve_checkpoint_identity(checkpoint_dir)
        checkpoint_module.verify_gene_names(resolved_checkpoint_identity.resolved_dir, gene_names)
        checkpoint_module.load_trainable_state(model, resolved_checkpoint_identity.resolved_dir)

    info = {
        "architecture_id": architecture_id,
        "kind": model_kind_for_architecture_id(architecture_id),
        "gene_basis": gene_basis,
        "gigapath_checkpoint_sha256": gigapath_checkpoint_sha256,
        "synchronized_init_manifest_path": synchronized_init_manifest_path,
        "architecture3_conditioner": conditioner_info,
        "checkpoint_identity": resolved_checkpoint_identity,
    }
    return model, info


def model_kind_for_architecture_id(architecture_id: str) -> str:
    """Gen3's own numeric `model.architecture` mapped onto the generic
    `model.kind` vocabulary (`"conditioner"` / `"flow"`) that Gen4/Gen5
    configs already carry directly via their own `model.kind` field.
    Architecture 4 IS Gen3's residual-flow architecture -- deterministic
    reconstruction training/`forward()` prediction for architectures 1-3
    was always exactly what `"conditioner"` means below; Architecture 4's
    `compute_losses()["flow_loss"]` training and
    `sample_predictive_distribution()` evaluation was always exactly what
    `"flow"` means. Integration audit finding #1: this lets
    `predict_for_metrics`/`compute_step_losses`/
    `compute_deterministic_reconstruction_losses` dispatch on ONE shared
    `kind` vocabulary for both Gen3 and Gen4/Gen5, instead of Gen3's
    numeric `architecture_id` -- a Gen4 flow config has no
    `model.architecture` at all, so the prior `architecture_id == "4"`
    check silently fell through to the `forward()` branch for it, which
    for `Gen4ResidualFlowModel` returns the FROZEN CONDITIONER's own
    output (by design, for the single-conditioner-pass loss discipline),
    not the flow model's real prediction -- and `Gen5LatentFlowModel` has
    no `forward()` at all, so it would have raised `NotImplementedError`
    outright."""
    return "flow" if str(architecture_id) == "4" else "conditioner"


def model_identifier_for_config(config: dict) -> str:
    """Stable model identity shared by Gen3, Gen4, and Gen5 artifacts."""
    model_cfg = config.get("model") or {}
    identifier = model_cfg.get("architecture")
    if identifier is None:
        identifier = model_cfg.get("arm")
    if identifier is None or str(identifier) == "":
        raise ValueError("model must declare either architecture (Gen3) or arm (Gen4/Gen5)")
    return str(identifier)


def model_kind_for_config(config: dict) -> str:
    """Return the generic execution kind for any supported generation."""
    model_cfg = config.get("model") or {}
    kind = model_cfg.get("kind")
    if kind is not None:
        kind = str(kind)
        if kind not in {"conditioner", "flow", "latent_flow"}:
            raise ValueError(f"unsupported model.kind {kind!r}")
        return kind
    return model_kind_for_architecture_id(str(model_cfg.get("architecture", "")))


def required_artifact_file_sha256(config: dict) -> dict[str, str]:
    """Hash every configured regular-file model artifact.

    Cache content is bound separately by preflight, and transactional
    conditioner checkpoints are bound by their verified bundle identity.
    This map covers the remaining mutable file paths (UNI2,
    scFoundation, STPath, LongNet, residual bases, and Gen5's
    autoencoder) so resume/evaluation cannot silently use replaced
    bytes at the same path.
    """
    hashes: dict[str, str] = {}
    for name, value in sorted((config.get("required_fingerprints") or {}).items()):
        if not value:
            continue
        path = Path(str(value))
        if path.is_file():
            hashes[str(name)] = file_sha256(path)
    return hashes


def predict_for_metrics(
    kind: str, model, inputs, *, generator: torch.Generator | None = None, n_samples: int | None = None,
) -> dict:
    """The metric-basis prediction for `inputs`, dispatched by `kind`
    (`"conditioner"`, `"flow"`, or `"latent_flow"` -- see
    `model_kind_for_architecture_id`'s docstring and Integration audit
    finding #1). Adam's Step 6 audit #1 of commit a32051b: "Architecture
    4 validation/evaluation/overfit must evaluate a deterministic
    fixed-seed predictive mean from `sample_predictive_distribution`, not
    `forward()`'s frozen conditioner. Log conditioner-only metrics
    separately." A `"conditioner"` model's `forward()` already IS the
    real model, so its `expression` output is used directly and there is
    no separate "conditioner-only" number to report.

    Returns `{"expression": <the metric-basis prediction>}` for
    `"conditioner"`, plus (`"flow"`/`"latent_flow"` only)
    `"conditioner_only_expression"` (the frozen conditioner's own,
    secondary, mean -- never the reported headline metric),
    `"predictive_std"` (per-query sampled-residual uncertainty) and
    `"predictive_samples"` (the raw per-draw field, `[n_samples,
    n_query, n_genes]`, for empirical-quantile calibration). `generator`,
    when given, makes the flow's stochastic sampling reproducible --
    required for exact resume/evaluation consistency (same audit item).
    `n_samples`, when given, overrides the model's own configured
    `n_flow_samples` -- Codex re-audit of commit f7bb8a1, secondary fix
    #4: "Treat Gaussian coverage from 8 flow samples as approximate;
    preferably use empirical quantiles with a larger configurable sample
    count." A caller that wants a more reliable calibration estimate than
    the training-time default can ask for more draws here without
    changing `n_flow_samples` itself.

    A `"flow"`/`"latent_flow"` model's frozen conditioner is read via
    `model.conditioner(inputs)` directly, never `model(inputs)` -- Gen5's
    `Gen5LatentFlowModel` has no ordinary `forward()` at all (only Gen3's
    `Architecture4`/Gen4's `Gen4ResidualFlowModel` happen to alias their
    own `forward()` to the conditioner pass; relying on that alias is
    exactly the bug this function's genericization fixes)."""
    if kind in ("flow", "latent_flow"):
        predictive = model.sample_predictive_distribution(inputs, n_samples=n_samples, generator=generator)
        conditioner_expression = predictive.get("deterministic_mean")
        if conditioner_expression is None:
            conditioner_expression = predictive.get(
                "conditioner_expression_diagnostic_only"
            )
        if conditioner_expression is None:
            # Backward-compatible guard for an external flow
            # implementation that follows the sampling API but predates
            # the deterministic_mean return field.  Every in-repository
            # Gen3/Gen4/Gen5 flow supplies it, so production evaluation
            # performs only one conditioner pass per prediction.
            conditioner_expression = model.conditioner(inputs)["expression"]
        return {
            "expression": predictive["predictive_mean"],
            "conditioner_only_expression": conditioner_expression,
            "predictive_std": predictive["predictive_std"],
            "predictive_samples": predictive["predictive_samples"],
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


def common_random_validation_seed(seed: int, stable_key: str) -> int:
    """Codex re-audit of commit 90f853e, launch blocker #6 (refined by
    the re-audit of commit f7bb8a1's secondary fix #1): "Seed
    Architecture 4 validation/evaluation from a stable identity:
    evaluation seed + sample_id + stratum + query_fingerprint, never
    item index." `item_index` (the prior version's key) is `val_loader`'s
    own enumeration position -- stable only as long as the validation
    mask BANK's on-disk record ordering never changes; a mask bank
    self-heal/regeneration that reorders (without changing) the same
    logical set of realized masks would silently reassign every item's
    noise draw to a DIFFERENT held-out mask. `stable_key` is the
    caller's own `f"{sample_id}:{stratum}:{query_fingerprint}"` (see
    `Gen3SpatialFieldDataset.item_identity`) -- a content-derived
    identity for the EXACT realized mask itself, invariant to
    reordering. Extracted into its own named function so this
    determinism/stability property is directly unit-testable, not just
    observable end-to-end: two calls with the SAME `(seed, stable_key)`
    return the SAME value regardless of training step, evaluation
    order, or mask-bank record order."""
    digest = hashlib.sha256(f"gen3-common-random-validation:{int(seed)}:{stable_key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**63)


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


def compute_step_losses(kind: str, model, inputs, target_expression: torch.Tensor,
                         query_coords: torch.Tensor, gradient_weight: float, k_neighbors: int,
                         flow_weight: float = 1.0, per_gene_scale: torch.Tensor | None = None,
                         flow_generator: torch.Generator | None = None) -> dict:
    """Dispatched by `kind` (see `model_kind_for_architecture_id`'s
    docstring, Integration audit finding #1): `"conditioner"` models
    (Gen3 Architectures 1/2/3, Gen4 conditioner arms) share one
    deterministic reconstruction objective (models/losses.py).

    `"flow"` models (Gen3 Architecture 4, Gen4 flow arms) additionally
    add their stopped-gradient flow-matching loss to a reconstruction
    loss computed from the SAME single conditioner pass
    (model.compute_losses) so the reconstruction and flow losses agree on
    the same dropout mask (3rd Codex re-audit finding, CONTRACT.md) --
    `model.compute_losses`'s own `out["expression"]` is the conditioner's
    deterministic mean, real and safe to combine with the flow loss for
    these models.

    `"latent_flow"` models (Gen5) train ONLY `flow_weight *
    out["flow_loss"]` -- GEN5_CONTRACT.md section 5 is explicit that
    Gen5's conditioner is already a frozen, separately-pretrained
    artifact (loaded from a real Gen4 conditioner checkpoint, never
    trained here) and that its own expression output must NEVER be
    combined with the flow loss; `Gen5LatentFlowModel.compute_losses`
    reflects this directly -- it returns
    `"conditioner_expression_diagnostic_only"`, not `"expression"`, and
    that key is intentionally never read here. There is no separate
    reconstruction term to add: the flow loss IS the entire training
    objective for a latent_flow model.

    `flow_weight` was previously hardcoded at every call site (Adam's
    Step 6 audit #9, confirmed) -- now always threaded through from
    `loss.flow_weight` in the resolved config. `flow_generator`, when
    given, makes the flow loss's own random t/x0 draw reproducible (used
    for validation logging only -- see
    `compute_deterministic_reconstruction_losses` for the metric actually
    used for model selection, audit #8)."""
    if kind == "flow":
        out = model.compute_losses(inputs, target_expression, generator=flow_generator)
        recon = combined_reconstruction_loss(
            out["expression"], target_expression, query_coords,
            gradient_weight=gradient_weight, k_neighbors=k_neighbors, per_gene_scale=per_gene_scale,
        )
        total = recon["total"] + flow_weight * out["flow_loss"]
        return {"total": total, "primary": recon["primary"], "gradient": recon["gradient"], "flow_loss": out["flow_loss"]}
    if kind == "latent_flow":
        out = model.compute_losses(inputs, target_expression, generator=flow_generator)
        total = flow_weight * out["flow_loss"]
        return {"total": total, "flow_loss": out["flow_loss"]}
    out = model(inputs)
    return combined_reconstruction_loss(
        out["expression"], target_expression, query_coords,
        gradient_weight=gradient_weight, k_neighbors=k_neighbors, per_gene_scale=per_gene_scale,
    )


def compute_deterministic_reconstruction_losses(
    kind: str, model, inputs, target_expression: torch.Tensor, query_coords: torch.Tensor,
    gradient_weight: float, k_neighbors: int, per_gene_scale: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> dict:
    """The reconstruction objective used for VALIDATION/model-selection
    (Adam's Step 6 audit #8, then corrected by audit #1 of commit
    a32051b). For `"conditioner"` models, `model(inputs)` (their real
    forward pass) is used directly. For `"flow"`/`"latent_flow"` models,
    audit #1 is explicit: "Architecture 4 validation/evaluation/overfit
    must evaluate a deterministic fixed-seed predictive mean from
    `sample_predictive_distribution`, not `forward()`'s frozen
    conditioner." `forward()`/the conditioner pass never touches the
    trained flow apparatus at all -- selecting on it would let the flow
    weights train for hours while the ONLY metric ever checked is blind
    to whether they learned anything. `predict_for_metrics` (with a
    caller-supplied, fixed-per-step `generator` for resume-exact
    reproducibility) is now used for every `kind`; the result
    additionally carries `conditioner_only_total` for `"flow"`/
    `"latent_flow"` -- a SECONDARY diagnostic logged alongside the real
    metric, never used for selection."""
    prediction = predict_for_metrics(kind, model, inputs, generator=generator)
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


# `resolved_config`/`config_fingerprint`/`config_identity_fingerprint`
# moved to `gen3_multiscale/config_identity.py` (Codex re-audit of
# commit 57f0e3c) and are imported/re-exported at the top of this file.


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


_CODE_STATE_IGNORED_PATH_PARTS = frozenset({
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git",
    "cache", "caches", "checkpoint", "checkpoints", "results", "result",
    "output", "outputs", "logs", "log", "wandb", "hest1k_cache", "hest1k",
    "evaluation_masks", "runs", "tmp",
})
_CODE_STATE_RELEVANT_SUFFIXES = frozenset({
    ".py", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".sh", ".md", ".txt",
})


def _worktree_diff_hash() -> str | None:
    """Best-effort sha256 over `git diff HEAD` (every tracked-file
    change, staged or not, relative to the current commit) plus the
    CONTENTS of relevant untracked source/config files -- Codex re-audit
    of commit 90f853e, launch blocker #10: "Bind code state on resume:
    exact commit plus clean-worktree status/diff hash." None (not an
    error) outside a git checkout or if git itself is unavailable,
    exactly like `_code_commit_hash`; an all-clean worktree still returns
    a real, stable hash, never None, so "clean" is distinguishable from
    "unknown".

    Codex re-audit of commit f7bb8a1, launch blocker #8: "Make code-state
    binding operational: hash tracked diffs plus contents of relevant
    untracked source/config files; ignore generated caches/results." The
    PRIOR version hashed `git status --porcelain`'s raw text -- which
    lists every untracked path's NAME but never its CONTENT, so editing
    an already-untracked file (a real, confirmed gap: a new source file
    added but not yet `git add`-ed, then edited again, changes nothing
    about `git status`'s own output) left this hash unchanged; and it
    included every untracked path repo-wide, so an unrelated generated
    data/cache/results/checkpoint directory appearing (nothing to do with
    CODE identity at all) could spuriously flag "code drift" and block a
    legitimate resume. Fixed: any path with an ignored directory name
    anywhere in it (`_CODE_STATE_IGNORED_PATH_PARTS`) is excluded
    entirely -- its presence, absence, or content never affects this
    hash. Every SURVIVING untracked path with a source/config-like
    extension (`_CODE_STATE_RELEVANT_SUFFIXES`) has its actual file
    CONTENT read and hashed, not merely its name; surviving paths that
    don't match (binary/data files that happen to sit outside an ignored
    directory) still contribute their NAME (so their appearance is not
    silently invisible), just not their content.

    Codex re-audit of commit 2162ff4, finding #6: "Untracked code inside
    a new directory is not reliably hashed. Plain `git status
    --porcelain` commonly reports `?? new_directory/`, not its files.
    Editing `new_directory/module.py` would then leave the worktree hash
    unchanged." Confirmed real and reproduced directly: `git status
    --porcelain` collapses an entirely-untracked directory into ONE
    summary line naming the directory, never descending into it -- the
    PRIOR version's per-line loop over that output could therefore never
    even SEE `new_directory/module.py` as a path to consider hashing,
    let alone hash its content, regardless of the ignored-path-parts
    filter or the relevant-suffix check. Untracked-file enumeration now
    uses `git ls-files --others --exclude-standard` instead, which lists
    every individual untracked file's real path (already respecting
    `.gitignore`, so generated files a project has chosen to ignore are
    excluded the standard way, on top of `_CODE_STATE_IGNORED_PATH_
    PARTS`'s own defense-in-depth for paths that are untracked but not
    gitignored). `git status --porcelain` is no longer used at all --
    everything it could report about TRACKED files is already covered by
    `git diff HEAD`."""
    repo_root = Path(__file__).resolve().parents[2]
    try:
        diff = subprocess.run(
            ["git", "diff", "HEAD"], cwd=repo_root, capture_output=True, text=True, timeout=10,
        )
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"], cwd=repo_root,
            capture_output=True, text=True, timeout=10,
        )
        if diff.returncode != 0 or untracked.returncode != 0:
            return None
        hasher = hashlib.sha256()
        hasher.update(diff.stdout.encode("utf-8"))
        hasher.update(b"\x00")
        relevant_paths = []
        for relpath in untracked.stdout.splitlines():
            relpath = relpath.strip()
            if not relpath:
                continue
            if any(part in _CODE_STATE_IGNORED_PATH_PARTS for part in Path(relpath).parts):
                continue  # a generated cache/results/output/checkpoint path -- never part of code identity
            relevant_paths.append(relpath)
        for relpath in sorted(relevant_paths):
            hasher.update(relpath.encode("utf-8"))
            hasher.update(b"\x00")
            full_path = repo_root / relpath
            if full_path.suffix in _CODE_STATE_RELEVANT_SUFFIXES:
                try:
                    hasher.update(full_path.read_bytes())
                except OSError:
                    pass
            hasher.update(b"\x00")
        return hasher.hexdigest()
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
    staged_conditioner_info: dict | None = None,
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
    model_identifier = model_identifier_for_config(config)
    model_kind = model_kind_for_config(config)
    staged_info = staged_conditioner_info or architecture3_conditioner_info or {}
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
        "model_architecture": model_identifier,
        "model_kind": model_kind,
        "model_arm": (config.get("model") or {}).get("arm"),
        "required_artifact_file_sha256": required_artifact_file_sha256(config),
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
        # Codex re-audit of commit 2162ff4, finding #4: "bind conditioner
        # bundle_id + manifest_sha256 as well as weights/step."
        "architecture3_conditioner_checkpoint_bundle_id": (
            (architecture3_conditioner_info or {}).get("checkpoint_bundle_id")
        ),
        "architecture3_conditioner_checkpoint_manifest_sha256": (
            (architecture3_conditioner_info or {}).get("checkpoint_manifest_sha256")
        ),
        # Gen4/Gen5 use the same staged-conditioner discipline as Gen3
        # Architecture 4, but their artifact is named
        # gen4_conditioner_checkpoint. Bind it through one generic set
        # of fields so resume and evaluation share the same checks.
        "staged_conditioner_checkpoint_sha256": staged_info.get("checkpoint_sha256"),
        "staged_conditioner_checkpoint_step": staged_info.get("checkpoint_step"),
        "staged_conditioner_checkpoint_bundle_id": staged_info.get("checkpoint_bundle_id"),
        "staged_conditioner_checkpoint_manifest_sha256": staged_info.get(
            "checkpoint_manifest_sha256"
        ),
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
    "config_identity_fingerprint", "dataset_manifest_fingerprint", "gene_panel_hash",
    "model_architecture", "model_kind", "model_arm", "required_artifact_file_sha256",
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
    # Codex re-audit of commit 2162ff4, finding #4: "bind conditioner
    # bundle_id + manifest_sha256 as well as weights/step" -- weights
    # sha256 + step alone cannot distinguish two DIFFERENT bundles that
    # happen to save byte-identical weights at the same step.
    "architecture3_conditioner_checkpoint_bundle_id", "architecture3_conditioner_checkpoint_manifest_sha256",
    "staged_conditioner_checkpoint_sha256", "staged_conditioner_checkpoint_step",
    "staged_conditioner_checkpoint_bundle_id", "staged_conditioner_checkpoint_manifest_sha256",
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
    resumed, for reasons none of the other fields here can see.
    `allow_code_drift=True` (threaded from `run_training(...,
    allow_code_drift=True)` / `--allow-code-drift`) is the explicit
    override: it does not silence the check, it is RECORDED by the
    caller into the new run manifest's `code_drift_acknowledged` field so
    the override is scientifically visible in the artifact itself, never
    a silent bypass.

    Codex re-audit of commit f7bb8a1, launch blocker #8: "fail closed
    when code identity is unknown unless an explicit recorded override is
    supplied." The PRIOR version SKIPPED this entire check whenever the
    OLD manifest recorded no commit at all (`old_commit is not None and
    ...`) -- a checkpoint from outside a git checkout, or one predating
    this field, was silently treated as "nothing to verify," which meant
    a resume against such a checkpoint could run under ARBITRARY code
    changes with no check at all, not even the override requirement.
    Fixed: an unknown commit on EITHER side now counts as drift itself
    (there is nothing to positively confirm code identity matched), so it
    requires the same explicit `allow_code_drift=True` override as a
    confirmed change -- this trainer would rather force an operator in a
    non-git or git-unavailable environment to pass the override on every
    resume than silently trust an unverifiable one."""
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
    code_identity_unknown = old_commit is None or new_commit is None
    code_drifted = code_identity_unknown or old_commit != new_commit or old_diff_hash != new_diff_hash
    if code_drifted and not allow_code_drift:
        reason = "code identity is unknown on at least one side" if code_identity_unknown else "code state changed"
        raise ValueError(
            f"resume refused: {reason} since the last checkpoint at "
            f"{old_run_manifest.get('checkpoint_dir')} (commit {old_commit!r} -> {new_commit!r}, "
            f"worktree_diff_hash {old_diff_hash!r} -> {new_diff_hash!r}) -- pass allow_code_drift=True "
            "(run_training(..., allow_code_drift=True) / --allow-code-drift) if resuming under different or "
            "unverifiable code is genuinely intended; the override is recorded in the new run_manifest.json, "
            "never silent"
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
    model, config: dict, gene_names: list[str], best_dir: str | Path, *, step: int, val_loss: float,
    run_manifest: dict,
) -> Path:
    """`best/` as a real transactional `checkpoint.py` bundle -- Codex
    re-audit of commit f7bb8a1, launch blocker #3: "Replace mutable
    best/ with immutable uniquely named inference bundles plus an atomic
    best pointer." The prior version staged a temp directory, then did
    `shutil.rmtree(best_dir)` FOLLOWED BY `os.replace(tmp_dir, best_dir)`
    -- two separate operations, not one atomic one (`os.replace` cannot
    atomically swap in a directory over a non-empty existing one on
    POSIX). A crash in the window between the rmtree and the replace left
    NO best checkpoint at all, confirmed real by this re-audit.

    Reusing `checkpoint_module.save_checkpoint` directly gives `best/`
    every crash-safety property already built and hardened there for
    free: a uniquely-named, never-deleted-in-place bundle; an atomic
    pointer written and fsync'd BEFORE pruning; and (via
    `run_manifest=run_manifest`) the run's COMPLETE identity -- config,
    dataset, gene panel, cache content, synchronized init, LongNet
    checkpoint, Architecture 3 conditioner bundle/step, residual basis,
    and code state -- bound INSIDE the bundle and hash-verified on every
    read, not merely recorded in a bespoke, separately-verified
    `best_info.json`. `model_config` is the real resolved training
    config (mirrors every other `save_checkpoint` call site in this
    module), so `best/` is independently reconstructable without needing
    the original YAML on hand. `keep_last=2` keeps one prior best bundle
    around defensively; `optimizer=None` since `best/` is an inference
    artifact, never meant to resume TRAINING from."""
    checkpoint_module.save_checkpoint(
        model, config, gene_names, best_dir, step=step,
        extra_metadata={"total": float(val_loss)}, keep_last=2, run_manifest=run_manifest,
    )
    return Path(best_dir)


# Codex re-audit of commit f7bb8a1, launch blocker #3: every identity
# field that must agree between the CURRENT run/evaluation and a
# checkpoint's own recorded identity before its weights are trusted --
# "verify exact config, dataset, cache-content mapping, gene panel, sync
# init, LongNet checkpoint, conditioner bundle/step, residual basis and
# code state." Shared by `verify_full_checkpoint_identity` below for
# BOTH `best/` and a live checkpoint_dir's latest state -- one complete
# check, not two partial ones.
_FULL_CHECKPOINT_IDENTITY_FIELDS = (
    "model_architecture", "model_kind", "model_arm", "required_artifact_file_sha256",
    "config_identity_fingerprint", "dataset_manifest_fingerprint", "gene_panel_hash",
    "synchronized_init_manifest_sha256", "gigapath_checkpoint_sha256",
    "gene_residual_basis_gene_names_hash", "gene_residual_basis_sha256",
    # Codex re-audit of commit 2162ff4, finding #4: "gene_scale_sha256 is
    # listed but never placed in expected_values, so it is skipped."
    # Confirmed real -- and, unlike every other field here, there is no
    # honest way to fix it by COMPUTING an expected value: gene_scale is
    # fit from TRAINING samples only (`compute_training_gene_scale`),
    # which this function's callers (the evaluator, the basis fitter)
    # never load (evaluation covers validation/test samples; basis
    # fitting covers training samples but for a DIFFERENT architecture's
    # checkpoint). `gene_scale.npy` is also never bound inside a
    # transactional bundle (unlike run_manifest.json) -- it is a root-
    # only artifact -- so there is no accessible bundled artifact to hash
    # and compare either. Removed from this function's scope entirely
    # (Codex's own offered alternative to a real fix: "or remove the
    # false claim that it is checked") rather than leaving a field that
    # LOOKS checked but structurally cannot be. `gene_scale_sha256`
    # remains in `_RESUME_CONSISTENCY_FIELDS`, where it IS genuinely
    # checked -- resume compares two run_manifest.json RECORDS (both
    # already computed by `run_training` itself from its own freshly-
    # loaded training samples), never a bundled file, so no such gap
    # exists there.
    "architecture3_conditioner_checkpoint_sha256", "architecture3_conditioner_checkpoint_step",
    # Codex re-audit of commit 2162ff4, finding #4: "bind conditioner
    # bundle_id + manifest_sha256 as well as weights/step."
    "architecture3_conditioner_checkpoint_bundle_id", "architecture3_conditioner_checkpoint_manifest_sha256",
    "staged_conditioner_checkpoint_sha256", "staged_conditioner_checkpoint_step",
    "staged_conditioner_checkpoint_bundle_id", "staged_conditioner_checkpoint_manifest_sha256",
)


def verify_full_checkpoint_identity(
    checkpoint_dir: str | Path, *, config: dict, dataset_manifest: dict, gene_names: list[str],
    cache_content_by_sample: dict[str, dict] | None = None, allow_code_drift: bool = False,
) -> dict:
    """The ONE shared, complete identity verifier for both `best/` and a
    live checkpoint_dir's latest state -- Codex re-audit of commit
    f7bb8a1, launch blockers #3/#4/#6: replaces the previous
    `verify_checkpoint_bundle_identity` (best/-only, best_info.json-
    based, only checked dataset/genes/architecture) and
    `verify_checkpoint_run_manifest_against_evaluation` (latest-only,
    missing cache-content/gene-scale/basis-gene-names-hash/conditioner-
    step) with one function that checks EVERY field in
    `_FULL_CHECKPOINT_IDENTITY_FIELDS` against BOTH sources of the
    checkpoint's recorded identity.

    Resolves through `checkpoint_module.resolve_checkpoint_identity`
    first (exact-weights verification, already fail-closed and hardened
    there) then reads the run manifest bound INSIDE that exact resolved
    bundle via `checkpoint_module.load_checkpoint_run_manifest` -- never
    a loose, unbound root-level file. Every field is REQUIRED to be
    present in the checkpoint's recorded identity; a missing field fails
    exactly like a mismatched one.

    `cache_content_by_sample` (launch blocker #2/#6: "Bind evaluation to
    cache content used by training. Compare the checkpoint's per-sample
    cache identities against the currently loaded evaluation samples
    before loading weights") is a PER-SAMPLE comparison, deliberately
    NOT the single combined `cache_content_fingerprint` resume-
    consistency uses: training's own preflight covers `train_ids +
    validation_ids`, while an evaluation call may cover a DIFFERENT
    (e.g. validation-only, or test-only) sample set, so the two combined
    fingerprints are never expected to match even when every individual
    sample's cache content genuinely agrees. Instead, every sample_id in
    the caller's `cache_content_by_sample` that the checkpoint's own
    recorded `cache_preflight_report.cache_content_by_sample` ALSO
    covers must match exactly -- a sample the checkpoint's training run
    never touched (e.g. a held-out test sample) has nothing to compare
    against and is skipped, never silently treated as a pass OR a
    failure for a sample outside the checkpoint's own training scope.

    Codex re-audit of commit 2162ff4, finding #4, three further fixes:
    (a) `allow_code_drift` (default False, matching `verify_resume_
    consistency`'s own fail-closed default) additionally compares the
    CURRENT process's code commit + operational worktree hash against
    the checkpoint's own recorded values, with the SAME "unknown-is-
    drift" override semantics -- code that changed between training and
    evaluation/basis-fitting could change results in ways nothing else
    here can see. (b) for every sample_id in `cache_content_by_sample`
    that ALSO appears in the CURRENT `dataset_manifest`'s own train/
    validation sample ids (i.e. provably within the checkpoint's own
    training-time preflight scope, since `dataset_manifest_fingerprint`
    is already verified equal above), a MISSING recorded cache identity
    now FAILS instead of being silently skipped -- skipping remains
    correct only for samples outside that scope (e.g. a held-out test
    sample an evaluation call newly introduces). (c) the resolved
    bundle's own `model_config.json` is independently re-fingerprinted
    and compared against the SAME bundle's `run_manifest.json`'s own
    recorded `config_identity_fingerprint` -- every other field here
    compares a CURRENT value against a RECORDED one, but nothing
    previously checked that the recorded fingerprint and the actual
    bundled config file agree with EACH OTHER, which would catch a
    hypothetical bug where `build_run_manifest` fingerprinted a
    different config object than the one actually passed to
    `save_checkpoint` as `model_config`."""
    checkpoint_dir = Path(checkpoint_dir)
    identity = checkpoint_module.resolve_checkpoint_identity(checkpoint_dir)
    if identity.weights_sha256 is None and identity.resolved_dir == checkpoint_dir and identity.step is None:
        raise ValueError(f"{checkpoint_dir}: no checkpoint bundle found -- nothing to verify or load")
    checkpoint_run_manifest = checkpoint_module.load_checkpoint_run_manifest(checkpoint_dir)
    if checkpoint_run_manifest is None:
        raise ValueError(
            f"{checkpoint_dir} (resolved bundle {identity.resolved_dir}) has no run_manifest.json bound "
            "inside it -- cannot verify this checkpoint's config/dataset/gene-panel/architecture/cache/"
            "synchronized-init/conditioner/basis identity before loading. Refusing to load an "
            "unverifiable checkpoint"
        )

    # Finding #4(c): the bundle's OWN model_config.json, independently
    # re-fingerprinted, must agree with the SAME bundle's run_manifest.json
    # recorded config_identity_fingerprint.
    bundled_model_config_path = identity.resolved_dir / "model_config.json"
    if not bundled_model_config_path.is_file():
        raise ValueError(
            f"{checkpoint_dir} (resolved bundle {identity.resolved_dir}): model_config.json is missing -- "
            "cannot verify the bundled config matches its own recorded config_identity_fingerprint"
        )
    bundled_model_config = json.loads(bundled_model_config_path.read_text())
    bundled_config_identity_fingerprint = config_identity_fingerprint(bundled_model_config)
    recorded_config_identity_fingerprint = checkpoint_run_manifest.get("config_identity_fingerprint")
    if bundled_config_identity_fingerprint != recorded_config_identity_fingerprint:
        raise ValueError(
            f"{checkpoint_dir} (resolved bundle {identity.resolved_dir}): the bundled model_config.json's "
            f"own fingerprint ({bundled_config_identity_fingerprint!r}) does not match this same bundle's "
            f"run_manifest.json recorded config_identity_fingerprint ({recorded_config_identity_fingerprint!r}) "
            "-- the bundle's own two identity records disagree with each other, refusing to trust either"
        )

    architecture_id = str((config.get("model") or {}).get("architecture", ""))
    expected_values = {
        "model_architecture": model_identifier_for_config(config),
        "model_kind": model_kind_for_config(config),
        "required_artifact_file_sha256": required_artifact_file_sha256(config),
        "config_identity_fingerprint": config_identity_fingerprint(config),
        "dataset_manifest_fingerprint": dataset_manifest_fingerprint(dataset_manifest),
        "gene_panel_hash": gene_panel_hash(gene_names),
    }
    if (config.get("model") or {}).get("arm") is not None:
        expected_values["model_arm"] = str((config.get("model") or {})["arm"])

    training_cfg = config.get("training") or {}
    synchronized_init_dir = training_cfg.get("synchronized_init_dir")
    if synchronized_init_dir:
        sync_manifest_path = Path(synchronized_init_dir) / "initialization_manifest.json"
        expected_values["synchronized_init_manifest_sha256"] = (
            file_sha256(sync_manifest_path) if sync_manifest_path.is_file() else None
        )

    model_params = (config.get("model") or {}).get("params") or {}
    required_fingerprints = config.get("required_fingerprints") or {}
    if model_params.get("use_global_slide"):
        gigapath_checkpoint_path = required_fingerprints.get("gigapath_checkpoint")
        expected_values["gigapath_checkpoint_sha256"] = (
            file_sha256(gigapath_checkpoint_path)
            if gigapath_checkpoint_path and Path(gigapath_checkpoint_path).is_file() else None
        )

    if architecture_id == "4":
        gene_residual_basis_path = required_fingerprints.get("gene_residual_basis")
        if gene_residual_basis_path:
            from gen3_multiscale.models.gene_basis import load_gene_residual_basis

            basis = load_gene_residual_basis(gene_residual_basis_path)
            expected_values["gene_residual_basis_gene_names_hash"] = basis.gene_names_hash
            expected_values["gene_residual_basis_sha256"] = hashlib.sha256(
                np.ascontiguousarray(basis.basis.detach().cpu().numpy()).tobytes()
            ).hexdigest()
        conditioner_checkpoint_dir = required_fingerprints.get("architecture3_conditioner_checkpoint")
        if conditioner_checkpoint_dir:
            conditioner_identity = checkpoint_module.resolve_checkpoint_identity(conditioner_checkpoint_dir)
            expected_values["architecture3_conditioner_checkpoint_sha256"] = conditioner_identity.weights_sha256
            expected_values["architecture3_conditioner_checkpoint_step"] = conditioner_identity.step
            # Finding #4(a): bind bundle_id + manifest_sha256 too.
            expected_values["architecture3_conditioner_checkpoint_bundle_id"] = conditioner_identity.bundle_dir
            expected_values["architecture3_conditioner_checkpoint_manifest_sha256"] = (
                conditioner_identity.manifest_sha256
            )

    staged_checkpoint_dir = (
        required_fingerprints.get("architecture3_conditioner_checkpoint")
        if architecture_id == "4"
        else required_fingerprints.get("gen4_conditioner_checkpoint")
    )
    if staged_checkpoint_dir:
        staged_identity = checkpoint_module.resolve_checkpoint_identity(staged_checkpoint_dir)
        expected_values.update({
            "staged_conditioner_checkpoint_sha256": staged_identity.weights_sha256,
            "staged_conditioner_checkpoint_step": staged_identity.step,
            "staged_conditioner_checkpoint_bundle_id": staged_identity.bundle_dir,
            "staged_conditioner_checkpoint_manifest_sha256": staged_identity.manifest_sha256,
        })

    for field in _FULL_CHECKPOINT_IDENTITY_FIELDS:
        if field not in expected_values:
            continue  # not applicable to this architecture/config (e.g. gigapath fields when use_global_slide=false)
        expected = expected_values[field]
        recorded = checkpoint_run_manifest.get(field)
        if recorded is None or recorded != expected:
            raise ValueError(
                f"{checkpoint_dir} (resolved bundle {identity.resolved_dir}): recorded {field}={recorded!r} "
                f"does not match this run's own {field}={expected!r} -- refusing to load a checkpoint whose "
                "recorded identity does not match (or never recorded) what is being trained/evaluated against"
            )

    # Finding #4(a): code-state binding, mirroring `verify_resume_
    # consistency`'s own fail-closed-on-unknown definition of drift and
    # its explicit `allow_code_drift` override -- checked separately from
    # the equality loop above because it needs that same override, not
    # because it is any less real.
    recorded_commit = checkpoint_run_manifest.get("code_commit_hash")
    current_commit = _code_commit_hash()
    recorded_diff_hash = checkpoint_run_manifest.get("code_worktree_diff_hash")
    current_diff_hash = _worktree_diff_hash()
    code_identity_unknown = recorded_commit is None or current_commit is None
    code_drifted = code_identity_unknown or recorded_commit != current_commit or recorded_diff_hash != current_diff_hash
    if code_drifted and not allow_code_drift:
        reason = "code identity is unknown on at least one side" if code_identity_unknown else "code state changed"
        raise ValueError(
            f"{checkpoint_dir} (resolved bundle {identity.resolved_dir}): {reason} since this checkpoint was "
            f"saved (commit {recorded_commit!r} -> {current_commit!r}, worktree_diff_hash "
            f"{recorded_diff_hash!r} -> {current_diff_hash!r}) -- pass allow_code_drift=True if evaluating/"
            "fitting under different or unverifiable code is genuinely intended"
        )

    if cache_content_by_sample:
        recorded_cache_by_sample = (
            (checkpoint_run_manifest.get("cache_preflight_report") or {}).get("cache_content_by_sample") or {}
        )
        # Finding #4(b): `dataset_manifest_fingerprint` is already
        # verified equal to the checkpoint's own recorded value above --
        # so any sample_id in THIS dataset_manifest's own train/
        # validation ids is PROVABLY within the checkpoint's own
        # training-time preflight scope (which always covers exactly
        # train_ids + validation_ids). A missing recorded cache identity
        # for such a sample can only mean the checkpoint's own recorded
        # cache_preflight_report is itself incomplete/corrupted, never a
        # legitimate "this sample was outside training scope" case --
        # that legitimate case (e.g. a held-out test sample an evaluation
        # call newly introduces) is the ONLY one still silently skipped.
        known_training_scope_sample_ids = set(dataset_manifest.get("train_sample_ids") or []) | set(
            dataset_manifest.get("validation_sample_ids") or []
        )
        for sample_id, current_identity in cache_content_by_sample.items():
            recorded_identity = recorded_cache_by_sample.get(sample_id)
            if recorded_identity is None:
                if sample_id in known_training_scope_sample_ids:
                    raise ValueError(
                        f"{checkpoint_dir} (resolved bundle {identity.resolved_dir}): sample {sample_id!r} is "
                        "one of this dataset manifest's own train/validation sample ids -- provably within "
                        "this checkpoint's own training-time preflight scope -- but has no recorded cache "
                        "content identity in cache_preflight_report.cache_content_by_sample. Refusing to "
                        "treat a checkpoint with an incomplete recorded cache identity as safe to load"
                    )
                continue  # genuinely outside the checkpoint's own training scope -- nothing to compare
            if recorded_identity != current_identity:
                raise ValueError(
                    f"{checkpoint_dir} (resolved bundle {identity.resolved_dir}): sample {sample_id!r}'s cache "
                    f"content identity recorded at training time ({recorded_identity!r}) does not match its "
                    f"currently-loaded cache content ({current_identity!r}) -- the cache was regenerated with "
                    "different content since this checkpoint was trained; refusing to load"
                )
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
    # Integration audit item 1: a Gen4/Gen5 config never sets
    # model.architecture (its own generic model.kind covers the same
    # role) -- `.get(..., "")` avoids a KeyError; `architecture_id` stays
    # "" for those configs, matching `verify_resume_consistency`'s own
    # already-`.get`-based read of the identical field (both sides must
    # compute the same value for resume verification to be meaningful).
    architecture_id = str((config.get("model") or {}).get("architecture", ""))
    model_identifier = model_identifier_for_config(config)
    is_gen4_or_gen5 = (config.get("model") or {}).get("arm") is not None

    manifest_path = data_cfg.get("gen3_manifest_path")
    if not manifest_path:
        raise ValueError("data.gen3_manifest_path must be set to a real, already-built dataset manifest")
    if not Path(manifest_path).is_file():
        raise FileNotFoundError(f"dataset manifest not found at {manifest_path} -- build it first")
    dataset_manifest = load_dataset_manifest(manifest_path)

    # Requirement #1: sample selection and the train/validation/test
    # split are read EXCLUSIVELY from the manifest -- never re-derived.
    declared_train_ids = list(dataset_manifest["train_sample_ids"])
    declared_validation_ids = list(dataset_manifest["validation_sample_ids"])

    def _resolved_split_ids(field: str, declared: list[str], *, allow_empty: bool) -> list[str]:
        if field not in data_cfg:
            return declared
        selected = [str(sample_id) for sample_id in data_cfg[field]]
        if len(selected) != len(set(selected)):
            raise ValueError(f"data.{field} contains duplicate sample ids")
        unknown = sorted(set(selected) - set(declared))
        if unknown:
            raise ValueError(
                f"data.{field} contains samples outside the manifest-declared split: {unknown}"
            )
        if not selected and not allow_empty:
            raise ValueError(f"data.{field} must select at least one manifest training sample")
        return selected

    # Used by the fixed-mask capacity gate: preserve the immutable full
    # manifest identity required by staged conditioners/autoencoders, but
    # deliberately restrict which declared samples this short diagnostic
    # iterates. The override is part of the config fingerprint/run
    # manifest and cannot silently affect a normal resolved config.
    train_ids = _resolved_split_ids(
        "train_sample_ids_override", declared_train_ids, allow_empty=False,
    )
    validation_ids = _resolved_split_ids(
        "validation_sample_ids_override", declared_validation_ids, allow_empty=True,
    )
    if not train_ids:
        raise ValueError("dataset manifest has zero train_sample_ids -- nothing to train on")

    checkpoint_dir = Path(training_cfg["checkpoint_dir"])

    # Requirements #3/#4: cache coverage + tile-encoder provenance
    # consistency, BEFORE any model/optimizer/DataLoader is constructed.
    # Preflighted over train+validation samples -- the only manifest
    # roles this trainer touches (test-split evaluation is Step 7's job).
    cfg_om = OmegaConf.create(config)
    if is_gen4_or_gen5:
        from gen3_multiscale.gen4.dataset_adapter import load_and_preflight_gen4_samples

        samples, preflight_report = load_and_preflight_gen4_samples(
            cfg_om,
            dataset_manifest,
            train_ids + validation_ids,
            config,
            require_resolved_artifacts=not (smoke and not staged_smoke),
        )
    else:
        expected_provenance = expected_tile_encoder_provenance(config)
        samples, preflight_report = load_and_preflight_samples(
            cfg_om, dataset_manifest, train_ids + validation_ids, expected_provenance,
        )
    # Codex re-audit of commit f7bb8a1, launch blocker #6: "Perform all
    # resume verification before writing preflight reports, mask banks,
    # gene scale, validation history or other checkpoint-directory
    # artifacts." Confirmed real: `save_gen3_preflight_report` used to
    # write here, unconditionally, BEFORE `verify_resume_consistency` --
    # a resume that gets refused must never have already overwritten the
    # PRIOR run's own preflight_report.json on its way to being refused
    # (exactly the same concern `gene_scale.npy` was already fixed for).
    # The disk WRITE is deferred to right after resume verification
    # passes, below; only the in-memory `preflight_report` (needed to
    # build the run manifest being verified) is computed here.

    strata = config["masking"]["strata"]
    train_samples = {sid: s for sid, s in samples.items() if sid in train_ids}
    val_samples = {sid: s for sid, s in samples.items() if sid in validation_ids}

    # A smoke run still has to construct a structurally valid stratified
    # schedule.  Using two training masks and one validation mask was only
    # valid for the small two-stratum test fixture; the real configs have
    # four strata, and the fail-closed scheduler correctly rejects fewer
    # items than strata.  Keep the smoke pool minimal while covering every
    # declared stratum once.  The loop below still executes exactly one
    # optimizer step and one validation step.
    n_smoke_masks = max(1, len(strata))
    n_training_masks = n_smoke_masks if smoke else int(data_cfg.get("n_training_masks_per_sample", 500))
    n_validation_masks = n_smoke_masks if smoke else int(data_cfg.get("n_validation_masks", 4))

    train_schedule = build_gen3_mask_schedule(
        dataset_manifest, train_samples, strata, role="train", n_training_masks_per_sample=n_training_masks,
    )
    val_schedule = None
    val_loader = None
    if val_samples:
        # Same launch-blocker-#6 deferral: `mask_bank_dir` deliberately
        # OMITTED here (never write validation mask banks to
        # checkpoint_dir before resume verification passes) --
        # `build_gen3_mask_schedule` is a pure, deterministic function of
        # (dataset_manifest, val_samples, strata, split_counts,
        # split_seeds) when given no `mask_bank_dir`, so the SAME
        # schedule/reports this run_manifest gets built from are
        # recomputed (cheaply) and PERSISTED to disk in a second call
        # further below, once verification has actually passed.
        val_schedule = build_gen3_mask_schedule(
            dataset_manifest, val_samples, strata, role="validation",
            split_counts={"validation": n_validation_masks}, split_seeds={"validation": 700_000},
        )

    novae_enabled = bool((data_cfg.get("novae") or {}).get("enabled", False))
    gene_names = list(dataset_manifest["gene_panel"])

    # Integration audit item 2/1: a Gen4/Gen5 config (`model.arm` present,
    # never set by a Gen3 config) builds its real (context, query)
    # examples through `gen4.dataset_adapter.Gen4SpatialFieldDataset`
    # instead -- the SAME real sample loading/mask schedule above
    # (`load_and_preflight_samples`/`build_gen3_mask_schedule`), only the
    # per-arm cache wiring (`__getitem__`) differs. See that class's own
    # docstring for exactly which caches each arm consumes.
    if is_gen4_or_gen5:
        from gen3_multiscale.gen4.dataset_adapter import Gen4SpatialFieldDataset

        dataset_cls = Gen4SpatialFieldDataset
        dataset_extra_args = (cfg_om, gene_names)
    else:
        dataset_cls = Gen3SpatialFieldDataset
        dataset_extra_args = ()

    # Requirement #5: `train_dataset` is indexed DIRECTLY by a
    # deterministic per-step index (deterministic_train_index_for_step,
    # below) rather than iterated through a shuffled DataLoader --  a
    # plain `shuffle=True` DataLoader reshuffles a fresh, UNSAVED
    # permutation every epoch, so a resumed run previously continued from
    # a different point in a different random ordering than the original
    # run would have reached by the same step.
    train_dataset = dataset_cls(
        dataset_manifest, train_samples, train_schedule, strata, *dataset_extra_args, novae_enabled=novae_enabled,
    )
    if val_schedule is not None:
        val_dataset = dataset_cls(
            dataset_manifest, val_samples, val_schedule, strata, *dataset_extra_args, novae_enabled=novae_enabled,
        )
        boundary_preflight = val_dataset.validate_boundary_schedule()
        print(
            f"validation boundary preflight: PASS "
            f"({boundary_preflight['n_items_checked']} fixed masks checked)",
            flush=True,
        )
        # requirement #9: deterministic FIXED-mask validation -- never shuffled.
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False, collate_fn=gen3_identity_collate,
        )

    seed = int(training_cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    sample_rng = random.Random(seed)

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
        cache_content_by_sample=preflight_report.get("cache_content_by_sample"),
        allow_code_drift=allow_code_drift,
    )
    kind = model_info["kind"]
    # Integration audit item 1: Gen4/Gen5's own normalized model_info
    # schema (gen4.trainer_adapter.build_gen4_or_gen5_model_for_inference)
    # has no gene_basis/gigapath_checkpoint_sha256/
    # synchronized_init_manifest_path/architecture3_conditioner keys at
    # all -- those are Gen3-Architecture-4-specific concepts with no
    # Gen4/5 equivalent. `.get(..., None)` makes every downstream use of
    # these (already None-safe -- e.g. "gene_basis.gene_names_hash if
    # gene_basis is not None else None") correctly produce an honestly
    # empty run-manifest field for a Gen4/5 run instead of a KeyError.
    gene_basis = model_info.get("gene_basis")
    gigapath_checkpoint_sha256 = model_info.get("gigapath_checkpoint_sha256")
    synchronized_init_manifest_path = model_info.get("synchronized_init_manifest_path")
    architecture3_conditioner_info = model_info.get("architecture3_conditioner")
    staged_conditioner_info = model_info.get("conditioner_info") or architecture3_conditioner_info

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
        staged_conditioner_info=staged_conditioner_info,
    )
    existing_run_manifest_path = checkpoint_dir / "run_manifest.json"
    # Codex re-audit of commit 2162ff4, finding #2: "Resume still trusts
    # the loose root run_manifest.json... does not require the bundle-
    # bound run manifest to equal that root manifest." Confirmed real:
    # the prior version read ONLY `existing_run_manifest_path` (the root
    # mirror -- documented elsewhere in this codebase as "only ever a
    # disposable convenience mirror") and compared THAT against the new
    # run, never cross-checking it against the actual bundle that would
    # be resumed from. A stale root mirror (a crash between a bundle
    # write and its root-mirror refresh, or a hand-edited root file)
    # could therefore describe DIFFERENT state than the weights actually
    # resumed. Fixed: `checkpoint_module.load_checkpoint_run_manifest`
    # resolves through the SAME verified-bundle path every other loader
    # uses and returns the CANONICAL, bundle-bound run_manifest.json --
    # this is now `old_run_manifest`, never the root file directly. When
    # a root mirror also exists, it is compared against the canonical
    # copy and rejected on any disagreement (a legitimate mirror is
    # always byte-for-byte identical to what it mirrors).
    canonical_old_run_manifest = checkpoint_module.load_checkpoint_run_manifest(checkpoint_dir)
    # "Enforce run-manifest/checkpoint-pointer consistency in both
    # directions" (launch blocker #6) -- real history bundles with no
    # CANONICAL bound run_manifest at all (deleted/lost, predates
    # run_manifest.json binding, or a bundle whose own copy went missing)
    # can only mean this trainer cannot honestly verify what that prior
    # run's identity was; refusing here is safer than silently treating
    # it as "nothing to resume from." (The converse -- a run_manifest.json
    # with no checkpoint pointer yet -- is a normal, safe state for a run
    # that crashed before its FIRST checkpoint ever saved;
    # `checkpoint_module.load_training_state`'s own hardening already
    # guarantees a pointer exists whenever `step > 0` is ever reported,
    # so there is no silent-mismatch case left to reject there.)
    if checkpoint_module.list_checkpoint_bundles(checkpoint_dir) and canonical_old_run_manifest is None:
        raise ValueError(
            f"{checkpoint_dir} has real checkpoint history bundles but no canonical, bundle-bound "
            "run_manifest.json -- cannot verify this checkpoint's identity before resuming. Either it "
            "was deleted after a real run, this checkpoint predates run_manifest.json binding, or the "
            "resolved bundle's own copy is missing; refusing to silently treat unverifiable history as "
            "safe to build on"
        )
    code_drift_acknowledged = False
    if canonical_old_run_manifest is not None:
        if existing_run_manifest_path.is_file():
            # Compared over the SAME identity fields `verify_resume_
            # consistency` itself checks -- never full-dict equality. The
            # root mirror is legitimately refreshed on EVERY run_training
            # call (including a no-op resume that saves no new
            # checkpoint), while the bundle-bound copy only changes on a
            # real save; non-identity bookkeeping fields (environment_
            # versions, total_steps, code_commit_hash for a commit made
            # between calls, ...) can therefore genuinely differ between
            # them without describing any real scientific drift. Code-
            # state fields are deliberately excluded here too -- they
            # have their own separate `allow_code_drift` override
            # mechanism inside `verify_resume_consistency` below, and
            # requiring root/canonical agreement on them here would
            # bypass that override.
            root_run_manifest = json.loads(existing_run_manifest_path.read_text())
            mismatched_fields = [
                field for field in _RESUME_CONSISTENCY_FIELDS
                if root_run_manifest.get(field) != canonical_old_run_manifest.get(field)
            ]
            if mismatched_fields:
                raise ValueError(
                    f"{checkpoint_dir}: the root run_manifest.json disagrees with the canonical, "
                    f"bundle-bound run_manifest.json on identity field(s) {mismatched_fields} -- the "
                    "root mirror is stale, corrupted, or was hand-edited relative to the actual "
                    "checkpoint bundle that would be resumed from; refusing to resume from an "
                    "inconsistent checkpoint_dir"
                )
        old_run_manifest = canonical_old_run_manifest
        verify_resume_consistency(old_run_manifest, run_manifest, allow_code_drift=allow_code_drift)
        # Launch blocker #10 (refined by the re-audit of commit f7bb8a1's
        # launch blocker #8, matching `verify_resume_consistency`'s own
        # fail-closed-on-unknown definition of drift): record whether the
        # override was actually NEEDED (not merely passed) -- a caller
        # passing allow_code_drift=True against a checkpoint whose code
        # state did NOT drift (and whose identity was fully known on both
        # sides) leaves this False, so the manifest only ever claims an
        # override happened when one genuinely did.
        old_commit = old_run_manifest.get("code_commit_hash")
        new_commit = run_manifest.get("code_commit_hash")
        code_identity_unknown = old_commit is None or new_commit is None
        code_drift_acknowledged = allow_code_drift and (
            code_identity_unknown or old_commit != new_commit
            or old_run_manifest.get("code_worktree_diff_hash") != run_manifest.get("code_worktree_diff_hash")
        )
    run_manifest["code_drift_acknowledged"] = code_drift_acknowledged

    # Only now, having passed resume-consistency verification (or there
    # being no prior run to verify against), is it safe to overwrite this
    # checkpoint_dir's gene_scale.npy -- audit #4.
    gene_scale_path = save_gene_scale(gene_scale, checkpoint_dir / "gene_scale.npy")
    # Launch blocker #6: preflight report and validation mask banks are
    # ALSO deferred here, for the identical reason -- see the comment at
    # this function's earlier (in-memory-only) preflight/mask-schedule
    # construction. Persisting the validation mask bank a SECOND time
    # (identical content, since `build_gen3_mask_schedule` is a pure
    # function of its non-mask_bank_dir arguments) is the cheap, correct
    # way to keep the disk write itself deferred without recomputing
    # `val_schedule`/`run_manifest`'s own reports differently.
    save_gen3_preflight_report(preflight_report, checkpoint_dir / "preflight_report.json")
    if val_samples:
        build_gen3_mask_schedule(
            dataset_manifest, val_samples, strata, role="validation",
            split_counts={"validation": n_validation_masks}, split_seeds={"validation": 700_000},
            mask_bank_dir=str(checkpoint_dir),
        )

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
    def _run_validation(current_step: int) -> dict | None:
        if val_loader is None:
            return None
        model.eval()
        with torch.no_grad():
            val_totals: list[float] = []
            val_conditioner_only_totals: list[float] = []
            val_flow_losses: list[float] = []
            for item_index, (val_inputs, val_targets) in enumerate(val_loader):
                # Codex re-audit of commit 90f853e, launch blocker #6,
                # refined by the re-audit of commit f7bb8a1's secondary
                # fix #1: "Use common random numbers for Architecture 4
                # validation: seed by fixed evaluation seed plus stable
                # mask identity [sample_id + stratum + query_fingerprint],
                # independent of training step and evaluation order [and
                # dataset/mask-bank record index]." The PRIOR generator
                # was reseeded from (seed, current_step) -- every
                # checkpoint therefore sampled Architecture 4's flow
                # apparatus with DIFFERENT noise on the SAME held-out
                # item, so best-checkpoint selection could partly reflect
                # Monte Carlo luck in the noise draw rather than a
                # genuine difference between checkpoints. A LATER fix
                # reseeded per `item_index` instead, confirmed still
                # real by this re-audit: `item_index` is only stable as
                # long as the on-disk mask bank's record ORDER never
                # changes -- a self-heal/regeneration that reorders the
                # identical logical set of realized masks would silently
                # reassign noise draws across items. Reseeding from the
                # item's own CONTENT identity (`sample_id`/`stratum`/
                # `query_fingerprint`, stable regardless of storage
                # order) closes that gap while keeping every other
                # property (identical noise on the identical item across
                # checkpoints/resumes, common-random-numbers comparison).
                item_identity = val_dataset.item_identity(item_index)
                stable_key = (
                    f"{item_identity['sample_id']}:{item_identity['stratum']}:{item_identity['query_fingerprint']}"
                )
                predictive_val_generator = torch.Generator(device=device).manual_seed(
                    common_random_validation_seed(seed, stable_key)
                )
                # The secondary flow-loss diagnostic must use common
                # random numbers too. A single generator created outside
                # this loop advanced across validation calls, making that
                # value change between checkpoints solely because it was
                # evaluated later. Use a separate stable per-item stream.
                flow_loss_generator = torch.Generator(device=device).manual_seed(
                    common_random_validation_seed(seed, f"flow_loss:{stable_key}")
                )
                val_target_expression = torch.as_tensor(val_targets.query_expression, dtype=torch.float32, device=device)
                val_query_coords = torch.as_tensor(val_inputs.query_coords, dtype=torch.float32, device=device)
                # Requirement #8/audit #1: model selection ALWAYS uses the
                # real predictive distribution (Architecture 4) or real
                # forward() (Architectures 1-3) -- never Architecture 4's
                # frozen-conditioner-only forward(), which used to be
                # silently used for its selection metric.
                det_losses = compute_deterministic_reconstruction_losses(
                    kind, model, val_inputs, val_target_expression, val_query_coords,
                    gradient_weight=gradient_weight, k_neighbors=k_neighbors, per_gene_scale=gene_scale_tensor,
                    generator=predictive_val_generator,
                )
                val_totals.append(float(det_losses["total"]))
                if "conditioner_only_total" in det_losses:
                    val_conditioner_only_totals.append(float(det_losses["conditioner_only_total"]))
                if kind == "flow":
                    # Gen3 Architecture 4 / Gen4 flow arms expose a
                    # dedicated compute_flow_matching_loss -- Gen5's
                    # latent_flow does not (see the `latent_flow` branch
                    # below), only the combined compute_losses() contract.
                    flow_loss = model.compute_flow_matching_loss(
                        val_inputs, val_target_expression, generator=flow_loss_generator,
                    )
                    val_flow_losses.append(float(flow_loss))
                elif kind == "latent_flow":
                    flow_loss = model.compute_losses(
                        val_inputs, val_target_expression, generator=flow_loss_generator,
                    )["flow_loss"]
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
            kind, model, inputs, target_expression, query_coords,
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
                            model, config, gene_names, checkpoint_dir / "best",
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
                keep_last=checkpoint_keep_last, optimizer=optimizer, rng=sample_rng, run_manifest=run_manifest,
            )
        step = completed_steps

    if not smoke and step > resume_step:
        checkpoint_module.save_checkpoint(
            model, config, gene_names, checkpoint_dir, step,
            extra_metadata={"n_skipped_nonfinite": n_skipped_nonfinite, "completion_reason": completion_reason},
            keep_last=checkpoint_keep_last, optimizer=optimizer, rng=sample_rng, run_manifest=run_manifest,
        )

    elapsed = time.time() - start_time
    summary = {
        "ok": True, "smoke": smoke, "architecture": model_identifier, "kind": kind,
        "final_step": step,
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
