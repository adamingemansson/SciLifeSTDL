"""The minimal real Step 7 evaluator -- Adam's explicit Step 6 audit #12
deliverable (of commit 27e1232): "Build the minimal Step 7 evaluator
before long training. On fixed patient-held-out masks, report per-
sample/per-mask PCC and RMSE with valid-gene counts, patient-level
aggregation and confidence intervals. Include harmonic, mean and simple
nearest-neighbour baselines. Keep ST-FID/ST-MMD secondary. Never select
using test samples."

Deliberately MINIMAL, per Adam's own earlier explicit instruction ("do
not implement a large new evaluation system yet") -- this is not Step
7's full evaluator, only the smallest real one that can honestly report
whether a checkpoint is learning anything on held-out data, using
already-audited primitives wherever one exists:
  - `evaluation/metrics.py::pearson_per_gene`/`rmse` for the per-item
    numbers, `aggregate_patient_metrics` (already built in Phase 7 for
    exactly this) for patient-level aggregation + 95% CIs.
  - `models/harmonic.py::harmonic_interpolation` for the harmonic
    baseline -- that module's own docstring already states "Harmonic,
    inverse-distance, and nearest-neighbour must also be computed as
    exact-mask external baselines for every arm."
  - `training/gen3_dataset.py`'s FIXED, deterministic held-out mask
    schedule (the exact same one `train.py`'s own validation loop uses)
    -- this evaluator never draws a fresh random mask.

Real, honest limits documented rather than hidden: mean/nearest-neighbour
baselines are new, small, deliberately simple functions (no existing
equivalent found by direct search); ST-FID/ST-MMD are computed only when
explicitly requested (`compute_st_fid_mmd=True`) and are never part of
the reported headline metric, matching "keep ST-FID/ST-MMD secondary."
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from gen3_multiscale.data.dataset_manifest import gene_panel_hash, load_dataset_manifest
from gen3_multiscale.evaluation.metrics import (
    aggregate_patient_metrics, embed_pca, gene_panel_metrics, nonzero_auc, pearson_per_gene, resolve_gene_panels,
    rmse, st_fid, st_mmd,
)
from gen3_multiscale.evaluation.train_gene_panels import load_train_derived_gene_panels
from gen3_multiscale.models.harmonic import harmonic_interpolation
from gen3_multiscale.training import checkpoint as checkpoint_module
from gen3_multiscale.training.gen3_dataset import Gen3SpatialFieldDataset, build_gen3_mask_schedule
from gen3_multiscale.training.gen3_preflight import load_and_preflight_samples
from gen3_multiscale.training.train import (
    _code_commit_hash, _worktree_diff_hash, build_model_for_inference, common_random_validation_seed,
    config_identity_fingerprint, dataset_manifest_fingerprint, expected_tile_encoder_provenance,
    predict_for_metrics, resolved_config, verify_full_checkpoint_identity,
)


def per_item_reconstruction_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """PCC and RMSE for ONE mask/hole, with the "valid-gene count" the
    audit explicitly asked for -- a gene is "valid" for PCC purposes when
    it has non-zero variance in the TRUE values for this item
    (`pearson_per_gene`'s own eligibility rule: a truth-constant gene is
    unidentifiable within this hole and is excluded, not silently scored
    as a perfect or zero match)."""
    if pred.shape != true.shape:
        raise ValueError(f"pred {pred.shape} and true {true.shape} must have the same shape")
    per_gene_pcc = pearson_per_gene(pred, true)
    valid = np.isfinite(per_gene_pcc)
    return {
        "pcc": float(np.nanmean(per_gene_pcc)) if valid.any() else float("nan"),
        "rmse": rmse(pred, true),
        "n_valid_genes": int(valid.sum()),
        "n_genes": int(per_gene_pcc.shape[0]),
        # Launch blocker #9: "nonzero AUC" -- can distinguish
        # zero/measured-absent vs. nonzero true expression from predicted
        # magnitude alone, flattened across every gene in this item.
        "nonzero_auc": nonzero_auc(pred, true),
    }


def mean_baseline_prediction(inputs) -> np.ndarray:
    """Predict every query spot as the plain mean of this item's own
    observed expression -- the simplest possible non-trivial baseline."""
    observed = np.asarray(inputs.observed_full_gene_expression, dtype=np.float64)
    mean_expr = observed.mean(axis=0)
    return np.tile(mean_expr, (inputs.query_coords.shape[0], 1)).astype(np.float32)


def nearest_neighbor_baseline_prediction(inputs) -> np.ndarray:
    """Predict every query spot as its single spatially-nearest OBSERVED
    spot's real expression -- the simple nearest-neighbour baseline
    `harmonic.py`'s own docstring names alongside harmonic/inverse-
    distance."""
    observed_coords = np.asarray(inputs.observed_coords, dtype=np.float64)
    query_coords = np.asarray(inputs.query_coords, dtype=np.float64)
    observed_expr = np.asarray(inputs.observed_full_gene_expression, dtype=np.float64)
    dists = np.linalg.norm(query_coords[:, None, :] - observed_coords[None, :, :], axis=-1)
    nearest_idx = np.argmin(dists, axis=1)
    return observed_expr[nearest_idx].astype(np.float32)


def harmonic_baseline_prediction(inputs) -> np.ndarray:
    """The exact-mask harmonic-interpolation baseline
    (`models/harmonic.py`) -- Architecture 2's own anchor input, reused
    here unmodified as an EVALUATION baseline for every architecture,
    never as a model input."""
    return harmonic_interpolation(
        np.asarray(inputs.observed_coords, dtype=np.float64),
        np.asarray(inputs.observed_full_gene_expression, dtype=np.float64),
        np.asarray(inputs.query_coords, dtype=np.float64),
    )


_BASELINE_PREDICTORS = {
    "mean": mean_baseline_prediction,
    "nearest_neighbor": nearest_neighbor_baseline_prediction,
    "harmonic": harmonic_baseline_prediction,
}


def load_configured_gene_panels(
    config: dict, dataset_manifest: dict | None = None,
) -> dict[str, list[str]]:
    """Codex re-audit of commit 90f853e, launch blocker #9: "configured
    named panels" -- `evaluation.gene_panels` (every architectureN.yaml
    already declares one: `{panel_name: json_path}`) was never actually
    read by this evaluator; `metrics.py::gene_panel_metrics`/
    `resolve_gene_panels` exist and are tested, but nothing called them
    with real gene NAMES, only the unresolved path strings. Mirrors
    `src/training/train.py::_resolved_evaluation_gene_panels`'s fixed-
    panel loading (a JSON file whose top-level `"genes"` key, or the
    payload itself if it is already a bare list, names the panel) --
    never the training-sample-variance-ranked panel half of that
    function, which has no equivalent config field here."""
    raw_panels = (config.get("evaluation") or {}).get("gene_panels") or {}
    panels: dict[str, list[str]] = {}
    for panel_name, raw_path in raw_panels.items():
        path = Path(str(raw_path))
        if not path.is_file():
            raise FileNotFoundError(f"configured evaluation gene panel {panel_name!r} is missing: {path}")
        payload = json.loads(path.read_text())
        genes = payload.get("genes") if isinstance(payload, dict) else payload
        if not isinstance(genes, list) or not genes:
            raise ValueError(f"evaluation gene panel {panel_name!r} must contain a non-empty gene list: {path}")
        panels[str(panel_name)] = list(dict.fromkeys(str(gene) for gene in genes))
    derived_path = (config.get("evaluation") or {}).get("train_gene_panel_artifact")
    if derived_path:
        if dataset_manifest is None:
            raise ValueError(
                "evaluation.train_gene_panel_artifact is configured, but no dataset manifest was supplied "
                "for its train-only provenance verification"
            )
        artifact = load_train_derived_gene_panels(derived_path, dataset_manifest)
        for panel_name, genes in artifact["panels"].items():
            if panel_name in panels:
                raise ValueError(f"duplicate configured/train-derived evaluation panel name {panel_name!r}")
            panels[panel_name] = list(genes)
    return panels


def architecture4_calibration_summary(standardized_residuals: list[float]) -> dict:
    """Codex re-audit of commit 90f853e, launch blocker #9: "Architecture
    4 interval coverage/calibration" -- Architecture 4 is the only
    architecture reporting a predictive standard deviation
    (`predict_for_metrics`'s `predictive_std`), and nothing previously
    checked whether it was CALIBRATED (does a nominal X% predictive
    interval actually contain the true value X% of the time?), only that
    it existed (`predictive_std_mean`). Standardized residuals
    `z = (true - pred) / predictive_std`, pooled across every query
    spot/gene/item, should have mean ~0 and std ~1 if well-calibrated;
    empirical coverage of the standard normal's own 68/90/95% intervals
    (|z| <= 1.0 / 1.645 / 1.96) should then match those nominal
    fractions. Systematically low coverage means the model is
    OVERCONFIDENT (intervals too narrow); systematically high coverage
    means it is UNDERCONFIDENT (intervals too wide).

    Codex re-audit of commit f7bb8a1, secondary fix #4: "Treat Gaussian
    coverage from 8 flow samples as approximate." `predictive_std` is
    itself estimated from only `n_flow_samples` (8 by default) draws, so
    ASSUMING those standardized residuals are exactly standard-normal
    (rather than checking the actual sample distribution) is itself an
    approximation on top of an approximation. `"method"` is recorded
    explicitly as `"gaussian_std_approximation"` so a caller cannot
    mistake this for a calibrated empirical measurement --
    `architecture4_empirical_calibration_summary` below is the
    higher-fidelity alternative, computed directly from the flow's own
    empirical quantiles rather than a Gaussian assumption."""
    z = np.asarray(standardized_residuals, dtype=np.float64)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return {"n_values": 0}
    abs_z = np.abs(z)
    return {
        "method": "gaussian_std_approximation",
        "n_values": int(z.size),
        "z_mean": float(z.mean()),
        "z_std": float(z.std()),
        "coverage_68": float(np.mean(abs_z <= 1.0)),
        "coverage_90": float(np.mean(abs_z <= 1.645)),
        "coverage_95": float(np.mean(abs_z <= 1.96)),
    }


_EMPIRICAL_COVERAGE_LEVELS = {"coverage_68": 0.68, "coverage_90": 0.90, "coverage_95": 0.95}


class _EmpiricalCoverageAccumulator:
    """Codex re-audit of commit f7bb8a1, secondary fix #4: "preferably
    use empirical quantiles with a larger configurable sample count."
    Accumulates running in-interval counts across items rather than
    storing per-element booleans/values, so memory stays bounded
    regardless of how many items or genes are evaluated -- the same
    bounded-memory discipline as `compute_training_residuals`'s memmap
    (Codex re-audit of commit 90f853e, launch blocker #11)."""

    def __init__(self) -> None:
        self.n_values = 0
        self.in_interval = {name: 0 for name in _EMPIRICAL_COVERAGE_LEVELS}

    def add_item(self, predictive_samples: np.ndarray, true_expression: np.ndarray) -> None:
        true64 = np.asarray(true_expression, dtype=np.float64)
        self.n_values += true64.size
        for name, level in _EMPIRICAL_COVERAGE_LEVELS.items():
            lower_q = (1.0 - level) / 2.0 * 100.0
            upper_q = 100.0 - lower_q
            lo = np.percentile(predictive_samples, lower_q, axis=0)
            hi = np.percentile(predictive_samples, upper_q, axis=0)
            self.in_interval[name] += int(np.sum((true64 >= lo) & (true64 <= hi)))

    def summary(self, n_samples_per_item: int) -> dict:
        if self.n_values == 0:
            return {"n_values": 0}
        return {
            "method": "empirical_quantiles",
            "n_samples_per_item": int(n_samples_per_item),
            "n_values": int(self.n_values),
            **{name: count / self.n_values for name, count in self.in_interval.items()},
        }


def _load_model_for_evaluation(
    config: dict, checkpoint_dir: str | Path, gene_names: list[str], device: torch.device,
    dataset_manifest: dict | None = None, cache_content_by_sample: dict[str, dict] | None = None,
):
    """Construct the real architecture and load a real, already-trained
    checkpoint's trainable weights onto it -- never a random/untrained
    model. Adam's Step 6 audit #2 of commit a32051b: this now calls the
    ONE shared `train.py::build_model_for_inference` pipeline, which
    additionally loads+freezes Architecture 4's exact Architecture 3
    conditioner -- a real, confirmed gap in the PRIOR version of this
    function, which built the architecture and loaded trainable weights
    but never called `maybe_load_pretrained_conditioner_for_architecture4`
    at all, so Architecture 4 evaluation was silently evaluating a
    conditioner that was never correctly loaded/frozen from the real
    Architecture 3 checkpoint the way training did."""
    model, info = build_model_for_inference(
        config, gene_names=gene_names, device=device, checkpoint_dir=checkpoint_dir, smoke=False,
        dataset_manifest=dataset_manifest, cache_content_by_sample=cache_content_by_sample,
    )
    model.eval()
    return model, info


def evaluate_gen3_checkpoint(
    config_path: str, checkpoint_dir: str | Path, *, split: str = "validation",
    n_masks_per_sample: int = 8, use_best: bool = True, compute_st_fid_mmd: bool = False,
    allow_test: bool = False, device_str: str = "cpu", evaluation_seed: int = 0,
    calibration_n_samples: int | None = None, allow_code_drift: bool = False,
) -> dict:
    """The real Step 7 evaluation entrypoint. Runs a REAL, already-trained
    checkpoint over the FIXED, deterministic held-out mask schedule for
    `split`, computing per-item/per-mask PCC+RMSE (with valid-gene
    counts) for the model AND every baseline, then
    `aggregate_patient_metrics` for patient-level means + 95% CIs.

    `split` must be "validation" unless `allow_test=True` is passed
    explicitly -- "Never select using test samples" is enforced here
    structurally, not just documented: this function refuses to touch
    test-split data at all by default.

    Adam's Step 6 audit #7 of commit a32051b extended this report with:
    retained per-item records (`per_item_records`, each carrying
    sample/patient/mask/stratum identity, so a caller can re-slice by any
    of those after the fact -- the prior version discarded every
    per-item value the moment it was folded into the aggregate); paired
    model-vs-baseline deltas with their own patient-level CIs
    (`per_arm_paired_delta_vs_model`, computed item-by-item, not by
    comparing two independently-aggregated means); Architecture 4's
    predictive uncertainty (`predictive_std_mean`, folded into
    `per_arm_patient_aggregated_metrics["model"]` when available); and a
    fail-closed check of the checkpoint's own bundle identity against
    THIS evaluation's dataset/gene-panel before any weights are loaded
    (audit #4's altered-cache/swapped-checkpoint adversarial scenario).

    Codex re-audit of commit f7bb8a1, launch blocker #7: `evaluation_seed`
    (default 0) combines with each item's own content-derived
    `stable_key` (`common_random_validation_seed`) to seed Architecture
    4's stochastic sampling -- never the loop's raw `idx`, which is only
    stable as long as the mask bank's on-disk record order never changes
    (see `common_random_validation_seed`'s own docstring). Two calls with
    the same `(evaluation_seed, checkpoint)` are exactly reproducible
    regardless of mask-bank regeneration order.

    Secondary fix #4: `calibration_n_samples`, when given, overrides
    Architecture 4's configured `n_flow_samples` for every item's flow
    draw and additionally reports `architecture4_empirical_calibration`
    -- empirical-quantile interval coverage computed directly from those
    draws, a higher-fidelity alternative to `architecture4_calibration`'s
    Gaussian-std approximation (see both functions' docstrings)."""
    if split == "test" and not allow_test:
        raise ValueError(
            "evaluate_gen3_checkpoint refuses split='test' unless allow_test=True is passed "
            "explicitly -- Adam's Step 6 audit #12: 'Never select using test samples.' Use "
            "split='validation' for any model-selection or development-time evaluation; test-split "
            "evaluation is a final, one-time report, never a decision input."
        )
    if split not in ("validation", "test"):
        raise ValueError(f"split must be 'validation' or 'test', got {split!r}")

    config = resolved_config(config_path)
    # Integration audit item 9: a Gen4/Gen5 config never sets
    # model.architecture -- `.get(..., "")` avoids a KeyError; `kind` is
    # derived from the real model_info `_load_model_for_evaluation`
    # returns below (works uniformly for Gen3 numeric architectures AND
    # Gen4/5's own model.kind, since build_model_for_inference's Gen3
    # branch also now returns a normalized "kind").
    architecture_id = str((config.get("model") or {}).get("architecture", ""))
    is_gen4_or_gen5 = (config.get("model") or {}).get("arm") is not None
    data_cfg = config["data"]
    dataset_manifest = load_dataset_manifest(data_cfg["gen3_manifest_path"])
    split_ids = list(dataset_manifest[f"{split}_sample_ids"])
    if not split_ids:
        raise ValueError(f"dataset manifest has zero {split}_sample_ids -- nothing to evaluate")

    cfg_om = OmegaConf.create(config)
    expected_provenance = expected_tile_encoder_provenance(config)
    samples, preflight_report = load_and_preflight_samples(cfg_om, dataset_manifest, split_ids, expected_provenance)

    gene_names = list(dataset_manifest["gene_panel"])
    device = torch.device(device_str)
    checkpoint_dir = Path(checkpoint_dir)
    # Codex re-audit of commit 90f853e, launch blocker #4 (and re-audit of
    # commit f7bb8a1, launch blockers #2/#3/#4/#6): "use_best=True
    # silently falls back to latest if best/ is missing... Latest-
    # checkpoint evaluation does not compare the checkpoint's run
    # manifest with the evaluation dataset... bind evaluation to cache
    # content used by training." Both use_best=True and use_best=False
    # now go through the SAME complete verifier
    # (`verify_full_checkpoint_identity`) -- best/ is no longer a
    # separately-verified, partially-checked bespoke bundle: it is a
    # real transactional checkpoint.py bundle, verified identically to
    # the live checkpoint_dir's own latest state, including per-sample
    # cache content identity against what THIS evaluation actually
    # loaded.
    weights_dir = checkpoint_dir / "best" if use_best else checkpoint_dir
    if use_best and not weights_dir.is_dir():
        raise ValueError(
            f"evaluate_gen3_checkpoint: use_best=True but no best/ bundle exists at {checkpoint_dir} "
            "-- refusing to silently fall back to evaluating the latest checkpoint instead. Pass "
            "use_best=False explicitly if evaluating the latest (not best-selected) checkpoint is "
            "genuinely intended."
        )
    # Codex re-audit of commit 66d65f2, finding #3: "Evaluation creates
    # mask-bank files before verifying the checkpoint... under
    # checkpoint_dir/evaluation_masks before identity verification. It
    # also reuses the same filenames for different --n-masks-per-sample,
    # so later evaluations with another mask count can fail as 'stale.'
    # Evaluation should construct deterministic masks in memory or use
    # fingerprinted immutable paths outside the checkpoint." Confirmed
    # real on both counts: mask-bank construction/persistence previously
    # ran BEFORE this checkpoint-identity verification (a checkpoint that
    # turns out to be invalid still left real files behind under
    # checkpoint_dir), and reused a FIXED per-sample filename
    # (`{sample_id}_stratified_mask_bank.json`) regardless of
    # `n_masks_per_sample` -- `mask_schedule.py::load_stratified_mask_
    # bank`'s own strata_fingerprint (which DOES include split_counts)
    # then correctly, but unhelpfully, refuses a second evaluation of the
    # SAME checkpoint at a DIFFERENT mask count as "stale," when nothing
    # is actually wrong. Fixed by verifying identity FIRST, below, then
    # building the mask schedule with no `mask_bank_dir` at all --
    # `build_gen3_mask_schedule` already builds a fully deterministic,
    # reproducible bank in memory (`build_stratified_mask_bank`, a pure
    # function of coords/slice_ids/obs_names/strata/split_counts/
    # split_seeds) and only persists to disk when explicitly given a
    # directory to write into. Evaluation is a one-shot, comparatively
    # cheap computation (unlike training's incremental resumability
    # need) -- Adam's own stated preference ("Prefer deterministic
    # in-memory evaluation mask banks") -- so it never needs the on-disk
    # cache at all, closing both the ordering gap and the stale-filename
    # collision at once.
    #
    # Codex re-audit of commit 7a2d819, finding #2: "The evaluator
    # verifies best/, resolves its identity, then later loads from the
    # mutable best/ pointer again. If training replaces best/ between
    # those operations, the report can identify bundle A while
    # predictions came from bundle B." Confirmed real: `weights_dir`
    # (`checkpoint_dir / "best"`, a MUTABLE path whose `latest_bundle.json`
    # pointer a concurrent training job can rewrite at any time) was
    # independently re-resolved by THREE separate calls below --
    # `verify_full_checkpoint_identity`, `resolve_checkpoint_identity`
    # (for the report), and `_load_model_for_evaluation` -- each of which
    # could in principle resolve to a DIFFERENT bundle if `best/` was
    # replaced in between. Fixed: `resolve_checkpoint_identity(weights_dir)`
    # is now called EXACTLY ONCE, up front, pinning an immutable
    # `pinned_identity.resolved_dir` (a real bundle directory with its own
    # `manifest.json` and no pointer of its own -- `checkpoint.py::
    # _resolve_checkpoint_source`'s own documented "caller directly
    # resolves a HISTORY BUNDLE'S OWN path" exception, the same pattern
    # `checkpoint.py`'s own tests already use to load a specific past
    # snapshot). Every subsequent operation -- verification, model
    # loading, and the report's own recorded identity -- now passes THAT
    # exact resolved directory, never the mutable `weights_dir`, so all
    # three are structurally guaranteed to describe the same bundle: if
    # `best/` is replaced after this point, this evaluation either keeps
    # using the ORIGINAL pinned bundle (verified content, immutable once
    # written) or fails outright -- it can never silently mix identities.
    pinned_identity = checkpoint_module.resolve_checkpoint_identity(weights_dir)
    checkpoint_run_manifest = verify_full_checkpoint_identity(
        pinned_identity.resolved_dir, config=config, dataset_manifest=dataset_manifest, gene_names=gene_names,
        cache_content_by_sample=preflight_report.get("cache_content_by_sample"), allow_code_drift=allow_code_drift,
    )
    strata = config["masking"]["strata"]
    schedule = build_gen3_mask_schedule(
        dataset_manifest, samples, strata, role=split,
        split_counts={split: n_masks_per_sample}, split_seeds={split: 700_000 if split == "validation" else 900_000},
    )
    if is_gen4_or_gen5:
        from gen3_multiscale.gen4.dataset_adapter import Gen4SpatialFieldDataset

        dataset = Gen4SpatialFieldDataset(dataset_manifest, samples, schedule, strata, cfg_om, gene_names)
    else:
        dataset = Gen3SpatialFieldDataset(dataset_manifest, samples, schedule, strata)
    # Codex re-audit of commit 66d65f2, finding #1: "Evaluation reports
    # are not bound to the exact checkpoint... records paths, not the
    # bundle ID, step, manifest SHA, or weights SHA. If best/ later
    # changes, the report no longer proves which weights produced it."
    # `checkpoint_identity` below IS `pinned_identity` -- the SAME
    # resolved identity used for verification and loading, never a fresh,
    # independently re-resolved one (finding #2, above).
    checkpoint_identity = pinned_identity
    recorded_commit = checkpoint_run_manifest.get("code_commit_hash")
    current_commit = _code_commit_hash()
    recorded_diff_hash = checkpoint_run_manifest.get("code_worktree_diff_hash")
    current_diff_hash = _worktree_diff_hash()
    code_drift_present = (
        recorded_commit is None or current_commit is None
        or recorded_commit != current_commit or recorded_diff_hash != current_diff_hash
    )
    model, model_info = _load_model_for_evaluation(
        config, pinned_identity.resolved_dir, gene_names, device, dataset_manifest=dataset_manifest,
        cache_content_by_sample=preflight_report.get("cache_content_by_sample"),
    )
    kind = model_info["kind"]

    gene_panels = load_configured_gene_panels(config, dataset_manifest)
    train_panel_path = (config.get("evaluation") or {}).get("train_gene_panel_artifact")
    train_panel_identity = None
    if train_panel_path:
        train_panel_artifact = load_train_derived_gene_panels(train_panel_path, dataset_manifest)
        train_panel_identity = {
            "path": str(train_panel_path),
            "artifact_sha256": train_panel_artifact["artifact_sha256"],
            "dataset_manifest_fingerprint": train_panel_artifact["dataset_manifest_fingerprint"],
            "method": train_panel_artifact["method"],
        }
    _panel_indices, panel_metadata = resolve_gene_panels(gene_names, gene_panels) if gene_panels else ({}, {})

    per_item_by_arm: dict[str, list[dict]] = {"model": []}
    for name in _BASELINE_PREDICTORS:
        per_item_by_arm[name] = []
    arm_names = ["model", *_BASELINE_PREDICTORS]
    # Secondary fix #3: named-panel metrics for EVERY arm (model AND every
    # baseline), not just the model -- otherwise a caller can never tell
    # whether the model actually beats a trivial baseline ON a clinically
    # relevant panel specifically, only on the full gene set.
    per_panel_items_by_arm: dict[str, dict[str, list[dict]]] = {
        arm: {panel: [] for panel in gene_panels} for arm in arm_names
    }
    per_stratum_model_items: dict[str, list[dict]] = {}
    per_stratum_patient_ids: dict[str, list[str]] = {}
    patient_ids: list[str] = []
    per_item_records: list[dict] = []
    predictive_stds: list[float] = []
    architecture4_standardized_residuals: list[float] = []
    empirical_coverage = _EmpiricalCoverageAccumulator()
    real_expression_for_embedding: list[np.ndarray] = []
    model_expression_for_embedding: list[np.ndarray] = []

    # Audit #1 of commit a32051b, refined by the re-audit of commit
    # f7bb8a1's launch blocker #7: Architecture 4's reported prediction
    # comes from `sample_predictive_distribution`'s predictive mean,
    # reseeded per item from a STABLE content-derived identity
    # (`evaluation_seed` + sample_id + stratum + query_fingerprint via
    # `common_random_validation_seed`), never the loop's raw `idx` --
    # see `common_random_validation_seed`'s own docstring for why `idx`
    # is not stable across a mask-bank regeneration.
    with torch.no_grad():
        for idx in range(len(dataset)):
            inputs, targets = dataset[idx]
            true_expression = np.asarray(targets.query_expression, dtype=np.float32)
            patient_ids.append(str(inputs.patient_id))
            item_identity = dataset.item_identity(idx)
            stable_key = (
                f"{item_identity['sample_id']}:{item_identity['stratum']}:{item_identity['query_fingerprint']}"
            )

            item_generator = torch.Generator(device=device).manual_seed(
                common_random_validation_seed(evaluation_seed, stable_key)
            )
            prediction = predict_for_metrics(
                kind, model, inputs, generator=item_generator, n_samples=calibration_n_samples,
            )
            model_pred = np.asarray(prediction["expression"].detach().cpu().numpy(), dtype=np.float32)
            model_item_metrics = per_item_reconstruction_metrics(model_pred, true_expression)
            if "predictive_std" in prediction:
                predictive_std_arr = prediction["predictive_std"].detach().cpu().numpy().astype(np.float64)
                item_predictive_std = float(predictive_std_arr.mean())
                model_item_metrics["predictive_std_mean"] = item_predictive_std
                predictive_stds.append(item_predictive_std)
                # Launch blocker #9: pooled standardized residuals for
                # `architecture4_calibration_summary` below -- every
                # query spot/gene in this item, never just the mean.
                safe_std = np.where(predictive_std_arr > 1e-8, predictive_std_arr, np.nan)
                z = (true_expression.astype(np.float64) - model_pred.astype(np.float64)) / safe_std
                architecture4_standardized_residuals.extend(z.flatten().tolist())
                # Secondary fix #4: empirical-quantile coverage computed
                # directly from this item's raw flow draws, only when the
                # caller actually asked for a (larger) calibration sample
                # count -- never silently swaps in the training-time
                # n_flow_samples=8 draws for this purpose, since that
                # count was never chosen for calibration reliability.
                if calibration_n_samples is not None and "predictive_samples" in prediction:
                    samples_arr = prediction["predictive_samples"].detach().cpu().numpy().astype(np.float64)
                    empirical_coverage.add_item(samples_arr, true_expression)
            per_item_by_arm["model"].append(model_item_metrics)

            stratum = item_identity["stratum"]
            record = {
                "idx": idx, "sample_id": item_identity["sample_id"], "patient_id": str(inputs.patient_id),
                "stratum": stratum, "query_fingerprint": item_identity["query_fingerprint"],
                "model": model_item_metrics,
            }
            if stratum is not None:
                per_stratum_model_items.setdefault(str(stratum), []).append(model_item_metrics)
                per_stratum_patient_ids.setdefault(str(stratum), []).append(str(inputs.patient_id))

            if gene_panels:
                model_panel_item_metrics = gene_panel_metrics(model_pred, true_expression, gene_names, gene_panels)
                record["gene_panels"] = model_panel_item_metrics
                for panel_name, panel_metrics in model_panel_item_metrics.items():
                    per_panel_items_by_arm["model"][panel_name].append(panel_metrics)

            for name, predictor in _BASELINE_PREDICTORS.items():
                baseline_pred = predictor(inputs)
                # `baseline_item_metrics` is appended AS-IS into
                # `per_item_by_arm[name]`, which `aggregate_patient_metrics`
                # later iterates key-by-key expecting every value to be a
                # plain float -- panel metrics must be attached to a
                # SEPARATE dict (`record`), never mutated into this one,
                # or aggregation would try to float()-cast a nested dict.
                baseline_item_metrics = per_item_reconstruction_metrics(baseline_pred, true_expression)
                per_item_by_arm[name].append(baseline_item_metrics)
                record[name] = baseline_item_metrics
                if gene_panels:
                    baseline_panel_item_metrics = gene_panel_metrics(
                        baseline_pred, true_expression, gene_names, gene_panels,
                    )
                    record.setdefault("baseline_gene_panels", {})[name] = baseline_panel_item_metrics
                    for panel_name, panel_metrics in baseline_panel_item_metrics.items():
                        per_panel_items_by_arm[name][panel_name].append(panel_metrics)

            per_item_records.append(record)

            if compute_st_fid_mmd:
                real_expression_for_embedding.append(true_expression)
                model_expression_for_embedding.append(model_pred)

    aggregated = {
        arm: aggregate_patient_metrics(items, patient_ids) for arm, items in per_item_by_arm.items()
    }
    if predictive_stds:
        aggregated["model"]["predictive_std_mean"] = float(np.mean(predictive_stds))

    # Launch blocker #9: "per-stratum results" -- the SAME patient-safe
    # aggregation used for the whole split, restricted to items sharing
    # one masking.strata name, so a caller can tell whether performance
    # is uniform across hole sizes/kinds rather than dominated by the
    # easiest stratum.
    per_stratum_aggregated_metrics = {
        stratum: aggregate_patient_metrics(items, per_stratum_patient_ids[stratum])
        for stratum, items in per_stratum_model_items.items()
    }

    # Launch blocker #9 + secondary fix #3: "configured named panels" for
    # EVERY arm, keyed `[panel][arm]` -- real per-panel patient-safe
    # aggregation for the model AND every baseline, not just per-item
    # numbers buried in per_item_records.
    per_panel_patient_aggregated_metrics = {
        panel: {
            arm: aggregate_patient_metrics(per_panel_items_by_arm[arm][panel], patient_ids) for arm in arm_names
        }
        for panel in gene_panels
    }

    # Launch blocker #9: "Architecture 4 interval coverage/calibration" --
    # empty ({"n_values": 0}, not omitted) for Architectures 1-3, which
    # report no predictive_std at all. Gaussian approximation only.
    architecture4_calibration = architecture4_calibration_summary(architecture4_standardized_residuals)
    # Secondary fix #4: higher-fidelity empirical-quantile alternative,
    # only populated when calibration_n_samples was actually requested.
    architecture4_empirical_calibration = empirical_coverage.summary(calibration_n_samples or 0)

    # Audit #7: PAIRED model-vs-baseline deltas -- computed ITEM BY ITEM
    # (same mask, same sample) rather than by differencing two
    # independently-aggregated means, then run through the same
    # `aggregate_patient_metrics` patient-level-CI machinery used for the
    # raw metrics. Positive pcc_delta / positive rmse_delta both mean
    # "the model beat this baseline" on that item.
    def _paired_deltas(model_items: list[dict], baseline_items: list[dict]) -> dict:
        deltas = [
            {
                "pcc_delta": model_item["pcc"] - baseline_item["pcc"],
                "rmse_delta": baseline_item["rmse"] - model_item["rmse"],
            }
            for model_item, baseline_item in zip(model_items, baseline_items)
        ]
        return aggregate_patient_metrics(deltas, patient_ids)

    paired_deltas = {
        name: _paired_deltas(per_item_by_arm["model"], per_item_by_arm[name]) for name in _BASELINE_PREDICTORS
    }
    # Secondary fix #3: "paired deltas for every baseline too" -- restricted
    # to a named panel, using the SAME patient_ids ordering (one entry per
    # item, regardless of panel), since every item contributes to every
    # configured panel.
    per_panel_paired_delta_vs_model = {
        panel: {
            name: _paired_deltas(per_panel_items_by_arm["model"][panel], per_panel_items_by_arm[name][panel])
            for name in _BASELINE_PREDICTORS
        }
        for panel in gene_panels
    }

    # Codex re-audit of commit 66d65f2, finding #1: a durable identity
    # binding, independent of `checkpoint_dir`/`weights_dir` remaining
    # whatever they currently point at -- a later `best/` replacement (or
    # any other change to those paths) cannot silently invalidate what
    # this ALREADY-COMPUTED report proves about which exact weights
    # produced it.
    checkpoint_identity_record = {
        "resolved_bundle_dir": checkpoint_identity.bundle_dir,
        "step": checkpoint_identity.step,
        "bundle_manifest_sha256": checkpoint_identity.manifest_sha256,
        "weights_sha256": checkpoint_identity.weights_sha256,
        "config_identity_fingerprint": config_identity_fingerprint(config),
        "dataset_manifest_fingerprint": dataset_manifest_fingerprint(dataset_manifest),
        "gene_panel_hash": gene_panel_hash(gene_names),
        "mask_schedule_fingerprint": hashlib.sha256(
            json.dumps(schedule.reports, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
        "use_best": bool(use_best),
        "n_masks_per_sample": int(n_masks_per_sample),
        "code_drift_present": bool(code_drift_present),
        "code_drift_acknowledged": bool(code_drift_present and allow_code_drift),
    }

    report = {
        "version": 6,
        "kind": "gen3_step7_evaluation_report",
        "config_path": str(config_path),
        "checkpoint_dir": str(checkpoint_dir),
        "weights_dir": str(weights_dir),
        "checkpoint_identity": checkpoint_identity_record,
        "split": split,
        "n_samples": len(split_ids),
        "n_items": len(dataset),
        "evaluation_seed": int(evaluation_seed),
        "cache_preflight_report": preflight_report,
        "per_arm_patient_aggregated_metrics": aggregated,
        "per_arm_paired_delta_vs_model": paired_deltas,
        "per_item_records": per_item_records,
        # Launch blocker #9.
        "per_stratum_patient_aggregated_metrics": per_stratum_aggregated_metrics,
        "gene_panel_metadata": panel_metadata,
        "train_gene_panel_artifact": train_panel_identity,
        "per_panel_patient_aggregated_metrics": per_panel_patient_aggregated_metrics,
        "per_panel_paired_delta_vs_model": per_panel_paired_delta_vs_model,
        "architecture4_calibration": architecture4_calibration,
        "architecture4_empirical_calibration": architecture4_empirical_calibration,
    }

    if compute_st_fid_mmd and len(real_expression_for_embedding) >= 2:
        # Secondary, distributional metric only -- never gates selection.
        # embed_pca needs a PCA fit on real data first; a small, local fit
        # here (never persisted/reused) is honest about being a cheap,
        # secondary diagnostic, not a calibrated embedding space.
        from sklearn.decomposition import PCA

        real_matrix = np.concatenate(real_expression_for_embedding, axis=0)
        model_matrix = np.concatenate(model_expression_for_embedding, axis=0)
        n_components = min(10, real_matrix.shape[0] - 1, real_matrix.shape[1])
        if n_components >= 2:
            pca_model = PCA(n_components=n_components).fit(real_matrix)
            real_embeddings = embed_pca(real_matrix, pca_model)
            model_embeddings = embed_pca(model_matrix, pca_model)
            report["secondary_st_fid"] = st_fid(real_embeddings, model_embeddings)
            report["secondary_st_mmd"] = st_mmd(real_embeddings, model_embeddings)

    return report


def save_evaluation_report(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)
    return path


def _print_evaluation_summary(report: dict) -> None:
    """A concise, human-readable stdout summary -- never a replacement
    for the persisted report itself, which already carries every number
    in full detail. Codex re-audit of commit 2162ff4, finding #1: "print
    a concise summary" as part of a real CLI."""
    model_metrics = report["per_arm_patient_aggregated_metrics"]["model"]
    lines = [
        f"split={report['split']} n_items={report['n_items']} n_samples={report['n_samples']}",
        f"model pcc: patient_mean={model_metrics['pcc']['patient_mean']:.4f} "
        f"(n_patients={model_metrics['pcc']['n_patients']})",
        f"model rmse: patient_mean={model_metrics['rmse']['patient_mean']:.4f}",
    ]
    for name, delta in report["per_arm_paired_delta_vs_model"].items():
        lines.append(f"model vs {name}: pcc_delta patient_mean={delta['pcc_delta']['patient_mean']:.4f}")
    calibration = report.get("architecture4_calibration") or {}
    if calibration.get("n_values", 0):
        lines.append(
            f"architecture4 calibration ({calibration['method']}): "
            f"coverage_68={calibration['coverage_68']:.3f} coverage_90={calibration['coverage_90']:.3f} "
            f"coverage_95={calibration['coverage_95']:.3f}"
        )
    print("\n".join(lines))


def main() -> None:
    """The real, runnable Step 7 evaluation CLI. Codex re-audit of commit
    2162ff4, finding #1 (confirmed real): this module had `evaluate_gen3_
    checkpoint`/`save_evaluation_report` as importable functions, but no
    `main()` or `__main__` block at all -- "python -m
    gen3_multiscale.evaluation.gen3_evaluator ... silently exits without
    evaluating or saving a report" was literally true, since there was no
    code path for it to run at all. Every parameter `evaluate_gen3_
    checkpoint` accepts is exposed here; the report is always atomically
    persisted (`save_evaluation_report`, tmp-then-`os.replace`, matching
    every other artifact write in this package); a raised exception
    (fail-closed identity/provenance checks, a missing checkpoint, an
    explicit test-split lock, ...) is reported on stderr and exits
    non-zero rather than propagating a raw traceback as the only signal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="A real gen3 architectureN.yaml (data.gen3_manifest_path must be set)")
    parser.add_argument("--checkpoint-dir", required=True, help="A real, already-trained checkpoint_dir")
    parser.add_argument("--output", required=True, help="Path to atomically write the evaluation report JSON to")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--n-masks-per-sample", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--evaluation-seed", type=int, default=0,
        help="Combines with each item's content-derived identity to seed Architecture 4's stochastic "
             "sampling -- two calls with the same seed against the same checkpoint reproduce bit-identical "
             "per-item predictions.",
    )
    parser.add_argument(
        "--calibration-n-samples", type=int, default=None,
        help="Architecture 4 only: draw this many flow samples per item (instead of the trained "
             "n_flow_samples) and report a higher-fidelity empirical-quantile calibration alongside the "
             "default Gaussian-std approximation.",
    )
    parser.add_argument("--compute-st-fid-mmd", action="store_true", help="Secondary, distributional diagnostic -- never gates selection.")
    parser.add_argument(
        "--allow-test", action="store_true",
        help="Explicit unlock required for --split test -- 'Never select using test samples.' Test-split "
             "evaluation is a final, one-time report, never a model-selection or development-time input.",
    )
    parser.add_argument(
        "--allow-code-drift", action="store_true",
        help="Explicit override to evaluate a checkpoint under a different git commit or dirty worktree "
             "than the one it was trained under. Codex re-audit of commit 2162ff4, finding #4. Never a "
             "silent bypass -- required whenever code state changed (or is unverifiable) since training.",
    )
    use_best = parser.add_mutually_exclusive_group()
    use_best.add_argument(
        "--use-best", dest="use_best", action="store_true", default=True,
        help="Evaluate the best/ bundle (default).",
    )
    use_best.add_argument(
        "--no-use-best", dest="use_best", action="store_false",
        help="Evaluate the latest checkpoint instead of best/.",
    )
    args = parser.parse_args()

    try:
        report = evaluate_gen3_checkpoint(
            args.config, args.checkpoint_dir, split=args.split, n_masks_per_sample=args.n_masks_per_sample,
            use_best=args.use_best, compute_st_fid_mmd=args.compute_st_fid_mmd, allow_test=args.allow_test,
            device_str=args.device, evaluation_seed=args.evaluation_seed,
            calibration_n_samples=args.calibration_n_samples, allow_code_drift=args.allow_code_drift,
        )
    except Exception as exc:
        print(f"gen3_evaluator: evaluation FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)

    saved_path = save_evaluation_report(report, args.output)
    print(f"evaluation report saved to {saved_path}")
    _print_evaluation_summary(report)


if __name__ == "__main__":
    main()
