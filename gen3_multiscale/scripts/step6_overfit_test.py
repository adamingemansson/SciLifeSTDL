#!/usr/bin/env python3
"""Step 6 deliverable: tiny single-sample overfit/capacity test.

Takes a real architectureN.yaml (with data.gen3_manifest_path already set
to a real, on-disk dataset manifest) and one manifest sample id, builds a
DERIVED manifest restricted to exactly that one training sample and ZERO
validation/test samples (written to a fresh temp file -- the original,
immutable manifest on disk is never modified), then runs
training/train.py's real `run_training` for a small, explicit number of
real (non-smoke) optimizer steps with no held-out masking at all.

This is the standard "can this architecture memorize one sample" sanity
check before trusting a longer multi-sample run -- reuses run_training's
entire real orchestration (preflight, dataset, model, synchronized init,
checkpointing, finite-loss checks) unmodified; the only thing this script
does is narrow the manifest and steps before calling it.

Real, confirmed gap fixed (Codex audit of commit 27e1232): "Merely
completing 200 steps is not a capacity test." A prior version of this
script only ran `run_training` and reported whatever summary it
returned -- a run that trains for 200 steps while learning NOTHING would
previously "pass" with no way to tell. `run_overfit_gate` now evaluates
one FIXED (context, query) mask -- the exact same draw, deterministic by
construction (n_training_masks_per_sample=1 makes item 0 a pure function
of sample_id, never re-sampled) -- BEFORE training (using the model's
real synchronized-init weights) and AFTER (using the real trained
checkpoint), and RAISES if RMSE did not improve by at least
`--min-rmse-improvement-fraction` (default 10%). Deliberately does not
default --n-steps high enough to look like a real training run: the
caller must explicitly ask for more steps if they want a longer capacity
check.

    python -m gen3_multiscale.scripts.step6_overfit_test \\
        --config gen3_multiscale/configs/architecture1.yaml \\
        --sample-id INT1 --n-steps 200 --checkpoint-dir /tmp/overfit_arch1
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from gen3_multiscale.data.dataset_manifest import load_dataset_manifest, save_dataset_manifest
from gen3_multiscale.evaluation.gen3_evaluator import per_item_reconstruction_metrics
from gen3_multiscale.training.gen3_dataset import Gen3SpatialFieldDataset, build_gen3_mask_schedule
from gen3_multiscale.training.gen3_preflight import load_and_preflight_samples
from gen3_multiscale.training.train import (
    build_model_for_inference, expected_tile_encoder_provenance, model_kind_for_architecture_id,
    predict_for_metrics, run_training,
)


def build_single_sample_config(config_path: str | Path, sample_id: str, n_steps: int, checkpoint_dir: str | Path) -> Path:
    """Derive a temp config + temp manifest restricted to one training
    sample and zero validation/test samples. Returns the temp config's
    path (the temp manifest it references lives alongside it)."""
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    manifest_path = (config.get("data") or {}).get("gen3_manifest_path")
    if not manifest_path:
        raise ValueError(f"{config_path}: data.gen3_manifest_path must be set to a real dataset manifest")
    manifest = load_dataset_manifest(manifest_path)
    if sample_id not in manifest.get("samples", {}):
        raise ValueError(f"{sample_id!r} is not a sample the dataset manifest at {manifest_path} declares")
    if int(n_steps) < 1:
        raise ValueError(f"--n-steps must be positive, got {n_steps}")

    overfit_manifest = dict(manifest)
    overfit_manifest["train_sample_ids"] = [sample_id]
    overfit_manifest["validation_sample_ids"] = []
    overfit_manifest["test_sample_ids"] = []

    tmp_dir = Path(tempfile.mkdtemp(prefix="gen3_step6_overfit_"))
    overfit_manifest_path = save_dataset_manifest(overfit_manifest, tmp_dir / "overfit_manifest.json")

    config["data"]["gen3_manifest_path"] = str(overfit_manifest_path)
    config["training"]["total_steps"] = int(n_steps)
    config["training"]["checkpoint_dir"] = str(checkpoint_dir)
    # No held-out samples exist in this derived manifest, so there is
    # nothing for periodic validation to evaluate against -- push it out
    # of reach rather than leaving a val_loader-is-None branch to be
    # silently skipped every step for no visible reason.
    config["training"]["eval_every_n_steps"] = int(n_steps) + 1
    overfit_config_path = tmp_dir / "overfit_config.yaml"
    OmegaConf.save(OmegaConf.create(config), overfit_config_path)
    return overfit_config_path


def _build_fixed_eval_item(config: dict, manifest: dict, sample_id: str):
    """One deterministic (context, query) draw on the single overfit
    sample. `n_training_masks_per_sample=1` makes item 0 a pure function
    of `sample_id` (mask_fingerprint's own per-sample seed derivation,
    Adam's Step 6 audit #10) -- IDENTICAL every time this is called,
    never re-sampled, so "before" and "after" genuinely evaluate the same
    hole."""
    cfg_om = OmegaConf.create(config)
    expected_provenance = expected_tile_encoder_provenance(config)
    samples, _preflight_report = load_and_preflight_samples(cfg_om, manifest, [sample_id], expected_provenance)
    strata = config["masking"]["strata"]
    schedule = build_gen3_mask_schedule(manifest, samples, strata, role="train", n_training_masks_per_sample=1)
    dataset = Gen3SpatialFieldDataset(manifest, samples, schedule, strata)
    return dataset[0]


def _evaluate_fixed_item(kind: str, model: torch.nn.Module, item, *, generator=None) -> dict:
    """Adam's Step 6 audit #1 of commit a32051b: Architecture 4's
    capacity-gate metric must come from `sample_predictive_distribution`'s
    predictive mean, not `forward()`'s frozen conditioner -- the prior
    version called `model(inputs)` unconditionally, so an Architecture 4
    overfit gate could "pass" purely because its ALREADY-TRAINED
    conditioner reconstructs its one memorized sample well, without the
    flow apparatus this test is meant to be exercising ever being
    evaluated at all. `generator`, when given, makes repeated calls
    (before/after training) reproducible. `kind` (Integration audit
    finding #1) is `model_kind_for_architecture_id(architecture_id)` --
    this script is Gen3-only (builds items via `Gen3SpatialFieldDataset`
    directly), so `kind` is always `"conditioner"` or `"flow"` here."""
    inputs, targets = item
    model.eval()
    with torch.no_grad():
        prediction = predict_for_metrics(kind, model, inputs, generator=generator)
    model.train()
    pred = np.asarray(prediction["expression"].detach().cpu().numpy(), dtype=np.float32)
    true = np.asarray(targets.query_expression, dtype=np.float32)
    return per_item_reconstruction_metrics(pred, true)


def run_overfit_gate(
    config_path: str, sample_id: str, n_steps: int, checkpoint_dir: str,
    *, min_rmse_improvement_fraction: float = 0.1,
) -> dict:
    """The real overfit/capacity GATE (not just a run): evaluates the
    SAME fixed mask before training (real synchronized-init weights, plus
    -- for Architecture 4 -- the real, already-trained Architecture 3
    conditioner) and after (the real trained checkpoint), and RAISES if
    RMSE did not improve by at least `min_rmse_improvement_fraction`.
    "Merely completing N steps is not a capacity test" (Adam's Step 6
    audit #4) -- this function is what actually tests capacity.

    Adam's Step 6 audit #2 of commit a32051b: model construction now goes
    through the ONE shared `train.py::build_model_for_inference` pipeline
    -- the prior version of this function built the architecture inline
    and never called `maybe_load_pretrained_conditioner_for_architecture4`
    at all, so an Architecture 4 overfit gate was silently evaluating a
    conditioner that was never correctly loaded/frozen from the real
    Architecture 3 checkpoint the way training did. Audit #1: both
    before/after evaluations now use `predict_for_metrics` (Architecture
    4's real predictive-mean sample, not `forward()`'s frozen
    conditioner), reseeded from the SAME fixed seed both times, so the
    ONLY thing that can differ between "before" and "after" is the
    trained flow weights themselves -- direct behavioral proof, not just
    documentation, that a passing gate reflects a real change in
    Architecture 4's reported prediction."""
    overfit_config_path = build_single_sample_config(config_path, sample_id, n_steps, checkpoint_dir)
    config = OmegaConf.to_container(OmegaConf.load(overfit_config_path), resolve=True)
    overfit_manifest = load_dataset_manifest(config["data"]["gen3_manifest_path"])
    fixed_item = _build_fixed_eval_item(config, overfit_manifest, sample_id)

    architecture_id = str(config["model"]["architecture"])
    kind = model_kind_for_architecture_id(architecture_id)
    gene_names = list(overfit_manifest["gene_panel"])
    seed = int(config["training"].get("seed", 0))
    device = torch.device(config["training"].get("device", "cpu") if torch.cuda.is_available() else "cpu")

    before_model, _before_info = build_model_for_inference(
        config, gene_names=gene_names, device=device, checkpoint_dir=None, smoke=False,
        dataset_manifest=overfit_manifest,
    )
    before_generator = torch.Generator(device=device).manual_seed(seed)
    before_metrics = _evaluate_fixed_item(kind, before_model, fixed_item, generator=before_generator)

    training_summary = run_training(str(overfit_config_path), smoke=False)

    after_model, _after_info = build_model_for_inference(
        config, gene_names=gene_names, device=device, checkpoint_dir=checkpoint_dir, smoke=False,
        dataset_manifest=overfit_manifest,
    )
    after_generator = torch.Generator(device=device).manual_seed(seed)
    after_metrics = _evaluate_fixed_item(kind, after_model, fixed_item, generator=after_generator)

    rmse_before, rmse_after = before_metrics["rmse"], after_metrics["rmse"]
    relative_improvement = (rmse_before - rmse_after) / rmse_before if rmse_before > 0 else float("nan")
    passed = bool(np.isfinite(relative_improvement) and relative_improvement >= min_rmse_improvement_fraction)

    gate_report = {
        "sample_id": sample_id, "n_steps": int(n_steps), "before": before_metrics, "after": after_metrics,
        "rmse_relative_improvement": float(relative_improvement),
        "min_rmse_improvement_fraction_required": float(min_rmse_improvement_fraction),
        "passed": passed, "training_summary": training_summary,
    }
    if not passed:
        raise RuntimeError(
            f"overfit gate FAILED for sample {sample_id!r}: RMSE went from {rmse_before:.6f} to "
            f"{rmse_after:.6f} ({relative_improvement:.1%} relative improvement, needed >= "
            f"{min_rmse_improvement_fraction:.1%}) -- this architecture/config combination does not "
            f"appear to be learning even on a single, memorizable sample. Full report: {gate_report}"
        )
    return gate_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="A real architectureN.yaml with data.gen3_manifest_path set")
    parser.add_argument("--sample-id", required=True, help="Exactly one manifest sample id to overfit on")
    parser.add_argument("--n-steps", type=int, default=200, help="Real optimizer steps (default: 200)")
    parser.add_argument("--checkpoint-dir", required=True, help="Fresh checkpoint dir for this overfit run")
    parser.add_argument(
        "--min-rmse-improvement-fraction", type=float, default=0.1,
        help="Required relative RMSE improvement on the fixed eval mask (default: 0.1 = 10%%)",
    )
    args = parser.parse_args()

    gate_report = run_overfit_gate(
        args.config, args.sample_id, args.n_steps, args.checkpoint_dir,
        min_rmse_improvement_fraction=args.min_rmse_improvement_fraction,
    )
    print(f"overfit gate PASSED: {json.dumps(gate_report, indent=2, default=str)}")


if __name__ == "__main__":
    main()
