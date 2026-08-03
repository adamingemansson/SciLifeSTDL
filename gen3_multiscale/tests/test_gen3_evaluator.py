"""Tests for gen3_multiscale/evaluation/gen3_evaluator.py -- Adam's Step
6 audit #12 deliverable: the minimal real Step 7 evaluator. Real, small,
end-to-end synthetic data throughout."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from gen3_multiscale.evaluation.gen3_evaluator import (
    evaluate_gen3_checkpoint, harmonic_baseline_prediction, load_configured_gene_panels,
    mean_baseline_prediction, nearest_neighbor_baseline_prediction, per_item_reconstruction_metrics,
    save_evaluation_report, zero_image_content,
)
from gen3_multiscale.tests._step6_fixtures import prepare_step6_experiment, write_step6_train_config
from gen3_multiscale.training import train as train_module


def test_per_item_reconstruction_metrics_reports_pcc_rmse_and_valid_gene_counts():
    rng = np.random.default_rng(0)
    true = rng.normal(size=(10, 5)).astype(np.float32)
    pred = true + rng.normal(scale=0.01, size=true.shape).astype(np.float32)
    metrics = per_item_reconstruction_metrics(pred, true)
    assert metrics["pcc"] > 0.9
    assert metrics["rmse"] < 0.1
    assert metrics["n_genes"] == 5
    assert metrics["n_valid_genes"] == 5
    # Launch blocker #9: "nonzero AUC" -- a near-perfect predictor should
    # separate zero/measured-absent from nonzero true expression almost
    # perfectly too.
    assert metrics["nonzero_auc"] > 0.9


def test_per_item_reconstruction_metrics_excludes_truth_constant_genes_from_valid_count():
    true = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]], dtype=np.float32)  # gene 1 is constant
    pred = true.copy()
    metrics = per_item_reconstruction_metrics(pred, true)
    assert metrics["n_genes"] == 2
    assert metrics["n_valid_genes"] == 1


def test_per_item_reconstruction_metrics_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        per_item_reconstruction_metrics(np.zeros((3, 2)), np.zeros((3, 3)))


class _FakeInputs:
    def __init__(self, observed_coords, observed_full_gene_expression, query_coords):
        self.observed_coords = observed_coords
        self.observed_full_gene_expression = observed_full_gene_expression
        self.query_coords = query_coords


def test_mean_baseline_prediction_is_the_observed_mean_broadcast_to_every_query():
    inputs = _FakeInputs(
        observed_coords=np.zeros((3, 2)),
        observed_full_gene_expression=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        query_coords=np.zeros((2, 2)),
    )
    pred = mean_baseline_prediction(inputs)
    assert pred.shape == (2, 2)
    assert np.allclose(pred, [3.0, 4.0])


def test_nearest_neighbor_baseline_prediction_picks_the_closest_observed_spot():
    inputs = _FakeInputs(
        observed_coords=np.array([[0.0, 0.0], [10.0, 10.0]]),
        observed_full_gene_expression=np.array([[1.0, 1.0], [9.0, 9.0]], dtype=np.float32),
        query_coords=np.array([[0.1, 0.1], [9.9, 9.9]]),
    )
    pred = nearest_neighbor_baseline_prediction(inputs)
    assert np.allclose(pred, [[1.0, 1.0], [9.0, 9.0]])


def test_harmonic_baseline_prediction_returns_the_right_shape():
    inputs = _FakeInputs(
        observed_coords=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        observed_full_gene_expression=np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32),
        query_coords=np.array([[0.5, 0.5]]),
    )
    pred = harmonic_baseline_prediction(inputs)
    assert pred.shape == (1, 1)


def test_zero_image_content_removes_every_image_path_but_preserves_expression_and_geometry():
    from gen3_multiscale.tests._gen4_fixtures import (
        synthetic_gen4_inputs, with_synthetic_uni2_features, with_synthetic_wsi_context,
    )

    inputs, _targets = synthetic_gen4_inputs(image_dim=8)
    inputs = with_synthetic_wsi_context(inputs, image_dim=8)
    inputs = with_synthetic_uni2_features(inputs, image_dim=8)

    ablated = zero_image_content(inputs)

    assert not np.any(ablated.observed_gigapath_features)
    assert not np.any(ablated.observed_image_available)
    assert not np.any(ablated.wsi_tile_features)
    assert not np.any(ablated.observed_uni2_features)
    np.testing.assert_array_equal(ablated.observed_full_gene_expression, inputs.observed_full_gene_expression)
    np.testing.assert_array_equal(ablated.observed_coords, inputs.observed_coords)
    np.testing.assert_array_equal(ablated.query_coords, inputs.query_coords)
    np.testing.assert_array_equal(ablated.wsi_tile_longnet_coords, inputs.wsi_tile_longnet_coords)
    np.testing.assert_array_equal(ablated.wsi_tile_regional_coords, inputs.wsi_tile_regional_coords)
    np.testing.assert_array_equal(ablated.boundary_idx, inputs.boundary_idx)
    # The original example is not mutated, which permits normal and
    # ablated predictions to be compared from the same dataset item.
    assert np.any(inputs.observed_gigapath_features)
    assert np.any(inputs.wsi_tile_features)
    assert np.any(inputs.observed_uni2_features)


def _build_synchronized_init_dir(tmp_path, manifest):
    from gen3_multiscale.models import model_factory as mf
    from gen3_multiscale.models.gene_basis import fit_gene_residual_basis
    from gen3_multiscale.tests._step6_fixtures import step6_model_params

    gene_names = list(manifest["gene_panel"])
    n_genes = len(gene_names)
    residuals = np.random.default_rng(1).normal(size=(10, n_genes)).astype(np.float32)
    basis = fit_gene_residual_basis(residuals, gene_names, rank=4)
    common = dict(n_genes=n_genes, gex_feature_dim=8, seed=0)
    models = {
        "architecture1": mf.build_architecture({"model": {"architecture": "1", "params": step6_model_params("1")}}, **common),
        "architecture2": mf.build_architecture({"model": {"architecture": "2", "params": step6_model_params("2", use_anchor_blend=True)}}, **common),
        "architecture3": mf.build_architecture({"model": {"architecture": "3", "params": step6_model_params("3", use_regional_he=True, use_global_gex=True)}}, **common),
        "architecture4": mf.build_architecture(
            {"model": {"architecture": "4", "params": step6_model_params("4", use_regional_he=True)}},
            **common, gene_basis=basis, gene_names=gene_names,
        ),
    }
    sync_dir = tmp_path / "sync"
    mf.persist_four_architecture_initializations(models, sync_dir)
    return sync_dir


def _train_a_real_checkpoint(tmp_path, cfg, manifest, manifest_path):
    """A REAL (non-smoke) checkpoint -- smoke mode never writes
    trainable_weights.pt/gene_names.json, so the evaluator (which loads a
    real checkpoint) needs a genuine one to evaluate."""
    sync_dir = _build_synchronized_init_dir(tmp_path, manifest)
    config_path = tmp_path / "arch1_config.yaml"
    checkpoint_dir = tmp_path / "arch1_ckpt"
    write_step6_train_config(
        cfg, manifest_path, config_path, architecture="1", checkpoint_dir=checkpoint_dir,
        synchronized_init_dir=str(sync_dir),
    )
    config = yaml.safe_load(config_path.read_text())
    config["training"]["total_steps"] = 1
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    train_module.run_training(str(config_path), smoke=False)
    return config_path, checkpoint_dir


def test_evaluate_gen3_checkpoint_writes_no_files_under_checkpoint_dir_when_identity_verification_fails(tmp_path, monkeypatch):
    """Codex re-audit of commit 66d65f2, finding #3: 'Evaluation creates
    mask-bank files before verifying the checkpoint... under
    checkpoint_dir/evaluation_masks before identity verification.' Trains
    a real checkpoint, then evaluates against a config whose dataset
    manifest has been swapped for a mutated one (so
    verify_full_checkpoint_identity fails on dataset_manifest_fingerprint)
    -- proves NO new file appears anywhere under checkpoint_dir as a
    side effect of the failed attempt."""
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    config_path, checkpoint_dir = _train_a_real_checkpoint(tmp_path, cfg, manifest, manifest_path)

    from gen3_multiscale.data.dataset_manifest import load_dataset_manifest, save_dataset_manifest

    mutated_manifest = dict(load_dataset_manifest(manifest_path))
    mutated_manifest["gene_panel"] = list(mutated_manifest["gene_panel"])[::-1]
    save_dataset_manifest(mutated_manifest, manifest_path)

    files_before = sorted(p.relative_to(checkpoint_dir) for p in checkpoint_dir.rglob("*") if p.is_file())
    with pytest.raises(ValueError, match="dataset_manifest_fingerprint"):
        evaluate_gen3_checkpoint(str(config_path), checkpoint_dir, split="validation", n_masks_per_sample=2)
    files_after = sorted(p.relative_to(checkpoint_dir) for p in checkpoint_dir.rglob("*") if p.is_file())
    assert files_after == files_before
    assert not (checkpoint_dir / "evaluation_masks").exists()


def test_evaluate_gen3_checkpoint_supports_different_mask_counts_without_stale_bank_collisions(tmp_path, monkeypatch):
    """Codex re-audit of commit 66d65f2, finding #3: '[evaluation] also
    reuses the same filenames for different --n-masks-per-sample, so
    later evaluations with another mask count can fail as stale.'
    Evaluating the SAME checkpoint twice with two DIFFERENT
    n_masks_per_sample values, back to back, must both succeed (in-memory
    mask construction has no persisted filename to collide over)."""
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    config_path, checkpoint_dir = _train_a_real_checkpoint(tmp_path, cfg, manifest, manifest_path)

    report_a = evaluate_gen3_checkpoint(str(config_path), checkpoint_dir, split="validation", n_masks_per_sample=2)
    report_b = evaluate_gen3_checkpoint(str(config_path), checkpoint_dir, split="validation", n_masks_per_sample=4)
    assert report_a["n_items"] != report_b["n_items"]
    assert report_a["checkpoint_identity"]["n_masks_per_sample"] == 2
    assert report_b["checkpoint_identity"]["n_masks_per_sample"] == 4
    assert not (checkpoint_dir / "evaluation_masks").exists()


def test_evaluate_gen3_checkpoint_never_mixes_identities_if_best_is_replaced_mid_evaluation(tmp_path, monkeypatch):
    """Codex re-audit of commit 7a2d819, finding #2: 'The evaluator
    verifies best/, resolves its identity, then later loads from the
    mutable best/ pointer again. If training replaces best/ between
    those operations, the report can identify bundle A while predictions
    came from bundle B. The new test only replaces best/ after
    evaluation, so it misses this.' This test replaces `best/` with a
    genuinely different bundle (different weights, different bundle_id)
    IMMEDIATELY AFTER the evaluator's own identity-pinning call resolves
    it -- i.e. inside the evaluation itself, before verification/loading
    run -- by monkeypatching `checkpoint_module.resolve_checkpoint_identity`
    to perform the replacement as a side effect the first time it is
    called with the mutable `best/` path (the ONE call the fix makes
    against that mutable path; every subsequent internal resolution
    operates on the already-pinned, immutable bundle directory, whose
    name is never "best"). Proves both the report's recorded identity
    AND the actual argument passed to model-loading are bound to the
    ORIGINAL (pre-replacement) bundle throughout, never mixing in the
    replacement bundle that appeared mid-evaluation."""
    from gen3_multiscale.training import checkpoint as checkpoint_module
    from gen3_multiscale.training.train import build_model_for_inference, resolved_config
    import gen3_multiscale.evaluation.gen3_evaluator as evaluator_module

    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    config_path, checkpoint_dir = _train_a_real_checkpoint(tmp_path, cfg, manifest, manifest_path)
    original_identity = checkpoint_module.resolve_checkpoint_identity(checkpoint_dir / "best")

    real_resolve = checkpoint_module.resolve_checkpoint_identity
    replaced = {"done": False}

    def _resolve_and_then_replace_best(path):
        result = real_resolve(path)
        if not replaced["done"] and Path(path).name == "best":
            replaced["done"] = True
            # Simulate a concurrent training job replacing best/ with a
            # genuinely different bundle (freshly re-initialized, never-
            # trained weights -- definitely different content, and a
            # definitely different bundle_id) right after this call.
            config = resolved_config(str(config_path))
            gene_names = list(manifest["gene_panel"])
            fresh_model, _info = build_model_for_inference(
                config, gene_names=gene_names, device=torch.device("cpu"), checkpoint_dir=None,
                smoke=False, dataset_manifest=manifest,
            )
            model_config_path = original_identity.resolved_dir / "model_config.json"
            model_config = json.loads(model_config_path.read_text())
            checkpoint_module.save_checkpoint(
                fresh_model, model_config, gene_names, checkpoint_dir / "best", step=original_identity.step,
            )
        return result

    monkeypatch.setattr(
        evaluator_module.checkpoint_module, "resolve_checkpoint_identity", _resolve_and_then_replace_best,
    )

    captured = {}
    real_load_model = evaluator_module._load_model_for_evaluation

    def _capture_load(config, checkpoint_dir_arg, *args, **kwargs):
        captured["checkpoint_dir_arg"] = Path(checkpoint_dir_arg)
        return real_load_model(config, checkpoint_dir_arg, *args, **kwargs)

    monkeypatch.setattr(evaluator_module, "_load_model_for_evaluation", _capture_load)

    report = evaluator_module.evaluate_gen3_checkpoint(
        str(config_path), checkpoint_dir, split="validation", n_masks_per_sample=2,
    )

    assert replaced["done"] is True  # confirm the mid-evaluation replacement actually ran
    new_identity = checkpoint_module.resolve_checkpoint_identity(checkpoint_dir / "best")
    assert new_identity.bundle_dir != original_identity.bundle_dir
    assert new_identity.weights_sha256 != original_identity.weights_sha256

    assert report["checkpoint_identity"]["resolved_bundle_dir"] == original_identity.bundle_dir
    assert report["checkpoint_identity"]["weights_sha256"] == original_identity.weights_sha256
    assert captured["checkpoint_dir_arg"] == original_identity.resolved_dir


def test_evaluate_gen3_checkpoint_refuses_test_split_by_default(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    config_path, checkpoint_dir = _train_a_real_checkpoint(tmp_path, cfg, manifest, manifest_path)
    with pytest.raises(ValueError, match="Never select using test samples"):
        evaluate_gen3_checkpoint(str(config_path), checkpoint_dir, split="test")


def test_evaluate_gen3_checkpoint_reports_model_and_baseline_metrics_on_validation(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    config_path, checkpoint_dir = _train_a_real_checkpoint(tmp_path, cfg, manifest, manifest_path)

    report = evaluate_gen3_checkpoint(str(config_path), checkpoint_dir, split="validation", n_masks_per_sample=2)
    assert report["split"] == "validation"
    assert report["n_items"] > 0
    arms = report["per_arm_patient_aggregated_metrics"]
    for arm_name in ("model", "mean", "nearest_neighbor", "harmonic"):
        assert arm_name in arms
        assert "pcc" in arms[arm_name]
        assert "n_patients" in arms[arm_name]["pcc"]
        assert "patient_ci95_low" in arms[arm_name]["pcc"]
    assert "secondary_st_fid" not in report  # compute_st_fid_mmd defaults to False
    assert report["input_ablation"] == {
        "zero_image_input": False,
        "image_feature_content": "unmodified",
        "observed_image_available": "unmodified",
        "spatial_geometry": "retained",
    }

    saved_path = save_evaluation_report(report, checkpoint_dir / "evaluation_validation.json")
    assert saved_path.is_file()


def test_evaluate_gen3_checkpoint_records_zero_image_ablation(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    config_path, checkpoint_dir = _train_a_real_checkpoint(tmp_path, cfg, manifest, manifest_path)

    report = evaluate_gen3_checkpoint(
        str(config_path), checkpoint_dir, split="validation", n_masks_per_sample=2,
        zero_image_input=True,
    )

    assert report["input_ablation"] == {
        "zero_image_input": True,
        "image_feature_content": "all_zero",
        "observed_image_available": "all_false",
        "spatial_geometry": "retained",
    }
    assert report["n_items"] > 0


def test_evaluate_gen3_checkpoint_report_is_bound_to_the_exact_checkpoint_identity(tmp_path, monkeypatch):
    """Codex re-audit of commit 66d65f2, finding #1: 'Evaluation reports
    are not bound to the exact checkpoint... records paths, not the
    bundle ID, step, manifest SHA, or weights SHA. If best/ later
    changes, the report no longer proves which weights produced it.'
    Proves the report's `checkpoint_identity` block genuinely identifies
    the exact bundle used -- and that it does NOT silently track whatever
    `best/` happens to point at LATER, by re-saving a DIFFERENT best/
    bundle after the report was already generated and confirming the
    OLD report's recorded identity still matches the OLD bundle, not the
    new one."""
    from gen3_multiscale.training import checkpoint as checkpoint_module

    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    config_path, checkpoint_dir = _train_a_real_checkpoint(tmp_path, cfg, manifest, manifest_path)

    report = evaluate_gen3_checkpoint(str(config_path), checkpoint_dir, split="validation", n_masks_per_sample=2)
    identity_record = report["checkpoint_identity"]
    best_identity = checkpoint_module.resolve_checkpoint_identity(checkpoint_dir / "best")
    assert identity_record["resolved_bundle_dir"] == best_identity.bundle_dir
    assert identity_record["step"] == best_identity.step
    assert identity_record["bundle_manifest_sha256"] == best_identity.manifest_sha256
    assert identity_record["weights_sha256"] == best_identity.weights_sha256
    assert identity_record["use_best"] is True
    assert identity_record["n_masks_per_sample"] == 2
    assert identity_record["code_drift_present"] is False
    assert identity_record["code_drift_acknowledged"] is False
    assert identity_record["config_identity_fingerprint"]
    assert identity_record["dataset_manifest_fingerprint"]
    assert identity_record["gene_panel_hash"]
    assert identity_record["mask_schedule_fingerprint"]

    # Re-save a genuinely DIFFERENT best/ bundle (same model, but a
    # second, distinct save produces a distinct bundle_id/manifest_sha256
    # -- see checkpoint.py's own "repeated save at the same step" test).
    from gen3_multiscale.training.train import build_model_for_inference, resolved_config

    config = resolved_config(str(config_path))
    gene_names = list(manifest["gene_panel"])
    model, _info = build_model_for_inference(
        config, gene_names=gene_names, device=torch.device("cpu"), checkpoint_dir=str(checkpoint_dir),
        smoke=False, dataset_manifest=manifest,
    )
    model_config_path = best_identity.resolved_dir / "model_config.json"
    model_config = json.loads(model_config_path.read_text())
    checkpoint_module.save_checkpoint(
        model, model_config, gene_names, checkpoint_dir / "best", step=best_identity.step,
    )
    new_best_identity = checkpoint_module.resolve_checkpoint_identity(checkpoint_dir / "best")
    assert new_best_identity.bundle_dir != best_identity.bundle_dir

    # The OLD report's recorded identity is untouched -- it still proves
    # exactly which (now-superseded) bundle it was computed from.
    assert identity_record["resolved_bundle_dir"] == best_identity.bundle_dir
    assert identity_record["resolved_bundle_dir"] != new_best_identity.bundle_dir


def test_evaluate_gen3_checkpoint_allow_test_true_permits_test_split(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    config_path, checkpoint_dir = _train_a_real_checkpoint(tmp_path, cfg, manifest, manifest_path)
    report = evaluate_gen3_checkpoint(
        str(config_path), checkpoint_dir, split="test", n_masks_per_sample=2, allow_test=True,
    )
    assert report["split"] == "test"


def _mutate_canonical_run_manifest(checkpoint_dir: Path, **field_updates) -> None:
    """Same pattern as test_train.py's own helper of the same name (kept
    local here to avoid importing across test modules): mutate the
    CANONICAL, bundle-bound run_manifest.json (re-signing the bundle's
    own manifest.json file-hash entry and the pointer's manifest_sha256)
    and mirror the identical content to checkpoint_dir's root
    run_manifest.json, simulating "this field's value was wrong when
    originally saved" without tripping the separate root-vs-canonical
    consistency check."""
    import hashlib

    from gen3_multiscale.training import checkpoint as checkpoint_module

    checkpoint_dir = Path(checkpoint_dir)
    bundle_dir = checkpoint_module.resolve_checkpoint_identity(checkpoint_dir).resolved_dir
    run_manifest = json.loads((bundle_dir / "run_manifest.json").read_text())
    run_manifest.update(field_updates)
    new_content = json.dumps(run_manifest, indent=2, sort_keys=True, default=str).encode()

    (bundle_dir / "run_manifest.json").write_bytes(new_content)
    manifest_path = bundle_dir / "manifest.json"
    bundle_manifest = json.loads(manifest_path.read_text())
    bundle_manifest["files"]["run_manifest.json"] = hashlib.sha256(new_content).hexdigest()
    manifest_path.write_text(json.dumps(bundle_manifest, indent=2, sort_keys=True))
    new_manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    pointer_path = checkpoint_dir / "latest_bundle.json"
    pointer = json.loads(pointer_path.read_text())
    pointer["manifest_sha256"] = new_manifest_sha256
    pointer_path.write_text(json.dumps(pointer, indent=2))

    (checkpoint_dir / "run_manifest.json").write_bytes(new_content)


# ---------------------------------------------------------------------------
# Codex re-audit of commit 2162ff4, finding #4: "'Full checkpoint identity'
# is not actually full" -- code commit/worktree hashes not compared,
# missing recorded cache identity for a validation sample silently
# skipped instead of rejected, model_config vs run_manifest not
# cross-checked.
# ---------------------------------------------------------------------------

def test_evaluate_gen3_checkpoint_refuses_when_code_state_drifted_unless_allowed(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    config_path, checkpoint_dir = _train_a_real_checkpoint(tmp_path, cfg, manifest, manifest_path)

    # `evaluate_gen3_checkpoint` defaults to use_best=True, which resolves
    # through checkpoint_dir/"best" -- a SEPARATE checkpoint bundle from
    # checkpoint_dir's own live/latest one -- so the mutation must target
    # THAT bundle to actually affect what gets verified/loaded.
    best_dir = checkpoint_dir / "best"
    persisted = json.loads((best_dir / "run_manifest.json").read_text())
    assert persisted["code_commit_hash"] is not None  # this repo IS a real git checkout
    _mutate_canonical_run_manifest(best_dir, code_commit_hash="deadbeef" * 5)

    with pytest.raises(ValueError, match="code state changed"):
        evaluate_gen3_checkpoint(str(config_path), checkpoint_dir, split="validation", n_masks_per_sample=2)

    # The explicit override still evaluates successfully.
    report = evaluate_gen3_checkpoint(
        str(config_path), checkpoint_dir, split="validation", n_masks_per_sample=2, allow_code_drift=True,
    )
    assert report["n_items"] > 0


def test_evaluate_gen3_checkpoint_cross_checks_bundled_model_config_against_its_own_run_manifest(tmp_path, monkeypatch):
    """Codex re-audit of commit 2162ff4, finding #4(c): the resolved
    bundle's own model_config.json, independently re-fingerprinted, must
    agree with that SAME bundle's run_manifest.json recorded
    config_identity_fingerprint -- catches a hypothetical "value was
    wrong when originally saved" bug where the two disagree with EACH
    OTHER, which the per-file sha256 checks (which only verify internal
    self-consistency, never cross-field consistency) cannot detect."""
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    config_path, checkpoint_dir = _train_a_real_checkpoint(tmp_path, cfg, manifest, manifest_path)

    import hashlib

    from gen3_multiscale.training import checkpoint as checkpoint_module

    best_dir = checkpoint_dir / "best"  # evaluate_gen3_checkpoint defaults to use_best=True
    bundle_dir = checkpoint_module.resolve_checkpoint_identity(best_dir).resolved_dir
    model_config_path = bundle_dir / "model_config.json"
    model_config = json.loads(model_config_path.read_text())
    model_config["params"] = dict(model_config.get("params") or {}, hidden_dim=999999)  # tamper the bundled config
    new_content = json.dumps(model_config, indent=2, default=str).encode()
    model_config_path.write_bytes(new_content)

    manifest_path_json = bundle_dir / "manifest.json"
    bundle_manifest = json.loads(manifest_path_json.read_text())
    bundle_manifest["files"]["model_config.json"] = hashlib.sha256(new_content).hexdigest()
    manifest_path_json.write_text(json.dumps(bundle_manifest, indent=2, sort_keys=True))
    new_manifest_sha256 = hashlib.sha256(manifest_path_json.read_bytes()).hexdigest()
    pointer_path = best_dir / "latest_bundle.json"
    pointer = json.loads(pointer_path.read_text())
    pointer["manifest_sha256"] = new_manifest_sha256
    pointer_path.write_text(json.dumps(pointer, indent=2))

    with pytest.raises(ValueError, match="disagree with each other"):
        evaluate_gen3_checkpoint(str(config_path), checkpoint_dir, split="validation", n_masks_per_sample=2)


def test_evaluate_gen3_checkpoint_fails_closed_on_missing_recorded_cache_identity_for_a_known_scope_sample(tmp_path, monkeypatch):
    """Codex re-audit of commit 2162ff4, finding #4(b): a sample_id
    within the checkpoint's OWN training-time preflight scope (provably
    so, since dataset_manifest_fingerprint is already verified equal)
    that has NO recorded cache content identity must fail closed, never
    be silently skipped -- skipping is only correct for a sample outside
    that scope entirely (e.g. a held-out test sample)."""
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    config_path, checkpoint_dir = _train_a_real_checkpoint(tmp_path, cfg, manifest, manifest_path)

    best_dir = checkpoint_dir / "best"  # evaluate_gen3_checkpoint defaults to use_best=True
    validation_sample_id = manifest["validation_sample_ids"][0]
    bundle_dir_before = json.loads((best_dir / "run_manifest.json").read_text())
    cache_preflight_report = dict(bundle_dir_before["cache_preflight_report"])
    cache_content_by_sample = dict(cache_preflight_report["cache_content_by_sample"])
    assert validation_sample_id in cache_content_by_sample
    del cache_content_by_sample[validation_sample_id]  # a "known-scope" sample with no recorded cache identity
    cache_preflight_report["cache_content_by_sample"] = cache_content_by_sample
    _mutate_canonical_run_manifest(best_dir, cache_preflight_report=cache_preflight_report)

    with pytest.raises(ValueError, match="no recorded cache"):
        evaluate_gen3_checkpoint(str(config_path), checkpoint_dir, split="validation", n_masks_per_sample=2)


def test_gen3_evaluator_cli_actually_evaluates_and_writes_a_report(tmp_path, monkeypatch):
    """Codex re-audit of commit 2162ff4, finding #1: "The documented
    evaluator command does nothing... running `python -m
    gen3_multiscale.evaluation.gen3_evaluator ...` silently exits without
    evaluating or saving a report." Confirmed real: the module had no
    `main()`/`__main__` at all. A REAL subprocess (not a monkeypatched or
    in-process call) is the only way to prove the documented command
    actually works -- an in-process call to `main()` could still pass
    even if `if __name__ == "__main__"` were wired wrong or `python -m`
    resolution were broken. The checkpoint/manifest/cache are all built
    beforehand in-process (the standard fixture flow, using the
    monkeypatched GigaPath stub); the spawned subprocess never needs that
    stub itself, since `load_gen3_spot_features` only ever reads the
    already-written, on-disk `.npz` cache -- it never re-invokes the tile
    encoder for a sample whose cache already exists."""
    import subprocess
    import sys

    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    config_path, checkpoint_dir = _train_a_real_checkpoint(tmp_path, cfg, manifest, manifest_path)
    output_path = tmp_path / "cli_evaluation_report.json"
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable, "-m", "gen3_multiscale.evaluation.gen3_evaluator",
            "--config", str(config_path), "--checkpoint-dir", str(checkpoint_dir),
            "--split", "validation", "--n-masks-per-sample", "2", "--output", str(output_path),
        ],
        cwd=repo_root, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "evaluation report saved to" in result.stdout
    assert "model pcc" in result.stdout

    assert output_path.is_file()
    written_report = json.loads(output_path.read_text())
    assert written_report["split"] == "validation"
    assert written_report["n_items"] > 0
    assert "model" in written_report["per_arm_patient_aggregated_metrics"]


def test_gen3_evaluator_cli_exits_nonzero_and_reports_the_error_on_failure(tmp_path, monkeypatch):
    """The same real subprocess, but asking for `--split test` without
    `--allow-test` -- must exit non-zero and print the real error to
    stderr, never silently succeed or exit 0 with no report written."""
    import subprocess
    import sys

    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    config_path, checkpoint_dir = _train_a_real_checkpoint(tmp_path, cfg, manifest, manifest_path)
    output_path = tmp_path / "cli_evaluation_report_failure.json"
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable, "-m", "gen3_multiscale.evaluation.gen3_evaluator",
            "--config", str(config_path), "--checkpoint-dir", str(checkpoint_dir),
            "--split", "test", "--n-masks-per-sample", "2", "--output", str(output_path),
        ],
        cwd=repo_root, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode != 0
    assert "Never select using test samples" in result.stderr
    assert not output_path.exists()


def test_evaluate_gen3_checkpoint_computes_secondary_st_fid_mmd_only_when_requested(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = prepare_step6_experiment(
        tmp_path, monkeypatch, samples_per_split={"train": 3, "validation": 2, "test": 1},
    )
    config_path, checkpoint_dir = _train_a_real_checkpoint(tmp_path, cfg, manifest, manifest_path)
    report = evaluate_gen3_checkpoint(
        str(config_path), checkpoint_dir, split="validation", n_masks_per_sample=3, compute_st_fid_mmd=True,
    )
    assert "secondary_st_fid" in report
    assert "secondary_st_mmd" in report
    assert np.isfinite(report["secondary_st_fid"])


# ---------------------------------------------------------------------------
# Codex re-audit of commit 90f853e, launch blocker #9: evaluator
# completeness (named panels, per-stratum results, per-item query
# fingerprints, Architecture 4 calibration).
# ---------------------------------------------------------------------------

def test_load_configured_gene_panels_reads_the_genes_key_from_each_json_file(tmp_path):
    panel_path = tmp_path / "my_panel.json"
    panel_path.write_text(json.dumps({"panel_name": "my_panel", "genes": ["GENE0", "GENE1", "GENE0"]}))
    panels = load_configured_gene_panels({"evaluation": {"gene_panels": {"my_panel": str(panel_path)}}})
    assert panels == {"my_panel": ["GENE0", "GENE1"]}


def test_load_configured_gene_panels_returns_empty_when_unconfigured():
    assert load_configured_gene_panels({}) == {}
    assert load_configured_gene_panels({"evaluation": {}}) == {}


def test_load_configured_gene_panels_requires_manifest_for_train_derived_artifact(tmp_path):
    with pytest.raises(ValueError, match="no dataset manifest"):
        load_configured_gene_panels(
            {"evaluation": {"train_gene_panel_artifact": str(tmp_path / "panels.json")}}
        )


def test_load_configured_gene_panels_raises_on_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        load_configured_gene_panels(
            {"evaluation": {"gene_panels": {"my_panel": str(tmp_path / "does_not_exist.json")}}}
        )


def test_evaluate_gen3_checkpoint_reports_configured_named_gene_panels(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    config_path, checkpoint_dir = _train_a_real_checkpoint(tmp_path, cfg, manifest, manifest_path)

    panel_path = tmp_path / "two_gene_panel.json"
    panel_path.write_text(json.dumps({"genes": ["GENE0", "GENE1", "not_a_real_gene"]}))
    config = yaml.safe_load(config_path.read_text())
    config["evaluation"] = {"gene_panels": {"two_gene_panel": str(panel_path)}}
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    report = evaluate_gen3_checkpoint(str(config_path), checkpoint_dir, split="validation", n_masks_per_sample=2)

    assert report["gene_panel_metadata"]["two_gene_panel"]["evaluated_count"] == 2
    assert report["gene_panel_metadata"]["two_gene_panel"]["missing_genes"] == ["not_a_real_gene"]
    # Secondary fix #3: named-panel metrics for EVERY arm (model AND every
    # baseline), keyed [panel][arm], plus per-panel paired deltas. Only
    # "model" is asserted to cover every item's pcc as a finite value --
    # "mean" broadcasts one constant vector to every query spot, which
    # makes its per-gene pcc structurally undefined (zero prediction
    # variance) whenever a query has more than one spot, a pre-existing
    # property of that baseline, not something this round changes.
    panel_by_arm = report["per_panel_patient_aggregated_metrics"]["two_gene_panel"]
    for arm_name in ("model", "mean", "nearest_neighbor", "harmonic"):
        assert arm_name in panel_by_arm
        assert "pcc" in panel_by_arm[arm_name] and "rmse" in panel_by_arm[arm_name]
    assert panel_by_arm["model"]["pcc"]["n_items"] == report["n_items"]
    panel_deltas = report["per_panel_paired_delta_vs_model"]["two_gene_panel"]
    for baseline_name in ("mean", "nearest_neighbor", "harmonic"):
        assert baseline_name in panel_deltas
        assert "pcc_delta" in panel_deltas[baseline_name]
    assert "gene_panels" in report["per_item_records"][0]
    assert "two_gene_panel" in report["per_item_records"][0]["gene_panels"]
    assert "two_gene_panel" in report["per_item_records"][0]["baseline_gene_panels"]["mean"]


def test_evaluate_gen3_checkpoint_reports_per_stratum_results(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    config_path, checkpoint_dir = _train_a_real_checkpoint(tmp_path, cfg, manifest, manifest_path)
    report = evaluate_gen3_checkpoint(str(config_path), checkpoint_dir, split="validation", n_masks_per_sample=2)

    per_stratum = report["per_stratum_patient_aggregated_metrics"]
    # The fixture's masking config declares exactly one stratum ("small");
    # every item must be attributed to it, and its aggregated item count
    # must equal the whole report's.
    assert set(per_stratum.keys()) == {"small"}
    assert per_stratum["small"]["pcc"]["n_items"] == report["n_items"]


def test_evaluate_gen3_checkpoint_per_item_query_fingerprint_is_stable_and_real(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    config_path, checkpoint_dir = _train_a_real_checkpoint(tmp_path, cfg, manifest, manifest_path)
    report_a = evaluate_gen3_checkpoint(str(config_path), checkpoint_dir, split="validation", n_masks_per_sample=2)
    report_b = evaluate_gen3_checkpoint(str(config_path), checkpoint_dir, split="validation", n_masks_per_sample=2)

    fingerprints_a = [record["query_fingerprint"] for record in report_a["per_item_records"]]
    fingerprints_b = [record["query_fingerprint"] for record in report_b["per_item_records"]]
    assert fingerprints_a == fingerprints_b  # deterministic, FIXED held-out schedule
    assert len(set(fingerprints_a)) == len(fingerprints_a)  # every item's mask is distinct
    for fp in fingerprints_a:
        assert isinstance(fp, str) and len(fp) == 64


def test_evaluate_gen3_checkpoint_reports_empty_architecture4_calibration_for_architecture1(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    config_path, checkpoint_dir = _train_a_real_checkpoint(tmp_path, cfg, manifest, manifest_path)
    report = evaluate_gen3_checkpoint(str(config_path), checkpoint_dir, split="validation", n_masks_per_sample=2)
    assert report["architecture4_calibration"] == {"n_values": 0}
