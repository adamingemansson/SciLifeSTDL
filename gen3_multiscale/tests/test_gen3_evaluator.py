"""Tests for gen3_multiscale/evaluation/gen3_evaluator.py -- Adam's Step
6 audit #12 deliverable: the minimal real Step 7 evaluator. Real, small,
end-to-end synthetic data throughout."""
from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from gen3_multiscale.evaluation.gen3_evaluator import (
    evaluate_gen3_checkpoint, harmonic_baseline_prediction, load_configured_gene_panels,
    mean_baseline_prediction, nearest_neighbor_baseline_prediction, per_item_reconstruction_metrics,
    save_evaluation_report,
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

    saved_path = save_evaluation_report(report, checkpoint_dir / "evaluation_validation.json")
    assert saved_path.is_file()


def test_evaluate_gen3_checkpoint_allow_test_true_permits_test_split(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    config_path, checkpoint_dir = _train_a_real_checkpoint(tmp_path, cfg, manifest, manifest_path)
    report = evaluate_gen3_checkpoint(
        str(config_path), checkpoint_dir, split="test", n_masks_per_sample=2, allow_test=True,
    )
    assert report["split"] == "test"


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
    panel_metrics = report["per_panel_patient_aggregated_metrics"]["two_gene_panel"]
    assert "pcc" in panel_metrics and "rmse" in panel_metrics
    assert panel_metrics["pcc"]["n_items"] == report["n_items"]
    assert "gene_panels" in report["per_item_records"][0]
    assert "two_gene_panel" in report["per_item_records"][0]["gene_panels"]


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
