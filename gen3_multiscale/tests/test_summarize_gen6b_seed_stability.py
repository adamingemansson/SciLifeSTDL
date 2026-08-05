import json
from pathlib import Path

import pytest

from gen3_multiscale.scripts.summarize_gen6b_seed_stability import summarize_gen6b_seed_stability


def _metric_block(patient_mean: float) -> dict:
    return {"patient_mean": patient_mean, "pooled_mean": patient_mean, "n_patients": 8, "n_items": 288}


def _fake_report(*, pcc: float, rmse: float, auc: float, n_items: int = 288, dataset_fp: str = "fp-shared") -> dict:
    return {
        "n_items": n_items,
        "checkpoint_identity": {"dataset_manifest_fingerprint": dataset_fp},
        "per_arm_patient_aggregated_metrics": {
            "model": {"pcc": _metric_block(pcc), "rmse": _metric_block(rmse), "nonzero_auc": _metric_block(auc)},
        },
        "per_panel_patient_aggregated_metrics": {
            "hvg_50": {"model": {"pcc": _metric_block(pcc * 3), "rmse": _metric_block(rmse * 0.9), "nonzero_auc": _metric_block(auc)}},
            "hvg_200": {"model": {"pcc": _metric_block(pcc * 2.5), "rmse": _metric_block(rmse * 0.95), "nonzero_auc": _metric_block(auc)}},
            "CCRCC_var_50genes": {"model": {"pcc": _metric_block(pcc * 1.2), "rmse": _metric_block(rmse), "nonzero_auc": _metric_block(auc)}},
        },
        "per_arm_paired_delta_vs_model": {
            "mean": {"pcc_delta": _metric_block(0.01), "rmse_delta": _metric_block(0.005)},
            "nearest_neighbor": {"pcc_delta": _metric_block(0.02), "rmse_delta": _metric_block(0.006)},
            "harmonic": {"pcc_delta": _metric_block(0.015), "rmse_delta": _metric_block(0.004)},
        },
    }


def _write(tmp_path: Path, name: str, report: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(report))
    return str(path)


def test_aggregates_mean_and_sd_across_three_seeds(tmp_path):
    reports = {
        "0": _write(tmp_path, "seed0.json", _fake_report(pcc=0.040, rmse=0.320, auc=0.55)),
        "1": _write(tmp_path, "seed1.json", _fake_report(pcc=0.046, rmse=0.315, auc=0.56)),
        "2": _write(tmp_path, "seed2.json", _fake_report(pcc=0.050, rmse=0.310, auc=0.57)),
    }
    summary = summarize_gen6b_seed_stability(evaluation_reports=reports)
    assert summary["seeds"] == ["0", "1", "2"]
    assert summary["n_items"] == 288
    pcc_block = summary["all_gene_patient_mean_across_seeds"]["pcc"]
    assert pcc_block["n"] == 3
    assert pcc_block["mean"] == pytest.approx((0.040 + 0.046 + 0.050) / 3)
    assert pcc_block["sd"] > 0
    assert pcc_block["values"] == [0.040, 0.046, 0.050]

    hvg50 = summary["per_panel_patient_mean_across_seeds"]["hvg_50"]["pcc"]
    assert hvg50["mean"] == pytest.approx(pcc_block["mean"] * 3)

    delta = summary["paired_delta_patient_mean_across_seeds"]["nearest_neighbor"]["pcc"]
    assert delta["mean"] == pytest.approx(0.02)


def test_refuses_mismatched_n_items(tmp_path):
    reports = {
        "0": _write(tmp_path, "seed0.json", _fake_report(pcc=0.04, rmse=0.32, auc=0.55, n_items=288)),
        "1": _write(tmp_path, "seed1.json", _fake_report(pcc=0.04, rmse=0.32, auc=0.55, n_items=200)),
    }
    with pytest.raises(ValueError, match="different n_items"):
        summarize_gen6b_seed_stability(evaluation_reports=reports)


def test_refuses_mismatched_dataset_fingerprint(tmp_path):
    reports = {
        "0": _write(tmp_path, "seed0.json", _fake_report(pcc=0.04, rmse=0.32, auc=0.55, dataset_fp="fp-a")),
        "1": _write(tmp_path, "seed1.json", _fake_report(pcc=0.04, rmse=0.32, auc=0.55, dataset_fp="fp-b")),
    }
    with pytest.raises(ValueError, match="dataset_manifest_fingerprint"):
        summarize_gen6b_seed_stability(evaluation_reports=reports)


def test_refuses_fewer_than_two_seeds(tmp_path):
    reports = {"0": _write(tmp_path, "seed0.json", _fake_report(pcc=0.04, rmse=0.32, auc=0.55))}
    with pytest.raises(ValueError, match="at least two"):
        summarize_gen6b_seed_stability(evaluation_reports=reports)


def test_reference_arm_spread_comparison(tmp_path):
    reports = {
        "0": _write(tmp_path, "seed0.json", _fake_report(pcc=0.040, rmse=0.320, auc=0.55)),
        "1": _write(tmp_path, "seed1.json", _fake_report(pcc=0.070, rmse=0.310, auc=0.57)),  # wide seed spread
    }
    summary = summarize_gen6b_seed_stability(
        evaluation_reports=reports,
        reference_arm_pcc={"gen6a": 0.0587, "gen6b": 0.045384, "gen6c": 0.044},
    )
    comparison = summary["seed_spread_vs_gen6_b_through_j_reference"]
    assert comparison["reference_pcc_spread"] == pytest.approx(0.0587 - 0.044)
    assert comparison["seed_pcc_spread"] == pytest.approx(0.070 - 0.040)
    assert comparison["seed_spread_exceeds_reference_spread"] is True


def test_run_context_reads_validation_history_when_checkpoint_dirs_given(tmp_path):
    ckpt0 = tmp_path / "ckpt0"
    ckpt0.mkdir()
    (ckpt0 / "validation_history.json").write_text(json.dumps([
        {"step": 100, "total": 0.5}, {"step": 200, "total": 0.4}, {"step": 300, "total": 0.45},
    ]))
    reports = {
        "0": _write(tmp_path, "seed0.json", _fake_report(pcc=0.04, rmse=0.32, auc=0.55)),
        "1": _write(tmp_path, "seed1.json", _fake_report(pcc=0.05, rmse=0.31, auc=0.56)),
    }
    summary = summarize_gen6b_seed_stability(
        evaluation_reports=reports, checkpoint_dirs={"0": str(ckpt0)},
    )
    assert summary["run_context"]["0"]["best_step"] == 200
    assert summary["run_context"]["0"]["best_validation_total"] == pytest.approx(0.4)
    assert summary["run_context"]["0"]["final_step"] == 300
    assert "best_step" not in summary["run_context"]["1"] or summary["run_context"]["1"].get("best_step") is None
