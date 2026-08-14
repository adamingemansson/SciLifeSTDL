import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gen3_multiscale.evaluation.frozen_feature_ridge_evaluator import (
    _CovariancePCA,
    _candidate_indices,
    _fit_covariance_pca,
    _positive_gene_scale,
    _save_fit_artifact,
    _solve_ridge,
    _stable_indices,
)
from gen3_multiscale.evaluation.frozen_feature_mlp_evaluator import FrozenFeatureMLP
from gen3_multiscale.evaluation.frozen_feature_ridge_artifact_evaluator import (
    _load_and_validate_fit,
)
from gen3_multiscale.scripts.summarize_hest_mk_benchmarks import rows_from_report
from gen3_multiscale.scripts.compare_mk_whole_slide_to_ridge import (
    _compatible_mk,
    _contract_from_ridge,
    _paired_patient_delta,
)


def test_stable_indices_are_reproducible_bounded_and_sample_specific():
    first = _stable_indices("A", 100, 12, 3)
    assert np.array_equal(first, _stable_indices("A", 100, 12, 3))
    assert not np.array_equal(first, _stable_indices("B", 100, 12, 3))
    assert len(first) == 12
    assert np.all(np.diff(first) > 0)


def test_frozen_feature_mlp_maps_pca_rows_to_the_full_gene_panel():
    model = FrozenFeatureMLP(8, 12, 17, dropout=0.1)
    output = model(torch.randn(5, 8))
    assert output.shape == (5, 17)
    assert sum(parameter.numel() for parameter in model.parameters()) == 329


def test_candidate_indices_match_zero_placeholder_contract():
    class Sample:
        class Adata:
            n_obs = 4

        adata = Adata()
        image_source_available = np.asarray([True, False, True, False])

    assert _candidate_indices(Sample(), "zero").tolist() == [0, 1, 2, 3]
    assert _candidate_indices(Sample(), "exclude").tolist() == [0, 2]


def test_covariance_pca_and_torch_ridge_avoid_numpy_lapack():
    rng = np.random.default_rng(4)
    features = rng.normal(size=(40, 6))
    pca = _fit_covariance_pca(
        features.sum(axis=0), features.T @ features,
        len(features), n_components=3, device="cpu",
    )
    projected = pca.transform(features)
    assert projected.shape == (40, 3)
    assert np.allclose(projected.mean(axis=0), 0.0, atol=1e-6)

    x = np.column_stack([projected, np.ones(len(projected))])
    truth = rng.normal(size=(40, 5))
    penalty = np.eye(x.shape[1])
    penalty[-1, -1] = 0.0
    actual = _solve_ridge(x.T @ x, x.T @ truth, penalty, "cpu")
    expected = np.linalg.solve(x.T @ x + penalty, x.T @ truth)
    assert np.allclose(actual, expected)


def test_gene_scale_floors_only_constant_genes_and_fit_is_saved(tmp_path: Path):
    target = np.asarray([
        [1.0, 2.0, 4.0],
        [1.0, 4.0, 4.0],
        [1.0, 6.0, 4.0],
    ])
    scale, n_floored = _positive_gene_scale(
        target.sum(axis=0), np.square(target).sum(axis=0), len(target),
    )
    assert n_floored == 2
    assert scale[0] == np.float32(1e-6)
    assert scale[1] > 0
    assert scale[2] == np.float32(1e-6)

    artifact = tmp_path / "ridge.model.npz"
    pca = _CovariancePCA(np.zeros(2), np.eye(2))
    _save_fit_artifact(
        artifact,
        pca=pca,
        coefficients=np.ones((3, 3)),
        per_gene_scale=scale,
        gene_names=["g0", "g1", "g2"],
    )
    with np.load(artifact, allow_pickle=False) as payload:
        assert payload["coefficients"].shape == (3, 3)
        assert np.all(payload["per_gene_scale"] > 0)
        assert payload["gene_names"].tolist() == ["g0", "g1", "g2"]


def test_saved_ridge_fit_reuse_is_fail_closed(tmp_path: Path):
    artifact = tmp_path / "ridge.model.npz"
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    _save_fit_artifact(
        artifact,
        pca=_CovariancePCA(np.zeros(3), np.eye(2, 3)),
        coefficients=np.ones((3, 2)),
        per_gene_scale=np.ones(2),
        gene_names=["g0", "g1"],
    )
    report = {
        "kind": "frozen_feature_pca_ridge_whole_slide_benchmark",
        "image_encoder": "uni2",
        "missing_image_policy": "zero",
        "manifest_path": str(manifest),
        "model_artifact": str(artifact),
        "pca_components": 2,
    }
    pca, coefficients, scale = _load_and_validate_fit(
        artifact, report, gene_names=["g0", "g1"], manifest_path=manifest,
        image_encoder="uni2", missing_image_policy="zero",
    )
    assert pca.components_.shape == (2, 3)
    assert coefficients.shape == (3, 2)
    assert scale.tolist() == [1.0, 1.0]
    with pytest.raises(ValueError, match="gene panel/order"):
        _load_and_validate_fit(
            artifact, report, gene_names=["g1", "g0"], manifest_path=manifest,
            image_encoder="uni2", missing_image_policy="zero",
        )


def test_summary_parser_emits_patient_macro_ridge_row(tmp_path: Path):
    report = {
        "kind": "frozen_feature_pca_ridge_whole_slide_benchmark",
        "image_encoder": "uni2",
        "pca_components": 256,
        "point_metrics_patient_aggregated": {
            "all_genes": {
                "pcc": {"patient_mean": 0.12, "n_patients": 9},
                "mean_spot_profile_pcc": {"patient_mean": 0.72, "n_patients": 9},
                "mean_gene_nmi": {"patient_mean": 0.21, "n_patients": 9},
                "mean_gene_js_divergence": {"patient_mean": 0.18, "n_patients": 9},
                "rmse": {"patient_mean": 0.34, "n_patients": 9},
                "median_gene_pcc": {"patient_mean": 0.08, "n_patients": 9},
                "fraction_gene_pcc_gt_0_3": {"patient_mean": 0.14, "n_patients": 9},
            },
        },
        "structured_metrics_patient_aggregated": {
            "all_genes": {
                "spatial_ssim.mean_per_gene_ssim": {
                    "patient_mean": 0.44, "n_patients": 9,
                },
                "moran_local.moran_i_pcc": {
                    "patient_mean": 0.55, "n_patients": 9,
                },
            },
        },
    }
    path = tmp_path / "ridge.json"
    path.write_text(json.dumps(report))
    rows = rows_from_report(path, report)
    assert len(rows) == 1
    assert rows[0]["method"] == "uni2_pca256_ridge"
    assert rows[0]["scope"] == "whole_slide"
    assert rows[0]["pcc"] == 0.12
    assert rows[0]["mean_spot_profile_pcc"] == 0.72
    assert rows[0]["mean_gene_nmi"] == 0.21
    assert rows[0]["mean_gene_js_divergence"] == 0.18
    assert rows[0]["median_gene_pcc"] == 0.08
    assert rows[0]["fraction_gene_pcc_gt_0_3"] == 0.14
    assert rows[0]["mean_per_gene_ssim"] == 0.44
    assert rows[0]["moran_i_pcc"] == 0.55
    assert rows[0]["n_patients"] == 9


def test_exact_whole_slide_comparator_rejects_different_spot_scope():
    ridge = {
        "kind": "frozen_feature_pca_ridge_whole_slide_benchmark",
        "split": "validation",
        "missing_image_policy": "zero",
        "n_validation_samples": 14,
        "per_slide_records": [
            {"sample_id": f"s{i}", "n_evaluated_spots": 100 + i}
            for i in range(14)
        ],
        "point_metrics_patient_aggregated": {"all_genes": {"pcc": {}}},
    }
    counts, _, contract = _contract_from_ridge(ridge)
    assert contract is None
    report = {
        "kind": "conditional_wae_supervisor_evaluation",
        "split": "validation",
        "query_gex_visible": False,
        "query_he_visible": True,
        "whole_slide_structured_field_evaluation": {
            "scope": "all_held_out_slides_every_spot_exactly_once",
            "primary_prediction": "deterministic_h_and_e_point_prediction",
            "target_gex_visible_to_model": False,
            "point_metrics_patient_aggregated": {"all_genes": {"pcc": {}}},
            "per_slide_records": [
                {"sample_id": f"s{i}", "n_spots": 100 + i}
                for i in range(14)
            ],
        },
    }
    assert _compatible_mk(report, counts) == (True, "compatible")
    report["whole_slide_structured_field_evaluation"]["per_slide_records"][0]["n_spots"] = 99
    assert _compatible_mk(report, counts) == (False, "cohort_or_spot_count_mismatch")


def test_paired_delta_bootstraps_patients_after_averaging_their_slides():
    reference = [
        {"sample_id": "s0", "patient_id": "p0", "point_metrics": {"all_genes": {"pcc": 0.1}}},
        {"sample_id": "s1", "patient_id": "p0", "point_metrics": {"all_genes": {"pcc": 0.3}}},
        {"sample_id": "s2", "patient_id": "p1", "point_metrics": {"all_genes": {"pcc": 0.4}}},
    ]
    candidate = [
        {"sample_id": "s0", "patient_id": "p0", "point_metrics": {"all_genes": {"pcc": 0.2}}},
        {"sample_id": "s1", "patient_id": "p0", "point_metrics": {"all_genes": {"pcc": 0.6}}},
        {"sample_id": "s2", "patient_id": "p1", "point_metrics": {"all_genes": {"pcc": 0.5}}},
    ]
    result = _paired_patient_delta(
        reference, candidate, panel="all_genes", metric="pcc",
        n_bootstrap=1000, seed=2,
    )
    # p0 mean delta=(.1+.3)/2=.2; p1=.1; patient macro=.15.
    assert result["patient_mean"] == pytest.approx(0.15)
    assert result["n_patients"] == 2
    assert result["ci95_low"] <= result["patient_mean"] <= result["ci95_high"]
