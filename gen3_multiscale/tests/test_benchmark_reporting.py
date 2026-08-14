import json
from pathlib import Path

import numpy as np

from gen3_multiscale.evaluation.frozen_feature_ridge_evaluator import (
    _CovariancePCA,
    _candidate_indices,
    _fit_covariance_pca,
    _positive_gene_scale,
    _save_fit_artifact,
    _solve_ridge,
    _stable_indices,
)
from gen3_multiscale.scripts.summarize_hest_mk_benchmarks import rows_from_report


def test_stable_indices_are_reproducible_bounded_and_sample_specific():
    first = _stable_indices("A", 100, 12, 3)
    assert np.array_equal(first, _stable_indices("A", 100, 12, 3))
    assert not np.array_equal(first, _stable_indices("B", 100, 12, 3))
    assert len(first) == 12
    assert np.all(np.diff(first) > 0)


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


def test_summary_parser_emits_patient_macro_ridge_row(tmp_path: Path):
    report = {
        "kind": "frozen_feature_pca_ridge_whole_slide_benchmark",
        "image_encoder": "uni2",
        "pca_components": 256,
        "point_metrics_patient_aggregated": {
            "all_genes": {
                "pcc": {"patient_mean": 0.12, "n_patients": 9},
                "rmse": {"patient_mean": 0.34, "n_patients": 9},
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
    assert rows[0]["n_patients"] == 9
