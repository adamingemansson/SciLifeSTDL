import numpy as np
import pytest

from gen3_multiscale.evaluation.structured_field_metrics import (
    coexpression_agreement,
    moran_i_agreement,
    noise_ceiling_adjusted_pcc,
    signed_gradient_agreement,
    spot_profile_agreement,
    structured_field_metrics,
    undirected_knn_edges,
)
from gen3_multiscale.evaluation.conditional_wae_evaluator import (
    _aggregate_whole_slide_structured_reports,
)


def _field(seed=0, side=8, genes=6):
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.arange(side), np.arange(side))
    coords = np.stack([x.ravel(), y.ravel()], axis=1).astype(np.float64)
    values = []
    for gene in range(genes):
        values.append(
            np.sin((gene + 1) * coords[:, 0] / side)
            + np.cos((gene + 2) * coords[:, 1] / side)
            + 0.03 * rng.normal(size=len(coords))
        )
    target = np.stack(values, axis=1).astype(np.float32)
    return coords, target


def test_perfect_field_scores_as_perfect_without_nan_collapse():
    coords, target = _field()
    scale = target.std(axis=0)
    report = structured_field_metrics(
        target.copy(), target, coords, scale,
        panel_indices={"panel": np.arange(4)}, local_k=4, wide_k=10,
    )
    for name in ("all_genes", "panel"):
        panel = report["panels"][name]
        assert panel["spot_profile"]["mean_spot_profile_pcc"] == pytest.approx(1.0)
        assert panel["moran_local"]["moran_i_pcc"] == pytest.approx(1.0)
        assert panel["gradient_local"]["signed_gradient_pcc"] == pytest.approx(1.0)
        assert panel["gradient_local"]["gradient_energy_ratio"] == pytest.approx(1.0)
    assert report["panels"]["panel"]["coexpression"]["correlation_matrix_pcc"] == pytest.approx(1.0)


def test_spatial_permutation_preserves_values_but_breaks_field_metrics():
    coords, target = _field(seed=2)
    rng = np.random.default_rng(8)
    predicted = target[rng.permutation(len(target))]
    perfect_moran = moran_i_agreement(target, target, coords, k_neighbors=4)
    shuffled_moran = moran_i_agreement(predicted, target, coords, k_neighbors=4)
    perfect_gradient = signed_gradient_agreement(
        target, target, coords, target.std(axis=0), k_neighbors=4,
    )
    shuffled_gradient = signed_gradient_agreement(
        predicted, target, coords, target.std(axis=0), k_neighbors=4,
    )
    assert shuffled_moran["moran_i_mae"] > perfect_moran["moran_i_mae"] + 0.1
    assert shuffled_gradient["signed_gradient_pcc"] < perfect_gradient["signed_gradient_pcc"] - 0.5


def test_flat_prediction_is_penalized_not_dropped():
    coords, target = _field(seed=3)
    predicted = np.broadcast_to(target.mean(axis=0), target.shape).copy()
    moran = moran_i_agreement(predicted, target, coords, k_neighbors=4)
    gradient = signed_gradient_agreement(
        predicted, target, coords, target.std(axis=0), k_neighbors=4,
    )
    assert moran["n_eligible_genes"] == target.shape[1]
    assert moran["moran_i_pcc"] == pytest.approx(0.0)
    assert gradient["signed_gradient_pcc"] == pytest.approx(0.0)
    assert gradient["gradient_energy_ratio"] == pytest.approx(0.0)


def test_coexpression_and_spot_profile_measure_different_axes():
    _coords, target = _field(seed=4)
    perfect_coexpression = coexpression_agreement(target, target)
    perfect_spots = spot_profile_agreement(target, target)
    assert perfect_coexpression["correlation_matrix_pcc"] == pytest.approx(1.0)
    assert perfect_spots["mean_spot_profile_pcc"] == pytest.approx(1.0)
    assert perfect_coexpression["n_gene_pairs"] == 15


def test_noise_ceiling_alignment_and_low_ceiling_exclusion():
    coords, target = _field(seed=5, genes=3)
    del coords
    report = noise_ceiling_adjusted_pcc(
        target, target, ["a", "b", "c"],
        {"a": 0.5, "b": 1.0, "c": 0.01},
        panel_indices={"pair": np.array([0, 1])}, minimum_ceiling=0.05,
    )
    assert report["all_genes"]["n_usable_genes"] == 2
    assert report["all_genes"]["mean_fraction_achievable"] == pytest.approx(1.5)
    assert report["pair"]["n_usable_genes"] == 2


def test_knn_edges_are_unique_undirected_and_validate_inputs():
    coords, _target = _field(side=3)
    edges = undirected_knn_edges(coords, k_neighbors=3)
    assert np.all(edges[:, 0] < edges[:, 1])
    assert len({tuple(row) for row in edges.tolist()}) == len(edges)
    with pytest.raises(ValueError, match="positive"):
        undirected_knn_edges(coords, k_neighbors=0)


def _whole_slide_record(sample, patient, organ, value):
    return {
        "sample_id": sample,
        "patient_id": patient,
        "organ": organ,
        "point_metrics": {"all_genes": {"pcc": value, "rmse": 1.0 - value}},
        "structured_field": {
            "panels": {
                "all_genes": {
                    "moran_local": {
                        "moran_i_pcc": value,
                        "n_eligible_genes": 4,
                    },
                },
            },
        },
        "noise_ceiling_adjusted_pcc": None,
    }


def test_whole_slide_aggregation_is_patient_macro_and_organ_stratified():
    report = _aggregate_whole_slide_structured_reports([
        _whole_slide_record("s0", "p0", "kidney", 0.2),
        _whole_slide_record("s1", "p0", "kidney", 0.4),
        _whole_slide_record("s2", "p1", "lung", 0.8),
    ])
    # p0 contributes its within-patient mean 0.3 once; p1 contributes 0.8.
    pcc = report["point_metrics_patient_aggregated"]["all_genes"]["pcc"]
    assert pcc["patient_mean"] == pytest.approx(0.55)
    assert pcc["n_patients"] == 2
    assert set(report["by_organ"]) == {"kidney", "lung"}
    kidney = report["by_organ"]["kidney"]["structured_metrics_patient_aggregated"]
    assert kidney["all_genes"]["moran_local.moran_i_pcc"]["patient_mean"] == pytest.approx(0.3)
