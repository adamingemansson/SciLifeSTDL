"""Phase 7 metrics: verbatim-copied pointwise/distributional functions
(smoke-tested here for the copy itself, not re-deriving gen2's own full
test battery) plus the new patient-aggregation, gene-panel, binning,
edge-gradient, and spatial-agreement functions this project needed built
from scratch (confirmed absent anywhere else in the repo)."""
import numpy as np
from scipy.stats import pearsonr

from gen3_multiscale.evaluation.metrics import (
    aggregate_patient_metrics, boundary_interior_bins, edge_gradient_agreement, gene_panel_metrics,
    graph_laplacian_agreement, hole_size_bins, nonzero_auc, pearson_per_gene, resolve_gene_panels, rmse,
    spatial_variogram_agreement, st_fid, st_mmd,
)


# ---------------------------------------------------------------------------
# Verbatim-copy smoke tests.
# ---------------------------------------------------------------------------
def test_pearson_per_gene_perfect_and_constant_cases():
    rng = np.random.default_rng(0)
    true = rng.normal(size=(20, 3))
    pred_perfect = true.copy()
    out = pearson_per_gene(pred_perfect, true)
    assert np.allclose(out, 1.0, atol=1e-6)

    true_with_constant = true.copy()
    true_with_constant[:, 0] = 5.0  # truth-constant gene -> NaN
    out2 = pearson_per_gene(pred_perfect, true_with_constant)
    assert np.isnan(out2[0])


def test_pearson_per_gene_vectorized_matches_scalar_reference_and_preserves_failure_semantics():
    rng = np.random.default_rng(17)
    true = rng.normal(size=(257, 31)).astype(np.float32)
    pred = (0.4 * true + rng.normal(size=true.shape)).astype(np.float32)
    true[:, 3] = 7.0       # truth-constant: ineligible -> NaN
    pred[:, 8] = -2.0      # prediction-constant with variable truth -> 0

    actual = pearson_per_gene(pred, true)
    expected = np.array([
        np.nan if np.std(true[:, gene]) < 1e-8
        else 0.0 if np.std(pred[:, gene]) < 1e-8
        else pearsonr(pred[:, gene], true[:, gene])[0]
        for gene in range(true.shape[1])
    ])
    assert np.allclose(actual, expected, atol=1e-7, equal_nan=True)


def test_rmse_zero_for_identical_arrays():
    a = np.random.default_rng(0).normal(size=(10, 4))
    assert rmse(a, a) == 0.0


def test_nonzero_auc_perfect_separation():
    true = np.array([[0.0, 1.0, 0.0, 1.0]])
    pred = np.array([[0.0, 1.0, 0.0, 1.0]])
    assert nonzero_auc(pred, true) == 1.0


def test_st_fid_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(50, 6))
    assert st_fid(embeddings, embeddings) < 1e-6


def test_st_mmd_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(50, 6))
    assert abs(st_mmd(embeddings, embeddings)) < 1e-6


# ---------------------------------------------------------------------------
# resolve_gene_panels / gene_panel_metrics
# ---------------------------------------------------------------------------
def test_resolve_gene_panels_reports_missing_genes_without_changing_panel_size():
    gene_names = ["A", "B", "C"]
    indices, metadata = resolve_gene_panels(gene_names, {"my_panel": ["A", "C", "Z"]})
    assert list(indices["my_panel"]) == [0, 2]
    assert metadata["my_panel"]["missing_genes"] == ["Z"]
    assert metadata["my_panel"]["evaluated_count"] == 2


def test_resolve_gene_panels_rejects_bad_panel_names():
    try:
        resolve_gene_panels(["A"], {"bad name!": ["A"]})
        assert False, "expected a ValueError"
    except ValueError as exc:
        assert "letters, digits" in str(exc)


def test_resolve_gene_panels_rejects_a_panel_with_no_matching_genes():
    try:
        resolve_gene_panels(["A", "B"], {"panel": ["Z"]})
        assert False, "expected a ValueError"
    except ValueError as exc:
        assert "no genes" in str(exc)


def test_gene_panel_metrics_restricts_pcc_and_rmse_to_the_named_panel():
    rng = np.random.default_rng(0)
    gene_names = ["A", "B", "C"]
    true = rng.normal(size=(30, 3))
    pred = true.copy()
    pred[:, 1] += rng.normal(size=30) * 10  # corrupt gene B only
    out = gene_panel_metrics(pred, true, gene_names, {"clean_panel": ["A", "C"]})
    assert out["clean_panel"]["rmse"] < 1e-6
    assert out["clean_panel"]["pcc"] > 0.99


# ---------------------------------------------------------------------------
# aggregate_patient_metrics
# ---------------------------------------------------------------------------
def test_aggregate_patient_metrics_macro_averages_across_patients_not_pooled_items():
    """Patient A has 1 item at value 0.0; patient B has 3 items at value
    1.0. The pooled mean is dominated by B (0.75); the patient-mean must
    weight A and B equally (0.5)."""
    per_item = [{"pcc": 0.0}, {"pcc": 1.0}, {"pcc": 1.0}, {"pcc": 1.0}]
    patients = ["A", "B", "B", "B"]
    out = aggregate_patient_metrics(per_item, patients)
    assert abs(out["pcc"]["pooled_mean"] - 0.75) < 1e-9
    assert abs(out["pcc"]["patient_mean"] - 0.5) < 1e-9
    assert out["pcc"]["n_patients"] == 2


def test_aggregate_patient_metrics_reports_ci_not_estimable_for_one_patient():
    per_item = [{"rmse": 1.0}, {"rmse": 2.0}]
    patients = ["only_patient", "only_patient"]
    out = aggregate_patient_metrics(per_item, patients)
    assert out["rmse"]["n_patients"] == 1
    assert out["rmse"]["ci_estimable"] == 0.0
    assert np.isnan(out["rmse"]["patient_ci95_low"])


def test_aggregate_patient_metrics_computes_a_ci_for_multiple_patients():
    per_item = [{"pcc": v} for v in [0.1, 0.2, 0.3, 0.9, 0.8, 0.7]]
    patients = ["A", "A", "A", "B", "B", "B"]
    out = aggregate_patient_metrics(per_item, patients)
    assert out["pcc"]["ci_estimable"] == 1.0
    assert out["pcc"]["patient_ci95_low"] < out["pcc"]["patient_mean"] < out["pcc"]["patient_ci95_high"]


def test_aggregate_patient_metrics_rejects_mismatched_lengths():
    try:
        aggregate_patient_metrics([{"pcc": 1.0}], ["A", "B"])
        assert False, "expected a ValueError"
    except ValueError as exc:
        assert "length" in str(exc)


# ---------------------------------------------------------------------------
# boundary_interior_bins / hole_size_bins
# ---------------------------------------------------------------------------
def test_boundary_interior_bins_groups_by_depth():
    per_query_metric = np.array([0.9, 0.9, 0.5, 0.5, 0.1, 0.1])
    depth = np.array([0, 0, 1, 1, 2, 2])
    out = boundary_interior_bins(per_query_metric, depth)
    assert abs(out["<0.5"]["mean"] - 0.9) < 1e-9
    assert abs(out["[0.5,1.5)"]["mean"] - 0.5) < 1e-9
    assert abs(out["[1.5,2.5)"]["mean"] - 0.1) < 1e-9


def test_hole_size_bins_groups_by_hole_size():
    per_item_metric = np.array([1.0, 1.0, 5.0, 5.0])
    hole_sizes = np.array([5, 5, 50, 50])
    out = hole_size_bins(per_item_metric, hole_sizes, bin_edges=np.array([10.0]))
    assert abs(out["<10"]["mean"] - 1.0) < 1e-9
    assert abs(out[">=10"]["mean"] - 5.0) < 1e-9


# ---------------------------------------------------------------------------
# edge_gradient_agreement
# ---------------------------------------------------------------------------
def test_edge_gradient_agreement_classifies_edges_and_returns_bounded_correlations():
    rng = np.random.default_rng(0)
    n = 8
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    n_query = coords.shape[0]
    true = rng.normal(size=(n_query, 5))
    predicted = true + rng.normal(size=(n_query, 5)) * 0.01
    out = edge_gradient_agreement(predicted, true, coords, k_neighbors=4)
    assert out["normal_n_edges"] + out["tangential_n_edges"] > 0
    for key in ("normal_pcc", "tangential_pcc"):
        assert np.isnan(out[key]) or -1.0 - 1e-6 <= out[key] <= 1.0 + 1e-6


def test_edge_gradient_agreement_rejects_row_count_mismatch():
    coords = np.random.default_rng(0).normal(size=(5, 2))
    pred = np.random.default_rng(0).normal(size=(4, 3))
    true = np.random.default_rng(0).normal(size=(4, 3))
    try:
        edge_gradient_agreement(pred, true, coords)
        assert False, "expected a ValueError"
    except ValueError as exc:
        assert "row" in str(exc)


# ---------------------------------------------------------------------------
# spatial_variogram_agreement / graph_laplacian_agreement
# ---------------------------------------------------------------------------
def test_spatial_variogram_agreement_is_near_perfect_for_identical_fields():
    rng = np.random.default_rng(0)
    n = 6
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    field = rng.normal(size=(coords.shape[0], 4)) + coords.sum(axis=1, keepdims=True)
    out = spatial_variogram_agreement(field, field, coords, n_bins=4)
    assert out["agreement_pcc"] > 0.99


def test_spatial_variogram_agreement_detects_a_spatially_scrambled_field():
    """Same per-point values, shuffled among coordinates -- the pointwise
    PCC/RMSE would be unaffected by this corruption (it's not what they
    measure), but the variogram (which depends on WHICH points are near
    each other) must disagree with the true, spatially-coherent field."""
    rng = np.random.default_rng(1)
    n = 6
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    true_field = coords.sum(axis=1, keepdims=True) + rng.normal(size=(coords.shape[0], 1)) * 0.01
    true_field = np.tile(true_field, (1, 4))
    scrambled = rng.permutation(true_field)
    out = spatial_variogram_agreement(scrambled, true_field, coords, n_bins=4)
    assert out["agreement_pcc"] < 0.9 or np.isnan(out["agreement_pcc"])


def test_graph_laplacian_agreement_ratio_is_one_for_identical_fields():
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(20, 2))
    field = rng.normal(size=(20, 5))
    out = graph_laplacian_agreement(field, field, coords, k_neighbors=4)
    assert abs(out["energy_ratio"] - 1.0) < 1e-9


def test_graph_laplacian_agreement_ratio_exceeds_one_for_a_noisier_prediction():
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(20, 2))
    true_field = coords.sum(axis=1, keepdims=True) * np.ones((1, 5))
    noisy_field = true_field + rng.normal(size=(20, 5)) * 5.0
    out = graph_laplacian_agreement(noisy_field, true_field, coords, k_neighbors=4)
    assert out["energy_ratio"] > 1.0
