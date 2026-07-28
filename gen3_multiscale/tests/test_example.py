"""Phase 1 (multiscale spatial-field handoff): the shared example object's
structural contract. Model-forward inputs (SpatialFieldInputs) and
loss-only targets (SpatialFieldTargets) are separate types on purpose --
these tests exercise both valid construction and every violation
validate_spatial_field_example is meant to catch."""
import numpy as np
import pytest

from gen3_multiscale.data.example import (
    SpatialFieldInputs, SpatialFieldTargets, validate_spatial_field_example,
)


def _valid_example(n_observed=6, n_query=3, n_genes=5, gex_dim=4, local_k=2):
    rng = np.random.default_rng(0)
    inputs = SpatialFieldInputs(
        sample_id="s1", patient_id="p1",
        observed_barcodes=np.array([f"o{i}" for i in range(n_observed)]),
        query_barcodes=np.array([f"q{i}" for i in range(n_query)]),
        observed_coords=rng.normal(size=(n_observed, 2)).astype(np.float32),
        query_coords=rng.normal(size=(n_query, 2)).astype(np.float32),
        observed_gex_conditioning=rng.normal(size=(n_observed, gex_dim)).astype(np.float32),
        observed_full_gene_expression=rng.normal(size=(n_observed, n_genes)).astype(np.float32),
        observed_gigapath_features=rng.normal(size=(n_observed, 1536)).astype(np.float32),
        query_local_neighbor_idx=rng.integers(0, n_observed, size=(n_query, local_k)),
        boundary_idx=np.array([0, 1, 2]),
        boundary_ring=np.array([1, 1, 2]),
        query_depth_to_boundary=np.array([0, 1, 2]),
    )
    targets = SpatialFieldTargets(query_expression=rng.normal(size=(n_query, n_genes)).astype(np.float32))
    return inputs, targets


def test_a_well_formed_example_passes_validation():
    inputs, targets = _valid_example()
    validate_spatial_field_example(inputs, targets)  # must not raise


def test_rejects_observed_and_query_barcode_overlap():
    inputs, targets = _valid_example()
    bad_query_barcodes = inputs.query_barcodes.copy()
    bad_query_barcodes[0] = inputs.observed_barcodes[0]  # leak: a query spot is also "observed"
    bad_inputs = _replace(inputs, query_barcodes=bad_query_barcodes)
    with pytest.raises(ValueError, match="overlap"):
        validate_spatial_field_example(bad_inputs, targets)


def test_rejects_empty_observed_set():
    inputs, targets = _valid_example()
    bad_inputs = _replace(
        inputs, observed_coords=np.zeros((0, 2), dtype=np.float32), observed_barcodes=np.array([]),
        observed_gex_conditioning=np.zeros((0, inputs.observed_gex_conditioning.shape[1]), dtype=np.float32),
        observed_full_gene_expression=np.zeros((0, inputs.observed_full_gene_expression.shape[1]), dtype=np.float32),
        observed_gigapath_features=np.zeros((0, 1536), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="observed_coords is empty"):
        validate_spatial_field_example(bad_inputs, targets)


def test_rejects_empty_query_set():
    inputs, targets = _valid_example()
    bad_inputs = _replace(inputs, query_coords=np.zeros((0, 2), dtype=np.float32), query_barcodes=np.array([]))
    with pytest.raises(ValueError, match="query_coords is empty"):
        validate_spatial_field_example(bad_inputs, targets)


def test_rejects_local_neighbor_index_outside_observed_range():
    inputs, targets = _valid_example(n_observed=6)
    bad_neighbors = inputs.query_local_neighbor_idx.copy()
    bad_neighbors[0, 0] = 999  # not a valid position in the observed_* arrays
    bad_inputs = _replace(inputs, query_local_neighbor_idx=bad_neighbors)
    with pytest.raises(ValueError, match="query_local_neighbor_idx"):
        validate_spatial_field_example(bad_inputs, targets)


def test_rejects_boundary_idx_outside_observed_range():
    inputs, targets = _valid_example(n_observed=6)
    bad_inputs = _replace(inputs, boundary_idx=np.array([0, 1, 999]))
    with pytest.raises(ValueError, match="boundary_idx"):
        validate_spatial_field_example(bad_inputs, targets)


def test_rejects_boundary_ring_outside_1_2_3():
    inputs, targets = _valid_example()
    bad_inputs = _replace(inputs, boundary_ring=np.array([1, 1, 7]))
    with pytest.raises(ValueError, match="boundary_ring"):
        validate_spatial_field_example(bad_inputs, targets)


def test_rejects_negative_depth_to_boundary():
    inputs, targets = _valid_example()
    bad_inputs = _replace(inputs, query_depth_to_boundary=np.array([0, -1, 2]))
    with pytest.raises(ValueError, match="non-negative"):
        validate_spatial_field_example(bad_inputs, targets)


def test_rejects_non_finite_observed_expression():
    inputs, targets = _valid_example()
    bad_expr = inputs.observed_full_gene_expression.copy()
    bad_expr[0, 0] = np.nan
    bad_inputs = _replace(inputs, observed_full_gene_expression=bad_expr)
    with pytest.raises(ValueError, match="non-finite"):
        validate_spatial_field_example(bad_inputs, targets)


def test_rejects_target_gene_panel_width_mismatch():
    inputs, targets = _valid_example(n_genes=5)
    bad_targets = SpatialFieldTargets(query_expression=np.zeros((inputs.query_coords.shape[0], 3), dtype=np.float32))
    with pytest.raises(ValueError, match="gene panel width"):
        validate_spatial_field_example(inputs, bad_targets)


def test_rejects_mismatched_query_raw_counts_shape():
    inputs, targets = _valid_example(n_query=3, n_genes=5)
    bad_targets = SpatialFieldTargets(
        query_expression=targets.query_expression, query_raw_counts=np.zeros((2, 5), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="query_raw_counts"):
        validate_spatial_field_example(inputs, bad_targets)


def test_targets_are_a_distinct_type_from_inputs():
    """Enforces the handoff's "separated by type and API, not merely by
    convention" requirement -- a model's forward(inputs: SpatialFieldInputs)
    type hint cannot silently accept a SpatialFieldTargets."""
    inputs, targets = _valid_example()
    assert type(inputs) is not type(targets)
    assert not hasattr(inputs, "query_expression")
    assert not hasattr(targets, "observed_full_gene_expression")


def _replace(inputs: SpatialFieldInputs, **overrides) -> SpatialFieldInputs:
    from dataclasses import replace
    return replace(inputs, **overrides)
