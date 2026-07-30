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


def _valid_example(n_observed=6, n_query=3, n_genes=5, local_k=2):
    rng = np.random.default_rng(0)
    inputs = SpatialFieldInputs(
        sample_id="s1", patient_id="p1",
        observed_barcodes=np.array([f"o{i}" for i in range(n_observed)]),
        query_barcodes=np.array([f"q{i}" for i in range(n_query)]),
        observed_coords=rng.normal(size=(n_observed, 2)).astype(np.float32),
        query_coords=rng.normal(size=(n_query, 2)).astype(np.float32),
        observed_full_gene_expression=rng.normal(size=(n_observed, n_genes)).astype(np.float32),
        observed_gigapath_features=rng.normal(size=(n_observed, 1536)).astype(np.float32),
        observed_image_available=np.ones(n_observed, dtype=bool),
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


def test_rejects_empty_boundary_before_model_attention():
    inputs, targets = _valid_example()
    bad_inputs = _replace(
        inputs,
        boundary_idx=np.array([], dtype=int),
        boundary_ring=np.array([], dtype=int),
    )
    with pytest.raises(ValueError, match="boundary_idx is empty"):
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


def test_rejects_observed_image_available_shape_mismatch():
    inputs, targets = _valid_example(n_observed=6)
    bad_inputs = _replace(inputs, observed_image_available=np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="observed_image_available must be"):
        validate_spatial_field_example(bad_inputs, targets)


def test_rejects_non_boolean_observed_image_available():
    inputs, targets = _valid_example(n_observed=6)
    bad_flags = np.zeros(6)
    bad_flags[0] = 0.5  # not a real boolean/0-1 value
    bad_inputs = _replace(inputs, observed_image_available=bad_flags)
    with pytest.raises(ValueError, match="boolean"):
        validate_spatial_field_example(bad_inputs, targets)


def test_rejects_a_nonzero_image_feature_for_an_unavailable_spot():
    """15th Codex re-audit (Step 5 acceptance criteria): missing images
    must never be represented as an ordinary (non-zero) image feature
    with no flag -- the converse must also be structurally impossible:
    a spot FLAGGED unavailable must have an explicitly zeroed feature,
    never a leftover real/garbage value."""
    inputs, targets = _valid_example(n_observed=6)
    flags = np.ones(6, dtype=bool)
    flags[0] = False
    bad_inputs = _replace(inputs, observed_image_available=flags)  # features NOT zeroed for spot 0
    with pytest.raises(ValueError, match="explicit zero"):
        validate_spatial_field_example(bad_inputs, targets)


def test_accepts_a_zeroed_image_feature_for_an_unavailable_spot():
    inputs, targets = _valid_example(n_observed=6)
    flags = np.ones(6, dtype=bool)
    flags[0] = False
    features = inputs.observed_gigapath_features.copy()
    features[0] = 0.0
    good_inputs = _replace(inputs, observed_image_available=flags, observed_gigapath_features=features)
    validate_spatial_field_example(good_inputs, targets)  # must not raise


_VALID_WSI_KWARGS = dict(
    wsi_tile_features=np.zeros((2, 1536), dtype=np.float32),
    wsi_tile_longnet_coords=np.asarray([[9000.0, 9000.0], [9100.0, 9100.0]], dtype=np.float32),
    wsi_tile_regional_coords=np.asarray([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float32),
    full_slide_coord_bounds=(-5.0, 5.0, -5.0, 5.0),
    slide_cache_namespace="content-hash:visible-set:checkpoint-sha256",
)


def test_rejects_wsi_fields_set_partially():
    inputs, targets = _valid_example()
    bad_inputs = _replace(inputs, wsi_tile_features=np.zeros((2, 1536), dtype=np.float32))
    with pytest.raises(ValueError, match="must be set together"):
        validate_spatial_field_example(bad_inputs, targets)


def test_rejects_wsi_context_with_no_slide_cache_namespace():
    """16th Codex re-audit (Step 5 Part 2 acceptance criteria),
    CONFIRMED: a prior version's all-or-none check did not include
    slide_cache_namespace at all -- WSI context could pass validation
    with slide_cache_namespace=None, defeating the whole point of a
    real, required LongNet cache-key binding."""
    inputs, targets = _valid_example()
    kwargs = dict(_VALID_WSI_KWARGS)
    kwargs["slide_cache_namespace"] = None
    bad_inputs = _replace(inputs, **kwargs)
    with pytest.raises(ValueError, match="must be set together"):
        validate_spatial_field_example(bad_inputs, targets)


def test_rejects_a_blank_slide_cache_namespace():
    inputs, targets = _valid_example()
    kwargs = dict(_VALID_WSI_KWARGS)
    kwargs["slide_cache_namespace"] = "   "
    bad_inputs = _replace(inputs, **kwargs)
    with pytest.raises(ValueError, match="non-empty string"):
        validate_spatial_field_example(bad_inputs, targets)


def test_rejects_degenerate_full_slide_coord_bounds():
    inputs, targets = _valid_example()
    kwargs = dict(_VALID_WSI_KWARGS)
    kwargs["full_slide_coord_bounds"] = (0.0, 0.0, -5.0, 5.0)  # xmax == xmin
    bad_inputs = _replace(inputs, **kwargs)
    with pytest.raises(ValueError, match="degenerate"):
        validate_spatial_field_example(bad_inputs, targets)


def test_rejects_a_visible_tile_outside_the_full_slide_bounds():
    """15th Codex re-audit (Step 5 acceptance criteria): regional-grid
    bounds must come from the COMPLETE slide -- a visible tile falling
    outside those bounds indicates a mismatched/stale bounds computation,
    which would make regional grid cell (i, j) refer to inconsistent
    physical regions across examples. Checked against the REGIONAL
    (normalized) coordinates, not the native GigaPath coordinates, which
    live in an entirely different, unnormalized scale."""
    inputs, targets = _valid_example()
    kwargs = dict(_VALID_WSI_KWARGS)
    kwargs["wsi_tile_features"] = np.zeros((1, 1536), dtype=np.float32)
    kwargs["wsi_tile_longnet_coords"] = np.asarray([[9000.0, 9000.0]], dtype=np.float32)
    kwargs["wsi_tile_regional_coords"] = np.asarray([[100.0, 100.0]], dtype=np.float32)  # well outside bounds
    bad_inputs = _replace(inputs, **kwargs)
    with pytest.raises(ValueError, match="outside full_slide_coord_bounds"):
        validate_spatial_field_example(bad_inputs, targets)


def test_accepts_valid_wsi_context_fields():
    inputs, targets = _valid_example()
    good_inputs = _replace(inputs, **_VALID_WSI_KWARGS)
    validate_spatial_field_example(good_inputs, targets)  # must not raise


def test_native_and_regional_wsi_coords_are_independently_scaled():
    """The core reason two separate fields exist: native GigaPath
    coordinates and normalized regional coordinates live on completely
    different scales for the same real tile -- one is not simply a
    scaled copy discoverable from the other without the same
    reference/scale example_builder.py used, so validation must accept
    them independently rather than assuming any fixed relationship
    between the two arrays' raw values."""
    inputs, targets = _valid_example()
    good_inputs = _replace(inputs, **_VALID_WSI_KWARGS)
    assert not np.allclose(
        good_inputs.wsi_tile_longnet_coords[:, 0], good_inputs.wsi_tile_regional_coords[:, 0],
    )
    validate_spatial_field_example(good_inputs, targets)  # must not raise


def test_rejects_a_scalar_wsi_tile_features_with_a_clean_valueerror():
    """18th Codex re-audit (Step 5 Part 2, "Other real gaps"), CONFIRMED:
    a prior version read wsi_tile_features.shape[0] BEFORE checking
    ndim -- a scalar (0-d) array has shape () and shape[0] raises a raw
    IndexError, not the intended, actionable ValueError."""
    inputs, targets = _valid_example()
    kwargs = dict(_VALID_WSI_KWARGS)
    kwargs["wsi_tile_features"] = np.asarray(7.0)  # a genuine 0-d scalar array
    bad_inputs = _replace(inputs, **kwargs)
    with pytest.raises(ValueError, match=r"wsi_tile_features must be \[N, F\]"):
        validate_spatial_field_example(bad_inputs, targets)


def test_rejects_a_wsi_coordinate_array_with_the_wrong_column_count():
    """17th Codex re-audit (Step 5 Part 2 launch blocker, "Important
    before Step 6/7"), CONFIRMED: a prior version only checked ROW
    counts against wsi_tile_features -- a [N, 3] or [N] coordinate array
    with a matching row count would have silently passed."""
    inputs, targets = _valid_example()
    kwargs = dict(_VALID_WSI_KWARGS)
    kwargs["wsi_tile_longnet_coords"] = np.asarray([[9000.0, 9000.0, 0.0], [9100.0, 9100.0, 0.0]], dtype=np.float32)
    bad_inputs = _replace(inputs, **kwargs)
    with pytest.raises(ValueError, match=r"wsi_tile_longnet_coords must be \[2, 2\]"):
        validate_spatial_field_example(bad_inputs, targets)

    kwargs2 = dict(_VALID_WSI_KWARGS)
    kwargs2["wsi_tile_regional_coords"] = np.asarray([-1.0, 1.0], dtype=np.float32)
    bad_inputs2 = _replace(inputs, **kwargs2)
    with pytest.raises(ValueError, match=r"wsi_tile_regional_coords must be \[2, 2\]"):
        validate_spatial_field_example(bad_inputs2, targets)


def test_rejects_duplicate_wsi_tile_coordinates_in_either_frame():
    """17th Codex re-audit (Step 5 Part 2 launch blocker, "Important
    before Step 6/7"), CONFIRMED: no check existed for duplicate
    coordinates in EITHER WSI frame independently -- a corrupted or
    mismatched dense_wsi_cache (coords/mask_coords sourced separately)
    could produce duplicate visible tiles in one frame without the other,
    double-counting that region's contribution to regional pooling or
    the LongNet global vector."""
    inputs, targets = _valid_example()
    kwargs = dict(_VALID_WSI_KWARGS)
    kwargs["wsi_tile_longnet_coords"] = np.asarray([[9000.0, 9000.0], [9000.0, 9000.0]], dtype=np.float32)
    bad_inputs = _replace(inputs, **kwargs)
    with pytest.raises(ValueError, match="wsi_tile_longnet_coords contains duplicate"):
        validate_spatial_field_example(bad_inputs, targets)

    kwargs2 = dict(_VALID_WSI_KWARGS)
    kwargs2["wsi_tile_regional_coords"] = np.asarray([[-1.0, -1.0], [-1.0, -1.0]], dtype=np.float32)
    bad_inputs2 = _replace(inputs, **kwargs2)
    with pytest.raises(ValueError, match="wsi_tile_regional_coords contains duplicate"):
        validate_spatial_field_example(bad_inputs2, targets)


def test_rejects_a_non_string_slide_cache_namespace():
    inputs, targets = _valid_example()
    kwargs = dict(_VALID_WSI_KWARGS)
    kwargs["slide_cache_namespace"] = 12345
    bad_inputs = _replace(inputs, **kwargs)
    with pytest.raises(ValueError, match="non-empty string"):
        validate_spatial_field_example(bad_inputs, targets)


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
