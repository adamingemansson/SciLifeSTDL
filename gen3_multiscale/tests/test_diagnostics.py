"""Phase 7 diagnostic evaluation interventions, exercised against real
forward passes (Architecture1/Architecture3) on the same synthetic-grid
pattern test_architectures.py uses, plus a block-level test of the
global-slide-vector intervention (see diagnostics.py's module docstring
for why that one can't yet be exercised against a full architecture
wrapper -- use_global_slide=True isn't wired into any forward() yet)."""
import numpy as np
import torch

from gen3_multiscale.data.boundary_graph import extract_boundary_and_local_context
from gen3_multiscale.data.example import SpatialFieldInputs, SpatialFieldTargets, validate_spatial_field_example
from gen3_multiscale.evaluation.diagnostics import (
    permute_boundary_order, shuffle_boundary_gex, shuffle_observed_gex, swap_global_slide_vector, zero_global_slide_vector,
    zero_he, zero_observed_gex,
)
from gen3_multiscale.models.architectures import Architecture1
from gen3_multiscale.models.backbone import SpatialFieldBackbone


def _synthetic_inputs(n_genes=6, gex_dim=4, image_dim=8, seed=0):
    rng = np.random.default_rng(seed)
    n = 15
    lo = -(n // 2)
    xs, ys = np.meshgrid(np.arange(lo, lo + n), np.arange(lo, lo + n))
    grid = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    dist = np.linalg.norm(grid, axis=1)
    observed_coords, query_coords = grid[dist > 2.5], grid[dist <= 2.5]
    n_observed, n_query = observed_coords.shape[0], query_coords.shape[0]

    result = extract_boundary_and_local_context(observed_coords, query_coords, k_neighbors=6, local_k=6, max_rings=3)

    inputs = SpatialFieldInputs(
        sample_id="s1", patient_id="p1",
        observed_barcodes=np.array([f"o{i}" for i in range(n_observed)]),
        query_barcodes=np.array([f"q{i}" for i in range(n_query)]),
        observed_coords=observed_coords.astype(np.float32),
        query_coords=query_coords.astype(np.float32),
        observed_full_gene_expression=rng.normal(size=(n_observed, n_genes)).astype(np.float32),
        observed_gigapath_features=rng.normal(size=(n_observed, image_dim)).astype(np.float32),
        observed_image_available=np.ones(n_observed, dtype=bool),
        query_local_neighbor_idx=result.query_local_neighbor_idx,
        boundary_idx=result.boundary_idx,
        boundary_ring=result.boundary_ring,
        query_depth_to_boundary=result.query_depth_to_boundary,
    )
    targets = SpatialFieldTargets(query_expression=rng.normal(size=(n_query, n_genes)).astype(np.float32))
    validate_spatial_field_example(inputs, targets)
    return inputs, targets, n_genes, gex_dim, image_dim


_MODEL_KWARGS = dict(hidden_dim=32, n_heads=4, n_blocks=2, dense_threshold=100)


def _forward(inputs, n_genes, gex_dim, image_dim, seed=0):
    torch.manual_seed(seed)
    model = Architecture1(n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim, **_MODEL_KWARGS)
    torch.manual_seed(123)  # keep forward-pass RNG (if any) independent of construction RNG
    with torch.no_grad():
        return model(inputs)["expression"]


def test_zero_observed_gex_produces_a_valid_example_and_changes_the_prediction():
    inputs, targets, n_genes, gex_dim, image_dim = _synthetic_inputs()
    perturbed = zero_observed_gex(inputs)
    validate_spatial_field_example(perturbed, targets)
    assert np.allclose(perturbed.observed_full_gene_expression, 0.0)

    baseline = _forward(inputs, n_genes, gex_dim, image_dim)
    ablated = _forward(perturbed, n_genes, gex_dim, image_dim)
    assert not torch.allclose(baseline, ablated)


def test_shuffle_observed_gex_preserves_the_value_multiset_and_changes_the_prediction():
    inputs, targets, n_genes, gex_dim, image_dim = _synthetic_inputs()
    perturbed = shuffle_observed_gex(inputs, seed=0)
    validate_spatial_field_example(perturbed, targets)
    assert np.allclose(
        np.sort(perturbed.observed_full_gene_expression, axis=0),
        np.sort(inputs.observed_full_gene_expression, axis=0),
    )

    baseline = _forward(inputs, n_genes, gex_dim, image_dim)
    ablated = _forward(perturbed, n_genes, gex_dim, image_dim)
    assert not torch.allclose(baseline, ablated)


def test_zero_he_produces_a_valid_example_and_changes_the_prediction():
    inputs, targets, n_genes, gex_dim, image_dim = _synthetic_inputs()
    perturbed = zero_he(inputs)
    validate_spatial_field_example(perturbed, targets)
    assert np.allclose(perturbed.observed_gigapath_features, 0.0)

    baseline = _forward(inputs, n_genes, gex_dim, image_dim)
    ablated = _forward(perturbed, n_genes, gex_dim, image_dim)
    assert not torch.allclose(baseline, ablated)


def test_permute_boundary_order_produces_a_valid_example_and_leaves_the_prediction_unchanged():
    """"Context-token order randomly permuted while predictions remain
    equivalent up to numerical tolerance" -- the integration-level version
    of ChunkedCrossAttention's own unit-level permutation-invariance gate
    (Phase 5)."""
    inputs, targets, n_genes, gex_dim, image_dim = _synthetic_inputs()
    perturbed = permute_boundary_order(inputs, seed=0)
    validate_spatial_field_example(perturbed, targets)
    assert set(perturbed.boundary_idx.tolist()) == set(inputs.boundary_idx.tolist())
    assert perturbed.boundary_idx.tolist() != inputs.boundary_idx.tolist()  # order actually changed

    baseline = _forward(inputs, n_genes, gex_dim, image_dim)
    reordered = _forward(perturbed, n_genes, gex_dim, image_dim)
    assert torch.allclose(baseline, reordered, atol=1e-4)


def test_shuffle_boundary_gex_leaves_locally_referenced_and_non_boundary_spots_untouched():
    """Hand-built minimal example with a known, deliberately overlapping
    local/boundary set so the "local neighbours remain intact" contract
    can be checked exactly rather than only observed statistically on a
    real grid."""
    n_observed, n_genes = 6, 3
    observed_coords = np.arange(n_observed, dtype=np.float32)[:, None] * np.array([1.0, 0.0], dtype=np.float32)
    full_expr = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0]],
        dtype=np.float32,
    )
    inputs = SpatialFieldInputs(
        sample_id="s", patient_id="p",
        observed_barcodes=np.array([f"o{i}" for i in range(n_observed)]),
        query_barcodes=np.array(["q0"]),
        observed_coords=observed_coords,
        query_coords=np.array([[10.0, 0.0]], dtype=np.float32),
        observed_full_gene_expression=full_expr,
        observed_gigapath_features=np.zeros((n_observed, 4), dtype=np.float32),
        observed_image_available=np.ones(n_observed, dtype=bool),
        query_local_neighbor_idx=np.array([[0, 1]]),  # local positions {0, 1}
        boundary_idx=np.array([1, 2, 3, 4]),  # position 1 overlaps with local
        boundary_ring=np.array([1, 1, 2, 2]),
        query_depth_to_boundary=np.array([0]),
    )
    targets = SpatialFieldTargets(query_expression=np.zeros((1, n_genes), dtype=np.float32))
    validate_spatial_field_example(inputs, targets)

    perturbed = shuffle_boundary_gex(inputs, seed=0)
    validate_spatial_field_example(perturbed, targets)

    # Position 1 is both a local neighbor AND a boundary spot -- must be untouched.
    assert np.array_equal(perturbed.observed_full_gene_expression[1], full_expr[1])
    # Position 0 (local-only) and position 5 (neither local nor boundary) untouched.
    assert np.array_equal(perturbed.observed_full_gene_expression[0], full_expr[0])
    assert np.array_equal(perturbed.observed_full_gene_expression[5], full_expr[5])
    # Positions {2, 3, 4} (boundary-only) are a PERMUTATION of their original rows.
    shuffle_only = [2, 3, 4]
    assert np.array_equal(
        np.sort(perturbed.observed_full_gene_expression[shuffle_only], axis=0),
        np.sort(full_expr[shuffle_only], axis=0),
    )


def test_shuffle_boundary_gex_is_a_noop_when_fewer_than_two_shufflable_positions_exist():
    n_observed = 4
    inputs = SpatialFieldInputs(
        sample_id="s", patient_id="p",
        observed_barcodes=np.array([f"o{i}" for i in range(n_observed)]),
        query_barcodes=np.array(["q0"]),
        observed_coords=np.arange(n_observed, dtype=np.float32)[:, None] * np.array([1.0, 0.0], dtype=np.float32),
        query_coords=np.array([[10.0, 0.0]], dtype=np.float32),
        observed_full_gene_expression=np.arange(n_observed * 2, dtype=np.float32).reshape(n_observed, 2),
        observed_gigapath_features=np.zeros((n_observed, 4), dtype=np.float32),
        observed_image_available=np.ones(n_observed, dtype=bool),
        query_local_neighbor_idx=np.array([[0, 1, 2]]),  # local covers everything boundary has
        boundary_idx=np.array([1, 2]),  # entirely overlapping with local -- nothing shufflable
        boundary_ring=np.array([1, 1]),
        query_depth_to_boundary=np.array([0]),
    )
    perturbed = shuffle_boundary_gex(inputs, seed=0)
    assert np.array_equal(perturbed.observed_full_gene_expression, inputs.observed_full_gene_expression)


def test_zero_and_swap_global_slide_vector_reject_or_apply_correctly():
    vector = torch.randn(8)
    assert torch.allclose(zero_global_slide_vector(vector), torch.zeros(8))

    other = torch.randn(8)
    assert torch.equal(swap_global_slide_vector(vector, other), other)

    try:
        swap_global_slide_vector(vector, torch.randn(5))
        assert False, "expected a ValueError"
    except ValueError as exc:
        assert "shape" in str(exc)


def test_global_slide_intervention_has_no_effect_at_initialization_but_measurable_effect_once_trained():
    """Two-part gate: (1) at a FRESH construction, GlobalConditioningFiLM
    is zero-initialized (Phase 5), so this diagnostic must show NO
    difference -- confirms the diagnostic doesn't manufacture a false
    positive on an untrained model. (2) after simulating training by
    perturbing the FiLM weights away from zero, the same zero/swap
    intervention MUST measurably change the block's output -- "otherwise
    the slide branch is functionally ignored" (the handoff's own Learning
    Tests gate), exercised here at the block level since no architecture
    wrapper exposes use_global_slide yet (see diagnostics.py docstring)."""
    hidden_dim, global_dim = 16, 8
    torch.manual_seed(0)
    backbone = SpatialFieldBackbone(
        n_blocks=1, hidden_dim=hidden_dim, n_heads=2, dense_threshold=100,
        use_global_slide=True, global_slide_dim=global_dim,
    )
    backbone.eval()  # dropout must be off -- otherwise per-call dropout masks would themselves
    # change the output between calls, confounding the effect actually under test.
    n_query, n_local, n_boundary = 5, 3, 4
    query_hidden = torch.randn(n_query, hidden_dim)
    query_coords = torch.randn(n_query, 2)
    local_hidden = torch.randn(n_query, n_local, hidden_dim)
    local_geometry = torch.randn(n_query, n_local, 3)
    boundary_hidden = torch.randn(n_boundary, hidden_dim)
    boundary_geometry = torch.randn(n_query, n_boundary, 3)
    global_vector = torch.randn(global_dim)
    other_vector = torch.randn(global_dim)

    kwargs = dict(
        query_coords=query_coords, local_hidden=local_hidden, local_geometry=local_geometry,
        boundary_hidden=boundary_hidden, boundary_geometry=boundary_geometry,
    )

    with torch.no_grad():
        out_original = backbone(query_hidden.clone(), global_slide_vector=global_vector, **kwargs)
        out_zeroed = backbone(query_hidden.clone(), global_slide_vector=zero_global_slide_vector(global_vector), **kwargs)
        out_swapped = backbone(query_hidden.clone(), global_slide_vector=swap_global_slide_vector(global_vector, other_vector), **kwargs)
    assert torch.allclose(out_original, out_zeroed, atol=1e-6)
    assert torch.allclose(out_original, out_swapped, atol=1e-6)

    film = backbone.blocks[0].global_slide_film
    with torch.no_grad():
        film.scale_proj.weight.normal_(std=0.5)
        film.shift_proj.weight.normal_(std=0.5)

    with torch.no_grad():
        out_original = backbone(query_hidden.clone(), global_slide_vector=global_vector, **kwargs)
        out_zeroed = backbone(query_hidden.clone(), global_slide_vector=zero_global_slide_vector(global_vector), **kwargs)
        out_swapped = backbone(query_hidden.clone(), global_slide_vector=swap_global_slide_vector(global_vector, other_vector), **kwargs)
    assert not torch.allclose(out_original, out_zeroed, atol=1e-4)
    assert not torch.allclose(out_original, out_swapped, atol=1e-4)
