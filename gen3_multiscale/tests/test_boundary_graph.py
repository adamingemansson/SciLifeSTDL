"""Phase 2 (multiscale spatial-field handoff): geometry-only spot graph,
boundary-ring (Rings 1-3) BFS extraction, true per-query local-k nearest
context. Exercises the handoff's own Phase 2 checklist and several of its
"Geometry and context tests" gates directly (opposite-side visibility,
locals differing across separated queries, no silent boundary truncation,
every boundary spot represented exactly once)."""
import numpy as np
import pytest

from gen3_multiscale.data.boundary_graph import build_knn_adjacency, extract_boundary_and_local_context
from gen3_multiscale.data.example import SpatialFieldInputs, SpatialFieldTargets, validate_spatial_field_example


def _square_grid(n=21, spacing=1.0):
    """An n x n grid of unit-spaced points, centered at the origin."""
    lo = -(n // 2)
    xs, ys = np.meshgrid(np.arange(lo, lo + n) * spacing, np.arange(lo, lo + n) * spacing)
    return np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)


def _split_by_circular_hole(grid, radius):
    dist = np.linalg.norm(grid, axis=1)
    query_mask = dist <= radius
    return grid[~query_mask], grid[query_mask]


def test_build_knn_adjacency_returns_k_neighbors_and_no_self_loops():
    grid = _square_grid(n=11)
    adjacency = build_knn_adjacency(grid, k_neighbors=6)
    assert len(adjacency) == grid.shape[0]
    for i, neighbors in enumerate(adjacency):
        assert i not in neighbors
        assert len(neighbors) <= 6


def test_opposite_sides_of_a_synthetic_hole_are_both_reachable_in_boundary():
    """Directly tests the handoff's "Opposite sides of a synthetic hole
    are both visible" gate."""
    grid = _square_grid(n=25)
    observed, query = _split_by_circular_hole(grid, radius=3.5)
    result = extract_boundary_and_local_context(observed, query, k_neighbors=6, local_k=8, max_rings=3)

    boundary_points = observed[result.boundary_idx]
    north = boundary_points[boundary_points[:, 1] > 0]
    south = boundary_points[boundary_points[:, 1] < 0]
    east = boundary_points[boundary_points[:, 0] > 0]
    west = boundary_points[boundary_points[:, 0] < 0]
    assert len(north) > 0 and len(south) > 0, "boundary must include both north and south of the hole"
    assert len(east) > 0 and len(west) > 0, "boundary must include both east and west of the hole"


def test_boundary_spots_are_represented_exactly_once():
    grid = _square_grid(n=21)
    observed, query = _split_by_circular_hole(grid, radius=3.0)
    result = extract_boundary_and_local_context(observed, query, k_neighbors=6, local_k=8, max_rings=3)
    assert len(np.unique(result.boundary_idx)) == len(result.boundary_idx)


def test_boundary_ring_values_reflect_increasing_distance_from_the_hole():
    grid = _square_grid(n=25)
    observed, query = _split_by_circular_hole(grid, radius=3.5)
    result = extract_boundary_and_local_context(observed, query, k_neighbors=6, local_k=8, max_rings=3)
    ring1_dist = np.linalg.norm(observed[result.boundary_idx[result.boundary_ring == 1]], axis=1).mean()
    ring3_dist = np.linalg.norm(observed[result.boundary_idx[result.boundary_ring == 3]], axis=1).mean()
    assert ring1_dist < ring3_dist, "ring 3 should be farther from the hole center than ring 1 on average"


def test_query_local_neighbors_differ_across_well_separated_queries():
    """Directly tests the handoff's "Local neighbours differ across
    suitably separated queries" gate."""
    grid = _square_grid(n=41)
    observed, query = _split_by_circular_hole(grid, radius=15.0)
    # pick two queries far apart within the (large) hole
    far_apart = np.argsort(np.linalg.norm(query - query[0], axis=1))[-1]
    picked_query = query[[0, far_apart]]
    result = extract_boundary_and_local_context(observed, picked_query, k_neighbors=6, local_k=8, max_rings=3)
    neighbors_a = set(result.query_local_neighbor_idx[0].tolist())
    neighbors_b = set(result.query_local_neighbor_idx[1].tolist())
    assert neighbors_a != neighbors_b


def test_query_depth_to_boundary_increases_toward_the_hole_interior():
    grid = _square_grid(n=25)
    observed, query = _split_by_circular_hole(grid, radius=5.0)
    result = extract_boundary_and_local_context(observed, query, k_neighbors=6, local_k=8, max_rings=3)
    dist_from_hole_center = np.linalg.norm(query, axis=1)
    # depth should correlate with distance from the hole's own CENTER (0,0)
    # -- points near the hole center are farthest from the rim (deep),
    # points near dist == radius sit right on the rim (shallow).
    order = np.argsort(dist_from_hole_center)  # ascending: [near center, ..., near rim]
    deep, shallow = order[:5], order[-5:]
    assert result.query_depth_to_boundary[deep].mean() > result.query_depth_to_boundary[shallow].mean()


def test_max_boundary_size_raises_instead_of_silently_truncating():
    """Directly tests the handoff's "Fail closed if a configured safety
    maximum is exceeded; never silently truncate it" requirement."""
    grid = _square_grid(n=25)
    observed, query = _split_by_circular_hole(grid, radius=3.5)
    with pytest.raises(ValueError, match="max_boundary_size"):
        extract_boundary_and_local_context(observed, query, k_neighbors=6, local_k=8, max_rings=3, max_boundary_size=1)


def test_diagnostic_manifest_has_the_phase_2_required_fields():
    """Phase 2 item 6: "Write a diagnostic manifest with total observed
    count, per-query local counts, ring counts, and boundary coverage."""
    grid = _square_grid(n=21)
    observed, query = _split_by_circular_hole(grid, radius=3.0)
    result = extract_boundary_and_local_context(observed, query, k_neighbors=6, local_k=8, max_rings=3)
    for key in (
        "n_observed", "n_query", "n_boundary", "n_boundary_ring_1", "n_boundary_ring_2",
        "n_boundary_ring_3", "local_k_requested", "local_k_effective",
    ):
        assert key in result.diagnostic


def test_local_k_pads_when_fewer_observed_spots_exist_than_requested():
    grid = _square_grid(n=9)
    observed, query = _split_by_circular_hole(grid, radius=1.5)
    result = extract_boundary_and_local_context(observed, query, k_neighbors=6, local_k=1000, max_rings=3)
    assert result.query_local_neighbor_idx.shape[1] == 1000
    assert result.diagnostic["local_k_padded"] is True


def test_empty_observed_or_query_raises():
    grid = _square_grid(n=5)
    with pytest.raises(ValueError, match="observed_coords is empty"):
        extract_boundary_and_local_context(np.zeros((0, 2)), grid)
    with pytest.raises(ValueError, match="query_coords is empty"):
        extract_boundary_and_local_context(grid, np.zeros((0, 2)))


def test_output_integrates_cleanly_into_the_shared_example_object():
    """Proves Phase 1's schema and Phase 2's builder actually fit
    together -- this exact check caught a real Phase 1 design bug
    (coords sizing) before any model code was built on top of it."""
    grid = _square_grid(n=15)
    observed, query = _split_by_circular_hole(grid, radius=2.5)
    result = extract_boundary_and_local_context(observed, query, k_neighbors=6, local_k=6, max_rings=3)

    n_observed, n_query, n_genes = observed.shape[0], query.shape[0], 5
    rng = np.random.default_rng(0)
    inputs = SpatialFieldInputs(
        sample_id="s1", patient_id="p1",
        observed_barcodes=np.array([f"o{i}" for i in range(n_observed)]),
        query_barcodes=np.array([f"q{i}" for i in range(n_query)]),
        observed_coords=observed.astype(np.float32),
        query_coords=query.astype(np.float32),
        observed_full_gene_expression=rng.normal(size=(n_observed, n_genes)).astype(np.float32),
        observed_gigapath_features=rng.normal(size=(n_observed, 1536)).astype(np.float32),
        observed_image_available=np.ones(n_observed, dtype=bool),
        query_local_neighbor_idx=result.query_local_neighbor_idx,
        boundary_idx=result.boundary_idx,
        boundary_ring=result.boundary_ring,
        query_depth_to_boundary=result.query_depth_to_boundary,
    )
    targets = SpatialFieldTargets(query_expression=rng.normal(size=(n_query, n_genes)).astype(np.float32))
    validate_spatial_field_example(inputs, targets)  # must not raise
