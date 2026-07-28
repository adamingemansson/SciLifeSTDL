import torch

from gen3_multiscale.models.geometry_utils import compute_hole_geometry, compute_relative_geometry, scatter_boundary_ring


def test_compute_relative_geometry_shared_candidates():
    query_coords = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    candidate_coords = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    out = compute_relative_geometry(query_coords, candidate_coords)
    assert out.shape == (2, 2, 3)
    # query 0 -> candidate 0: delta (1,0), distance 1
    assert torch.allclose(out[0, 0], torch.tensor([1.0, 0.0, 1.0]))


def test_compute_relative_geometry_per_query_candidates():
    query_coords = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    candidate_coords = torch.stack([
        torch.tensor([[1.0, 0.0]]),  # query 0's own candidate
        torch.tensor([[10.0, 3.0]]),  # query 1's own candidate
    ])
    out = compute_relative_geometry(query_coords, candidate_coords)
    assert out.shape == (2, 1, 3)
    assert torch.allclose(out[1, 0], torch.tensor([0.0, 3.0, 3.0]))


def test_compute_relative_geometry_rejects_bad_shapes():
    query_coords = torch.zeros(3, 2)
    try:
        compute_relative_geometry(query_coords, torch.zeros(5, 2, 2))
        assert False, "expected a ValueError"
    except ValueError as exc:
        assert "candidate_coords" in str(exc)
    try:
        compute_relative_geometry(query_coords, torch.zeros(4))
        assert False, "expected a ValueError"
    except ValueError as exc:
        assert "2-D or 3-D" in str(exc)


def test_compute_hole_geometry_shape_and_broadcast():
    query_coords = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    out = compute_hole_geometry(query_coords)
    assert out.shape == (3, 2)
    # "area" (count-derived) feature is identical across all queries in the item
    assert torch.allclose(out[:, 0], out[0, 0].expand(3))


def test_compute_hole_geometry_distance_increases_away_from_centroid():
    query_coords = torch.tensor([[0.0, 0.0], [5.0, 0.0]])  # centroid at (2.5, 0)
    out = compute_hole_geometry(query_coords)
    # both points are equidistant from the centroid here -- use an asymmetric case instead
    query_coords2 = torch.tensor([[0.0, 0.0], [0.0, 0.0], [10.0, 0.0]])
    out2 = compute_hole_geometry(query_coords2)
    assert out2[2, 1] > out2[0, 1]


def test_scatter_boundary_ring_places_values_at_the_right_positions():
    out = scatter_boundary_ring(n_observed=5, boundary_idx=torch.tensor([1, 3]), boundary_ring=torch.tensor([1, 2]))
    assert out.tolist() == [0, 1, 0, 2, 0]


def test_scatter_boundary_ring_handles_empty_boundary():
    out = scatter_boundary_ring(n_observed=4, boundary_idx=torch.tensor([], dtype=torch.long), boundary_ring=torch.tensor([], dtype=torch.long))
    assert out.tolist() == [0, 0, 0, 0]
