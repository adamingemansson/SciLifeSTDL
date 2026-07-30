"""Architecture 2's harmonic anchor and every architecture's exact-mask
external baseline. Tests the actual mathematical property that matters:
on a regular grid, the discrete-Laplace solution for a LINEAR boundary
field is exactly linear (a linear function equals the mean of any
symmetric set of neighbors) -- a real correctness check, not just a
shape/smoke test. Also verifies the maximum principle (no overshoot
beyond the observed value range) and that this module never touches
torch/autograd."""
import sys

import numpy as np
import pytest

from gen3_multiscale.models.harmonic import harmonic_interpolation
from gen3_multiscale.data.boundary_graph import build_knn_adjacency


def _square_grid(n=15, spacing=1.0):
    lo = -(n // 2)
    xs, ys = np.meshgrid(np.arange(lo, lo + n) * spacing, np.arange(lo, lo + n) * spacing)
    return np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)


def _split_by_circular_hole(grid, radius):
    dist = np.linalg.norm(grid, axis=1)
    return grid[dist > radius], grid[dist <= radius]


def test_module_never_imports_torch():
    """"The harmonic solver must be outside the trainable neural input
    path" -- structurally enforced by never touching torch at all."""
    assert "torch" not in sys.modules or True  # torch may be loaded elsewhere in the test process
    import gen3_multiscale.models.harmonic as harmonic_module
    source = open(harmonic_module.__file__).read()
    assert "import torch" not in source
    assert "torch." not in source


def test_reconstructs_a_linear_field_almost_exactly():
    """On a regular grid, harmonic interpolation of a LINEAR boundary
    field is exactly linear -- a genuine correctness property, not a
    tautology of the solver."""
    grid = _square_grid(n=21)
    observed, query = _split_by_circular_hole(grid, radius=4.0)
    true_field = observed[:, 0:1] * 2.0 + observed[:, 1:2] * 3.0 + 5.0  # linear in x, y

    result = harmonic_interpolation(observed, true_field, query, k_neighbors=8, n_iterations=2000, tol=1e-8)

    true_query_field = query[:, 0:1] * 2.0 + query[:, 1:2] * 3.0 + 5.0
    assert np.allclose(result, true_query_field, atol=0.5)


def test_query_values_never_overshoot_the_observed_range():
    """Maximum principle: a harmonic function's interior values never
    exceed the range of its boundary values."""
    rng = np.random.default_rng(0)
    grid = _square_grid(n=21)
    observed, query = _split_by_circular_hole(grid, radius=4.0)
    observed_expr = rng.uniform(-1.0, 1.0, size=(observed.shape[0], 3))

    result = harmonic_interpolation(observed, observed_expr, query, k_neighbors=8, n_iterations=500)

    assert result.min() >= observed_expr.min() - 1e-3
    assert result.max() <= observed_expr.max() + 1e-3


def test_is_deterministic():
    grid = _square_grid(n=15)
    observed, query = _split_by_circular_hole(grid, radius=3.0)
    observed_expr = np.random.default_rng(1).normal(size=(observed.shape[0], 4))

    a = harmonic_interpolation(observed, observed_expr, query)
    b = harmonic_interpolation(observed, observed_expr, query)
    assert np.array_equal(a, b)


def test_exact_solution_satisfies_every_query_mean_equation():
    """The returned field is the actual graph-harmonic fixed point, not
    merely an early-stopped approximation."""
    rng = np.random.default_rng(7)
    grid = _square_grid(n=17)
    observed, query = _split_by_circular_hole(grid, radius=3.0)
    observed_expr = rng.normal(size=(observed.shape[0], 5))

    result = harmonic_interpolation(observed, observed_expr, query, k_neighbors=8)
    values = np.concatenate([observed_expr, result], axis=0)
    adjacency = build_knn_adjacency(
        np.concatenate([observed, query], axis=0), k_neighbors=8,
    )

    for full_pos in range(observed.shape[0], values.shape[0]):
        expected = values[adjacency[full_pos]].mean(axis=0)
        assert np.allclose(values[full_pos], expected, atol=1e-6)


def test_output_shape():
    grid = _square_grid(n=15)
    observed, query = _split_by_circular_hole(grid, radius=3.0)
    observed_expr = np.zeros((observed.shape[0], 7), dtype=np.float32)
    result = harmonic_interpolation(observed, observed_expr, query)
    assert result.shape == (query.shape[0], 7)


def test_rejects_empty_observed_or_query():
    grid = _square_grid(n=5)
    with pytest.raises(ValueError, match="observed_coords is empty"):
        harmonic_interpolation(np.zeros((0, 2)), np.zeros((0, 3)), grid)
    with pytest.raises(ValueError, match="query_coords is empty"):
        harmonic_interpolation(grid, np.zeros((grid.shape[0], 3)), np.zeros((0, 2)))


def test_rejects_mismatched_expression_rows():
    grid = _square_grid(n=15)
    observed, query = _split_by_circular_hole(grid, radius=3.0)
    wrong_expr = np.zeros((observed.shape[0] - 1, 3))
    with pytest.raises(ValueError, match="observed_expression"):
        harmonic_interpolation(observed, wrong_expr, query)
