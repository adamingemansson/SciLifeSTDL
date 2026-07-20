import torch

from src.models.spatial_baselines import (
    global_mean_interpolate,
    nearest_interpolate,
    local_mean_interpolate,
    idw_interpolate,
    harmonic_interpolate,
)


def _line_problem():
    coords = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    expr = torch.tensor([[0.0, 2.0], [2.0, 4.0]])
    query = torch.tensor([[1.0, 0.0]])
    return coords, expr, query


def test_baselines_have_expected_shapes_and_simple_values():
    coords, expr, query = _line_problem()
    expected = torch.tensor([[1.0, 3.0]])
    assert torch.allclose(global_mean_interpolate(coords, expr, query), expected)
    assert nearest_interpolate(coords, expr, query).shape == expected.shape
    assert torch.allclose(local_mean_interpolate(coords, expr, query, k=2), expected)
    assert torch.allclose(idw_interpolate(coords, expr, query, k=2), expected, atol=1e-5)
    harmonic = harmonic_interpolate(coords, expr, query, k=2)
    assert harmonic.shape == expected.shape
    assert torch.isfinite(harmonic).all()
    assert torch.allclose(harmonic, expected, atol=2e-3)


def test_harmonic_anchor_is_differentiable_in_context_expression():
    coords, expr, query = _line_problem()
    expr.requires_grad_(True)
    harmonic_interpolate(coords, expr, query, k=2).sum().backward()
    assert expr.grad is not None
    assert torch.isfinite(expr.grad).all()
