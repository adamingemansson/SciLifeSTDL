"""
Smoke test for the conditioning encoder (src/models/conditioning.py) —
tiny synthetic data, no real dataset needed. Checks shapes, no crashes, no
NaNs, and that it handles both Track A (2D) and Track B (3D) coordinate
dimensionality with the same module.

Run with: python tests/test_conditioning.py
(needs the st3d conda env active — docs/environment_setup.md)
"""
import torch

from src.models.conditioning import SpatialContextEncoder


def _run_case(coord_dim: int, n_context: int, n_query: int, n_genes: int, label: str):
    torch.manual_seed(0)
    encoder = SpatialContextEncoder(n_genes=n_genes, coord_dim=coord_dim, hidden_dim=64,
                                     n_message_layers=2, k_neighbors=5, rff_features=16)

    context_coords = torch.randn(n_context, coord_dim)
    context_expression = torch.rand(n_context, n_genes)
    query_coords = torch.randn(n_query, coord_dim)

    c = encoder(context_coords, context_expression, query_coords)

    assert c.shape == (n_query, 64), f"[{label}] expected shape ({n_query}, 64), got {tuple(c.shape)}"
    assert torch.isfinite(c).all(), f"[{label}] output contains NaN/Inf"
    print(f"[{label}] OK — output shape {tuple(c.shape)}, "
          f"mean={c.mean().item():.4f}, std={c.std().item():.4f}")


def _run_edge_cases():
    # fewer context points than k_neighbors — _knn_indices should clip k, not crash
    encoder = SpatialContextEncoder(n_genes=10, coord_dim=2, hidden_dim=32, k_neighbors=20)
    context_coords = torch.randn(3, 2)
    context_expression = torch.rand(3, 10)
    query_coords = torch.randn(2, 2)
    c = encoder(context_coords, context_expression, query_coords)
    assert c.shape == (2, 32)
    assert torch.isfinite(c).all()
    print("[edge: fewer context points than k] OK")

    # single query point
    encoder2 = SpatialContextEncoder(n_genes=10, coord_dim=3, hidden_dim=32, k_neighbors=5)
    c2 = encoder2(torch.randn(20, 3), torch.rand(20, 10), torch.randn(1, 3))
    assert c2.shape == (1, 32)
    print("[edge: single query point] OK")


if __name__ == "__main__":
    _run_case(coord_dim=2, n_context=200, n_query=30, n_genes=50, label="Track A (2D, intra-slice)")
    _run_case(coord_dim=3, n_context=500, n_query=80, n_genes=2000, label="Track B (3D, inter-slice)")
    _run_edge_cases()
    print("\nAll conditioning encoder smoke tests passed.")
