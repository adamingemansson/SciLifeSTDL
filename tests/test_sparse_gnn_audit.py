import torch

from src.models.storm_lite_encoder import _GNNBlock, _knn_edge_index


def test_sparse_knn_edge_index_and_gradient():
    torch.manual_seed(0)
    n, k, d = 37, 5, 16
    coords = torch.randn(n, 3)
    edges = _knn_edge_index(coords, k=k, chunk_size=8)
    assert edges.shape == (2, n * k)
    assert int(edges.min()) >= 0 and int(edges.max()) < n

    # Every receiver has exactly k incoming neighbor entries.
    counts = torch.bincount(edges[0], minlength=n)
    assert torch.equal(counts, torch.full((n,), k, dtype=counts.dtype))

    x = torch.randn(1, n, d, requires_grad=True)
    out = _GNNBlock(d)(x, edges)
    out.square().mean().backward()
    assert out.shape == x.shape
    assert x.grad is not None and torch.isfinite(x.grad).all()
