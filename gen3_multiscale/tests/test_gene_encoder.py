"""Verbatim-copy tests for WeightedGeneExpressionEncoder (Codex audit
finding #7's fix) -- gen3_multiscale's own copy, independently verified
rather than imported, matching this project's established
copy-provenance discipline (e.g. slide_context.py's tests)."""
import torch

from gen3_multiscale.models.gene_encoder import WeightedGeneExpressionEncoder


def test_output_shape():
    encoder = WeightedGeneExpressionEncoder(n_genes=10, output_dim=6)
    out = encoder(torch.randn(4, 10))
    assert out.shape == (4, 6)


def test_rejects_non_2d_input():
    encoder = WeightedGeneExpressionEncoder(n_genes=10, output_dim=6)
    try:
        encoder(torch.randn(4, 10, 1))
        assert False, "expected a ValueError"
    except ValueError as exc:
        assert "[N,G]" in str(exc)


def test_is_exactly_linear_a_weighted_bag_of_gene_embeddings():
    """"x @ W is exactly a weighted bag of gene embeddings" -- verified
    directly: output for a sum of two expression vectors must equal the
    sum of their individual outputs (linearity), and the module holds no
    bias term."""
    encoder = WeightedGeneExpressionEncoder(n_genes=5, output_dim=3)
    assert encoder.projection.bias is None
    a, b = torch.randn(2, 5), torch.randn(2, 5)
    assert torch.allclose(encoder(a) + encoder(b), encoder(a + b), atol=1e-5)


def test_gradients_flow_to_the_projection_weight():
    encoder = WeightedGeneExpressionEncoder(n_genes=5, output_dim=3)
    out = encoder(torch.randn(4, 5, requires_grad=True))
    out.sum().backward()
    assert encoder.projection.weight.grad is not None
    assert torch.isfinite(encoder.projection.weight.grad).all()
