"""Phase 5 items 3-5 (multiscale spatial-field handoff): relative-geometry
attention bias, chunked boundary cross-attention (numerical correctness
vs an unchunked reference, and fail-closed on an oversized context), and
query-query dense/sparse switching at the handoff's own named threshold."""
import torch

from gen3_multiscale.models.attention import ChunkedCrossAttention, QueryQuerySelfAttention, RelativeGeometryBias


def _relative_geometry(query_coords, context_coords):
    delta = context_coords[None, :, :] - query_coords[:, None, :]
    distance = torch.linalg.norm(delta, dim=-1, keepdim=True)
    return torch.cat([delta, distance], dim=-1)


def test_relative_geometry_bias_output_shape():
    bias = RelativeGeometryBias(n_heads=4)
    out = bias(torch.randn(5, 6, 3))
    assert out.shape == (5, 6, 4)


class TestChunkedCrossAttention:
    def test_output_shape(self):
        attn = ChunkedCrossAttention(hidden_dim=32, n_heads=4, chunk_size=8)
        query_hidden = torch.randn(5, 32)
        context_hidden = torch.randn(20, 32)
        geometry = _relative_geometry(torch.randn(5, 2), torch.randn(20, 2))
        out = attn(query_hidden, context_hidden, geometry)
        assert out.shape == (5, 32)
        assert torch.isfinite(out).all()

    def test_chunked_output_matches_a_single_large_chunk_regardless_of_chunk_size(self):
        """The online-softmax recurrence must be mathematically exact --
        splitting the SAME context into different chunk sizes must give
        numerically identical output (up to float precision), never an
        approximation. chunk_size >= n_context reduces to one chunk (the
        direct/"unchunked" computation), which this test uses as its own
        reference rather than a separate hand-written implementation."""
        torch.manual_seed(0)
        attn = ChunkedCrossAttention(hidden_dim=32, n_heads=4, chunk_size=1000)
        query_hidden = torch.randn(6, 32)
        context_hidden = torch.randn(37, 32)  # deliberately doesn't divide evenly
        geometry = _relative_geometry(torch.randn(6, 2), torch.randn(37, 2))

        reference = attn(query_hidden, context_hidden, geometry)
        for chunk_size in (1, 3, 10, 37, 100):
            attn.chunk_size = chunk_size
            out = attn(query_hidden, context_hidden, geometry)
            assert torch.allclose(out, reference, atol=1e-4), f"mismatch at chunk_size={chunk_size}"

    def test_max_context_size_raises_instead_of_truncating(self):
        attn = ChunkedCrossAttention(hidden_dim=16, n_heads=2, chunk_size=4, max_context_size=10)
        query_hidden = torch.randn(3, 16)
        context_hidden = torch.randn(20, 16)
        geometry = _relative_geometry(torch.randn(3, 2), torch.randn(20, 2))
        try:
            attn(query_hidden, context_hidden, geometry)
            assert False, "expected a ValueError"
        except ValueError as exc:
            assert "max_context_size" in str(exc)

    def test_gradients_flow_through_query_context_and_geometry(self):
        attn = ChunkedCrossAttention(hidden_dim=16, n_heads=2, chunk_size=3)
        query_hidden = torch.randn(4, 16, requires_grad=True)
        context_hidden = torch.randn(11, 16, requires_grad=True)
        geometry = _relative_geometry(torch.randn(4, 2), torch.randn(11, 2)).requires_grad_(True)
        out = attn(query_hidden, context_hidden, geometry)
        out.sum().backward()
        assert query_hidden.grad is not None and torch.isfinite(query_hidden.grad).all()
        assert context_hidden.grad is not None and torch.isfinite(context_hidden.grad).all()
        assert geometry.grad is not None and torch.isfinite(geometry.grad).all()

    def test_output_is_invariant_to_context_ordering(self):
        """"Context and boundary processing is permutation-equivariant/
        invariant; arbitrary barcode or file order cannot become a
        positional cue" -- permuting the context set (and its matching
        geometry) together must not change the aggregate output."""
        torch.manual_seed(1)
        attn = ChunkedCrossAttention(hidden_dim=16, n_heads=2, chunk_size=5)
        query_hidden = torch.randn(3, 16)
        context_hidden = torch.randn(13, 16)
        geometry = _relative_geometry(torch.randn(3, 2), torch.randn(13, 2))

        out_original = attn(query_hidden, context_hidden, geometry)
        perm = torch.randperm(13)
        out_permuted = attn(query_hidden, context_hidden[perm], geometry[:, perm])
        assert torch.allclose(out_original, out_permuted, atol=1e-5)


class TestQueryQuerySelfAttention:
    def test_uses_dense_mode_at_or_below_the_threshold(self):
        attn = QueryQuerySelfAttention(hidden_dim=16, n_heads=2, dense_threshold=10)
        out, mode = attn(torch.randn(10, 16), torch.randn(10, 2))
        assert mode == "dense"
        assert out.shape == (10, 16)

    def test_uses_sparse_mode_above_the_threshold(self):
        attn = QueryQuerySelfAttention(hidden_dim=16, n_heads=2, dense_threshold=10, sparse_k=4)
        out, mode = attn(torch.randn(30, 16), torch.randn(30, 2))
        assert mode == "sparse"
        assert out.shape == (30, 16)

    def test_matches_the_handoffs_own_default_threshold_of_256(self):
        attn = QueryQuerySelfAttention(hidden_dim=16, n_heads=2)
        assert attn.dense_threshold == 256

    def test_gradients_flow_in_both_modes(self):
        for n_query, expected_mode in ((8, "dense"), (30, "sparse")):
            attn = QueryQuerySelfAttention(hidden_dim=16, n_heads=2, dense_threshold=10, sparse_k=4)
            query_hidden = torch.randn(n_query, 16, requires_grad=True)
            query_coords = torch.randn(n_query, 2)
            out, mode = attn(query_hidden, query_coords)
            assert mode == expected_mode
            out.sum().backward()
            assert query_hidden.grad is not None and torch.isfinite(query_hidden.grad).all()

    def test_finite_output_for_a_single_query(self):
        attn = QueryQuerySelfAttention(hidden_dim=16, n_heads=2, dense_threshold=10)
        out, mode = attn(torch.randn(1, 16), torch.randn(1, 2))
        assert mode == "dense"
        assert torch.isfinite(out).all()
