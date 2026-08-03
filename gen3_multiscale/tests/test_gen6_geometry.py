import torch

from gen3_multiscale.gen6.geometry import Gen6QuerySelfAttention


def test_relative_and_fourier_bias_are_coordinate_scale_invariant():
    coords = torch.tensor([[0.0, 0.0], [2.0, 1.0], [4.0, -1.0]])
    for mode in ("relative_bias", "fourier_attention"):
        torch.manual_seed(2)
        attention = Gen6QuerySelfAttention(
            hidden_dim=12, n_heads=3, dense_threshold=10, sparse_k=2, geometry_mode=mode,
        )
        torch.testing.assert_close(
            attention._full_bias(coords), attention._full_bias(coords * 10_000),
            rtol=1e-3, atol=1e-4,
        )


def test_auto_scaled_frame_bias_is_translation_and_scale_invariant():
    coords = torch.tensor([[0.0, 0.0], [2.0, 1.0], [4.0, -1.0], [3.0, 3.0]])
    torch.manual_seed(3)
    attention = Gen6QuerySelfAttention(
        hidden_dim=12, n_heads=3, dense_threshold=10, sparse_k=2,
        geometry_mode="frame_averaging", coord_scale=0.0,
    )
    first = attention._full_bias(coords)
    transformed = attention._full_bias(coords * 5000 + torch.tensor([12000.0, -9000.0]))
    torch.testing.assert_close(first, transformed, rtol=2e-4, atol=2e-4)


def test_sparse_geometry_never_materializes_full_pairwise_bias():
    hidden = torch.randn(12, 12)
    coords = torch.randn(12, 2)
    for mode in ("relative_bias", "fourier_attention", "frame_averaging"):
        attention = Gen6QuerySelfAttention(
            hidden_dim=12, n_heads=3, dense_threshold=4, sparse_k=3,
            geometry_mode=mode, coord_scale=0.0,
        )

        def forbidden(_):
            raise AssertionError("sparse path materialized full N-by-N bias")

        attention._full_bias = forbidden
        output, path = attention(hidden, coords)
        assert path == "sparse"
        assert output.shape == hidden.shape
        assert torch.isfinite(output).all()
