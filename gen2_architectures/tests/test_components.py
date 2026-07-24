import torch

from gen2_architectures.models.components import (
    CoordEmbedding, ConfidenceEmbedding, build_local_transformer, StagedGeneLoss,
)


def test_coord_embedding_shape_and_gradient():
    ce = CoordEmbedding(feat_dim=64, num_fourier_features=32, coord_scale=1000.0)
    xy = (torch.randn(10, 2) * 500).requires_grad_(True)  # leaf tensor, so .grad actually populates
    out = ce(xy)
    assert out.shape == (10, 64)
    out.sum().backward()
    assert xy.grad is not None and xy.grad.abs().sum() > 0


def test_confidence_embedding_shape():
    conf = ConfidenceEmbedding(embed_dim=16)
    out = conf(torch.tensor([True, False, True]))
    assert out.shape == (3, 16)
    assert not torch.equal(out[0], out[1]), "observed vs masked must produce different embeddings"


def test_local_transformer_shape():
    tf = build_local_transformer(hidden_dim=64, n_layers=2, n_heads=4)
    tokens = torch.randn(2, 10, 64)
    out = tf(tokens)
    assert out.shape == (2, 10, 64)


def test_staged_loss_schedule_breakpoints():
    loss_fn = StagedGeneLoss(stage1_end=0.2, stage2_end=0.6, stage2_pearson_weight=0.1, stage3_pearson_weight=0.3)
    pred = torch.randn(32, 50, requires_grad=True)
    target = torch.randn(32, 50)
    expectations = [(0.0, 0.0), (0.1, 0.0), (0.19, 0.0), (0.2, 0.1), (0.5, 0.1), (0.6, 0.3), (1.0, 0.3)]
    for progress, expected_weight in expectations:
        result = loss_fn(pred, target, progress)
        assert result["pearson_weight"] == expected_weight, (progress, result["pearson_weight"])


def test_staged_loss_gradient_flows():
    loss_fn = StagedGeneLoss()
    pred = torch.randn(32, 50, requires_grad=True)
    target = torch.randn(32, 50)
    result = loss_fn(pred, target, progress=1.0)
    result["loss"].backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0


def test_staged_loss_pearson_penalty_zero_for_perfect_correlation():
    loss_fn = StagedGeneLoss()
    target = torch.randn(32, 50)
    pred = target.clone().requires_grad_(True)
    result = loss_fn(pred, target, progress=1.0)
    assert result["pearson_penalty"].item() < 1e-4


def test_staged_loss_single_spot_falls_back_to_mse():
    """Pearson is undefined for a single observation -- must not crash or NaN."""
    loss_fn = StagedGeneLoss()
    pred = torch.randn(1, 50, requires_grad=True)
    target = torch.randn(1, 50)
    result = loss_fn(pred, target, progress=1.0)
    assert result["pearson_weight"] == 0.0
    assert torch.isfinite(result["loss"])
