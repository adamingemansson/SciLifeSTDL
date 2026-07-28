"""Phase 7: the shared deterministic training objective -- primary MSE
plus the weak spatial-gradient term on query graph edges. Directly tests
the handoff's own framing: "This loss must match gradients rather than
force neighbouring spots to be identical.\""""
import numpy as np
import torch

from gen3_multiscale.models.losses import combined_reconstruction_loss, primary_reconstruction_loss, spatial_gradient_loss


def _grid_query_coords(n=6):
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    return torch.as_tensor(np.stack([xs.ravel(), ys.ravel()], axis=1), dtype=torch.float32)


def test_primary_reconstruction_loss_is_zero_for_a_perfect_prediction():
    target = torch.randn(10, 5)
    assert primary_reconstruction_loss(target.clone(), target).item() == 0.0


def test_primary_reconstruction_loss_rejects_shape_mismatch():
    try:
        primary_reconstruction_loss(torch.randn(3, 4), torch.randn(3, 5))
        assert False, "expected a ValueError"
    except ValueError as exc:
        assert "shape" in str(exc)


def test_primary_reconstruction_loss_gradients_flow():
    pred = torch.randn(4, 3, requires_grad=True)
    target = torch.randn(4, 3)
    loss = primary_reconstruction_loss(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_spatial_gradient_loss_is_zero_for_a_perfect_prediction():
    coords = _grid_query_coords()
    target = torch.randn(coords.shape[0], 5)
    loss = spatial_gradient_loss(target.clone(), target, coords, k_neighbors=4)
    assert loss.item() < 1e-6


def test_spatial_gradient_loss_matches_gradients_not_absolute_levels():
    """The core design claim: a prediction that is uniformly OFFSET from
    the truth by the same constant vector at every query has IDENTICAL
    edge-to-edge differences to the truth, so it must incur (near) zero
    gradient loss even though it incurs large primary reconstruction
    loss -- "match gradients ... not force neighbouring spots to be
    identical.\""""
    coords = _grid_query_coords()
    n_query = coords.shape[0]
    target = torch.randn(n_query, 5)
    offset = torch.tensor([3.0, -2.0, 1.0, 0.5, -1.5])
    predicted = target + offset[None, :]

    gradient_loss = spatial_gradient_loss(predicted, target, coords, k_neighbors=4)
    primary_loss = primary_reconstruction_loss(predicted, target)

    assert gradient_loss.item() < 1e-6
    assert primary_loss.item() > 1.0


def test_spatial_gradient_loss_is_positive_when_edge_differences_disagree():
    coords = _grid_query_coords()
    n_query = coords.shape[0]
    torch.manual_seed(0)
    target = torch.randn(n_query, 5)
    predicted = torch.randn(n_query, 5)  # unrelated to target -- edges will disagree
    loss = spatial_gradient_loss(predicted, target, coords, k_neighbors=4)
    assert loss.item() > 0.1


def test_spatial_gradient_loss_rejects_row_count_mismatch_with_coords():
    coords = _grid_query_coords()
    pred = torch.randn(coords.shape[0] - 1, 3)
    target = torch.randn(coords.shape[0] - 1, 3)
    try:
        spatial_gradient_loss(pred, target, coords, k_neighbors=4)
        assert False, "expected a ValueError"
    except ValueError as exc:
        assert "row" in str(exc)


def test_spatial_gradient_loss_gradients_flow_and_do_not_require_a_scale():
    coords = _grid_query_coords()
    n_query = coords.shape[0]
    pred = torch.randn(n_query, 4, requires_grad=True)
    target = torch.randn(n_query, 4)
    loss = spatial_gradient_loss(pred, target, coords, k_neighbors=4)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_spatial_gradient_loss_accepts_an_explicit_per_gene_scale():
    coords = _grid_query_coords()
    n_query = coords.shape[0]
    torch.manual_seed(1)
    target = torch.randn(n_query, 4)
    predicted = target + torch.randn(n_query, 4) * 0.01
    scale = torch.full((4,), 100.0)  # deliberately huge scale should shrink the loss toward 0
    loss_default = spatial_gradient_loss(predicted, target, coords, k_neighbors=4)
    loss_scaled = spatial_gradient_loss(predicted, target, coords, k_neighbors=4, per_gene_scale=scale)
    assert loss_scaled.item() < loss_default.item()


def test_combined_reconstruction_loss_applies_the_stated_default_weight():
    coords = _grid_query_coords()
    n_query = coords.shape[0]
    torch.manual_seed(2)
    predicted = torch.randn(n_query, 5)
    target = torch.randn(n_query, 5)
    out = combined_reconstruction_loss(predicted, target, coords, k_neighbors=4)
    expected_total = out["primary"] + 0.05 * out["gradient"]
    assert torch.allclose(out["total"], expected_total)


def test_combined_reconstruction_loss_rejects_negative_weight():
    coords = _grid_query_coords()
    n_query = coords.shape[0]
    predicted = torch.randn(n_query, 3)
    target = torch.randn(n_query, 3)
    try:
        combined_reconstruction_loss(predicted, target, coords, gradient_weight=-0.1)
        assert False, "expected a ValueError"
    except ValueError as exc:
        assert "gradient_weight" in str(exc)
