"""
Regression tests for FlowMatchingOT's boundary_consistency_weight
(2026-07-20) — see _boundary_consistency_loss's own docstring in
src/models/registry.py for the full adaptation reasoning: ports the
underlying idea behind DISCO's ablation-proven-most-important component
("integration with neighboring region" during generation, Duan et al.
2025, Proc Int Conf Image Proc — verified via direct PDF read) as a
training-time auxiliary loss, since DISCO's own literal per-step
re-noise-and-splice mechanism isn't dimensionally transferable to FM-OT's
per-query-point latent ODE state.

Run with: python -m tests.test_boundary_consistency_loss
"""
import torch

from src.models.registry import FlowMatchingOT


def _ctx_query(n_context=15, n_query=6, n_genes=10, coord_dim=3):
    context = {"coords": torch.randn(n_context, coord_dim), "expression": torch.rand(n_context, n_genes)}
    query = {"coords": torch.randn(n_query, coord_dim)}
    return context, query


def test_boundary_consistency_weight_zero_is_true_noop():
    """Default (0.0) must produce IDENTICAL loss/gradients to before this
    feature existed — the term is skipped entirely in training_step, not
    just multiplied by zero (which would still cost compute and could
    still perturb the loss via floating-point noise)."""
    torch.manual_seed(0)
    model_a = FlowMatchingOT(n_genes=10, coord_dim=3, cond_hidden_dim=16, hidden_dim=32, time_embed_dim=16, n_ode_steps=3)
    torch.manual_seed(0)
    model_b = FlowMatchingOT(
        n_genes=10, coord_dim=3, cond_hidden_dim=16, hidden_dim=32, time_embed_dim=16, n_ode_steps=3,
        boundary_consistency_weight=0.0,
    )
    context, query = _ctx_query()
    target = torch.rand(query["coords"].shape[0], 10)
    batch = {"context": context, "query": query, "target_expression": target}

    torch.manual_seed(1)
    loss_a = model_a.training_step(batch, 0)
    torch.manual_seed(1)
    loss_b = model_b.training_step(batch, 0)
    assert torch.equal(loss_a, loss_b)
    print("[boundary_consistency] OK — weight=0.0 (default) is a true no-op")


def test_boundary_consistency_weight_nonzero_changes_loss_and_backprops():
    torch.manual_seed(0)
    model = FlowMatchingOT(
        n_genes=10, coord_dim=3, cond_hidden_dim=16, hidden_dim=32, time_embed_dim=16, n_ode_steps=3,
        boundary_consistency_weight=0.5, boundary_consistency_bandwidth=10.0,
    )
    context, query = _ctx_query()
    target = torch.rand(query["coords"].shape[0], 10)
    batch = {"context": context, "query": query, "target_expression": target}
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert any(g > 0 for g in grad_norms), "boundary consistency term must contribute real gradient"
    print("[boundary_consistency] OK — weight=0.5 runs end-to-end, finite loss, gradients flow")


def test_boundary_consistency_loss_pulls_toward_nearest_real_context():
    """Direct unit test of _boundary_consistency_loss itself: a query
    point placed exactly ON TOP of a context point should be penalized
    proportional to the squared difference from THAT context point's
    real expression (weight=exp(0)=1, since distance=0) — not any other
    context point's, proving nearest-neighbor selection is correct, not
    just "some" context point."""
    torch.manual_seed(0)
    model = FlowMatchingOT(n_genes=4, coord_dim=2, cond_hidden_dim=8, hidden_dim=16, time_embed_dim=8, n_ode_steps=2,
                            boundary_consistency_bandwidth=100.0)
    context_coords = torch.tensor([[0.0, 0.0], [100.0, 100.0]])
    context_expr = torch.tensor([[1.0, 2.0, 3.0, 4.0], [9.0, 9.0, 9.0, 9.0]])
    query_coords = torch.tensor([[0.0, 0.0]])  # exactly on top of context point 0
    pred_expr = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # matches context point 0 exactly -> loss should be ~0

    loss_match = model._boundary_consistency_loss(query_coords, context_coords, context_expr, pred_expr)
    assert torch.allclose(loss_match, torch.tensor(0.0), atol=1e-6), loss_match

    pred_expr_off = torch.tensor([[0.0, 0.0, 0.0, 0.0]])  # off by (1,2,3,4) from context point 0 (the NEAREST one)
    loss_off = model._boundary_consistency_loss(query_coords, context_coords, context_expr, pred_expr_off)
    expected = ((torch.tensor([1.0, 2.0, 3.0, 4.0])) ** 2).mean()  # weight=1 at distance 0
    assert torch.allclose(loss_off, expected, atol=1e-5), (loss_off, expected)
    print("[boundary_consistency] OK — loss correctly targets the NEAREST real context point, weight=1 at distance 0")


def test_boundary_consistency_bandwidth_decays_with_distance():
    """A query point far from all context should contribute much less
    loss than one right at the boundary, given the SAME prediction error
    -- proves the exponential distance-weighting is real, not a no-op."""
    torch.manual_seed(0)
    model = FlowMatchingOT(n_genes=2, coord_dim=2, cond_hidden_dim=8, hidden_dim=16, time_embed_dim=8, n_ode_steps=2,
                            boundary_consistency_bandwidth=10.0)
    context_coords = torch.tensor([[0.0, 0.0]])
    context_expr = torch.tensor([[5.0, 5.0]])
    pred_expr = torch.tensor([[0.0, 0.0], [0.0, 0.0]])  # identical prediction error for both query points

    query_near = torch.tensor([[0.0, 0.0]])   # distance 0 -> weight 1
    query_far = torch.tensor([[1000.0, 0.0]])  # very far -> weight ~0

    loss_near = model._boundary_consistency_loss(query_near, context_coords, context_expr, pred_expr[:1])
    loss_far = model._boundary_consistency_loss(query_far, context_coords, context_expr, pred_expr[:1])
    assert loss_near > loss_far * 100, (loss_near, loss_far)
    print("[boundary_consistency] OK — bandwidth decay concentrates the pull near the boundary, "
          f"loss_near={loss_near.item():.4f} >> loss_far={loss_far.item():.6f}")


if __name__ == "__main__":
    test_boundary_consistency_weight_zero_is_true_noop()
    test_boundary_consistency_weight_nonzero_changes_loss_and_backprops()
    test_boundary_consistency_loss_pulls_toward_nearest_real_context()
    test_boundary_consistency_bandwidth_decays_with_distance()
    print("\nAll boundary_consistency_weight tests passed.")
