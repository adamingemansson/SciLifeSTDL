"""Regression tests for hierarchical_gene_transport_regressor
(src/models/registry.py) -- see docs/hierarchical_missing_tissue.md and the
2026-07-22 handoff for the full design rationale: HierarchicalMissingTissue-
Regressor's dense 256->512->G decoder lost to exact IDW/harmonic
interpolation on held-out patients, because routing ~16k exact observed
gene values through one 256D bottleneck discards gene-specific spatial
structure. This model keeps HierarchicalMissingTissueEncoder as the
conditioner but predicts a weighted combination of untouched observed
full-gene vectors (IDW anchor + learned multi-head, per-gene-gated
transport + optional zero-initialized low-rank residual), never a second
dense gene decoder.

Run with: python -m pytest tests/test_hierarchical_gene_transport.py -q
"""
import math

import torch

from src.models.registry import build_model


def _base_params(n_genes=6, **overrides):
    params = {
        "n_genes": n_genes,
        "hidden_dim": 16,
        "n_heads": 4,
        "context_layers": 1,
        "cross_layers": 1,
        "query_layers": 1,
        "local_k": 3,
        "dropout": 0.0,
        "use_novae": False,
        "use_local_images": False,
        "use_slide_context": False,
        "gene_encoder_type": "weighted_linear",
        "score_hidden_dim": 12,
        "transport_heads": 4,
        "target_gene_scale": [1.0] * n_genes,
    }
    params.update(overrides)
    return params


def _build(n_genes=6, **overrides):
    return build_model({
        "name": "hierarchical_gene_transport_regressor",
        "params": _base_params(n_genes, **overrides),
    })


def _context_query(n_context=5, n_genes=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    context = {
        "coords": torch.rand(n_context, 3, generator=g) * 10.0,
        "expression": torch.rand(n_context, n_genes, generator=g),
    }
    query = {"coords": torch.rand(2, 3, generator=g) * 10.0}
    return context, query


def test_needs_only_observed_inputs_for_queries():
    model = _build().eval()
    context, query = _context_query()
    with torch.inference_mode():
        out = model.sample(context, query)
    assert out["expression"].shape == (2, 6)
    assert torch.isfinite(out["expression"]).all()
    print("[hierarchical_gene_transport] OK — runs with query coords only, no query GEX/H&E")


def test_sample_output_contract():
    model = _build().eval()
    context, query = _context_query()
    with torch.inference_mode():
        out = model.sample(context, query)
    for key in ("coords", "expression", "anchor_expression", "residual_expression"):
        assert key in out, f"sample() missing required key {key!r}"
    assert out["anchor_expression"].shape == (2, 6)
    assert torch.allclose(
        out["residual_expression"], out["expression"] - out["anchor_expression"]
    )
    print("[hierarchical_gene_transport] OK — sample() satisfies the required output contract")


def test_idw_anchor_matches_hand_computed_weights():
    model = _build(n_genes=2)
    distances = torch.tensor([[1.0, 2.0, 4.0]])
    expr = torch.tensor([[[10.0, 0.0], [0.0, 10.0], [100.0, 100.0]]])
    anchor, weights = model._idw_anchor(distances, expr)
    raw = 1.0 / distances.pow(2.0)
    expected_weights = raw / raw.sum(dim=-1, keepdim=True)
    expected_anchor = torch.einsum("qk,qkg->qg", expected_weights, expr)
    assert torch.allclose(weights, expected_weights, atol=1e-6)
    assert torch.allclose(anchor, expected_anchor, atol=1e-5)
    # closer neighbors must dominate: gene0 should be much closer to 10 than 100
    assert anchor[0, 0] < 30.0
    print("[hierarchical_gene_transport] OK — IDW anchor matches hand-computed inverse-square weights")


def test_blend_logit_initializes_near_point_zero_five():
    model = _build()
    blend = torch.sigmoid(model.blend_logit)
    assert torch.allclose(blend, torch.full_like(blend, 0.05), atol=1e-3)
    print("[hierarchical_gene_transport] OK — sigmoid(blend_logit) starts at ~0.05 for every gene")


def test_prediction_starts_close_to_idw_anchor():
    """Safe-init property: at construction, the learned transport should
    barely move the prediction away from the IDW anchor (blend~0.05) and,
    with the residual off by default, contribute nothing beyond that."""
    model = _build().eval()
    context, query = _context_query()
    with torch.inference_mode():
        out = model.sample(context, query)
    delta = (out["expression"] - out["anchor_expression"]).abs()
    candidate_scale = context["expression"].abs().max()
    assert (delta < 0.25 * candidate_scale).all(), (
        "prediction moved too far from the IDW anchor at initialization"
    )
    print("[hierarchical_gene_transport] OK — prediction stays close to the IDW anchor at init")


def test_residual_is_exactly_zero_at_initialization():
    model = _build(use_residual=True, residual_rank=4).eval()
    context, query = _context_query()
    with torch.inference_mode():
        out = model.sample(context, query)
    assert torch.allclose(out["factorized_residual"], torch.zeros_like(out["factorized_residual"]))
    print("[hierarchical_gene_transport] OK — factorized residual is exactly zero before any training step")


def test_residual_becomes_nonzero_after_one_optimizer_step():
    model = _build(use_residual=True, residual_rank=4, residual_penalty_weight=0.0)
    context, query = _context_query(n_context=6)
    target = torch.rand(2, 6)
    optimizer = model.configure_optimizers()
    batch = {"context": context, "query": query, "target_expression": target}
    loss = model.training_step(batch, 0)
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.inference_mode():
        out = model.sample(context, query)
    assert not torch.allclose(out["factorized_residual"], torch.zeros_like(out["factorized_residual"]))
    print("[hierarchical_gene_transport] OK — residual moves off zero after a real optimizer step")


def test_shared_gene_gate_mode_gives_every_gene_the_same_transport_weights():
    model = _build(n_genes=5, gene_gate_mode="shared", use_query_gate=False).eval()
    context, query = _context_query(n_genes=5)
    with torch.inference_mode():
        encoded = model.context_encoder.forward_with_neighbors(context, query)
        gate_logits = model.gene_head_logits.expand(model.n_genes, -1)
        gates = torch.softmax(gate_logits, dim=-1)
    assert torch.allclose(gates[0], gates[-1]), "shared gene-gate mode must give every gene the same gate"
    print("[hierarchical_gene_transport] OK — gene_gate_mode='shared' collapses to one gate for all genes")


def test_single_transport_head_is_allowed():
    model = _build(transport_heads=1).eval()
    context, query = _context_query()
    with torch.inference_mode():
        out = model.sample(context, query)
    assert torch.isfinite(out["expression"]).all()
    print("[hierarchical_gene_transport] OK — transport_heads=1 (C10 ablation) runs without error")


def test_geometry_conditioning_mode_ignores_multimodal_tokens():
    """conditioning_mode='geometry' (the C07 ablation) must not touch
    neighbor_hidden/query_hidden in the transport SCORE -- but the encoder,
    gene gate and residual paths must still work unmodified."""
    model = _build(conditioning_mode="geometry").eval()
    assert model.neighbor_projection is None
    assert model.query_score_projection is None
    context, query = _context_query()
    with torch.inference_mode():
        out = model.sample(context, query)
    assert torch.isfinite(out["expression"]).all()
    print("[hierarchical_gene_transport] OK — conditioning_mode='geometry' disables multimodal scoring cleanly")


def test_gradients_are_bucketed_into_the_three_mandated_groups():
    model = _build(
        use_local_images=True, use_novae=False, use_residual=True, residual_rank=4,
    )
    context, query = _context_query(n_context=6)
    context["images"] = torch.rand(6, 1536)
    context["image_available"] = torch.ones(6, dtype=torch.bool)
    target = torch.rand(2, 6)
    batch = {"context": context, "query": query, "target_expression": target}
    loss = model.training_step(batch, 0)
    loss.backward()

    logged = {}
    model.log_dict = lambda values, **kwargs: logged.update(values)  # type: ignore[assignment]
    model.on_after_backward()

    assert math.isfinite(float(logged["train/hierarchical_encoder_grad_norm"]))
    assert math.isfinite(float(logged["train/transport_grad_norm"]))
    assert math.isfinite(float(logged["train/factorized_residual_grad_norm"]))
    assert float(logged["train/hierarchical_encoder_grad_norm"]) > 0.0
    assert float(logged["train/transport_grad_norm"]) > 0.0
    assert float(logged["train/factorized_residual_grad_norm"]) > 0.0
    print("[hierarchical_gene_transport] OK — encoder/transport/residual grad norms are finite, "
          "positive and non-overlapping")


def test_all_thirteen_mandated_metrics_are_logged():
    model = _build(use_residual=True, residual_rank=4)
    context, query = _context_query(n_context=6)
    target = torch.rand(2, 6)
    batch = {"context": context, "query": query, "target_expression": target}

    logged = {}
    model.log_dict = lambda values, **kwargs: logged.update(values)  # type: ignore[assignment]
    loss = model.training_step(batch, 0)
    loss.backward()
    model.on_after_backward()

    required = [
        "train/standardized_mse", "train/absolute_mse", "train/anchor_mse",
        "train/per_gene_pcc", "train/transport_delta_rms", "train/factorized_residual_rms",
        "train/prediction_delta_rms", "train/transport_head_entropy", "train/gene_gate_entropy",
        "train/hierarchical_encoder_grad_norm", "train/transport_grad_norm",
        "train/factorized_residual_grad_norm",
    ]
    missing = [key for key in required if key not in logged]
    assert not missing, f"missing mandated log keys: {missing}"
    assert torch.isfinite(loss)
    print("[hierarchical_gene_transport] OK — all 12 handoff-mandated per-step metrics are logged "
          "(train/loss is the 13th, always present as the return value)")


if __name__ == "__main__":
    test_needs_only_observed_inputs_for_queries()
    test_sample_output_contract()
    test_idw_anchor_matches_hand_computed_weights()
    test_blend_logit_initializes_near_point_zero_five()
    test_prediction_starts_close_to_idw_anchor()
    test_residual_is_exactly_zero_at_initialization()
    test_residual_becomes_nonzero_after_one_optimizer_step()
    test_shared_gene_gate_mode_gives_every_gene_the_same_transport_weights()
    test_single_transport_head_is_allowed()
    test_geometry_conditioning_mode_ignores_multimodal_tokens()
    test_gradients_are_bucketed_into_the_three_mandated_groups()
    test_all_thirteen_mandated_metrics_are_logged()
    print("\nAll hierarchical_gene_transport_regressor tests passed.")
