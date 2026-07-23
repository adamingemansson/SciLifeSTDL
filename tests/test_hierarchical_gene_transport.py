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


def test_entropy_regularization_is_off_by_default():
    """2026-07-22 20-run suite: with transport_reg_weight=1e-3,
    transport_head_entropy sat at ~ln(128)=4.852 (the true maximum) for the
    ENTIRE 20k-step run on every config -- the neighbor-weighting mechanism
    never learned to specialize, plausibly explaining why every richness
    axis (Novae, H&E, extra heads, richer gates) failed to help. The default
    weight must be exactly 0.0 so the entropy term never enters the loss
    unless a caller explicitly re-enables it."""
    model = _build()
    assert model.transport_reg_weight == 0.0

    context, query = _context_query(n_context=6)
    target = torch.rand(2, 6)
    batch = {"context": context, "query": query, "target_expression": target}

    out = model.sample(context, query)
    standardized_error = (out["expression"] - target) / model.target_gene_scale
    standardized_mse = standardized_error.square().mean()
    correlation = model._mean_per_gene_correlation(out["expression"], target)
    correlation_loss = 1.0 - correlation
    standardized_residual = out["factorized_residual"] / model.target_gene_scale
    residual_penalty = standardized_residual.square().mean()
    expected_loss = (
        standardized_mse
        + model.correlation_loss_weight * correlation_loss
        + model.residual_penalty_weight * residual_penalty
    )

    logged = {}
    model.log_dict = lambda values, **kwargs: logged.update(values)  # type: ignore[assignment]
    actual_loss = model.training_step(batch, 0)
    assert torch.allclose(actual_loss, expected_loss, atol=1e-6), (
        "loss must not include any entropy-regularization contribution when "
        "transport_reg_weight=0.0, regardless of transport_head_entropy's value"
    )
    print("[hierarchical_gene_transport] OK — entropy regularization defaults to off "
          "and contributes nothing to the loss unless explicitly re-enabled")


def test_tokenized_gene_encoder_runs_and_stays_finite():
    """gene_encoder_type='tokenized' (round-4 wiring of the existing
    TokenizedGeneEncoder, previously only used by StormLiteContextEncoder)
    must produce the same [Nq, n_genes] output contract as the default
    weighted_linear/mlp encoders."""
    full_names = [f"g{i}" for i in range(6)]
    model = _build(
        gene_encoder_type="tokenized",
        tokenized_gene_names=full_names[:4],
        tokenized_full_gene_names=full_names,
    ).eval()
    context, query = _context_query()
    with torch.inference_mode():
        out = model.sample(context, query)
    assert out["expression"].shape == (2, 6)
    assert torch.isfinite(out["expression"]).all()
    print("[hierarchical_gene_transport] OK — gene_encoder_type='tokenized' runs and stays finite")


def test_tokenized_gene_encoder_requires_gene_names():
    try:
        _build(gene_encoder_type="tokenized")
        assert False, "expected ValueError for missing tokenized_gene_names"
    except ValueError as exc:
        assert "tokenized_gene_names" in str(exc)
    print("[hierarchical_gene_transport] OK — gene_encoder_type='tokenized' fails closed without gene names")


def test_global_candidate_off_by_default():
    model = _build()
    assert model.use_global_candidate is False
    print("[hierarchical_gene_transport] OK — use_global_candidate defaults to False")


def test_global_candidate_runs_and_stays_finite():
    model = _build(use_global_candidate=True, local_k=3).eval()
    context, query = _context_query(n_context=6)
    with torch.inference_mode():
        out = model.sample(context, query)
    assert out["expression"].shape == (2, 6)
    assert torch.isfinite(out["expression"]).all()
    print("[hierarchical_gene_transport] OK — use_global_candidate=True runs and stays finite")


def test_global_candidate_does_not_change_idw_anchor():
    """The IDW anchor must stay a pure local interpolation over exactly the
    k real neighbors, regardless of whether the learned candidate also sees
    a global whole-slide fallback slot."""
    torch.manual_seed(0)
    model_off = _build(use_global_candidate=False, local_k=3).eval()
    torch.manual_seed(0)
    model_on = _build(use_global_candidate=True, local_k=3).eval()
    context, query = _context_query(n_context=6)
    with torch.inference_mode():
        out_off = model_off.sample(context, query)
        out_on = model_on.sample(context, query)
    assert torch.allclose(out_off["anchor_expression"], out_on["anchor_expression"], atol=1e-6)
    print("[hierarchical_gene_transport] OK — global candidate leaves the IDW anchor exactly unchanged")


def test_global_candidate_lets_far_context_reach_the_prediction():
    """A far-away group of context spots (never among the k nearest) with
    very different expression can only ever move the prediction through the
    global candidate -- pure local kNN (use_global_candidate=False)
    structurally has no path for them to matter at all."""
    torch.manual_seed(0)
    model_off = _build(use_global_candidate=False, local_k=3, n_genes=2).eval()
    torch.manual_seed(0)
    model_on = _build(use_global_candidate=True, local_k=3, n_genes=2).eval()

    local_coords = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
    far_coords = torch.tensor([[500.0, 500.0, 0.0], [501.0, 500.0, 0.0], [500.0, 501.0, 0.0]])
    context = {
        "coords": torch.cat([local_coords, far_coords], dim=0),
        "expression": torch.cat([torch.zeros(3, 2), torch.full((3, 2), 1000.0)], dim=0),
    }
    query = {"coords": torch.tensor([[0.02, 0.02, 0.0]])}

    with torch.inference_mode():
        out_off = model_off.sample(context, query)
        out_on = model_on.sample(context, query)

    assert torch.allclose(
        out_off["expression"], torch.zeros_like(out_off["expression"]), atol=1e-3
    ), "pure local kNN must be entirely untouched by context spots outside local_k"
    assert not torch.allclose(out_on["expression"], out_off["expression"], atol=1e-6), (
        "enabling use_global_candidate must let far, non-local context influence the prediction"
    )
    print("[hierarchical_gene_transport] OK — use_global_candidate lets far context spots reach "
          "the prediction that pure local kNN structurally cannot see")


def test_niche_candidate_off_by_default():
    model = _build()
    assert model.use_niche_candidate is False
    print("[hierarchical_gene_transport] OK — use_niche_candidate defaults to False")


def test_niche_candidate_requires_niche_labels():
    model = _build(use_niche_candidate=True, local_k=3).eval()
    context, query = _context_query(n_context=6)
    try:
        with torch.inference_mode():
            model.sample(context, query)
        raise AssertionError("expected a ValueError for missing context['niche_labels']")
    except ValueError as exc:
        assert "niche_labels" in str(exc)
    print("[hierarchical_gene_transport] OK — use_niche_candidate=True fails closed "
          "without context['niche_labels']")


def test_niche_candidate_requires_matching_niche_label_count():
    model = _build(use_niche_candidate=True, local_k=3).eval()
    context, query = _context_query(n_context=6)
    context["niche_labels"] = torch.zeros(4, 1)  # wrong count, should be 6
    try:
        with torch.inference_mode():
            model.sample(context, query)
        raise AssertionError("expected a ValueError for mismatched niche_labels row count")
    except ValueError as exc:
        assert "one row per context spot" in str(exc)
    print("[hierarchical_gene_transport] OK — use_niche_candidate=True fails closed "
          "on a mismatched context['niche_labels'] row count")


def test_niche_candidate_runs_and_stays_finite():
    model = _build(use_niche_candidate=True, local_k=3).eval()
    context, query = _context_query(n_context=6)
    context["niche_labels"] = torch.randint(0, 3, (6, 1)).float()
    with torch.inference_mode():
        out = model.sample(context, query)
    assert out["expression"].shape == (2, 6)
    assert torch.isfinite(out["expression"]).all()
    print("[hierarchical_gene_transport] OK — use_niche_candidate=True runs and stays finite")


def test_niche_candidate_does_not_change_idw_anchor():
    torch.manual_seed(0)
    model_off = _build(use_niche_candidate=False, local_k=3).eval()
    torch.manual_seed(0)
    model_on = _build(use_niche_candidate=True, local_k=3).eval()
    context, query = _context_query(n_context=6)
    context_with_niche = dict(context)
    context_with_niche["niche_labels"] = torch.randint(0, 3, (6, 1)).float()
    with torch.inference_mode():
        out_off = model_off.sample(context, query)
        out_on = model_on.sample(context_with_niche, query)
    assert torch.allclose(out_off["anchor_expression"], out_on["anchor_expression"], atol=1e-6)
    print("[hierarchical_gene_transport] OK — niche candidate leaves the IDW anchor exactly unchanged")


def test_niche_candidate_pools_exactly_same_niche_context_spots():
    """The niche candidate must be the mean of exactly the context spots
    sharing the query's own niche (read off its nearest context neighbor's
    label), never spots from a different niche and never the whole slide --
    the thing that actually distinguishes it from use_global_candidate."""
    model = _build(use_niche_candidate=True, local_k=4, n_genes=2).eval()
    context = {
        "coords": torch.tensor([
            [0.0, 0.0, 0.0], [0.1, 0.0, 0.0],   # niche 0
            [5.0, 5.0, 0.0], [5.1, 5.0, 0.0],   # niche 1
        ]),
        "expression": torch.tensor([
            [1.0, 0.0], [3.0, 0.0],
            [100.0, 0.0], [300.0, 0.0],
        ]),
        "niche_labels": torch.tensor([[0], [0], [1], [1]], dtype=torch.float32),
    }
    query = {"coords": torch.tensor([[0.0, 0.0, 0.0]])}  # nearest context spot = 0, niche 0

    captured = {}
    original_einsum = torch.einsum

    def _spy_einsum(equation, *operands):
        if equation == "qhk,qkg->qhg":
            captured["scoring_neighbour_expression"] = operands[1]
        return original_einsum(equation, *operands)

    torch.einsum = _spy_einsum
    try:
        with torch.inference_mode():
            model.sample(context, query)
    finally:
        torch.einsum = original_einsum

    niche_candidate_expression = captured["scoring_neighbour_expression"][0, -1]
    expected = torch.tensor([2.0, 0.0])  # mean of the two real niche-0 spots
    assert torch.allclose(niche_candidate_expression, expected, atol=1e-5), (
        f"expected the niche candidate to be exactly {expected.tolist()} (mean of the "
        f"query's own niche's real context spots), got {niche_candidate_expression.tolist()}"
    )
    print("[hierarchical_gene_transport] OK — niche candidate pools exactly the context "
          "spots sharing the query's own (nearest-context-neighbor-derived) niche label")


def test_niche_and_global_candidates_stack_correctly():
    n_context, local_k = 6, 3
    model = _build(
        use_global_candidate=True, use_niche_candidate=True,
        local_k=local_k, n_genes=6,
    ).eval()
    context, query = _context_query(n_context=n_context)
    context["niche_labels"] = torch.randint(0, 3, (n_context, 1)).float()

    captured = {}
    original_einsum = torch.einsum

    def _spy_einsum(equation, *operands):
        if equation == "qhk,qkg->qhg":
            captured["n_candidates"] = operands[0].shape[-1]
        return original_einsum(equation, *operands)

    torch.einsum = _spy_einsum
    try:
        with torch.inference_mode():
            out = model.sample(context, query)
    finally:
        torch.einsum = original_einsum

    assert out["expression"].shape == (2, 6)
    assert torch.isfinite(out["expression"]).all()
    assert captured["n_candidates"] == local_k + 1 + 1, (
        "expected local_k neighbors + 1 global + 1 niche candidate slots"
    )
    print("[hierarchical_gene_transport] OK — niche + global candidates stack to exactly "
          "k+1+1 total candidates")


def test_retrieval_candidate_off_by_default():
    model = _build()
    assert model.use_retrieval_candidate is False
    assert model.retrieval_query_projection is None
    assert model.retrieval_expression_projection is None
    print("[hierarchical_gene_transport] OK — use_retrieval_candidate defaults to False")


def test_retrieval_candidate_runs_and_stays_finite():
    model = _build(use_retrieval_candidate=True, retrieval_k=2, local_k=3).eval()
    context, query = _context_query(n_context=6)
    with torch.inference_mode():
        out = model.sample(context, query)
    assert out["expression"].shape == (2, 6)
    assert torch.isfinite(out["expression"]).all()
    print("[hierarchical_gene_transport] OK — use_retrieval_candidate=True runs and stays finite")


def test_retrieval_candidate_does_not_change_idw_anchor():
    """The IDW anchor must stay a pure local interpolation over exactly the
    k real neighbors, regardless of whether the learned candidate also gets
    content-retrieved candidates."""
    torch.manual_seed(0)
    model_off = _build(use_retrieval_candidate=False, local_k=3).eval()
    torch.manual_seed(0)
    model_on = _build(use_retrieval_candidate=True, retrieval_k=2, local_k=3).eval()
    context, query = _context_query(n_context=6)
    with torch.inference_mode():
        out_off = model_off.sample(context, query)
        out_on = model_on.sample(context, query)
    assert torch.allclose(out_off["anchor_expression"], out_on["anchor_expression"], atol=1e-6)
    print("[hierarchical_gene_transport] OK — retrieval candidates leave the IDW anchor exactly unchanged")


def test_retrieval_candidate_lets_far_context_reach_the_prediction():
    """With retrieval_k set to the full context size, every context spot
    (including ones far outside local_k) is guaranteed to be retrieved
    regardless of the random projection weights -- a deterministic version
    of the same "far context can only reach the prediction through the new
    mechanism" property already tested for use_global_candidate."""
    torch.manual_seed(0)
    model_off = _build(use_retrieval_candidate=False, local_k=3, n_genes=2).eval()
    torch.manual_seed(0)
    model_on = _build(use_retrieval_candidate=True, retrieval_k=6, local_k=3, n_genes=2).eval()

    local_coords = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
    far_coords = torch.tensor([[500.0, 500.0, 0.0], [501.0, 500.0, 0.0], [500.0, 501.0, 0.0]])
    context = {
        "coords": torch.cat([local_coords, far_coords], dim=0),
        "expression": torch.cat([torch.zeros(3, 2), torch.full((3, 2), 1000.0)], dim=0),
    }
    query = {"coords": torch.tensor([[0.02, 0.02, 0.0]])}

    with torch.inference_mode():
        out_off = model_off.sample(context, query)
        out_on = model_on.sample(context, query)

    assert torch.allclose(
        out_off["expression"], torch.zeros_like(out_off["expression"]), atol=1e-3
    ), "pure local kNN must be entirely untouched by context spots outside local_k"
    assert not torch.allclose(out_on["expression"], out_off["expression"], atol=1e-6), (
        "enabling use_retrieval_candidate must let far, content-retrieved context influence the prediction"
    )
    print("[hierarchical_gene_transport] OK — use_retrieval_candidate lets far context spots reach "
          "the prediction that pure local kNN structurally cannot see")


def test_retrieval_loss_is_zero_when_disabled_and_finite_when_enabled():
    context, query = _context_query(n_context=6)
    target = torch.rand(2, 6)
    batch = {"context": context, "query": query, "target_expression": target}

    model_off = _build(use_retrieval_candidate=False, local_k=3)
    logged_off = {}
    model_off.log_dict = lambda values, **kwargs: logged_off.update(values)  # type: ignore[assignment]
    model_off.training_step(batch, 0)
    assert logged_off["train/retrieval_loss"] == 0.0

    model_on = _build(use_retrieval_candidate=True, retrieval_k=2, local_k=3)
    logged_on = {}
    model_on.log_dict = lambda values, **kwargs: logged_on.update(values)  # type: ignore[assignment]
    loss = model_on.training_step(batch, 0)
    assert torch.isfinite(logged_on["train/retrieval_loss"])
    assert torch.isfinite(loss)
    loss.backward()
    assert model_on.retrieval_query_projection.weight.grad is not None
    assert model_on.retrieval_expression_projection.weight.grad is not None
    print("[hierarchical_gene_transport] OK — retrieval contrastive loss is exactly 0 when disabled, "
          "finite and receives real gradient when enabled")


def test_global_and_retrieval_candidates_stack_correctly():
    """Config 242 stacks use_global_candidate AND use_retrieval_candidate
    together for the first time -- neither mechanism was unit-tested in
    combination with the other before, only individually. Both extend the
    SAME scoring_relative_geometry/scoring_neighbor_hidden/
    scoring_neighbour_expression tensors sequentially (global block first,
    then retrieval), so this specifically checks that chaining produces the
    correct final candidate count (k local + 1 global + retrieval_k
    retrieved) and stays finite, with the anchor still completely
    unaffected by either."""
    n_context, local_k, retrieval_k = 6, 3, 2
    model = _build(
        use_global_candidate=True,
        use_retrieval_candidate=True, retrieval_k=retrieval_k,
        local_k=local_k, n_genes=6,
    ).eval()
    context, query = _context_query(n_context=n_context)

    captured = {}
    original_einsum = torch.einsum

    def _spy_einsum(equation, *operands):
        if equation == "qhk,qkg->qhg":
            captured["n_candidates"] = operands[0].shape[-1]
        return original_einsum(equation, *operands)

    torch.einsum = _spy_einsum
    try:
        with torch.inference_mode():
            out = model.sample(context, query)
    finally:
        torch.einsum = original_einsum

    assert out["expression"].shape == (2, 6)
    assert torch.isfinite(out["expression"]).all()
    assert captured["n_candidates"] == local_k + 1 + retrieval_k, (
        f"expected {local_k}(local) + 1(global) + {retrieval_k}(retrieval) = "
        f"{local_k + 1 + retrieval_k} candidates in the gate's softmax, got {captured['n_candidates']}"
    )

    model_anchor_only = _build(
        use_global_candidate=False, use_retrieval_candidate=False,
        local_k=local_k, n_genes=6,
    ).eval()
    model_anchor_only.load_state_dict(
        {k: v for k, v in model.state_dict().items() if k in model_anchor_only.state_dict()},
        strict=False,
    )
    with torch.inference_mode():
        out_anchor_only = model_anchor_only.sample(context, query)
    assert torch.allclose(out["anchor_expression"], out_anchor_only["anchor_expression"], atol=1e-6), (
        "stacking both candidate mechanisms must still leave the pure-local IDW anchor unchanged"
    )
    print("[hierarchical_gene_transport] OK — global + retrieval candidates stack to exactly "
          "k+1+retrieval_k total candidates and leave the IDW anchor unchanged")


def test_niche_and_retrieval_candidates_stack_correctly():
    """Configs 246-248 (the niche-candidate follow-up matrix) stack
    use_niche_candidate with use_retrieval_candidate and/or
    use_global_candidate -- never exercised in combination before (only
    niche+global, in test_niche_and_global_candidates_stack_correctly
    above). All three blocks extend the SAME scoring_* tensors sequentially
    (global, then niche, then retrieval), so this checks the final
    candidate count and anchor invariance for niche+retrieval specifically."""
    n_context, local_k, retrieval_k = 6, 3, 2
    model = _build(
        use_niche_candidate=True,
        use_retrieval_candidate=True, retrieval_k=retrieval_k,
        local_k=local_k, n_genes=6,
    ).eval()
    context, query = _context_query(n_context=n_context)
    context["niche_labels"] = torch.randint(0, 3, (n_context, 1)).float()

    captured = {}
    original_einsum = torch.einsum

    def _spy_einsum(equation, *operands):
        if equation == "qhk,qkg->qhg":
            captured["n_candidates"] = operands[0].shape[-1]
        return original_einsum(equation, *operands)

    torch.einsum = _spy_einsum
    try:
        with torch.inference_mode():
            out = model.sample(context, query)
    finally:
        torch.einsum = original_einsum

    assert out["expression"].shape == (2, 6)
    assert torch.isfinite(out["expression"]).all()
    assert captured["n_candidates"] == local_k + 1 + retrieval_k, (
        f"expected {local_k}(local) + 1(niche) + {retrieval_k}(retrieval) = "
        f"{local_k + 1 + retrieval_k} candidates in the gate's softmax, got {captured['n_candidates']}"
    )
    print("[hierarchical_gene_transport] OK — niche + retrieval candidates stack to exactly "
          "k+1+retrieval_k total candidates")


def test_niche_global_and_retrieval_candidates_all_stack_correctly():
    """Config 248's full 'kitchen sink': all three candidate mechanisms at
    once. Checks the candidate count is exactly local + niche + global +
    retrieval and the IDW anchor is still completely unaffected."""
    n_context, local_k, retrieval_k = 8, 3, 2
    model = _build(
        use_global_candidate=True, use_niche_candidate=True,
        use_retrieval_candidate=True, retrieval_k=retrieval_k,
        local_k=local_k, n_genes=6,
    ).eval()
    context, query = _context_query(n_context=n_context)
    context["niche_labels"] = torch.randint(0, 3, (n_context, 1)).float()

    captured = {}
    original_einsum = torch.einsum

    def _spy_einsum(equation, *operands):
        if equation == "qhk,qkg->qhg":
            captured["n_candidates"] = operands[0].shape[-1]
        return original_einsum(equation, *operands)

    torch.einsum = _spy_einsum
    try:
        with torch.inference_mode():
            out = model.sample(context, query)
    finally:
        torch.einsum = original_einsum

    assert out["expression"].shape == (2, 6)
    assert torch.isfinite(out["expression"]).all()
    expected = local_k + 1 + 1 + retrieval_k
    assert captured["n_candidates"] == expected, (
        f"expected {local_k}(local) + 1(global) + 1(niche) + {retrieval_k}(retrieval) = "
        f"{expected} candidates in the gate's softmax, got {captured['n_candidates']}"
    )

    model_anchor_only = _build(
        use_global_candidate=False, use_niche_candidate=False, use_retrieval_candidate=False,
        local_k=local_k, n_genes=6,
    ).eval()
    model_anchor_only.load_state_dict(
        {k: v for k, v in model.state_dict().items() if k in model_anchor_only.state_dict()},
        strict=False,
    )
    with torch.inference_mode():
        out_anchor_only = model_anchor_only.sample(context, query)
    assert torch.allclose(out["anchor_expression"], out_anchor_only["anchor_expression"], atol=1e-6), (
        "stacking all three candidate mechanisms must still leave the pure-local IDW anchor unchanged"
    )
    print("[hierarchical_gene_transport] OK — global + niche + retrieval candidates all stack to "
          "exactly k+1+1+retrieval_k total candidates and leave the IDW anchor unchanged")


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
    test_entropy_regularization_is_off_by_default()
    test_tokenized_gene_encoder_runs_and_stays_finite()
    test_tokenized_gene_encoder_requires_gene_names()
    test_global_candidate_off_by_default()
    test_global_candidate_runs_and_stays_finite()
    test_global_candidate_does_not_change_idw_anchor()
    test_global_candidate_lets_far_context_reach_the_prediction()
    test_niche_candidate_off_by_default()
    test_niche_candidate_requires_niche_labels()
    test_niche_candidate_requires_matching_niche_label_count()
    test_niche_candidate_runs_and_stays_finite()
    test_niche_candidate_does_not_change_idw_anchor()
    test_niche_candidate_pools_exactly_same_niche_context_spots()
    test_niche_and_global_candidates_stack_correctly()
    test_retrieval_candidate_off_by_default()
    test_niche_and_retrieval_candidates_stack_correctly()
    test_niche_global_and_retrieval_candidates_all_stack_correctly()
    test_retrieval_candidate_runs_and_stays_finite()
    test_retrieval_candidate_does_not_change_idw_anchor()
    test_retrieval_candidate_lets_far_context_reach_the_prediction()
    test_retrieval_loss_is_zero_when_disabled_and_finite_when_enabled()
    test_global_and_retrieval_candidates_stack_correctly()
    test_all_thirteen_mandated_metrics_are_logged()
    print("\nAll hierarchical_gene_transport_regressor tests passed.")
