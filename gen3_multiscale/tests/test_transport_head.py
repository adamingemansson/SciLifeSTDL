"""Phase 4 (multiscale spatial-field handoff): the gene-value-preserving
transport head, extracted from the audited HierarchicalGeneTransportRegressor
into a standalone, encoder-agnostic module. Tests directly exercise the
handoff's own named pre-launch gates:

- "The selected conditioning encoder is noncollapsed and receives
  gradients" / "The transport scorer and gene gates receive gradients."
- "Transport weights are finite, normalized, and vary across queries on
  a nontrivial example."
- "The low-rank residual begins at exactly zero and cannot become a free
  sample-specific gene bias."
- "Architectures 1, 3, and 4 contain no computational path from any
  interpolation output to their predictions or losses."
- "Verify untouched observed full-gene values -- not compressed
  reconstructions -- are the values mixed by transport."
"""
import torch

from gen3_multiscale.models.transport_head import GeneValueTransportHead


def _inputs(n_query=5, n_candidates=6, hidden_dim=16, n_genes=8, requires_grad=False):
    torch.manual_seed(0)
    query_hidden = torch.randn(n_query, hidden_dim, requires_grad=requires_grad)
    candidate_hidden = torch.randn(n_query, n_candidates, hidden_dim, requires_grad=requires_grad)
    candidate_relative_geometry = torch.randn(n_query, n_candidates, 3, requires_grad=requires_grad)
    candidate_expression = torch.randn(n_query, n_candidates, n_genes)
    return query_hidden, candidate_hidden, candidate_relative_geometry, candidate_expression


def test_forward_shapes_are_correct_in_anchor_free_mode():
    head = GeneValueTransportHead(n_genes=8, hidden_dim=16, transport_heads=4)
    query_hidden, candidate_hidden, geometry, expr = _inputs()
    out = head(query_hidden, candidate_hidden, geometry, expr)
    assert out["expression"].shape == (5, 8)
    assert out["anchor_expression"] is None
    assert out["blend"] is None


def test_head_weights_are_convex_over_candidates():
    """"Transport weights are finite, normalized..." gate."""
    head = GeneValueTransportHead(n_genes=8, hidden_dim=16, transport_heads=4)
    query_hidden, candidate_hidden, geometry, expr = _inputs()
    out = head(query_hidden, candidate_hidden, geometry, expr)
    w = out["head_weights"]  # [Nq, heads, C]
    assert torch.isfinite(w).all()
    assert (w >= 0).all()
    assert torch.allclose(w.sum(dim=-1), torch.ones(w.shape[:2]), atol=1e-5)


def test_gene_gates_are_convex_over_heads():
    head = GeneValueTransportHead(n_genes=8, hidden_dim=16, transport_heads=4)
    query_hidden, candidate_hidden, geometry, expr = _inputs()
    out = head(query_hidden, candidate_hidden, geometry, expr)
    g = out["gene_gates"]  # [Nq, G, heads]
    assert torch.isfinite(g).all()
    assert (g >= 0).all()
    assert torch.allclose(g.sum(dim=-1), torch.ones(g.shape[:2]), atol=1e-5)


def test_transport_weights_vary_across_queries_on_a_nontrivial_example():
    """"...and vary across queries on a nontrivial example" gate."""
    head = GeneValueTransportHead(n_genes=8, hidden_dim=16, transport_heads=4)
    query_hidden, candidate_hidden, geometry, expr = _inputs(n_query=5)
    out = head(query_hidden, candidate_hidden, geometry, expr)
    w = out["head_weights"]
    assert not torch.allclose(w[0], w[1])


def test_untouched_observed_values_are_the_values_actually_mixed():
    """Direct test of the handoff's own gate: swapping candidate_expression
    for a different real matrix (same shape) must change the prediction --
    proves the output is genuinely built FROM those real values, not
    computed independently of them (e.g. via some compressed bottleneck)."""
    head = GeneValueTransportHead(n_genes=8, hidden_dim=16, transport_heads=4)
    query_hidden, candidate_hidden, geometry, expr = _inputs()
    out_a = head(query_hidden, candidate_hidden, geometry, expr)
    out_b = head(query_hidden, candidate_hidden, geometry, expr * 0.0 + 5.0)
    assert not torch.allclose(out_a["expression"], out_b["expression"])
    # and if every candidate has the identical real expression, transport
    # (a convex combination of them) must reproduce that value exactly
    uniform_expr = torch.full_like(expr, 3.0)
    out_uniform = head(query_hidden, candidate_hidden, geometry, uniform_expr)
    assert torch.allclose(out_uniform["candidate_expression"], torch.full((5, 8), 3.0), atol=1e-4)


def test_gradients_flow_through_scorer_query_hidden_candidate_hidden_and_gene_gates():
    """"The transport scorer and gene gates receive gradients" gate."""
    head = GeneValueTransportHead(n_genes=8, hidden_dim=16, transport_heads=4)
    query_hidden, candidate_hidden, geometry, expr = _inputs(requires_grad=True)
    out = head(query_hidden, candidate_hidden, geometry, expr)
    out["expression"].sum().backward()
    assert query_hidden.grad is not None and torch.isfinite(query_hidden.grad).all()
    assert candidate_hidden.grad is not None and torch.isfinite(candidate_hidden.grad).all()
    assert geometry.grad is not None and torch.isfinite(geometry.grad).all()
    assert head.gene_head_logits.grad is not None
    assert not torch.allclose(head.gene_head_logits.grad, torch.zeros_like(head.gene_head_logits.grad))


def test_low_rank_residual_begins_at_exactly_zero():
    """"The low-rank residual begins at exactly zero and cannot become a
    free sample-specific gene bias" gate."""
    torch.manual_seed(1)
    head_no_residual = GeneValueTransportHead(n_genes=8, hidden_dim=16, transport_heads=4, use_residual=False)
    torch.manual_seed(1)
    head_with_residual = GeneValueTransportHead(n_genes=8, hidden_dim=16, transport_heads=4, use_residual=True)

    query_hidden, candidate_hidden, geometry, expr = _inputs()
    out_no_res = head_no_residual(query_hidden, candidate_hidden, geometry, expr)
    out_with_res = head_with_residual(query_hidden, candidate_hidden, geometry, expr)

    assert torch.allclose(out_with_res["residual_expression"], torch.zeros(5, 8))
    assert torch.allclose(out_no_res["expression"], out_with_res["expression"], atol=1e-6)


def test_residual_rank_is_low_not_a_free_per_query_per_gene_bias():
    """A rank-r residual can only express r degrees of freedom per gene
    across queries -- verify residual_gene_embedding actually has the
    configured rank shape, not [n_query, n_genes] worth of free parameters."""
    head = GeneValueTransportHead(n_genes=8, hidden_dim=16, transport_heads=4, use_residual=True, residual_rank=3)
    assert head.residual_gene_embedding.shape == (8, 3)
    assert head.residual_query_projection.out_features == 3


def test_anchor_free_head_has_no_blend_parameter_at_all():
    """"Architectures 1, 3, and 4... contain no computational path from
    any interpolation output to their predictions or losses" -- verified
    structurally: blend_logit must not exist as a parameter when
    use_anchor_blend=False, not merely be near zero."""
    head = GeneValueTransportHead(n_genes=8, hidden_dim=16, transport_heads=4, use_anchor_blend=False)
    assert head.blend_logit is None
    assert "blend_logit" not in dict(head.named_parameters())


def test_anchor_free_head_rejects_an_anchor_argument():
    head = GeneValueTransportHead(n_genes=8, hidden_dim=16, transport_heads=4, use_anchor_blend=False)
    query_hidden, candidate_hidden, geometry, expr = _inputs()
    try:
        head(query_hidden, candidate_hidden, geometry, expr, anchor_expression=torch.zeros(5, 8))
        assert False, "expected a ValueError"
    except ValueError as exc:
        assert "anchor-free" in str(exc) or "use_anchor_blend=False" in str(exc)


def test_anchor_blend_head_requires_an_anchor_argument():
    head = GeneValueTransportHead(n_genes=8, hidden_dim=16, transport_heads=4, use_anchor_blend=True)
    query_hidden, candidate_hidden, geometry, expr = _inputs()
    try:
        head(query_hidden, candidate_hidden, geometry, expr)
        assert False, "expected a ValueError"
    except ValueError as exc:
        assert "anchor_expression" in str(exc)


def test_anchor_blend_head_stays_close_to_the_anchor_at_initialization():
    """Matches the audited default: blend_logit_init = logit(0.05), so a
    freshly-constructed Architecture 2 head starts close to the anchor,
    with a live (small) gradient path into the learned candidate."""
    head = GeneValueTransportHead(n_genes=8, hidden_dim=16, transport_heads=4, use_anchor_blend=True)
    assert torch.allclose(torch.sigmoid(head.blend_logit), torch.full((8,), 0.05), atol=1e-3)

    query_hidden, candidate_hidden, geometry, expr = _inputs()
    anchor = torch.randn(5, 8)
    out = head(query_hidden, candidate_hidden, geometry, expr, anchor_expression=anchor)
    # final expression should be MUCH closer to the anchor than to the raw candidate at init
    dist_to_anchor = (out["expression"] - anchor).abs().mean()
    dist_to_candidate = (out["expression"] - out["candidate_expression"]).abs().mean()
    assert dist_to_anchor < dist_to_candidate


def test_architecture_2_outputs_are_all_independently_loggable():
    """"Verify Architecture 2's harmonic anchor, learned candidate, blend,
    residual, and final outputs can be logged independently" gate."""
    head = GeneValueTransportHead(
        n_genes=8, hidden_dim=16, transport_heads=4, use_anchor_blend=True, use_residual=True,
    )
    query_hidden, candidate_hidden, geometry, expr = _inputs()
    anchor = torch.randn(5, 8)
    out = head(query_hidden, candidate_hidden, geometry, expr, anchor_expression=anchor)

    assert torch.allclose(out["anchor_expression"], anchor)
    assert out["candidate_expression"].shape == (5, 8)
    assert out["blend"].shape == (1, 8)
    assert out["residual_expression"].shape == (5, 8)
    assert out["expression"].shape == (5, 8)
    # each is a genuinely distinct tensor, not silently aliased
    assert out["expression"] is not out["candidate_expression"]
    assert out["expression"] is not out["anchor_expression"]
    assert out["expression"] is not out["transport_expression"]
    # residual is zero-initialized (see test_low_rank_residual_begins_at_exactly_zero),
    # so at a fresh init transport_expression == expression -- perturb the
    # residual weight to prove the two fields genuinely diverge once
    # training moves it, rather than always being aliased to each other.
    with torch.no_grad():
        head.residual_query_projection.weight.add_(0.1)
    out_perturbed = head(query_hidden, candidate_hidden, geometry, expr, anchor_expression=anchor)
    assert not torch.allclose(out_perturbed["transport_expression"], out_perturbed["expression"])


def test_rejects_mismatched_candidate_expression_gene_width():
    head = GeneValueTransportHead(n_genes=8, hidden_dim=16, transport_heads=4)
    query_hidden, candidate_hidden, geometry, _ = _inputs()
    wrong_expr = torch.randn(5, 6, 3)  # wrong gene width
    try:
        head(query_hidden, candidate_hidden, geometry, wrong_expr)
        assert False, "expected a ValueError"
    except ValueError as exc:
        assert "candidate_expression" in str(exc)
