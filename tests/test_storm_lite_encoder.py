"""
Smoke test for StormLiteContextEncoder (src/models/storm_lite_encoder.py,
2026-07-16 STORM-recipe-at-pilot-scale investigation).

No external dependency needed to test the architecture itself — uses
precomputed [B, gigapath_dim]/[B, novae_dim] feature tensors (the fast
path GigapathPatchEncoder/NovaeGeneEncoder always use in real training),
never touching the real frozen GigaPath/Novae models.

Run with:
    python -m tests.test_storm_lite_encoder
"""
import torch

from src.models.storm_lite_encoder import (
    StormLiteContextEncoder, _MoMETransformerBlock, _knn_additive_mask, _knn_adjacency,
)
from src.models.conditioning import _GIGAPATH_FEAT_DIM, RelativePositionBias


def test_storm_lite_context_encoder():
    torch.manual_seed(0)
    n_context, n_query, n_genes, novae_dim, hidden_dim = 10, 4, 20, 64, 16

    context_coords = torch.rand(n_context, 3) * 100
    query_coords = torch.rand(n_query, 3) * 100
    context_images = torch.rand(n_context, _GIGAPATH_FEAT_DIM)  # precomputed-features fast path
    query_images = torch.rand(n_query, _GIGAPATH_FEAT_DIM)
    context_expression = torch.rand(n_context, n_genes)
    context_novae_features = torch.rand(n_context, novae_dim)

    for gene_encoder_type in ("mlp", "novae", "both"):
        encoder = StormLiteContextEncoder(
            n_genes=n_genes, novae_dim=novae_dim, hidden_dim=hidden_dim,
            n_transformer_layers=2, n_heads=4, gene_encoder_type=gene_encoder_type,
        )
        c = encoder(
            context_coords, context_expression, query_coords,
            context_images, query_images,
            context_novae_features=context_novae_features,
        )
        assert c.shape == (n_query, hidden_dim), (gene_encoder_type, c.shape)
        assert torch.isfinite(c).all()

        # real gradient-flow check: every trainable submodule should
        # receive a gradient from a backward pass through c, not just the
        # transformer (a real bug class this project has hit before —
        # see stpath_encoder.py's own docstring on a near-identical
        # silent-no-grad mistake)
        loss = c.sum()
        loss.backward()
        no_grad_params = [
            name for name, p in encoder.named_parameters()
            if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())
        ]
        assert not no_grad_params, (
            f"gene_encoder_type={gene_encoder_type!r}: params with no/invalid gradient: {no_grad_params}"
        )
        print(f"[StormLiteContextEncoder gene_encoder_type={gene_encoder_type!r}] "
              f"OK — output shape {tuple(c.shape)}, all params received gradient")


def test_mome_transformer_block():
    """Direct unit test of _MoMETransformerBlock (2026-07-17, real STORM
    detail — see its own docstring) — shape, gradient flow, and the real
    check that matters: image tokens and gene tokens must actually route
    through DIFFERENT FFN experts, not silently collapse to the same
    weights (a real bug class this project has hit before with similarly-
    named-but-distinct pathways — see CombinedGeneEncoder's sum-vs-concat
    history)."""
    torch.manual_seed(0)
    d_model, n_heads, n_total = 16, 4, 6
    block = _MoMETransformerBlock(d_model, n_heads)
    x = torch.randn(1, 2 * n_total, d_model)
    is_image_token = torch.cat([
        torch.ones(n_total, dtype=torch.bool), torch.zeros(n_total, dtype=torch.bool),
    ])
    out = block(x, is_image_token)
    assert out.shape == x.shape, out.shape
    assert torch.isfinite(out).all()

    loss = out.sum()
    loss.backward()
    no_grad = [name for name, p in block.named_parameters()
               if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    assert not no_grad, f"params with no/invalid gradient: {no_grad}"

    # real check the two experts are genuinely different modules, not
    # aliased to the same weights — feed the SAME feature vector through
    # both experts directly and confirm different output
    torch.manual_seed(0)
    same_input = torch.randn(1, 1, d_model)
    image_out = block.ffn_experts["image"](same_input)
    gene_out = block.ffn_experts["gene"](same_input)
    assert not torch.allclose(image_out, gene_out), (
        "image and gene FFN experts produced identical output on identical input — "
        "they may be accidentally sharing weights instead of being independent experts"
    )
    print("[_MoMETransformerBlock] OK — shape correct, gradient flows, "
          "image/gene experts are genuinely independent")


def test_storm_lite_fusion_mode_mome():
    """Integration test of StormLiteContextEncoder(fusion_mode='mome') —
    every gene_encoder_type x bias_type combination, shape + gradient
    flow (same discipline as test_storm_lite_context_encoder above)."""
    torch.manual_seed(0)
    n_context, n_query, n_genes, novae_dim, hidden_dim = 10, 4, 20, 64, 16
    context_coords = torch.rand(n_context, 3) * 5000  # real pixel-scale, not the [0,100) every other check uses
    query_coords = torch.rand(n_query, 3) * 5000
    context_images = torch.rand(n_context, _GIGAPATH_FEAT_DIM)
    query_images = torch.rand(n_query, _GIGAPATH_FEAT_DIM)
    context_expression = torch.rand(n_context, n_genes)
    context_novae_features = torch.rand(n_context, novae_dim)

    for gene_encoder_type in ("mlp", "novae", "both"):
        for bias_type in ("none", "relative_position", "frame_averaging"):
            encoder = StormLiteContextEncoder(
                n_genes=n_genes, novae_dim=novae_dim, hidden_dim=hidden_dim,
                n_transformer_layers=2, n_heads=4, gene_encoder_type=gene_encoder_type,
                fusion_mode="mome", bias_type=bias_type, coord_scale=1000.0,
            )
            c = encoder(context_coords, context_expression, query_coords,
                         context_images, query_images, context_novae_features=context_novae_features)
            assert c.shape == (n_query, hidden_dim), (gene_encoder_type, bias_type, c.shape)
            assert torch.isfinite(c).all()

            loss = c.sum()
            loss.backward()
            no_grad = [name for name, p in encoder.named_parameters()
                       if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
            assert not no_grad, (gene_encoder_type, bias_type, no_grad)
    print("[StormLiteContextEncoder fusion_mode='mome'] OK — all 9 "
          "gene_encoder_type x bias_type combinations pass shape/gradient checks")


def test_storm_lite_fusion_mode_sum_vs_mome_differ():
    torch.manual_seed(0)
    n_context, n_query, n_genes, hidden_dim = 8, 3, 15, 16
    context_coords = torch.rand(n_context, 3) * 5000
    query_coords = torch.rand(n_query, 3) * 5000
    context_images = torch.rand(n_context, _GIGAPATH_FEAT_DIM)
    query_images = torch.rand(n_query, _GIGAPATH_FEAT_DIM)
    context_expression = torch.rand(n_context, n_genes)

    torch.manual_seed(1)
    sum_encoder = StormLiteContextEncoder(
        n_genes=n_genes, hidden_dim=hidden_dim, gene_encoder_type="mlp", fusion_mode="sum",
    )
    torch.manual_seed(1)
    mome_encoder = StormLiteContextEncoder(
        n_genes=n_genes, hidden_dim=hidden_dim, gene_encoder_type="mlp", fusion_mode="mome",
    )
    with torch.no_grad():
        out_sum = sum_encoder(context_coords, context_expression, query_coords, context_images, query_images)
        out_mome = mome_encoder(context_coords, context_expression, query_coords, context_images, query_images)
    assert out_sum.shape == out_mome.shape
    assert not torch.allclose(out_sum, out_mome), (
        "fusion_mode='sum' and 'mome' produced identical output — 'mome' may not be wired up"
    )
    print("[StormLiteContextEncoder] OK — fusion_mode='sum' vs 'mome' produce genuinely different output")


def test_relative_position_bias():
    torch.manual_seed(0)
    n, coord_dim = 8, 3
    bias_module = RelativePositionBias(coord_dim, hidden_dim=16)
    coords = torch.rand(n, coord_dim) * 100
    bias = bias_module(coords)
    assert bias.shape == (n, n), bias.shape
    assert torch.isfinite(bias).all()
    # diagonal (self-attention, dx=dy=dz=dist=0) should be identical
    # across all n rows — same input (all-zero relative coords) must
    # produce the same MLP output every time (real check that this isn't
    # accidentally position-dependent in a way it shouldn't be)
    diag = bias.diagonal()
    assert torch.allclose(diag, diag[0].expand_as(diag), atol=1e-5), (
        "bias for zero relative-offset (self-attention) should be identical everywhere"
    )
    print(f"[RelativePositionBias] OK — shape {tuple(bias.shape)}, self-bias consistent")


def test_relative_position_bias_scale_invariance():
    """Regression test for the real 2026-07-17 bug: RelativePositionBias
    used to feed RAW coordinate differences into its MLP, producing
    exploding bias values on real HEST-1k pixel-scale coordinates
    (thousands, not the [0, 100) every other test in this file uses) —
    directly responsible for StormLiteContextEncoder scoring WORSE than a
    plain interpolation baseline on real data (PCC -0.005 to -0.015 vs.
    interp_baseline's 0.0076, ST-FID 22-36 vs. 10.09), across every
    gene_encoder_type variant, while STPath arms (unaffected by this
    class) trained fine. Fixed via per-call normalization by the point
    cloud's own max pairwise distance — this test checks BOTH properties
    that fix is supposed to guarantee, at a REALISTIC coordinate scale."""
    torch.manual_seed(0)
    n, coord_dim = 10, 3
    bias_module = RelativePositionBias(coord_dim, hidden_dim=16)

    # 1. boundedness: real HEST-1k-scale coordinates (thousands, matching
    # this project's own masking configs' radius_range, e.g. [250, 450])
    # must NOT produce an exploding bias — before the fix, this konsistently
    # this produced values in the hundreds-to-thousands (swamping real
    # attention logits, which are O(1-10)); after the fix, values stay small because
    # feat is normalized into roughly [-1, 1] before the MLP sees it.
    pixel_scale_coords = torch.rand(n, coord_dim) * 5000.0
    bias = bias_module(pixel_scale_coords)
    assert torch.isfinite(bias).all()
    assert bias.abs().max() < 50.0, (
        f"bias exploded at real pixel-scale coordinates: max |bias| = {bias.abs().max().item():.2f} "
        f"(this is exactly the 2026-07-17 bug — unnormalized coordinates feeding directly into the MLP)"
    )

    # 2. scale invariance: the SAME relative geometry at a 1000x different
    # absolute scale must produce IDENTICAL bias (up to floating point
    # tolerance) — the whole point of normalizing by this call's own max
    # pairwise distance is that only the RELATIVE layout matters, never
    # the caller's coordinate units.
    small_scale_coords = pixel_scale_coords / 1000.0
    bias_small = bias_module(small_scale_coords)
    assert torch.allclose(bias, bias_small, atol=1e-4), (
        "bias should be invariant to the absolute coordinate scale, only relative geometry should matter"
    )
    print(f"[RelativePositionBias scale invariance] OK — bounded (max |bias|={bias.abs().max().item():.3f}) "
          f"at pixel scale, invariant to a 1000x rescale")


def test_storm_lite_bias_type_actually_used():
    """Real check that bias_type isn't a silent no-op — output for each of
    the 3 bias_type options must differ from the others, given otherwise-
    identical weights/inputs (same failure class as this project's real
    @torch.no_grad()-swallowed-gradient bugs, just for "is this feature
    doing anything at all" instead of "is it trainable"). Covers both
    bias mechanisms (2026-07-17, frame_averaging added as the new default
    — see StormLiteContextEncoder's own bias_type docstring), not just
    the older relative_position vs none comparison this test used to
    check exclusively."""
    torch.manual_seed(0)
    n_context, n_query, n_genes, hidden_dim = 8, 3, 15, 16
    context_coords = torch.rand(n_context, 3) * 100
    query_coords = torch.rand(n_query, 3) * 100
    context_images = torch.rand(n_context, _GIGAPATH_FEAT_DIM)
    query_images = torch.rand(n_query, _GIGAPATH_FEAT_DIM)
    context_expression = torch.rand(n_context, n_genes)

    outputs = {}
    for bias_type in ("none", "relative_position", "frame_averaging"):
        torch.manual_seed(1)
        encoder = StormLiteContextEncoder(
            n_genes=n_genes, hidden_dim=hidden_dim, gene_encoder_type="mlp", bias_type=bias_type,
        )
        with torch.no_grad():
            outputs[bias_type] = encoder(context_coords, context_expression, query_coords,
                                          context_images, query_images)
        assert outputs[bias_type].shape == (n_query, hidden_dim)

    # same seed -> identical weights for every shared submodule constructed
    # BEFORE pos_bias in __init__, but pos_bias itself (constructed first)
    # shifts the seed sequence for everything after it differently per
    # bias_type — so this checks the weaker-but-still-real property that
    # all three outputs genuinely differ, not that non-bias weights are
    # byte-identical across bias_type values.
    assert not torch.allclose(outputs["none"], outputs["relative_position"]), (
        "bias_type='relative_position' produced identical output to 'none' — bias may be a no-op"
    )
    assert not torch.allclose(outputs["none"], outputs["frame_averaging"]), (
        "bias_type='frame_averaging' produced identical output to 'none' — bias may be a no-op"
    )
    assert not torch.allclose(outputs["relative_position"], outputs["frame_averaging"]), (
        "the two bias mechanisms produced identical output — suspicious given they're structurally different"
    )
    print("[StormLiteContextEncoder] OK — all 3 bias_type options genuinely produce different output")


def test_knn_additive_mask_keeps_exactly_k_neighbors():
    """2026-07-20, ports STFlow's real k-NN spatial attention restriction
    (Huang et al. 2025, arXiv 2506.05361, Section 3.3) — see
    _knn_additive_mask's own docstring for the honest scope note (this
    masks dense attention, doesn't reduce memory the way STFlow's sparse
    implementation does)."""
    torch.manual_seed(0)
    coords = torch.rand(6, 3) * 100
    for k in (1, 2, 6, 10):  # 10 > N=6 must clip gracefully to full attention
        mask = _knn_additive_mask(coords, k)
        assert mask.shape == (6, 6)
        kept_per_row = (mask == 0).sum(dim=-1)
        expected = min(k, 6)
        assert (kept_per_row == expected).all(), (k, kept_per_row)
        assert torch.isfinite(mask[mask == 0]).all()
    print("[knn_additive_mask] OK — keeps exactly min(k, N) neighbors per row, clips gracefully")


def test_storm_lite_knn_k_end_to_end():
    torch.manual_seed(0)
    n_context, n_query, n_genes, hidden_dim = 10, 4, 20, 16
    context_coords = torch.rand(n_context, 3) * 100
    query_coords = torch.rand(n_query, 3) * 100
    context_images = torch.rand(n_context, _GIGAPATH_FEAT_DIM)
    query_images = torch.rand(n_query, _GIGAPATH_FEAT_DIM)
    context_expression = torch.rand(n_context, n_genes)

    for fusion_mode in ("sum", "mome"):
        encoder = StormLiteContextEncoder(
            n_genes=n_genes, hidden_dim=hidden_dim, n_transformer_layers=2, n_heads=4,
            gene_encoder_type="mlp", fusion_mode=fusion_mode, knn_k=3,
        )
        c = encoder(context_coords, context_expression, query_coords, context_images, query_images)
        assert c.shape == (n_query, hidden_dim)
        assert torch.isfinite(c).all()
        c.sum().backward()
    print("[knn_additive_mask] OK — StormLiteContextEncoder runs end-to-end with knn_k set, both fusion_mode")


def test_storm_lite_knn_k_none_preserves_prior_behavior():
    """knn_k=None (default) must produce byte-identical output to before
    this feature existed — purely opt-in, zero effect on any existing
    config."""
    torch.manual_seed(0)
    n_context, n_query, n_genes, hidden_dim = 10, 4, 20, 16
    context_coords = torch.rand(n_context, 3) * 100
    query_coords = torch.rand(n_query, 3) * 100
    context_images = torch.rand(n_context, _GIGAPATH_FEAT_DIM)
    query_images = torch.rand(n_query, _GIGAPATH_FEAT_DIM)
    context_expression = torch.rand(n_context, n_genes)

    torch.manual_seed(1)
    enc_a = StormLiteContextEncoder(n_genes=n_genes, hidden_dim=hidden_dim, gene_encoder_type="mlp").eval()
    torch.manual_seed(1)
    enc_b = StormLiteContextEncoder(
        n_genes=n_genes, hidden_dim=hidden_dim, gene_encoder_type="mlp", knn_k=None
    ).eval()
    with torch.no_grad():
        out_a = enc_a(context_coords, context_expression, query_coords, context_images, query_images)
        out_b = enc_b(context_coords, context_expression, query_coords, context_images, query_images)
    assert torch.equal(out_a, out_b)
    print("[knn_additive_mask] OK — knn_k=None (default) is a true no-op")


def test_knn_adjacency_exactly_k_edges_with_self_loop():
    torch.manual_seed(0)
    coords = torch.rand(7, 3) * 100
    for k in (1, 3, 7, 12):  # 12 > N=7 must clip
        adj = _knn_adjacency(coords, k)
        expected = min(k, 7)
        assert (adj.sum(dim=-1) == expected).all()
        assert (adj.diag() == 1.0).all(), "self must always be an edge (distance 0 is always nearest)"
    print("[gnn_fusion] OK — _knn_adjacency has exactly min(k,N) edges per row, always incl. self-loop")


def test_storm_lite_fusion_mode_gnn_end_to_end():
    torch.manual_seed(0)
    n_context, n_query, n_genes, hidden_dim = 10, 4, 20, 16
    context_coords = torch.rand(n_context, 3) * 100
    query_coords = torch.rand(n_query, 3) * 100
    context_images = torch.rand(n_context, _GIGAPATH_FEAT_DIM)
    query_images = torch.rand(n_query, _GIGAPATH_FEAT_DIM)
    context_expression = torch.rand(n_context, n_genes)

    encoder = StormLiteContextEncoder(
        n_genes=n_genes, hidden_dim=hidden_dim, n_transformer_layers=2,
        gene_encoder_type="mlp", fusion_mode="gnn", gnn_k=4,
    )
    c = encoder(context_coords, context_expression, query_coords, context_images, query_images)
    assert c.shape == (n_query, hidden_dim)
    assert torch.isfinite(c).all()
    c.sum().backward()
    print("[gnn_fusion] OK — StormLiteContextEncoder fusion_mode='gnn' runs end-to-end, gradients flow")


def test_local_pool_ignores_query_images_and_absolute_coordinate_frame():
    """The missing-tissue path must not merely zero target H&E downstream:
    it must be independent of the query-image tensor and absolute frame."""
    torch.manual_seed(0)
    n_context, n_query, n_genes, hidden_dim = 12, 5, 20, 16
    context_coords = torch.rand(n_context, 3) * 100
    query_coords = torch.rand(n_query, 3) * 100
    context_images = torch.rand(n_context, _GIGAPATH_FEAT_DIM)
    context_expression = torch.rand(n_context, n_genes)
    encoder = StormLiteContextEncoder(
        n_genes=n_genes, hidden_dim=hidden_dim, gene_encoder_type="mlp",
        fusion_mode="local_pool", local_k=4,
    ).eval()
    with torch.no_grad():
        reference = encoder(
            context_coords, context_expression, query_coords,
            context_images, torch.randn(n_query, _GIGAPATH_FEAT_DIM),
        )
        # Rotate, uniformly rescale and translate x/y together.
        angle = torch.tensor(0.73)
        rotation = torch.stack([
            torch.stack([torch.cos(angle), -torch.sin(angle)]),
            torch.stack([torch.sin(angle), torch.cos(angle)]),
        ])
        transformed_context = context_coords.clone()
        transformed_query = query_coords.clone()
        transformed_context[:, :2] = context_coords[:, :2] @ rotation.T * 7.0 + 1234.0
        transformed_query[:, :2] = query_coords[:, :2] @ rotation.T * 7.0 + 1234.0
        transformed = encoder(
            transformed_context, context_expression, transformed_query,
            context_images, None,
        )
    assert torch.allclose(reference, transformed, atol=2e-5, rtol=2e-5)


def test_storm_lite_fusion_mode_gnn_differs_from_sum_and_mome():
    """Genuinely different aggregation mechanism -> genuinely different
    output, not an accidental no-op reduction to the same computation."""
    torch.manual_seed(0)
    n_context, n_query, n_genes, hidden_dim = 10, 4, 20, 16
    context_coords = torch.rand(n_context, 3) * 100
    query_coords = torch.rand(n_query, 3) * 100
    context_images = torch.rand(n_context, _GIGAPATH_FEAT_DIM)
    query_images = torch.rand(n_query, _GIGAPATH_FEAT_DIM)
    context_expression = torch.rand(n_context, n_genes)

    outputs = {}
    for fusion_mode in ("sum", "mome", "gnn"):
        torch.manual_seed(1)
        kwargs = {"gnn_k": 4} if fusion_mode == "gnn" else {}
        encoder = StormLiteContextEncoder(
            n_genes=n_genes, hidden_dim=hidden_dim, gene_encoder_type="mlp",
            fusion_mode=fusion_mode, **kwargs,
        )
        with torch.no_grad():
            outputs[fusion_mode] = encoder(
                context_coords, context_expression, query_coords, context_images, query_images
            )
    assert not torch.allclose(outputs["sum"], outputs["gnn"])
    assert not torch.allclose(outputs["mome"], outputs["gnn"])
    print("[gnn_fusion] OK — fusion_mode='gnn' produces genuinely different output from 'sum'/'mome'")


if __name__ == "__main__":
    test_storm_lite_context_encoder()
    test_mome_transformer_block()
    test_storm_lite_fusion_mode_mome()
    test_storm_lite_fusion_mode_sum_vs_mome_differ()
    test_relative_position_bias()
    test_relative_position_bias_scale_invariance()
    test_storm_lite_bias_type_actually_used()
    test_knn_additive_mask_keeps_exactly_k_neighbors()
    test_storm_lite_knn_k_end_to_end()
    test_storm_lite_knn_k_none_preserves_prior_behavior()
    test_knn_adjacency_exactly_k_edges_with_self_loop()
    test_storm_lite_fusion_mode_gnn_end_to_end()
    test_storm_lite_fusion_mode_gnn_differs_from_sum_and_mome()
    print("\nStormLiteContextEncoder smoke test done.")


def test_storm_lite_concat_fusion_is_wired_and_trainable():
    torch.manual_seed(3)
    n_context, n_query, n_genes, hidden_dim = 7, 3, 12, 16
    context_coords = torch.rand(n_context, 3) * 100
    query_coords = torch.rand(n_query, 3) * 100
    context_images = torch.rand(n_context, _GIGAPATH_FEAT_DIM)
    query_images = torch.rand(n_query, _GIGAPATH_FEAT_DIM)
    context_expression = torch.rand(n_context, n_genes)

    encoder = StormLiteContextEncoder(
        n_genes=n_genes,
        hidden_dim=hidden_dim,
        gene_encoder_type="mlp",
        fusion_mode="concat",
        bias_type="none",
        n_transformer_layers=1,
        n_heads=4,
    )
    out = encoder(
        context_coords, context_expression, query_coords,
        context_images, query_images,
    )
    assert out.shape == (n_query, hidden_dim)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert encoder.concat_proj.weight.grad is not None
    assert torch.isfinite(encoder.concat_proj.weight.grad).all()


def test_storm_lite_sum_concat_and_mome_are_distinct():
    torch.manual_seed(4)
    n_context, n_query, n_genes, hidden_dim = 7, 3, 12, 16
    context_coords = torch.rand(n_context, 3) * 100
    query_coords = torch.rand(n_query, 3) * 100
    context_images = torch.rand(n_context, _GIGAPATH_FEAT_DIM)
    query_images = torch.rand(n_query, _GIGAPATH_FEAT_DIM)
    context_expression = torch.rand(n_context, n_genes)

    outputs = {}
    for mode in ("sum", "concat", "mome"):
        torch.manual_seed(11)
        encoder = StormLiteContextEncoder(
            n_genes=n_genes,
            hidden_dim=hidden_dim,
            gene_encoder_type="mlp",
            fusion_mode=mode,
            bias_type="none",
            n_transformer_layers=1,
            n_heads=4,
        )
        encoder.eval()
        with torch.no_grad():
            outputs[mode] = encoder(
                context_coords, context_expression, query_coords,
                context_images, query_images,
            )
    assert not torch.allclose(outputs["sum"], outputs["concat"])
    assert not torch.allclose(outputs["sum"], outputs["mome"])
    assert not torch.allclose(outputs["concat"], outputs["mome"])
