"""
Tests for PanelInvariantGeneDecoder (src/models/conditioning.py) and its
decoder_type="panel_invariant" wiring into WAEGAN/FlowMatchingOT/
VQVAEAutoregressive (src/models/registry.py) — the "GEX decoder ...
predicted expression in target platform space" box from the user's
2026-07-17 architecture roadmap (diagram 5). See conditioning.py's
PanelInvariantGeneDecoder docstring for the full reasoning.

Not validated against real cross-platform data (none available yet — see
docs/possible_extensions.md). These tests check the mechanism itself:
shape/gradient correctness, that querying a different gene subset actually
changes only that subset's output, that missing genes raise loudly, that
tech conditioning changes output, and that every generator family runs
end-to-end with decoder_type="panel_invariant" the same way it does with
"dense".

Run with: python -m tests.test_panel_invariant_decoder
"""
import torch

from src.models.conditioning import PanelInvariantGeneDecoder
from src.models.registry import WAEGAN, FlowMatchingOT, VQVAEAutoregressive


_GENES = [f"gene_{i}" for i in range(30)]


def test_shape_default_matches_full_vocab():
    torch.manual_seed(0)
    dec = PanelInvariantGeneDecoder(gene_names=_GENES, gene_embed_dim=8, in_dim=16, hidden_dim=24)
    h = torch.randn(5, 16)
    out = dec(h)  # gene_names=None -> full vocab, in original order
    assert out.shape == (5, len(_GENES)), out.shape
    assert torch.isfinite(out).all()
    print("[shape] OK — gene_names=None matches full training vocabulary width")


def test_shape_gene_subset():
    torch.manual_seed(0)
    dec = PanelInvariantGeneDecoder(gene_names=_GENES, gene_embed_dim=8, in_dim=16, hidden_dim=24)
    h = torch.randn(5, 16)
    subset = ["gene_3", "gene_0", "gene_17"]
    out = dec(h, gene_names=subset)
    assert out.shape == (5, 3), out.shape
    print("[shape] OK — arbitrary gene subset/order produces matching output width")


def test_subset_matches_full_columns():
    """The real panel-invariance property: scoring a subset of genes must
    give the exact same numbers as scoring the full panel and slicing —
    the decoder's output for a gene can't depend on which OTHER genes were
    also requested in the same call, otherwise it isn't really per-gene."""
    torch.manual_seed(0)
    dec = PanelInvariantGeneDecoder(gene_names=_GENES, gene_embed_dim=8, in_dim=16, hidden_dim=24)
    dec.eval()
    h = torch.randn(4, 16)
    with torch.no_grad():
        full = dec(h)  # [4, 30], full vocab in original order
        subset = ["gene_29", "gene_5"]
        out_subset = dec(h, gene_names=subset)  # [4, 2]
    expected = full[:, [29, 5]]
    assert torch.allclose(out_subset, expected, atol=1e-6), (
        "scoring a gene subset gave different values than slicing the full-panel output — "
        "decoder output for a gene must not depend on which other genes were co-queried"
    )
    print("[panel-invariance] OK — subset scoring matches full-panel slicing exactly")


def test_missing_gene_raises():
    dec = PanelInvariantGeneDecoder(gene_names=_GENES, gene_embed_dim=8, in_dim=16, hidden_dim=24)
    h = torch.randn(2, 16)
    try:
        dec(h, gene_names=["gene_0", "not_in_vocab"])
        raise AssertionError("expected an assertion error for an out-of-vocabulary gene")
    except AssertionError as e:
        assert "not in decoder vocabulary" in str(e), e
    print("[missing-gene] OK — querying an unknown gene raises loudly")


def test_tech_conditioning_changes_output():
    torch.manual_seed(0)
    dec = PanelInvariantGeneDecoder(
        gene_names=_GENES, gene_embed_dim=8, in_dim=16, hidden_dim=24,
        tech_vocab=["Visium", "Xenium"],
    )
    dec.eval()
    h = torch.randn(3, 16)
    with torch.no_grad():
        out_visium = dec(h, tech="Visium")
        out_xenium = dec(h, tech="Xenium")
        out_none = dec(h, tech=None)
    assert not torch.allclose(out_visium, out_xenium), (
        "different tech strings produced identical output — tech conditioning may be a no-op"
    )
    assert torch.allclose(out_none, out_none)  # sanity: no crash with tech=None + tech_vocab set
    print("[tech-conditioning] OK — different target-platform tech strings change output")


def test_gradient_flow():
    torch.manual_seed(0)
    dec = PanelInvariantGeneDecoder(gene_names=_GENES, gene_embed_dim=8, in_dim=16, hidden_dim=24)
    h = torch.randn(4, 16, requires_grad=True)
    out = dec(h, gene_names=["gene_1", "gene_2"])
    out.sum().backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()
    assert dec.gene_embed.weight.grad is not None and torch.isfinite(dec.gene_embed.weight.grad).all()
    print("[gradient] OK — gradients flow into both the decoder input and the gene embedding table")


def _make_batch(n_context, n_query, n_genes, coord_dim, tech=None):
    context = {
        "coords": torch.randn(n_context, coord_dim),
        "expression": torch.rand(n_context, n_genes),
    }
    query = {"coords": torch.randn(n_query, coord_dim)}
    if tech is not None:
        context["tech"] = tech
        query["tech"] = tech
    target_expression = torch.rand(n_query, n_genes)
    return {"context": context, "query": query, "target_expression": target_expression}


def test_wae_gan_panel_invariant_end_to_end():
    torch.manual_seed(0)
    gene_names = _GENES[:20]
    model = WAEGAN(n_genes=20, coord_dim=3, latent_dim=8, hidden_dim=32,
                    cond_hidden_dim=32, disc_hidden_dim=16,
                    decoder_type="panel_invariant", decoder_gene_names=gene_names)
    batch = _make_batch(n_context=40, n_query=10, n_genes=20, coord_dim=3)

    out = model.sample(batch["context"], batch["query"])
    assert out["expression"].shape == (10, 20), out["expression"].shape
    assert torch.isfinite(out["expression"]).all()

    opt_ae, opt_disc = model.configure_optimizers()
    model.optimizers = lambda: (opt_ae, opt_disc)
    model.manual_backward = lambda loss: loss.backward()
    model.log_dict = lambda *a, **k: None
    before = model.decoder.gene_embed.weight.clone()
    model.training_step(batch, batch_idx=0)
    after = model.decoder.gene_embed.weight
    assert not torch.allclose(before, after), "gene embedding table did not update"
    print("[WAEGAN] OK — decoder_type='panel_invariant' runs sample()/training_step() end-to-end")


def test_fm_ot_panel_invariant_end_to_end():
    torch.manual_seed(0)
    gene_names = _GENES[:20]
    model = FlowMatchingOT(n_genes=20, coord_dim=3, cond_hidden_dim=32,
                            hidden_dim=64, time_embed_dim=16, n_ode_steps=5,
                            decoder_type="panel_invariant", decoder_gene_names=gene_names)
    batch = _make_batch(n_context=40, n_query=10, n_genes=20, coord_dim=3)

    out = model.sample(batch["context"], batch["query"])
    assert out["expression"].shape == (10, 20), out["expression"].shape
    assert torch.isfinite(out["expression"]).all()

    opt = model.configure_optimizers()
    model.optimizers = lambda: opt
    model.manual_backward = lambda loss: loss.backward()
    model.log_dict = lambda *a, **k: None
    loss = model.training_step(batch, batch_idx=0)
    opt.zero_grad()
    loss.backward()
    assert torch.isfinite(loss)
    print("[FlowMatchingOT] OK — decoder_type='panel_invariant' runs sample()/training_step() end-to-end")


def test_vqvae_ar_panel_invariant_end_to_end():
    torch.manual_seed(0)
    gene_names = _GENES[:20]
    model = VQVAEAutoregressive(n_genes=20, coord_dim=3, cond_hidden_dim=32,
                                 latent_dim=8, ae_hidden_dim=32, codebook_size=16,
                                 transformer_dim=32, n_transformer_layers=2, n_heads=2,
                                 max_seq_len=64,
                                 decoder_type="panel_invariant", decoder_gene_names=gene_names)
    model.eval()
    batch = _make_batch(n_context=30, n_query=6, n_genes=20, coord_dim=3)

    with torch.no_grad():
        out = model.sample(batch["context"], batch["query"])
    assert out["expression"].shape == (6, 20), out["expression"].shape
    assert torch.isfinite(out["expression"]).all()

    model.train()
    loss = model.training_step(batch, batch_idx=0)
    loss.backward()
    assert torch.isfinite(loss)
    print("[VQVAEAutoregressive] OK — decoder_type='panel_invariant' runs sample()/training_step() end-to-end")


def test_combine_mode_add_shape_and_finiteness():
    torch.manual_seed(0)
    dec = PanelInvariantGeneDecoder(gene_names=_GENES, gene_embed_dim=8, in_dim=16,
                                     hidden_dim=24, combine_mode="add")
    h = torch.randn(5, 16)
    out = dec(h, gene_names=["gene_3", "gene_0", "gene_17"])
    assert out.shape == (5, 3), out.shape
    assert torch.isfinite(out).all()
    print("[combine_mode=add] OK — correct shape, finite output")


def test_combine_mode_add_vs_concat_differ():
    """Not just a smaller tensor -- combine_mode should genuinely change
    what the decoder computes, not produce the same numbers via a
    different path."""
    torch.manual_seed(0)
    dec_concat = PanelInvariantGeneDecoder(gene_names=_GENES, gene_embed_dim=8, in_dim=16,
                                            hidden_dim=24, combine_mode="concat")
    torch.manual_seed(0)
    dec_add = PanelInvariantGeneDecoder(gene_names=_GENES, gene_embed_dim=8, in_dim=16,
                                         hidden_dim=24, combine_mode="add")
    dec_concat.eval()
    dec_add.eval()
    h = torch.randn(4, 16)
    with torch.no_grad():
        out_concat = dec_concat(h, gene_names=["gene_1", "gene_2"])
        out_add = dec_add(h, gene_names=["gene_1", "gene_2"])
    assert not torch.allclose(out_concat, out_add), (
        "combine_mode='add' produced identical output to 'concat' — combine_mode may be a no-op"
    )
    print("[combine_mode] OK — 'add' and 'concat' produce genuinely different output")


def test_combine_mode_add_gradient_flow():
    torch.manual_seed(0)
    dec = PanelInvariantGeneDecoder(gene_names=_GENES, gene_embed_dim=8, in_dim=16,
                                     hidden_dim=24, combine_mode="add")
    h = torch.randn(4, 16, requires_grad=True)
    out = dec(h, gene_names=["gene_1", "gene_2"])
    out.sum().backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()
    assert dec.gene_embed.weight.grad is not None and torch.isfinite(dec.gene_embed.weight.grad).all()
    print("[combine_mode=add] OK — gradients flow correctly")


def test_hidden_dim_and_mlp_depth_override():
    """decoder_hidden_dim/decoder_mlp_depth (registry.py's _build_decoder)
    give the panel-invariant decoder capacity independent of whatever the
    caller's own dense-decoder width happens to be."""
    torch.manual_seed(0)
    dec = PanelInvariantGeneDecoder(gene_names=_GENES, gene_embed_dim=8, in_dim=16,
                                     hidden_dim=64, mlp_depth=3)
    # mlp_depth=3 -> 3 hidden Linear+GELU blocks then the final Linear(hidden_dim, 1)
    linear_layers = [m for m in dec.out_mlp if isinstance(m, torch.nn.Linear)]
    assert len(linear_layers) == 4, (
        f"mlp_depth=3 should produce 4 Linear layers total (3 hidden + 1 output), "
        f"got {len(linear_layers)}"
    )
    h = torch.randn(4, 16, requires_grad=True)
    out = dec(h, gene_names=["gene_1", "gene_2"])
    assert out.shape == (4, 2)
    out.sum().backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()
    print("[hidden_dim/mlp_depth] OK — independent width/depth knobs work correctly")


def test_target_gene_subset_smaller_than_training_panel():
    """The actual cross-platform use case (not yet exercised on real data,
    but the mechanism should work today): querying FEWER genes than the
    decoder was constructed with, e.g. a smaller target panel."""
    torch.manual_seed(0)
    gene_names = _GENES[:20]
    dec = PanelInvariantGeneDecoder(gene_names=gene_names, gene_embed_dim=8, in_dim=16, hidden_dim=24)
    h = torch.randn(5, 16)
    small_panel = gene_names[:5]  # a "different, smaller gene panel" stand-in
    out = dec(h, gene_names=small_panel)
    assert out.shape == (5, 5), out.shape
    assert torch.isfinite(out).all()
    print("[cross-panel] OK — querying a smaller target gene panel than the training vocabulary works")


def test_fm_ot_panel_invariant_add_combine_mode_end_to_end():
    """Registry-level wiring check for decoder_combine_mode (2026-07-17,
    scGPT-grounded memory-efficiency follow-up — see PanelInvariantGeneDecoder's
    own docstring)."""
    torch.manual_seed(0)
    gene_names = _GENES[:20]
    model = FlowMatchingOT(n_genes=20, coord_dim=3, cond_hidden_dim=32,
                            hidden_dim=64, time_embed_dim=16, n_ode_steps=5,
                            decoder_type="panel_invariant", decoder_gene_names=gene_names,
                            decoder_combine_mode="add", decoder_hidden_dim=48,
                            decoder_mlp_depth=2)
    assert model.decoder.combine_mode == "add"
    batch = _make_batch(n_context=40, n_query=10, n_genes=20, coord_dim=3)

    out = model.sample(batch["context"], batch["query"])
    assert out["expression"].shape == (10, 20), out["expression"].shape
    assert torch.isfinite(out["expression"]).all()

    opt = model.configure_optimizers()
    model.optimizers = lambda: opt
    model.manual_backward = lambda loss: loss.backward()
    model.log_dict = lambda *a, **k: None
    loss = model.training_step(batch, batch_idx=0)
    opt.zero_grad()
    loss.backward()
    assert torch.isfinite(loss)
    print("[FlowMatchingOT] OK — decoder_combine_mode='add' + decoder_hidden_dim/mlp_depth "
          "override run end-to-end")


if __name__ == "__main__":
    test_shape_default_matches_full_vocab()
    test_shape_gene_subset()
    test_subset_matches_full_columns()
    test_missing_gene_raises()
    test_tech_conditioning_changes_output()
    test_gradient_flow()
    test_wae_gan_panel_invariant_end_to_end()
    test_fm_ot_panel_invariant_end_to_end()
    test_vqvae_ar_panel_invariant_end_to_end()
    test_target_gene_subset_smaller_than_training_panel()
    test_combine_mode_add_shape_and_finiteness()
    test_combine_mode_add_vs_concat_differ()
    test_combine_mode_add_gradient_flow()
    test_hidden_dim_and_mlp_depth_override()
    test_fm_ot_panel_invariant_add_combine_mode_end_to_end()
    print("\nAll PanelInvariantGeneDecoder tests passed.")
