"""
Tests for GeneAttentionDecoder and LLOKIStyleDecoder (src/models/conditioning.py)
and their decoder_type="gene_attention"/"lloki" wiring into
WAEGAN/FlowMatchingOT/VQVAEAutoregressive (src/models/registry.py) —
2026-07-17 literature research follow-up to PanelInvariantGeneDecoder
(see docs/results_log.md's dedicated entry for the full research writeup,
and each class's own docstring for what's faithfully ported vs. what
isn't).

GeneAttentionDecoder: Geneformer-inspired (genes as self-attended tokens,
shared per-position output head), guarded against the full ~16570-gene
vocabulary it's NOT designed for (MAX_SAFE_PANEL_SIZE).

LLOKIStyleDecoder: faithfully ports LLOKI-CAE's real, verified
conditional-autoencoder mechanism (technology token concatenated at
input, multi-layer ReLU stack) — NOT panel-invariant (fixed n_genes
width), a deliberate, documented limitation inherited from the real
source architecture, not a bug.

Not validated against real cross-platform data (none available yet — see
docs/possible_extensions.md).

Run with: python -m tests.test_alternative_decoders
"""
import torch

from src.models.conditioning import GeneAttentionDecoder, LLOKIStyleDecoder
from src.models.registry import FlowMatchingOT, WAEGAN


_GENES = [f"gene_{i}" for i in range(30)]
_SMALL_PANEL = _GENES[:10]


# ---------------------------------------------------------------------------
# GeneAttentionDecoder
# ---------------------------------------------------------------------------

def test_gene_attention_shape_and_finiteness():
    torch.manual_seed(0)
    dec = GeneAttentionDecoder(gene_names=_GENES, gene_embed_dim=8, in_dim=16, hidden_dim=24)
    h = torch.randn(5, 16)
    out = dec(h, gene_names=_SMALL_PANEL)
    assert out.shape == (5, 10), out.shape
    assert torch.isfinite(out).all()
    print("[gene_attention] OK — correct shape, finite output")


def test_gene_attention_requires_explicit_gene_names():
    dec = GeneAttentionDecoder(gene_names=_GENES, gene_embed_dim=8, in_dim=16, hidden_dim=24)
    h = torch.randn(2, 16)
    try:
        dec(h)  # gene_names=None, no full-vocabulary default (unlike PanelInvariantGeneDecoder)
        raise AssertionError("expected an assertion error for missing gene_names")
    except AssertionError as e:
        assert "requires an explicit gene_names" in str(e), e
    print("[gene_attention] OK — refuses to default to the full vocabulary")


def test_gene_attention_panel_size_guard():
    dec = GeneAttentionDecoder(gene_names=_GENES, gene_embed_dim=8, in_dim=16, hidden_dim=24)
    dec.MAX_SAFE_PANEL_SIZE = 5  # shrink the guard for a fast test, don't build a 4096-gene vocab
    h = torch.randn(2, 16)
    try:
        dec(h, gene_names=_GENES[:10])  # 10 > shrunk guard of 5
        raise AssertionError("expected an assertion error for exceeding MAX_SAFE_PANEL_SIZE")
    except AssertionError as e:
        assert "MAX_SAFE_PANEL_SIZE" in str(e), e
    print("[gene_attention] OK — refuses an oversized panel instead of silently attempting O(n^2) attention")


def test_gene_attention_genes_interact_via_self_attention():
    """The real, load-bearing property that distinguishes this from
    PanelInvariantGeneDecoder: gene predictions are allowed to depend on
    which OTHER genes are in the same query panel, because they
    self-attend to each other. PanelInvariantGeneDecoder is explicitly
    invariant to this (see its own test_subset_matches_full_columns) —
    this decoder should NOT be, or the self-attention layer isn't doing
    anything."""
    torch.manual_seed(0)
    dec = GeneAttentionDecoder(gene_names=_GENES, gene_embed_dim=8, in_dim=16, hidden_dim=24, n_layers=1)
    dec.eval()
    h = torch.randn(3, 16)
    with torch.no_grad():
        out_small = dec(h, gene_names=["gene_0", "gene_1"])
        out_large = dec(h, gene_names=["gene_0", "gene_1", "gene_2", "gene_3", "gene_4"])
    # gene_0's prediction in the 2-gene panel vs. the 5-gene panel should differ,
    # since self-attention lets the other co-queried genes influence it
    assert not torch.allclose(out_small[:, 0], out_large[:, 0]), (
        "gene_0's prediction was identical regardless of which other genes were "
        "co-queried — self-attention among gene tokens doesn't seem to be doing anything"
    )
    print("[gene_attention] OK — gene predictions genuinely depend on co-queried genes (self-attention is live)")


def test_gene_attention_tech_conditioning_changes_output():
    torch.manual_seed(0)
    dec = GeneAttentionDecoder(gene_names=_GENES, gene_embed_dim=8, in_dim=16, hidden_dim=24,
                                tech_vocab=["Visium", "Xenium"])
    dec.eval()
    h = torch.randn(3, 16)
    with torch.no_grad():
        out_visium = dec(h, gene_names=_SMALL_PANEL, tech="Visium")
        out_xenium = dec(h, gene_names=_SMALL_PANEL, tech="Xenium")
    assert not torch.allclose(out_visium, out_xenium), (
        "different tech strings produced identical output — tech conditioning may be a no-op"
    )
    print("[gene_attention] OK — tech conditioning changes output")


def test_gene_attention_gradient_flow():
    torch.manual_seed(0)
    dec = GeneAttentionDecoder(gene_names=_GENES, gene_embed_dim=8, in_dim=16, hidden_dim=24)
    h = torch.randn(4, 16, requires_grad=True)
    out = dec(h, gene_names=_SMALL_PANEL)
    out.sum().backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()
    assert dec.gene_embed.weight.grad is not None and torch.isfinite(dec.gene_embed.weight.grad).all()
    print("[gene_attention] OK — gradients flow into both the decoder input and the gene embedding table")


# ---------------------------------------------------------------------------
# LLOKIStyleDecoder
# ---------------------------------------------------------------------------

def test_lloki_shape_and_finiteness():
    torch.manual_seed(0)
    dec = LLOKIStyleDecoder(in_dim=16, n_genes=30, tech_vocab=["Visium", "Xenium"],
                             tech_embed_dim=6, hidden_dims=(32, 16))
    h = torch.randn(5, 16)
    out = dec(h, tech="Visium")
    assert out.shape == (5, 30), out.shape  # fixed width, NOT panel-invariant — see class docstring
    assert torch.isfinite(out).all()
    print("[lloki] OK — correct fixed-width shape, finite output")


def test_lloki_requires_tech_vocab_at_construction():
    try:
        LLOKIStyleDecoder(in_dim=16, n_genes=30, tech_vocab=[])
        raise AssertionError("expected an assertion error for empty tech_vocab")
    except AssertionError as e:
        assert "requires a non-empty tech_vocab" in str(e), e
    print("[lloki] OK — refuses construction without a tech_vocab")


def test_lloki_unknown_tech_raises():
    dec = LLOKIStyleDecoder(in_dim=16, n_genes=30, tech_vocab=["Visium"])
    h = torch.randn(2, 16)
    try:
        dec(h, tech="Xenium")
        raise AssertionError("expected an assertion error for an unknown tech string")
    except AssertionError as e:
        assert "not in decoder tech_vocab" in str(e), e
    print("[lloki] OK — querying an unknown tech string raises loudly")


def test_lloki_tech_conditioning_changes_output():
    torch.manual_seed(0)
    dec = LLOKIStyleDecoder(in_dim=16, n_genes=30, tech_vocab=["Visium", "Xenium"],
                             tech_embed_dim=6, hidden_dims=(32, 16))
    dec.eval()
    h = torch.randn(3, 16)
    with torch.no_grad():
        out_visium = dec(h, tech="Visium")
        out_xenium = dec(h, tech="Xenium")
    assert not torch.allclose(out_visium, out_xenium), (
        "different tech strings produced identical output — the whole mechanism this "
        "decoder exists for (technology-conditioned decoding) may be a no-op"
    )
    print("[lloki] OK — different technology strings genuinely change output")


def test_lloki_gradient_flow():
    torch.manual_seed(0)
    dec = LLOKIStyleDecoder(in_dim=16, n_genes=30, tech_vocab=["Visium"],
                             tech_embed_dim=6, hidden_dims=(32, 16))
    h = torch.randn(4, 16, requires_grad=True)
    out = dec(h, tech="Visium")
    out.sum().backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()
    assert dec.tech_embed.weight.grad is not None and torch.isfinite(dec.tech_embed.weight.grad).all()
    print("[lloki] OK — gradients flow into both the decoder input and the technology embedding table")


# ---------------------------------------------------------------------------
# Registry-level end-to-end wiring
# ---------------------------------------------------------------------------

def _make_batch(n_context, n_query, n_genes, coord_dim, tech):
    context = {
        "coords": torch.randn(n_context, coord_dim),
        "expression": torch.rand(n_context, n_genes),
        "tech": tech,
    }
    query = {"coords": torch.randn(n_query, coord_dim), "tech": tech}
    target_expression = torch.rand(n_query, n_genes)
    return {"context": context, "query": query, "target_expression": target_expression}


def test_fm_ot_gene_attention_end_to_end():
    """Uses a genuinely RESTRICTED target panel (12 of the 20 training
    genes, not all of them) — the real, load-bearing case
    _slice_target_for_decoder exists for. An earlier version of this test
    accidentally used a full-width panel, which never exercised the
    shape-mismatch bug this fix addresses (recon width != target_expression
    width whenever gene_attention's panel is smaller than n_genes)."""
    torch.manual_seed(0)
    full_gene_names = _GENES[:20]
    target_panel = _GENES[:12]  # deliberately a SUBSET, not the full 20
    model = FlowMatchingOT(n_genes=20, coord_dim=3, cond_hidden_dim=32,
                            hidden_dim=64, time_embed_dim=16, n_ode_steps=5,
                            decoder_type="gene_attention", decoder_gene_names=target_panel,
                            full_gene_names=full_gene_names,
                            decoder_attn_n_heads=2, decoder_attn_n_layers=1)
    batch = _make_batch(n_context=40, n_query=10, n_genes=20, coord_dim=3, tech=None)

    out = model.sample(batch["context"], batch["query"])
    assert out["expression"].shape == (10, 12), out["expression"].shape  # restricted width, not 20
    assert torch.isfinite(out["expression"]).all()

    opt = model.configure_optimizers()
    model.optimizers = lambda: opt
    model.manual_backward = lambda loss: loss.backward()
    model.log_dict = lambda *a, **k: None
    loss = model.training_step(batch, batch_idx=0)  # would crash with a shape mismatch if the slicing fix were missing
    opt.zero_grad()
    loss.backward()
    assert torch.isfinite(loss)
    print("[FlowMatchingOT] OK — decoder_type='gene_attention' with a genuinely restricted "
          "panel (12/20 genes) runs sample()/training_step() end-to-end")


def test_fm_ot_gene_attention_requires_full_gene_names():
    gene_names = _GENES[:20]
    try:
        FlowMatchingOT(n_genes=20, coord_dim=3, cond_hidden_dim=32,
                        hidden_dim=64, time_embed_dim=16, n_ode_steps=5,
                        decoder_type="gene_attention", decoder_gene_names=gene_names)
        raise AssertionError("expected an assertion error for missing full_gene_names")
    except AssertionError as e:
        assert "requires full_gene_names" in str(e), e
    print("[FlowMatchingOT] OK — decoder_type='gene_attention' refuses construction without full_gene_names")


def test_slice_target_for_decoder_selects_correct_columns():
    """Direct check of BaseGenerativeModel._slice_target_for_decoder: the
    sliced target must contain exactly the training-panel columns
    corresponding to the decoder's restricted gene panel, in the right
    order — not just the right shape."""
    torch.manual_seed(0)
    full_gene_names = _GENES[:10]
    target_panel = ["gene_7", "gene_2", "gene_9"]  # out of order, real subset
    model = FlowMatchingOT(n_genes=10, coord_dim=3, cond_hidden_dim=16,
                            hidden_dim=32, time_embed_dim=8, n_ode_steps=3,
                            decoder_type="gene_attention", decoder_gene_names=target_panel,
                            full_gene_names=full_gene_names)
    target_expression = torch.arange(50).reshape(5, 10).float()  # column i == value i, easy to verify
    sliced = model._slice_target_for_decoder(target_expression)
    expected = target_expression[:, [7, 2, 9]]
    assert torch.equal(sliced, expected), "sliced columns don't match the decoder's target gene panel"
    print("[_slice_target_for_decoder] OK — selects the correct columns in the correct order")


def test_fm_ot_gene_attention_rejects_oversized_panel_at_construction():
    """The registry-level guard (_build_decoder), not just the class's own
    forward()-time guard — should fail at model construction, not wait
    until the first forward pass."""
    gene_names = [f"g_{i}" for i in range(5000)]  # exceeds MAX_SAFE_PANEL_SIZE=4096
    try:
        FlowMatchingOT(n_genes=5000, coord_dim=3, cond_hidden_dim=32,
                        hidden_dim=64, time_embed_dim=16, n_ode_steps=5,
                        decoder_type="gene_attention", decoder_gene_names=gene_names)
        raise AssertionError("expected an assertion error for an oversized gene_attention panel")
    except AssertionError as e:
        assert "MAX_SAFE_PANEL_SIZE" in str(e), e
    print("[FlowMatchingOT] OK — _build_decoder rejects an oversized gene_attention panel at construction time")


def test_fm_ot_lloki_end_to_end():
    torch.manual_seed(0)
    model = FlowMatchingOT(n_genes=20, coord_dim=3, cond_hidden_dim=32,
                            hidden_dim=64, time_embed_dim=16, n_ode_steps=5,
                            decoder_type="lloki", tech_vocab=["Visium", "Xenium"],
                            decoder_lloki_tech_embed_dim=6, decoder_lloki_hidden_dims=[32, 16])
    batch = _make_batch(n_context=40, n_query=10, n_genes=20, coord_dim=3, tech="Visium")

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
    print("[FlowMatchingOT] OK — decoder_type='lloki' runs sample()/training_step() end-to-end")


def test_wae_gan_lloki_end_to_end():
    """Regression test for a real pre-existing bug found while wiring this
    in (2026-07-17): WAEGAN.training_step called self.decoder(...) directly
    instead of through the self._decode(...) dispatcher every other call
    site (and WAEGAN's own sample()) already used — silently skipped
    `tech` conditioning during training for every decoder_type, and would
    have crashed outright for decoder_type='lloki' specifically (its
    forward() requires `tech` as a non-optional argument). Fixed alongside
    this decoder's own implementation, not a pre-existing test gap this
    file happens to newly cover."""
    torch.manual_seed(0)
    model = WAEGAN(n_genes=20, coord_dim=3, latent_dim=8, hidden_dim=32,
                    cond_hidden_dim=32, disc_hidden_dim=16,
                    decoder_type="lloki", tech_vocab=["Visium", "Xenium"],
                    decoder_lloki_tech_embed_dim=6, decoder_lloki_hidden_dims=[32, 16])
    batch = _make_batch(n_context=40, n_query=10, n_genes=20, coord_dim=3, tech="Visium")

    out = model.sample(batch["context"], batch["query"])
    assert out["expression"].shape == (10, 20), out["expression"].shape
    assert torch.isfinite(out["expression"]).all()

    opt_ae, opt_disc = model.configure_optimizers()
    model.optimizers = lambda: (opt_ae, opt_disc)
    model.manual_backward = lambda loss: loss.backward()
    model.log_dict = lambda *a, **k: None
    model.training_step(batch, batch_idx=0)  # would raise TypeError pre-fix (missing required 'tech')
    print("[WAEGAN] OK — decoder_type='lloki' runs sample()/training_step() end-to-end "
          "(regression check for the missed _decode call site)")


if __name__ == "__main__":
    test_gene_attention_shape_and_finiteness()
    test_gene_attention_requires_explicit_gene_names()
    test_gene_attention_panel_size_guard()
    test_gene_attention_genes_interact_via_self_attention()
    test_gene_attention_tech_conditioning_changes_output()
    test_gene_attention_gradient_flow()
    test_lloki_shape_and_finiteness()
    test_lloki_requires_tech_vocab_at_construction()
    test_lloki_unknown_tech_raises()
    test_lloki_tech_conditioning_changes_output()
    test_lloki_gradient_flow()
    test_fm_ot_gene_attention_end_to_end()
    test_fm_ot_gene_attention_requires_full_gene_names()
    test_slice_target_for_decoder_selects_correct_columns()
    test_fm_ot_gene_attention_rejects_oversized_panel_at_construction()
    test_fm_ot_lloki_end_to_end()
    test_wae_gan_lloki_end_to_end()
    print("\nAll GeneAttentionDecoder/LLOKIStyleDecoder tests passed.")
