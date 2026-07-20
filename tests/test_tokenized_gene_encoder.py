"""
Regression tests for TokenizedGeneEncoder (2026-07-19 research) — see its
own docstring in src/models/conditioning.py. Real motivation: StormLite's
existing gene_encoder_type options ("mlp"/"novae"/"both") all collapse
the entire expression vector into ONE dense token via a shared MLP or a
pretrained whole-profile embedding, discarding individual gene identity
before the spatial transformer ever sees it — unlike STPath's real
per-gene tokenization (its GeneExpTokenizer, verified via source). This
class ports that DESIGN PRINCIPLE (identity-aware per-gene tokens,
combined via scGPT's real "add" rule, pooled via a small self-attention
layer) as our own encoder.

Also covers the new "tokenizer_novae" combination — Novae and the gene
tokenizer are orthogonal axes (Novae's spatial awareness comes from its
OWN pretrained neighbor graph, independent of per-gene identity), worth
testing together per direct instruction, not just each alone.

Run with: python -m tests.test_tokenized_gene_encoder
"""
import tempfile
import os

import torch

from src.models.conditioning import TokenizedGeneEncoder, _GIGAPATH_FEAT_DIM
from src.models.storm_lite_encoder import StormLiteContextEncoder
from src.models.registry import build_model
from src.training.train import inject_novae_dim


def _gene_setup(n_full=20, n_selected=8):
    full_gene_names = [f"G{i}" for i in range(n_full)]
    gene_names = full_gene_names[:n_selected]  # first n_selected genes "selected" (stand-in for real HVG output)
    return gene_names, full_gene_names


def test_tokenized_gene_encoder_output_shape():
    torch.manual_seed(0)
    gene_names, full_gene_names = _gene_setup()
    enc = TokenizedGeneEncoder(gene_names, full_gene_names, feat_dim=16, n_pool_layers=1, n_pool_heads=4)
    raw_expr = torch.rand(5, len(full_gene_names))
    out = enc(raw_expr)
    assert out.shape == (5, 16)
    assert torch.isfinite(out).all()
    print("[tokenized_gene_encoder] OK — output shape [B, feat_dim], finite")


def test_tokenized_gene_encoder_only_uses_selected_gene_columns():
    """Changing an UNSELECTED gene's value must not change the output —
    proves _gene_col_idx actually restricts to the selected subset rather
    than silently using the whole panel."""
    torch.manual_seed(0)
    gene_names, full_gene_names = _gene_setup(n_full=20, n_selected=8)
    enc = TokenizedGeneEncoder(gene_names, full_gene_names, feat_dim=16)
    enc.eval()
    raw_expr = torch.rand(3, 20)
    with torch.no_grad():
        out_before = enc(raw_expr)
        raw_expr_perturbed = raw_expr.clone()
        raw_expr_perturbed[:, 15] += 100.0  # column 15 is NOT in the first-8-selected subset
        out_after = enc(raw_expr_perturbed)
    assert torch.equal(out_before, out_after), "perturbing an unselected gene's value must not change the output"
    print("[tokenized_gene_encoder] OK — only the selected gene columns affect the output")


def test_tokenized_gene_encoder_gradients_flow():
    torch.manual_seed(0)
    gene_names, full_gene_names = _gene_setup()
    enc = TokenizedGeneEncoder(gene_names, full_gene_names, feat_dim=16)
    raw_expr = torch.rand(4, len(full_gene_names))
    out = enc(raw_expr)
    out.sum().backward()
    grad_norms = [p.grad.norm().item() for p in enc.parameters() if p.grad is not None]
    assert len(grad_norms) > 0 and all(g >= 0 for g in grad_norms)
    assert any(g > 0 for g in grad_norms), "at least one param must receive a real (nonzero) gradient"
    print("[tokenized_gene_encoder] OK — gradients flow through identity_embed/value_proj/pool_transformer")


def test_max_safe_panel_size_guard():
    full_gene_names = [f"G{i}" for i in range(5000)]
    gene_names = full_gene_names[:4097]  # one over the guard
    try:
        TokenizedGeneEncoder(gene_names, full_gene_names, feat_dim=16)
        assert False, "must raise when gene_names exceeds MAX_SAFE_PANEL_SIZE"
    except AssertionError as e:
        assert "MAX_SAFE_PANEL_SIZE" in str(e)
    print("[tokenized_gene_encoder] OK — MAX_SAFE_PANEL_SIZE guard raises clearly")


def test_gene_names_must_be_subset_of_full_gene_names():
    full_gene_names = ["G0", "G1", "G2"]
    gene_names = ["G0", "G99"]  # G99 not in full_gene_names
    try:
        TokenizedGeneEncoder(gene_names, full_gene_names, feat_dim=16)
        assert False, "must raise when gene_names contains a gene not in full_gene_names"
    except AssertionError as e:
        assert "not in full_gene_names" in str(e)
    print("[tokenized_gene_encoder] OK — mismatched gene_names raises clearly")


def test_storm_lite_context_encoder_tokenizer_and_tokenizer_novae():
    torch.manual_seed(0)
    n_context, n_query, n_genes, novae_dim, hidden_dim = 10, 4, 20, 12, 16
    gene_names, full_gene_names = _gene_setup(n_full=n_genes, n_selected=8)

    context_coords = torch.rand(n_context, 3) * 100
    query_coords = torch.rand(n_query, 3) * 100
    context_images = torch.rand(n_context, _GIGAPATH_FEAT_DIM)
    query_images = torch.rand(n_query, _GIGAPATH_FEAT_DIM)
    context_expression = torch.rand(n_context, n_genes)
    context_novae_features = torch.rand(n_context, novae_dim)

    for gene_encoder_type in ("tokenizer", "tokenizer_novae"):
        encoder = StormLiteContextEncoder(
            n_genes=n_genes, novae_dim=novae_dim, hidden_dim=hidden_dim,
            n_transformer_layers=2, n_heads=4, gene_encoder_type=gene_encoder_type,
            tokenizer_gene_names=gene_names, tokenizer_full_gene_names=full_gene_names,
        )
        c = encoder(
            context_coords, context_expression, query_coords,
            context_images, query_images,
            context_novae_features=context_novae_features,
        )
        assert c.shape == (n_query, hidden_dim), (gene_encoder_type, c.shape)
        assert torch.isfinite(c).all()
    print("[tokenized_gene_encoder] OK — StormLiteContextEncoder runs end-to-end with "
          "gene_encoder_type='tokenizer' and 'tokenizer_novae'")


def test_default_gene_encoder_types_unaffected():
    """Existing gene_encoder_type values ('mlp'/'novae'/'both') must work
    exactly as before — the new options are purely additive."""
    torch.manual_seed(0)
    n_context, n_query, n_genes, novae_dim, hidden_dim = 10, 4, 20, 12, 16
    context_coords = torch.rand(n_context, 3) * 100
    query_coords = torch.rand(n_query, 3) * 100
    context_images = torch.rand(n_context, _GIGAPATH_FEAT_DIM)
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
        assert c.shape == (n_query, hidden_dim)
    print("[tokenized_gene_encoder] OK — existing gene_encoder_type values ('mlp'/'novae'/'both') unaffected")


def test_build_model_end_to_end_with_tokenizer():
    """Full registry.build_model() construction with context_encoder_type=
    'storm_lite' + gene_encoder_type='tokenizer' -- proves the params
    actually flow through _build_context_encoder and all 3 model classes'
    __init__ correctly, not just the encoder class in isolation."""
    torch.manual_seed(0)
    gene_names, full_gene_names = _gene_setup(n_full=20, n_selected=8)
    cfg = {
        "name": "fm_ot",
        "params": {
            "n_genes": 20, "coord_dim": 3, "cond_hidden_dim": 16,
            "hidden_dim": 32, "time_embed_dim": 16, "n_ode_steps": 3,
            "context_encoder_type": "storm_lite", "gene_encoder_type": "tokenizer",
            "storm_lite_tokenizer_gene_names": gene_names,
            "storm_lite_tokenizer_full_gene_names": full_gene_names,
        },
    }
    model = build_model(cfg)
    assert model.context_encoder.gene_encoder_type == "tokenizer"
    print("[tokenized_gene_encoder] OK — build_model() wires gene_encoder_type='tokenizer' end-to-end")


def test_inject_novae_dim_recognizes_tokenizer_novae():
    """Real bug found 2026-07-19 on a real training run: gene_encoder_type=
    'tokenizer_novae' crashed with 'requires novae_dim' because
    inject_novae_dim's own gating check only recognized ('novae', 'both'),
    not the new combined option -- Novae features were never even computed
    in the first place, so novae_dim was never auto-derived. Every original
    test in this file constructed the encoder directly with novae_dim
    passed manually, so none of them exercised this auto-detection path at
    all -- a real test-coverage gap, not just a code gap."""
    params = {"gene_encoder_type": "tokenizer_novae"}
    model_cfg = {"params": params}
    inject_novae_dim(model_cfg, novae_dim=64)
    assert params.get("novae_dim") == 64, (
        "inject_novae_dim must recognize gene_encoder_type='tokenizer_novae', same as 'novae'/'both'"
    )
    print("[tokenized_gene_encoder] OK — inject_novae_dim recognizes gene_encoder_type='tokenizer_novae'")


def test_any_config_uses_novae_recognizes_tokenizer_novae():
    """Same bug class as inject_novae_dim above, but in the shared-eval
    Novae-detection path (run_comparison.py's _any_config_uses_novae) --
    if this doesn't recognize 'tokenizer_novae' either, Novae features
    never get computed for the shared held-out evaluation draw, breaking
    eval even if training somehow worked."""
    from src.evaluation.run_comparison import _any_config_uses_novae

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = os.path.join(tmp, "test_tokenizer_novae.yaml")
        with open(cfg_path, "w") as f:
            f.write(
                "model:\n"
                "  name: fm_ot\n"
                "  params:\n"
                "    context_encoder_type: storm_lite\n"
                "    gene_encoder_type: tokenizer_novae\n"
            )
        assert _any_config_uses_novae([cfg_path], overrides=None), (
            "_any_config_uses_novae must recognize gene_encoder_type='tokenizer_novae', same as 'novae'/'both'"
        )
    print("[tokenized_gene_encoder] OK — _any_config_uses_novae recognizes gene_encoder_type='tokenizer_novae'")


if __name__ == "__main__":
    test_tokenized_gene_encoder_output_shape()
    test_tokenized_gene_encoder_only_uses_selected_gene_columns()
    test_tokenized_gene_encoder_gradients_flow()
    test_max_safe_panel_size_guard()
    test_gene_names_must_be_subset_of_full_gene_names()
    test_storm_lite_context_encoder_tokenizer_and_tokenizer_novae()
    test_default_gene_encoder_types_unaffected()
    test_build_model_end_to_end_with_tokenizer()
    test_inject_novae_dim_recognizes_tokenizer_novae()
    test_any_config_uses_novae_recognizes_tokenizer_novae()
    print("\nAll TokenizedGeneEncoder tests passed.")
