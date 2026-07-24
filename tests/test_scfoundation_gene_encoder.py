"""Tests for the scFoundation gene-encoder integration (2026-07-24).

Covers everything testable WITHOUT the real scFoundation checkpoint/repo
(no CUDA, no ~2GB checkpoint download available in this environment):
ScFoundationGeneEncoder's forward/gradient/shape contract,
_build_gene_encoder's 'scfoundation' dispatch, model_uses_scfoundation's
gating logic, and full end-to-end wiring through both registry models
(simple_cross_attn_dense_decoder, stpath_backbone_simple_gene) using a
FAKE precomputed-features tensor in place of what
precompute_scfoundation_features would really produce.

NOT covered here (needs real hardware): precompute_scfoundation_features
itself against the real cloned repo + downloaded checkpoint on a CUDA
machine -- that function's preprocessing (gene alignment, the two
resolution/depth tokens, gatherData-based sparse tokenization, the real
4-way pooling) was written by directly cloning
github.com/biomap-research/scFoundation and reading get_embedding.py/
load.py/mae_autobin.py verbatim (2026-07-24), not guessed from a
secondhand description, but has NOT been numerically verified against a
real forward pass. Run precompute_scfoundation_features once on the real
training server (with the repo cloned and a checkpoint downloaded) and
sanity-check its output before trusting it in a real training run.

Run with: python -m tests.test_scfoundation_gene_encoder
"""
import numpy as np
import torch
import torch.nn as nn

from src.data.context_features import model_uses_scfoundation
from src.models.conditioning import ScFoundationGeneEncoder, _scfoundation_align_genes
from src.models.registry import build_model
from src.models.simple_fusion_encoder import _build_gene_encoder


def test_scfoundation_align_genes_reorders_and_zero_pads():
    # Real per-spot expression has 3 genes, only 2 of which are in
    # scFoundation's (here, toy) vocabulary; the vocabulary also has an
    # extra gene never measured locally -- must be zero-padded, not
    # dropped, matching scFoundation's own real main_gene_selection
    # contract (fixed-width vocabulary, missing genes = 0).
    expression = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ], dtype=np.float32)
    gene_names = ["GENE_B", "GENE_UNKNOWN_LOCALLY", "GENE_A"]
    vocab_genes = ["GENE_A", "GENE_B", "GENE_C"]

    aligned = _scfoundation_align_genes(expression, gene_names, vocab_genes)
    assert aligned.shape == (2, 3)
    # Row 0: GENE_A=3.0 (from local col 2), GENE_B=1.0 (from local col 0), GENE_C=0.0 (never measured)
    assert np.allclose(aligned[0], [3.0, 1.0, 0.0])
    assert np.allclose(aligned[1], [6.0, 4.0, 0.0])
    print("[_scfoundation_align_genes] OK — reorders onto the real vocabulary, "
          "zero-pads genes never measured locally, drops genes not in the vocabulary")


def test_scfoundation_gene_encoder_shape_and_gradient():
    enc = ScFoundationGeneEncoder(scfoundation_dim=2048, feat_dim=64)
    assert isinstance(enc.embedding_norm, nn.LayerNorm)
    assert isinstance(enc.proj, nn.Linear)
    assert enc.proj.in_features == 2048
    assert enc.proj.out_features == 64

    fake_precomputed = torch.randn(6, 2048)
    out = enc(fake_precomputed)
    assert out.shape == (6, 64)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert enc.proj.weight.grad is not None and enc.proj.weight.grad.abs().sum() > 0
    print("[ScFoundationGeneEncoder] OK — shape (6, 64), gradient reaches the trainable projection")


def test_scfoundation_gene_encoder_rejects_wrong_rank():
    enc = ScFoundationGeneEncoder(scfoundation_dim=128, feat_dim=16)
    try:
        enc(torch.randn(4, 3, 128))
        raise AssertionError("expected ValueError for a non-2D input")
    except ValueError:
        pass
    print("[ScFoundationGeneEncoder] OK — rejects non-[B, scfoundation_dim] input")


def test_build_gene_encoder_scfoundation_dispatch():
    enc = _build_gene_encoder(
        "scfoundation", n_genes=20, feat_dim=32, gene_names=None, gene_voc_path=None,
        scfoundation_dim=2048,
    )
    assert isinstance(enc, ScFoundationGeneEncoder)
    assert enc.proj.in_features == 2048 and enc.proj.out_features == 32

    try:
        _build_gene_encoder(
            "scfoundation", n_genes=20, feat_dim=32, gene_names=None, gene_voc_path=None,
            scfoundation_dim=None,
        )
        raise AssertionError("expected ValueError when scfoundation_dim is missing")
    except ValueError:
        pass
    print("[_build_gene_encoder] OK — 'scfoundation' dispatch builds correctly, "
          "raises without scfoundation_dim")


def test_model_uses_scfoundation():
    assert model_uses_scfoundation({"gene_encoder_type": "scfoundation"}) is True
    assert model_uses_scfoundation({"gene_encoder_type": "local_mlp"}) is False
    assert model_uses_scfoundation({"gene_encoder_type": "scfoundation", "context_encoder_type": "stpath"}) is False
    print("[model_uses_scfoundation] OK — gates correctly on gene_encoder_type/context_encoder_type")


def _run_registry_case(model_name: str, extra_params: dict):
    from src.models.conditioning import _GIGAPATH_FEAT_DIM

    scfoundation_dim = 2048
    n_genes = 20
    model = build_model({
        "name": model_name,
        "params": {
            "n_genes": n_genes, "gene_encoder_type": "scfoundation",
            "scfoundation_dim": scfoundation_dim, "input_already_log1p": True,
            "lr": 1e-3, **extra_params,
        },
    })
    n_context, n_query = 10, 4
    # Simulates what train.py's context_gene_feature_provider mechanism
    # would really substitute into context["expression"] -- a precomputed
    # scFoundation embedding, NOT raw counts (input_already_log1p=True
    # above tells the encoder not to log1p it again).
    context = {
        "coords": torch.randn(n_context, 3),
        "expression": torch.randn(n_context, scfoundation_dim),
        "images": torch.randn(n_context, _GIGAPATH_FEAT_DIM),
    }
    query = {
        "coords": torch.randn(n_query, 3),
        "images": torch.randn(n_query, _GIGAPATH_FEAT_DIM),
    }
    out = model.sample(context, query)
    assert out["expression"].shape == (n_query, n_genes)
    assert torch.isfinite(out["expression"]).all()

    batch = {"context": context, "query": query, "target_expression": torch.rand(n_query, n_genes)}
    loss = model.training_step(batch, 0)
    loss.backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable)
    assert model.configure_optimizers() is not None
    print(f"[{model_name} + scfoundation] OK — sample/training_step/backward/optimizer, loss={loss.item():.4f}")


def test_wired_into_simple_cross_attn_dense_decoder():
    _run_registry_case("simple_cross_attn_dense_decoder", {
        "hidden_dim": 32, "n_heads": 4, "n_layers": 2, "knn_k": 5,
    })


def test_wired_into_stpath_backbone_simple_gene():
    try:
        import stpath  # noqa: F401
    except ImportError:
        print("[stpath_backbone_simple_gene + scfoundation] SKIPPED — stpath package not installed")
        return
    _run_registry_case("stpath_backbone_simple_gene", {
        "hidden_dim": 32, "n_layers": 2, "n_heads": 4,
    })


if __name__ == "__main__":
    test_scfoundation_align_genes_reorders_and_zero_pads()
    test_scfoundation_gene_encoder_shape_and_gradient()
    test_scfoundation_gene_encoder_rejects_wrong_rank()
    test_build_gene_encoder_scfoundation_dispatch()
    test_model_uses_scfoundation()
    test_wired_into_simple_cross_attn_dense_decoder()
    test_wired_into_stpath_backbone_simple_gene()
    print("\nAll scFoundation gene-encoder tests passed (structural — see this file's own "
          "docstring for what still needs real-hardware verification).")
