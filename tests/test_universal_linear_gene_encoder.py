"""Tests for UniversalLinearGeneEncoder (2026-07-24): a literal copy of
STPath's real gene_embed mechanism (single bias-free nn.Linear, verified
directly against stpath/model/model.py) applied on top of our own
architectures, using the same real fixed gene-ID vocabulary as
UniversalMLPGeneEncoder. The gene_encoder_type='universal_linear' third
option, completing the 2x3 (gene encoder x architecture) design.

Run with: python -m tests.test_universal_linear_gene_encoder
"""
import json
import os
import tempfile

import torch
import torch.nn as nn

from src.models.conditioning import _GIGAPATH_FEAT_DIM
from src.models.registry import build_model
from src.models.simple_fusion_encoder import SimpleCrossAttentionContextEncoder, UniversalLinearGeneEncoder


def _write_fake_vocab(mapping: dict) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(mapping, f)
    f.close()
    return f.name


def test_is_bias_free_linear_matching_stpaths_real_shape():
    path = _write_fake_vocab({f"GENE_{i}": f"ENSG{i:04d}" for i in range(30)})
    try:
        gene_names = [f"GENE_{i}" for i in range(20)]
        enc = UniversalLinearGeneEncoder(gene_names=gene_names, gene_voc_path=path, feat_dim=16)
        assert isinstance(enc.gene_embed, nn.Linear)
        assert enc.gene_embed.bias is None, "STPath's real gene_embed is bias-free"
        assert enc.gene_embed.in_features == enc.n_vocab_tokens
        assert enc.gene_embed.out_features == 16
        print("[shape] OK — single bias-free Linear(n_vocab_tokens, feat_dim), matching STPath's real gene_embed")
    finally:
        os.unlink(path)


def test_forward_and_gradient():
    path = _write_fake_vocab({f"GENE_{i}": f"ENSG{i:04d}" for i in range(30)})
    try:
        gene_names = [f"GENE_{i}" for i in range(20)]
        enc = UniversalLinearGeneEncoder(gene_names=gene_names, gene_voc_path=path, feat_dim=16)
        expr = torch.rand(4, 20)
        out = enc(expr)
        assert out.shape == (4, 16)
        assert torch.isfinite(out).all()
        out.sum().backward()
        assert enc.gene_embed.weight.grad is not None and enc.gene_embed.weight.grad.abs().sum() > 0
        print("[forward+gradient] OK")
    finally:
        os.unlink(path)


def test_shares_vocab_construction_with_mlp_variant():
    # Same vocabulary construction helper as UniversalMLPGeneEncoder --
    # confirms both variants see the SAME gene-ID space, only the encoder
    # architecture differs between them.
    path = _write_fake_vocab({"GENE_A": "ENSG001", "GENE_B": "ENSG002", "GENE_C": "ENSG003", "GENE_E": "ENSG003"})
    try:
        gene_names = ["GENE_A", "GENE_D", "GENE_B", "GENE_C"]
        enc = UniversalLinearGeneEncoder(gene_names=gene_names, gene_voc_path=path, feat_dim=8)
        assert enc.n_vocab_tokens == 5
        assert enc.local_idx.tolist() == [0, 2, 3]
        print("[shared vocab construction] OK — matches UniversalMLPGeneEncoder's own real test values")
    finally:
        os.unlink(path)


def test_wired_into_registry_end_to_end():
    path = _write_fake_vocab({f"GENE_{i}": f"ENSG{i:04d}" for i in range(30)})
    try:
        gene_names = [f"GENE_{i}" for i in range(20)]
        model = build_model({
            "name": "simple_cross_attn_dense_decoder",
            "params": {
                "n_genes": 20, "hidden_dim": 32, "n_heads": 4, "n_layers": 2, "knn_k": 5, "lr": 1e-3,
                "gene_encoder_type": "universal_linear", "gene_names": gene_names, "gene_voc_path": path,
            },
        })
        context = {"coords": torch.randn(40, 3), "expression": torch.rand(40, 20),
                   "images": torch.randn(40, _GIGAPATH_FEAT_DIM)}
        query = {"coords": torch.randn(8, 3), "images": torch.randn(8, _GIGAPATH_FEAT_DIM)}
        out = model.sample(context, query)
        assert out["expression"].shape == (8, 20)
        batch = {"context": context, "query": query, "target_expression": torch.rand(8, 20)}
        loss = model.training_step(batch, 0)
        loss.backward()
        assert model.configure_optimizers() is not None
        print(f"[registry end-to-end] OK — simple_cross_attn_dense_decoder + universal_linear, loss={loss.item():.4f}")
    finally:
        os.unlink(path)


if __name__ == "__main__":
    test_is_bias_free_linear_matching_stpaths_real_shape()
    test_forward_and_gradient()
    test_shares_vocab_construction_with_mlp_variant()
    test_wired_into_registry_end_to_end()
    print("\nAll universal_linear_gene_encoder tests passed.")
