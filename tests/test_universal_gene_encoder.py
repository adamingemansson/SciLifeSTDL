"""Tests for UniversalMLPGeneEncoder (2026-07-24): our own MLPGeneEncoder
(2-layer, LayerNorm+GELU), fed through STPath's real fixed gene-ID
vocabulary space instead of this dataset's local ad-hoc column order.
Vocabulary construction is verified against STPath's real
GeneExpTokenizer logic (stpath/tokenization/ge_tokenizer.py): symbol ->
symbol2gene[symbol] -> gene2id[that value], where gene2id enumerates the
SORTED SET of unique symbol2gene values, offset by 2.

Run with: python -m tests.test_universal_gene_encoder
"""
import json
import os
import tempfile

import torch

from src.models.conditioning import _GIGAPATH_FEAT_DIM
from src.models.registry import build_model
from src.models.simple_fusion_encoder import (
    SimpleCrossAttentionContextEncoder,
    UniversalMLPGeneEncoder,
)


def _write_fake_vocab(mapping: dict) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(mapping, f)
    f.close()
    return f.name


def test_vocab_construction_matches_stpaths_real_scheme():
    # 4 symbols -> 3 unique ensembl ids (GENE_C and GENE_E intentionally
    # share one, same as two symbols aliasing one real gene in STPath's
    # own real symbol2ensembl.json).
    path = _write_fake_vocab({
        "GENE_A": "ENSG001", "GENE_B": "ENSG002", "GENE_C": "ENSG003", "GENE_E": "ENSG003",
    })
    try:
        gene_names = ["GENE_A", "GENE_D", "GENE_B", "GENE_C"]  # GENE_D is out-of-vocabulary
        enc = UniversalMLPGeneEncoder(gene_names=gene_names, gene_voc_path=path, feat_dim=16)
        assert enc.n_vocab_tokens == 5, "3 unique ensembl ids + 2 reserved (pad/mask) offset"
        assert enc.local_idx.tolist() == [0, 2, 3], "GENE_D (index 1) must be excluded, OOV"
        assert sorted(enc.vocab_idx.tolist()) == [2, 3, 4], "ids start at 2, matching STPath's reserved 0/1"
        print("[vocab construction] OK — matches STPath's real sorted-unique-value + offset-2 scheme, "
              "OOV genes correctly excluded")
    finally:
        os.unlink(path)


def test_oov_genes_contribute_nothing():
    path = _write_fake_vocab({"GENE_A": "ENSG001", "GENE_B": "ENSG002"})
    try:
        gene_names = ["GENE_A", "GENE_D", "GENE_B"]
        enc = UniversalMLPGeneEncoder(gene_names=gene_names, gene_voc_path=path, feat_dim=8)
        torch.manual_seed(0)
        base = torch.rand(3, 3)
        perturbed = base.clone()
        perturbed[:, 1] = 999.0  # only touch GENE_D's (out-of-vocab) column
        out_base = enc(base)
        out_perturbed = enc(perturbed)
        assert torch.allclose(out_base, out_perturbed), (
            "an OOV gene's value changed the output — it must be silently dropped, "
            "exactly like STPath's own real OOV handling"
        )
        print("[OOV isolation] OK — an out-of-vocabulary gene's value has zero effect on the output")
    finally:
        os.unlink(path)


def test_gradient_flows_through_scatter():
    path = _write_fake_vocab({f"GENE_{i}": f"ENSG{i:04d}" for i in range(10)})
    try:
        gene_names = [f"GENE_{i}" for i in range(10)]
        enc = UniversalMLPGeneEncoder(gene_names=gene_names, gene_voc_path=path, feat_dim=8, hidden_dim=16, bottleneck_dim=8)
        expr = torch.rand(4, 10, requires_grad=False)
        out = enc(expr)
        assert out.shape == (4, 8)
        out.sum().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())
        print("[gradient flow] OK — gradient reaches the inner MLPGeneEncoder through the scatter")
    finally:
        os.unlink(path)


def test_wired_into_cross_attention_encoder():
    path = _write_fake_vocab({f"GENE_{i}": f"ENSG{i:04d}" for i in range(30)})
    try:
        gene_names = [f"GENE_{i}" for i in range(20)]
        enc = SimpleCrossAttentionContextEncoder(
            n_genes=20, hidden_dim=32, n_heads=4, knn_k=5, n_layers=2,
            gene_encoder_type="universal_mlp", gene_names=gene_names, gene_voc_path=path,
        )
        context_coords, context_expression = torch.randn(40, 3), torch.rand(40, 20)
        context_images = torch.randn(40, _GIGAPATH_FEAT_DIM)
        query_coords, query_images = torch.randn(8, 3), torch.randn(8, _GIGAPATH_FEAT_DIM)
        out = enc(context_coords, context_expression, query_coords, context_images, query_images)
        assert out.shape == (8, 32)
        assert torch.isfinite(out).all()
        print("[wired into SimpleCrossAttentionContextEncoder] OK")
    finally:
        os.unlink(path)


def test_wired_into_registry_model_end_to_end():
    path = _write_fake_vocab({f"GENE_{i}": f"ENSG{i:04d}" for i in range(30)})
    try:
        gene_names = [f"GENE_{i}" for i in range(20)]
        model = build_model({
            "name": "simple_cross_attn_dense_decoder",
            "params": {
                "n_genes": 20, "hidden_dim": 32, "n_heads": 4, "n_layers": 2, "knn_k": 5, "lr": 1e-3,
                "gene_encoder_type": "universal_mlp", "gene_names": gene_names, "gene_voc_path": path,
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
        print(f"[registry end-to-end] OK — simple_cross_attn_dense_decoder + universal_mlp, loss={loss.item():.4f}")
    finally:
        os.unlink(path)


if __name__ == "__main__":
    test_vocab_construction_matches_stpaths_real_scheme()
    test_oov_genes_contribute_nothing()
    test_gradient_flows_through_scatter()
    test_wired_into_cross_attention_encoder()
    test_wired_into_registry_model_end_to_end()
    print("\nAll universal_gene_encoder tests passed.")
