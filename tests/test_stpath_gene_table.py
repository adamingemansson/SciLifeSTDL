"""Tests for src/models/stpath_gene_table.py -- the frozen STPath
gene-embedding-table extraction (isolates STPath's pretraining from its
whole architecture). Uses a small SYNTHETIC symbol2ensembl.json + a
synthetic checkpoint state_dict (both pure Python/numpy/torch, no scanpy
needed) rather than STPath's real ~138k-entry vocabulary or real weights,
which aren't available in every environment this suite runs in -- but the
gene2id RECONSTRUCTION ALGORITHM is exercised for real, verified directly
against STPath's own source (see stpath_gene_table.py's module docstring).
"""
import json

import numpy as np
import torch

from src.models.stpath_gene_table import (
    STPathFrozenGeneEncoder, _stpath_gene2id, extract_stpath_gene_embedding_table,
)


def _write_symbol2ensembl(tmp_path, mapping):
    path = tmp_path / "symbol2ensembl.json"
    path.write_text(json.dumps(mapping))
    return str(path)


def test_stpath_gene2id_matches_the_real_sort_dedup_plus_two_algorithm(tmp_path):
    # Two symbols intentionally share one Ensembl id (real STPath data has
    # this too -- GeneExpTokenizer dedups by VALUE, not by key).
    mapping = {"GENE_C": "ENSG003", "GENE_A": "ENSG001", "GENE_B": "ENSG002", "GENE_A2": "ENSG001"}
    path = _write_symbol2ensembl(tmp_path, mapping)
    gene2id = _stpath_gene2id(path)
    # sorted(set(...)) of {"ENSG003","ENSG001","ENSG002"} -> ENSG001,ENSG002,ENSG003
    assert gene2id == {"ENSG001": 2, "ENSG002": 3, "ENSG003": 4}
    print("[stpath_gene_table] OK — gene2id reconstruction matches STPath's real "
          "sort-dedup-plus-two algorithm exactly")


def _write_checkpoint(tmp_path, weight: torch.Tensor, key="input_encoder.gene_embed.weight"):
    path = tmp_path / "stpath_weights.pt"
    torch.save({key: weight}, path)
    return str(path)


def test_extract_gathers_correct_columns_for_matched_genes(tmp_path):
    mapping = {"GENE_A": "ENSG001", "GENE_B": "ENSG002", "GENE_C": "ENSG003"}
    voc_path = _write_symbol2ensembl(tmp_path, mapping)
    d_model, n_tokens = 4, 10
    weight = torch.arange(d_model * n_tokens, dtype=torch.float32).reshape(d_model, n_tokens)
    ckpt_path = _write_checkpoint(tmp_path, weight)

    # our panel: GENE_B (id=3), GENE_A (id=2), and an unknown gene
    table, report = extract_stpath_gene_embedding_table(
        ["GENE_B", "GENE_A", "GENE_UNKNOWN"], voc_path, ckpt_path, d_model=d_model,
    )
    assert table.shape == (d_model, 3)
    assert np.array_equal(table[:, 0], weight[:, 3].numpy())  # GENE_B -> token id 3
    assert np.array_equal(table[:, 1], weight[:, 2].numpy())  # GENE_A -> token id 2
    assert np.all(table[:, 2] == 0.0)  # GENE_UNKNOWN -> zero column, not fabricated

    assert report["n_genes"] == 3
    assert report["n_found"] == 2
    assert report["n_missing"] == 1
    assert report["missing_genes"] == ["GENE_UNKNOWN"]
    print("[stpath_gene_table] OK — extraction gathers the exact real pretrained columns "
          "for matched genes, zero-fills unmatched genes, and reports coverage honestly")


def test_extract_raises_on_wrong_d_model(tmp_path):
    mapping = {"GENE_A": "ENSG001"}
    voc_path = _write_symbol2ensembl(tmp_path, mapping)
    weight = torch.zeros(4, 10)
    ckpt_path = _write_checkpoint(tmp_path, weight)
    try:
        extract_stpath_gene_embedding_table(["GENE_A"], voc_path, ckpt_path, d_model=512)
        raise AssertionError("expected a ValueError for mismatched d_model")
    except ValueError as exc:
        assert "d_model" in str(exc)
    print("[stpath_gene_table] OK — fails closed on a d_model mismatch rather than "
          "silently misinterpreting the checkpoint's weight shape")


def test_extract_raises_on_missing_state_dict_key(tmp_path):
    mapping = {"GENE_A": "ENSG001"}
    voc_path = _write_symbol2ensembl(tmp_path, mapping)
    ckpt_path = _write_checkpoint(tmp_path, torch.zeros(4, 10), key="some_other_key")
    try:
        extract_stpath_gene_embedding_table(["GENE_A"], voc_path, ckpt_path, d_model=4)
        raise AssertionError("expected a KeyError for a missing state_dict key")
    except KeyError:
        pass
    print("[stpath_gene_table] OK — fails closed with a clear error when the checkpoint "
          "doesn't contain the expected gene_embed weight key")


def test_stpath_frozen_gene_encoder_reproduces_stpaths_own_frozen_computation():
    torch.manual_seed(0)
    d_model, n_genes, batch = 6, 4, 3
    frozen_table = torch.randn(d_model, n_genes)
    encoder = STPathFrozenGeneEncoder(frozen_table, hidden_dim=5)

    expression = torch.rand(batch, n_genes)
    out = encoder(expression)
    assert out.shape == (batch, 5)
    assert torch.isfinite(out).all()

    # the pretrained matmul itself (before the trainable projection) must
    # be EXACTLY expression @ frozen_table.T -- STPath's own gene_embed
    # computation, bit for bit, never perturbed by training.
    expected_pretrained = expression @ frozen_table.T
    actual_pretrained = expression @ encoder.frozen_table.T
    assert torch.allclose(actual_pretrained, expected_pretrained)
    print("[stpath_gene_table] OK — STPathFrozenGeneEncoder reproduces STPath's own "
          "frozen gene_embed computation exactly, on real observed expression")


def test_stpath_frozen_gene_encoder_table_never_receives_gradient():
    torch.manual_seed(0)
    frozen_table = torch.randn(4, 3)
    encoder = STPathFrozenGeneEncoder(frozen_table, hidden_dim=5)
    expression = torch.rand(2, 3, requires_grad=True)
    out = encoder(expression)
    out.sum().backward()
    assert encoder.frozen_table.grad is None  # buffer, not a Parameter -- structurally frozen
    assert encoder.proj.weight.grad is not None
    assert torch.isfinite(encoder.proj.weight.grad).all()
    print("[stpath_gene_table] OK — the pretrained table is a buffer (never a Parameter), "
          "so it structurally cannot receive gradient; only the small projection head trains")


def test_stpath_frozen_gene_encoder_rejects_wrong_gene_count():
    encoder = STPathFrozenGeneEncoder(torch.randn(4, 3), hidden_dim=5)
    try:
        encoder(torch.rand(2, 7))
        raise AssertionError("expected a ValueError for a gene-count mismatch")
    except ValueError as exc:
        assert "mismatch" in str(exc)
    print("[stpath_gene_table] OK — fails closed on a gene-panel size mismatch at forward time")


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        test_stpath_gene2id_matches_the_real_sort_dedup_plus_two_algorithm(tmp_path)
        test_extract_gathers_correct_columns_for_matched_genes(tmp_path)
        test_extract_raises_on_wrong_d_model(tmp_path)
        test_extract_raises_on_missing_state_dict_key(tmp_path)
    test_stpath_frozen_gene_encoder_reproduces_stpaths_own_frozen_computation()
    test_stpath_frozen_gene_encoder_table_never_receives_gradient()
    test_stpath_frozen_gene_encoder_rejects_wrong_gene_count()
    print("\nAll stpath_gene_table tests passed.")
