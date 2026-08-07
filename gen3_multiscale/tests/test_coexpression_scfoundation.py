"""Tests for conditional_wae/coexpression_scfoundation.py: extracting a
per-gene embedding table from a (faked, no real checkpoint needed)
scFoundation encoder and rank-reducing it into a GeneResidualBasis via
the SAME fit_gene_residual_basis the from-scratch source uses."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gen3_multiscale.conditional_wae.coexpression import load_conditional_wae_gene_coexpression_basis, \
    save_conditional_wae_gene_coexpression_basis
from gen3_multiscale.conditional_wae.coexpression_scfoundation import (
    _REQUIRED_METADATA_FIELDS_SCFOUNDATION,
    extract_scfoundation_gene_embedding_table,
    fit_conditional_wae_gene_coexpression_basis_from_scfoundation,
)
from gen3_multiscale.gen4.providers import EncoderIdentity


def _fake_identity() -> EncoderIdentity:
    return EncoderIdentity(
        encoder_name="scfoundation", checkpoint_sha256="a" * 64, pinned_revision="b" * 40,
        package_version="git:c" * 8, preprocessing_spec="stub_scfoundation_v1", output_dim=3072,
    )


def _fake_encoder(gene_names, scfoundation_vocab, embedding_dim=8, seed=0):
    """Mirrors FrozenSCFoundationEncoder's public contract this module
    reads from -- .gene_names, .scfoundation_vocab, .model.pos_emb.weight,
    .identity -- without loading any real checkpoint."""
    rng = np.random.default_rng(seed)
    # +2 rows for the appended resolution/depth tokens, matching the real
    # class's own vocab-size-plus-two pos_emb convention.
    pos_emb_weight = torch.as_tensor(
        rng.normal(size=(len(scfoundation_vocab) + 2, embedding_dim)).astype(np.float32),
    )
    model = SimpleNamespace(pos_emb=torch.nn.Embedding.from_pretrained(pos_emb_weight, freeze=True))
    return SimpleNamespace(
        gene_names=tuple(gene_names), scfoundation_vocab=list(scfoundation_vocab),
        model=model, identity=_fake_identity(),
    )


# -- extract_scfoundation_gene_embedding_table --------------------------------

def test_extract_returns_the_right_shape_and_orientation():
    gene_names = ["G0", "G1", "G2"]
    vocab = ["G0", "G1", "G2", "OTHER"]
    encoder = _fake_encoder(gene_names, vocab, embedding_dim=6)
    table, report = extract_scfoundation_gene_embedding_table(encoder)
    assert table.shape == (6, 3)
    assert report["n_genes"] == 3
    assert report["n_found"] == 3
    assert report["n_missing"] == 0
    assert report["missing_genes"] == []
    assert report["embedding_dim"] == 6


def test_extract_zero_fills_and_reports_genes_missing_from_the_vocabulary():
    gene_names = ["G0", "G1", "MISSING_GENE"]
    vocab = ["G0", "G1"]
    encoder = _fake_encoder(gene_names, vocab, embedding_dim=4)
    table, report = extract_scfoundation_gene_embedding_table(encoder)
    assert report["n_found"] == 2
    assert report["n_missing"] == 1
    assert report["missing_genes"] == ["MISSING_GENE"]
    np.testing.assert_array_equal(table[:, 2], np.zeros(4, dtype=np.float32))


def test_extract_uses_the_real_pretrained_embedding_for_a_matched_gene():
    gene_names = ["G0"]
    vocab = ["OTHER0", "G0", "OTHER1"]
    encoder = _fake_encoder(gene_names, vocab, embedding_dim=5)
    table, _report = extract_scfoundation_gene_embedding_table(encoder)
    expected = encoder.model.pos_emb.weight[1].detach().numpy()  # G0 is vocab index 1
    np.testing.assert_allclose(table[:, 0], expected)


def test_extract_never_treats_the_two_appended_resolution_tokens_as_genes():
    """The last two pos_emb rows are the resolution/depth tokens, never a
    real gene -- a gene name colliding with a vocab index beyond the real
    gene count must still be excluded."""
    gene_names = ["G0"]
    vocab = ["G0"]  # embedding_dim+2 rows exist in pos_emb, but only index 0 is a real gene
    encoder = _fake_encoder(gene_names, vocab, embedding_dim=3)
    table, report = extract_scfoundation_gene_embedding_table(encoder)
    assert report["n_found"] == 1
    expected = encoder.model.pos_emb.weight[0].detach().numpy()
    np.testing.assert_allclose(table[:, 0], expected)


def test_extract_rejects_zero_overlap_with_the_vocabulary():
    gene_names = ["NOT_IN_VOCAB"]
    vocab = ["G0", "G1"]
    encoder = _fake_encoder(gene_names, vocab)
    with pytest.raises(ValueError, match="none of the"):
        extract_scfoundation_gene_embedding_table(encoder)


# -- fit_conditional_wae_gene_coexpression_basis_from_scfoundation -----------

def test_fit_produces_a_valid_gene_residual_basis():
    gene_names = [f"G{i}" for i in range(6)]
    vocab = gene_names
    encoder = _fake_encoder(gene_names, vocab, embedding_dim=10)
    table, report = extract_scfoundation_gene_embedding_table(encoder)
    basis, metadata = fit_conditional_wae_gene_coexpression_basis_from_scfoundation(
        table, gene_names, encoder.identity, report, rank=4, seed=0,
    )
    assert basis.rank == 4
    assert basis.gene_names == tuple(gene_names)
    assert set(_REQUIRED_METADATA_FIELDS_SCFOUNDATION).issubset(metadata)
    assert metadata["kind"] == "conditional_wae_gene_coexpression_basis_scfoundation"
    assert metadata["scfoundation_checkpoint_sha256"] == encoder.identity.checkpoint_sha256
    assert metadata["n_genes_found_in_scfoundation_vocab"] == 6
    assert metadata["n_genes_missing_from_scfoundation_vocab"] == 0


def test_fit_is_deterministic_given_the_same_embedding_table_and_seed():
    gene_names = [f"G{i}" for i in range(6)]
    encoder = _fake_encoder(gene_names, gene_names, embedding_dim=10)
    table, report = extract_scfoundation_gene_embedding_table(encoder)
    basis_a, _ = fit_conditional_wae_gene_coexpression_basis_from_scfoundation(
        table, gene_names, encoder.identity, report, rank=4, seed=0,
    )
    basis_b, _ = fit_conditional_wae_gene_coexpression_basis_from_scfoundation(
        table, gene_names, encoder.identity, report, rank=4, seed=0,
    )
    assert torch.equal(basis_a.basis, basis_b.basis)


def test_fit_rejects_a_gene_dimension_mismatch():
    gene_names = ["G0", "G1", "G2"]
    wrong_shape_table = np.zeros((5, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="expected"):
        fit_conditional_wae_gene_coexpression_basis_from_scfoundation(
            wrong_shape_table, gene_names, _fake_identity(), {"n_found": 0, "n_missing": 3, "missing_genes": gene_names},
        )


def test_fit_rejects_non_finite_embedding_values():
    gene_names = ["G0", "G1"]
    bad_table = np.full((3, 2), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        fit_conditional_wae_gene_coexpression_basis_from_scfoundation(
            bad_table, gene_names, _fake_identity(), {"n_found": 2, "n_missing": 0, "missing_genes": []},
        )


# -- round trip through the shared save/load machinery ------------------------

def test_save_then_load_round_trips_through_the_shared_coexpression_loader(tmp_path):
    """The generic loader (used identically by _build_model for every
    coexpression arm) must accept a scFoundation-sourced basis file even
    though its metadata shape differs from the from-scratch source's."""
    gene_names = [f"G{i}" for i in range(6)]
    encoder = _fake_encoder(gene_names, gene_names, embedding_dim=10)
    table, report = extract_scfoundation_gene_embedding_table(encoder)
    basis, metadata = fit_conditional_wae_gene_coexpression_basis_from_scfoundation(
        table, gene_names, encoder.identity, report, rank=4, seed=0,
    )
    path = save_conditional_wae_gene_coexpression_basis(
        basis, metadata, tmp_path / "basis_scfoundation.pt", required_fields=_REQUIRED_METADATA_FIELDS_SCFOUNDATION,
    )
    loaded_basis, loaded_metadata = load_conditional_wae_gene_coexpression_basis(str(path), gene_names)
    assert torch.equal(loaded_basis.basis, basis.basis)
    assert loaded_metadata["kind"] == "conditional_wae_gene_coexpression_basis_scfoundation"


def test_save_rejects_scfoundation_metadata_missing_a_required_field(tmp_path):
    gene_names = [f"G{i}" for i in range(6)]
    encoder = _fake_encoder(gene_names, gene_names, embedding_dim=10)
    table, report = extract_scfoundation_gene_embedding_table(encoder)
    basis, metadata = fit_conditional_wae_gene_coexpression_basis_from_scfoundation(
        table, gene_names, encoder.identity, report, rank=4, seed=0,
    )
    del metadata["scfoundation_checkpoint_sha256"]
    with pytest.raises(ValueError, match="missing required field"):
        save_conditional_wae_gene_coexpression_basis(
            basis, metadata, tmp_path / "basis.pt", required_fields=_REQUIRED_METADATA_FIELDS_SCFOUNDATION,
        )
