"""Tests for gen3_multiscale/conditional_wae/coexpression.py: fitting a
low-rank gene-coexpression basis from training-split-only expression,
save/load with fail-closed provenance, and the zero-init-safe
`GeneCoexpressionRefinement` decoder-side module."""
import numpy as np
import pytest
import torch

from gen3_multiscale.conditional_wae.coexpression import (
    GeneCoexpressionRefinement, fit_conditional_wae_gene_coexpression_basis,
    load_conditional_wae_gene_coexpression_basis, save_conditional_wae_gene_coexpression_basis,
)
from gen3_multiscale.conditional_wae.model import Architecture1ImageConditioner, ConditionalWAE
from gen3_multiscale.models.gene_basis import fit_gene_residual_basis


def _synthetic_expression(gene_names, n_spots=20, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_spots, len(gene_names))).astype(np.float32)


# -- fit_conditional_wae_gene_coexpression_basis -----------------------------

def test_fit_pools_expression_from_exactly_the_given_train_sample_ids():
    gene_names = [f"G{i}" for i in range(6)]
    expression_by_sample = {
        "S0": _synthetic_expression(gene_names, seed=0),
        "S1": _synthetic_expression(gene_names, seed=1),
    }
    basis, metadata = fit_conditional_wae_gene_coexpression_basis(
        expression_by_sample, ["S0", "S1"], gene_names, rank=3, seed=0,
    )
    assert basis.rank == 3
    assert basis.gene_names == tuple(gene_names)
    assert metadata["train_sample_ids"] == ["S0", "S1"]
    assert metadata["n_residual_rows"] == 40
    assert metadata["normalization"] == "normalize_log1p"
    assert metadata["fit_seed"] == 0
    assert metadata["kind"] == "conditional_wae_gene_coexpression_basis"
    assert len(metadata["residual_content_sha256"]) == 64


def test_fit_ignores_samples_not_listed_in_train_sample_ids():
    """Leakage guard: a validation/test sample's expression sitting in
    the same `expression_by_sample` mapping must never affect the fitted
    basis unless its id is explicitly in `train_sample_ids`."""
    gene_names = [f"G{i}" for i in range(6)]
    train_expression = {
        "S0": _synthetic_expression(gene_names, seed=0),
        "S1": _synthetic_expression(gene_names, seed=1),
    }
    basis_without_extra, _ = fit_conditional_wae_gene_coexpression_basis(
        dict(train_expression), ["S0", "S1"], gene_names, rank=3, seed=0,
    )
    with_extra = dict(train_expression)
    with_extra["VALIDATION_SAMPLE"] = _synthetic_expression(gene_names, seed=999)  # deliberately different
    basis_with_extra, metadata_with_extra = fit_conditional_wae_gene_coexpression_basis(
        with_extra, ["S0", "S1"], gene_names, rank=3, seed=0,
    )
    assert torch.equal(basis_without_extra.basis, basis_with_extra.basis)
    assert metadata_with_extra["train_sample_ids"] == ["S0", "S1"]
    assert "VALIDATION_SAMPLE" not in metadata_with_extra["train_sample_ids"]


def test_fit_rejects_empty_train_sample_ids():
    gene_names = ["G0", "G1"]
    with pytest.raises(ValueError, match="non-empty"):
        fit_conditional_wae_gene_coexpression_basis({}, [], gene_names, rank=1)


def test_fit_rejects_a_train_sample_id_missing_from_expression_by_sample():
    gene_names = ["G0", "G1"]
    with pytest.raises(KeyError):
        fit_conditional_wae_gene_coexpression_basis(
            {"S0": _synthetic_expression(gene_names)}, ["S0", "S1"], gene_names, rank=1,
        )


def test_fit_rejects_a_gene_dimension_mismatch():
    gene_names = ["G0", "G1", "G2"]
    wrong_shape = np.zeros((5, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="expected"):
        fit_conditional_wae_gene_coexpression_basis({"S0": wrong_shape}, ["S0"], gene_names, rank=1)


def test_fit_rejects_non_finite_expression():
    gene_names = ["G0", "G1"]
    bad = np.full((3, 2), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        fit_conditional_wae_gene_coexpression_basis({"S0": bad}, ["S0"], gene_names, rank=1)


def test_fit_accepts_real_scipy_sparse_expression_like_hest1k_adata_x():
    """Regression test: real HEST-1k adata.X is a scipy.sparse matrix, not
    a dense numpy array (see data/loaders.py's own densify-only-a-slice
    convention). np.asarray on a sparse matrix silently wraps it instead
    of densifying it, which used to fail opaquely deep inside
    np.concatenate/SVD the first time this ran against real data."""
    import scipy.sparse as sp

    gene_names = [f"G{i}" for i in range(6)]
    dense = _synthetic_expression(gene_names, n_spots=10, seed=0)
    sparse_expression = {"S0": sp.csr_matrix(dense)}
    basis, metadata = fit_conditional_wae_gene_coexpression_basis(
        sparse_expression, ["S0"], gene_names, rank=3, seed=0,
    )
    dense_basis, dense_metadata = fit_conditional_wae_gene_coexpression_basis(
        {"S0": dense}, ["S0"], gene_names, rank=3, seed=0,
    )
    assert torch.equal(basis.basis, dense_basis.basis)
    assert metadata["residual_content_sha256"] == dense_metadata["residual_content_sha256"]


# -- save/load -----------------------------------------------------------------

def _fit(gene_names=("G0", "G1", "G2", "G3"), rank=2):
    gene_names = list(gene_names)
    expression_by_sample = {"S0": _synthetic_expression(gene_names, n_spots=10, seed=0)}
    return fit_conditional_wae_gene_coexpression_basis(expression_by_sample, ["S0"], gene_names, rank=rank, seed=0)


def test_save_then_load_round_trips(tmp_path):
    basis, metadata = _fit()
    path = save_conditional_wae_gene_coexpression_basis(basis, metadata, tmp_path / "basis.pt")
    loaded_basis, loaded_metadata = load_conditional_wae_gene_coexpression_basis(str(path), list(basis.gene_names))
    assert torch.equal(loaded_basis.basis, basis.basis)
    assert loaded_basis.gene_names == basis.gene_names
    assert loaded_metadata == metadata


def test_save_rejects_metadata_missing_required_fields(tmp_path):
    basis, metadata = _fit()
    del metadata["rank"]
    with pytest.raises(ValueError, match="missing required field"):
        save_conditional_wae_gene_coexpression_basis(basis, metadata, tmp_path / "basis.pt")


def test_load_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="fit one with"):
        load_conditional_wae_gene_coexpression_basis(tmp_path / "nope.pt", ["G0", "G1"])


def test_load_rejects_a_gene_panel_mismatch(tmp_path):
    basis, metadata = _fit(gene_names=("G0", "G1", "G2", "G3"))
    path = save_conditional_wae_gene_coexpression_basis(basis, metadata, tmp_path / "basis.pt")
    with pytest.raises(ValueError, match="does not match the panel"):
        load_conditional_wae_gene_coexpression_basis(str(path), ["G0", "G1", "G2", "DIFFERENT"])


def test_load_rejects_a_reordered_gene_panel(tmp_path):
    basis, metadata = _fit(gene_names=("G0", "G1", "G2", "G3"))
    path = save_conditional_wae_gene_coexpression_basis(basis, metadata, tmp_path / "basis.pt")
    with pytest.raises(ValueError, match="does not match the panel"):
        load_conditional_wae_gene_coexpression_basis(str(path), ["G1", "G0", "G2", "G3"])


def test_load_rejects_a_corrupted_gene_names_hash(tmp_path):
    basis, metadata = _fit()
    path = save_conditional_wae_gene_coexpression_basis(basis, metadata, tmp_path / "basis.pt")
    payload = torch.load(path)
    payload["gene_names_hash"] = "corrupted"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="gene_names_hash"):
        load_conditional_wae_gene_coexpression_basis(str(path), list(basis.gene_names))


def test_load_rejects_a_metadata_rank_mismatch(tmp_path):
    basis, metadata = _fit()
    path = save_conditional_wae_gene_coexpression_basis(basis, metadata, tmp_path / "basis.pt")
    payload = torch.load(path)
    payload["metadata"]["rank"] = payload["metadata"]["rank"] + 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="metadata rank"):
        load_conditional_wae_gene_coexpression_basis(str(path), list(basis.gene_names))


# -- GeneCoexpressionRefinement ------------------------------------------------

def test_refinement_is_an_exact_identity_at_construction():
    basis = fit_gene_residual_basis(
        np.random.default_rng(0).normal(size=(10, 8)).astype(np.float32), [f"G{i}" for i in range(8)], rank=4,
    )
    refinement = GeneCoexpressionRefinement(basis)
    prediction = torch.randn(5, 8)
    assert torch.equal(refinement(prediction), prediction)


def test_refinement_rejects_a_gene_dimension_mismatch():
    basis = fit_gene_residual_basis(
        np.random.default_rng(0).normal(size=(10, 8)).astype(np.float32), [f"G{i}" for i in range(8)], rank=4,
    )
    refinement = GeneCoexpressionRefinement(basis)
    with pytest.raises(ValueError, match="n_genes"):
        refinement(torch.randn(5, 6))


def test_refinement_changes_prediction_after_weights_move_away_from_zero_init():
    basis = fit_gene_residual_basis(
        np.random.default_rng(0).normal(size=(10, 8)).astype(np.float32), [f"G{i}" for i in range(8)], rank=4,
    )
    refinement = GeneCoexpressionRefinement(basis)
    with torch.no_grad():
        for parameter in refinement.refine.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)
    prediction = torch.randn(5, 8)
    assert not torch.equal(refinement(prediction), prediction)


# -- ConditionalWAE wiring -----------------------------------------------------

def _tiny_conditioner(n_genes, image_feature_dim=8, hidden_dim=16):
    return Architecture1ImageConditioner(
        n_genes=n_genes, image_feature_dim=image_feature_dim, gex_feature_dim=4,
        hidden_dim=hidden_dim, n_heads=2, n_blocks=1, dense_threshold=64, sparse_k=4,
    )


def test_conditional_wae_decode_is_unaffected_at_init_with_a_coexpression_basis():
    """Zero-init discipline end to end: decode()'s reconstruction must be
    IDENTICAL with and without the coexpression basis at construction."""
    n_genes = 8
    gene_names = [f"G{i}" for i in range(n_genes)]
    basis = fit_gene_residual_basis(
        np.random.default_rng(0).normal(size=(10, n_genes)).astype(np.float32), gene_names, rank=3,
    )
    torch.manual_seed(0)
    plain = ConditionalWAE(n_genes, _tiny_conditioner(n_genes), regularizer="gan", latent_dim=4, autoencoder_hidden_dim=16)
    torch.manual_seed(0)
    with_coexpression = ConditionalWAE(
        n_genes, _tiny_conditioner(n_genes), regularizer="gan", latent_dim=4, autoencoder_hidden_dim=16,
        gene_coexpression_basis=basis,
    )
    assert with_coexpression.coexpression_refinement is not None
    context = torch.randn(3, 16)  # matches _tiny_conditioner's hidden_dim
    z = torch.randn(3, 4)
    plain_reconstruction, plain_mean = plain.decode(z, context)
    coexpression_reconstruction, coexpression_mean = with_coexpression.decode(z, context)
    assert torch.equal(plain_mean, coexpression_mean)
    # decoders were constructed with the same seed so their conditional_mean_head/
    # residual_decoder weights match exactly; the coexpression refinement adds
    # nothing at init, so both reconstructions should match too.
    assert torch.allclose(plain_reconstruction, coexpression_reconstruction)


def test_conditional_wae_rejects_a_coexpression_basis_with_the_wrong_n_genes():
    basis = fit_gene_residual_basis(
        np.random.default_rng(0).normal(size=(10, 6)).astype(np.float32), [f"G{i}" for i in range(6)], rank=2,
    )
    with pytest.raises(ValueError, match="same n_genes"):
        ConditionalWAE(
            8, _tiny_conditioner(8), regularizer="gan", latent_dim=4, autoencoder_hidden_dim=16,
            gene_coexpression_basis=basis,
        )
