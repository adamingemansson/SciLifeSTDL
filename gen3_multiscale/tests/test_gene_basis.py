"""Architecture 4's fixed low-rank gene residual basis. Tests real
mathematical properties (orthonormality, reconstruction quality scaling
with rank, exact reconstruction at full rank) plus the gene-identity
fail-closed verification mirroring checkpoint.verify_gene_names."""
import numpy as np
import pytest
import torch

from gen3_multiscale.models.gene_basis import fit_gene_residual_basis, verify_gene_residual_basis


def _residuals(n_samples=50, n_genes=10, seed=0):
    rng = np.random.default_rng(seed)
    # low effective rank by construction: mostly driven by a handful of
    # latent factors, so a modest rank should reconstruct it well
    latent = rng.normal(size=(n_samples, 3))
    loadings = rng.normal(size=(3, n_genes))
    signal = latent @ loadings
    noise = rng.normal(scale=0.01, size=(n_samples, n_genes))
    return (signal + noise).astype(np.float64)


def test_basis_rows_are_orthonormal():
    gene_names = [f"g{i}" for i in range(10)]
    basis = fit_gene_residual_basis(_residuals(), gene_names, rank=5)
    gram = basis.basis @ basis.basis.T
    assert torch.allclose(gram, torch.eye(5), atol=1e-4)


def test_higher_rank_reconstructs_better():
    gene_names = [f"g{i}" for i in range(10)]
    residuals = _residuals()
    residuals_t = torch.from_numpy(residuals).float()

    basis_low = fit_gene_residual_basis(residuals, gene_names, rank=1)
    basis_high = fit_gene_residual_basis(residuals, gene_names, rank=5)

    err_low = (basis_low.from_coefficients(basis_low.to_coefficients(residuals_t)) - residuals_t).square().mean()
    err_high = (basis_high.from_coefficients(basis_high.to_coefficients(residuals_t)) - residuals_t).square().mean()
    assert err_high < err_low


def test_full_rank_reconstructs_almost_exactly():
    gene_names = [f"g{i}" for i in range(10)]
    residuals = _residuals(n_samples=50, n_genes=10)
    residuals_t = torch.from_numpy(residuals).float()
    basis = fit_gene_residual_basis(residuals, gene_names, rank=10)  # full rank (n_genes=10)
    reconstructed = basis.from_coefficients(basis.to_coefficients(residuals_t))
    assert torch.allclose(reconstructed, residuals_t, atol=1e-2)


def test_rank_is_capped_at_the_smaller_matrix_dimension():
    gene_names = [f"g{i}" for i in range(10)]
    residuals = _residuals(n_samples=5, n_genes=10)  # fewer samples than requested rank
    basis = fit_gene_residual_basis(residuals, gene_names, rank=64)
    assert basis.rank == 5


def test_coefficients_shape():
    gene_names = [f"g{i}" for i in range(10)]
    basis = fit_gene_residual_basis(_residuals(), gene_names, rank=4)
    residual = torch.randn(7, 10)
    coeff = basis.to_coefficients(residual)
    assert coeff.shape == (7, 4)
    back = basis.from_coefficients(coeff)
    assert back.shape == (7, 10)


def test_verify_gene_residual_basis_passes_for_the_same_panel():
    gene_names = [f"g{i}" for i in range(10)]
    basis = fit_gene_residual_basis(_residuals(), gene_names, rank=4)
    verify_gene_residual_basis(basis, gene_names)  # must not raise


def test_verify_gene_residual_basis_raises_on_reordered_panel():
    gene_names = [f"g{i}" for i in range(10)]
    basis = fit_gene_residual_basis(_residuals(), gene_names, rank=4)
    reordered = gene_names[::-1]
    with pytest.raises(ValueError, match="gene panel"):
        verify_gene_residual_basis(basis, reordered)


def test_rejects_mismatched_gene_names_length():
    with pytest.raises(ValueError, match="gene_names"):
        fit_gene_residual_basis(_residuals(n_genes=10), [f"g{i}" for i in range(5)], rank=4)


def test_rejects_non_finite_residuals():
    residuals = _residuals()
    residuals[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fit_gene_residual_basis(residuals, [f"g{i}" for i in range(10)], rank=4)


def test_randomized_svd_uses_explicit_qr_normalization_and_bounded_iterations(monkeypatch):
    import sklearn.utils.extmath

    captured = {}
    real_randomized_svd = sklearn.utils.extmath.randomized_svd

    def recording_randomized_svd(*args, **kwargs):
        captured.update(kwargs)
        return real_randomized_svd(*args, **kwargs)

    monkeypatch.setattr(sklearn.utils.extmath, "randomized_svd", recording_randomized_svd)
    fit_gene_residual_basis(_residuals(), [f"g{i}" for i in range(10)], rank=4)

    assert captured["power_iteration_normalizer"] == "QR"
    assert captured["n_iter"] == 4
    assert captured["random_state"] == 0
