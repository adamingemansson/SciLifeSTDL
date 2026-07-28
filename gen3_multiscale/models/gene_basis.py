"""Fixed low-rank gene residual basis -- Architecture 4's mechanism for
mapping between full-gene residual vectors and a compact coefficient
field, per the handoff: "Generate a low-rank residual coefficient field
around the deterministic transport mean... represented through a learned
low-rank gene basis... Use a fixed orthonormal gene basis fitted on
training expression only... Record its gene ordering and hash. It is not
one independent flow per spot and it does not run a dense flow directly
over approximately 17,000 genes."

Deliberately pure numpy for the FIT step -- like harmonic.py, never
touches torch until the final result is wrapped for use by the velocity
network, keeping "fitted on training expression only, fixed thereafter"
structurally simple to audit: fit_gene_residual_basis takes a plain numpy
residuals matrix (the caller's responsibility to ensure it's TRAINING-
split residuals only -- this module has no way to enforce that from
inside, same limitation harmonic.py and target_gene_scale already have).
The returned GeneResidualBasis holds its basis as a torch buffer-ready
tensor but the basis itself is never an nn.Parameter anywhere in this
codebase -- callers must not wrap it in one.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class GeneResidualBasis:
    basis: torch.Tensor  # [rank, n_genes], orthonormal rows (basis @ basis.T ~= I)
    gene_names: tuple[str, ...]
    gene_names_hash: str

    @property
    def rank(self) -> int:
        return self.basis.shape[0]

    @property
    def n_genes(self) -> int:
        return self.basis.shape[1]

    def to_coefficients(self, residual: torch.Tensor) -> torch.Tensor:
        """[..., n_genes] -> [..., rank]."""
        if residual.shape[-1] != self.n_genes:
            raise ValueError(f"residual's last dim ({residual.shape[-1]}) must equal n_genes ({self.n_genes})")
        return residual @ self.basis.T

    def from_coefficients(self, coefficients: torch.Tensor) -> torch.Tensor:
        """[..., rank] -> [..., n_genes]."""
        if coefficients.shape[-1] != self.rank:
            raise ValueError(f"coefficients' last dim ({coefficients.shape[-1]}) must equal rank ({self.rank})")
        return coefficients @ self.basis


def fit_gene_residual_basis(residuals: np.ndarray, gene_names: list[str], rank: int = 64) -> GeneResidualBasis:
    """Fit a FIXED orthonormal low-rank basis via truncated SVD on a
    TRAINING-only residuals matrix [n_samples, n_genes]. Not centered --
    this approximates the raw residual space directly (no separate mean
    to track/store alongside the basis for reconstruction), a truncated-
    SVD low-rank approximation of "target - deterministic_mean" as
    computed on training data. gene_names' exact order is hashed and
    stored alongside the basis so a caller can verify (mirroring
    checkpoint.verify_gene_names' fail-closed discipline elsewhere in
    this project) that a later use of this basis has the same gene
    ordering it was fit against.
    """
    if residuals.ndim != 2:
        raise ValueError(f"residuals must be 2-D [n_samples, n_genes], got shape {residuals.shape}")
    if residuals.shape[1] != len(gene_names):
        raise ValueError(
            f"residuals has {residuals.shape[1]} gene columns but {len(gene_names)} gene_names were given"
        )
    if residuals.shape[0] < 2:
        raise ValueError("need at least 2 residual samples to fit a basis")
    if not np.all(np.isfinite(residuals)):
        raise ValueError("residuals contains non-finite values")
    max_rank = min(residuals.shape)
    effective_rank = min(int(rank), max_rank)
    if effective_rank < 1:
        raise ValueError("rank must be positive")

    _u, _s, vt = np.linalg.svd(residuals, full_matrices=False)
    basis = vt[:effective_rank]  # [effective_rank, n_genes], orthonormal rows by construction

    gene_names_hash = hashlib.sha256("\0".join(gene_names).encode()).hexdigest()
    return GeneResidualBasis(
        basis=torch.from_numpy(np.ascontiguousarray(basis)).float(),
        gene_names=tuple(gene_names),
        gene_names_hash=gene_names_hash,
    )


def verify_gene_residual_basis(basis: GeneResidualBasis, gene_names: list[str]) -> None:
    """Fail-closed check mirroring checkpoint.verify_gene_names: raises if
    a caller's current ordered gene panel doesn't match the exact panel
    this basis was fit against."""
    current_hash = hashlib.sha256("\0".join(gene_names).encode()).hexdigest()
    if current_hash != basis.gene_names_hash:
        raise ValueError(
            "gene panel does not match the panel this GeneResidualBasis was fit against "
            f"(expected hash {basis.gene_names_hash}, got {current_hash}) -- refusing to use a "
            "residual basis fit on a different, possibly reordered, gene panel"
        )
