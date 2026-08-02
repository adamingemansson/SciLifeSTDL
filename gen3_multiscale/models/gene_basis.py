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
import os
from dataclasses import dataclass
from pathlib import Path

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


def fit_gene_residual_basis(
    residuals: np.ndarray,
    gene_names: list[str],
    rank: int = 64,
    *,
    random_state: int = 0,
    svd_device: str = "cpu",
    n_iter: int = 4,
    n_oversamples: int = 16,
) -> GeneResidualBasis:
    """Fit a FIXED orthonormal low-rank basis via a DETERMINISTIC
    randomized truncated SVD on a TRAINING-only residuals matrix
    [n_samples, n_genes]. Not centered -- this approximates the raw
    residual space directly (no separate mean to track/store alongside
    the basis for reconstruction), a truncated-SVD low-rank approximation
    of "target - deterministic_mean" as computed on training data.
    gene_names' exact order is hashed and stored alongside the basis so a
    caller can verify (mirroring checkpoint.verify_gene_names' fail-closed
    discipline elsewhere in this project) that a later use of this basis
    has the same gene ordering it was fit against.

    Adam's Step 6 audit #6 of commit a32051b: "Replace full residual SVD
    with deterministic randomized/incremental truncated SVD." A real
    ~17,000-gene panel with many pooled training rows made
    `np.linalg.svd(residuals, full_matrices=False)` -- a FULL dense SVD
    over the whole matrix -- large, memory- and compute-heavy for a
    result that only ever keeps `rank` (typically 32-64) singular
    vectors. `sklearn.utils.extmath.randomized_svd` computes only the
    requested number of components directly, with a FIXED `random_state`
    (never left to the global numpy RNG) so the fitted basis stays
    exactly reproducible given the same residuals/rank/random_state --
    "fixed thereafter" per this module's own docstring must also mean
    the FIT ITSELF is deterministic, not just its later use."""
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
    if int(n_iter) < 0:
        raise ValueError("n_iter must be non-negative")
    if int(n_oversamples) < 0:
        raise ValueError("n_oversamples must be non-negative")

    device = torch.device(svd_device)
    if device.type == "cpu":
        from sklearn.utils.extmath import randomized_svd

        # Explicit QR normalization avoids sklearn's AUTO -> LU path and its
        # LAPACK SLASWP calls, which have failed on the real 90k x 17k memmap.
        _u, _s, vt = randomized_svd(
            residuals,
            n_components=effective_rank,
            n_iter=int(n_iter),
            n_oversamples=int(n_oversamples),
            power_iteration_normalizer="QR",
            random_state=random_state,
        )
        basis = vt[:effective_rank]
    elif device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"svd_device={svd_device!r} requested but CUDA is unavailable")
        # The real residual matrix is about 6.2 GB and fits comfortably on
        # the project's 80-GB A100s. Keep the same randomized rank,
        # oversampling, and iteration controls while moving the matrix
        # multiplications off the prohibitively slow CPU path.
        q = min(max_rank, effective_rank + int(n_oversamples))
        torch.manual_seed(int(random_state))
        torch.cuda.manual_seed_all(int(random_state))
        matrix = torch.as_tensor(residuals, dtype=torch.float32, device=device)
        try:
            _u, _s, v = torch.svd_lowrank(matrix, q=q, niter=int(n_iter), M=None)
            basis_tensor = v[:, :effective_rank].T.contiguous()
            # SVD vector signs are arbitrary. Canonicalize them so a fixed
            # numerical decomposition has a stable saved orientation.
            pivot = basis_tensor.abs().argmax(dim=1)
            row = torch.arange(effective_rank, device=device)
            signs = torch.sign(basis_tensor[row, pivot])
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            basis = (basis_tensor * signs[:, None]).cpu().numpy()
        finally:
            del matrix
    else:
        raise ValueError(f"svd_device must be 'cpu' or a CUDA device, got {svd_device!r}")

    gene_names_hash = hashlib.sha256("\0".join(gene_names).encode()).hexdigest()
    return GeneResidualBasis(
        basis=torch.from_numpy(np.ascontiguousarray(basis)).float(),
        gene_names=tuple(gene_names),
        gene_names_hash=gene_names_hash,
    )


def save_gene_residual_basis(basis: GeneResidualBasis, path: str | Path) -> Path:
    """Persist a fitted `GeneResidualBasis` -- Step 6's real trainer needs
    a real on-disk artifact for Architecture 4's `required_fingerprints.
    gene_basis` config field to point at ("gene_basis must be a
    GeneResidualBasis already fit on TRAINING-split residuals... fit
    offline, outside this class" -- architecture4.yaml's own docs; no
    persistence for that "offline" step existed anywhere in this
    codebase before this function). Atomic write, mirroring every other
    artifact in this package."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(
        {
            "basis": basis.basis, "gene_names": list(basis.gene_names),
            "gene_names_hash": basis.gene_names_hash,
        },
        tmp,
    )
    os.replace(tmp, path)
    return path


def load_gene_residual_basis(path: str | Path) -> GeneResidualBasis:
    """Load-and-verify a `save_gene_residual_basis` artifact: the saved
    `gene_names_hash` must match a FRESH hash of the saved `gene_names`
    (fail closed on a hand-edited or corrupted file, same discipline as
    `checkpoint.verify_gene_names`), and the basis's own numeric shape
    must agree with `gene_names`'s length."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"gene residual basis not found at {path} -- fit and save one first")
    payload = torch.load(path, map_location="cpu")
    gene_names = list(payload["gene_names"])
    basis_tensor = payload["basis"]
    expected_hash = hashlib.sha256("\0".join(gene_names).encode()).hexdigest()
    if payload.get("gene_names_hash") != expected_hash:
        raise ValueError(
            f"gene residual basis at {path} has a gene_names_hash that does not match its own "
            "saved gene_names -- corrupted or hand-edited file"
        )
    if basis_tensor.shape[1] != len(gene_names):
        raise ValueError(
            f"gene residual basis at {path} has {basis_tensor.shape[1]} basis columns but "
            f"{len(gene_names)} gene_names -- corrupted file"
        )
    return GeneResidualBasis(basis=basis_tensor, gene_names=tuple(gene_names), gene_names_hash=expected_hash)


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
