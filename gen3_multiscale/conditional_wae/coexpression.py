"""Gene-coexpression ablation for the MK conditional WAE.

Reuses `models/gene_basis.py::GeneResidualBasis` exactly as Architecture 4
does (fixed orthonormal low-rank basis, fit once via deterministic
randomized SVD, gene ordering hashed and verified on every later use) --
but fit here on TRAINING-split real, normalized full-gene expression
directly (never a frozen model's residuals, since no pretrained
deterministic checkpoint is a prerequisite for this arm), capturing which
genes co-vary together across real training spots. Never touches
validation/test GEX: `fit_conditional_wae_gene_coexpression_basis` only
ever reads the exact `train_sample_ids` its caller passes.

`GeneCoexpressionRefinement` is the model-side piece: a small, zero-init
residual correction applied to the ordinary full-gene prediction
(`ConditionalWAE.decode`'s `reconstruction`, never `conditional_mean`) in
the basis's low-rank coefficient space. Zero-initialized final layer
means it is an EXACT identity at construction -- the same "strict
superset" discipline `_FiLMGenerator` already uses -- so the coexpression
component can only ever nudge the base decoder's prediction, never
replace it.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from gen3_multiscale.models.gene_basis import (
    GeneResidualBasis, fit_gene_residual_basis, verify_gene_residual_basis,
)

_REQUIRED_METADATA_FIELDS = {
    "kind", "train_sample_ids", "normalization", "rank", "n_residual_rows",
    "fit_seed", "residual_content_sha256",
}


def fit_conditional_wae_gene_coexpression_basis(
    expression_by_sample: dict,
    train_sample_ids: list[str],
    gene_names: list[str],
    *,
    rank: int = 64,
    seed: int = 0,
) -> tuple[GeneResidualBasis, dict]:
    """Fit a low-rank gene-coexpression basis from real, normalized
    full-gene expression pooled ONLY across `train_sample_ids` -- the
    caller's dataset-manifest-declared training split, never validation
    or test (this function has no way to enforce that from inside; the
    same limitation `fit_gene_residual_basis`'s own docstring already
    documents for its caller-supplied residuals matrix).

    `expression_by_sample` must map every id in `train_sample_ids` to a
    real `[n_spots, n_genes]` array in `gene_names` order -- e.g.
    `example_builder.load_sample_for_examples(manifest, sample_id)[0].X`
    (QC'd/gene-panel-aligned, `normalize_log1p`, this codebase's
    pipeline-standard transform, `data/loaders.py::basic_qc_and_
    normalize`). Deliberately a plain array mapping, not `Gen3SampleData`
    -- fitting this basis is pure gene-expression structure and has no
    dependency on any image tile-encoder cache existing."""
    if not train_sample_ids:
        raise ValueError("train_sample_ids must be non-empty")
    sorted_ids = sorted(str(sample_id) for sample_id in train_sample_ids)
    rows = []
    for sample_id in sorted_ids:
        if sample_id not in expression_by_sample:
            raise KeyError(f"{sample_id}: not present in the given expression_by_sample mapping")
        raw = expression_by_sample[sample_id]
        # Real HEST-1k adata.X is a scipy.sparse matrix (see data/loaders.py's
        # own "densify only the slice you need" convention) -- np.asarray on
        # a sparse matrix does NOT densify it, it wraps it in a 0-d object
        # array, which later fails opaquely inside np.concatenate/SVD.
        dense = raw.toarray() if hasattr(raw, "toarray") else raw
        matrix = np.asarray(dense, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != len(gene_names):
            raise ValueError(
                f"{sample_id}: expression has shape {matrix.shape}, expected [*, {len(gene_names)}]"
            )
        rows.append(matrix)
    pooled = np.concatenate(rows, axis=0)
    if not np.all(np.isfinite(pooled)):
        raise ValueError("pooled training expression contains non-finite values")

    basis = fit_gene_residual_basis(pooled, gene_names, rank=rank, random_state=seed)
    content_hash = hashlib.sha256(np.ascontiguousarray(pooled).tobytes()).hexdigest()
    metadata = {
        "kind": "conditional_wae_gene_coexpression_basis",
        "train_sample_ids": sorted_ids,
        "normalization": "normalize_log1p",
        "rank": int(basis.rank),
        "n_residual_rows": int(pooled.shape[0]),
        "fit_seed": int(seed),
        "residual_content_sha256": content_hash,
    }
    return basis, metadata


def save_conditional_wae_gene_coexpression_basis(basis: GeneResidualBasis, metadata: dict, path: str | Path) -> Path:
    """Atomic write, mirroring `gene_basis.save_gene_residual_basis` --
    the basis tensor, its own gene-order hash, AND the fitting metadata
    (training sample ids, normalization, rank, artifact hashes) all in
    one file, so a loader can never see one without the other."""
    missing = _REQUIRED_METADATA_FIELDS - set(metadata)
    if missing:
        raise ValueError(f"metadata is missing required field(s): {sorted(missing)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(
        {
            "basis": basis.basis, "gene_names": list(basis.gene_names),
            "gene_names_hash": basis.gene_names_hash, "metadata": dict(metadata),
        },
        tmp,
    )
    os.replace(tmp, path)
    return path


def load_conditional_wae_gene_coexpression_basis(
    path: str | Path, gene_names: list[str],
) -> tuple[GeneResidualBasis, dict]:
    """Load-and-verify: internal gene_names_hash consistency (mirrors
    `gene_basis.load_gene_residual_basis`), required metadata fields
    present and self-consistent (rank matches the basis's own rank), AND
    that the CURRENT gene panel exactly matches the panel this basis was
    fit against (`verify_gene_residual_basis` -- fail closed on any
    reordering or panel drift since fitting)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"gene coexpression basis not found at {path} -- fit one with "
            "scripts/fit_conditional_wae_gene_coexpression_basis.py before training"
        )
    payload = torch.load(path, map_location="cpu")
    for field in ("basis", "gene_names", "gene_names_hash", "metadata"):
        if field not in payload:
            raise ValueError(f"gene coexpression basis {path} is missing field {field!r} -- corrupted file")
    saved_gene_names = list(payload["gene_names"])
    expected_hash = hashlib.sha256("\0".join(saved_gene_names).encode()).hexdigest()
    if payload["gene_names_hash"] != expected_hash:
        raise ValueError(
            f"gene coexpression basis {path} has a gene_names_hash that does not match its own "
            "saved gene_names -- corrupted or hand-edited file"
        )
    basis = GeneResidualBasis(
        basis=payload["basis"], gene_names=tuple(saved_gene_names), gene_names_hash=expected_hash,
    )
    if basis.basis.shape[1] != len(saved_gene_names):
        raise ValueError(f"gene coexpression basis {path} basis shape does not match its saved gene_names")

    metadata = dict(payload["metadata"])
    missing_meta = _REQUIRED_METADATA_FIELDS - set(metadata)
    if missing_meta:
        raise ValueError(f"gene coexpression basis {path} metadata is missing field(s): {sorted(missing_meta)}")
    if int(metadata["rank"]) != basis.rank:
        raise ValueError(
            f"gene coexpression basis {path} metadata rank ({metadata['rank']}) does not match the "
            f"basis's own rank ({basis.rank}) -- corrupted file"
        )

    # Fail closed: the CURRENT caller's gene panel must exactly match the
    # panel this basis was fit against, not merely what was saved inside
    # the file (a stale/mismatched caller-supplied panel is exactly the
    # scenario this check exists to catch).
    verify_gene_residual_basis(basis, gene_names)
    return basis, metadata


class GeneCoexpressionRefinement(nn.Module):
    """Small residual refinement over a fixed, training-only-fit
    low-rank gene coexpression basis. Operates ONLY on the ordinary
    full-gene prediction as an additive correction -- projects into the
    basis's rank-R coefficient space, applies a small MLP, projects back
    to gene space. The final linear layer is zero-initialized so this
    module computes an EXACT identity at construction (matches
    `_FiLMGenerator`'s zero-init discipline in model.py): the
    coexpression component can only ever nudge the base decoder's
    prediction, never replace it."""

    def __init__(self, basis: GeneResidualBasis, hidden_dim: int = 64):
        super().__init__()
        self.register_buffer("basis_matrix", basis.basis.clone(), persistent=False)
        self.gene_names_hash = basis.gene_names_hash
        rank = basis.rank
        self.refine = nn.Sequential(
            nn.LayerNorm(rank), nn.Linear(rank, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, rank),
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    @property
    def n_genes(self) -> int:
        return self.basis_matrix.shape[1]

    def forward(self, prediction: torch.Tensor) -> torch.Tensor:
        if prediction.shape[-1] != self.n_genes:
            raise ValueError(
                f"prediction's last dim ({prediction.shape[-1]}) must equal the coexpression basis's "
                f"n_genes ({self.n_genes})"
            )
        coefficients = prediction @ self.basis_matrix.T
        delta_coefficients = self.refine(coefficients)
        delta = delta_coefficients @ self.basis_matrix
        return prediction + delta
