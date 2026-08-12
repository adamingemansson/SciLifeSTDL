"""Training-only gene structure for deterministic MK field predictors.

The basis is deliberately fitted after subtracting each training slide's
gene-wise mean.  Rows are weighted so every organ contributes equally and
every slide within an organ contributes equally.  This prevents a basis that
mostly encodes organ/slide offsets or the largest slide.  Validation and test
expression are structurally unavailable to the fitter.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from gen3_multiscale.models.gene_basis import (
    GeneResidualBasis,
    fit_gene_residual_basis,
    verify_gene_residual_basis,
)


@dataclass(frozen=True)
class CenteredGeneStructureArtifact:
    basis: GeneResidualBasis
    global_gene_mean: torch.Tensor
    per_gene_scale: torch.Tensor
    metadata: dict

    def __post_init__(self) -> None:
        mean = torch.as_tensor(self.global_gene_mean, dtype=torch.float32)
        scale = torch.as_tensor(self.per_gene_scale, dtype=torch.float32)
        if mean.shape != (self.basis.n_genes,) or not torch.isfinite(mean).all():
            raise ValueError("global_gene_mean must be finite with one value per gene")
        if scale.shape != (self.basis.n_genes,):
            raise ValueError("per_gene_scale must have one value per gene")
        if not torch.isfinite(scale).all() or not torch.all(scale > 0):
            raise ValueError("per_gene_scale must be finite and strictly positive")
        object.__setattr__(self, "global_gene_mean", mean)
        object.__setattr__(self, "per_gene_scale", scale)


def _dense_expression(value, sample_id: str, n_genes: int) -> np.ndarray:
    value = value.toarray() if hasattr(value, "toarray") else value
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != n_genes:
        raise ValueError(
            f"{sample_id}: expression has shape {matrix.shape}, expected [*, {n_genes}]"
        )
    if matrix.shape[0] < 2:
        raise ValueError(f"{sample_id}: at least two spots are required")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{sample_id}: expression contains non-finite values")
    return matrix


def fit_centered_organ_balanced_gene_structure(
    expression_by_sample: dict,
    train_sample_ids: list[str],
    organ_by_sample: dict[str, str],
    gene_names: list[str],
    *,
    rank: int = 64,
    seed: int = 0,
    svd_device: str = "cpu",
) -> CenteredGeneStructureArtifact:
    """Fit centered gene programs and gradient scales from training only.

    Each slide is centered independently.  Its centered rows are scaled by
    ``1/sqrt(n_spots * n_slides_in_organ)``; consequently each organ has the
    same total squared weight and each of its slides contributes equally.
    """
    sample_ids = sorted(str(value) for value in train_sample_ids)
    if not sample_ids or not gene_names:
        raise ValueError("train_sample_ids and gene_names must be non-empty")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("train_sample_ids contains duplicates")
    missing_organs = [sid for sid in sample_ids if sid not in organ_by_sample]
    if missing_organs:
        raise ValueError(f"training samples missing organ metadata: {missing_organs}")

    organs = {sid: str(organ_by_sample[sid]) for sid in sample_ids}
    if any(not value for value in organs.values()):
        raise ValueError("organ labels must be non-empty")
    slides_per_organ: dict[str, int] = {}
    for organ in organs.values():
        slides_per_organ[organ] = slides_per_organ.get(organ, 0) + 1

    centered_by_sample: dict[str, np.ndarray] = {}
    means_by_organ: dict[str, list[np.ndarray]] = {}
    within_second_moments: dict[str, list[np.ndarray]] = {}
    row_counts = {}
    for sample_id in sample_ids:
        if sample_id not in expression_by_sample:
            raise KeyError(f"{sample_id}: missing from expression_by_sample")
        matrix = _dense_expression(expression_by_sample[sample_id], sample_id, len(gene_names))
        slide_mean = matrix.mean(axis=0)
        centered = matrix - slide_mean[None, :]
        organ = organs[sample_id]
        centered_by_sample[sample_id] = centered
        means_by_organ.setdefault(organ, []).append(slide_mean)
        within_second_moments.setdefault(organ, []).append(
            np.mean(np.square(centered, dtype=np.float64), axis=0)
        )
        row_counts[sample_id] = int(matrix.shape[0])

    # Equal mean over slides within each organ, then equal mean over organs.
    organ_moments = [
        np.mean(np.stack(within_second_moments[organ], axis=0), axis=0)
        for organ in sorted(within_second_moments)
    ]
    scale = np.sqrt(np.mean(np.stack(organ_moments, axis=0), axis=0))
    scale = np.maximum(scale, 1e-6).astype(np.float32)

    organ_means = [
        np.mean(np.stack(means_by_organ[organ], axis=0), axis=0)
        for organ in sorted(means_by_organ)
    ]
    global_mean = np.mean(np.stack(organ_means, axis=0), axis=0).astype(np.float32)

    weighted_rows = []
    for sample_id in sample_ids:
        organ = organs[sample_id]
        centered = centered_by_sample[sample_id] / scale[None, :]
        row_scale = 1.0 / np.sqrt(centered.shape[0] * slides_per_organ[organ])
        weighted_rows.append(centered * np.float32(row_scale))

    weighted = np.concatenate(weighted_rows, axis=0).astype(np.float32, copy=False)
    basis = fit_gene_residual_basis(
        weighted, gene_names, rank=rank, random_state=seed, svd_device=svd_device,
    )
    metadata = {
        "kind": "mk_centered_organ_balanced_gene_structure",
        "version": 1,
        "train_sample_ids": sample_ids,
        "organ_by_sample": organs,
        "slides_per_organ": dict(sorted(slides_per_organ.items())),
        "row_counts": row_counts,
        "normalization": "normalize_log1p_then_training_within_slide_standardization",
        "centering": "per_slide_gene_mean",
        "weighting": "equal_organ_equal_slide",
        "rank": int(basis.rank),
        "n_rows": int(weighted.shape[0]),
        "fit_seed": int(seed),
        "basis_sha256": hashlib.sha256(
            np.ascontiguousarray(basis.basis.cpu().numpy()).tobytes()
        ).hexdigest(),
        "centered_weighted_content_sha256": hashlib.sha256(
            np.ascontiguousarray(weighted).tobytes()
        ).hexdigest(),
        "per_gene_scale_sha256": hashlib.sha256(
            np.ascontiguousarray(scale).tobytes()
        ).hexdigest(),
        "global_gene_mean_sha256": hashlib.sha256(
            np.ascontiguousarray(global_mean).tobytes()
        ).hexdigest(),
    }
    return CenteredGeneStructureArtifact(
        basis=basis,
        global_gene_mean=torch.from_numpy(global_mean),
        per_gene_scale=torch.from_numpy(scale),
        metadata=metadata,
    )


def save_centered_gene_structure_artifact(
    artifact: CenteredGeneStructureArtifact, path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save({
        "basis": artifact.basis.basis,
        "gene_names": list(artifact.basis.gene_names),
        "gene_names_hash": artifact.basis.gene_names_hash,
        "global_gene_mean": artifact.global_gene_mean,
        "per_gene_scale": artifact.per_gene_scale,
        "metadata": dict(artifact.metadata),
    }, temporary)
    os.replace(temporary, path)
    return path


def load_centered_gene_structure_artifact(
    path: str | Path, gene_names: list[str],
) -> CenteredGeneStructureArtifact:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"centered gene-structure artifact not found: {path}")
    payload = torch.load(path, map_location="cpu")
    required = {
        "basis", "gene_names", "gene_names_hash", "global_gene_mean",
        "per_gene_scale", "metadata",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"centered gene-structure artifact is missing {sorted(missing)}")
    saved_names = tuple(str(value) for value in payload["gene_names"])
    expected_hash = hashlib.sha256("\0".join(saved_names).encode()).hexdigest()
    if payload["gene_names_hash"] != expected_hash:
        raise ValueError("centered gene-structure artifact has a corrupt gene-name hash")
    basis = GeneResidualBasis(
        basis=torch.as_tensor(payload["basis"], dtype=torch.float32),
        gene_names=saved_names,
        gene_names_hash=expected_hash,
    )
    verify_gene_residual_basis(basis, gene_names)
    metadata = dict(payload["metadata"])
    if metadata.get("kind") != "mk_centered_organ_balanced_gene_structure":
        raise ValueError("artifact kind is not centered organ-balanced MK gene structure")
    if metadata.get("centering") != "per_slide_gene_mean":
        raise ValueError("artifact was not fit with per-slide centering")
    if metadata.get("weighting") != "equal_organ_equal_slide":
        raise ValueError("artifact was not fit with equal-organ/equal-slide weighting")
    if int(metadata.get("rank", -1)) != basis.rank:
        raise ValueError("artifact metadata rank does not match its basis")
    basis_hash = hashlib.sha256(
        np.ascontiguousarray(basis.basis.cpu().numpy()).tobytes()
    ).hexdigest()
    if basis_hash != metadata.get("basis_sha256"):
        raise ValueError("artifact basis hash does not match its contents")
    scale = torch.as_tensor(payload["per_gene_scale"], dtype=torch.float32)
    mean = torch.as_tensor(payload["global_gene_mean"], dtype=torch.float32)
    scale_hash = hashlib.sha256(np.ascontiguousarray(scale.numpy()).tobytes()).hexdigest()
    if scale_hash != metadata.get("per_gene_scale_sha256"):
        raise ValueError("artifact per-gene scale hash does not match its contents")
    mean_hash = hashlib.sha256(np.ascontiguousarray(mean.numpy()).tobytes()).hexdigest()
    if mean_hash != metadata.get("global_gene_mean_sha256"):
        raise ValueError("artifact global gene-mean hash does not match its contents")
    return CenteredGeneStructureArtifact(
        basis=basis, global_gene_mean=mean, per_gene_scale=scale, metadata=metadata,
    )


class CenteredGeneStructureRefinement(nn.Module):
    """Residual decoder in a fixed, standardized within-slide gene basis."""

    def __init__(self, artifact: CenteredGeneStructureArtifact, hidden_dim: int = 64):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        self.register_buffer("basis_matrix", artifact.basis.basis.clone(), persistent=False)
        self.register_buffer("global_gene_mean", artifact.global_gene_mean.clone(), persistent=False)
        self.register_buffer("per_gene_scale", artifact.per_gene_scale.clone(), persistent=False)
        rank = artifact.basis.rank
        self.refine = nn.Sequential(
            nn.LayerNorm(rank), nn.Linear(rank, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, rank),
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, prediction: torch.Tensor) -> torch.Tensor:
        if prediction.ndim != 2 or prediction.shape[1] != self.basis_matrix.shape[1]:
            raise ValueError("prediction must be [N, n_genes] in the artifact's gene order")
        standardized = (prediction - self.global_gene_mean) / self.per_gene_scale
        coefficients = standardized @ self.basis_matrix.T
        delta_standardized = self.refine(coefficients) @ self.basis_matrix
        return prediction + delta_standardized * self.per_gene_scale
