"""Gen4 residual-basis fitting primitive -- GEN4_CONTRACT.md section 10.

Reuses `models/gene_basis.py::fit_gene_residual_basis`/`save_gene_residual_basis`
unmodified -- this module only supplies the Gen4-specific residual
collection loop (running a frozen `Gen4Conditioner` over training-only
examples), mirroring `scripts/fit_architecture4_residual_basis.py::
compute_training_residuals`'s own memmap-backed, bounded-memory approach.

Scope note (see GEN4_CONTRACT.md section 13): this module takes an
already-built iterable of training `(inputs, targets)` pairs, not a
manifest path + mask-schedule spec -- wiring a real manifest-driven
training-mask iterator for Gen4 (mirroring
`training.gen3_dataset.Gen3SpatialFieldDataset`/`build_gen3_mask_schedule`,
extended to attach each arm's `context_gex_embedding`) is real-data/real-
weight-dependent integration work, listed explicitly as a follow-up in
RUNBOOK.md rather than half-built here.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from gen3_multiscale.models.gene_basis import GeneResidualBasis, fit_gene_residual_basis, save_gene_residual_basis


def compute_gen4_training_residuals(
    conditioner, train_examples: list, device: torch.device, *, memmap_path: str | Path,
) -> np.memmap:
    """`conditioner` is a frozen (eval-mode) `Gen4Conditioner` for one arm.
    `train_examples` is a real, materialized list of `(inputs, targets)`
    pairs drawn ONLY from training-split samples/masks -- the caller's
    responsibility (the same limit `fit_architecture4_residual_basis.py`'s
    own `compute_training_residuals` already documents for Gen3); this
    function has no way to independently verify a caller passed
    training-only data, matching the rest of this codebase's "fit offline,
    outside this class" discipline for anything that touches the
    train/validation/test boundary."""
    if not train_examples:
        raise ValueError("compute_gen4_training_residuals: train_examples is empty")
    row_counts = [np.asarray(targets.query_expression).shape[0] for _inputs, targets in train_examples]
    total_rows = int(sum(row_counts))
    if total_rows == 0:
        raise ValueError("compute_gen4_training_residuals: zero residual rows")
    n_genes = np.asarray(train_examples[0][1].query_expression).shape[1]

    memmap_path = Path(memmap_path)
    memmap_path.parent.mkdir(parents=True, exist_ok=True)
    residuals = np.lib.format.open_memmap(memmap_path, mode="w+", dtype=np.float32, shape=(total_rows, n_genes))
    conditioner.eval()
    with torch.no_grad():
        offset = 0
        for inputs, targets in train_examples:
            target_expression = torch.as_tensor(targets.query_expression, dtype=torch.float32, device=device)
            out = conditioner(inputs)
            residual = target_expression - torch.as_tensor(out["expression"], dtype=torch.float32, device=device)
            n_rows = residual.shape[0]
            residuals[offset:offset + n_rows] = residual.detach().cpu().numpy().astype(np.float32)
            offset += n_rows
    residuals.flush()
    if not np.all(np.isfinite(residuals)):
        raise ValueError("compute_gen4_training_residuals: computed residuals contain non-finite values")
    return residuals


def fit_gen4_residual_basis(
    conditioner, train_examples: list, gene_names: list[str], *, rank: int = 64,
    device: torch.device | None = None, output_basis_path: str | Path,
) -> GeneResidualBasis:
    device = device or torch.device("cpu")
    memmap_path = Path(f"{output_basis_path}.residuals.tmp.{os.getpid()}.npy")
    try:
        residuals = compute_gen4_training_residuals(conditioner, train_examples, device, memmap_path=memmap_path)
        basis = fit_gene_residual_basis(residuals, gene_names, rank=rank)
        del residuals
    finally:
        memmap_path.unlink(missing_ok=True)
    save_gene_residual_basis(basis, output_basis_path)
    return basis
