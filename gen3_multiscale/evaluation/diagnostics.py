"""Evaluation-time modality-ablation interventions -- Phase 7 of the
handoff: "Add evaluation interventions using the same trained checkpoint:
global slide token zeroed; global slide token swapped with another
held-out slide; observed GEX shuffled among coordinates; all observed GEX
zeroed; all H&E zeroed; boundary GEX shuffled while local neighbours
remain intact; context-token order randomly permuted while predictions
remain equivalent up to numerical tolerance. These are diagnostic
evaluations, not extra 24-hour training arms."

Each intervention below returns a NEW SpatialFieldInputs (frozen
dataclasses.replace, never in-place mutation) built from an existing,
already-validated example, so a caller runs a real architecture's
forward() on the original and the perturbed inputs and compares the two
outputs -- these functions only build the perturbed input, they do not
run a model or compute a diagnostic verdict themselves (that's the
caller's job, exercised directly in tests/test_diagnostics.py against
real Architecture1/3 forward passes).

SCOPE LIMITATION, documented rather than silently worked around: the
global-slide-token zero/swap diagnostic requires a real LongNet global
token wired into an architecture's forward() -- CONTRACT.md records that
this wiring (`use_global_slide=True`) is not yet implemented in
architectures.py (raises NotImplementedError). zero_global_slide_vector/
swap_global_slide_vector below operate directly on a plain tensor and are
exercised against MultiscaleBlock/SpatialFieldBackbone's own
use_global_slide path (which DOES already accept a global_slide_vector
argument, per backbone.py) -- proving the mechanism itself responds
correctly now, ready for the day an architecture wrapper actually
supplies a real vector, rather than leaving this diagnostic entirely
unbuilt.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from gen3_multiscale.data.example import SpatialFieldInputs


def zero_observed_gex(inputs: SpatialFieldInputs) -> SpatialFieldInputs:
    """"All observed GEX zeroed" -- both the compact conditioning
    features and the untouched full-gene values every candidate's
    transport weight is scored against."""
    return replace(
        inputs,
        observed_gex_conditioning=np.zeros_like(inputs.observed_gex_conditioning),
        observed_full_gene_expression=np.zeros_like(inputs.observed_full_gene_expression),
    )


def shuffle_observed_gex(inputs: SpatialFieldInputs, seed: int = 0) -> SpatialFieldInputs:
    """"Observed GEX shuffled among coordinates" -- coordinates and
    H&E/candidate IDENTITY stay in place; the GEX profile that lands at
    each observed position is permuted, decoupling expression from real
    spatial location while keeping the same multiset of values."""
    n_observed = inputs.observed_coords.shape[0]
    if n_observed < 2:
        return inputs
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_observed)
    return replace(
        inputs,
        observed_gex_conditioning=inputs.observed_gex_conditioning[perm],
        observed_full_gene_expression=inputs.observed_full_gene_expression[perm],
    )


def zero_he(inputs: SpatialFieldInputs) -> SpatialFieldInputs:
    """"All H&E zeroed" -- local GigaPath tile features for every
    candidate spot, plus any dense WSI regional tiles if present."""
    updates: dict = {"observed_gigapath_features": np.zeros_like(inputs.observed_gigapath_features)}
    if inputs.wsi_tile_features is not None:
        updates["wsi_tile_features"] = np.zeros_like(inputs.wsi_tile_features)
    return replace(inputs, **updates)


def shuffle_boundary_gex(inputs: SpatialFieldInputs, seed: int = 0) -> SpatialFieldInputs:
    """"Boundary GEX shuffled while local neighbours remain intact."

    A subtlety this function handles explicitly: architectures.py's
    _candidate_pool concatenates local (query_local_neighbor_idx) and
    boundary (boundary_idx) candidates from the SAME
    observed_full_gene_expression array, and an observed spot can appear
    in both sets (a query's true-nearest local neighbor is often also a
    Ring-1 boundary spot -- CONTRACT.md's documented non-deduplication
    simplification). Shuffling every boundary_idx position would corrupt
    those doubly-referenced spots' LOCAL role too, violating "local
    neighbours remain intact." Only observed positions that are in
    boundary_idx and NOT in any query's local_neighbor set are shuffled;
    if fewer than 2 such positions exist, the input is returned unchanged
    (nothing can be shuffled without touching a local neighbor)."""
    local_positions = {int(p) for p in np.asarray(inputs.query_local_neighbor_idx).reshape(-1)}
    boundary_positions = [int(p) for p in np.asarray(inputs.boundary_idx)]
    shuffle_only = np.asarray([p for p in boundary_positions if p not in local_positions], dtype=int)
    if shuffle_only.shape[0] < 2:
        return inputs
    rng = np.random.default_rng(seed)
    permuted = rng.permutation(shuffle_only)
    new_full = np.array(inputs.observed_full_gene_expression, copy=True)
    new_cond = np.array(inputs.observed_gex_conditioning, copy=True)
    new_full[shuffle_only] = inputs.observed_full_gene_expression[permuted]
    new_cond[shuffle_only] = inputs.observed_gex_conditioning[permuted]
    return replace(inputs, observed_full_gene_expression=new_full, observed_gex_conditioning=new_cond)


def permute_boundary_order(inputs: SpatialFieldInputs, seed: int = 0) -> SpatialFieldInputs:
    """"Context-token order randomly permuted while predictions remain
    equivalent up to numerical tolerance" -- reorders boundary_idx/
    boundary_ring (aligned together) WITHOUT changing which observed
    positions they reference or how many rings each belongs to. A correct
    architecture's forward() output must be unchanged (within floating-
    point tolerance) between the original and this permuted input, since
    ChunkedCrossAttention (Phase 5) is permutation-invariant to context
    ordering by construction -- this is the integration-level check of
    that unit-level guarantee."""
    n_boundary = inputs.boundary_idx.shape[0]
    if n_boundary < 2:
        return inputs
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_boundary)
    return replace(
        inputs,
        boundary_idx=inputs.boundary_idx[perm],
        boundary_ring=inputs.boundary_ring[perm],
    )


def zero_global_slide_vector(vector: torch.Tensor) -> torch.Tensor:
    """"Global slide token zeroed" -- see module docstring's scope note:
    exercised against MultiscaleBlock/SpatialFieldBackbone directly
    (use_global_slide=True), not yet against a full architecture wrapper."""
    return torch.zeros_like(vector)


def swap_global_slide_vector(vector: torch.Tensor, other_vector: torch.Tensor) -> torch.Tensor:
    """"Global slide token swapped with another held-out slide" -- see
    module docstring's scope note."""
    if other_vector.shape != vector.shape:
        raise ValueError(
            f"swap_global_slide_vector requires a replacement of shape {tuple(vector.shape)}, "
            f"got {tuple(other_vector.shape)}"
        )
    return other_vector
