"""
Gap simulators: take a COMPLETE, real dataset and hide part of it, so we have
ground truth to evaluate reconstruction against. This is the shared machinery
behind both tasks in the project:

  - hold_out_slice()   -> simulates a missing intermediate section (3D task)
  - mask_region_2d()   -> simulates torn/damaged tissue within one slice
                          (intra-slice task)

Keeping both in one module because they should share config plumbing (e.g.
"fraction of tissue removed") and both feed the same evaluation code.
"""
from __future__ import annotations
import numpy as np


def hold_out_slice(z_values: np.ndarray, slice_id: str | int, slice_ids: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (context_mask, query_mask): boolean arrays over all points.
    query = the held-out slice (ground truth for evaluation, hidden from the model)
    context = everything else (what the model conditions on)
    """
    query_mask = slice_ids == slice_id
    context_mask = ~query_mask
    return context_mask, query_mask


def mask_region_2d(coords_xy: np.ndarray, slice_ids: np.ndarray, target_slice,
                    center: tuple[float, float], radius: float
                    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate a circular 'damaged tissue' hole of given radius centered at
    `center`, only within `target_slice`. Everything outside that circle (in
    the target slice) plus all other slices form the context.

    For more realistic damage patterns (tears, folds) later: replace the
    circular mask with an irregular polygon, or better, sample real damage
    masks from QC-flagged regions in actual lab data if available.
    """
    in_slice = slice_ids == target_slice
    dist = np.linalg.norm(coords_xy - np.array(center), axis=1)
    hole = in_slice & (dist <= radius)
    context_mask = ~hole
    query_mask = hole
    return context_mask, query_mask


def random_dropout_patches(coords_xy: np.ndarray, slice_ids: np.ndarray,
                            n_patches: int = 3, radius_range=(50, 150),
                            seed: int | None = None
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Multiple random circular holes per call — for stress-testing / making
    a held-out evaluation set with varied difficulty."""
    rng = np.random.default_rng(seed)
    query_mask = np.zeros(len(coords_xy), dtype=bool)
    unique_slices = np.unique(slice_ids)
    for _ in range(n_patches):
        s = rng.choice(unique_slices)
        in_slice_idx = np.where(slice_ids == s)[0]
        if len(in_slice_idx) == 0:
            continue
        center_idx = rng.choice(in_slice_idx)
        center = coords_xy[center_idx]
        radius = rng.uniform(*radius_range)
        dist = np.linalg.norm(coords_xy - center, axis=1)
        query_mask |= (slice_ids == s) & (dist <= radius)
    context_mask = ~query_mask
    return context_mask, query_mask
