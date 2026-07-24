"""Boundary-patch leakage check for target_zero image masking.

Extracted from src/data/slide_context.py (2026-07-25) — that file's other
functions (load_slide_context, visible_slide_context) implement a separate
WSI-dense-tile mechanism used only by the hierarchical_slide model family,
which none of the gen2 architectures use. This one function is a real,
general leakage-prevention check (a broken/missing tissue region removes
PIXELS, not just query-centered patches — an observed context patch whose
footprint overlaps the hole still leaks part of the "absent" H&E into the
model unless excluded), so it's kept on its own here rather than dragging
in the whole slide-tiling subsystem.
"""
from __future__ import annotations

import numpy as np


def _overlaps_query_hole(
    tile_centers: np.ndarray,
    tile_half_size: float,
    query_centers: np.ndarray,
    query_half_size: float,
    chunk_size: int = 8192,
) -> np.ndarray:
    """Conservative axis-aligned tile/target-patch intersection test."""
    overlap = np.zeros(tile_centers.shape[0], dtype=bool)
    limit = float(tile_half_size + query_half_size)
    for start in range(0, tile_centers.shape[0], chunk_size):
        block = tile_centers[start : start + chunk_size]
        delta = np.abs(block[:, None, :] - query_centers[None, :, :])
        overlap[start : start + len(block)] = np.any(
            (delta[..., 0] < limit) & (delta[..., 1] < limit), axis=1
        )
    return overlap


def nonoverlapping_context_patch_mask(
    context_coords: np.ndarray,
    query_coords: np.ndarray,
    patch_size: float,
) -> np.ndarray:
    """Context spot patches whose raw pixels do not intersect the hole."""
    return ~_overlaps_query_hole(
        np.asarray(context_coords[:, :2], dtype=np.float32),
        float(patch_size) / 2.0,
        np.asarray(query_coords[:, :2], dtype=np.float32),
        float(patch_size) / 2.0,
    )
