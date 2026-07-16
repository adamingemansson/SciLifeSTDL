"""
Coordinate-space data augmentation (2026-07-16, third autonomous-research
addition after organ/tech conditioning + varied mask geometries): training-
time random rotation/reflection of the xy plane, applied identically to an
entire context+query masking draw's coordinates at once.

Motivation: tissue orientation on a slide is an artifact of how it was
mounted/imaged, not a real feature of the underlying gene expression
pattern. A model that only ever sees "north" as a fixed direction has no
reason to generalize its spatial interpolation to a sample mounted at a
different angle, or a different sample entirely (whose orientation on the
slide is arbitrary relative to this one). Applying the SAME rigid
transform to every point in one draw preserves every pairwise relationship
the architecture actually reasons over (k-NN structure in
SpatialContextEncoder, self-attention in StormLiteContextEncoder,
RelativePositionBias's relative offsets) — it only changes the arbitrary
GLOBAL frame. This also functions as an explicit regularizer against
RandomFourierFeatures' absolute-position encoding becoming an exploitable
shortcut (e.g. memorizing "gene X is always high near coordinate (800,
1200)" instead of learning genuine local spatial structure).

z is deliberately left untouched: it's slice depth (Track B) or a constant
placeholder (Track A) — never spatially exchangeable with x/y (rotating it
into the xy-plane would fabricate a fake spatial relationship between
depth and position that doesn't exist in the real tissue geometry).

Opt-in only (see cfg.training.augment_coords in src/training/train.py) —
every existing config's behavior is completely unchanged unless a config
explicitly turns this on."""
from __future__ import annotations
import numpy as np


def augment_coords_xy(coords3d: np.ndarray, seed: int | None = None,
                       reflect: bool = True) -> np.ndarray:
    """coords3d: [N, 3]. Returns a NEW array (input never mutated) with x/y
    rotated by a random angle in [0, 2*pi) about the point set's own
    centroid, and independently reflected across the (post-rotation) x and
    y axes with 50% probability each when reflect=True (the default).
    Reflection is a genuinely separate degree of freedom from rotation —
    real tissue can be mounted "face up" or "face down" on a slide, a flip
    no amount of in-plane rotation alone can reproduce.

    A pure rotation+reflection is an ISOMETRY (distance-preserving) by
    construction — every pairwise distance, and therefore every k-NN
    graph/relative-position bias computed downstream, is identical before
    and after this transform. Only the arbitrary absolute frame changes."""
    rng = np.random.default_rng(seed)
    coords = coords3d.copy()
    xy = coords[:, :2]
    centroid = xy.mean(axis=0)
    theta = rng.uniform(0, 2 * np.pi)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    xy_rot = (xy - centroid) @ rotation.T
    if reflect:
        if rng.random() < 0.5:
            xy_rot[:, 0] *= -1
        if rng.random() < 0.5:
            xy_rot[:, 1] *= -1
    coords[:, :2] = xy_rot + centroid
    return coords
