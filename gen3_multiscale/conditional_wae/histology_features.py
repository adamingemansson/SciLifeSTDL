"""Deterministic, multiscale H&E morphology/context features for the MK
conditional-WAE histology-structure ablation.

Complementary to, not a duplicate of, `Architecture1ImageConditioner`
(see the audit this module's design is grounded in): that conditioner
already mixes context across spots via geometry-aware self-attention
over the OPAQUE 1536-d frozen GigaPath embedding -- either fully dense
(<=`dense_threshold` visible spots) or an expanding sparse k-NN
neighborhood, in both cases a LEARNED, uninterpretable signal. This
module instead computes EXPLICIT, deterministic, interpretable
descriptors (color/stain statistics, texture) directly from the same
224x224 H&E tile every spot already has, pooled at four scales (its own
tile, its immediate spatial neighbors, a larger regional radius, and the
whole visible slide) -- a genuinely different signal type, not more
mixing of the same embedding.

No nuclear segmentation: a real, code-search audit of this repository
found no validated nuclear-segmentation method anywhere (the one hit was
an unrelated antialiasing-preprocessing docstring). Per the explicit
instruction "do not silently use random models or claim nuclear
morphology without actual segmentation," nuclear density/size features
are deliberately NOT computed here -- only stain-deconvolution color
statistics and GLCM texture, both fixed, deterministic formulas (skimage
"HED" stain matrix -- the standard Ruifrok & Johnston deconvolution --
and grey-level co-occurrence matrix texture), never a trained or
randomly-initialized model.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

_TILE_FEATURE_DIM = 14
# own-tile + neighbor(mean,std) + regional(mean,std) + slide(mean,std)
FEATURE_DIM = _TILE_FEATURE_DIM * 7
FEATURE_SPEC = "histology_morphology_v1:rgb6+hed4+glcm4:neighbor+regional+slide_mean_std"
_GLCM_GRAY_LEVELS = 32
_GLCM_DISTANCES = (1, 3)
_GLCM_ANGLES = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)


def compute_tile_morphology_features(patch: np.ndarray) -> np.ndarray:
    """Deterministic feature vector from ONE [H, W, 3] uint8 (or float in
    [0, 255]/[0, 1]) H&E tile: RGB channel mean+std (6), HED
    (hematoxylin/eosin) stain-deconvolved channel mean+std (4, skimage's
    standard Ruifrok matrix via `skimage.color.rgb2hed`), and grey-level
    co-occurrence-matrix texture (contrast, homogeneity, energy,
    correlation -- averaged over multiple distances/angles for rotation
    robustness) (4). Always exactly `_TILE_FEATURE_DIM` (14) values,
    finite by construction for a real non-degenerate tile."""
    from skimage.color import rgb2hed
    from skimage.feature import graycomatrix, graycoprops

    patch = np.asarray(patch)
    if patch.ndim != 3 or patch.shape[-1] != 3:
        raise ValueError(f"patch must be [H, W, 3], got shape {patch.shape}")
    if patch.max() > 1.5:  # heuristically 0-255 range, matches this codebase's own convention
        rgb01 = patch.astype(np.float64) / 255.0
    else:
        rgb01 = patch.astype(np.float64)

    rgb_stats = np.concatenate([rgb01.mean(axis=(0, 1)), rgb01.std(axis=(0, 1))])

    hed = rgb2hed(rgb01)
    hed_stats = np.concatenate([
        hed[..., :2].mean(axis=(0, 1)), hed[..., :2].std(axis=(0, 1)),
    ])  # H and E channels only -- the third (DAB) channel is irrelevant for H&E-stained tissue

    gray = (rgb01.mean(axis=-1) * (_GLCM_GRAY_LEVELS - 1)).round().astype(np.uint8)
    glcm = graycomatrix(
        gray, distances=list(_GLCM_DISTANCES), angles=list(_GLCM_ANGLES),
        levels=_GLCM_GRAY_LEVELS, symmetric=True, normed=True,
    )
    glcm_stats = np.array([
        graycoprops(glcm, prop).mean()
        for prop in ("contrast", "homogeneity", "energy", "correlation")
    ])

    features = np.concatenate([rgb_stats, hed_stats, glcm_stats]).astype(np.float32)
    if features.shape != (_TILE_FEATURE_DIM,):
        raise RuntimeError(f"internal error: tile feature vector has shape {features.shape}, expected (14,)")
    if not np.isfinite(features).all():
        # A real but degenerate tile (e.g. a uniform-color patch makes GLCM
        # correlation's variance denominator zero) -- skimage returns NaN
        # for that specific statistic; replace non-finite entries with 0.0
        # (a legitimate "no texture variation" value) rather than propagate
        # NaN into a cache a later training run would then reject.
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


def compute_spot_tile_features(patches: np.ndarray, image_source_available: np.ndarray) -> np.ndarray:
    """[N, H, W, 3] + [N] bool -> [N, `_TILE_FEATURE_DIM`], an explicit
    zero row for every spot with `image_source_available=False` --
    mirrors `spot_feature_cache.py`'s own zero-placeholder contract; the
    tile-morphology function is never called on an unavailable/
    placeholder patch."""
    patches = np.asarray(patches)
    availability = np.asarray(image_source_available, dtype=bool)
    if patches.shape[0] != availability.shape[0]:
        raise ValueError("patches and image_source_available must be row-aligned")
    n = patches.shape[0]
    features = np.zeros((n, _TILE_FEATURE_DIM), dtype=np.float32)
    for index in np.flatnonzero(availability):
        features[index] = compute_tile_morphology_features(patches[index])
    return features


def compute_multiscale_histology_features(
    tile_features: np.ndarray,
    coords: np.ndarray,
    image_source_available: np.ndarray,
    *,
    neighbor_k: int = 6,
    regional_radius_multiplier: float = 5.0,
) -> np.ndarray:
    """[N, `_TILE_FEATURE_DIM`] real per-tile features + [N, 2] real
    physical spot coordinates -> [N, `FEATURE_DIM`]: for every spot,
    concatenate its own tile features with (mean, std) pooled over three
    progressively larger scales -- its `neighbor_k` nearest OTHER
    AVAILABLE spots, every AVAILABLE spot within `regional_radius_
    multiplier` times the median nearest-neighbor spacing (a coarser
    "regional" scale), and every AVAILABLE spot on the whole visible
    slide (one shared vector, identical for every spot in this sample).
    Unavailable spots are excluded from every aggregate they would
    otherwise pollute (their own zero-placeholder tile feature is never
    averaged into a real neighbor's context) -- their OWN row keeps a
    zero own-tile feature (matching `compute_spot_tile_features`) but
    still receives real neighbor/regional/slide aggregates from nearby
    AVAILABLE spots, exactly like GigaPath features are handled
    elsewhere for a spot with a real coordinate but no matching H&E
    patch. A scale with zero eligible spots falls back to the next
    coarser scale (neighbor -> regional -> slide); if the ENTIRE slide
    has zero available spots, every aggregate is zero (nothing to
    aggregate)."""
    tile_features = np.asarray(tile_features, dtype=np.float32)
    coords = np.asarray(coords, dtype=np.float64)
    availability = np.asarray(image_source_available, dtype=bool)
    n = tile_features.shape[0]
    if tile_features.shape != (n, _TILE_FEATURE_DIM):
        raise ValueError(f"tile_features must be [N, {_TILE_FEATURE_DIM}], got {tile_features.shape}")
    if coords.shape != (n, 2):
        raise ValueError(f"coords must be [{n}, 2], got {coords.shape}")
    if availability.shape != (n,):
        raise ValueError(f"image_source_available must be [{n}], got {availability.shape}")

    available_idx = np.flatnonzero(availability)
    slide_stats = _mean_std(tile_features[available_idx]) if available_idx.size else _zero_mean_std()

    if available_idx.size >= 2:
        tree = cKDTree(coords[available_idx])
        nn_distances, _ = tree.query(coords[available_idx], k=2)
        median_spacing = max(float(np.median(nn_distances[:, 1])), 1e-6)
        radius = regional_radius_multiplier * median_spacing
    else:
        tree = None
        radius = 0.0

    out = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    for row in range(n):
        own = tile_features[row]
        neighbor_stats = slide_stats
        regional_stats = slide_stats
        if tree is not None:
            k = min(neighbor_k + 1, available_idx.size)  # +1: query point may include itself
            distances, local_idx = tree.query(coords[row], k=k)
            local_idx = np.atleast_1d(local_idx)
            neighbor_global_idx = available_idx[local_idx]
            neighbor_global_idx = neighbor_global_idx[neighbor_global_idx != row]
            if neighbor_global_idx.size:
                neighbor_stats = _mean_std(tile_features[neighbor_global_idx])

            regional_local_idx = tree.query_ball_point(coords[row], r=radius)
            regional_global_idx = available_idx[np.asarray(regional_local_idx, dtype=np.int64)]
            regional_global_idx = regional_global_idx[regional_global_idx != row]
            if regional_global_idx.size:
                regional_stats = _mean_std(tile_features[regional_global_idx])
            elif neighbor_global_idx.size:
                regional_stats = neighbor_stats
        out[row] = np.concatenate([own, neighbor_stats, regional_stats, slide_stats])
    return out


def _mean_std(rows: np.ndarray) -> np.ndarray:
    return np.concatenate([rows.mean(axis=0), rows.std(axis=0)]).astype(np.float32)


def _zero_mean_std() -> np.ndarray:
    return np.zeros(_TILE_FEATURE_DIM * 2, dtype=np.float32)
