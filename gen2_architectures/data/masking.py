"""
Gap simulators: take a COMPLETE, real dataset and hide part of it, so we have
ground truth to evaluate reconstruction against. This is the shared machinery
behind both tasks in the project:

  - hold_out_slice()   -> simulates a missing intermediate section (3D task)
  - mask_region_2d()   -> simulates torn/damaged tissue within one slice
                          (intra-slice task)

Keeping both in one module because they should share config plumbing (e.g.
"fraction of tissue removed") and both feed the same evaluation code.

2026-07-16 follow-up ("better masks", item 3 of the overnight autonomous-
work plan): every hole shape above this date was a perfect circle. Real
tissue damage/folding/tearing is essentially never a perfect circle, and a
model trained exclusively against circular gaps risks learning a
shape-specific shortcut (e.g. "context always forms a clean boundary at a
fixed distance from a hole's center") rather than genuine spatial
interpolation. Added below: elliptical holes (rotation + independent
axes — cheapest generalization of a circle), irregular "torn" blobs (a
randomized angular radius function — organic, non-convex boundaries, no
polygon/shapely dependency needed), and sparse single-spot dropout (a
structurally DIFFERENT failure mode from a contiguous hole — scattered
low-quality spots dropped independently, as real QC pipelines do, not one
contiguous missing region). random_dropout_patches gained a `shape` switch
covering the first two; mixed_dropout (new) combines patches of varied
shape with sparse dropout in one call, for training runs that want every
geometry represented across a single dataset rather than picking one.
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

    For non-circular geometry, see elliptical_hole_2d/irregular_blob_2d
    below (2026-07-16) — this function is kept exactly as-is (still the
    simplest/cheapest shape, and used directly by callers/tests that don't
    need the generalization)."""
    in_slice = slice_ids == target_slice
    dist = np.linalg.norm(coords_xy - np.array(center), axis=1)
    hole = in_slice & (dist <= radius)
    context_mask = ~hole
    query_mask = hole
    return context_mask, query_mask


def elliptical_hole_2d(coords_xy: np.ndarray, slice_ids: np.ndarray, target_slice,
                        center: tuple[float, float], radius_a: float, radius_b: float,
                        angle: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Elliptical generalization of mask_region_2d (2026-07-16) —
    radius_a/radius_b are the semi-axes (radius_a==radius_b reduces exactly
    to a circle of that radius), `angle` rotates the ellipse (radians,
    counterclockwise) so holes aren't always axis-aligned. Cheapest real
    generalization of "damage is round": actual tissue tears/folds are
    rarely isotropic — a knife nick or a fold line stretches much further
    in one direction than the other, which a circular hole can never
    represent regardless of radius."""
    in_slice = slice_ids == target_slice
    offset = coords_xy - np.array(center)
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    # rotate offset into the ellipse's own (unrotated) frame, then apply
    # the standard axis-aligned ellipse membership test
    x_rot = offset[:, 0] * cos_a - offset[:, 1] * sin_a
    y_rot = offset[:, 0] * sin_a + offset[:, 1] * cos_a
    inside_ellipse = (x_rot / radius_a) ** 2 + (y_rot / radius_b) ** 2 <= 1.0
    hole = in_slice & inside_ellipse
    context_mask = ~hole
    query_mask = hole
    return context_mask, query_mask


def irregular_blob_2d(coords_xy: np.ndarray, slice_ids: np.ndarray, target_slice,
                       center: tuple[float, float], base_radius: float,
                       n_harmonics: int = 4, strength: float = 0.35,
                       seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Organic, non-convex "torn tissue" hole (2026-07-16) — the boundary's
    distance from `center` varies smoothly with angle via a random,
    band-limited Fourier series, rather than being constant (a circle) or
    a fixed ellipse. This is the shape family real tissue tears/irregular
    necrotic regions actually look like: a lumpy, non-convex blob, not a
    conic section.

    radius(theta) = base_radius * (1 + strength * sum_k a_k * cos(k*theta + phi_k))
    for k in 1..n_harmonics, with random unit-amplitude a_k/phase phi_k
    drawn once per call (seed-controlled for reproducibility, same
    convention as every other seeded function in this module). strength
    controls how far the boundary deviates from a plain circle of
    base_radius — kept < ~0.5 by convention (not enforced) so the radius
    function stays positive almost everywhere for any reasonable n_harmonics
    (a strict positivity guarantee isn't needed here: a radius that dips
    below zero at some angle just means "no hole at that angle", not a
    crash — np.clip below simply forbids inverting the sign entirely)."""
    rng = np.random.default_rng(seed)
    amplitudes = rng.uniform(0.5, 1.0, size=n_harmonics)
    phases = rng.uniform(0, 2 * np.pi, size=n_harmonics)
    harmonics = np.arange(1, n_harmonics + 1)

    in_slice = slice_ids == target_slice
    offset = coords_xy - np.array(center)
    theta = np.arctan2(offset[:, 1], offset[:, 0])          # [N]
    dist = np.linalg.norm(offset, axis=1)                    # [N]

    # radius(theta) for every point at once: [N, n_harmonics] broadcast sum
    wobble = (amplitudes[None, :] * np.cos(harmonics[None, :] * theta[:, None] + phases[None, :])).sum(axis=1)
    radius_at_theta = base_radius * np.clip(1.0 + strength * wobble, 0.05, None)

    hole = in_slice & (dist <= radius_at_theta)
    context_mask = ~hole
    query_mask = hole
    return context_mask, query_mask


def sparse_spot_dropout(coords_xy: np.ndarray, slice_ids: np.ndarray,
                         fraction: float = 0.05, seed: int | None = None
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Drop individual spots UNIFORMLY AT RANDOM across every slice
    (2026-07-16) — a structurally different failure mode from every hole
    shape above: real Visium/Xenium QC pipelines routinely drop scattered
    individual low-quality spots (tissue-fold artifacts, low read depth,
    doublets) that are NOT spatially contiguous, unlike a torn/damaged
    region. A model that only ever sees contiguous-hole masking during
    training has no pressure to handle "one isolated missing point deep
    inside otherwise-dense context" well specifically (it can always lean
    on nearby-in-every-direction context for a hole's interior, but a
    sparse dropped spot's neighbors are ALSO sometimes dropped at random,
    changing the local context density in a qualitatively different way).

    fraction: expected fraction of ALL points (across every slice) dropped
    into the query set, independently per point (i.i.d. Bernoulli(fraction),
    not a fixed count) — matches how QC dropout actually behaves (each spot
    independently at risk, not "exactly N% every time")."""
    rng = np.random.default_rng(seed)
    query_mask = rng.random(len(coords_xy)) < fraction
    context_mask = ~query_mask
    return context_mask, query_mask


def random_dropout_patches(coords_xy: np.ndarray, slice_ids: np.ndarray,
                            n_patches: int = 3, radius_range=(50, 150),
                            seed: int | None = None,
                            shape: str = "circle",
                            radius_unit: str = "coordinate",
                            aspect_ratio_range=(1.5, 3.0),
                            blob_strength_range=(0.2, 0.45),
                            blob_harmonics: int = 4,
                            center_mode: str = "random",
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Multiple random holes per call — for stress-testing / making a held-
    out evaluation set with varied difficulty.

    shape (2026-07-16, "better masks" follow-up — was always "circle"
    before this): "circle" (default, unchanged behavior/output for any
    existing config not opting in), "ellipse" (elliptical_hole_2d, random
    aspect ratio drawn from aspect_ratio_range and random rotation per
    patch), "irregular" (irregular_blob_2d, random per-patch strength
    drawn from blob_strength_range), or "mixed" (each of the n_patches
    holes independently picks circle/ellipse/irregular uniformly at
    random — the single call that best matches "real damage isn't one
    consistent shape across a whole dataset"). radius_range is reused as
    the base/semi-major radius for every shape so existing configs' tuned
    radius_range values stay meaningful without retuning per shape.

    ``radius_unit="coordinate"`` is the exact historical behavior.  The
    opt-in ``spot_spacing`` mode expresses radii in median nearest-neighbour
    distances on the selected slice, making hole size comparable across
    slides with different pixel coordinate scales.

    ``center_mode="random"`` (default, unchanged behavior) picks a uniformly
    random spot on the selected slice as each patch's center. The opt-in
    ``"geometric_median"`` mode (2026-07-24, added to reproduce a reference
    STPath benchmark notebook's masking scheme for a direct comparison)
    instead always centers the patch on the slice's own geometric-median
    spot (nearest observed spot to the coordinate-wise median, same
    selection rule as that notebook's `choose_central_window`) -- every
    hole for a given slice lands in the same place, only n_patches'
    per-patch radius/shape randomness still varies it. Deliberately NOT the
    default: a hole that always sits in a slide's richest, most
    representative region is easier to interpolate than one placed
    uniformly at random (which can land near edges/sparse tissue), so
    "random" stays the right choice for a task meant to reflect real,
    arbitrarily-located tissue damage.
    """
    assert shape in ("circle", "ellipse", "irregular", "mixed"), f"unknown shape {shape!r}"
    assert radius_unit in ("coordinate", "spot_spacing"), f"unknown radius_unit {radius_unit!r}"
    assert center_mode in ("random", "geometric_median"), f"unknown center_mode {center_mode!r}"
    rng = np.random.default_rng(seed)
    query_mask = np.zeros(len(coords_xy), dtype=bool)
    unique_slices = np.unique(slice_ids)
    for i in range(n_patches):
        s = rng.choice(unique_slices)
        in_slice_idx = np.where(slice_ids == s)[0]
        if len(in_slice_idx) == 0:
            continue
        if center_mode == "geometric_median":
            slice_coords = coords_xy[in_slice_idx]
            target = np.median(slice_coords, axis=0)
            center_idx = in_slice_idx[int(np.argmin(np.linalg.norm(slice_coords - target, axis=1)))]
        else:
            center_idx = rng.choice(in_slice_idx)
        center = coords_xy[center_idx]
        radius = rng.uniform(*radius_range)
        if radius_unit == "spot_spacing":
            from scipy.spatial import cKDTree

            slice_coords = np.asarray(coords_xy[in_slice_idx], dtype=np.float64)
            if len(slice_coords) < 2:
                continue
            neighbour_distances, _ = cKDTree(slice_coords).query(slice_coords, k=2)
            spacing = float(np.median(neighbour_distances[:, 1]))
            if not np.isfinite(spacing) or spacing <= 0:
                raise ValueError(f"could not determine positive spot spacing for slice {s!r}")
            radius *= spacing
        patch_seed = None if seed is None else seed * 10_000 + i  # deterministic-but-distinct per patch

        this_shape = shape
        if shape == "mixed":
            this_shape = rng.choice(["circle", "ellipse", "irregular"])

        if this_shape == "circle":
            dist = np.linalg.norm(coords_xy - center, axis=1)
            hole = (slice_ids == s) & (dist <= radius)
        elif this_shape == "ellipse":
            aspect = rng.uniform(*aspect_ratio_range)
            angle = rng.uniform(0, 2 * np.pi)
            _, hole_mask = elliptical_hole_2d(
                coords_xy, slice_ids, s, tuple(center),
                radius_a=radius, radius_b=radius / aspect, angle=angle,
            )
            hole = hole_mask
        else:  # "irregular"
            strength = rng.uniform(*blob_strength_range)
            _, hole_mask = irregular_blob_2d(
                coords_xy, slice_ids, s, tuple(center), base_radius=radius,
                n_harmonics=blob_harmonics, strength=strength, seed=patch_seed,
            )
            hole = hole_mask
        query_mask |= hole
    context_mask = ~query_mask
    return context_mask, query_mask


def mixed_dropout(coords_xy: np.ndarray, slice_ids: np.ndarray,
                   n_patches: int = 3, radius_range=(50, 150),
                   sparse_fraction: float = 0.02,
                   shape: str = "mixed",
                   seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One call combining contiguous varied-shape holes (random_dropout_patches,
    shape="mixed" by default) with sparse independent single-spot dropout
    (sparse_spot_dropout) — 2026-07-16, "better masks" follow-up. The single
    richest masking strategy in this module: exercises large contiguous
    gaps of varied geometry AND scattered isolated missing spots in the
    same training draw, rather than a config having to pick just one
    masking strategy for an entire run. sparse_fraction applies on top of
    whatever random_dropout_patches already removed (independent draws,
    unioned) — a spot inside a patch hole that's ALSO independently chosen
    by the sparse draw is simply still held out, no double-counting issue
    since query_mask is a boolean OR."""
    patch_seed = seed
    sparse_seed = None if seed is None else seed + 1  # distinct draw from the patch RNG
    _, patch_query = random_dropout_patches(
        coords_xy, slice_ids, n_patches=n_patches, radius_range=radius_range,
        seed=patch_seed, shape=shape,
    )
    _, sparse_query = sparse_spot_dropout(
        coords_xy, slice_ids, fraction=sparse_fraction, seed=sparse_seed,
    )
    query_mask = patch_query | sparse_query
    context_mask = ~query_mask
    return context_mask, query_mask
