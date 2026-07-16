"""
Smoke tests for the mask-geometry additions (2026-07-16, "better masks" —
item 3 of the overnight autonomous-work plan): elliptical_hole_2d,
irregular_blob_2d, sparse_spot_dropout, random_dropout_patches's new
`shape` switch, and mixed_dropout (src/data/masking.py). No external
dependency — pure numpy, same as the rest of this module.

Run with: python -m tests.test_masking
"""
import numpy as np

from src.data.masking import (
    mask_region_2d, elliptical_hole_2d, irregular_blob_2d,
    sparse_spot_dropout, random_dropout_patches, mixed_dropout,
)


def _grid(n_side=40, spacing=5.0):
    """Regular grid centered near the origin — makes circle/ellipse/blob
    membership easy to reason about geometrically."""
    xs = np.arange(n_side) * spacing - (n_side * spacing) / 2
    xx, yy = np.meshgrid(xs, xs)
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1)
    slice_ids = np.zeros(len(coords), dtype=int)
    return coords, slice_ids


def test_elliptical_hole_reduces_to_circle():
    coords, slice_ids = _grid()
    # radius_a == radius_b, angle=0 -> must match mask_region_2d exactly
    _, circle_query = mask_region_2d(coords, slice_ids, 0, center=(0, 0), radius=50)
    _, ellipse_query = elliptical_hole_2d(coords, slice_ids, 0, center=(0, 0),
                                           radius_a=50, radius_b=50, angle=0.0)
    assert np.array_equal(circle_query, ellipse_query), (
        "elliptical_hole_2d with equal semi-axes must exactly match mask_region_2d"
    )
    print("[elliptical_hole_2d] OK — reduces exactly to a circle when radius_a==radius_b")


def test_elliptical_hole_is_actually_elongated():
    coords, slice_ids = _grid()
    # a very eccentric ellipse (radius_a >> radius_b) must include points
    # far along the major axis that a circle of the SAME area would not,
    # and exclude points along the minor axis that the circle WOULD include
    _, ellipse_query = elliptical_hole_2d(coords, slice_ids, 0, center=(0, 0),
                                           radius_a=100, radius_b=20, angle=0.0)
    far_on_major_axis = np.argmin(np.linalg.norm(coords - np.array([90, 0]), axis=1))
    far_on_minor_axis = np.argmin(np.linalg.norm(coords - np.array([0, 90]), axis=1))
    assert ellipse_query[far_on_major_axis], "point far along the major axis should be inside the ellipse"
    assert not ellipse_query[far_on_minor_axis], "point far along the minor axis should be outside the ellipse"
    print("[elliptical_hole_2d] OK — genuinely elongated, not just a renamed circle")


def test_elliptical_hole_rotation_changes_shape():
    coords, slice_ids = _grid()
    _, query_0 = elliptical_hole_2d(coords, slice_ids, 0, center=(0, 0),
                                     radius_a=100, radius_b=20, angle=0.0)
    _, query_90 = elliptical_hole_2d(coords, slice_ids, 0, center=(0, 0),
                                      radius_a=100, radius_b=20, angle=np.pi / 2)
    assert not np.array_equal(query_0, query_90), "rotating the ellipse 90 degrees must change membership"
    # rotating exactly 90 degrees swaps the roles of x/y for an ellipse
    # centered at the origin on a symmetric grid — same point COUNT
    assert query_0.sum() == query_90.sum(), "90-degree rotation on a symmetric grid should preserve the point count"
    print("[elliptical_hole_2d] OK — rotation changes which points are covered")


def test_irregular_blob_differs_from_circle_but_is_bounded():
    coords, slice_ids = _grid()
    _, circle_query = mask_region_2d(coords, slice_ids, 0, center=(0, 0), radius=80)
    _, blob_query = irregular_blob_2d(coords, slice_ids, 0, center=(0, 0),
                                       base_radius=80, strength=0.4, seed=0)
    assert not np.array_equal(circle_query, blob_query), (
        "irregular_blob_2d should not degenerate into an exact circle"
    )
    # every masked point must still be within a bounded distance of the
    # center — worst case, all n_harmonics=4 cosine terms align to +1
    # simultaneously at some theta, each with amplitude up to 1.0, so
    # radius(theta) maxes out at base_radius * (1 + strength * n_harmonics)
    n_harmonics, strength = 4, 0.4
    max_possible_radius = 80 * (1 + strength * n_harmonics)
    dist = np.linalg.norm(coords[blob_query], axis=1)
    assert dist.max() <= max_possible_radius, (
        f"blob extends further than its theoretical bound allows: "
        f"max dist {dist.max()} > {max_possible_radius}"
    )
    # reproducible given the same seed
    _, blob_query_again = irregular_blob_2d(coords, slice_ids, 0, center=(0, 0),
                                             base_radius=80, strength=0.4, seed=0)
    assert np.array_equal(blob_query, blob_query_again), "same seed must reproduce the exact same blob"
    # different seed -> different blob
    _, blob_query_seed1 = irregular_blob_2d(coords, slice_ids, 0, center=(0, 0),
                                             base_radius=80, strength=0.4, seed=1)
    assert not np.array_equal(blob_query, blob_query_seed1), "different seeds should produce different blobs"
    print("[irregular_blob_2d] OK — non-circular, bounded, seed-reproducible, seed-varying")


def test_sparse_spot_dropout():
    coords, slice_ids = _grid(n_side=60)  # 3600 points, enough for a stable fraction estimate
    _, query_mask = sparse_spot_dropout(coords, slice_ids, fraction=0.05, seed=0)
    frac = query_mask.mean()
    assert 0.03 < frac < 0.07, f"expected ~5% dropout, got {frac:.3f}"

    # real structural check vs. every hole-shape function above: dropped
    # points should NOT be spatially contiguous — most dropped points
    # should have at least one non-dropped immediate grid neighbor,
    # unlike a contiguous hole where interior points are surrounded only
    # by other dropped points
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    dropped_idx = np.where(query_mask)[0]
    isolated_count = 0
    for idx in dropped_idx[:200]:  # sample for speed
        neighbor_idx = tree.query(coords[idx], k=5)[1][1:]  # nearest 4 neighbors, excluding self
        if not query_mask[neighbor_idx].all():
            isolated_count += 1
    assert isolated_count > 0.5 * min(200, len(dropped_idx)), (
        "sparse_spot_dropout should scatter drops, not cluster them into a contiguous region"
    )
    print(f"[sparse_spot_dropout] OK — fraction={frac:.3f}, drops are spatially scattered")


def test_random_dropout_patches_shape_switch():
    coords, slice_ids = _grid()
    for shape in ("circle", "ellipse", "irregular", "mixed"):
        _, query_mask = random_dropout_patches(
            coords, slice_ids, n_patches=3, radius_range=(30, 60), shape=shape, seed=0,
        )
        assert query_mask.any(), f"shape={shape!r} produced an empty query mask"
        assert query_mask.sum() < len(coords), f"shape={shape!r} masked every point"
    print("[random_dropout_patches] OK — shape switch produces valid, non-degenerate masks for all 4 modes")


def test_mixed_dropout_combines_both_mechanisms():
    coords, slice_ids = _grid(n_side=50)
    _, patches_only = random_dropout_patches(coords, slice_ids, n_patches=2, radius_range=(40, 70),
                                              shape="circle", seed=0)
    _, mixed_query = mixed_dropout(coords, slice_ids, n_patches=2, radius_range=(40, 70),
                                    sparse_fraction=0.03, shape="circle", seed=0)
    # mixed_dropout must mask AT LEAST as many points as the patch-only
    # draw (sparse dropout only adds, via OR, never removes)
    assert mixed_query.sum() >= patches_only.sum(), (
        "mixed_dropout should mask at least as many points as its patch component alone"
    )
    # and strictly more in practice (sparse_fraction=0.03 on a 2500-point
    # grid should add scattered points outside the patches with overwhelming probability)
    assert mixed_query.sum() > patches_only.sum(), (
        "mixed_dropout's sparse component doesn't appear to be contributing anything"
    )
    print(f"[mixed_dropout] OK — patches alone: {patches_only.sum()}, "
          f"patches+sparse: {mixed_query.sum()}")


if __name__ == "__main__":
    test_elliptical_hole_reduces_to_circle()
    test_elliptical_hole_is_actually_elongated()
    test_elliptical_hole_rotation_changes_shape()
    test_irregular_blob_differs_from_circle_but_is_bounded()
    test_sparse_spot_dropout()
    test_random_dropout_patches_shape_switch()
    test_mixed_dropout_combines_both_mechanisms()
    print("\nAll mask-geometry smoke tests passed.")
