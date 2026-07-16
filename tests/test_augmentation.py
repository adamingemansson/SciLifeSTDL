"""
Smoke tests for coordinate-space augmentation (2026-07-16,
src/data/augmentation.py augment_coords_xy — fourth autonomous-research
addition this session, after organ/tech conditioning, relative-position
attention bias, and varied mask geometries). Pure numpy, no external
dependency.

Run with: python -m tests.test_augmentation
"""
import numpy as np

from src.data.augmentation import augment_coords_xy


def test_isometry_preserves_pairwise_distances():
    """The whole point of using rotation+reflection specifically: every
    pairwise distance (and therefore every k-NN graph / relative-position
    bias downstream) must be EXACTLY preserved, only the absolute frame
    changes."""
    rng = np.random.default_rng(0)
    n = 30
    coords3d = np.zeros((n, 3))
    coords3d[:, :2] = rng.uniform(0, 500, size=(n, 2))
    coords3d[:, 2] = rng.uniform(0, 10, size=n)  # nonzero z, to also check it's untouched

    augmented = augment_coords_xy(coords3d, seed=0)

    dist_before = np.linalg.norm(coords3d[:, None, :2] - coords3d[None, :, :2], axis=-1)
    dist_after = np.linalg.norm(augmented[:, None, :2] - augmented[None, :, :2], axis=-1)
    assert np.allclose(dist_before, dist_after, atol=1e-8), (
        "pairwise xy distances must be exactly preserved by a rotation+reflection"
    )
    print("[augment_coords_xy] OK — pairwise distances preserved (isometry)")


def test_z_untouched():
    rng = np.random.default_rng(1)
    n = 20
    coords3d = np.zeros((n, 3))
    coords3d[:, :2] = rng.uniform(0, 500, size=(n, 2))
    coords3d[:, 2] = rng.uniform(-50, 50, size=n)

    augmented = augment_coords_xy(coords3d, seed=1)
    assert np.array_equal(coords3d[:, 2], augmented[:, 2]), "z must be left completely untouched"
    print("[augment_coords_xy] OK — z coordinate untouched")


def test_input_not_mutated():
    rng = np.random.default_rng(2)
    coords3d = rng.uniform(0, 500, size=(15, 3))
    original = coords3d.copy()
    _ = augment_coords_xy(coords3d, seed=2)
    assert np.array_equal(coords3d, original), "augment_coords_xy must not mutate its input in place"
    print("[augment_coords_xy] OK — input array not mutated")


def test_reproducible_given_seed():
    rng = np.random.default_rng(3)
    coords3d = rng.uniform(0, 500, size=(10, 3))
    a = augment_coords_xy(coords3d, seed=42)
    b = augment_coords_xy(coords3d, seed=42)
    assert np.array_equal(a, b), "same seed must reproduce the exact same augmented coordinates"
    print("[augment_coords_xy] OK — reproducible given the same seed")


def test_different_seeds_differ():
    rng = np.random.default_rng(4)
    coords3d = rng.uniform(0, 500, size=(10, 3))
    a = augment_coords_xy(coords3d, seed=1)
    b = augment_coords_xy(coords3d, seed=2)
    assert not np.allclose(a, b), "different seeds should (almost surely) produce different transforms"
    print("[augment_coords_xy] OK — different seeds produce different transforms")


def test_reflect_false_is_pure_rotation():
    """With reflect=False, the transform must be orientation-preserving
    (a proper rotation, determinant +1) — real check that reflect=False
    doesn't silently still flip half the time."""
    rng = np.random.default_rng(5)
    n = 3
    coords3d = np.zeros((n, 3))
    # a small right-handed triangle, signed area sign is the orientation check
    coords3d[:, :2] = np.array([[0, 0], [10, 0], [0, 10]])

    def signed_area(xy):
        return 0.5 * ((xy[1, 0] - xy[0, 0]) * (xy[2, 1] - xy[0, 1])
                       - (xy[2, 0] - xy[0, 0]) * (xy[1, 1] - xy[0, 1]))

    original_sign = np.sign(signed_area(coords3d[:, :2]))
    for seed in range(20):
        augmented = augment_coords_xy(coords3d, seed=seed, reflect=False)
        assert np.sign(signed_area(augmented[:, :2])) == original_sign, (
            f"reflect=False produced a flipped (orientation-reversed) triangle at seed={seed}"
        )
    print("[augment_coords_xy] OK — reflect=False never flips orientation across 20 seeds")


if __name__ == "__main__":
    test_isometry_preserves_pairwise_distances()
    test_z_untouched()
    test_input_not_mutated()
    test_reproducible_given_seed()
    test_different_seeds_differ()
    test_reflect_false_is_pure_rotation()
    print("\nAll augmentation smoke tests passed.")
