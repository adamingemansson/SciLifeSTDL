"""
Smoke test for HEST-1k H&E patch loading (src/data/loaders.py, task #17).
Builds a tiny synthetic patches/*.h5 file matching the REAL format
(h5py keys 'img'/'coords'/'barcode' — 'barcode' singular, [N,1] shaped —
confirmed 2026-07-15 against an actual downloaded file, correcting an
earlier version of this test based on source-reading alone) plus a
matching synthetic AnnData, in a shuffled order, to check alignment
actually reorders correctly rather than assuming barcode order matches.

Run with: python -m tests.test_hest_patches
"""
import tempfile
from pathlib import Path

import numpy as np
import h5py
import anndata as ad

from src.data.loaders import load_hest_patches, align_patches_to_adata


def test_load_and_align():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        patches_dir = tmp_dir / "patches"
        patches_dir.mkdir()

        n = 20
        rng = np.random.default_rng(0)
        barcodes = np.array([f"spot_{i}" for i in range(n)])
        img = rng.integers(0, 255, size=(n, 8, 8, 3), dtype=np.uint8)
        coords = rng.integers(0, 1000, size=(n, 2)).astype(np.int32)

        with h5py.File(patches_dir / "TEST_INT1.h5", "w") as f:
            f.create_dataset("img", data=img)
            f.create_dataset("coords", data=coords)
            f.create_dataset("barcode", data=[[b.encode()] for b in barcodes])  # [N, 1], real shape

        loaded_patches, loaded_barcodes = load_hest_patches(tmp_dir, "INT1")
        assert loaded_patches.shape == (n, 8, 8, 3)
        assert set(loaded_barcodes) == set(barcodes)
        print(f"[load_hest_patches] OK — loaded {loaded_patches.shape[0]} patches")

        # adata in a DIFFERENT (shuffled) order than the .h5 file — alignment
        # must actually reorder, not just assume matching order
        shuffled = rng.permutation(barcodes)
        adata = ad.AnnData(X=np.zeros((n, 5)))
        adata.obs_names = shuffled

        aligned_adata, aligned_patches = align_patches_to_adata(adata, loaded_patches, loaded_barcodes)
        assert aligned_patches.shape == (n, 8, 8, 3)
        assert aligned_adata.n_obs == n
        barcode_to_idx = {b: i for i, b in enumerate(loaded_barcodes)}
        for i, name in enumerate(aligned_adata.obs_names):
            expected = loaded_patches[barcode_to_idx[name]]
            assert np.array_equal(aligned_patches[i], expected), f"misaligned patch at row {i} ({name})"
        print("[align_patches_to_adata] OK — patches correctly reordered to match adata.obs_names")

        # partial gap (real-world case, e.g. INT1: 49/1080 spots missing a
        # patch) must SUBSET, not raise — HEST-1k's own patch extraction
        # naturally drops some spots, that's not an error
        adata_partial = ad.AnnData(X=np.zeros((n + 3, 5)))
        adata_partial.obs_names = list(shuffled) + ["no_patch_1", "no_patch_2", "no_patch_3"]
        filtered_adata, filtered_patches = align_patches_to_adata(adata_partial, loaded_patches, loaded_barcodes)
        assert filtered_adata.n_obs == n, f"expected {n} spots to survive, got {filtered_adata.n_obs}"
        assert filtered_patches.shape == (n, 8, 8, 3)
        assert "no_patch_1" not in filtered_adata.obs_names
        print(f"[align_patches_to_adata] OK — partial gap subsets correctly "
              f"({filtered_adata.n_obs}/{adata_partial.n_obs} spots kept)")

        # zero-overlap case must still raise loudly — that indicates a real
        # data-mismatch bug, not normal partial coverage
        adata_none = ad.AnnData(X=np.zeros((1, 5)))
        adata_none.obs_names = ["not_a_real_spot"]
        try:
            align_patches_to_adata(adata_none, loaded_patches, loaded_barcodes)
            raised = False
        except ValueError:
            raised = True
        assert raised, "zero barcode overlap should raise, not silently return empty data"
        print("[align_patches_to_adata] OK — raises when there is zero overlap")


if __name__ == "__main__":
    test_load_and_align()
    print("\nAll HEST-1k H&E patch loading smoke tests passed.")
