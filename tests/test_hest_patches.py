"""
Smoke test for HEST-1k H&E patch loading (src/data/loaders.py, task #17).
Builds a tiny synthetic patches/*.h5 file matching the real format
(h5py keys 'img'/'coords'/'barcodes', confirmed against HESTData's actual
dump_patches() source — see load_hest_patches() docstring) plus a matching
synthetic AnnData, in a shuffled order, to check alignment actually
reorders correctly rather than assuming barcode order already matches.

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
            f.create_dataset("barcodes", data=[b.encode() for b in barcodes])

        loaded_patches, loaded_barcodes = load_hest_patches(tmp_dir, "INT1")
        assert loaded_patches.shape == (n, 8, 8, 3)
        assert set(loaded_barcodes) == set(barcodes)
        print(f"[load_hest_patches] OK — loaded {loaded_patches.shape[0]} patches")

        # adata in a DIFFERENT (shuffled) order than the .h5 file — alignment
        # must actually reorder, not just assume matching order
        shuffled = rng.permutation(barcodes)
        adata = ad.AnnData(X=np.zeros((n, 5)))
        adata.obs_names = shuffled

        aligned = align_patches_to_adata(adata, loaded_patches, loaded_barcodes)
        assert aligned.shape == (n, 8, 8, 3)
        barcode_to_idx = {b: i for i, b in enumerate(loaded_barcodes)}
        for i, name in enumerate(adata.obs_names):
            expected = loaded_patches[barcode_to_idx[name]]
            assert np.array_equal(aligned[i], expected), f"misaligned patch at row {i} ({name})"
        print("[align_patches_to_adata] OK — patches correctly reordered to match adata.obs_names")

        # missing-barcode case must raise loudly, not silently misalign
        adata_missing = ad.AnnData(X=np.zeros((1, 5)))
        adata_missing.obs_names = ["not_a_real_spot"]
        try:
            align_patches_to_adata(adata_missing, loaded_patches, loaded_barcodes)
            raised = False
        except ValueError:
            raised = True
        assert raised, "missing barcode should raise, not silently return misaligned data"
        print("[align_patches_to_adata] OK — raises on missing barcode")


if __name__ == "__main__":
    test_load_and_align()
    print("\nAll HEST-1k H&E patch loading smoke tests passed.")
