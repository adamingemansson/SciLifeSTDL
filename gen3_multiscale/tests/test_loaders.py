"""Direct unit tests for gen3_multiscale/data/loaders.py's
align_patches_to_adata -- 19th Codex re-audit (Step 5 Part 2, remaining
gap #6): a corrupted/malformed patches .h5 file with duplicate barcodes,
or a barcodes/patches row-count mismatch, must be rejected explicitly
rather than silently resolved via dict-overwrite."""
import anndata as ad
import numpy as np
import pandas as pd
import pytest

from gen3_multiscale.data.loaders import align_patches_to_adata


def _adata(barcodes: list[str]) -> "ad.AnnData":
    n = len(barcodes)
    return ad.AnnData(
        X=np.zeros((n, 2), dtype=np.float32),
        obs=pd.DataFrame(index=pd.Index(barcodes)),
        var=pd.DataFrame(index=pd.Index(["G0", "G1"])),
    )


def _patches(n: int, value_fn=None) -> np.ndarray:
    if value_fn is None:
        return np.zeros((n, 2, 2, 3), dtype=np.uint8)
    return np.stack([np.full((2, 2, 3), value_fn(i), dtype=np.uint8) for i in range(n)])


def test_align_patches_to_adata_rejects_duplicate_barcodes():
    adata = _adata(["S0-1", "S1-1", "S2-1"])
    barcodes = np.array(["S0-1", "S1-1", "S1-1"])  # S1-1 duplicated
    patches = _patches(3, value_fn=lambda i: i * 10)

    with pytest.raises(ValueError, match="duplicate"):
        align_patches_to_adata(adata, patches, barcodes)


def test_align_patches_to_adata_rejects_a_barcodes_patches_row_count_mismatch():
    adata = _adata(["S0-1", "S1-1"])
    barcodes = np.array(["S0-1", "S1-1"])
    patches = _patches(3)  # 3 rows of pixels but only 2 barcodes

    with pytest.raises(ValueError, match="patches.shape\\[0\\]"):
        align_patches_to_adata(adata, patches, barcodes)


def test_align_patches_to_adata_does_not_misattribute_a_duplicated_barcodes_patch():
    """Positive-control companion to the duplicate-rejection test above:
    confirms the bug this guards against would otherwise be real -- with
    the old dict-comprehension lookup, the LAST occurrence of a
    duplicated barcode would silently win, misattributing a real spot's
    patch. Once duplicates are rejected outright, this scenario can no
    longer reach that silent-misattribution path at all."""
    adata = _adata(["S0-1", "S1-1"])
    barcodes = np.array(["S0-1", "S1-1", "S1-1"])
    patches = _patches(3, value_fn=lambda i: i * 10)

    with pytest.raises(ValueError, match="duplicate"):
        align_patches_to_adata(adata, patches, barcodes)


def test_align_patches_to_adata_accepts_well_formed_unique_barcodes():
    adata = _adata(["S0-1", "S1-1", "S2-1"])
    barcodes = np.array(["S2-1", "S0-1", "S1-1"])  # deliberately out of adata's order
    patches = _patches(3, value_fn=lambda i: i * 10)

    _, aligned_patches, image_source_available = align_patches_to_adata(adata, patches, barcodes)

    assert image_source_available.all()
    # S0-1 -> barcodes index 1 -> value 10; S1-1 -> index 2 -> value 20; S2-1 -> index 0 -> value 0
    assert aligned_patches[0, 0, 0, 0] == 10
    assert aligned_patches[1, 0, 0, 0] == 20
    assert aligned_patches[2, 0, 0, 0] == 0
