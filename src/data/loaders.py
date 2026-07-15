"""
Data loading built around AnnData / SpatialData, the standard ST data model.

An AnnData object `adata` is expected to have, at minimum:
    adata.X                    -> [n_cells/spots, n_genes] expression matrix
    adata.obsm['spatial']      -> [n_cells/spots, 2] in-plane coordinates
    adata.obs['slice_id']      -> which physical section a point came from
    adata.obs['z']             -> z-axis / depth coordinate for that slice
                                   (fill in from known section spacing if not
                                   provided by the source dataset)
    adata.obs['cell_type']     -> optional, for cell-type-aware models
"""
from __future__ import annotations
from pathlib import Path

import anndata as ad
import numpy as np


def load_multi_slice(paths: list[str | Path], z_positions: list[float] | None = None
                      ) -> ad.AnnData:
    """
    Load and concatenate several single-slice files (h5ad/Visium/etc.) into
    one AnnData with a 'z' column, so downstream code always sees one 3D
    point cloud regardless of the source format.

    z_positions: physical depth of each slice (e.g. mm from a reference
    section). If None, uses integer slice index as a placeholder — REPLACE
    with real spacing before doing anything quantitative with the z-axis.
    """
    adatas = []
    for i, p in enumerate(paths):
        a = ad.read_h5ad(p) if str(p).endswith(".h5ad") else ad.read(p)
        z = z_positions[i] if z_positions is not None else float(i)
        a.obs["slice_id"] = Path(p).stem
        a.obs["z"] = z
        adatas.append(a)
    combined = ad.concat(adatas, join="outer", label="slice_id_batch")
    return combined


def basic_qc_and_normalize(adata: ad.AnnData, min_genes: int = 200,
                            min_cells: int = 3) -> ad.AnnData:
    """Standard scanpy-style QC + normalization. Adjust thresholds per dataset."""
    import scanpy as sc
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def get_coords_3d(adata: ad.AnnData) -> np.ndarray:
    """[N, 3] array of (x, y, z) for every point — the shared spatial index
    that both the inter-slice and intra-slice tasks are built on."""
    xy = adata.obsm["spatial"]
    z = adata.obs["z"].to_numpy()[:, None]
    return np.concatenate([xy, z], axis=1)


def load_hest_patches(hest_data_dir: str | Path, sample_id: str
                       ) -> tuple[np.ndarray, np.ndarray]:
    """
    H&E patches for one HEST-1k sample (task #17, docs/architecture_plan.md
    "Known gaps" — the H&E branch). HEST-1k pre-extracts per-spot patches
    into patches/{sample_id}.h5. Real h5py keys, confirmed 2026-07-15 by
    inspecting an actual downloaded file (corrects the earlier
    2026-07-14 note, which was based on HESTData.dump_patches() source
    reading alone and got two details wrong): 'img' [N,224,224,3] uint8
    (224x224, NOT 256x256), 'coords' [N,2], 'barcode' [N,1] object
    (singular key name, NOT 'barcodes', and 2D not 1D). No raw
    WSI/openslide handling needed.

    Same download command as load_hest_sample already pulls this file
    (docs/dataset_notes.md's `allow_patterns=["*INT1[_.]**"]` matches
    "patches/INT1.h5" the same way it matches "st/INT1.h5ad") — if this
    raises FileNotFoundError, re-run that download command rather than
    assuming a separate one is needed.

    Returns (patches [N,224,224,3] uint8, barcodes [N] str) in whatever
    order the .h5 file stores them — NOT necessarily aligned to any
    AnnData's obs order. Use align_patches_to_adata() for that.
    """
    import h5py
    hest_data_dir = Path(hest_data_dir)
    matches = [m for m in hest_data_dir.rglob(f"*{sample_id}*.h5") if "patches" in m.parts]
    if not matches:
        raise FileNotFoundError(
            f"No patches/{sample_id}.h5 file found under {hest_data_dir}. "
            "Re-run the HEST-1k download command in docs/dataset_notes.md — "
            "the same allow_patterns already covers this file."
        )
    with h5py.File(matches[0], "r") as f:
        patches = f["img"][:]
        raw_barcodes = f["barcode"][:, 0]  # [N, 1] object array -> [N]
        barcodes = np.array([b.decode() if isinstance(b, bytes) else b for b in raw_barcodes])
    return patches, barcodes


def align_patches_to_adata(adata: ad.AnnData, patches: np.ndarray, barcodes: np.ndarray
                            ) -> tuple[ad.AnnData, np.ndarray]:
    """Subset adata to only the spots with a matching H&E patch, and
    return (filtered_adata, aligned_patches) in that same (filtered)
    order. HEST-1k's own patch extraction naturally drops some spots
    (tissue-mask/WSI-border edge cases) — confirmed 2026-07-15 on real
    INT1 data: 49/1080 spots had no matching patch, a normal ~4.5% gap in
    HEST-1k's own pipeline, not a data-mismatch bug (an earlier version of
    this function raised on ANY gap, which was too strict for real data).
    Only raises if NONE of the spots match at all — that would indicate a
    genuine version/sample mismatch, not normal partial coverage."""
    barcode_to_idx = {b: i for i, b in enumerate(barcodes)}
    has_patch = np.array([name in barcode_to_idx for name in adata.obs_names])
    if not has_patch.any():
        raise ValueError(
            "None of the adata spots matched any H&E patch barcode — patches "
            "and expression data are likely from different downloads/versions "
            "of this sample."
        )
    n_dropped = int((~has_patch).sum())
    if n_dropped:
        print(f"align_patches_to_adata: dropping {n_dropped}/{adata.n_obs} spots "
              f"with no matching H&E patch (HEST-1k's own patch extraction misses "
              f"some spots at tissue/WSI edges — this is normal)")
    filtered_adata = adata[has_patch].copy()
    order = [barcode_to_idx[name] for name in filtered_adata.obs_names]
    return filtered_adata, patches[order]


def load_hest_sample(hest_data_dir: str | Path, sample_id: str) -> ad.AnnData:
    """
    Load one HEST-1k sample (docs/dataset_notes.md Track A primary pick) —
    a single 2D section, not part of a serial z-series, so `z` is a constant
    placeholder. Track A only; Track B needs load_multi_slice with real
    z-spacing instead.

    Expects hest_data_dir already populated via HEST-1k's own download_hest
    (huggingface_hub-based, needs a free HF account + auth token — see
    docs/dataset_notes.md). Searches for the sample's .h5ad file by pattern
    rather than assuming an exact folder nesting depth, since that layout
    wasn't independently confirmed byte-for-byte from documentation alone.
    """
    hest_data_dir = Path(hest_data_dir)
    matches = list(hest_data_dir.rglob(f"*{sample_id}*.h5ad"))
    if not matches:
        raise FileNotFoundError(
            f"No .h5ad file found for sample_id={sample_id!r} under {hest_data_dir}. "
            "Download it first via HEST-1k's download_hest (docs/dataset_notes.md)."
        )
    adata = ad.read_h5ad(matches[0])
    # HEST-1k already uses the standard scanpy spatial convention
    # (adata.obsm['spatial']), so no coordinate remapping needed here.
    adata.obs["slice_id"] = sample_id
    adata.obs["z"] = 0.0
    return adata
