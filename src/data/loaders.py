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
