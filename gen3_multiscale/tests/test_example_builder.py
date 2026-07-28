"""Tests for the real Gen3 example builder (Step 2 of the real data
builder/trainer, CONTRACT.md section 30)."""
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest

from gen3_multiscale.data.dataset_manifest import build_dataset_manifest
from gen3_multiscale.data.example_builder import (
    build_spatial_field_example, load_sample_for_examples,
)

_N_FEATURES = 8


def _stub_image_feature_fn(patches: np.ndarray) -> np.ndarray:
    """Cheap, deterministic per-patch feature: the patch's own mean pixel
    value repeated across a small feature width -- no real GigaPath
    checkpoint needed for these tests, matching every other pluggable-
    feature-function pattern in this codebase."""
    means = patches.reshape(patches.shape[0], -1).astype(np.float32).mean(axis=1)
    return np.tile(means[:, None], (1, _N_FEATURES))


def _square_grid_adata(n_side: int = 6, spacing: float = 10.0, n_genes: int = 5, seed: int = 0) -> "ad.AnnData":
    rng = np.random.default_rng(seed)
    coords = np.array([[x * spacing, y * spacing] for x in range(n_side) for y in range(n_side)], dtype=np.float64)
    n = coords.shape[0]
    barcodes = [f"SPOT{i}-1" for i in range(n)]
    gene_names = [f"GENE{i}" for i in range(n_genes)]
    counts = rng.poisson(5, size=(n, n_genes)).astype(np.float32)
    adata = ad.AnnData(
        X=counts, obs=pd.DataFrame(index=pd.Index(barcodes)), var=pd.DataFrame(index=pd.Index(gene_names)),
    )
    adata.obsm["spatial"] = coords
    return adata


def _matching_patches(adata: "ad.AnnData", patch_value_fn=None) -> np.ndarray:
    n = adata.n_obs
    if patch_value_fn is None:
        return np.zeros((n, 4, 4, 3), dtype=np.uint8)
    return np.stack([np.full((4, 4, 3), patch_value_fn(i), dtype=np.uint8) for i in range(n)])


def test_build_spatial_field_example_basic_shapes_and_disjointness():
    adata = _square_grid_adata()
    patches = _matching_patches(adata)
    barcodes = list(adata.obs_names)
    query = barcodes[14:16]  # two central-ish spots
    context = [b for b in barcodes if b not in query]

    inputs, targets = build_spatial_field_example(
        adata, patches, context, query, _stub_image_feature_fn,
        sample_id="S0", patient_id="P0", patch_size_fullres=1.0,  # tiny -- this test isn't about overlap exclusion
        require_full_sample_coords=False,
    )
    assert set(inputs.observed_barcodes.tolist()).isdisjoint(set(inputs.query_barcodes.tolist()))
    assert inputs.observed_full_gene_expression.shape[0] == inputs.observed_coords.shape[0]
    assert inputs.observed_gigapath_features.shape == (inputs.observed_coords.shape[0], _N_FEATURES)
    assert targets.query_expression.shape[0] == len(query)
    assert inputs.sample_id == "S0" and inputs.patient_id == "P0"


def test_build_spatial_field_example_excludes_context_patches_overlapping_the_hole():
    """The core physical-safety requirement this module exists for: a
    context spot whose barcode is NOT a query barcode, but whose patch
    footprint physically overlaps a query spot's footprint, must be
    excluded from the observed set entirely."""
    adata = _square_grid_adata(n_side=6, spacing=10.0)
    patches = _matching_patches(adata)
    barcodes = list(adata.obs_names)
    coords = adata.obsm["spatial"]

    # Pick one query spot; find a DIFFERENT spot whose coordinate is very
    # close to it (adjacent on the grid, spacing=10 apart -- well within
    # a large patch_size_fullres) but not itself a query barcode.
    query_idx = 14
    query_barcode = barcodes[query_idx]
    query_xy = coords[query_idx]
    distances = np.linalg.norm(coords - query_xy, axis=1)
    distances[query_idx] = np.inf
    neighbor_idx = int(np.argmin(distances))
    neighbor_barcode = barcodes[neighbor_idx]

    context = [b for b in barcodes if b not in (query_barcode,)]
    inputs, _ = build_spatial_field_example(
        adata, patches, context, [query_barcode], _stub_image_feature_fn,
        sample_id="S0", patient_id="P0", patch_size_fullres=30.0,  # large enough to guarantee overlap with the 10-unit-spaced neighbor
        require_full_sample_coords=False,
    )
    assert neighbor_barcode not in inputs.observed_barcodes.tolist()
    assert inputs.provenance["n_context_excluded_for_physical_he_overlap"] >= 1

    # With a tiny patch size, the same neighbor must NOT be excluded.
    inputs_small_patch, _ = build_spatial_field_example(
        adata, patches, context, [query_barcode], _stub_image_feature_fn,
        sample_id="S0", patient_id="P0", patch_size_fullres=0.01,
        require_full_sample_coords=False,
    )
    assert neighbor_barcode in inputs_small_patch.observed_barcodes.tolist()
    assert inputs_small_patch.provenance["n_context_excluded_for_physical_he_overlap"] == 0


def test_build_spatial_field_example_rejects_missing_barcodes():
    adata = _square_grid_adata()
    patches = _matching_patches(adata)
    with pytest.raises(ValueError, match="absent from the aligned sample data"):
        build_spatial_field_example(
            adata, patches, ["NOT-A-REAL-BARCODE"], [list(adata.obs_names)[0]], _stub_image_feature_fn,
            sample_id="S0", patient_id="P0", require_full_sample_coords=False,
        )


def test_build_spatial_field_example_rejects_context_query_overlap():
    adata = _square_grid_adata()
    patches = _matching_patches(adata)
    shared = list(adata.obs_names)[:5]
    with pytest.raises(ValueError, match="overlap"):
        build_spatial_field_example(
            adata, patches, shared, shared[:1], _stub_image_feature_fn,
            sample_id="S0", patient_id="P0", require_full_sample_coords=False,
        )


def test_build_spatial_field_example_requires_full_sample_coords_by_default():
    """10th Codex re-audit of commit 9592d9e, finding #5 (confirmed):
    the real trainer must always supply the complete aligned sample
    lattice for coordinate-scaling normalization, not silently fall
    back to a mask-dependent subset. Regression: omitting
    full_sample_coords now raises by default; the escape hatch is only
    available via an explicit require_full_sample_coords=False."""
    adata = _square_grid_adata()
    patches = _matching_patches(adata)
    barcodes = list(adata.obs_names)
    query = barcodes[:1]
    context = barcodes[1:]
    with pytest.raises(ValueError, match="full_sample_coords is required"):
        build_spatial_field_example(
            adata, patches, context, query, _stub_image_feature_fn,
            sample_id="S0", patient_id="P0", patch_size_fullres=1.0,
        )
    # Explicit opt-out still works (the test-only escape hatch).
    build_spatial_field_example(
        adata, patches, context, query, _stub_image_feature_fn,
        sample_id="S0", patient_id="P0", patch_size_fullres=1.0, require_full_sample_coords=False,
    )


def test_build_spatial_field_example_rejects_non_2d_image_features():
    adata = _square_grid_adata()
    patches = _matching_patches(adata)
    barcodes = list(adata.obs_names)
    query = barcodes[:1]
    context = barcodes[1:]
    with pytest.raises(ValueError, match="2D"):
        build_spatial_field_example(
            adata, patches, context, query, lambda p: np.zeros(len(p)),  # 1D, not [N, feature_dim]
            sample_id="S0", patient_id="P0", patch_size_fullres=1.0, require_full_sample_coords=False,
        )


def test_build_spatial_field_example_rejects_non_finite_image_features():
    adata = _square_grid_adata()
    patches = _matching_patches(adata)
    barcodes = list(adata.obs_names)
    query = barcodes[:1]
    context = barcodes[1:]

    def _nan_feature_fn(p: np.ndarray) -> np.ndarray:
        out = _stub_image_feature_fn(p)
        out[0, 0] = np.nan
        return out

    with pytest.raises(ValueError, match="non-finite"):
        build_spatial_field_example(
            adata, patches, context, query, _nan_feature_fn,
            sample_id="S0", patient_id="P0", patch_size_fullres=1.0, require_full_sample_coords=False,
        )


def test_build_spatial_field_example_rejects_an_unexpected_feature_width():
    adata = _square_grid_adata()
    patches = _matching_patches(adata)
    barcodes = list(adata.obs_names)
    query = barcodes[:1]
    context = barcodes[1:]
    with pytest.raises(ValueError, match="feature width"):
        build_spatial_field_example(
            adata, patches, context, query, _stub_image_feature_fn,
            sample_id="S0", patient_id="P0", patch_size_fullres=1.0, require_full_sample_coords=False,
            expected_feature_width=_N_FEATURES + 1,
        )


def test_build_spatial_field_example_rejects_a_patches_adata_row_count_mismatch():
    adata = _square_grid_adata()
    patches = _matching_patches(adata)[:-1]  # one row short
    barcodes = list(adata.obs_names)
    query = barcodes[:1]
    context = barcodes[1:]
    with pytest.raises(ValueError, match="patches has .* rows but adata has"):
        build_spatial_field_example(
            adata, patches, context, query, _stub_image_feature_fn,
            sample_id="S0", patient_id="P0", patch_size_fullres=1.0, require_full_sample_coords=False,
        )


def test_build_spatial_field_example_rejects_duplicate_spot_coordinates():
    adata = _square_grid_adata()
    patches = _matching_patches(adata)
    adata.obsm["spatial"][1] = adata.obsm["spatial"][0]  # corrupt: two spots at the identical location
    barcodes = list(adata.obs_names)
    query = barcodes[:1]
    context = barcodes[1:]
    with pytest.raises(ValueError, match="duplicate spot coordinates"):
        build_spatial_field_example(
            adata, patches, context, query, _stub_image_feature_fn,
            sample_id="S0", patient_id="P0", patch_size_fullres=1.0, require_full_sample_coords=False,
        )


def test_build_spatial_field_example_rejects_a_mismatched_feature_function():
    adata = _square_grid_adata()
    patches = _matching_patches(adata)
    barcodes = list(adata.obs_names)
    query = barcodes[:1]
    context = barcodes[1:]
    with pytest.raises(ValueError, match="image_feature_fn returned"):
        build_spatial_field_example(
            adata, patches, context, query, lambda p: np.zeros((1, _N_FEATURES)),  # wrong row count
            sample_id="S0", patient_id="P0", patch_size_fullres=1.0, require_full_sample_coords=False,
        )


def test_build_spatial_field_example_normalizes_coordinates_not_raw_pixels():
    """Coordinates must be centered and spot-spacing-scaled, not raw
    physical pixel/micron values -- SpatialFieldInputs' own documented
    contract, and an explicit, repeated audit recommendation."""
    adata = _square_grid_adata(n_side=6, spacing=37.5)  # a real, non-trivial physical spacing
    patches = _matching_patches(adata)
    barcodes = list(adata.obs_names)
    query = barcodes[14:16]
    context = [b for b in barcodes if b not in query]

    inputs, _ = build_spatial_field_example(
        adata, patches, context, query, _stub_image_feature_fn,
        sample_id="S0", patient_id="P0", patch_size_fullres=1.0, require_full_sample_coords=False,
    )
    combined = np.concatenate([inputs.observed_coords, inputs.query_coords], axis=0)
    assert np.allclose(combined.mean(axis=0), 0.0, atol=1e-4)  # centered
    # Scaled to spot-spacing units: nearest-neighbor spacing in the
    # RAW data was 37.5; after scaling it should be close to 1.0, not 37.5.
    from scipy.spatial import cKDTree
    tree = cKDTree(inputs.observed_coords)
    dist, _ = tree.query(inputs.observed_coords, k=2)
    assert 0.5 < np.median(dist[:, 1]) < 2.0


def _make_synthetic_hest1k_with_patches(
    tmp_path: Path, organ_sample_ids: dict[str, list[str]], n_spots_per_sample: int = 6,
    gene_names: list[str] | None = None, seed: int = 0,
) -> tuple[Path, Path, list[str]]:
    gene_names = gene_names or [f"GENE{i}" for i in range(8)]
    rng = np.random.default_rng(seed)
    hest_dir = tmp_path / "hest1k"
    (hest_dir / "st").mkdir(parents=True, exist_ok=True)
    (hest_dir / "patches").mkdir(parents=True, exist_ok=True)
    rows = []
    for organ, ids in organ_sample_ids.items():
        for sid in ids:
            barcodes = [f"{sid}-SPOT{i}-1" for i in range(n_spots_per_sample)]
            counts = rng.poisson(5, size=(len(barcodes), len(gene_names))).astype(np.float32)
            coords = np.array([[x * 10.0, 0.0] for x in range(len(barcodes))], dtype=np.float64)
            adata = ad.AnnData(
                X=counts, obs=pd.DataFrame(index=pd.Index(barcodes)), var=pd.DataFrame(index=pd.Index(gene_names)),
            )
            adata.obsm["spatial"] = coords
            adata.write_h5ad(hest_dir / "st" / f"{sid}.h5ad")

            with h5py.File(hest_dir / "patches" / f"{sid}.h5", "w") as f:
                f.create_dataset("img", data=np.zeros((len(barcodes), 4, 4, 3), dtype=np.uint8))
                f.create_dataset("barcode", data=np.array([[b.encode()] for b in barcodes]))

            rows.append({
                "id": sid, "organ": organ, "st_technology": "Visium",
                "species": "Homo sapiens", "nb_genes": len(gene_names),
            })
    meta_path = tmp_path / "meta.csv"
    pd.DataFrame(rows).to_csv(meta_path, index=False)
    return hest_dir, meta_path, gene_names


def test_load_sample_for_examples_matches_the_manifests_declared_gene_panel_and_barcodes(tmp_path):
    hest_dir, meta_path, gene_names = _make_synthetic_hest1k_with_patches(
        tmp_path, {"Lung": [f"L{i}" for i in range(3)]},
    )
    manifest = build_dataset_manifest(
        hest_dir, str(meta_path), organs="all", min_samples_per_organ=3,
        n_validation_per_organ=0, n_test_per_organ=0, split_seed=0,
        check_gene_panel_compatibility=False, min_nb_genes=None,
        gene_min_genes_per_spot=1, gene_min_cells=0,
    )
    adata, patches = load_sample_for_examples(manifest, "L0")
    assert list(adata.var_names) == manifest["gene_panel"]
    assert set(adata.obs_names).issubset(set(manifest["samples"]["L0"]["barcodes"]))
    assert patches.shape[0] == adata.n_obs


def test_end_to_end_manifest_to_example(tmp_path):
    """The real pipeline: manifest -> load_sample_for_examples ->
    build_spatial_field_example, on real (synthetic) data end to end."""
    hest_dir, meta_path, gene_names = _make_synthetic_hest1k_with_patches(
        tmp_path, {"Lung": [f"L{i}" for i in range(3)]}, n_spots_per_sample=8,
    )
    manifest = build_dataset_manifest(
        hest_dir, str(meta_path), organs="all", min_samples_per_organ=3,
        n_validation_per_organ=0, n_test_per_organ=0, split_seed=0,
        check_gene_panel_compatibility=False, min_nb_genes=None,
        gene_min_genes_per_spot=1, gene_min_cells=0,
    )
    adata, patches = load_sample_for_examples(manifest, "L0")
    barcodes = list(adata.obs_names)
    query = barcodes[3:5]
    context = [b for b in barcodes if b not in query]

    inputs, targets = build_spatial_field_example(
        adata, patches, context, query, _stub_image_feature_fn,
        sample_id="L0", patient_id=manifest["samples"]["L0"]["patient_id"], patch_size_fullres=1.0,
        full_sample_coords=adata.obsm["spatial"],  # the recommended real usage, not the test-only escape hatch
    )
    assert set(inputs.query_barcodes.tolist()) == set(query)
    assert set(inputs.observed_barcodes.tolist()).issubset(set(context))
