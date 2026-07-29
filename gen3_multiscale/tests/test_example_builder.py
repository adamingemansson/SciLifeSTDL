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


def test_build_spatial_field_example_requires_exactly_one_of_image_feature_fn_or_precomputed_spot_features():
    """21st Codex re-audit, Step 5/6 boundary #2: precomputed_spot_features
    must be mutually exclusive with image_feature_fn -- never both, never
    neither."""
    adata = _square_grid_adata()
    patches = _matching_patches(adata)
    barcodes = list(adata.obs_names)
    query = barcodes[14:16]
    context = [b for b in barcodes if b not in query]

    with pytest.raises(ValueError, match="exactly one"):
        build_spatial_field_example(
            adata, patches, context, query, None,
            sample_id="S0", patient_id="P0", require_full_sample_coords=False,
        )
    with pytest.raises(ValueError, match="exactly one"):
        build_spatial_field_example(
            adata, patches, context, query, _stub_image_feature_fn,
            sample_id="S0", patient_id="P0", require_full_sample_coords=False,
            precomputed_spot_features=np.zeros((adata.n_obs, _N_FEATURES), dtype=np.float32),
        )


def test_build_spatial_field_example_precomputed_spot_features_matches_image_feature_fn_output():
    """The production path (precomputed_spot_features) must select
    exactly the same rows image_feature_fn(patches[...]) would have
    computed from scratch -- a real correctness check, not just a shape
    check."""
    adata = _square_grid_adata()
    patches = _matching_patches(adata)
    barcodes = list(adata.obs_names)
    query = barcodes[14:16]
    context = [b for b in barcodes if b not in query]
    kwargs = dict(sample_id="S0", patient_id="P0", patch_size_fullres=1.0, require_full_sample_coords=False)

    inputs_fn, _ = build_spatial_field_example(adata, patches, context, query, _stub_image_feature_fn, **kwargs)
    precomputed = _stub_image_feature_fn(patches)  # the SAME per-spot features, precomputed for every spot
    inputs_precomputed, _ = build_spatial_field_example(
        adata, patches, context, query, None, precomputed_spot_features=precomputed, **kwargs,
    )
    np.testing.assert_array_equal(inputs_fn.observed_gigapath_features, inputs_precomputed.observed_gigapath_features)
    np.testing.assert_array_equal(inputs_fn.observed_image_available, inputs_precomputed.observed_image_available)


def test_build_spatial_field_example_precomputed_spot_features_ignores_query_row_changes():
    """21st Codex re-audit: 'Add adversarial tests proving that changing
    cached query rows... cannot change the constructed model input.'
    Query spots never appear in context_pos, so precomputed_spot_features
    rows at query positions must have ZERO effect on the built example --
    a real leakage-guard test, not just a documentation claim."""
    adata = _square_grid_adata()
    patches = _matching_patches(adata)
    barcodes = list(adata.obs_names)
    query = barcodes[14:16]
    context = [b for b in barcodes if b not in query]
    query_pos = [barcodes.index(b) for b in query]
    kwargs = dict(sample_id="S0", patient_id="P0", patch_size_fullres=1.0, require_full_sample_coords=False)

    precomputed_a = np.arange(adata.n_obs * _N_FEATURES, dtype=np.float32).reshape(adata.n_obs, _N_FEATURES)
    precomputed_b = precomputed_a.copy()
    precomputed_b[query_pos] = -999.0  # corrupt ONLY the query rows

    inputs_a, _ = build_spatial_field_example(
        adata, patches, context, query, None, precomputed_spot_features=precomputed_a, **kwargs,
    )
    inputs_b, _ = build_spatial_field_example(
        adata, patches, context, query, None, precomputed_spot_features=precomputed_b, **kwargs,
    )
    np.testing.assert_array_equal(inputs_a.observed_gigapath_features, inputs_b.observed_gigapath_features)


def test_build_spatial_field_example_precomputed_spot_features_ignores_physically_hidden_rows():
    """21st Codex re-audit: 'Add adversarial tests proving that changing
    ...rows hidden by physical overlap—cannot change the constructed
    model input.' A context spot whose H&E physically overlaps the query
    hole must be zeroed in observed_gigapath_features regardless of what
    value precomputed_spot_features actually holds for that row -- the
    same "never leak an unavailable feature" guarantee image_feature_fn
    already has, now proven for the precomputed path too."""
    adata = _square_grid_adata(n_side=6, spacing=10.0)
    patches = _matching_patches(adata)
    barcodes = list(adata.obs_names)
    coords = adata.obsm["spatial"]

    query_idx = 14
    query_barcode = barcodes[query_idx]
    query_xy = coords[query_idx]
    distances = np.linalg.norm(coords - query_xy, axis=1)
    distances[query_idx] = np.inf
    neighbor_idx = int(np.argmin(distances))  # close enough to physically overlap -> unavailable
    neighbor_barcode = barcodes[neighbor_idx]

    context = [b for b in barcodes if b != query_barcode]
    kwargs = dict(
        sample_id="S0", patient_id="P0", patch_size_fullres=30.0, require_full_sample_coords=False,
    )
    precomputed_a = np.zeros((adata.n_obs, _N_FEATURES), dtype=np.float32)
    precomputed_b = precomputed_a.copy()
    precomputed_b[neighbor_idx] = 12345.0  # a distinctive, real-looking value at the physically hidden row

    inputs_a, _ = build_spatial_field_example(
        adata, patches, context, [query_barcode], None, precomputed_spot_features=precomputed_a, **kwargs,
    )
    inputs_b, _ = build_spatial_field_example(
        adata, patches, context, [query_barcode], None, precomputed_spot_features=precomputed_b, **kwargs,
    )
    np.testing.assert_array_equal(inputs_a.observed_gigapath_features, inputs_b.observed_gigapath_features)
    neighbor_pos = inputs_b.observed_barcodes.tolist().index(neighbor_barcode)
    assert inputs_b.observed_image_available[neighbor_pos] == False  # noqa: E712 -- real numpy bool
    assert np.all(inputs_b.observed_gigapath_features[neighbor_pos] == 0.0)


def test_build_spatial_field_example_flags_but_retains_context_patches_overlapping_the_hole():
    """15th Codex re-audit (Step 5 acceptance criteria): a context spot
    whose barcode is NOT a query barcode, but whose patch footprint
    physically overlaps a query spot's footprint, must be RETAINED in
    the observed set (its GEX is real and available) but explicitly
    flagged observed_image_available=False, with its
    observed_gigapath_features row zeroed -- never dropped entirely, and
    never a real/garbage feature with no flag."""
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
    assert neighbor_barcode in inputs.observed_barcodes.tolist()  # retained, not dropped
    neighbor_pos = inputs.observed_barcodes.tolist().index(neighbor_barcode)
    assert inputs.observed_image_available[neighbor_pos] == False  # noqa: E712 -- real numpy bool, not `is False`
    assert np.all(inputs.observed_gigapath_features[neighbor_pos] == 0.0)
    assert inputs.provenance["n_context_image_unavailable_for_physical_he_overlap"] >= 1

    # With a tiny patch size, the same neighbor must be marked available.
    inputs_small_patch, _ = build_spatial_field_example(
        adata, patches, context, [query_barcode], _stub_image_feature_fn,
        sample_id="S0", patient_id="P0", patch_size_fullres=0.01,
        require_full_sample_coords=False,
    )
    assert neighbor_barcode in inputs_small_patch.observed_barcodes.tolist()
    small_pos = inputs_small_patch.observed_barcodes.tolist().index(neighbor_barcode)
    assert inputs_small_patch.observed_image_available[small_pos] == True  # noqa: E712
    assert inputs_small_patch.provenance["n_context_image_unavailable_for_physical_he_overlap"] == 0


def test_build_spatial_field_example_all_he_unavailable_requires_expected_feature_width():
    """15th Codex re-audit (Step 5 acceptance criteria): when EVERY
    context spot's H&E overlaps the hole, image_feature_fn is never
    called at all (nothing available to feed it) -- expected_feature_width
    must be supplied so a correctly-shaped all-zero
    observed_gigapath_features can still be built; omitting it must fail
    loudly rather than guess a width."""
    adata = _square_grid_adata(n_side=3, spacing=10.0)
    patches = _matching_patches(adata)
    barcodes = list(adata.obs_names)
    query = [barcodes[4]]  # center spot
    context = [b for b in barcodes if b != barcodes[4]]

    with pytest.raises(ValueError, match="expected_feature_width"):
        build_spatial_field_example(
            adata, patches, context, query, _stub_image_feature_fn,
            sample_id="S0", patient_id="P0", patch_size_fullres=1000.0,  # huge -- overlaps every context spot
            require_full_sample_coords=False,
        )

    inputs, _ = build_spatial_field_example(
        adata, patches, context, query, _stub_image_feature_fn,
        sample_id="S0", patient_id="P0", patch_size_fullres=1000.0,
        require_full_sample_coords=False, expected_feature_width=_N_FEATURES,
    )
    assert not inputs.observed_image_available.any()
    assert inputs.observed_gigapath_features.shape == (len(context), _N_FEATURES)
    assert np.all(inputs.observed_gigapath_features == 0.0)


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


def test_build_spatial_field_example_rejects_a_malformed_full_sample_coords():
    """11th Codex re-audit of commit 9dab8fe, finding #3 ("validate
    full_sample_coords itself -- shape, finiteness, uniqueness, row
    count and agreement with the aligned sample"): a caller-supplied
    full_sample_coords must be held to the same fail-closed standard as
    adata.obsm['spatial'] itself."""
    adata = _square_grid_adata()
    patches = _matching_patches(adata)
    barcodes = list(adata.obs_names)
    query = barcodes[:1]
    context = barcodes[1:]
    real_coords = adata.obsm["spatial"]

    with pytest.raises(ValueError, match=r"full_sample_coords must be \[M, 2\]"):
        build_spatial_field_example(
            adata, patches, context, query, _stub_image_feature_fn,
            sample_id="S0", patient_id="P0", patch_size_fullres=1.0,
            full_sample_coords=real_coords[:, :1],  # wrong shape, [N, 1]
        )

    non_finite = real_coords.copy()
    non_finite[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        build_spatial_field_example(
            adata, patches, context, query, _stub_image_feature_fn,
            sample_id="S0", patient_id="P0", patch_size_fullres=1.0,
            full_sample_coords=non_finite,
        )

    duplicated = real_coords.copy()
    duplicated[1] = duplicated[0]
    with pytest.raises(ValueError, match="duplicate rows"):
        build_spatial_field_example(
            adata, patches, context, query, _stub_image_feature_fn,
            sample_id="S0", patient_id="P0", patch_size_fullres=1.0,
            full_sample_coords=duplicated,
        )

    too_few_rows = real_coords[:-5]
    with pytest.raises(ValueError, match="fewer than"):
        build_spatial_field_example(
            adata, patches, context, query, _stub_image_feature_fn,
            sample_id="S0", patient_id="P0", patch_size_fullres=1.0,
            full_sample_coords=too_few_rows,
        )

    mismatched = real_coords.copy()
    mismatched[0] = [-999.0, -999.0]  # replaces a real row with an unrelated value -- no longer agrees
    with pytest.raises(ValueError, match="does not agree with the aligned sample"):
        build_spatial_field_example(
            adata, patches, context, query, _stub_image_feature_fn,
            sample_id="S0", patient_id="P0", patch_size_fullres=1.0,
            full_sample_coords=mismatched,
        )


def test_build_spatial_field_example_accepts_a_genuinely_larger_full_sample_coords_superset():
    """The recommended real usage: full_sample_coords is the sample's
    COMPLETE lattice, which can legitimately have MORE rows than the
    aligned adata (e.g. spots dropped for missing H&E patches) as long
    as every one of the aligned sample's own coordinates is present."""
    adata = _square_grid_adata()
    patches = _matching_patches(adata)
    barcodes = list(adata.obs_names)
    query = barcodes[:1]
    context = barcodes[1:]
    real_coords = adata.obsm["spatial"]
    superset = np.concatenate([real_coords, np.array([[1000.0, 1000.0]])], axis=0)

    inputs, _ = build_spatial_field_example(
        adata, patches, context, query, _stub_image_feature_fn,
        sample_id="S0", patient_id="P0", patch_size_fullres=1.0, full_sample_coords=superset,
    )
    assert inputs.observed_coords.shape[0] == len(context)


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


def test_build_spatial_field_example_rejects_a_malformed_precomputed_spot_features():
    adata = _square_grid_adata()
    patches = _matching_patches(adata)
    barcodes = list(adata.obs_names)
    query = barcodes[:1]
    context = barcodes[1:]
    kwargs = dict(sample_id="S0", patient_id="P0", patch_size_fullres=1.0, require_full_sample_coords=False)

    with pytest.raises(ValueError, match="precomputed_spot_features"):
        build_spatial_field_example(  # wrong row count
            adata, patches, context, query, None,
            precomputed_spot_features=np.zeros((adata.n_obs - 1, _N_FEATURES), dtype=np.float32), **kwargs,
        )
    with pytest.raises(ValueError, match="precomputed_spot_features"):
        build_spatial_field_example(  # 1D, not [N, feature_dim]
            adata, patches, context, query, None,
            precomputed_spot_features=np.zeros(adata.n_obs, dtype=np.float32), **kwargs,
        )
    with pytest.raises(ValueError, match="non-finite"):
        bad = np.zeros((adata.n_obs, _N_FEATURES), dtype=np.float32)
        bad[0, 0] = np.nan
        build_spatial_field_example(
            adata, patches, context, query, None, precomputed_spot_features=bad, **kwargs,
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


def _wsi_slide_context(
    level0_xys: list[tuple[float, float]], tile_size: float = 20.0,
    longnet_scale: float = 2.0, longnet_offset: tuple[float, float] = (5000.0, 5000.0),
) -> dict:
    """A minimal, hand-built slide_context.load_slide_context()-shaped
    dict with TWO GENUINELY DIFFERENT coordinate frames, matching a real
    dense_wsi_cache whose source MPP differs from GigaPath's target MPP:
    `mask_coords` (level-0/HEST-aligned, the SAME physical frame
    spot_coords/adata.obsm['spatial'] use) and `coords` (GigaPath
    LongNet's own target-MPP frame -- here a different scale AND a large
    coordinate-origin offset, simulating a real level-0-pixel-vs-target-
    MPP mismatch) -- so a test using this fixture would FAIL if
    example_builder.py ever mixed the two frames up again (17th Codex
    re-audit, Step 5 Part 2 launch blocker #1)."""
    level0 = np.asarray(level0_xys, dtype=np.float32)
    longnet = level0 * longnet_scale + np.asarray(longnet_offset, dtype=np.float32)
    features = np.stack([np.full(8, float(i + 1), dtype=np.float32) for i in range(len(level0_xys))])
    return {
        "features": features, "coords": longnet, "mask_coords": level0,
        "tile_size": tile_size, "mask_tile_size": tile_size, "coords_are_centers": True,
        "context_id": "wsi-unit-test", "source": "dense_wsi_cache",
    }


# Three level-0 tile centers; y-values differ so full_slide_coord_bounds is
# non-degenerate. The x=20 tile's footprint overlaps a query hole at
# (20, 20) with patch_size_fullres=15.0 (limit=17.5); x=0/x=40 do not
# (see test_slide_context.py's identical overlap-threshold arithmetic).
_WSI_LEVEL0_TILES = [(0.0, 10.0), (20.0, 20.0), (40.0, 30.0)]


def test_build_spatial_field_example_wires_wsi_context_with_separate_coordinate_frames():
    """17th Codex re-audit (Step 5 Part 2 launch blocker #1), CONFIRMED:
    wsi_tile_longnet_coords must stay GigaPath LongNet's own frame
    (visible_slide_context's `coords`, UNCHANGED -- never recentered/
    rescaled), while wsi_tile_regional_coords must be derived from the
    level-0/HEST-aligned frame (`level0_coords`) via the documented
    (level0 - reference) / scale transform -- the SAME physical frame
    observed_coords/query_coords already use. This fixture's `coords`
    and `mask_coords` are deliberately on different scales/origins (as a
    real non-0.5-MPP dense_wsi_cache would be), so the two must never be
    interchangeable in either direction."""
    adata = _square_grid_adata(n_side=6, spacing=10.0)
    patches = _matching_patches(adata)
    barcodes = list(adata.obs_names)
    query_barcode = barcodes[14]  # coords (20, 20) -- see _square_grid_adata's x*spacing,y*spacing layout
    context = [b for b in barcodes if b != query_barcode]
    slide_context = _wsi_slide_context(_WSI_LEVEL0_TILES)

    inputs, _ = build_spatial_field_example(
        adata, patches, context, [query_barcode], _stub_image_feature_fn,
        sample_id="S0", patient_id="P0", patch_size_fullres=15.0,
        full_sample_coords=adata.obsm["spatial"], slide_context=slide_context,
    )

    assert inputs.slide_cache_namespace and inputs.slide_cache_namespace.strip()
    assert inputs.wsi_tile_longnet_coords.shape == (2, 2)
    assert inputs.wsi_tile_regional_coords.shape == (2, 2)
    assert inputs.wsi_tile_features.shape == (2, 8)

    # reference/scale are now derived from the COMPLETE sample lattice
    # (17th Codex re-audit launch blocker #2) -- for this uniform 6x6,
    # spacing=10 grid, that's exactly (25, 25) and 10.0, deterministically,
    # regardless of which mask was realized.
    scale = inputs.provenance["spot_spacing_scale"]
    reference = np.asarray(inputs.provenance["coordinate_reference"])
    assert np.allclose(reference, [25.0, 25.0])
    assert np.isclose(scale, 10.0)

    # wsi_tile_longnet_coords is EXACTLY visible_slide_context's `coords`
    # (the LongNet frame) -- large-magnitude, UNTOUCHED by centering/scaling.
    expected_longnet = np.asarray([(0.0, 10.0), (40.0, 30.0)], dtype=np.float32) * 2.0 + np.asarray([5000.0, 5000.0])
    assert np.allclose(inputs.wsi_tile_longnet_coords, expected_longnet)

    # wsi_tile_regional_coords is derived from level0_coords -- NOT from
    # wsi_tile_longnet_coords. Reconstructing regional coords from the
    # (wrong) longnet frame would give a completely different, large-
    # magnitude result -- explicitly asserted absent here.
    expected_regional = (np.asarray([(0.0, 10.0), (40.0, 30.0)], dtype=np.float32) - reference) / scale
    assert np.allclose(inputs.wsi_tile_regional_coords, expected_regional, atol=1e-4)
    wrong_regional_from_longnet_frame = (inputs.wsi_tile_longnet_coords - reference) / scale
    assert not np.allclose(inputs.wsi_tile_regional_coords, wrong_regional_from_longnet_frame)

    # full_slide_coord_bounds comes from the COMPLETE (3-tile) level-0
    # tile-center set, in the same normalized frame -- the removed tile
    # (x=20) lies strictly inside the bounds even though it's absent from
    # wsi_tile_regional_coords itself.
    xmin, xmax, ymin, ymax = inputs.full_slide_coord_bounds
    removed_tile_regional_x = (20.0 - reference[0]) / scale
    assert xmin <= removed_tile_regional_x <= xmax
    assert removed_tile_regional_x not in inputs.wsi_tile_regional_coords[:, 0].tolist()
    assert inputs.provenance["wsi_context_available"] is True


def test_build_spatial_field_example_excludes_hole_overlapping_wsi_tiles_but_keeps_visible_ones():
    """Direct acceptance-criteria test: changing the content of a WSI
    tile whose footprint overlaps the query hole must NOT change any
    field of the built example; changing a visible tile's content MUST."""
    adata = _square_grid_adata(n_side=6, spacing=10.0)
    patches = _matching_patches(adata)
    barcodes = list(adata.obs_names)
    query_barcode = barcodes[14]
    context = [b for b in barcodes if b != query_barcode]

    def _build(hole_tile_value: float, visible_tile_value: float):
        slide_context = _wsi_slide_context(_WSI_LEVEL0_TILES)
        slide_context["features"][1, :] = hole_tile_value  # x=20 -- overlaps the hole
        slide_context["features"][0, :] = visible_tile_value  # x=0 -- visible
        inputs, _ = build_spatial_field_example(
            adata, patches, context, [query_barcode], _stub_image_feature_fn,
            sample_id="S0", patient_id="P0", patch_size_fullres=15.0,
            full_sample_coords=adata.obsm["spatial"], slide_context=slide_context,
        )
        return inputs

    baseline = _build(hole_tile_value=1.0, visible_tile_value=5.0)
    changed_hole_only = _build(hole_tile_value=999.0, visible_tile_value=5.0)  # hole tile corrupted
    changed_visible = _build(hole_tile_value=1.0, visible_tile_value=999.0)  # visible tile corrupted

    assert np.array_equal(baseline.wsi_tile_features, changed_hole_only.wsi_tile_features)
    assert not np.array_equal(baseline.wsi_tile_features, changed_visible.wsi_tile_features)


def test_build_spatial_field_example_wsi_regional_coordinates_are_stable_across_different_masks():
    """17th/18th Codex re-audits (Step 5 Part 2 launch blocker #2),
    CONFIRMED: a prior version derived `reference` from ONLY this
    example's own observed+query subset -- the SAME physical WSI tile
    would land at a DIFFERENT regional coordinate depending purely on
    which mask happened to be realized on the identical sample,
    contradicting the slide-stable regional-grid contract. The 18th
    re-audit specifically flagged that using the COMPLETE non-query
    complement as context (as an earlier version of this test did) is
    too easy a case -- the original bug was about CAPPED, RESERVED, or
    otherwise FILTERED/incomplete context, not just a different query.
    Two DIFFERENT single-spot queries, each with a DIFFERENT, genuinely
    CAPPED (far short of the full non-query complement) and DISJOINT
    context subset, on the SAME sample/slide_context, must still agree
    exactly on the regional coordinate of a tile visible under both, and
    on full_slide_coord_bounds and the recorded coordinate_reference."""
    adata = _square_grid_adata(n_side=6, spacing=10.0)
    patches = _matching_patches(adata)
    barcodes = list(adata.obs_names)
    full_coords = adata.obsm["spatial"]

    def _build(query_idx: int, context_idx_range: range):
        query_barcode = barcodes[query_idx]
        # A genuinely CAPPED context: 14 of the sample's 35 non-query
        # spots, not the full complement -- and, across the two calls
        # below, a DISJOINT range, so this is real, different, filtered
        # context, exactly the "capped/reserved/filtered" scenario the
        # 18th Codex re-audit named.
        context = [barcodes[i] for i in context_idx_range if i != query_idx]
        slide_context = _wsi_slide_context(_WSI_LEVEL0_TILES)
        inputs, _ = build_spatial_field_example(
            adata, patches, context, [query_barcode], _stub_image_feature_fn,
            sample_id="S0", patient_id="P0", patch_size_fullres=15.0,
            full_sample_coords=full_coords, slide_context=slide_context,
        )
        return inputs

    # index 14 -> (20, 20); index 20 -> (30, 20) (_square_grid_adata's
    # index = x*n_side+y, spacing=10.0). Both holes leave the level-0
    # tile at (0, 10) visible (verified by hand against the same
    # overlap-threshold arithmetic _wsi_slide_context's docstring uses),
    # while removing different OTHER tiles -- a real, different mask.
    inputs_a = _build(14, range(0, 15))  # capped context: barcodes 0..14 only
    inputs_b = _build(20, range(20, 35))  # capped, DISJOINT-from-A context: barcodes 20..34 only
    assert inputs_a.provenance["coordinate_reference"] == inputs_b.provenance["coordinate_reference"]
    assert inputs_a.provenance["spot_spacing_scale"] == inputs_b.provenance["spot_spacing_scale"]
    assert inputs_a.full_slide_coord_bounds == inputs_b.full_slide_coord_bounds

    common_tile_regional_a = inputs_a.wsi_tile_regional_coords[0]  # tile (0, 10) is row 0 in both
    common_tile_regional_b = inputs_b.wsi_tile_regional_coords[0]
    assert np.allclose(common_tile_regional_a, common_tile_regional_b)
    assert np.allclose(common_tile_regional_a, [-2.5, -1.5])  # (0-25)/10, (10-25)/10


def test_build_spatial_field_example_slide_context_requires_full_sample_coords():
    """17th Codex re-audit (Step 5 Part 2 launch blocker #2): WSI
    regional-grid stability across masks fundamentally depends on a
    coordinate reference derived from the COMPLETE sample lattice --
    combining slide_context with the require_full_sample_coords=False
    escape hatch must fail closed, not silently produce a mask-dependent
    (and therefore unstable) regional frame."""
    adata = _square_grid_adata(n_side=6, spacing=10.0)
    patches = _matching_patches(adata)
    barcodes = list(adata.obs_names)
    query_barcode = barcodes[14]
    context = [b for b in barcodes if b != query_barcode]
    slide_context = _wsi_slide_context(_WSI_LEVEL0_TILES)

    with pytest.raises(ValueError, match="slide_context requires full_sample_coords"):
        build_spatial_field_example(
            adata, patches, context, [query_barcode], _stub_image_feature_fn,
            sample_id="S0", patient_id="P0", patch_size_fullres=15.0,
            require_full_sample_coords=False, slide_context=slide_context,
        )


def test_build_spatial_field_example_rejects_unsupported_image_modes():
    """17th Codex re-audit (Step 5 Part 2 launch blocker #4): image
    intervention semantics ("all_zero" removes WSI context but leaves
    spot H&E features populated; "shuffled" does not actually shuffle
    WSI features; "full" still removes spot patches overlapping the
    hole) are not implemented consistently yet -- the real builder must
    fail closed on anything but "target_zero" rather than silently
    producing an inconsistent example."""
    adata = _square_grid_adata(n_side=6, spacing=10.0)
    patches = _matching_patches(adata)
    barcodes = list(adata.obs_names)
    query_barcode = barcodes[14]
    context = [b for b in barcodes if b != query_barcode]
    slide_context = _wsi_slide_context(_WSI_LEVEL0_TILES)

    for unsupported in ("all_zero", "full", "shuffled"):
        with pytest.raises(ValueError, match="not yet implemented consistently"):
            build_spatial_field_example(
                adata, patches, context, [query_barcode], _stub_image_feature_fn,
                sample_id="S0", patient_id="P0", patch_size_fullres=15.0,
                full_sample_coords=adata.obsm["spatial"], slide_context=slide_context,
                image_mode=unsupported,
            )


def test_build_spatial_field_example_rejects_unsupported_image_modes_even_without_slide_context():
    """18th Codex re-audit (Step 5 Part 2, "Other real gaps"), CONFIRMED
    real: a prior version only validated image_mode when slide_context
    was ALSO given -- an Architecture 1/2 caller (no WSI context at all)
    could pass image_mode="all_zero"/"full"/"shuffled" and it would be
    silently ignored, giving the false impression the mode had some
    real effect. Must raise unconditionally now."""
    adata = _square_grid_adata(n_side=6, spacing=10.0)
    patches = _matching_patches(adata)
    barcodes = list(adata.obs_names)
    query_barcode = barcodes[14]
    context = [b for b in barcodes if b != query_barcode]

    for unsupported in ("all_zero", "full", "shuffled"):
        with pytest.raises(ValueError, match="not yet implemented consistently"):
            build_spatial_field_example(
                adata, patches, context, [query_barcode], _stub_image_feature_fn,
                sample_id="S0", patient_id="P0", patch_size_fullres=15.0,
                require_full_sample_coords=False, image_mode=unsupported,  # NO slide_context at all
            )


def test_build_spatial_field_example_no_slide_context_leaves_wsi_fields_unset():
    adata = _square_grid_adata()
    patches = _matching_patches(adata)
    barcodes = list(adata.obs_names)
    query = barcodes[14:16]
    context = [b for b in barcodes if b not in query]

    inputs, _ = build_spatial_field_example(
        adata, patches, context, query, _stub_image_feature_fn,
        sample_id="S0", patient_id="P0", patch_size_fullres=1.0,
        require_full_sample_coords=False,
    )
    assert inputs.wsi_tile_features is None
    assert inputs.slide_cache_namespace is None
    assert inputs.provenance["wsi_context_available"] is False


def _make_synthetic_hest1k_with_patches(
    tmp_path: Path, organ_sample_ids: dict[str, list[str]], n_spots_per_sample: int = 6,
    gene_names: list[str] | None = None, seed: int = 0, missing_patch_barcodes: set[str] | None = None,
) -> tuple[Path, Path, list[str]]:
    gene_names = gene_names or [f"GENE{i}" for i in range(8)]
    missing_patch_barcodes = missing_patch_barcodes or set()
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

            # 18th Codex re-audit (Step 5 Part 2 launch blocker #1): a
            # real HEST-1k patches .h5 can be missing a barcode that IS
            # present (and expression-QC-valid) in the .h5ad -- this is
            # the normal ~4.5% tissue/WSI-edge gap, simulated here by
            # simply omitting `missing_patch_barcodes` members from the
            # patches file entirely.
            patch_barcodes = [b for b in barcodes if b not in missing_patch_barcodes]
            with h5py.File(hest_dir / "patches" / f"{sid}.h5", "w") as f:
                f.create_dataset("img", data=np.zeros((len(patch_barcodes), 4, 4, 3), dtype=np.uint8))
                f.create_dataset("barcode", data=np.array([[b.encode()] for b in patch_barcodes]))

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
    adata, patches, image_source_available = load_sample_for_examples(manifest, "L0")
    assert list(adata.var_names) == manifest["gene_panel"]
    # 18th Codex re-audit (Step 5 Part 2 launch blocker #1): the barcode
    # universe is now EXACTLY the manifest's declared set, never a
    # silently-shrunk subset (align_patches_to_adata no longer drops
    # spots for missing H&E patches).
    assert set(adata.obs_names) == set(manifest["samples"]["L0"]["barcodes"])
    assert patches.shape[0] == adata.n_obs
    assert image_source_available.shape == (adata.n_obs,)
    assert image_source_available.dtype == bool


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
    adata, patches, image_source_available = load_sample_for_examples(manifest, "L0")
    barcodes = list(adata.obs_names)
    query = barcodes[3:5]
    context = [b for b in barcodes if b not in query]

    inputs, targets = build_spatial_field_example(
        adata, patches, context, query, _stub_image_feature_fn,
        sample_id="L0", patient_id=manifest["samples"]["L0"]["patient_id"], patch_size_fullres=1.0,
        full_sample_coords=adata.obsm["spatial"],  # the recommended real usage, not the test-only escape hatch
        image_source_available=image_source_available,
    )
    assert set(inputs.query_barcodes.tolist()) == set(query)
    assert set(inputs.observed_barcodes.tolist()).issubset(set(context))


def test_a_mask_referencing_a_gex_valid_spot_with_no_h_e_patch_builds_successfully(tmp_path):
    """18th Codex re-audit (Step 5 Part 2 launch blocker #1), CONFIRMED
    real: masks (mask_bank.py) are realized against the MANIFEST's
    expression-QC spot set, which never accounts for H&E-patch
    availability -- a prior version of align_patches_to_adata silently
    DROPPED any spot with no matching patch, so a mask referencing that
    barcode as CONTEXT would make build_spatial_field_example raise
    "mask references barcodes absent from the aligned sample data" far
    downstream of manifest/mask construction, or (worse) silently use a
    different spot universe than the mask was built against. Fixed: the
    spot is retained end to end -- through load_sample_for_examples,
    through the real (context, query) split, all the way to a
    successfully-built example with observed_image_available=False and
    a zeroed feature row for exactly that spot, nothing else."""
    missing_barcode = "L0-SPOT3-1"
    hest_dir, meta_path, gene_names = _make_synthetic_hest1k_with_patches(
        tmp_path, {"Lung": [f"L{i}" for i in range(3)]}, n_spots_per_sample=8,
        missing_patch_barcodes={missing_barcode},
    )
    manifest = build_dataset_manifest(
        hest_dir, str(meta_path), organs="all", min_samples_per_organ=3,
        n_validation_per_organ=0, n_test_per_organ=0, split_seed=0,
        check_gene_panel_compatibility=False, min_nb_genes=None,
        gene_min_genes_per_spot=1, gene_min_cells=0,
    )
    # The manifest's declared barcode set is built from expression QC
    # alone -- it includes missing_barcode, exactly like a real mask
    # realized against it would.
    assert missing_barcode in manifest["samples"]["L0"]["barcodes"]

    adata, patches, image_source_available = load_sample_for_examples(manifest, "L0")
    barcodes = list(adata.obs_names)
    assert missing_barcode in barcodes  # RETAINED, not dropped
    missing_pos = barcodes.index(missing_barcode)
    assert image_source_available[missing_pos] == False  # noqa: E712 -- real numpy bool
    assert np.all(patches[missing_pos] == 0)  # explicit zero placeholder, never a garbage/misaligned patch

    # A mask (built exactly as mask_bank.py would, from the manifest's
    # own declared set) that puts missing_barcode in CONTEXT, far from
    # any query spot -- must build successfully, not raise.
    query = [barcodes[6]]
    context = [b for b in barcodes if b != barcodes[6]]
    assert missing_barcode in context

    inputs, targets = build_spatial_field_example(
        adata, patches, context, query, _stub_image_feature_fn,
        sample_id="L0", patient_id=manifest["samples"]["L0"]["patient_id"], patch_size_fullres=1.0,
        full_sample_coords=adata.obsm["spatial"], image_source_available=image_source_available,
        expected_feature_width=_N_FEATURES,
    )
    observed_barcodes = inputs.observed_barcodes.tolist()
    assert missing_barcode in observed_barcodes  # RETAINED in the built example too
    missing_observed_pos = observed_barcodes.index(missing_barcode)
    assert inputs.observed_image_available[missing_observed_pos] == False  # noqa: E712
    assert np.all(inputs.observed_gigapath_features[missing_observed_pos] == 0.0)
    assert inputs.provenance["n_context_image_unavailable_for_missing_source_patch"] == 1


def test_build_spatial_field_example_rejects_a_malformed_image_source_available(tmp_path):
    adata = _square_grid_adata()
    patches = _matching_patches(adata)
    barcodes = list(adata.obs_names)
    query = barcodes[:1]
    context = barcodes[1:]
    wrong_shape = np.ones(adata.n_obs - 1, dtype=bool)  # one row short
    with pytest.raises(ValueError, match="image_source_available has shape"):
        build_spatial_field_example(
            adata, patches, context, query, _stub_image_feature_fn,
            sample_id="S0", patient_id="P0", patch_size_fullres=1.0,
            require_full_sample_coords=False, image_source_available=wrong_shape,
        )
