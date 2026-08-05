"""Tests for the histology-structure-context ablation: deterministic
morphology feature extraction (histology_features.py), the manifest-
driven spot-feature cache (histology_cache.py), and the zero-init-safe
`HistologyContextInjector` composition wrapper (histology_context.py)."""
import numpy as np
import pytest
import torch

from gen3_multiscale.conditional_wae.histology_cache import (
    build_histology_feature_cache, cfg_cache_root, load_gen3_histology_features, load_histology_features,
)
from gen3_multiscale.conditional_wae.histology_context import HistologyContextInjector
from gen3_multiscale.conditional_wae.histology_features import (
    FEATURE_DIM, _TILE_FEATURE_DIM, compute_multiscale_histology_features, compute_spot_tile_features,
    compute_tile_morphology_features,
)
from gen3_multiscale.conditional_wae.inputs import FullImageExpressionInputs
from gen3_multiscale.conditional_wae.model import Architecture1ImageConditioner


def _synthetic_tile(seed=0, size=16):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)


def _synthetic_slide(n=8, n_unavailable=2, seed=0, size=16):
    rng = np.random.default_rng(seed)
    barcodes = np.array([f"S{i}-1" for i in range(n)])
    coords = np.array([[float(i), float(i % 3)] for i in range(n)]) * 100.0
    patches = np.stack([rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8) for _ in range(n)])
    image_source_available = np.ones(n, dtype=bool)
    image_source_available[:n_unavailable] = False
    patches[:n_unavailable] = 0
    return barcodes, coords, patches, image_source_available


# -- compute_tile_morphology_features -----------------------------------------

def test_compute_tile_morphology_features_has_the_expected_shape_and_is_finite():
    features = compute_tile_morphology_features(_synthetic_tile())
    assert features.shape == (_TILE_FEATURE_DIM,)
    assert np.isfinite(features).all()


def test_compute_tile_morphology_features_is_deterministic():
    tile = _synthetic_tile(seed=3)
    first = compute_tile_morphology_features(tile)
    second = compute_tile_morphology_features(tile)
    np.testing.assert_array_equal(first, second)


def test_compute_tile_morphology_features_rejects_a_wrong_shape():
    with pytest.raises(ValueError, match="H, W, 3"):
        compute_tile_morphology_features(np.zeros((16, 16), dtype=np.uint8))


def test_compute_tile_morphology_features_handles_a_uniform_flat_tile_without_nan():
    flat = np.full((16, 16, 3), 100, dtype=np.uint8)
    features = compute_tile_morphology_features(flat)
    assert np.isfinite(features).all()


# -- compute_spot_tile_features ------------------------------------------------

def test_compute_spot_tile_features_zeroes_unavailable_rows():
    _barcodes, _coords, patches, availability = _synthetic_slide(n=5, n_unavailable=2)
    features = compute_spot_tile_features(patches, availability)
    assert features.shape == (5, _TILE_FEATURE_DIM)
    np.testing.assert_array_equal(features[~availability], np.zeros((2, _TILE_FEATURE_DIM), dtype=np.float32))
    assert np.abs(features[availability]).sum() > 0


def test_compute_spot_tile_features_rejects_misaligned_inputs():
    patches = np.zeros((5, 16, 16, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="row-aligned"):
        compute_spot_tile_features(patches, np.ones(4, dtype=bool))


# -- compute_multiscale_histology_features ------------------------------------

def test_compute_multiscale_histology_features_has_the_expected_shape():
    _barcodes, coords, patches, availability = _synthetic_slide(n=8, n_unavailable=2)
    tile_features = compute_spot_tile_features(patches, availability)
    features = compute_multiscale_histology_features(tile_features, coords, availability)
    assert features.shape == (8, FEATURE_DIM)
    assert np.isfinite(features).all()


def test_own_tile_slice_is_zero_for_unavailable_spots():
    _barcodes, coords, patches, availability = _synthetic_slide(n=8, n_unavailable=2)
    tile_features = compute_spot_tile_features(patches, availability)
    features = compute_multiscale_histology_features(tile_features, coords, availability)
    unavailable_idx = np.flatnonzero(~availability)
    own_slice = features[unavailable_idx, :_TILE_FEATURE_DIM]
    np.testing.assert_array_equal(own_slice, np.zeros((unavailable_idx.size, _TILE_FEATURE_DIM), dtype=np.float32))


def test_unavailable_spots_still_receive_real_neighbor_aggregates():
    """An unavailable spot's OWN tile feature is zero, but it must still
    receive real (non-degenerate) neighbor/regional/slide aggregates
    pooled from nearby available spots -- it is not simply dropped."""
    _barcodes, coords, patches, availability = _synthetic_slide(n=8, n_unavailable=2)
    tile_features = compute_spot_tile_features(patches, availability)
    features = compute_multiscale_histology_features(tile_features, coords, availability)
    unavailable_idx = np.flatnonzero(~availability)
    neighbor_and_beyond = features[unavailable_idx, _TILE_FEATURE_DIM:]
    assert np.abs(neighbor_and_beyond).sum() > 0


def test_an_unavailable_spots_zero_placeholder_never_pollutes_a_real_neighbors_aggregate():
    """Leakage-adjacent guard: replacing an UNAVAILABLE spot's real (but
    hidden) tile content with a completely different image must not
    change any AVAILABLE spot's aggregated features, since the
    unavailable spot's zero placeholder is excluded from every
    aggregate."""
    _barcodes, coords, patches, availability = _synthetic_slide(n=8, n_unavailable=2, seed=1)
    tile_features_a = compute_spot_tile_features(patches, availability)
    features_a = compute_multiscale_histology_features(tile_features_a, coords, availability)

    patches_b = patches.copy()
    patches_b[~availability] = np.random.default_rng(999).integers(0, 255, size=patches_b[~availability].shape, dtype=np.uint8)
    tile_features_b = compute_spot_tile_features(patches_b, availability)
    features_b = compute_multiscale_histology_features(tile_features_b, coords, availability)

    np.testing.assert_array_equal(features_a[availability], features_b[availability])


def test_compute_multiscale_histology_features_rejects_a_coords_shape_mismatch():
    _barcodes, coords, patches, availability = _synthetic_slide(n=5, n_unavailable=1)
    tile_features = compute_spot_tile_features(patches, availability)
    with pytest.raises(ValueError, match="coords"):
        compute_multiscale_histology_features(tile_features, coords[:-1], availability)


def test_compute_multiscale_histology_features_handles_zero_available_spots():
    n = 4
    tile_features = np.zeros((n, _TILE_FEATURE_DIM), dtype=np.float32)
    coords = np.zeros((n, 2), dtype=np.float64)
    availability = np.zeros(n, dtype=bool)
    features = compute_multiscale_histology_features(tile_features, coords, availability)
    assert features.shape == (n, FEATURE_DIM)
    np.testing.assert_array_equal(features, np.zeros((n, FEATURE_DIM), dtype=np.float32))


# -- histology_cache: build/load round trip -----------------------------------

def test_build_then_load_round_trips(tmp_path):
    barcodes, coords, patches, availability = _synthetic_slide(n=6, n_unavailable=1)
    build_histology_feature_cache(tmp_path, "S0", barcodes, coords, patches, availability)
    loaded = load_histology_features(tmp_path, "S0", barcodes, coords, patches, availability)
    assert loaded["features"].shape == (6, FEATURE_DIM)
    np.testing.assert_array_equal(loaded["barcodes"], barcodes)
    np.testing.assert_array_equal(loaded["image_source_available"], availability)
    assert loaded["provenance"]["feature_dim"] == FEATURE_DIM


def test_load_rejects_a_missing_cache(tmp_path):
    barcodes, coords, patches, availability = _synthetic_slide(n=4, n_unavailable=0)
    with pytest.raises(FileNotFoundError, match="precompute_gen3_histology_features"):
        load_histology_features(tmp_path, "MISSING", barcodes, coords, patches, availability)


def test_load_rejects_a_cache_missing_required_fields(tmp_path):
    barcodes, coords, patches, availability = _synthetic_slide(n=4, n_unavailable=0)
    path = build_histology_feature_cache(tmp_path, "S0", barcodes, coords, patches, availability)
    payload = dict(np.load(path, allow_pickle=False))
    del payload["feature_spec"]
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="missing fields"):
        load_histology_features(tmp_path, "S0", barcodes, coords, patches, availability)


def test_load_rejects_a_barcode_order_mismatch(tmp_path):
    barcodes, coords, patches, availability = _synthetic_slide(n=4, n_unavailable=0)
    build_histology_feature_cache(tmp_path, "S0", barcodes, coords, patches, availability)
    shuffled = barcodes[::-1].copy()
    with pytest.raises(ValueError, match="barcode identity/order mismatch"):
        load_histology_features(tmp_path, "S0", shuffled, coords, patches, availability)


def test_load_rejects_changed_coordinates(tmp_path):
    barcodes, coords, patches, availability = _synthetic_slide(n=4, n_unavailable=0)
    build_histology_feature_cache(tmp_path, "S0", barcodes, coords, patches, availability)
    changed_coords = coords + 1.0
    with pytest.raises(ValueError, match="content_sha256"):
        load_histology_features(tmp_path, "S0", barcodes, changed_coords, patches, availability)


def test_load_rejects_changed_patch_content(tmp_path):
    barcodes, coords, patches, availability = _synthetic_slide(n=4, n_unavailable=0)
    build_histology_feature_cache(tmp_path, "S0", barcodes, coords, patches, availability)
    changed_patches = patches.copy()
    changed_patches[availability] = 255 - changed_patches[availability]
    with pytest.raises(ValueError, match="content_sha256"):
        load_histology_features(tmp_path, "S0", barcodes, coords, changed_patches, availability)


def test_load_rejects_a_corrupted_own_tile_slice_for_an_unavailable_spot(tmp_path):
    barcodes, coords, patches, availability = _synthetic_slide(n=4, n_unavailable=1)
    path = build_histology_feature_cache(tmp_path, "S0", barcodes, coords, patches, availability)
    payload = dict(np.load(path, allow_pickle=False))
    corrupted = payload["features"].copy()
    unavailable_idx = np.flatnonzero(~availability)
    corrupted[unavailable_idx[0], 0] = 1.0
    payload["features"] = corrupted
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="nonzero own-tile slice"):
        load_histology_features(tmp_path, "S0", barcodes, coords, patches, availability)


def test_build_rejects_duplicate_barcodes(tmp_path):
    barcodes, coords, patches, availability = _synthetic_slide(n=4, n_unavailable=0)
    barcodes = barcodes.copy()
    barcodes[1] = barcodes[0]
    with pytest.raises(ValueError, match="duplicate"):
        build_histology_feature_cache(tmp_path, "S0", barcodes, coords, patches, availability)


def test_cfg_cache_root_prefers_the_explicit_override():
    from pathlib import Path

    from omegaconf import OmegaConf
    cfg = OmegaConf.create(
        {"data": {"gen3_histology_feature_cache_dir": "/explicit", "hest_cache_dir": "/fallback"}},
    )
    assert cfg_cache_root(cfg) == Path("/explicit")


def test_cfg_cache_root_falls_back_to_hest_cache_dir():
    from pathlib import Path

    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"data": {"hest_cache_dir": "/fallback", "hest_data_dir": "/data"}})
    assert cfg_cache_root(cfg) == Path("/fallback")


def test_load_gen3_histology_features_returns_the_feature_matrix_directly(tmp_path):
    from omegaconf import OmegaConf
    barcodes, coords, patches, availability = _synthetic_slide(n=4, n_unavailable=0)
    cfg = OmegaConf.create({"data": {"gen3_histology_feature_cache_dir": str(tmp_path)}})
    build_histology_feature_cache(tmp_path, "S0", barcodes, coords, patches, availability)
    features = load_gen3_histology_features(cfg, "S0", barcodes, coords, patches, availability)
    assert features.shape == (4, FEATURE_DIM)


# -- HistologyContextInjector ---------------------------------------------------

def _tiny_conditioner(n_genes=5, image_feature_dim=8, hidden_dim=16):
    return Architecture1ImageConditioner(
        n_genes=n_genes, image_feature_dim=image_feature_dim, gex_feature_dim=4,
        hidden_dim=hidden_dim, n_heads=2, n_blocks=1, dense_threshold=64, sparse_k=4,
    )


def _tiny_inputs(n=5, n_query=2, image_feature_dim=8, histology_feature_dim=FEATURE_DIM, seed=0):
    rng = np.random.default_rng(seed)
    image_available = np.ones(n, dtype=bool)
    query_mask = np.zeros(n, dtype=bool)
    query_mask[:n_query] = True
    image_features = rng.normal(size=(n, image_feature_dim)).astype(np.float32)
    coords = rng.normal(size=(n, 2)).astype(np.float32)
    histology_features = rng.normal(size=(n, histology_feature_dim)).astype(np.float32)
    return FullImageExpressionInputs(
        sample_id="S0", image_features=image_features, coords=coords, image_available=image_available,
        query_mask=query_mask, histology_features=histology_features,
    )


def test_injector_is_an_exact_identity_at_construction_in_eval_mode():
    torch.manual_seed(0)
    conditioner = _tiny_conditioner()
    injector = HistologyContextInjector(conditioner, histology_feature_dim=FEATURE_DIM)
    injector.eval()
    inputs = _tiny_inputs()
    plain_context = conditioner(inputs)
    injected_context = injector(inputs)
    assert torch.equal(plain_context, injected_context)


def test_injector_changes_output_after_weights_move_away_from_zero_init():
    torch.manual_seed(0)
    conditioner = _tiny_conditioner()
    injector = HistologyContextInjector(conditioner, histology_feature_dim=FEATURE_DIM)
    injector.eval()
    inputs = _tiny_inputs()
    before = injector(inputs)
    with torch.no_grad():
        for parameter in injector.project.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)
    after = injector(inputs)
    assert not torch.equal(before, after)


def test_injector_requires_histology_features_on_the_inputs():
    conditioner = _tiny_conditioner()
    injector = HistologyContextInjector(conditioner, histology_feature_dim=FEATURE_DIM)
    inputs = _tiny_inputs()
    inputs = FullImageExpressionInputs(**{**inputs.__dict__, "histology_features": None})
    with pytest.raises(ValueError, match="histology_features"):
        injector(inputs)


def test_injector_rejects_a_histology_feature_dim_mismatch():
    conditioner = _tiny_conditioner()
    injector = HistologyContextInjector(conditioner, histology_feature_dim=FEATURE_DIM)
    inputs = _tiny_inputs(histology_feature_dim=FEATURE_DIM - 1)
    with pytest.raises(ValueError, match="histology_features has"):
        injector(inputs)


def test_injector_rejects_a_row_count_mismatch():
    conditioner = _tiny_conditioner()
    injector = HistologyContextInjector(conditioner, histology_feature_dim=FEATURE_DIM)
    inputs = _tiny_inputs(n=5)
    bad_histology = FullImageExpressionInputs(
        **{**inputs.__dict__, "histology_features": np.zeros((4, FEATURE_DIM), dtype=np.float32)},
    )
    with pytest.raises(ValueError, match="histology_features must be"):
        injector(bad_histology)


def test_injector_slices_histology_features_by_query_mask_not_by_row_order():
    """The wrapped conditioner returns context for QUERY rows only, in
    slide-row order restricted to query_mask -- verify the injector adds
    the histology rows for exactly those slide positions, not the first
    len(query_rows) rows of the full histology_features array."""
    torch.manual_seed(0)
    n_genes = 5
    conditioner = _tiny_conditioner(n_genes=n_genes)
    injector = HistologyContextInjector(conditioner, histology_feature_dim=FEATURE_DIM, hidden_dim=8)
    injector.eval()
    with torch.no_grad():
        for parameter in injector.project.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)

    n = 6
    rng = np.random.default_rng(1)
    image_available = np.ones(n, dtype=bool)
    query_mask = np.array([False, True, False, True, False, False])
    image_features = rng.normal(size=(n, 8)).astype(np.float32)
    coords = rng.normal(size=(n, 2)).astype(np.float32)
    histology_features = rng.normal(size=(n, FEATURE_DIM)).astype(np.float32)
    inputs = FullImageExpressionInputs(
        sample_id="S0", image_features=image_features, coords=coords, image_available=image_available,
        query_mask=query_mask, histology_features=histology_features,
    )
    context = conditioner(inputs)
    expected = context + injector.project(torch.as_tensor(histology_features[query_mask], dtype=context.dtype))
    actual = injector(inputs)
    assert torch.allclose(actual, expected)


def test_injector_exposes_hidden_dim_and_n_genes_from_the_wrapped_conditioner():
    conditioner = _tiny_conditioner(n_genes=7, hidden_dim=24)
    injector = HistologyContextInjector(conditioner, histology_feature_dim=FEATURE_DIM)
    assert injector.hidden_dim == 24
    assert injector.n_genes == 7
