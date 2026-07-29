"""Tests for gen3_multiscale/data/spot_feature_cache.py -- Gen3's own
manifest-driven GigaPath spot-feature cache (20th Codex re-audit, Step
5/6 boundary #3: "the legacy scripts/precompute_gigapath_samples.py
cache is not a valid Gen3 input... Build a Gen3-specific, manifest-
driven spot-feature cache/provider")."""
import numpy as np
import pytest
from omegaconf import OmegaConf

from gen3_multiscale.data.spot_feature_cache import (
    _patch_content_sha256, build_gen3_spot_feature_cache, encode_gen3_spot_feature_cache,
    load_gen3_spot_features, load_gigapath_tile_encoder_for_gen3,
)

_VALID_HF_REVISION = "d072f48609bec7ec4d2c43889262b3029bb1279f"  # 40 lowercase hex chars
_GIGAPATH_FEAT_DIM = 1536


def _cfg(tmp_path):
    return OmegaConf.create({"data": {"hest_data_dir": str(tmp_path / "hest1k")}})


def _synthetic_sample(n=5, n_unavailable=1, seed=0):
    rng = np.random.default_rng(seed)
    barcodes = np.array([f"S{i}-1" for i in range(n)])
    patches = np.stack([
        np.full((4, 4, 3), rng.integers(0, 255), dtype=np.uint8) for _ in range(n)
    ])
    image_source_available = np.ones(n, dtype=bool)
    image_source_available[:n_unavailable] = False
    patches[:n_unavailable] = 0  # zero placeholder, matching align_patches_to_adata's contract
    return barcodes, patches, image_source_available


def _stub_gigapath(monkeypatch, feat_dim=_GIGAPATH_FEAT_DIM):
    """Monkeypatch the real (network/GPU-dependent) GigaPath loading and
    encoding functions with cheap, deterministic stubs -- mirrors this
    session's established pattern (e.g. the A100 smoke test's LongNet
    call-counting wrapper) of testing real orchestration logic without
    needing the real heavyweight model. The real
    gigapath_tile_encoder_provenance is left UNSTUBBED (it only needs a
    real state_dict(), which nn.Linear provides) -- except `timm` itself
    is injected into sys.modules, since it is not installed in this
    sandbox and gigapath_tile_encoder_provenance's real, correct
    behavior (a mandatory nonblank timm_version on load) would otherwise
    make every cache built here fail its own real validation."""
    import sys
    import types

    import torch.nn as nn

    def _fake_load(revision=None):
        assert revision is not None
        return nn.Linear(4, 4)

    def _fake_encode(tile_encoder, tensor):
        means = tensor.mean(dim=(1, 2, 3)).view(tensor.shape[0], 1)
        return means.expand(tensor.shape[0], feat_dim).clone()

    fake_timm = types.ModuleType("timm")
    fake_timm.__version__ = "0.0.0-test-stub"
    monkeypatch.setitem(sys.modules, "timm", fake_timm)
    monkeypatch.setattr("src.models.conditioning._load_gigapath_tile_encoder", _fake_load)
    monkeypatch.setattr("src.models.conditioning._gigapath_preprocess_and_encode", _fake_encode)


def test_build_gen3_spot_feature_cache_rejects_an_unpinned_revision(tmp_path, monkeypatch):
    _stub_gigapath(monkeypatch)
    barcodes, patches, availability = _synthetic_sample()
    with pytest.raises(ValueError):
        build_gen3_spot_feature_cache(
            _cfg(tmp_path), "S0", barcodes, patches, availability,
            tile_encoder_revision="main", device="cpu",
        )


def test_build_gen3_spot_feature_cache_rejects_a_barcodes_patches_length_mismatch(tmp_path, monkeypatch):
    _stub_gigapath(monkeypatch)
    barcodes, patches, availability = _synthetic_sample()
    with pytest.raises(ValueError, match="row-aligned"):
        build_gen3_spot_feature_cache(
            _cfg(tmp_path), "S0", barcodes, patches[:-1], availability,
            tile_encoder_revision=_VALID_HF_REVISION, device="cpu",
        )


def test_build_gen3_spot_feature_cache_rejects_duplicate_barcodes(tmp_path, monkeypatch):
    _stub_gigapath(monkeypatch)
    barcodes, patches, availability = _synthetic_sample()
    barcodes[1] = barcodes[0]
    with pytest.raises(ValueError, match="duplicate"):
        build_gen3_spot_feature_cache(
            _cfg(tmp_path), "S0", barcodes, patches, availability,
            tile_encoder_revision=_VALID_HF_REVISION, device="cpu",
        )


def test_build_then_load_gen3_spot_feature_cache_round_trips(tmp_path, monkeypatch):
    _stub_gigapath(monkeypatch)
    barcodes, patches, availability = _synthetic_sample(n=6, n_unavailable=2)
    cfg = _cfg(tmp_path)

    build_gen3_spot_feature_cache(
        cfg, "S0", barcodes, patches, availability,
        tile_encoder_revision=_VALID_HF_REVISION, device="cpu", batch_size=2,
    )
    loaded = load_gen3_spot_features(cfg, "S0", barcodes, patches, availability)

    assert loaded["features"].shape == (6, _GIGAPATH_FEAT_DIM)
    assert np.array_equal(loaded["barcodes"], barcodes)
    assert np.array_equal(loaded["image_source_available"], availability)
    assert loaded["tile_encoder_provenance"]["hf_revision"] == _VALID_HF_REVISION
    # unavailable rows must be an explicit, real zero -- never run through the encoder
    assert np.array_equal(loaded["features"][~availability], np.zeros((2, _GIGAPATH_FEAT_DIM), dtype=np.float32))
    # available rows must be real (nonzero, since the stub encoder returns the patch mean)
    assert np.all(loaded["features"][availability].any(axis=1))


def test_load_gen3_spot_features_raises_a_clear_error_when_no_cache_exists(tmp_path):
    cfg = _cfg(tmp_path)
    barcodes, patches, availability = _synthetic_sample()
    with pytest.raises(FileNotFoundError, match="precompute_gen3_spot_features"):
        load_gen3_spot_features(cfg, "S0", barcodes, patches, availability)


def test_load_gen3_spot_features_rejects_a_barcode_order_mismatch(tmp_path, monkeypatch):
    _stub_gigapath(monkeypatch)
    barcodes, patches, availability = _synthetic_sample(n=4, n_unavailable=0)
    cfg = _cfg(tmp_path)
    build_gen3_spot_feature_cache(
        cfg, "S0", barcodes, patches, availability,
        tile_encoder_revision=_VALID_HF_REVISION, device="cpu",
    )
    reordered = barcodes[::-1].copy()
    with pytest.raises(ValueError, match="barcode identity/order"):
        load_gen3_spot_features(cfg, "S0", reordered, patches[::-1].copy(), availability[::-1].copy())


def test_load_gen3_spot_features_rejects_a_changed_image_source_available(tmp_path, monkeypatch):
    _stub_gigapath(monkeypatch)
    barcodes, patches, availability = _synthetic_sample(n=4, n_unavailable=1)
    cfg = _cfg(tmp_path)
    build_gen3_spot_feature_cache(
        cfg, "S0", barcodes, patches, availability,
        tile_encoder_revision=_VALID_HF_REVISION, device="cpu",
    )
    different_availability = availability.copy()
    different_availability[:] = True  # pretend every spot now has a patch
    with pytest.raises(ValueError, match="image_source_available"):
        load_gen3_spot_features(cfg, "S0", barcodes, patches, different_availability)


def test_load_gen3_spot_features_rejects_patches_that_changed_since_the_cache_was_built(tmp_path, monkeypatch):
    """'validate patch content... on load' (20th Codex re-audit): a cache
    built from patches that have since changed on disk must be rejected,
    not silently trusted."""
    _stub_gigapath(monkeypatch)
    barcodes, patches, availability = _synthetic_sample(n=4, n_unavailable=0)
    cfg = _cfg(tmp_path)
    build_gen3_spot_feature_cache(
        cfg, "S0", barcodes, patches, availability,
        tile_encoder_revision=_VALID_HF_REVISION, device="cpu",
    )
    changed_patches = patches.copy()
    changed_patches[0] = 255 - changed_patches[0]
    with pytest.raises(ValueError, match="patch_content_sha256"):
        load_gen3_spot_features(cfg, "S0", barcodes, changed_patches, availability)


def test_load_gen3_spot_features_rejects_a_cache_missing_provenance_fields(tmp_path):
    """An old-format or legacy (e.g. scripts/precompute_gigapath_samples.py)
    cache carries no tile-encoder provenance at all -- must fail closed,
    not be silently trusted as a valid Gen3 input."""
    cfg = _cfg(tmp_path)
    cache_dir = tmp_path / "hest1k" / "gigapath_gen3_spot_cache"
    cache_dir.mkdir(parents=True)
    barcodes, patches, availability = _synthetic_sample()
    np.savez(  # deliberately missing every tile_encoder_* field, like the legacy cache
        cache_dir / "S0.npz",
        features=np.zeros((5, _GIGAPATH_FEAT_DIM), dtype=np.float32),
        barcodes=barcodes,
    )
    with pytest.raises(ValueError, match="missing fields"):
        load_gen3_spot_features(cfg, "S0", barcodes, patches, availability)


def test_load_gen3_spot_features_rejects_a_malformed_tile_encoder_revision(tmp_path, monkeypatch):
    """The load-side validation must apply the SAME strict provenance
    checks load_slide_context uses for the dense WSI cache -- a
    hand-corrupted cache with an unpinned-looking revision must be
    rejected even though it otherwise round-trips."""
    _stub_gigapath(monkeypatch)
    barcodes, patches, availability = _synthetic_sample(n=3, n_unavailable=0)
    cfg = _cfg(tmp_path)
    build_gen3_spot_feature_cache(
        cfg, "S0", barcodes, patches, availability,
        tile_encoder_revision=_VALID_HF_REVISION, device="cpu",
    )
    cache_path = tmp_path / "hest1k" / "gigapath_gen3_spot_cache" / "S0.npz"
    cached = dict(np.load(cache_path, allow_pickle=False))
    cached["tile_encoder_hf_revision"] = np.asarray("main")  # hand-corrupt to an unpinned ref
    np.savez(cache_path, **cached)

    with pytest.raises(ValueError, match="tile_encoder_hf_revision"):
        load_gen3_spot_features(cfg, "S0", barcodes, patches, availability)


def test_two_samples_built_with_different_pinned_revisions_have_different_provenance(tmp_path, monkeypatch):
    """Sanity check that the cache genuinely records whichever revision
    was passed at build time (not a hardcoded/ignored value) -- the
    preflight gate (tile_encoder_preflight.py) depends on this being real."""
    _stub_gigapath(monkeypatch)
    barcodes, patches, availability = _synthetic_sample(n=3, n_unavailable=0)
    cfg = _cfg(tmp_path)
    other_revision = "1234567890abcdef1234567890abcdef12345678"

    build_gen3_spot_feature_cache(
        cfg, "S0", barcodes, patches, availability,
        tile_encoder_revision=_VALID_HF_REVISION, device="cpu",
    )
    build_gen3_spot_feature_cache(
        cfg, "S1", barcodes, patches, availability,
        tile_encoder_revision=other_revision, device="cpu",
    )
    prov_s0 = load_gen3_spot_features(cfg, "S0", barcodes, patches, availability)["tile_encoder_provenance"]
    prov_s1 = load_gen3_spot_features(cfg, "S1", barcodes, patches, availability)["tile_encoder_provenance"]
    assert prov_s0["hf_revision"] != prov_s1["hf_revision"]


def test_encode_gen3_spot_feature_cache_rejects_a_non_positive_batch_size(tmp_path, monkeypatch):
    """21st Codex re-audit hardening: a negative batch_size makes
    range(0, n, batch_size) an EMPTY range, silently leaving every
    available spot's feature row at its zero-initialized default with no
    error at all -- must be rejected explicitly instead."""
    _stub_gigapath(monkeypatch)
    barcodes, patches, availability = _synthetic_sample()
    encoder, provenance = load_gigapath_tile_encoder_for_gen3(_VALID_HF_REVISION, device="cpu")
    for bad_batch_size in (0, -1, -32):
        with pytest.raises(ValueError, match="batch_size"):
            encode_gen3_spot_feature_cache(
                _cfg(tmp_path), "S0", barcodes, patches, availability, encoder, provenance,
                device="cpu", batch_size=bad_batch_size,
            )


def test_encode_gen3_spot_feature_cache_rejects_a_malformed_encoder_output_shape(tmp_path, monkeypatch):
    """21st Codex re-audit hardening: 'validate encoder output shape...
    before writing' -- a broken/mismatched encoder must be caught at the
    exact batch that produced bad output, not discovered later as a
    generic whole-array shape mismatch."""
    _stub_gigapath(monkeypatch)
    monkeypatch.setattr(
        "src.models.conditioning._gigapath_preprocess_and_encode",
        lambda tile_encoder, tensor: tensor.mean(dim=(1, 2, 3)).view(tensor.shape[0], 1).expand(
            tensor.shape[0], _GIGAPATH_FEAT_DIM - 1,  # deliberately wrong width
        ).clone(),
    )
    barcodes, patches, availability = _synthetic_sample()
    encoder, provenance = load_gigapath_tile_encoder_for_gen3(_VALID_HF_REVISION, device="cpu")
    with pytest.raises(ValueError, match="shape"):
        encode_gen3_spot_feature_cache(
            _cfg(tmp_path), "S0", barcodes, patches, availability, encoder, provenance, device="cpu",
        )


def test_encode_gen3_spot_feature_cache_rejects_non_finite_encoder_output(tmp_path, monkeypatch):
    _stub_gigapath(monkeypatch)
    monkeypatch.setattr(
        "src.models.conditioning._gigapath_preprocess_and_encode",
        lambda tile_encoder, tensor: tensor.mean(dim=(1, 2, 3)).view(tensor.shape[0], 1).expand(
            tensor.shape[0], _GIGAPATH_FEAT_DIM,
        ).clone().fill_(float("nan")),
    )
    barcodes, patches, availability = _synthetic_sample()
    encoder, provenance = load_gigapath_tile_encoder_for_gen3(_VALID_HF_REVISION, device="cpu")
    with pytest.raises(ValueError, match="non-finite"):
        encode_gen3_spot_feature_cache(
            _cfg(tmp_path), "S0", barcodes, patches, availability, encoder, provenance, device="cpu",
        )


def test_load_gigapath_tile_encoder_for_gen3_is_called_only_once_across_many_samples(tmp_path, monkeypatch):
    """21st Codex re-audit hardening: 'prefer loading the tile encoder
    once per worker and reusing it across samples.' Proves the real
    multi-sample usage pattern (load once, encode_gen3_spot_feature_cache
    many times) never reloads the model, by counting real calls to
    _load_gigapath_tile_encoder rather than merely asserting it."""
    _stub_gigapath(monkeypatch)
    load_call_count = {"n": 0}
    real_load = __import__("src.models.conditioning", fromlist=["_load_gigapath_tile_encoder"])._load_gigapath_tile_encoder

    def _counting_load(revision=None):
        load_call_count["n"] += 1
        return real_load(revision=revision)

    monkeypatch.setattr("src.models.conditioning._load_gigapath_tile_encoder", _counting_load)

    cfg = _cfg(tmp_path)
    barcodes, patches, availability = _synthetic_sample(n=3, n_unavailable=0)
    encoder, provenance = load_gigapath_tile_encoder_for_gen3(_VALID_HF_REVISION, device="cpu")
    assert load_call_count["n"] == 1
    for sample_id in ("S0", "S1", "S2"):
        encode_gen3_spot_feature_cache(
            cfg, sample_id, barcodes, patches, availability, encoder, provenance, device="cpu",
        )
    assert load_call_count["n"] == 1, "the tile encoder must be loaded once and reused, not per sample"
    for sample_id in ("S0", "S1", "S2"):
        loaded = load_gen3_spot_features(cfg, sample_id, barcodes, patches, availability)
        assert loaded["tile_encoder_provenance"]["hf_revision"] == _VALID_HF_REVISION


def test_load_gen3_spot_features_rejects_a_nonzero_feature_row_for_an_unavailable_spot(tmp_path, monkeypatch):
    """21st Codex re-audit hardening: 'require unavailable feature rows
    to be exactly zero on load' -- a hand-corrupted cache with a real,
    nonzero row for a spot marked image_source_available=False must be
    rejected, not silently trusted as a real feature."""
    _stub_gigapath(monkeypatch)
    barcodes, patches, availability = _synthetic_sample(n=4, n_unavailable=1)
    cfg = _cfg(tmp_path)
    build_gen3_spot_feature_cache(
        cfg, "S0", barcodes, patches, availability,
        tile_encoder_revision=_VALID_HF_REVISION, device="cpu",
    )
    cache_path = tmp_path / "hest1k" / "gigapath_gen3_spot_cache" / "S0.npz"
    cached = dict(np.load(cache_path, allow_pickle=False))
    corrupted_features = cached["features"].copy()
    unavailable_row = int(np.flatnonzero(~availability)[0])
    corrupted_features[unavailable_row] = 1.0  # hand-corrupt: a "real" value where there must be none
    cached["features"] = corrupted_features
    np.savez(cache_path, **cached)

    with pytest.raises(ValueError, match="nonzero feature row"):
        load_gen3_spot_features(cfg, "S0", barcodes, patches, availability)


def test_patch_content_sha256_is_sensitive_to_shape_and_dtype_not_just_raw_bytes():
    """21st Codex re-audit hardening: 'include patch shape and dtype in
    patch_content_sha256' -- two arrays with the SAME underlying bytes
    but a different shape/dtype view must hash differently."""
    barcodes = np.array(["S0-1", "S1-1"])
    availability = np.array([True, True])
    patches_a = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    patches_a[0] = 7
    patches_b = patches_a.reshape(2, 48).astype(np.uint8)  # same bytes, different shape

    digest_a = _patch_content_sha256(barcodes, availability, patches_a)
    digest_b = _patch_content_sha256(barcodes, availability, patches_b)
    assert digest_a != digest_b
