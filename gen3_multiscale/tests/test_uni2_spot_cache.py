"""Tests for the new Gen3/MK-specific parts of gen3_multiscale/gen4/
uni2_spot_cache.py: strict provenance validation, the preflight
consistency gate, and the cfg-driven load_gen3_uni2_spot_features wrapper
(the UNI2 analogue of data/spot_feature_cache.py's GigaPath discipline)."""
import numpy as np
import pytest
from omegaconf import OmegaConf

from gen3_multiscale.gen4.providers import EncoderIdentity
from gen3_multiscale.gen4.uni2_spot_cache import (
    build_uni2_spot_feature_cache, cfg_cache_root, load_gen3_uni2_spot_features,
    load_uni2_spot_features, require_consistent_uni2_tile_encoder_provenance,
    validate_uni2_tile_encoder_provenance,
)

_VALID_REVISION = "d072f48609bec7ec4d2c43889262b3029bb1279f"  # 40 lowercase hex chars
_VALID_SHA256 = "a" * 64
_EXPECTED_SPEC = "uni2_tile_v2:vit_giant_patch14_224:resize224_bicubic_antialias:imagenet_norm"


def _valid_provenance(**overrides) -> dict:
    provenance = {
        "checkpoint_sha256": _VALID_SHA256,
        "pinned_revision": _VALID_REVISION,
        "package_version": "timm-1.0.0",
        "preprocessing_spec": _EXPECTED_SPEC,
        "output_dim": 1536,
        "schema_version": 1,
    }
    provenance.update(overrides)
    return provenance


class _RealShapedStubUNI2Encoder:
    """Deterministic ImageContextProvider stand-in whose `.identity`
    fields are real-UNI2-SHAPED (valid hex revision/digest, exact
    preprocessing spec, output_dim=1536) -- unlike `_gen4_fixtures.py::
    StubUNI2Encoder`, which deliberately uses non-hex placeholder values
    and is only used against Gen4's general-purpose (unvalidated)
    `load_uni2_spot_features`, never the strict Gen3/MK path this file
    tests."""

    def __init__(self):
        self.identity = EncoderIdentity(
            encoder_name="uni2", checkpoint_sha256=_VALID_SHA256, pinned_revision=_VALID_REVISION,
            package_version="timm-1.0.0", preprocessing_spec=_EXPECTED_SPEC, output_dim=1536,
        )

    def encode_available_patches(self, patches: np.ndarray) -> np.ndarray:
        means = patches.astype(np.float32).reshape(patches.shape[0], -1).mean(axis=1, keepdims=True)
        return np.tile(means, (1, 1536)).astype(np.float32)


def _synthetic_sample(n=5, n_unavailable=1, seed=0):
    rng = np.random.default_rng(seed)
    barcodes = np.array([f"S{i}-1" for i in range(n)])
    patches = np.stack([
        np.full((4, 4, 3), rng.integers(0, 255), dtype=np.uint8) for _ in range(n)
    ])
    image_source_available = np.ones(n, dtype=bool)
    image_source_available[:n_unavailable] = False
    patches[:n_unavailable] = 0
    return barcodes, patches, image_source_available


# -- validate_uni2_tile_encoder_provenance -----------------------------------

def test_validate_uni2_tile_encoder_provenance_accepts_a_well_formed_dict():
    validate_uni2_tile_encoder_provenance("src", _valid_provenance())


def test_validate_uni2_tile_encoder_provenance_rejects_a_malformed_revision():
    with pytest.raises(ValueError, match="pinned_revision"):
        validate_uni2_tile_encoder_provenance("src", _valid_provenance(pinned_revision="not-hex"))


def test_validate_uni2_tile_encoder_provenance_rejects_a_short_revision():
    with pytest.raises(ValueError, match="pinned_revision"):
        validate_uni2_tile_encoder_provenance("src", _valid_provenance(pinned_revision=_VALID_REVISION[:39]))


def test_validate_uni2_tile_encoder_provenance_rejects_a_malformed_checksum():
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        validate_uni2_tile_encoder_provenance("src", _valid_provenance(checkpoint_sha256="stub" * 16))


def test_validate_uni2_tile_encoder_provenance_rejects_a_blank_package_version():
    with pytest.raises(ValueError, match="package_version"):
        validate_uni2_tile_encoder_provenance("src", _valid_provenance(package_version="  "))


def test_validate_uni2_tile_encoder_provenance_rejects_a_wrong_preprocessing_spec():
    with pytest.raises(ValueError, match="preprocessing_spec"):
        validate_uni2_tile_encoder_provenance("src", _valid_provenance(preprocessing_spec="stub_uni2_v1"))


def test_validate_uni2_tile_encoder_provenance_rejects_a_wrong_output_dim():
    with pytest.raises(ValueError, match="output_dim"):
        validate_uni2_tile_encoder_provenance("src", _valid_provenance(output_dim=8))


def test_validate_uni2_tile_encoder_provenance_rejects_an_unsupported_schema_version():
    with pytest.raises(ValueError, match="schema_version"):
        validate_uni2_tile_encoder_provenance("src", _valid_provenance(schema_version=99))


# -- require_consistent_uni2_tile_encoder_provenance -------------------------

def test_require_consistent_uni2_tile_encoder_provenance_accepts_identical_entries():
    provenance = _valid_provenance()
    require_consistent_uni2_tile_encoder_provenance(
        {"s1:spot_features": provenance, "s2:spot_features": dict(provenance)},
        {"pinned_revision": _VALID_REVISION},
    )


def test_require_consistent_uni2_tile_encoder_provenance_rejects_a_mismatched_revision():
    other_revision = "0" * 40
    with pytest.raises(ValueError, match="mismatch"):
        require_consistent_uni2_tile_encoder_provenance(
            {
                "s1:spot_features": _valid_provenance(),
                "s2:spot_features": _valid_provenance(pinned_revision=other_revision),
            },
            {"pinned_revision": _VALID_REVISION},
        )


def test_require_consistent_uni2_tile_encoder_provenance_rejects_disagreement_with_expected():
    with pytest.raises(ValueError, match="mismatch"):
        require_consistent_uni2_tile_encoder_provenance(
            {"s1:spot_features": _valid_provenance()},
            {"pinned_revision": "0" * 40},
        )


def test_require_consistent_uni2_tile_encoder_provenance_rejects_empty_input():
    with pytest.raises(ValueError, match="no provenance entries"):
        require_consistent_uni2_tile_encoder_provenance({}, {"pinned_revision": _VALID_REVISION})


def test_require_consistent_uni2_tile_encoder_provenance_requires_expected_provenance():
    with pytest.raises(ValueError, match="pinned_revision"):
        require_consistent_uni2_tile_encoder_provenance({"s1:spot_features": _valid_provenance()}, {})


def test_require_consistent_uni2_tile_encoder_provenance_rejects_a_malformed_entry():
    with pytest.raises(ValueError, match="pinned_revision"):
        require_consistent_uni2_tile_encoder_provenance(
            {"s1:spot_features": _valid_provenance(pinned_revision="bad")},
            {"pinned_revision": _VALID_REVISION},
        )


def test_require_consistent_uni2_tile_encoder_provenance_rejects_an_entry_missing_fields():
    incomplete = {"pinned_revision": _VALID_REVISION}
    with pytest.raises(ValueError, match="missing field"):
        require_consistent_uni2_tile_encoder_provenance(
            {"s1:spot_features": incomplete}, {"pinned_revision": _VALID_REVISION},
        )


# -- cfg_cache_root ------------------------------------------------------------

def test_cfg_cache_root_respects_an_explicit_override(tmp_path):
    override = tmp_path / "custom_uni2_root"
    cfg = OmegaConf.create({"data": {"gen3_uni2_spot_feature_cache_dir": str(override), "hest_data_dir": str(tmp_path)}})
    assert cfg_cache_root(cfg) == override


def test_cfg_cache_root_accepts_the_concrete_uni2_cache_directory(tmp_path):
    concrete = tmp_path / "uni2_gen3_spot_cache"
    cfg = OmegaConf.create({"data": {
        "gen3_uni2_spot_feature_cache_dir": str(concrete),
        "hest_data_dir": str(tmp_path),
    }})
    assert cfg_cache_root(cfg) == tmp_path


def test_cfg_cache_root_falls_back_to_hest_cache_dir(tmp_path):
    cfg = OmegaConf.create({"data": {"hest_data_dir": str(tmp_path / "hest1k")}})
    assert cfg_cache_root(cfg) == tmp_path / "hest1k"


# -- load_gen3_uni2_spot_features (cfg-driven wrapper) ------------------------

def test_build_then_load_gen3_uni2_spot_features_round_trips(tmp_path):
    cfg = OmegaConf.create({"data": {"hest_data_dir": str(tmp_path / "hest1k")}})
    barcodes, patches, availability = _synthetic_sample()
    encoder = _RealShapedStubUNI2Encoder()
    build_uni2_spot_feature_cache(cfg_cache_root(cfg), "S0", barcodes, patches, availability, encoder)

    loaded = load_gen3_uni2_spot_features(cfg, "S0", barcodes, patches, availability)
    assert loaded["features"].shape == (5, 1536)
    assert np.array_equal(loaded["barcodes"], barcodes)
    assert np.array_equal(loaded["image_source_available"], availability)
    assert loaded["tile_encoder_provenance"]["pinned_revision"] == _VALID_REVISION


def test_load_gen3_uni2_spot_features_rejects_a_cache_with_a_malformed_provenance_field(tmp_path):
    """A hand-corrupted or legacy-built cache with a non-real-shaped
    provenance field must be rejected by the Gen3/MK-specific loader,
    even though it would still load fine through the general-purpose
    `load_uni2_spot_features` (Gen4's own, deliberately less strict,
    entry point)."""
    cfg = OmegaConf.create({"data": {"hest_data_dir": str(tmp_path / "hest1k")}})
    barcodes, patches, availability = _synthetic_sample()

    class _BadEncoder:
        identity = EncoderIdentity(
            encoder_name="uni2", checkpoint_sha256=_VALID_SHA256, pinned_revision=_VALID_REVISION,
            package_version="timm-1.0.0", preprocessing_spec="not_the_real_uni2_spec",  # corrupted/legacy
            output_dim=1536,
        )

        def encode_available_patches(self, patches):
            return np.zeros((patches.shape[0], 1536), dtype=np.float32)

    build_uni2_spot_feature_cache(cfg_cache_root(cfg), "S0", barcodes, patches, availability, _BadEncoder())

    # The general-purpose loader still accepts it (Gen4's own contract).
    loaded = load_uni2_spot_features(cfg_cache_root(cfg), "S0", barcodes, patches, availability)
    assert loaded["provenance"]["preprocessing_spec"] == "not_the_real_uni2_spec"

    # -- but the Gen3/MK-specific loader must fail closed.
    with pytest.raises(ValueError, match="preprocessing_spec"):
        load_gen3_uni2_spot_features(cfg, "S0", barcodes, patches, availability)


def test_load_gen3_uni2_spot_features_rejects_a_missing_cache_without_rebuilding(tmp_path):
    """No cache has ever been written for this sample -- the loader must
    fail closed with a clear message, never silently construct an
    encoder or fall back to a build path."""
    cfg = OmegaConf.create({"data": {"hest_data_dir": str(tmp_path / "hest1k")}})
    barcodes, patches, availability = _synthetic_sample()
    with pytest.raises(FileNotFoundError, match="missing"):
        load_gen3_uni2_spot_features(cfg, "S0", barcodes, patches, availability)


def test_load_gen3_uni2_spot_features_rejects_a_corrupt_cache_missing_required_fields(tmp_path):
    """A cache file that exists but is missing required provenance/data
    fields (e.g. truncated write, wrong builder version) must fail
    closed, not be silently accepted with defaults."""
    cfg = OmegaConf.create({"data": {"hest_data_dir": str(tmp_path / "hest1k")}})
    barcodes, patches, availability = _synthetic_sample()
    path = cfg_cache_root(cfg) / "uni2_gen3_spot_cache" / "S0.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, features=np.zeros((5, 1536), dtype=np.float32), barcodes=barcodes)
    with pytest.raises(ValueError, match="missing fields"):
        load_gen3_uni2_spot_features(cfg, "S0", barcodes, patches, availability)


def test_load_gen3_uni2_spot_features_rejects_a_barcode_order_mismatch(tmp_path):
    cfg = OmegaConf.create({"data": {"hest_data_dir": str(tmp_path / "hest1k")}})
    barcodes, patches, availability = _synthetic_sample()
    encoder = _RealShapedStubUNI2Encoder()
    build_uni2_spot_feature_cache(cfg_cache_root(cfg), "S0", barcodes, patches, availability, encoder)

    reordered = barcodes[::-1].copy()
    with pytest.raises(ValueError, match="mismatch"):
        load_gen3_uni2_spot_features(cfg, "S0", reordered, patches[::-1].copy(), availability[::-1].copy())


def test_gigapath_default_path_behavior_is_unaffected_by_the_uni2_addition(tmp_path, monkeypatch):
    """Regression: adding the uni2 branch must not change GigaPath's own
    (pre-existing, separately audited) spot-feature cache module at all."""
    import sys
    import types

    import torch.nn as nn

    from gen3_multiscale.data.spot_feature_cache import (
        build_gen3_spot_feature_cache, load_gen3_spot_features,
    )

    def _fake_load(revision=None):
        assert revision is not None
        return nn.Linear(4, 4)

    def _fake_encode(tile_encoder, tensor):
        means = tensor.mean(dim=(1, 2, 3)).view(tensor.shape[0], 1)
        return means.expand(tensor.shape[0], 1536).clone()

    fake_timm = types.ModuleType("timm")
    fake_timm.__version__ = "0.0.0-test-stub"
    monkeypatch.setitem(sys.modules, "timm", fake_timm)
    monkeypatch.setattr("src.models.conditioning._load_gigapath_tile_encoder", _fake_load)
    monkeypatch.setattr("src.models.conditioning._gigapath_preprocess_and_encode", _fake_encode)

    cfg = OmegaConf.create({"data": {"hest_data_dir": str(tmp_path / "hest1k")}})
    barcodes, patches, availability = _synthetic_sample()
    build_gen3_spot_feature_cache(
        cfg, "S0", barcodes, patches, availability, tile_encoder_revision=_VALID_REVISION, device="cpu",
    )
    loaded = load_gen3_spot_features(cfg, "S0", barcodes, patches, availability)
    assert loaded["features"].shape == (5, 1536)
    assert loaded["tile_encoder_provenance"]["hf_repo_id"] == "prov-gigapath/prov-gigapath"
