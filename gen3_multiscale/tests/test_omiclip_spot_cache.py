"""Tests for gen3_multiscale/gen4/omiclip_spot_cache.py: strict provenance
validation, the preflight consistency gate, and the cfg-driven
load_gen3_omiclip_spot_features wrapper -- mirrors test_uni2_spot_cache.py
exactly, adapted to OmiCLIP's own field names/shapes (output_dim=768,
coca_ViT-L-14 preprocessing spec)."""
import numpy as np
import pytest
from omegaconf import OmegaConf

from gen3_multiscale.gen4.providers import EncoderIdentity
from gen3_multiscale.gen4.omiclip_spot_cache import (
    build_omiclip_spot_feature_cache, cfg_cache_root, load_gen3_omiclip_spot_features,
    load_omiclip_spot_features, require_consistent_omiclip_tile_encoder_provenance,
    validate_omiclip_tile_encoder_provenance,
)

_VALID_REVISION = "d072f48609bec7ec4d2c43889262b3029bb1279f"  # 40 lowercase hex chars
_VALID_SHA256 = "a" * 64
_EXPECTED_SPEC = "omiclip_tile_v1:coca_ViT-L-14:open_clip_eval_transform"


def _valid_provenance(**overrides) -> dict:
    provenance = {
        "checkpoint_sha256": _VALID_SHA256,
        "pinned_revision": _VALID_REVISION,
        "package_version": "open_clip_torch-2.24.0",
        "preprocessing_spec": _EXPECTED_SPEC,
        "output_dim": 768,
        "schema_version": 1,
    }
    provenance.update(overrides)
    return provenance


class _RealShapedStubOmiCLIPEncoder:
    def __init__(self):
        self.identity = EncoderIdentity(
            encoder_name="omiclip", checkpoint_sha256=_VALID_SHA256, pinned_revision=_VALID_REVISION,
            package_version="open_clip_torch-2.24.0", preprocessing_spec=_EXPECTED_SPEC, output_dim=768,
        )

    def encode_available_patches(self, patches: np.ndarray) -> np.ndarray:
        means = patches.astype(np.float32).reshape(patches.shape[0], -1).mean(axis=1, keepdims=True)
        return np.tile(means, (1, 768)).astype(np.float32)


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


# -- validate_omiclip_tile_encoder_provenance --------------------------------

def test_validate_omiclip_tile_encoder_provenance_accepts_a_well_formed_dict():
    validate_omiclip_tile_encoder_provenance("src", _valid_provenance())


def test_validate_omiclip_tile_encoder_provenance_rejects_a_malformed_revision():
    with pytest.raises(ValueError, match="pinned_revision"):
        validate_omiclip_tile_encoder_provenance("src", _valid_provenance(pinned_revision="not-hex"))


def test_validate_omiclip_tile_encoder_provenance_rejects_a_short_revision():
    with pytest.raises(ValueError, match="pinned_revision"):
        validate_omiclip_tile_encoder_provenance("src", _valid_provenance(pinned_revision=_VALID_REVISION[:39]))


def test_validate_omiclip_tile_encoder_provenance_rejects_a_malformed_checksum():
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        validate_omiclip_tile_encoder_provenance("src", _valid_provenance(checkpoint_sha256="stub" * 16))


def test_validate_omiclip_tile_encoder_provenance_rejects_a_blank_package_version():
    with pytest.raises(ValueError, match="package_version"):
        validate_omiclip_tile_encoder_provenance("src", _valid_provenance(package_version="  "))


def test_validate_omiclip_tile_encoder_provenance_rejects_a_wrong_preprocessing_spec():
    with pytest.raises(ValueError, match="preprocessing_spec"):
        validate_omiclip_tile_encoder_provenance("src", _valid_provenance(preprocessing_spec="stub_omiclip_v1"))


def test_validate_omiclip_tile_encoder_provenance_rejects_a_wrong_output_dim():
    with pytest.raises(ValueError, match="output_dim"):
        validate_omiclip_tile_encoder_provenance("src", _valid_provenance(output_dim=8))


def test_validate_omiclip_tile_encoder_provenance_rejects_an_unsupported_schema_version():
    with pytest.raises(ValueError, match="schema_version"):
        validate_omiclip_tile_encoder_provenance("src", _valid_provenance(schema_version=99))


# -- require_consistent_omiclip_tile_encoder_provenance ----------------------

def test_require_consistent_omiclip_tile_encoder_provenance_accepts_identical_entries():
    provenance = _valid_provenance()
    require_consistent_omiclip_tile_encoder_provenance(
        {"s1:spot_features": provenance, "s2:spot_features": dict(provenance)},
        {"pinned_revision": _VALID_REVISION},
    )


def test_require_consistent_omiclip_tile_encoder_provenance_rejects_a_mismatched_revision():
    other_revision = "0" * 40
    with pytest.raises(ValueError, match="mismatch"):
        require_consistent_omiclip_tile_encoder_provenance(
            {
                "s1:spot_features": _valid_provenance(),
                "s2:spot_features": _valid_provenance(pinned_revision=other_revision),
            },
            {"pinned_revision": _VALID_REVISION},
        )


def test_require_consistent_omiclip_tile_encoder_provenance_rejects_disagreement_with_expected():
    with pytest.raises(ValueError, match="mismatch"):
        require_consistent_omiclip_tile_encoder_provenance(
            {"s1:spot_features": _valid_provenance()},
            {"pinned_revision": "0" * 40},
        )


def test_require_consistent_omiclip_tile_encoder_provenance_rejects_empty_input():
    with pytest.raises(ValueError, match="no provenance entries"):
        require_consistent_omiclip_tile_encoder_provenance({}, {"pinned_revision": _VALID_REVISION})


def test_require_consistent_omiclip_tile_encoder_provenance_requires_expected_provenance():
    with pytest.raises(ValueError, match="pinned_revision"):
        require_consistent_omiclip_tile_encoder_provenance({"s1:spot_features": _valid_provenance()}, {})


def test_require_consistent_omiclip_tile_encoder_provenance_rejects_a_malformed_entry():
    with pytest.raises(ValueError, match="pinned_revision"):
        require_consistent_omiclip_tile_encoder_provenance(
            {"s1:spot_features": _valid_provenance(pinned_revision="bad")},
            {"pinned_revision": _VALID_REVISION},
        )


def test_require_consistent_omiclip_tile_encoder_provenance_rejects_an_entry_missing_fields():
    incomplete = {"pinned_revision": _VALID_REVISION}
    with pytest.raises(ValueError, match="missing field"):
        require_consistent_omiclip_tile_encoder_provenance(
            {"s1:spot_features": incomplete}, {"pinned_revision": _VALID_REVISION},
        )


# -- cfg_cache_root ------------------------------------------------------------

def test_cfg_cache_root_respects_an_explicit_override(tmp_path):
    override = tmp_path / "custom_omiclip_root"
    cfg = OmegaConf.create({"data": {"gen3_omiclip_spot_feature_cache_dir": str(override), "hest_data_dir": str(tmp_path)}})
    assert cfg_cache_root(cfg) == override


def test_cfg_cache_root_falls_back_to_hest_cache_dir(tmp_path):
    cfg = OmegaConf.create({"data": {"hest_data_dir": str(tmp_path / "hest1k")}})
    assert cfg_cache_root(cfg) == tmp_path / "hest1k"


# -- load_gen3_omiclip_spot_features (cfg-driven wrapper) ---------------------

def test_build_then_load_gen3_omiclip_spot_features_round_trips(tmp_path):
    cfg = OmegaConf.create({"data": {"hest_data_dir": str(tmp_path / "hest1k")}})
    barcodes, patches, availability = _synthetic_sample()
    encoder = _RealShapedStubOmiCLIPEncoder()
    build_omiclip_spot_feature_cache(cfg_cache_root(cfg), "S0", barcodes, patches, availability, encoder)

    loaded = load_gen3_omiclip_spot_features(cfg, "S0", barcodes, patches, availability)
    assert loaded["features"].shape == (5, 768)
    assert np.array_equal(loaded["barcodes"], barcodes)
    assert np.array_equal(loaded["image_source_available"], availability)
    assert loaded["tile_encoder_provenance"]["pinned_revision"] == _VALID_REVISION


def test_load_gen3_omiclip_spot_features_rejects_a_cache_with_a_malformed_provenance_field(tmp_path):
    cfg = OmegaConf.create({"data": {"hest_data_dir": str(tmp_path / "hest1k")}})
    barcodes, patches, availability = _synthetic_sample()

    class _BadEncoder:
        identity = EncoderIdentity(
            encoder_name="omiclip", checkpoint_sha256=_VALID_SHA256, pinned_revision=_VALID_REVISION,
            package_version="open_clip_torch-2.24.0", preprocessing_spec="not_the_real_omiclip_spec",
            output_dim=768,
        )

        def encode_available_patches(self, patches):
            return np.zeros((patches.shape[0], 768), dtype=np.float32)

    build_omiclip_spot_feature_cache(cfg_cache_root(cfg), "S0", barcodes, patches, availability, _BadEncoder())

    loaded = load_omiclip_spot_features(cfg_cache_root(cfg), "S0", barcodes, patches, availability)
    assert loaded["provenance"]["preprocessing_spec"] == "not_the_real_omiclip_spec"

    with pytest.raises(ValueError, match="preprocessing_spec"):
        load_gen3_omiclip_spot_features(cfg, "S0", barcodes, patches, availability)


def test_load_gen3_omiclip_spot_features_rejects_a_missing_cache_without_rebuilding(tmp_path):
    cfg = OmegaConf.create({"data": {"hest_data_dir": str(tmp_path / "hest1k")}})
    barcodes, patches, availability = _synthetic_sample()
    with pytest.raises(FileNotFoundError, match="missing"):
        load_gen3_omiclip_spot_features(cfg, "S0", barcodes, patches, availability)


def test_load_gen3_omiclip_spot_features_rejects_a_corrupt_cache_missing_required_fields(tmp_path):
    cfg = OmegaConf.create({"data": {"hest_data_dir": str(tmp_path / "hest1k")}})
    barcodes, patches, availability = _synthetic_sample()
    path = cfg_cache_root(cfg) / "omiclip_gen3_spot_cache" / "S0.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, features=np.zeros((5, 768), dtype=np.float32), barcodes=barcodes)
    with pytest.raises(ValueError, match="missing fields"):
        load_gen3_omiclip_spot_features(cfg, "S0", barcodes, patches, availability)


def test_load_gen3_omiclip_spot_features_rejects_a_barcode_order_mismatch(tmp_path):
    cfg = OmegaConf.create({"data": {"hest_data_dir": str(tmp_path / "hest1k")}})
    barcodes, patches, availability = _synthetic_sample()
    encoder = _RealShapedStubOmiCLIPEncoder()
    build_omiclip_spot_feature_cache(cfg_cache_root(cfg), "S0", barcodes, patches, availability, encoder)

    reordered = barcodes[::-1].copy()
    with pytest.raises(ValueError, match="mismatch"):
        load_gen3_omiclip_spot_features(cfg, "S0", reordered, patches[::-1].copy(), availability[::-1].copy())
