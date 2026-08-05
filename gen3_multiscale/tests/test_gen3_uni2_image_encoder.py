"""Real, end-to-end tests for the `data.image_encoder="uni2"` path added
across gen3_dataset.py, gen3_preflight.py, training/train.py, and
conditional_wae/contract.py -- the UNI2 analogue of GigaPath's existing
Gen3 spot-feature-cache discipline. Builds on the SAME real synthetic
experiment `test_gen3_dataset.py`/`test_gen3_preflight.py` already use
(`_step6_fixtures.build_synthetic_gen3_experiment`), so this exercises
real `load_gen3_sample_data`/`load_and_preflight_samples` codepaths, not
a reimplemented shortcut."""
import copy

import numpy as np
import pytest

from gen3_multiscale.data import example_builder
from gen3_multiscale.gen4.providers import EncoderIdentity
from gen3_multiscale.gen4.uni2_spot_cache import build_uni2_spot_feature_cache, cfg_cache_root
from gen3_multiscale.tests._step6_fixtures import build_synthetic_gen3_experiment
from gen3_multiscale.training import train
from gen3_multiscale.training.gen3_dataset import load_gen3_sample_data
from gen3_multiscale.training.gen3_preflight import load_and_preflight_samples

_VALID_REVISION = "d072f48609bec7ec4d2c43889262b3029bb1279f"  # 40 lowercase hex chars
_VALID_SHA256 = "a" * 64
_EXPECTED_SPEC = "uni2_tile_v2:vit_giant_patch14_224:resize224_bicubic_antialias:imagenet_norm"


class _RealShapedStubUNI2Encoder:
    """Deterministic ImageContextProvider stand-in whose `.identity`
    fields are real-UNI2-shaped -- satisfies validate_uni2_tile_encoder_
    provenance, unlike `_gen4_fixtures.py::StubUNI2Encoder`."""

    def __init__(self):
        self.identity = EncoderIdentity(
            encoder_name="uni2", checkpoint_sha256=_VALID_SHA256, pinned_revision=_VALID_REVISION,
            package_version="timm-1.0.0", preprocessing_spec=_EXPECTED_SPEC, output_dim=1536,
        )

    def encode_available_patches(self, patches: np.ndarray) -> np.ndarray:
        means = patches.astype(np.float32).reshape(patches.shape[0], -1).mean(axis=1, keepdims=True)
        return np.tile(means, (1, 1536)).astype(np.float32)


def _uni2_cfg(cfg, manifest):
    """A real gigapath-built synthetic experiment's cfg, switched to
    image_encoder="uni2" and with a real UNI2 spot-feature cache built
    for every sample from the SAME real patches/adata the gigapath cache
    was built from."""
    uni2_cfg = copy.deepcopy(cfg)
    uni2_cfg.data.image_encoder = "uni2"
    encoder = _RealShapedStubUNI2Encoder()
    cache_root = cfg_cache_root(uni2_cfg)
    for sample_id in manifest["samples"]:
        adata, patches, image_source_available = example_builder.load_sample_for_examples(manifest, sample_id)
        build_uni2_spot_feature_cache(
            cache_root, sample_id, np.asarray(adata.obs_names), patches, image_source_available, encoder,
        )
    return uni2_cfg


def test_load_gen3_sample_data_uses_the_uni2_cache_when_image_encoder_is_uni2(tmp_path, monkeypatch):
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch)
    uni2_cfg = _uni2_cfg(cfg, manifest)
    sample_id = next(iter(manifest["samples"]))

    sample = load_gen3_sample_data(uni2_cfg, manifest, sample_id)
    assert sample.precomputed_spot_features.shape[1] == 1536
    provenance = sample.tile_encoder_provenance["spot_features"]
    assert provenance["pinned_revision"] == _VALID_REVISION
    assert "hf_repo_id" not in provenance  # UNI2-shaped, not GigaPath-shaped


def test_load_gen3_sample_data_defaults_to_gigapath_when_image_encoder_is_unset(tmp_path, monkeypatch):
    """The default codepath (every existing config that never sets
    data.image_encoder) must be completely unaffected."""
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch)
    assert "image_encoder" not in cfg.data
    sample_id = next(iter(manifest["samples"]))

    sample = load_gen3_sample_data(cfg, manifest, sample_id)
    assert "hf_repo_id" in sample.tile_encoder_provenance["spot_features"]


def test_load_gen3_sample_data_rejects_an_unknown_image_encoder(tmp_path, monkeypatch):
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch)
    cfg.data.image_encoder = "not-a-real-encoder"
    sample_id = next(iter(manifest["samples"]))
    with pytest.raises(ValueError, match="image_encoder"):
        load_gen3_sample_data(cfg, manifest, sample_id)


def test_expected_tile_encoder_provenance_uni2_path_requires_pinned_revision():
    config = {"data": {"image_encoder": "uni2"}}
    with pytest.raises(ValueError, match="uni2_pinned_revision"):
        train.expected_tile_encoder_provenance(config)


def test_expected_tile_encoder_provenance_uni2_path_returns_pinned_revision():
    config = {"data": {"image_encoder": "uni2", "uni2_pinned_revision": _VALID_REVISION}}
    assert train.expected_tile_encoder_provenance(config) == {
        "pinned_revision": _VALID_REVISION, "schema_version": 1,
    }


def test_expected_tile_encoder_provenance_still_defaults_to_gigapath():
    config = {"data": {"tile_encoder_revision": "d072f48609bec7ec4d2c43889262b3029bb1279f"}}
    provenance = train.expected_tile_encoder_provenance(config)
    assert provenance["hf_repo_id"] == "prov-gigapath/prov-gigapath"


def test_expected_tile_encoder_provenance_rejects_an_unknown_image_encoder():
    with pytest.raises(ValueError, match="image_encoder"):
        train.expected_tile_encoder_provenance({"data": {"image_encoder": "bogus"}})


def test_load_and_preflight_samples_passes_end_to_end_for_a_real_uni2_experiment(tmp_path, monkeypatch):
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch)
    uni2_cfg = _uni2_cfg(cfg, manifest)
    sample_ids = list(manifest["samples"])

    samples, report = load_and_preflight_samples(
        uni2_cfg, manifest, sample_ids, {"pinned_revision": _VALID_REVISION},
    )
    assert set(samples) == set(sample_ids)
    assert report["passed"] is True


def test_load_and_preflight_samples_rejects_uni2_combined_with_dense_wsi(tmp_path, monkeypatch):
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch)
    uni2_cfg = _uni2_cfg(cfg, manifest)
    uni2_cfg.model = {"params": {"use_global_slide": True}}
    sample_ids = list(manifest["samples"])
    with pytest.raises(ValueError, match="use_regional_he/use_global_slide"):
        load_and_preflight_samples(uni2_cfg, manifest, sample_ids, {"pinned_revision": _VALID_REVISION})
