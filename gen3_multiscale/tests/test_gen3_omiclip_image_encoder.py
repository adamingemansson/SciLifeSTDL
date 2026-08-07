"""Real, end-to-end tests for the `data.image_encoder="omiclip"` path added
across gen3_dataset.py, gen3_preflight.py, training/train.py, and
conditional_wae/contract.py -- mirrors test_gen3_uni2_image_encoder.py
exactly, adapted to OmiCLIP's own field shapes (output_dim=768,
coca_ViT-L-14 preprocessing spec)."""
import copy

import numpy as np
import pytest

from gen3_multiscale.data import example_builder
from gen3_multiscale.gen4.providers import EncoderIdentity
from gen3_multiscale.gen4.omiclip_spot_cache import build_omiclip_spot_feature_cache, cfg_cache_root
from gen3_multiscale.tests._step6_fixtures import build_synthetic_gen3_experiment
from gen3_multiscale.training import train
from gen3_multiscale.training.gen3_dataset import load_gen3_sample_data
from gen3_multiscale.training.gen3_preflight import load_and_preflight_samples

_VALID_REVISION = "d072f48609bec7ec4d2c43889262b3029bb1279f"  # 40 lowercase hex chars
_VALID_SHA256 = "a" * 64
_EXPECTED_SPEC = "omiclip_tile_v1:coca_ViT-L-14:open_clip_eval_transform"


class _RealShapedStubOmiCLIPEncoder:
    def __init__(self):
        self.identity = EncoderIdentity(
            encoder_name="omiclip", checkpoint_sha256=_VALID_SHA256, pinned_revision=_VALID_REVISION,
            package_version="open_clip_torch-2.24.0", preprocessing_spec=_EXPECTED_SPEC, output_dim=768,
        )

    def encode_available_patches(self, patches: np.ndarray) -> np.ndarray:
        means = patches.astype(np.float32).reshape(patches.shape[0], -1).mean(axis=1, keepdims=True)
        return np.tile(means, (1, 768)).astype(np.float32)


def _omiclip_cfg(cfg, manifest):
    omiclip_cfg = copy.deepcopy(cfg)
    omiclip_cfg.data.image_encoder = "omiclip"
    encoder = _RealShapedStubOmiCLIPEncoder()
    cache_root = cfg_cache_root(omiclip_cfg)
    for sample_id in manifest["samples"]:
        adata, patches, image_source_available = example_builder.load_sample_for_examples(manifest, sample_id)
        build_omiclip_spot_feature_cache(
            cache_root, sample_id, np.asarray(adata.obs_names), patches, image_source_available, encoder,
        )
    return omiclip_cfg


def test_load_gen3_sample_data_uses_the_omiclip_cache_when_image_encoder_is_omiclip(tmp_path, monkeypatch):
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch)
    omiclip_cfg = _omiclip_cfg(cfg, manifest)
    sample_id = next(iter(manifest["samples"]))

    sample = load_gen3_sample_data(omiclip_cfg, manifest, sample_id)
    assert sample.precomputed_spot_features.shape[1] == 768
    provenance = sample.tile_encoder_provenance["spot_features"]
    assert provenance["pinned_revision"] == _VALID_REVISION
    assert "hf_repo_id" not in provenance  # OmiCLIP-shaped, not GigaPath-shaped


def test_load_gen3_sample_data_rejects_an_unknown_image_encoder(tmp_path, monkeypatch):
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch)
    cfg.data.image_encoder = "not-a-real-encoder"
    sample_id = next(iter(manifest["samples"]))
    with pytest.raises(ValueError, match="image_encoder"):
        load_gen3_sample_data(cfg, manifest, sample_id)


def test_expected_tile_encoder_provenance_omiclip_path_requires_pinned_revision():
    config = {"data": {"image_encoder": "omiclip"}}
    with pytest.raises(ValueError, match="omiclip_pinned_revision"):
        train.expected_tile_encoder_provenance(config)


def test_expected_tile_encoder_provenance_omiclip_path_returns_pinned_revision():
    config = {"data": {"image_encoder": "omiclip", "omiclip_pinned_revision": _VALID_REVISION}}
    assert train.expected_tile_encoder_provenance(config) == {
        "pinned_revision": _VALID_REVISION, "schema_version": 1,
    }


def test_expected_tile_encoder_provenance_rejects_an_unknown_image_encoder():
    with pytest.raises(ValueError, match="image_encoder"):
        train.expected_tile_encoder_provenance({"data": {"image_encoder": "bogus"}})


def test_load_and_preflight_samples_passes_end_to_end_for_a_real_omiclip_experiment(tmp_path, monkeypatch):
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch)
    omiclip_cfg = _omiclip_cfg(cfg, manifest)
    sample_ids = list(manifest["samples"])

    samples, report = load_and_preflight_samples(
        omiclip_cfg, manifest, sample_ids, {"pinned_revision": _VALID_REVISION},
    )
    assert set(samples) == set(sample_ids)
    assert report["passed"] is True


def test_load_and_preflight_samples_rejects_omiclip_combined_with_dense_wsi(tmp_path, monkeypatch):
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch)
    omiclip_cfg = _omiclip_cfg(cfg, manifest)
    omiclip_cfg.model = {"params": {"use_global_slide": True}}
    sample_ids = list(manifest["samples"])
    with pytest.raises(ValueError, match="use_regional_he/use_global_slide"):
        load_and_preflight_samples(omiclip_cfg, manifest, sample_ids, {"pinned_revision": _VALID_REVISION})
