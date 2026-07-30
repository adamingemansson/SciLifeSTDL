"""Integration audit item 2: Gen4SpatialFieldDataset real-path tests --
built on the SAME real, small, end-to-end synthetic Gen3 experiment every
other Gen3 dataset test uses (`tests/_step6_fixtures.py::
build_synthetic_gen3_experiment`), with a real UNI2 spot cache, a real
UNI2 dense-WSI cache, and a real scFoundation cache layered on top --
never mocked cache-loading functions, only stubbed ENCODERS (matching
`tests/_gen4_fixtures.py`'s own StubUNI2Encoder/StubSCFoundationEncoder,
the established discipline for the two real external dependencies not
installed in this sandbox).

Verifies the user's own per-arm wiring spec against the real
`gen4.model_factory.ARM_TABLE`:
  gen4a/gen4c: UNI2 spot features (SWAPS OUT Gen3's own GigaPath cache)
    + UNI2 dense-WSI context, tagged wsi_tile_feature_provenance="uni2".
  gen4c only: scFoundation gex_context_embedding on top.
  gen4b: Gen3's own GigaPath precomputed_spot_features REUSED UNCHANGED
    + scFoundation gex_context_embedding.
  gen4d: Gen3's own GigaPath precomputed_spot_features reused unchanged
    (structurally required placeholder, real image path is live STPath)
    + the sample's real manifest organ.
  gen4e: Gen3's own GigaPath precomputed_spot_features reused unchanged
    + a SEPARATE UNI2 spot-feature lookup (observed_uni2_features)
    + scFoundation gex_context_embedding + organ."""
from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from gen3_multiscale.gen4 import scfoundation_cache, uni2_spot_cache
from gen3_multiscale.gen4.dataset_adapter import Gen4SpatialFieldDataset
from gen3_multiscale.tests._gen4_fixtures import StubSCFoundationEncoder, StubUNI2Encoder
from gen3_multiscale.tests._step6_fixtures import build_synthetic_gen3_experiment
from gen3_multiscale.training.gen3_dataset import build_gen3_mask_schedule, load_gen3_sample_data

_STRATA = [
    {"name": "small", "radius_range": [1.5, 2.5], "radius_unit": "spot_spacing", "shape": "circle"},
]


def _build_experiment(tmp_path, monkeypatch):
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch, n_side=6, spacing=300.0, n_genes=6)
    train_ids = manifest["train_sample_ids"]
    samples = {sid: load_gen3_sample_data(cfg, manifest, sid) for sid in train_ids}
    return cfg, manifest, samples


def _write_uni2_caches(cfg, samples, *, output_dim=8):
    cache_root = str(cfg.data.hest_cache_dir)
    encoder = StubUNI2Encoder(output_dim=output_dim)
    for sample_id, sample in samples.items():
        obs_names = np.asarray(sample.adata.obs_names, dtype=str)
        uni2_spot_cache.build_uni2_spot_feature_cache(
            cache_root, sample_id, obs_names, sample.patches, sample.image_source_available, encoder,
        )
        dense_dir = cfg.data.hest_cache_dir + "/uni2_dense_wsi_cache"
        import os

        os.makedirs(dense_dir, exist_ok=True)
        coords = sample.full_sample_coords.astype(np.float32)
        features = np.random.default_rng(1).normal(size=(coords.shape[0], output_dim)).astype(np.float32)
        np.savez(
            f"{dense_dir}/{sample_id}.npz",
            features=features, coords=coords, level0_coords=coords,
            tile_size=np.asarray(256.0, dtype=np.float32), level0_tile_size=np.asarray(256.0, dtype=np.float32),
            coords_are_centers=np.asarray(True), wsi_dimensions=np.asarray([4000.0, 4000.0], dtype=np.float64),
            uni2_checkpoint_sha256=np.asarray(encoder.identity.checkpoint_sha256),
            uni2_pinned_revision=np.asarray(encoder.identity.pinned_revision),
            uni2_package_version=np.asarray(encoder.identity.package_version),
            uni2_preprocessing_spec=np.asarray(encoder.identity.preprocessing_spec),
            uni2_output_dim=np.asarray(output_dim), uni2_schema_version=np.asarray(1),
        )


def _write_scfoundation_cache(cfg, manifest, samples, *, output_dim=12):
    from gen3_multiscale.data.dataset_manifest import gene_panel_hash

    cache_root = str(cfg.data.hest_cache_dir)
    gene_names = list(manifest["gene_panel"])
    encoder = StubSCFoundationEncoder(gene_names, output_dim=output_dim)
    for sample_id, sample in samples.items():
        obs_names = np.asarray(sample.adata.obs_names, dtype=str)
        expression = np.asarray(
            sample.adata.X.toarray() if hasattr(sample.adata.X, "toarray") else sample.adata.X, dtype=np.float32,
        )
        raw_library_size = np.asarray(sample.adata.obs["_scilifestdl_raw_library_size"], dtype=np.float32)
        scfoundation_cache.build_scfoundation_spot_feature_cache(
            cache_root, sample_id, obs_names, expression, gene_panel_hash(gene_names), encoder,
            raw_library_size=raw_library_size,
        )


def _dataset(config: dict, cfg, manifest, samples, *, n_masks=3):
    schedule = build_gen3_mask_schedule(manifest, samples, _STRATA, role="train", n_training_masks_per_sample=n_masks)
    config_om = OmegaConf.merge(cfg, OmegaConf.create(config))
    return Gen4SpatialFieldDataset(manifest, samples, schedule, _STRATA, config_om, manifest["gene_panel"])


def test_gen4a_swaps_uni2_spot_features_and_uni2_dense_context(tmp_path, monkeypatch):
    cfg, manifest, samples = _build_experiment(tmp_path, monkeypatch)
    _write_uni2_caches(cfg, samples, output_dim=8)
    dataset = _dataset({"model": {"arm": "gen4a", "kind": "conditioner"}}, cfg, manifest, samples)

    inputs, targets = dataset[0]
    sample = samples[dataset._items[0].sample_id]
    gigapath_features = sample.precomputed_spot_features
    # The context row-selected UNI2 feature must NOT equal the GigaPath
    # feature at the same barcode -- a real swap happened, not a
    # coincidental pass-through.
    assert inputs.observed_gigapath_features.shape[1] == 8  # UNI2's own output_dim, not GigaPath's 1536
    assert not np.allclose(inputs.observed_gigapath_features.mean(), gigapath_features.mean())
    assert inputs.wsi_tile_feature_provenance == "uni2"
    assert inputs.context_gex_embedding is None  # arm A has no frozen GEX-context provider


def test_gen4c_attaches_scfoundation_context_on_top_of_uni2_swap(tmp_path, monkeypatch):
    cfg, manifest, samples = _build_experiment(tmp_path, monkeypatch)
    _write_uni2_caches(cfg, samples, output_dim=8)
    _write_scfoundation_cache(cfg, manifest, samples, output_dim=12)
    dataset = _dataset({"model": {"arm": "gen4c", "kind": "conditioner"}}, cfg, manifest, samples)

    inputs, targets = dataset[0]
    assert inputs.observed_gigapath_features.shape[1] == 8
    assert inputs.wsi_tile_feature_provenance == "uni2"
    assert inputs.context_gex_embedding is not None
    assert inputs.context_gex_embedding.shape == (inputs.observed_coords.shape[0], 12)


def test_gen4b_reuses_gigapath_features_unchanged_and_attaches_scfoundation(tmp_path, monkeypatch):
    cfg, manifest, samples = _build_experiment(tmp_path, monkeypatch)
    _write_scfoundation_cache(cfg, manifest, samples, output_dim=12)
    dataset = _dataset({"model": {"arm": "gen4b", "kind": "conditioner"}}, cfg, manifest, samples)

    inputs, targets = dataset[0]
    sample = samples[dataset._items[0].sample_id]
    obs_names = np.asarray(sample.adata.obs_names, dtype=str)
    barcode_to_row = {b: sample.precomputed_spot_features[i] for i, b in enumerate(obs_names)}
    expected = np.stack([barcode_to_row[str(b)] for b in inputs.observed_barcodes])
    np.testing.assert_array_equal(inputs.observed_gigapath_features, expected)
    assert inputs.wsi_tile_feature_provenance is None
    assert inputs.context_gex_embedding is not None


def test_gen4d_attaches_sample_organ_and_reuses_gigapath_placeholder(tmp_path, monkeypatch):
    cfg, manifest, samples = _build_experiment(tmp_path, monkeypatch)
    dataset = _dataset({"model": {"arm": "gen4d", "kind": "conditioner"}}, cfg, manifest, samples)

    inputs, targets = dataset[0]
    sample = samples[dataset._items[0].sample_id]
    assert inputs.sample_organ == manifest["samples"][sample.sample_id]["organ"]
    assert inputs.observed_gigapath_features.shape[1] == 1536  # Gen3's own GigaPath width, unchanged
    assert inputs.observed_uni2_features is None
    assert inputs.context_gex_embedding is None


def test_gen4e_attaches_hybrid_uni2_scfoundation_and_organ(tmp_path, monkeypatch):
    cfg, manifest, samples = _build_experiment(tmp_path, monkeypatch)
    _write_uni2_caches(cfg, samples, output_dim=8)
    _write_scfoundation_cache(cfg, manifest, samples, output_dim=12)
    dataset = _dataset({"model": {"arm": "gen4e", "kind": "conditioner"}}, cfg, manifest, samples)

    inputs, targets = dataset[0]
    sample = samples[dataset._items[0].sample_id]
    assert inputs.observed_gigapath_features.shape[1] == 1536  # arm 4's STPath placeholder, reused GigaPath cache
    assert inputs.observed_uni2_features is not None
    assert inputs.observed_uni2_features.shape == (inputs.observed_coords.shape[0], 8)
    assert inputs.context_gex_embedding is not None
    assert inputs.sample_organ == manifest["samples"][sample.sample_id]["organ"]


def test_gen4_dataset_adapter_rejects_an_unknown_arm(tmp_path, monkeypatch):
    cfg, manifest, samples = _build_experiment(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        _dataset({"model": {"arm": "not_a_real_arm", "kind": "conditioner"}}, cfg, manifest, samples)
