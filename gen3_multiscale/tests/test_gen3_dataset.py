"""Tests for gen3_multiscale/training/gen3_dataset.py -- Step 6's
manifest-backed dataset. Real, small, end-to-end synthetic data
(gen3_multiscale/tests/_step6_fixtures.py) exercising every real module
this dataset wires together: dataset_manifest, example_builder,
spot_feature_cache, slide_context, mask_fingerprint, mask_schedule."""
import numpy as np
import pytest

from gen3_multiscale.tests._step6_fixtures import build_synthetic_gen3_experiment
from gen3_multiscale.training.gen3_dataset import (
    Gen3SampleData, Gen3SpatialFieldDataset, build_gen3_mask_schedule, gen3_identity_collate,
    load_gen3_sample_data,
)

_STRATA = [
    {"name": "small", "radius_range": [1.5, 2.5], "radius_unit": "spot_spacing", "shape": "circle"},
]


def _load_split_samples(cfg, manifest, split):
    ids = manifest[f"{split}_sample_ids"]
    return {sid: load_gen3_sample_data(cfg, manifest, sid) for sid in ids}


def test_load_gen3_sample_data_returns_a_verified_sample(tmp_path, monkeypatch):
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch)
    sample_id = manifest["train_sample_ids"][0]
    sample = load_gen3_sample_data(cfg, manifest, sample_id)
    assert isinstance(sample, Gen3SampleData)
    assert sample.sample_id == sample_id
    assert sample.split == "train"
    n = sample.adata.n_obs
    assert sample.patches.shape[0] == n
    assert sample.image_source_available.shape == (n,)
    assert sample.precomputed_spot_features.shape == (n, 1536)
    assert sample.tile_encoder_provenance["dense_wsi"] is not None
    assert sample.tile_encoder_provenance["spot_features"] is not None
    assert sample.tile_encoder_provenance["dense_wsi"]["hf_revision"] == sample.tile_encoder_provenance["spot_features"]["hf_revision"]


def test_load_gen3_sample_data_rejects_an_unknown_sample_id(tmp_path, monkeypatch):
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="not a sample"):
        load_gen3_sample_data(cfg, manifest, "NOT_A_REAL_SAMPLE")


def test_build_gen3_mask_schedule_for_train_samples_passes_its_own_report(tmp_path, monkeypatch):
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch)
    samples = _load_split_samples(cfg, manifest, "train")
    schedule = build_gen3_mask_schedule(
        manifest, samples, _STRATA, role="train", n_training_masks_per_sample=6,
    )
    assert len(schedule.train_items) == 6 * len(samples)
    for sid in samples:
        assert schedule.reports[sid]["passed"] is True


def test_build_gen3_mask_schedule_for_held_out_samples_passes_its_own_report(tmp_path, monkeypatch):
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch)
    samples = _load_split_samples(cfg, manifest, "validation")
    schedule = build_gen3_mask_schedule(
        manifest, samples, _STRATA, role="validation",
        split_counts={"validation": 3}, split_seeds={"validation": 700_000},
    )
    assert len(schedule.held_out_items) == 3 * len(samples)
    for sid in samples:
        assert schedule.reports[sid]["passed"] is True


def test_build_gen3_mask_schedule_rejects_a_sample_with_the_wrong_split(tmp_path, monkeypatch):
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch)
    train_samples = _load_split_samples(cfg, manifest, "train")
    with pytest.raises(ValueError, match="requested role"):
        build_gen3_mask_schedule(manifest, train_samples, _STRATA, role="validation")


def test_gen3_dataset_train_role_builds_real_examples(tmp_path, monkeypatch):
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch)
    samples = _load_split_samples(cfg, manifest, "train")
    schedule = build_gen3_mask_schedule(manifest, samples, _STRATA, role="train", n_training_masks_per_sample=4)
    dataset = Gen3SpatialFieldDataset(manifest, samples, schedule, _STRATA)
    assert len(dataset) == 4 * len(samples)
    inputs, targets = dataset[0]
    assert inputs.observed_full_gene_expression.shape[1] == len(manifest["gene_panel"])
    assert inputs.observed_gigapath_features.shape[1] == 1536
    assert targets.query_expression.shape[0] == inputs.query_coords.shape[0]


def test_gen3_dataset_held_out_role_is_deterministic_across_epochs(tmp_path, monkeypatch):
    """Mandatory requirement #9: deterministic FIXED-mask validation --
    the same index must produce the identical example every time, not a
    freshly re-sampled one."""
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch)
    samples = _load_split_samples(cfg, manifest, "validation")
    schedule = build_gen3_mask_schedule(
        manifest, samples, _STRATA, role="validation",
        split_counts={"validation": 3}, split_seeds={"validation": 700_000},
    )
    dataset = Gen3SpatialFieldDataset(manifest, samples, schedule, _STRATA)
    inputs_a, _ = dataset[0]
    inputs_b, _ = dataset[0]
    assert np.array_equal(inputs_a.query_barcodes, inputs_b.query_barcodes)
    assert np.array_equal(inputs_a.observed_barcodes, inputs_b.observed_barcodes)
    assert np.allclose(inputs_a.observed_full_gene_expression, inputs_b.observed_full_gene_expression)


def test_gen3_dataset_rejects_mixing_a_sample_from_the_wrong_role(tmp_path, monkeypatch):
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch)
    train_samples = _load_split_samples(cfg, manifest, "train")
    val_samples = _load_split_samples(cfg, manifest, "validation")
    schedule = build_gen3_mask_schedule(manifest, val_samples, _STRATA, role="validation",
                                         split_counts={"validation": 2}, split_seeds={"validation": 700_000})
    mixed = {**train_samples, **val_samples}
    with pytest.raises(ValueError, match="manifest split"):
        Gen3SpatialFieldDataset(manifest, mixed, schedule, _STRATA)


def test_gen3_dataset_never_calls_the_tile_encoder(tmp_path, monkeypatch):
    """Mandatory requirement #5 ('never execute the GigaPath tile encoder
    per training example') -- proven, not just documented: raise if the
    real encoder functions are ever called while building dataset
    items."""
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch)
    samples = _load_split_samples(cfg, manifest, "train")
    schedule = build_gen3_mask_schedule(manifest, samples, _STRATA, role="train", n_training_masks_per_sample=3)
    dataset = Gen3SpatialFieldDataset(manifest, samples, schedule, _STRATA)

    def _fail(*args, **kwargs):
        raise AssertionError("the tile encoder must never be called while building a dataset item")

    monkeypatch.setattr("src.models.conditioning._load_gigapath_tile_encoder", _fail)
    monkeypatch.setattr("src.models.conditioning._gigapath_preprocess_and_encode", _fail)
    for i in range(len(dataset)):
        dataset[i]  # must not raise AssertionError


def test_gen3_identity_collate_requires_batch_size_one():
    assert gen3_identity_collate([("a", "b")]) == ("a", "b")
    with pytest.raises(ValueError, match="batch_size=1"):
        gen3_identity_collate([("a", "b"), ("c", "d")])
