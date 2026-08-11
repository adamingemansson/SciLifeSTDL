"""Tests for gen3_multiscale/training/gen3_dataset.py -- Step 6's
manifest-backed dataset. Real, small, end-to-end synthetic data
(gen3_multiscale/tests/_step6_fixtures.py) exercising every real module
this dataset wires together: dataset_manifest, example_builder,
spot_feature_cache, slide_context, mask_fingerprint, mask_schedule."""
from pathlib import Path

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
    # Adam's Step 6 audit #6 of commit a32051b: the dense-WSI cache is now
    # only loaded for an architecture that actually consumes it
    # (use_regional_he/use_global_slide) -- this test's OWN point is to
    # exercise that loaded path, so it must ask for one explicitly.
    cfg.model = {"params": {"use_regional_he": True}}
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


def test_load_gen3_sample_data_skips_dense_wsi_cache_when_architecture_does_not_consume_it(tmp_path, monkeypatch):
    """Adam's Step 6 audit #6 of commit a32051b: "do not load dense WSI
    caches for architectures that do not consume them." A config with
    neither use_regional_he nor use_global_slide set (or no model.params
    section at all, e.g. Architecture 1/2) must never pay for the dense-
    WSI cache load, even when data.slide_context_source is configured to
    dense_wsi_cache."""
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch)
    sample_id = manifest["train_sample_ids"][0]
    sample = load_gen3_sample_data(cfg, manifest, sample_id)
    assert sample.slide_context_record is None
    assert sample.tile_encoder_provenance["dense_wsi"] is None
    assert sample.tile_encoder_provenance["spot_features"] is not None

    cfg.model = {"params": {"use_regional_he": False, "use_global_slide": False}}
    sample_explicit_false = load_gen3_sample_data(cfg, manifest, sample_id)
    assert sample_explicit_false.tile_encoder_provenance["dense_wsi"] is None


def test_load_gen3_sample_data_explicit_dense_override_beats_inherited_model_flags(tmp_path, monkeypatch):
    """Adapters with a canonical arm table can state the real cache need.

    This prevents a newly selected model from silently inheriting stale
    ``use_global_slide`` flags from the comparison config used to prepare it.
    """
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch)
    sample_id = manifest["train_sample_ids"][0]
    cfg.model = {"params": {"use_regional_he": False, "use_global_slide": False}}
    required = load_gen3_sample_data(
        cfg, manifest, sample_id, require_dense_wsi=True,
    )
    assert required.slide_context_record is not None
    assert required.tile_encoder_provenance["dense_wsi"] is not None

    cfg.model = {"params": {"use_regional_he": True, "use_global_slide": True}}
    forbidden = load_gen3_sample_data(
        cfg, manifest, sample_id, require_dense_wsi=False,
    )
    assert forbidden.slide_context_record is None
    assert forbidden.tile_encoder_provenance["dense_wsi"] is None


def test_load_gen3_sample_data_rejects_an_unknown_sample_id(tmp_path, monkeypatch):
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="not a sample"):
        load_gen3_sample_data(cfg, manifest, "NOT_A_REAL_SAMPLE")


def test_load_gen3_sample_data_rejects_an_h5ad_file_that_changed_since_the_manifest_was_built(tmp_path, monkeypatch):
    """Regression test for a real, confirmed gap (Codex audit of commit
    27e1232): 'a changed h5ad with identical genes/barcodes can pass.'
    Mutates the real h5ad file on disk (same barcodes/genes, different
    expression values) after the manifest was built, and confirms the
    trainer's own load path now catches it via re-hashing."""
    import anndata as ad

    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch)
    sample_id = manifest["train_sample_ids"][0]
    h5ad_path = next(Path(str(cfg.data.hest_data_dir)).glob(f"st/{sample_id}.h5ad"))
    adata = ad.read_h5ad(h5ad_path)
    adata.X = adata.X + 1.0  # same shape/barcodes/genes, different real content
    adata.write_h5ad(h5ad_path)

    with pytest.raises(ValueError, match="SHA256"):
        load_gen3_sample_data(cfg, manifest, sample_id)


def test_gen3_sample_data_rejects_mismatched_precomputed_spot_features_barcodes():
    """Regression test: a bare row-COUNT check cannot catch a caller-side
    mix-up between two samples with the SAME n_spots -- barcode identity
    AND order is the real invariant Gen3SampleData.__post_init__ must
    enforce."""
    import anndata as ad
    import pandas as pd

    barcodes = [f"S0-SPOT{i}-1" for i in range(4)]
    adata = ad.AnnData(
        X=np.zeros((4, 2), dtype=np.float32),
        obs=pd.DataFrame(index=pd.Index(barcodes)), var=pd.DataFrame(index=pd.Index(["G0", "G1"])),
    )
    adata.obsm["spatial"] = np.zeros((4, 2), dtype=np.float64)
    features = np.zeros((4, 1536), dtype=np.float32)
    wrong_barcodes = np.asarray([f"S1-SPOT{i}-1" for i in range(4)])  # same length, wrong identity
    import hashlib
    with pytest.raises(ValueError, match="do not exactly equal"):
        Gen3SampleData(
            sample_id="S0", patient_id="P0", split="train", adata=adata,
            patches=np.zeros((4, 4, 4, 3), dtype=np.uint8), image_source_available=np.ones(4, dtype=bool),
            precomputed_spot_features=features, precomputed_spot_features_barcodes=wrong_barcodes,
            precomputed_spot_features_digest=hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest(),
            full_sample_coords=np.zeros((4, 2), dtype=np.float64), coords3d=np.zeros((4, 3), dtype=np.float64),
            slice_ids=np.full(4, "S0", dtype=object), slide_context_record=None,
            tile_encoder_provenance={"dense_wsi": None, "spot_features": None},
        )


def _sample_kwargs(n=4, **overrides):
    import hashlib
    import anndata as ad
    adata = ad.AnnData(np.zeros((n, 3), dtype=np.float32))
    adata.obs_names = [f"S0-SPOT{i}-1" for i in range(n)]
    adata.obsm["spatial"] = np.arange(2 * n, dtype=np.float64).reshape(n, 2)
    features = np.zeros((n, 1536), dtype=np.float32)
    kwargs = dict(
        sample_id="S0", patient_id="P0", split="train", adata=adata,
        patches=np.zeros((n, 4, 4, 3), dtype=np.uint8),
        image_source_available=np.ones(n, dtype=bool),
        precomputed_spot_features=features,
        precomputed_spot_features_barcodes=np.asarray(adata.obs_names, dtype=str),
        precomputed_spot_features_digest=hashlib.sha256(
            np.ascontiguousarray(features).tobytes()
        ).hexdigest(),
        full_sample_coords=adata.obsm["spatial"],
        coords3d=np.zeros((n, 3), dtype=np.float64),
        slice_ids=np.full(n, "S0", dtype=object), slide_context_record=None,
        tile_encoder_provenance={"dense_wsi": None, "spot_features": None},
    )
    kwargs.update(overrides)
    return kwargs


def test_n_patch_rows_is_derived_from_patches_when_not_supplied():
    assert Gen3SampleData(**_sample_kwargs(n=5)).n_patch_rows == 5


def test_released_patches_keep_the_real_row_count_for_the_alignment_check():
    """data.retain_patches_in_memory=false drops ~376 MB/slide of pixels
    AFTER the spot-feature cache has re-hashed them, but the row count that
    guards patch/adata alignment must survive the release."""
    sample = Gen3SampleData(**_sample_kwargs(n=6, patches=None, n_patch_rows=6))
    assert sample.patches is None
    assert sample.n_patch_rows == 6


def test_releasing_patches_without_a_row_count_is_refused():
    with pytest.raises(ValueError, match="n_patch_rows is required"):
        Gen3SampleData(**_sample_kwargs(patches=None))


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
    boundary_report = dataset.validate_boundary_schedule()
    assert boundary_report == {
        "role": "validation", "n_items_checked": len(dataset), "passed": True,
    }
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


def test_build_gen3_mask_schedule_gives_different_samples_different_raw_query_index_schedules(tmp_path, monkeypatch):
    """Regression test for a real, confirmed gap (Codex audit of commit
    27e1232): the trainer used to hardcode base_seed=0 for EVERY training
    sample -- two samples with an IDENTICAL regular lattice (this
    fixture's own samples all share the exact same n_side x n_side grid
    and spacing) would then draw their first item's raw query positions
    from the SAME seed, risking identical realized schedules across
    DIFFERENT samples. Compares realized query POSITIONS (index within
    each sample's own obs_names, not the barcode string itself, which
    trivially differs by sample_id prefix) for the first training item of
    two different samples."""
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch)
    samples = _load_split_samples(cfg, manifest, "train")
    sample_ids = sorted(samples)
    assert len(sample_ids) >= 2
    schedule = build_gen3_mask_schedule(manifest, samples, _STRATA, role="train", n_training_masks_per_sample=4)
    dataset = Gen3SpatialFieldDataset(manifest, samples, schedule, _STRATA)

    first_item_idx_by_sample = {}
    for idx, item in enumerate(schedule.train_items):
        if item.sample_id not in first_item_idx_by_sample:
            first_item_idx_by_sample[item.sample_id] = idx

    query_positions_by_sample = {}
    for sid in sample_ids[:2]:
        obs_names = np.asarray(samples[sid].adata.obs_names, dtype=str)
        item = schedule.train_items[first_item_idx_by_sample[sid]]
        _, query_obs_names = dataset._resolve_barcodes(item)
        query_positions_by_sample[sid] = sorted(np.flatnonzero(np.isin(obs_names, query_obs_names)).tolist())

    assert query_positions_by_sample[sample_ids[0]] != query_positions_by_sample[sample_ids[1]]


def test_sample_seed_namespace_is_stable_and_differs_across_samples_and_salts():
    from gen3_multiscale.training.gen3_dataset import sample_seed_namespace

    assert sample_seed_namespace("S0", salt="train") == sample_seed_namespace("S0", salt="train")
    assert sample_seed_namespace("S0", salt="train") != sample_seed_namespace("S1", salt="train")
    assert sample_seed_namespace("S0", salt="train") != sample_seed_namespace("S0", salt="validation")


def test_gen3_identity_collate_requires_batch_size_one():
    assert gen3_identity_collate([("a", "b")]) == ("a", "b")
    with pytest.raises(ValueError, match="batch_size=1"):
        gen3_identity_collate([("a", "b"), ("c", "d")])
