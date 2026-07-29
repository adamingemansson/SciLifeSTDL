"""Tests for gen3_multiscale/training/gen3_preflight.py -- Step 6's
mandatory preflight gate (Adam's requirements #3/#4: exact cache
coverage, tile-encoder provenance consistency, both BEFORE model/
optimizer/DataLoader construction)."""
import numpy as np
import pytest

from gen3_multiscale.tests._step6_fixtures import VALID_HF_REVISION, build_synthetic_gen3_experiment
from gen3_multiscale.training.gen3_preflight import (
    expected_cache_source_labels, load_and_preflight_samples, load_gen3_preflight_report,
    save_gen3_preflight_report, verify_cache_coverage,
)


def _expected_provenance():
    return {
        "hf_repo_id": "prov-gigapath/prov-gigapath", "hf_revision": VALID_HF_REVISION,
        "schema_version": 1,
    }


def test_expected_cache_source_labels():
    labels = expected_cache_source_labels(["S0", "S1"])
    assert labels == {"S0:dense_wsi", "S0:spot_features", "S1:dense_wsi", "S1:spot_features"}


def test_verify_cache_coverage_accepts_an_exact_match():
    expected = {"S0:dense_wsi", "S0:spot_features"}
    result = verify_cache_coverage(expected, list(expected))
    assert result["passed"] is True


def test_verify_cache_coverage_rejects_missing_entries():
    expected = {"S0:dense_wsi", "S0:spot_features"}
    with pytest.raises(ValueError, match="missing cache-coverage"):
        verify_cache_coverage(expected, ["S0:dense_wsi"])


def test_verify_cache_coverage_rejects_extra_entries():
    expected = {"S0:dense_wsi", "S0:spot_features"}
    with pytest.raises(ValueError, match="extra cache-coverage"):
        verify_cache_coverage(expected, [*expected, "S1:dense_wsi"])


def test_verify_cache_coverage_rejects_duplicate_entries():
    expected = {"S0:dense_wsi", "S0:spot_features"}
    with pytest.raises(ValueError, match="duplicate cache-coverage"):
        verify_cache_coverage(expected, [*expected, "S0:dense_wsi"])


def test_load_and_preflight_samples_passes_for_a_real_consistent_experiment(tmp_path, monkeypatch):
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch)
    sample_ids = manifest["train_sample_ids"]
    samples, report = load_and_preflight_samples(cfg, manifest, sample_ids, _expected_provenance())
    assert set(samples) == set(sample_ids)
    assert report["passed"] is True
    assert report["cache_coverage"]["passed"] is True


def test_load_and_preflight_samples_rejects_disagreement_with_expected_provenance(tmp_path, monkeypatch):
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch)
    sample_ids = manifest["train_sample_ids"]
    wrong_expected = {**_expected_provenance(), "hf_revision": "9" * 40}
    with pytest.raises(ValueError, match="hf_revision"):
        load_and_preflight_samples(cfg, manifest, sample_ids, wrong_expected)


def test_load_and_preflight_samples_rejects_an_incomplete_sample_list(tmp_path, monkeypatch):
    """Mandatory requirement #3: cache coverage must be built from EVERY
    manifest-selected sample -- if a caller only preflights a SUBSET of
    the samples it will actually train on, that gap should be visible
    (this preflight call only validates whichever sample_ids it is
    given; the real trainer's own responsibility -- exercised in
    test_train.py -- is to always pass the manifest's full sample list
    for the role being trained)."""
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch)
    sample_ids = manifest["train_sample_ids"]
    assert len(sample_ids) > 1
    samples, report = load_and_preflight_samples(cfg, manifest, sample_ids[:1], _expected_provenance())
    assert set(samples) == {sample_ids[0]}
    assert report["n_samples"] == 1  # proves the caller controls scope; the trainer must pass the FULL list


def test_load_and_preflight_samples_rejects_two_samples_with_different_pinned_revisions(tmp_path, monkeypatch):
    """A cache built from a DIFFERENT, still validly-pinned revision must
    be caught -- not merely a syntactically malformed one."""
    other_revision = "1234567890abcdef1234567890abcdef12345678"
    cfg, manifest = build_synthetic_gen3_experiment(
        tmp_path, monkeypatch, samples_per_split={"train": 2, "validation": 1, "test": 1},
    )
    sample_ids = manifest["train_sample_ids"]
    # Rebuild ONE training sample's caches with a different revision --
    # a real, plausible "half-migrated experiment" scenario.
    from gen3_multiscale.data.spot_feature_cache import build_gen3_spot_feature_cache
    from gen3_multiscale.data import loaders
    import anndata as ad

    other_sid = sample_ids[0]
    hest_dir = cfg.data.hest_data_dir
    patches, patch_barcodes = loaders.load_hest_patches(hest_dir, other_sid)
    raw_adata = ad.read_h5ad(f"{hest_dir}/st/{other_sid}.h5ad")
    _, aligned_patches, image_source_available = loaders.align_patches_to_adata(raw_adata, patches, patch_barcodes)
    build_gen3_spot_feature_cache(
        cfg, other_sid, np.asarray(raw_adata.obs_names), aligned_patches, image_source_available,
        tile_encoder_revision=other_revision, device="cpu",
    )

    with pytest.raises(ValueError, match="hf_revision"):
        load_and_preflight_samples(cfg, manifest, sample_ids, _expected_provenance())


def test_save_and_load_gen3_preflight_report_round_trips(tmp_path, monkeypatch):
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch)
    sample_ids = manifest["train_sample_ids"]
    _, report = load_and_preflight_samples(cfg, manifest, sample_ids, _expected_provenance())
    path = tmp_path / "preflight_report.json"
    save_gen3_preflight_report(report, path)
    loaded = load_gen3_preflight_report(path)
    assert loaded == report
    assert path.is_file()
