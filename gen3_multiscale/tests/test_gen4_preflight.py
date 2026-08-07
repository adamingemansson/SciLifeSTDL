"""Static preflight/audit tests -- GEN4_CONTRACT.md section 9/12."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from gen3_multiscale.gen4.preflight import (
    audit_gen4_manifest_cache_coverage, audit_scfoundation_cache_matches_config,
    audit_uni2_cache_matches_config, static_audit_gen4_config,
)
from gen3_multiscale.gen4.scfoundation_cache import build_scfoundation_spot_feature_cache
from gen3_multiscale.gen4.uni2_spot_cache import build_uni2_spot_feature_cache
from gen3_multiscale.tests._gen4_fixtures import StubSCFoundationEncoder, StubUNI2Encoder

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "gen4"


def test_static_audit_rejects_unknown_arm():
    config = yaml.safe_load((_CONFIG_DIR / "gen4a_conditioner.yaml").read_text())
    config["model"]["arm"] = "not_a_real_arm"
    with pytest.raises(ValueError, match="model.arm"):
        static_audit_gen4_config(config)


def test_static_audit_rejects_missing_required_fingerprint_key():
    config = yaml.safe_load((_CONFIG_DIR / "gen4b_conditioner.yaml").read_text())
    del config["required_fingerprints"]["scfoundation_checkpoint"]
    with pytest.raises(ValueError, match="required_fingerprints"):
        static_audit_gen4_config(config)


def test_static_audit_rejects_empty_masking_strata():
    config = yaml.safe_load((_CONFIG_DIR / "gen4a_conditioner.yaml").read_text())
    config["masking"]["strata"] = []
    with pytest.raises(ValueError, match="masking.strata"):
        static_audit_gen4_config(config)


def test_static_audit_flow_config_requires_basis_and_conditioner_checkpoint_keys():
    config = yaml.safe_load((_CONFIG_DIR / "gen4a_flow.yaml").read_text())
    del config["required_fingerprints"]["gene_residual_basis"]
    with pytest.raises(ValueError, match="gene_residual_basis"):
        static_audit_gen4_config(config)


def test_uni2_cache_audit_reports_missing_cache_without_raising(tmp_path):
    config = yaml.safe_load((_CONFIG_DIR / "gen4a_conditioner.yaml").read_text())
    report = audit_uni2_cache_matches_config(tmp_path, "no-such-sample", config)
    assert report["cache_exists"] is False


def test_uni2_cache_audit_fails_closed_on_dim_mismatch(tmp_path):
    config = yaml.safe_load((_CONFIG_DIR / "gen4a_conditioner.yaml").read_text())
    config["model"]["params"]["image_feature_dim"] = 999  # deliberately does not match the built cache
    encoder = StubUNI2Encoder(output_dim=6)
    barcodes = np.array(["a", "b"])
    patches = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    availability = np.array([True, True])
    build_uni2_spot_feature_cache(tmp_path, "s1", barcodes, patches, availability, encoder)
    with pytest.raises(ValueError, match="does not match config"):
        audit_uni2_cache_matches_config(tmp_path, "s1", config)


def test_scfoundation_cache_audit_fails_closed_on_dim_mismatch(tmp_path):
    config = yaml.safe_load((_CONFIG_DIR / "gen4b_conditioner.yaml").read_text())
    config["model"]["params"]["gex_context_embedding_dim"] = 999
    gene_names = [f"g{i}" for i in range(4)]
    encoder = StubSCFoundationEncoder(gene_names, output_dim=7)
    barcodes = np.array(["a", "b"])
    expression = np.zeros((2, 4), dtype=np.float32)
    build_scfoundation_spot_feature_cache(tmp_path, "s1", barcodes, expression, "hash123", encoder)
    with pytest.raises(ValueError, match="does not match config"):
        audit_scfoundation_cache_matches_config(tmp_path, "s1", config)


# Item 2 (six-launch-blocker audit): "Make preflight require exact
# manifest-derived cache coverage and compare cache checkpoint/revision/
# vocabulary/preprocessing identities against the resolved experiment.
# Missing or wrong-modality caches must fail before model construction."

def test_uni2_cache_audit_require_exists_raises_on_missing_cache(tmp_path):
    config = yaml.safe_load((_CONFIG_DIR / "gen4a_conditioner.yaml").read_text())
    with pytest.raises(FileNotFoundError, match="UNI2 spot-feature cache required"):
        audit_uni2_cache_matches_config(tmp_path, "no-such-sample", config, require_exists=True)


def test_scfoundation_cache_audit_require_exists_raises_on_missing_cache(tmp_path):
    config = yaml.safe_load((_CONFIG_DIR / "gen4b_conditioner.yaml").read_text())
    with pytest.raises(FileNotFoundError, match="scFoundation spot-feature cache required"):
        audit_scfoundation_cache_matches_config(tmp_path, "no-such-sample", config, require_exists=True)


def test_uni2_cache_audit_fails_closed_on_preprocessing_spec_mismatch(tmp_path):
    config = yaml.safe_load((_CONFIG_DIR / "gen4a_conditioner.yaml").read_text())
    config["required_fingerprints"]["uni2_preprocessing_spec"] = "expected_spec_v2"
    encoder = StubUNI2Encoder(output_dim=6)
    barcodes = np.array(["a", "b"])
    patches = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    availability = np.array([True, True])
    build_uni2_spot_feature_cache(tmp_path, "s1", barcodes, patches, availability, encoder)
    with pytest.raises(ValueError, match="uni2_preprocessing_spec"):
        audit_uni2_cache_matches_config(tmp_path, "s1", config)


def test_uni2_cache_audit_fails_closed_on_checkpoint_content_mismatch(tmp_path):
    config = yaml.safe_load((_CONFIG_DIR / "gen4a_conditioner.yaml").read_text())
    checkpoint = tmp_path / "uni2.pt"
    checkpoint.write_bytes(b"real checkpoint bytes")
    config["required_fingerprints"]["uni2_checkpoint"] = str(checkpoint)
    encoder = StubUNI2Encoder(output_dim=6)  # identity.checkpoint_sha256 does NOT match the real file above
    barcodes = np.array(["a", "b"])
    patches = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    availability = np.array([True, True])
    build_uni2_spot_feature_cache(tmp_path, "s1", barcodes, patches, availability, encoder)
    with pytest.raises(ValueError, match="uni2_checkpoint content sha256"):
        audit_uni2_cache_matches_config(tmp_path, "s1", config)


def test_scfoundation_cache_audit_fails_closed_on_preprocessing_spec_mismatch(tmp_path):
    config = yaml.safe_load((_CONFIG_DIR / "gen4b_conditioner.yaml").read_text())
    config["required_fingerprints"]["scfoundation_preprocessing_spec"] = "expected_spec_v2"
    gene_names = [f"g{i}" for i in range(4)]
    encoder = StubSCFoundationEncoder(gene_names, output_dim=7)
    barcodes = np.array(["a", "b"])
    expression = np.zeros((2, 4), dtype=np.float32)
    build_scfoundation_spot_feature_cache(tmp_path, "s1", barcodes, expression, "hash123", encoder)
    with pytest.raises(ValueError, match="scfoundation_preprocessing_spec"):
        audit_scfoundation_cache_matches_config(tmp_path, "s1", config)


def _write_uni2_dense_cache(cache_root: Path, sample_id: str, uni2: StubUNI2Encoder, output_dim: int) -> None:
    cache_dir = Path(cache_root) / "uni2_dense_wsi_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    coords = np.asarray([[0.0, 0.0], [256.0, 0.0]], dtype=np.float32)
    np.savez(
        cache_dir / f"{sample_id}.npz",
        features=np.ones((2, output_dim), dtype=np.float32), coords=coords, level0_coords=coords,
        tile_size=np.asarray(256.0, dtype=np.float32), level0_tile_size=np.asarray(256.0, dtype=np.float32),
        coords_are_centers=np.asarray(True), wsi_dimensions=np.asarray([1024.0, 1024.0], dtype=np.float64),
        uni2_checkpoint_sha256=np.asarray(uni2.identity.checkpoint_sha256),
        uni2_pinned_revision=np.asarray(uni2.identity.pinned_revision),
        uni2_package_version=np.asarray(uni2.identity.package_version),
        uni2_preprocessing_spec=np.asarray(uni2.identity.preprocessing_spec),
        uni2_output_dim=np.asarray(output_dim), uni2_schema_version=np.asarray(1),
    )


def test_audit_gen4_manifest_cache_coverage_requires_every_resolved_sample(tmp_path):
    """gen4c needs uni2, uni2_dense, AND scfoundation caches for EVERY
    resolved sample -- one sample missing one modality is a hard
    failure, never a silently-incomplete "mostly covered" success."""
    config = yaml.safe_load((_CONFIG_DIR / "gen4c_conditioner.yaml").read_text())
    config["model"]["params"]["image_feature_dim"] = 6
    config["model"]["params"]["gex_context_embedding_dim"] = 7
    uni2 = StubUNI2Encoder(output_dim=6)
    gene_names = [f"g{i}" for i in range(4)]
    scfoundation = StubSCFoundationEncoder(gene_names, output_dim=7)
    barcodes = np.array(["a", "b"])
    patches = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    availability = np.array([True, True])
    expression = np.zeros((2, 4), dtype=np.float32)
    for sample_id in ("s1", "s2"):
        build_uni2_spot_feature_cache(tmp_path, sample_id, barcodes, patches, availability, uni2)
        _write_uni2_dense_cache(tmp_path, sample_id, uni2, output_dim=6)
        build_scfoundation_spot_feature_cache(tmp_path, sample_id, barcodes, expression, "hash123", scfoundation)
    # Full coverage -- succeeds.
    report = audit_gen4_manifest_cache_coverage(tmp_path, ["s1", "s2"], config)
    assert report["n_samples"] == 2
    assert set(report["samples"]) == {"s1", "s2"}
    assert set(report["modalities"]) == {"uni2", "uni2_dense", "scfoundation"}
    # A third resolved sample with NO cache at all -- must hard-fail, not
    # silently report partial coverage as success.
    with pytest.raises(FileNotFoundError):
        audit_gen4_manifest_cache_coverage(tmp_path, ["s1", "s2", "s3"], config)


def test_audit_gen4_manifest_cache_coverage_only_checks_modalities_the_arm_needs(tmp_path):
    """gen4d (STPath-only) needs no per-sample UNI2/scFoundation cache at
    all -- coverage must succeed with zero caches built, never demand a
    modality this arm doesn't consume."""
    config = yaml.safe_load((_CONFIG_DIR / "gen4d_conditioner.yaml").read_text())
    report = audit_gen4_manifest_cache_coverage(tmp_path, ["s1"], config)
    assert report["modalities"] == []
    assert report["samples"] == {"s1": {}}


def test_audit_gen4_manifest_cache_coverage_resolves_a_gen5_style_arm(tmp_path):
    """Integration audit item 7 ("generalize cache preflight by arm"):
    a Gen5 config's model.arm is a Gen5-style key (e.g. "gen5a"), never
    one of ARM_TABLE's own Gen4 keys directly -- this must resolve
    through GEN5_TO_GEN4_ARM (gen5a -> gen4a) rather than raising
    "unknown model.arm"."""
    config = {"model": {"arm": "gen5a", "kind": "latent_flow"}}
    uni2 = StubUNI2Encoder(output_dim=6)
    barcodes = np.array(["a", "b"])
    patches = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    availability = np.array([True, True])
    build_uni2_spot_feature_cache(tmp_path, "s1", barcodes, patches, availability, uni2)
    _write_uni2_dense_cache(tmp_path, "s1", uni2, output_dim=6)

    report = audit_gen4_manifest_cache_coverage(tmp_path, ["s1"], config)
    assert report["arm"] == "gen4a"
    assert set(report["modalities"]) == {"uni2", "uni2_dense"}


def test_manifest_cache_coverage_hashes_large_artifacts_only_once(tmp_path, monkeypatch):
    """External weights are content-verified once per preflight, not once
    per sample.  This is a runtime property, not a weakened cache check."""
    config = yaml.safe_load((_CONFIG_DIR / "gen4b_conditioner.yaml").read_text())
    gene_names = [f"g{i}" for i in range(4)]
    encoder = StubSCFoundationEncoder(gene_names, output_dim=7)
    config["model"]["params"]["gex_context_embedding_dim"] = 7
    checkpoint = tmp_path / "scfoundation.ckpt"
    vocab = tmp_path / "vocab.tsv"
    checkpoint.write_bytes(b"checkpoint")
    vocab.write_bytes(b"vocab")
    config["required_fingerprints"].update({
        "scfoundation_checkpoint": str(checkpoint),
        "scfoundation_vocab": str(vocab),
        "scfoundation_package_version": encoder.identity.package_version,
        "scfoundation_preprocessing_spec": encoder.identity.preprocessing_spec,
    })
    barcodes = np.array(["a", "b"])
    expression = np.zeros((2, 4), dtype=np.float32)
    for sample_id in ("s1", "s2", "s3"):
        build_scfoundation_spot_feature_cache(
            tmp_path, sample_id, barcodes, expression, "hash123", encoder,
        )

    calls = []

    def fake_hash(path):
        calls.append(Path(path))
        return (
            encoder.identity.checkpoint_sha256
            if Path(path) == checkpoint
            else encoder.identity.pinned_revision
        )

    monkeypatch.setattr("gen3_multiscale.gen4.preflight._sha256_file", fake_hash)
    report = audit_gen4_manifest_cache_coverage(tmp_path, ["s1", "s2", "s3"], config)
    assert report["n_samples"] == 3
    assert calls.count(checkpoint) == 1
    assert calls.count(vocab) == 1
