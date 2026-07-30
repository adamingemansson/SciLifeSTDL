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


def test_audit_gen4_manifest_cache_coverage_requires_every_resolved_sample(tmp_path):
    """gen4c needs BOTH uni2 and scfoundation caches for EVERY resolved
    sample -- one sample missing one modality is a hard failure, never a
    silently-incomplete "mostly covered" success."""
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
        build_scfoundation_spot_feature_cache(tmp_path, sample_id, barcodes, expression, "hash123", scfoundation)
    # Full coverage -- succeeds.
    report = audit_gen4_manifest_cache_coverage(tmp_path, ["s1", "s2"], config)
    assert report["n_samples"] == 2
    assert set(report["samples"]) == {"s1", "s2"}
    assert set(report["modalities"]) == {"uni2", "scfoundation"}
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
