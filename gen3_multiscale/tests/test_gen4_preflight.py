"""Static preflight/audit tests -- GEN4_CONTRACT.md section 9/12."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from gen3_multiscale.gen4.preflight import (
    audit_scfoundation_cache_matches_config, audit_uni2_cache_matches_config, static_audit_gen4_config,
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
