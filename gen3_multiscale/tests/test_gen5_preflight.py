"""Gen5 static preflight/audit tests."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gen3_multiscale.gen5.preflight import static_audit_gen5_config

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "gen5"


def test_static_audit_rejects_unknown_arm():
    config = yaml.safe_load((_CONFIG_DIR / "gen5a.yaml").read_text())
    config["model"]["arm"] = "not_a_real_arm"
    with pytest.raises(ValueError, match="model.arm"):
        static_audit_gen5_config(config)


def test_static_audit_rejects_wrong_kind():
    config = yaml.safe_load((_CONFIG_DIR / "gen5a.yaml").read_text())
    config["model"]["kind"] = "conditioner"
    with pytest.raises(ValueError, match="model.kind"):
        static_audit_gen5_config(config)


def test_static_audit_requires_autoencoder_and_conditioner_checkpoint_keys():
    config = yaml.safe_load((_CONFIG_DIR / "gen5a.yaml").read_text())
    del config["required_fingerprints"]["expression_autoencoder_checkpoint"]
    with pytest.raises(ValueError, match="expression_autoencoder_checkpoint"):
        static_audit_gen5_config(config)


def test_static_audit_requires_context_embedding_dim_for_frozen_context_arms():
    config = yaml.safe_load((_CONFIG_DIR / "gen5b.yaml").read_text())
    config["model"]["params"]["gex_context_embedding_dim"] = None
    with pytest.raises(ValueError, match="gex_context_embedding_dim"):
        static_audit_gen5_config(config)


def test_static_audit_rejects_empty_masking_strata():
    config = yaml.safe_load((_CONFIG_DIR / "gen5a.yaml").read_text())
    config["masking"]["strata"] = []
    with pytest.raises(ValueError, match="masking.strata"):
        static_audit_gen5_config(config)
