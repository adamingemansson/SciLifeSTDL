"""Tests for gen3_multiscale/scripts/resolve_experiment_config.py -- the
first bounded piece of the deployment/orchestration system tracked as
unbuilt across CONTRACT.md sections 53-54 ("a dedicated config-resolution
CLI that refuses unresolved/null fields and records resolved-config
hashes"). Exercises the real, committed configs/architectureN.yaml
templates directly, and proves a resolved config is actually loadable
and runnable by the real trainer (train.py::run_training), not merely
schema-valid YAML.

Codex re-audit of commit 7a2d819, findings #4/#5: a resolved config is
now a real TRANSACTIONAL BUNDLE directory (config.yaml + identity.json,
staged then atomically renamed into place), never two independently-
written loose files, and every deployment field is validated as a
non-blank string / correctly-shaped value before use, never stringified
first and validated after. Tests below cover both."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from gen3_multiscale.scripts.resolve_experiment_config import (
    load_verified_resolved_config, resolve_and_save_experiment_config, resolve_experiment_config,
)
from gen3_multiscale.tests._step6_fixtures import prepare_step6_experiment
from gen3_multiscale.training import train as train_module

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_A_REVISION = "a" * 40  # a valid, 40-char lowercase-hex placeholder revision


def test_resolve_experiment_config_fills_deployment_fields_and_drops_dead_ones():
    resolved = resolve_experiment_config(
        _CONFIG_DIR / "architecture1.yaml",
        gen3_manifest_path="/tmp/manifest.json", tile_encoder_revision=_A_REVISION,
        synchronized_init_dir="/tmp/sync_init",
    )
    assert resolved["data"]["gen3_manifest_path"] == "/tmp/manifest.json"
    assert resolved["data"]["tile_encoder_revision"] == _A_REVISION
    assert resolved["training"]["synchronized_init_dir"] == "/tmp/sync_init"
    # Dead fields (never read by train.py -- see the module's own
    # docstring) are dropped entirely, never left as misleading nulls.
    assert "n_genes" not in resolved["model"]["params"]
    assert "gex_feature_dim" not in resolved["model"]["params"]
    for field in ("gene_vocabulary", "train_mask_bank", "validation_mask_bank", "test_mask_bank"):
        assert field not in resolved["required_fingerprints"]
    # training.checkpoint_dir is NOT overridden when no override is given --
    # the base config's own default survives untouched.
    assert resolved["training"]["checkpoint_dir"] == "gen3_multiscale/results/architecture1"


def test_resolve_experiment_config_overrides_checkpoint_dir_when_given():
    resolved = resolve_experiment_config(
        _CONFIG_DIR / "architecture1.yaml",
        gen3_manifest_path="/tmp/manifest.json", tile_encoder_revision=_A_REVISION,
        synchronized_init_dir="/tmp/sync_init", checkpoint_dir="/tmp/my_ckpt_dir",
    )
    assert resolved["training"]["checkpoint_dir"] == "/tmp/my_ckpt_dir"


def test_resolve_experiment_config_requires_gigapath_checkpoint_when_use_global_slide():
    """architecture3.yaml has model.params.use_global_slide=true --
    resolving it without a gigapath checkpoint must fail closed, never
    silently produce a config that would later raise deep inside a real
    training run."""
    with pytest.raises(ValueError, match="gigapath_checkpoint"):
        resolve_experiment_config(
            _CONFIG_DIR / "architecture3.yaml",
            gen3_manifest_path="/tmp/manifest.json", tile_encoder_revision=_A_REVISION,
            synchronized_init_dir="/tmp/sync_init",
        )
    # Supplying it resolves cleanly.
    resolved = resolve_experiment_config(
        _CONFIG_DIR / "architecture3.yaml",
        gen3_manifest_path="/tmp/manifest.json", tile_encoder_revision=_A_REVISION,
        synchronized_init_dir="/tmp/sync_init", gigapath_checkpoint="/tmp/gigapath.pt",
    )
    assert resolved["required_fingerprints"]["gigapath_checkpoint"] == "/tmp/gigapath.pt"


def test_resolve_experiment_config_requires_architecture4_specific_fields():
    base_kwargs = dict(
        gen3_manifest_path="/tmp/manifest.json", tile_encoder_revision=_A_REVISION,
        synchronized_init_dir="/tmp/sync_init", gigapath_checkpoint="/tmp/gigapath.pt",
    )
    with pytest.raises(ValueError, match="gene_residual_basis"):
        resolve_experiment_config(_CONFIG_DIR / "architecture4.yaml", **base_kwargs)
    with pytest.raises(ValueError, match="architecture3_conditioner_checkpoint"):
        resolve_experiment_config(
            _CONFIG_DIR / "architecture4.yaml", **base_kwargs, gene_residual_basis="/tmp/basis.pt",
        )
    resolved = resolve_experiment_config(
        _CONFIG_DIR / "architecture4.yaml", **base_kwargs,
        gene_residual_basis="/tmp/basis.pt", architecture3_conditioner_checkpoint="/tmp/arch3_ckpt",
    )
    assert resolved["required_fingerprints"]["gene_residual_basis"] == "/tmp/basis.pt"
    assert resolved["required_fingerprints"]["architecture3_conditioner_checkpoint"] == "/tmp/arch3_ckpt"


# ---------------------------------------------------------------------------
# Codex re-audit of commit 7a2d819, finding #5: reject None/blank strings
# (never silently stringified to the truthy "None"), unknown architecture
# IDs, and non-40-hex tile revisions.
# ---------------------------------------------------------------------------

def test_resolve_experiment_config_rejects_none_instead_of_stringifying_it():
    """`str(None) == "None"`, a TRUTHY string that would have silently
    passed the old `if not value:` missing-field check -- confirmed real
    gap, now rejected explicitly and immediately, never stringified
    first."""
    with pytest.raises(ValueError, match="gen3_manifest_path"):
        resolve_experiment_config(
            _CONFIG_DIR / "architecture1.yaml",
            gen3_manifest_path=None, tile_encoder_revision=_A_REVISION, synchronized_init_dir="/tmp/sync_init",
        )
    with pytest.raises(ValueError, match="tile_encoder_revision"):
        resolve_experiment_config(
            _CONFIG_DIR / "architecture1.yaml",
            gen3_manifest_path="/tmp/manifest.json", tile_encoder_revision=None,
            synchronized_init_dir="/tmp/sync_init",
        )
    with pytest.raises(ValueError, match="synchronized_init_dir"):
        resolve_experiment_config(
            _CONFIG_DIR / "architecture1.yaml",
            gen3_manifest_path="/tmp/manifest.json", tile_encoder_revision=_A_REVISION,
            synchronized_init_dir=None,
        )


def test_resolve_experiment_config_rejects_blank_and_whitespace_only_strings():
    with pytest.raises(ValueError, match="gen3_manifest_path"):
        resolve_experiment_config(
            _CONFIG_DIR / "architecture1.yaml",
            gen3_manifest_path="   ", tile_encoder_revision=_A_REVISION, synchronized_init_dir="/tmp/sync_init",
        )
    with pytest.raises(ValueError, match="synchronized_init_dir"):
        resolve_experiment_config(
            _CONFIG_DIR / "architecture1.yaml",
            gen3_manifest_path="/tmp/manifest.json", tile_encoder_revision=_A_REVISION, synchronized_init_dir="",
        )


def test_resolve_experiment_config_rejects_a_non_40_hex_tile_revision():
    with pytest.raises(ValueError, match="40-character hexadecimal"):
        resolve_experiment_config(
            _CONFIG_DIR / "architecture1.yaml",
            gen3_manifest_path="/tmp/manifest.json", tile_encoder_revision="not-a-real-revision",
            synchronized_init_dir="/tmp/sync_init",
        )
    with pytest.raises(ValueError, match="40-character hexadecimal"):
        resolve_experiment_config(  # 39 chars -- one short
            _CONFIG_DIR / "architecture1.yaml",
            gen3_manifest_path="/tmp/manifest.json", tile_encoder_revision="a" * 39,
            synchronized_init_dir="/tmp/sync_init",
        )
    with pytest.raises(ValueError, match="40-character hexadecimal"):
        resolve_experiment_config(  # uppercase -- must be lowercase hex
            _CONFIG_DIR / "architecture1.yaml",
            gen3_manifest_path="/tmp/manifest.json", tile_encoder_revision="A" * 40,
            synchronized_init_dir="/tmp/sync_init",
        )


def test_resolve_experiment_config_rejects_an_unknown_architecture_id(tmp_path):
    tampered = yaml.safe_load((_CONFIG_DIR / "architecture1.yaml").read_text())
    tampered["model"]["architecture"] = "99"
    tampered_path = tmp_path / "tampered.yaml"
    tampered_path.write_text(yaml.safe_dump(tampered, sort_keys=False))
    with pytest.raises(ValueError, match="not one of"):
        resolve_experiment_config(
            tampered_path,
            gen3_manifest_path="/tmp/manifest.json", tile_encoder_revision=_A_REVISION,
            synchronized_init_dir="/tmp/sync_init",
        )


# ---------------------------------------------------------------------------
# Codex re-audit of commit 7a2d819, finding #4: a resolved config is a real
# transactional bundle (config.yaml + identity.json, staged then atomically
# renamed), never two independently-written loose files; identity.json
# additionally records config_yaml_sha256/base_template_sha256; a verified
# loader recomputes both file hash and semantic fingerprints before trusting
# a bundle. force/--force is removed entirely.
# ---------------------------------------------------------------------------

def test_resolve_and_save_writes_a_transactional_bundle_with_a_complete_identity_record(tmp_path):
    from gen3_multiscale.training.train import config_fingerprint, config_identity_fingerprint

    output_dir = tmp_path / "resolved_architecture1"
    resolved = resolve_and_save_experiment_config(
        _CONFIG_DIR / "architecture1.yaml", output_dir,
        gen3_manifest_path="/tmp/manifest.json", tile_encoder_revision=_A_REVISION,
        synchronized_init_dir="/tmp/sync_init",
    )
    assert output_dir.is_dir()
    config_path = output_dir / "config.yaml"
    identity_path = output_dir / "identity.json"
    assert config_path.is_file()
    assert identity_path.is_file()
    # No staging directory left behind.
    assert list(tmp_path.glob(".resolved_architecture1.staging.*")) == []

    on_disk = yaml.safe_load(config_path.read_text())
    assert on_disk == resolved

    identity = json.loads(identity_path.read_text())
    assert identity["kind"] == "gen3_resolved_experiment_config_identity"
    assert identity["base_config_path"] == str(_CONFIG_DIR / "architecture1.yaml")
    assert identity["config_fingerprint"] == config_fingerprint(resolved)
    assert identity["config_identity_fingerprint"] == config_identity_fingerprint(resolved)

    import hashlib

    assert identity["config_yaml_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert identity["base_template_sha256"] == hashlib.sha256((_CONFIG_DIR / "architecture1.yaml").read_bytes()).hexdigest()


def test_resolve_and_save_refuses_to_overwrite_an_existing_bundle_and_has_no_force_override():
    import inspect

    assert "force" not in inspect.signature(resolve_and_save_experiment_config).parameters


def test_resolve_and_save_refuses_an_existing_output_dir(tmp_path):
    output_dir = tmp_path / "resolved_architecture1"
    resolve_and_save_experiment_config(
        _CONFIG_DIR / "architecture1.yaml", output_dir,
        gen3_manifest_path="/tmp/manifest.json", tile_encoder_revision=_A_REVISION,
        synchronized_init_dir="/tmp/sync_init",
    )
    with pytest.raises(FileExistsError, match="already exists"):
        resolve_and_save_experiment_config(
            _CONFIG_DIR / "architecture1.yaml", output_dir,
            gen3_manifest_path="/tmp/manifest.json", tile_encoder_revision=_A_REVISION,
            synchronized_init_dir="/tmp/sync_init_2",
        )
    # The correct way to "replace" a resolved config is a distinct bundle.
    other_output_dir = tmp_path / "resolved_architecture1_v2"
    resolve_and_save_experiment_config(
        _CONFIG_DIR / "architecture1.yaml", other_output_dir,
        gen3_manifest_path="/tmp/manifest.json", tile_encoder_revision=_A_REVISION,
        synchronized_init_dir="/tmp/sync_init_2",
    )
    on_disk = yaml.safe_load((other_output_dir / "config.yaml").read_text())
    assert on_disk["training"]["synchronized_init_dir"] == "/tmp/sync_init_2"
    # The original bundle is untouched.
    original_on_disk = yaml.safe_load((output_dir / "config.yaml").read_text())
    assert original_on_disk["training"]["synchronized_init_dir"] == "/tmp/sync_init"


def test_resolve_and_save_leaves_no_partial_bundle_when_resolution_fails(tmp_path):
    """A crash/failure DURING resolution (before any file is even staged)
    must leave nothing behind at output_dir."""
    output_dir = tmp_path / "resolved_architecture3"
    with pytest.raises(ValueError, match="gigapath_checkpoint"):
        resolve_and_save_experiment_config(
            _CONFIG_DIR / "architecture3.yaml", output_dir,
            gen3_manifest_path="/tmp/manifest.json", tile_encoder_revision=_A_REVISION,
            synchronized_init_dir="/tmp/sync_init",
        )
    assert not output_dir.exists()
    assert list(tmp_path.glob(".resolved_architecture3.staging.*")) == []


def test_load_verified_resolved_config_returns_the_same_config_when_untampered(tmp_path):
    output_dir = tmp_path / "resolved_architecture1"
    resolved = resolve_and_save_experiment_config(
        _CONFIG_DIR / "architecture1.yaml", output_dir,
        gen3_manifest_path="/tmp/manifest.json", tile_encoder_revision=_A_REVISION,
        synchronized_init_dir="/tmp/sync_init",
    )
    loaded = load_verified_resolved_config(output_dir)
    assert loaded == resolved


def test_load_verified_resolved_config_rejects_a_tampered_config_yaml(tmp_path):
    output_dir = tmp_path / "resolved_architecture1"
    resolve_and_save_experiment_config(
        _CONFIG_DIR / "architecture1.yaml", output_dir,
        gen3_manifest_path="/tmp/manifest.json", tile_encoder_revision=_A_REVISION,
        synchronized_init_dir="/tmp/sync_init",
    )
    config_path = output_dir / "config.yaml"
    tampered = yaml.safe_load(config_path.read_text())
    tampered["training"]["synchronized_init_dir"] = "/tmp/a_different_dir"
    config_path.write_text(yaml.safe_dump(tampered, sort_keys=False))
    with pytest.raises(ValueError, match="config_yaml_sha256"):
        load_verified_resolved_config(output_dir)


def test_load_verified_resolved_config_rejects_a_tampered_identity_json(tmp_path):
    output_dir = tmp_path / "resolved_architecture1"
    resolve_and_save_experiment_config(
        _CONFIG_DIR / "architecture1.yaml", output_dir,
        gen3_manifest_path="/tmp/manifest.json", tile_encoder_revision=_A_REVISION,
        synchronized_init_dir="/tmp/sync_init",
    )
    identity_path = output_dir / "identity.json"
    identity = json.loads(identity_path.read_text())
    identity["config_fingerprint"] = "0" * 64
    identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True))
    with pytest.raises(ValueError, match="config_fingerprint"):
        load_verified_resolved_config(output_dir)


def test_load_verified_resolved_config_rejects_an_incomplete_bundle(tmp_path):
    output_dir = tmp_path / "not_a_real_bundle"
    output_dir.mkdir()
    (output_dir / "config.yaml").write_text("experiment_name: x\n")
    with pytest.raises(ValueError, match="not a complete resolved-config bundle"):
        load_verified_resolved_config(output_dir)


def test_a_resolved_config_is_actually_runnable_by_the_real_trainer(tmp_path, monkeypatch):
    """The real integration proof: a config produced by this script,
    written to disk and loaded back exactly like any other config, is
    genuinely runnable by `train.py::run_training` -- not merely schema-
    valid YAML that happens to look right."""
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)

    from gen3_multiscale.models import model_factory as mf

    gene_names = list(manifest["gene_panel"])
    n_genes = len(gene_names)
    common = dict(n_genes=n_genes, gex_feature_dim=8, seed=0)
    models = {"architecture1": mf.build_architecture(
        {"model": {"architecture": "1", "params": {
            "image_feature_dim": 1536, "hidden_dim": 16, "n_heads": 2, "n_blocks": 1,
            "dense_threshold": 256, "sparse_k": 10, "chunk_size": 1024, "max_boundary_size": None,
            "transport_heads": 2, "transport_temperature": 1.0, "gene_gate_mode": "per_gene",
            "use_query_gate": True, "use_residual": False, "residual_rank": 4,
            "use_anchor_blend": False, "use_regional_he": False, "use_global_gex": False,
            "use_global_slide": False, "global_slide_dim": 16, "n_gex_inducing": 4,
            "harmonic_k_neighbors": 4, "gene_encoder_type": "weighted_linear", "init_seed": 0,
        }}}, **common,
    )}
    sync_dir = tmp_path / "sync_init"
    mf.persist_synchronized_initializations(models, sync_dir)

    output_dir = tmp_path / "resolved_architecture1"
    resolve_and_save_experiment_config(
        _CONFIG_DIR / "architecture1.yaml", output_dir,
        gen3_manifest_path=str(manifest_path), tile_encoder_revision="d072f48609bec7ec4d2c43889262b3029bb1279f",
        synchronized_init_dir=str(sync_dir), checkpoint_dir=str(tmp_path / "ckpt"),
    )
    # Adapt the resolved config's data/masking sections to this test's
    # small synthetic fixture (the committed template targets a real,
    # much larger HEST-1k deployment) -- mirrors _step6_fixtures.py's own
    # write_step6_train_config shape for the fields a real deployment
    # would otherwise set identically to the committed template.
    config_path = output_dir / "config.yaml"
    resolved = yaml.safe_load(config_path.read_text())
    resolved["data"]["hest_data_dir"] = str(cfg.data.hest_data_dir)
    resolved["data"]["hest_cache_dir"] = str(cfg.data.hest_cache_dir)
    resolved["data"]["slide_context_source"] = "dense_wsi_cache"
    resolved["data"]["gex_feature_dim"] = 8
    resolved["masking"]["strata"] = [
        {"name": "small", "radius_range": [1.5, 2.5], "radius_unit": "spot_spacing", "shape": "circle"},
    ]
    resolved["model"]["params"]["image_feature_dim"] = 1536
    resolved["model"]["params"]["hidden_dim"] = 16
    resolved["model"]["params"]["n_heads"] = 2
    resolved["model"]["params"]["n_blocks"] = 1
    resolved["model"]["params"]["dense_threshold"] = 256
    resolved["model"]["params"]["sparse_k"] = 10
    resolved["model"]["params"]["chunk_size"] = 1024
    resolved["model"]["params"]["transport_heads"] = 2
    resolved["model"]["params"]["n_gex_inducing"] = 4
    resolved["model"]["params"]["harmonic_k_neighbors"] = 4
    # Written directly (not via resolve_and_save_experiment_config, so no
    # identity.json re-signing needed here) -- this test's own point is
    # exercising train.py::run_training, not load_verified_resolved_config
    # (covered separately above).
    config_path.write_text(yaml.safe_dump(resolved, sort_keys=False))

    summary = train_module.run_training(str(config_path), smoke=True)
    assert summary["ok"] is True
    assert summary["architecture"] == "1"


def test_resolve_experiment_config_cli_actually_writes_a_resolved_config_bundle(tmp_path):
    """Real subprocess test -- proves `python -m gen3_multiscale.scripts.
    resolve_experiment_config ...` genuinely works end to end, not merely
    that the importable functions do."""
    output_dir = tmp_path / "resolved"
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable, "-m", "gen3_multiscale.scripts.resolve_experiment_config",
            "--base-config", str(_CONFIG_DIR / "architecture1.yaml"), "--output-dir", str(output_dir),
            "--gen3-manifest-path", "/tmp/manifest.json", "--tile-encoder-revision", _A_REVISION,
            "--synchronized-init-dir", "/tmp/sync_init",
        ],
        cwd=repo_root, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert (output_dir / "config.yaml").is_file()
    assert (output_dir / "identity.json").is_file()


def test_resolve_experiment_config_cli_exits_nonzero_on_a_missing_required_field(tmp_path):
    output_dir = tmp_path / "resolved"
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable, "-m", "gen3_multiscale.scripts.resolve_experiment_config",
            "--base-config", str(_CONFIG_DIR / "architecture3.yaml"), "--output-dir", str(output_dir),
            "--gen3-manifest-path", "/tmp/manifest.json", "--tile-encoder-revision", _A_REVISION,
            "--synchronized-init-dir", "/tmp/sync_init",  # missing --gigapath-checkpoint (use_global_slide=true)
        ],
        cwd=repo_root, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    assert "gigapath_checkpoint" in result.stderr
    assert not output_dir.exists()


def test_resolve_experiment_config_cli_has_no_force_flag(tmp_path):
    """Codex re-audit of commit 7a2d819, finding #5: 'avoid --force.'
    Passing an unrecognized --force must fail argument parsing -- proves
    the flag is genuinely gone from the CLI, not merely undocumented."""
    output_dir = tmp_path / "resolved"
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable, "-m", "gen3_multiscale.scripts.resolve_experiment_config",
            "--base-config", str(_CONFIG_DIR / "architecture1.yaml"), "--output-dir", str(output_dir),
            "--gen3-manifest-path", "/tmp/manifest.json", "--tile-encoder-revision", _A_REVISION,
            "--synchronized-init-dir", "/tmp/sync_init", "--force",
        ],
        cwd=repo_root, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "unrecognized argument" in result.stderr.lower() or "unrecognized argument" in result.stdout.lower()
