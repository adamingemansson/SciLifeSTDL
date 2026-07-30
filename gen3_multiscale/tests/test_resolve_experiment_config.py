"""Tests for gen3_multiscale/scripts/resolve_experiment_config.py -- the
first bounded piece of the deployment/orchestration system tracked as
unbuilt across CONTRACT.md sections 53-54 ("a dedicated config-resolution
CLI that refuses unresolved/null fields and records resolved-config
hashes"). Exercises the real, committed configs/architectureN.yaml
templates directly, and proves a resolved config is actually loadable
and runnable by the real trainer (train.py::run_training), not merely
schema-valid YAML."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from gen3_multiscale.scripts.resolve_experiment_config import (
    resolve_and_save_experiment_config, resolve_experiment_config,
)
from gen3_multiscale.tests._step6_fixtures import prepare_step6_experiment
from gen3_multiscale.training import train as train_module

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_resolve_experiment_config_fills_deployment_fields_and_drops_dead_ones():
    resolved = resolve_experiment_config(
        _CONFIG_DIR / "architecture1.yaml",
        gen3_manifest_path="/tmp/manifest.json", tile_encoder_revision="a" * 40,
        synchronized_init_dir="/tmp/sync_init",
    )
    assert resolved["data"]["gen3_manifest_path"] == "/tmp/manifest.json"
    assert resolved["data"]["tile_encoder_revision"] == "a" * 40
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
        gen3_manifest_path="/tmp/manifest.json", tile_encoder_revision="a" * 40,
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
            gen3_manifest_path="/tmp/manifest.json", tile_encoder_revision="a" * 40,
            synchronized_init_dir="/tmp/sync_init",
        )
    # Supplying it resolves cleanly.
    resolved = resolve_experiment_config(
        _CONFIG_DIR / "architecture3.yaml",
        gen3_manifest_path="/tmp/manifest.json", tile_encoder_revision="a" * 40,
        synchronized_init_dir="/tmp/sync_init", gigapath_checkpoint="/tmp/gigapath.pt",
    )
    assert resolved["required_fingerprints"]["gigapath_checkpoint"] == "/tmp/gigapath.pt"


def test_resolve_experiment_config_requires_architecture4_specific_fields():
    base_kwargs = dict(
        gen3_manifest_path="/tmp/manifest.json", tile_encoder_revision="a" * 40,
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


def test_resolve_and_save_writes_a_resolved_config_and_an_identity_sidecar(tmp_path):
    from gen3_multiscale.training.train import config_fingerprint, config_identity_fingerprint

    output_path = tmp_path / "resolved_architecture1.yaml"
    resolved = resolve_and_save_experiment_config(
        _CONFIG_DIR / "architecture1.yaml", output_path,
        gen3_manifest_path="/tmp/manifest.json", tile_encoder_revision="a" * 40,
        synchronized_init_dir="/tmp/sync_init",
    )
    assert output_path.is_file()
    on_disk = yaml.safe_load(output_path.read_text())
    assert on_disk == resolved

    identity_path = tmp_path / "resolved_architecture1.yaml.resolved_identity.json"
    assert identity_path.is_file()
    identity = json.loads(identity_path.read_text())
    assert identity["kind"] == "gen3_resolved_experiment_config_identity"
    assert identity["base_config_path"] == str(_CONFIG_DIR / "architecture1.yaml")
    assert identity["config_fingerprint"] == config_fingerprint(resolved)
    assert identity["config_identity_fingerprint"] == config_identity_fingerprint(resolved)


def test_resolve_and_save_refuses_to_overwrite_an_existing_resolved_config(tmp_path):
    output_path = tmp_path / "resolved_architecture1.yaml"
    resolve_and_save_experiment_config(
        _CONFIG_DIR / "architecture1.yaml", output_path,
        gen3_manifest_path="/tmp/manifest.json", tile_encoder_revision="a" * 40,
        synchronized_init_dir="/tmp/sync_init",
    )
    with pytest.raises(FileExistsError, match="already exists"):
        resolve_and_save_experiment_config(
            _CONFIG_DIR / "architecture1.yaml", output_path,
            gen3_manifest_path="/tmp/manifest.json", tile_encoder_revision="a" * 40,
            synchronized_init_dir="/tmp/sync_init_2",
        )
    # force=True is a genuine, explicit override.
    resolve_and_save_experiment_config(
        _CONFIG_DIR / "architecture1.yaml", output_path, force=True,
        gen3_manifest_path="/tmp/manifest.json", tile_encoder_revision="a" * 40,
        synchronized_init_dir="/tmp/sync_init_2",
    )
    on_disk = yaml.safe_load(output_path.read_text())
    assert on_disk["training"]["synchronized_init_dir"] == "/tmp/sync_init_2"


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

    output_path = tmp_path / "resolved_architecture1.yaml"
    resolve_and_save_experiment_config(
        _CONFIG_DIR / "architecture1.yaml", output_path,
        gen3_manifest_path=str(manifest_path), tile_encoder_revision="d072f48609bec7ec4d2c43889262b3029bb1279f",
        synchronized_init_dir=str(sync_dir), checkpoint_dir=str(tmp_path / "ckpt"),
    )
    # Adapt the resolved config's data/masking sections to this test's
    # small synthetic fixture (the committed template targets a real,
    # much larger HEST-1k deployment) -- mirrors _step6_fixtures.py's own
    # write_step6_train_config shape for the fields a real deployment
    # would otherwise set identically to the committed template.
    resolved = yaml.safe_load(output_path.read_text())
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
    output_path.write_text(yaml.safe_dump(resolved, sort_keys=False))

    summary = train_module.run_training(str(output_path), smoke=True)
    assert summary["ok"] is True
    assert summary["architecture"] == "1"


def test_resolve_experiment_config_cli_actually_writes_a_resolved_config(tmp_path):
    """Real subprocess test -- proves `python -m gen3_multiscale.scripts.
    resolve_experiment_config ...` genuinely works end to end, not merely
    that the importable functions do."""
    output_path = tmp_path / "resolved.yaml"
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable, "-m", "gen3_multiscale.scripts.resolve_experiment_config",
            "--base-config", str(_CONFIG_DIR / "architecture1.yaml"), "--output", str(output_path),
            "--gen3-manifest-path", "/tmp/manifest.json", "--tile-encoder-revision", "a" * 40,
            "--synchronized-init-dir", "/tmp/sync_init",
        ],
        cwd=repo_root, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert output_path.is_file()
    assert (tmp_path / "resolved.yaml.resolved_identity.json").is_file()


def test_resolve_experiment_config_cli_exits_nonzero_on_a_missing_required_field(tmp_path):
    output_path = tmp_path / "resolved.yaml"
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable, "-m", "gen3_multiscale.scripts.resolve_experiment_config",
            "--base-config", str(_CONFIG_DIR / "architecture3.yaml"), "--output", str(output_path),
            "--gen3-manifest-path", "/tmp/manifest.json", "--tile-encoder-revision", "a" * 40,
            "--synchronized-init-dir", "/tmp/sync_init",  # missing --gigapath-checkpoint (use_global_slide=true)
        ],
        cwd=repo_root, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    assert "gigapath_checkpoint" in result.stderr
    assert not output_path.exists()
