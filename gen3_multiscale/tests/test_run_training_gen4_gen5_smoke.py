"""Integration audit items 1/6: a real call through
`training.train.run_training(config_path, smoke=True)` for each real
`model.kind` -- "conditioner" (gen4a), "flow" (gen4c_flow), and
"latent_flow" (gen5a) -- proving the FULL real path (preflight, the
Gen4-aware dataset adapter, generic kind-dispatched loss/prediction, one
real optimizer step, one real validation step) actually executes for
Gen4/Gen5, not merely that the individual pieces construct in isolation.

Built on the same real, small, end-to-end synthetic Gen3 experiment
(`tests/_step6_fixtures.py::build_synthetic_gen3_experiment`) plus a real
UNI2 spot cache, a real UNI2 dense-WSI cache, and (for the arms that need
it) a real scFoundation cache -- exactly `tests/test_gen4_dataset_adapter.
py`'s own fixture discipline, reused directly rather than duplicated."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from gen3_multiscale.data.dataset_manifest import save_dataset_manifest
from gen3_multiscale.tests._step6_fixtures import VALID_HF_REVISION, build_synthetic_gen3_experiment
from gen3_multiscale.tests.test_gen4_dataset_adapter import (
    _STRATA, _write_scfoundation_cache, _write_uni2_caches,
)
from gen3_multiscale.training.gen3_dataset import load_gen3_sample_data
from gen3_multiscale.training.train import run_training

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

_TINY_PARAMS = dict(
    gex_feature_dim=4, image_feature_dim=8, hidden_dim=16, n_heads=2, n_blocks=1,
    dense_threshold=100, sparse_k=4, chunk_size=64, max_boundary_size=None,
    transport_heads=2, transport_temperature=1.0, gene_gate_mode="per_gene",
    use_query_gate=True, use_residual=False, residual_rank=4,
    use_regional_he=False, regional_grid_size=2, global_slide_dim=8,
    n_gex_inducing=4, harmonic_k_neighbors=4, init_seed=0,
)


def _write_config(config: dict, path, *, cfg, manifest_path, checkpoint_dir) -> None:
    config["masking"]["strata"] = _STRATA
    config["data"]["hest_data_dir"] = str(cfg.data.hest_data_dir)
    config["data"]["hest_cache_dir"] = str(cfg.data.hest_cache_dir)
    config["data"]["slide_context_source"] = str(cfg.data.slide_context_source)
    config["data"]["gen3_manifest_path"] = str(manifest_path)
    config["data"]["tile_encoder_revision"] = VALID_HF_REVISION
    config["data"]["gex_feature_dim"] = _TINY_PARAMS["gex_feature_dim"]
    config["data"]["n_training_masks_per_sample"] = 1
    config["data"]["n_validation_masks"] = 1
    config["training"]["device"] = "cpu"
    config["training"]["checkpoint_dir"] = str(checkpoint_dir)
    config["training"]["synchronized_init_dir"] = None
    with open(path, "w") as handle:
        yaml.safe_dump(config, handle)


def _prepare_experiment(tmp_path, monkeypatch):
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch, n_side=6, spacing=300.0, n_genes=6)
    manifest_path = tmp_path / "manifest.json"
    save_dataset_manifest(manifest, manifest_path)
    # run_training preflights train+validation samples together -- every
    # Gen4-only cache must cover both roles, not just the training split.
    all_ids = manifest["train_sample_ids"] + manifest["validation_sample_ids"]
    samples = {sid: load_gen3_sample_data(cfg, manifest, sid) for sid in all_ids}
    return cfg, manifest, manifest_path, samples


def _pin_uni2_test_artifacts_for_real_run(config: dict, cfg, samples, tmp_path) -> None:
    """Make the synthetic UNI2 caches satisfy the real-run provenance gate.

    The shared lightweight encoder fixture records a deliberately fake
    checkpoint hash because construction-only smoke tests do not load a
    real checkpoint.  A non-smoke integration test must instead bind the
    caches to an actual file whose content hash the production preflight
    can recompute.
    """
    checkpoint_path = tmp_path / "uni2_test_checkpoint.bin"
    checkpoint_path.write_bytes(b"deterministic UNI2 integration-test checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    for sample_id in samples:
        for cache_path in (
            Path(str(cfg.data.hest_cache_dir)) / "uni2_gen3_spot_cache" / f"{sample_id}.npz",
            Path(str(cfg.data.hest_cache_dir)) / "uni2_dense_wsi_cache" / f"{sample_id}.npz",
        ):
            with np.load(cache_path, allow_pickle=False) as cached:
                payload = {key: cached[key] for key in cached.files}
            payload["uni2_checkpoint_sha256"] = np.asarray(checkpoint_sha256)
            with cache_path.open("wb") as handle:
                np.savez(handle, **payload)

    config["required_fingerprints"].update(
        {
            "uni2_checkpoint": str(checkpoint_path),
            "uni2_revision": "0" * 40,
            "uni2_package_version": "stub-0.0.0",
            "uni2_preprocessing_spec": "stub_uni2_v1",
        }
    )


def test_run_training_smoke_gen4a_conditioner_kind(tmp_path, monkeypatch):
    cfg, manifest, manifest_path, samples = _prepare_experiment(tmp_path, monkeypatch)
    _write_uni2_caches(cfg, samples, output_dim=_TINY_PARAMS["image_feature_dim"])

    config = yaml.safe_load((_CONFIG_DIR / "gen4" / "gen4a_conditioner.yaml").read_text())
    config["model"]["params"].update(_TINY_PARAMS)
    config_path = tmp_path / "gen4a_config.yaml"
    checkpoint_dir = tmp_path / "ckpt_gen4a"
    _write_config(config, config_path, cfg=cfg, manifest_path=manifest_path, checkpoint_dir=checkpoint_dir)

    result = run_training(str(config_path), smoke=True)
    assert result["ok"] is True


def test_run_training_smoke_gen4c_flow_kind(tmp_path, monkeypatch):
    cfg, manifest, manifest_path, samples = _prepare_experiment(tmp_path, monkeypatch)
    _write_uni2_caches(cfg, samples, output_dim=_TINY_PARAMS["image_feature_dim"])
    _write_scfoundation_cache(cfg, manifest, samples, output_dim=6)

    config = yaml.safe_load((_CONFIG_DIR / "gen4" / "gen4c_flow.yaml").read_text())
    config["model"]["params"].update(_TINY_PARAMS)
    config["model"]["params"]["gex_context_embedding_dim"] = 6
    config["model"]["params"]["n_flow_blocks"] = 1
    config["model"]["params"]["n_flow_samples"] = 2
    config["model"]["params"]["n_ode_steps"] = 2
    config_path = tmp_path / "gen4c_flow_config.yaml"
    _write_config(config, config_path, cfg=cfg, manifest_path=manifest_path, checkpoint_dir=tmp_path / "ckpt_gen4c_flow")

    result = run_training(str(config_path), smoke=True)
    assert result["ok"] is True


def test_run_training_smoke_gen5a_latent_flow_kind(tmp_path, monkeypatch):
    cfg, manifest, manifest_path, samples = _prepare_experiment(tmp_path, monkeypatch)
    _write_uni2_caches(cfg, samples, output_dim=_TINY_PARAMS["image_feature_dim"])

    config = yaml.safe_load((_CONFIG_DIR / "gen5" / "gen5a.yaml").read_text())
    config["model"]["params"].update(_TINY_PARAMS)
    config["model"]["params"]["latent_dim"] = 4
    config["model"]["params"]["n_flow_blocks"] = 1
    config["model"]["params"]["n_flow_samples"] = 2
    config["model"]["params"]["n_ode_steps"] = 2
    config_path = tmp_path / "gen5a_config.yaml"
    _write_config(config, config_path, cfg=cfg, manifest_path=manifest_path, checkpoint_dir=tmp_path / "ckpt_gen5a")

    result = run_training(str(config_path), smoke=True)
    assert result["ok"] is True


def test_run_training_fails_closed_when_gen4_cache_missing(tmp_path, monkeypatch):
    """The arm-specific cache gate runs before model/dataset construction."""
    cfg, manifest, manifest_path, _samples = _prepare_experiment(tmp_path, monkeypatch)

    config = yaml.safe_load((_CONFIG_DIR / "gen4" / "gen4a_conditioner.yaml").read_text())
    config["model"]["params"].update(_TINY_PARAMS)
    config_path = tmp_path / "gen4a_config_missing_cache.yaml"
    _write_config(
        config,
        config_path,
        cfg=cfg,
        manifest_path=manifest_path,
        checkpoint_dir=tmp_path / "ckpt_gen4a_missing",
    )

    with pytest.raises(FileNotFoundError, match="UNI2"):
        run_training(str(config_path), smoke=True)


def test_run_training_non_smoke_gen4a_then_evaluate(tmp_path, monkeypatch):
    """Exercise the real train -> checkpoint -> evaluator Gen4 path."""
    cfg, manifest, manifest_path, samples = _prepare_experiment(tmp_path, monkeypatch)
    _write_uni2_caches(cfg, samples, output_dim=_TINY_PARAMS["image_feature_dim"])

    config = yaml.safe_load((_CONFIG_DIR / "gen4" / "gen4a_conditioner.yaml").read_text())
    config["model"]["params"].update(_TINY_PARAMS)
    config["training"]["total_steps"] = 1
    config["evaluation"]["gene_panels"] = {}
    _pin_uni2_test_artifacts_for_real_run(config, cfg, samples, tmp_path)
    config_path = tmp_path / "gen4a_nonsmoke_config.yaml"
    checkpoint_dir = tmp_path / "ckpt_gen4a_nonsmoke"
    _write_config(
        config,
        config_path,
        cfg=cfg,
        manifest_path=manifest_path,
        checkpoint_dir=checkpoint_dir,
    )

    result = run_training(str(config_path), smoke=False)
    assert result["ok"] is True
    assert result["final_step"] == 1

    from gen3_multiscale.evaluation.gen3_evaluator import evaluate_gen3_checkpoint

    report = evaluate_gen3_checkpoint(
        str(config_path),
        checkpoint_dir,
        split="validation",
        use_best=False,
    )
    assert report["kind"] == "gen3_step7_evaluation_report"
