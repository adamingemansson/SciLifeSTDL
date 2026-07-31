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

from pathlib import Path

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
    """Integration audit item 7: the mandatory Gen4 preflight gate runs
    BEFORE any model/optimizer/DataLoader construction -- a resolved
    manifest sample missing its real UNI2 cache must fail here, not
    surface later as an opaque error deep inside the dataset's first
    __getitem__ call."""
    import pytest

    cfg, manifest, manifest_path, samples = _prepare_experiment(tmp_path, monkeypatch)
    # Deliberately never call _write_uni2_caches -- gen4a's required
    # uni2/uni2_dense caches do not exist for any resolved sample.

    config = yaml.safe_load((_CONFIG_DIR / "gen4" / "gen4a_conditioner.yaml").read_text())
    config["model"]["params"].update(_TINY_PARAMS)
    config_path = tmp_path / "gen4a_config_missing_cache.yaml"
    _write_config(config, config_path, cfg=cfg, manifest_path=manifest_path, checkpoint_dir=tmp_path / "ckpt_gen4a_missing")

    with pytest.raises(FileNotFoundError, match="UNI2"):
        run_training(str(config_path), smoke=True)


def test_run_training_non_smoke_gen4a_then_evaluate(tmp_path, monkeypatch):
    """A real (non-smoke) one-step gen4a run writes a real checkpoint
    (a plain --smoke run deliberately never does, matching every other
    Gen3 architecture's own "construction-only" smoke contract) -- then
    a real evaluate_gen3_checkpoint call over that checkpoint proves the
    evaluator's own Gen4-aware dataset dispatch and kind-derived
    prediction path (Integration audit item 9) actually execute."""
    cfg, manifest, manifest_path, samples = _prepare_experiment(tmp_path, monkeypatch)
    _write_uni2_caches(cfg, samples, output_dim=_TINY_PARAMS["image_feature_dim"])

    config = yaml.safe_load((_CONFIG_DIR / "gen4" / "gen4a_conditioner.yaml").read_text())
    config["model"]["params"].update(_TINY_PARAMS)
    config["training"]["total_steps"] = 1
    config["evaluation"]["gene_panels"] = {}  # the real named panel file isn't present in this sandbox
    config_path = tmp_path / "gen4a_nonsmoke_config.yaml"
    checkpoint_dir = tmp_path / "ckpt_gen4a_nonsmoke"
    _write_config(config, config_path, cfg=cfg, manifest_path=manifest_path, checkpoint_dir=checkpoint_dir)

    result = run_training(str(config_path), smoke=False)
    assert result["ok"] is True
    assert result["final_step"] == 1

    from gen3_multiscale.evaluation.gen3_evaluator import evaluate_gen3_checkpoint

    report = evaluate_gen3_checkpoint(str(config_path), checkpoint_dir, split="validation", use_best=False)
    assert report["kind"] == "gen3_step7_evaluation_report"
