import json

import pytest
import yaml

from gen3_multiscale.scripts import prepare_wae_gan_histology_ablation_suite as suite_module


def _write_comparison_config(tmp_path):
    (tmp_path / "gen3_multiscale").mkdir()
    config_path = tmp_path / "gen3_multiscale" / "results" / "run" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config = {
        "model": {
            "architecture": "1",
            "params": {
                "image_feature_dim": 1536, "hidden_dim": 512, "n_heads": 8,
                "n_blocks": 3, "dense_threshold": 400, "sparse_k": 16,
            },
        },
        "data": {"gex_feature_dim": 256, "tile_encoder_revision": "abc123"},
        "training": {"seed": 42},
    }
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def _write_manifest(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"gene_panel": ["G0", "G1"], "train_sample_ids": ["S0"]}))
    return manifest_path


@pytest.fixture(autouse=True)
def _stub_train_gene_panels(monkeypatch):
    monkeypatch.setattr(
        suite_module,
        "load_train_derived_gene_panels",
        lambda _path, _manifest: {
            "panels": {"train_log1p_variance_top50": ["G0"], "train_log1p_variance_top200": ["G0", "G1"]},
        },
    )


def test_prepare_writes_a_control_and_a_histology_arm_with_the_right_overrides(tmp_path):
    comparison_config = _write_comparison_config(tmp_path)
    manifest = _write_manifest(tmp_path)
    panels = tmp_path / "panels.json"
    panels.write_text("{}")
    output_root = tmp_path / "suite"

    plan = suite_module.prepare_wae_gan_histology_ablation_suite(
        comparison_config=str(comparison_config), manifest=str(manifest),
        train_gene_panels=str(panels), output_root=str(output_root),
        hours=8.0, gpus=(0, 2), cpu_threads=12,
    )

    assert set(plan["arms"]) == set(suite_module.ARM_ORDER)
    configs = {
        arm: yaml.safe_load((output_root / "configs" / f"{arm}.yaml").read_text())
        for arm in suite_module.ARM_ORDER
    }

    control = configs["wae_he_gan_histology_control"]
    assert control["model"]["params"]["use_histology_context"] is False
    assert "use_histology_features" not in control["data"]
    assert control["training"]["lr"] == 1e-4
    assert control["training"]["device"] == "cuda:0"

    treatment = configs["wae_he_gan_histology"]
    assert treatment["model"]["params"]["use_histology_context"] is True
    assert treatment["data"]["use_histology_features"] is True
    assert treatment["training"]["lr"] == 1e-4  # training hyperparams held fixed
    assert treatment["training"]["device"] == "cuda:2"

    checkpoint_dirs = {config["training"]["checkpoint_dir"] for config in configs.values()}
    log_dirs = {config["evaluation"]["tensorboard"]["log_dir"] for config in configs.values()}
    assert len(checkpoint_dirs) == 2
    assert len(log_dirs) == 2

    pointer = tmp_path / "LATEST_WAE_GAN_HISTOLOGY_ABLATION_SUITE_ROOT.txt"
    assert pointer.read_text().strip() == str(output_root.resolve())


def test_prepare_accepts_an_explicit_cache_dir_override(tmp_path):
    comparison_config = _write_comparison_config(tmp_path)
    manifest = _write_manifest(tmp_path)
    panels = tmp_path / "panels.json"
    panels.write_text("{}")
    output_root = tmp_path / "suite"
    cache_dir = tmp_path / "histology_cache"

    plan = suite_module.prepare_wae_gan_histology_ablation_suite(
        comparison_config=str(comparison_config), manifest=str(manifest),
        train_gene_panels=str(panels), output_root=str(output_root),
        histology_feature_cache_dir=str(cache_dir),
    )

    treatment = yaml.safe_load((output_root / "configs" / "wae_he_gan_histology.yaml").read_text())
    assert treatment["data"]["gen3_histology_feature_cache_dir"] == str(cache_dir)
    control = yaml.safe_load((output_root / "configs" / "wae_he_gan_histology_control.yaml").read_text())
    assert "gen3_histology_feature_cache_dir" not in control["data"]
    assert plan["arms"]["wae_he_gan_histology"]["use_histology_context"] is True
    assert plan["arms"]["wae_he_gan_histology_control"]["use_histology_context"] is False


def test_prepare_refuses_to_overwrite_an_existing_suite_root(tmp_path):
    comparison_config = _write_comparison_config(tmp_path)
    manifest = _write_manifest(tmp_path)
    panels = tmp_path / "panels.json"
    panels.write_text("{}")
    output_root = tmp_path / "suite"
    output_root.mkdir()
    with pytest.raises(FileExistsError):
        suite_module.prepare_wae_gan_histology_ablation_suite(
            comparison_config=str(comparison_config), manifest=str(manifest),
            train_gene_panels=str(panels), output_root=str(output_root),
        )


def test_prepare_rejects_wrong_gpu_count(tmp_path):
    comparison_config = _write_comparison_config(tmp_path)
    manifest = _write_manifest(tmp_path)
    panels = tmp_path / "panels.json"
    panels.write_text("{}")
    with pytest.raises(ValueError, match="gpus"):
        suite_module.prepare_wae_gan_histology_ablation_suite(
            comparison_config=str(comparison_config), manifest=str(manifest),
            train_gene_panels=str(panels), output_root=str(tmp_path / "suite"), gpus=(0, 1, 2),
        )


def test_matched_except_declared_rejects_an_undeclared_divergence():
    control = {"model": {"regularizer": "gan"}, "loss": {"pcc_weight": 0.1}}
    other = {"model": {"regularizer": "mmd"}, "loss": {"pcc_weight": 0.1}}
    with pytest.raises(ValueError, match="undeclared divergence"):
        suite_module._assert_matched_except_declared(control, other, "wae_he_gan_histology")


def test_matched_except_declared_allows_only_declared_histology_fields():
    control = {"model": {"params": {"use_histology_context": False}}, "data": {}}
    other = {
        "model": {"params": {"use_histology_context": True}},
        "data": {"use_histology_features": True, "gen3_histology_feature_cache_dir": "/x"},
    }
    suite_module._assert_matched_except_declared(control, other, "wae_he_gan_histology")
