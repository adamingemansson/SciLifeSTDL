import json

import pytest
import yaml

from gen3_multiscale.scripts import prepare_wae_gan_ablation_suite as suite_module


def _write_comparison_config(tmp_path):
    # A marker directory makes _source_repository_root resolvable from a
    # config living under tmp_path, mirroring a real repo checkout.
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


def test_prepare_writes_four_arms_with_the_specified_gpu_lr_and_dim_overrides(tmp_path):
    comparison_config = _write_comparison_config(tmp_path)
    manifest = _write_manifest(tmp_path)
    panels = tmp_path / "panels.json"
    panels.write_text("{}")
    output_root = tmp_path / "suite"

    plan = suite_module.prepare_wae_gan_ablation_suite(
        comparison_config=str(comparison_config), manifest=str(manifest),
        train_gene_panels=str(panels), output_root=str(output_root),
        hours=8.0, gpus=(0, 2, 3, 5), cpu_threads=12,
    )

    assert set(plan["arms"]) == set(suite_module.ARM_ORDER)
    configs = {
        arm: yaml.safe_load((output_root / "configs" / f"{arm}.yaml").read_text())
        for arm in suite_module.ARM_ORDER
    }

    control = configs["wae_he_gan_control"]
    assert control["model"]["task"] == "he_to_st"
    assert control["model"]["regularizer"] == "gan"
    assert control["model"]["include_observed_gex"] is False
    assert control["model"]["params"]["latent_dim"] == 256
    assert control["model"]["params"]["autoencoder_hidden_dim"] == 1024
    assert control["model"]["params"]["hidden_dim"] == 512  # inherited, unchanged
    assert control["model"]["params"]["dropout"] == 0.1
    assert control["training"]["lr"] == 1e-4
    assert control["training"]["gradient_accumulation_steps"] == 1
    assert control["training"]["optimizer"]["weight_decay"] == 0.01
    assert control["training"]["device"] == "cuda:0"
    assert control["training"]["cpu_threads"] == 12
    assert control["training"]["max_wall_clock_hours"] == 8.0
    assert control["training"]["seed"] == 42

    accum8 = configs["wae_he_gan_accum8"]
    assert accum8["training"]["gradient_accumulation_steps"] == 8
    assert accum8["training"]["lr"] == 1e-4
    assert accum8["training"]["device"] == "cuda:2"
    assert accum8["model"]["params"]["latent_dim"] == 256

    lr3e5 = configs["wae_he_gan_lr3e5"]
    assert lr3e5["training"]["lr"] == 3e-5
    assert lr3e5["training"]["gradient_accumulation_steps"] == 1
    assert lr3e5["training"]["device"] == "cuda:3"
    assert lr3e5["model"]["params"]["latent_dim"] == 256

    small = configs["wae_he_gan_small"]
    assert small["model"]["params"]["latent_dim"] == 128
    assert small["model"]["params"]["autoencoder_hidden_dim"] == 512
    assert small["model"]["params"]["hidden_dim"] == 256
    assert small["model"]["params"]["discriminator_hidden_dim"] == 256  # unchanged
    assert small["model"]["params"]["dropout"] == 0.1  # no extra regularization
    assert small["training"]["lr"] == 1e-4
    assert small["training"]["gradient_accumulation_steps"] == 1
    assert small["training"]["device"] == "cuda:5"

    # Each arm gets its own new checkpoint dir and TensorBoard log dir.
    checkpoint_dirs = {config["training"]["checkpoint_dir"] for config in configs.values()}
    log_dirs = {config["evaluation"]["tensorboard"]["log_dir"] for config in configs.values()}
    assert len(checkpoint_dirs) == 4
    assert len(log_dirs) == 4

    pointer = tmp_path / "LATEST_WAE_GAN_ABLATION_SUITE_ROOT.txt"
    assert pointer.read_text().strip() == str(output_root.resolve())


def test_prepare_refuses_to_overwrite_an_existing_suite_root(tmp_path):
    comparison_config = _write_comparison_config(tmp_path)
    manifest = _write_manifest(tmp_path)
    panels = tmp_path / "panels.json"
    panels.write_text("{}")
    output_root = tmp_path / "suite"
    output_root.mkdir()
    with pytest.raises(FileExistsError):
        suite_module.prepare_wae_gan_ablation_suite(
            comparison_config=str(comparison_config), manifest=str(manifest),
            train_gene_panels=str(panels), output_root=str(output_root),
        )


def test_prepare_rejects_a_non_architecture1_comparison_config(tmp_path):
    (tmp_path / "gen3_multiscale").mkdir()
    config_path = tmp_path / "gen3_multiscale" / "config.yaml"
    config_path.write_text(yaml.safe_dump({"model": {"architecture": "3"}}))
    manifest = _write_manifest(tmp_path)
    panels = tmp_path / "panels.json"
    panels.write_text("{}")
    with pytest.raises(ValueError, match="Architecture 1"):
        suite_module.prepare_wae_gan_ablation_suite(
            comparison_config=str(config_path), manifest=str(manifest),
            train_gene_panels=str(panels), output_root=str(tmp_path / "suite"),
        )


def test_prepare_rejects_wrong_gpu_count(tmp_path):
    comparison_config = _write_comparison_config(tmp_path)
    manifest = _write_manifest(tmp_path)
    panels = tmp_path / "panels.json"
    panels.write_text("{}")
    with pytest.raises(ValueError, match="gpus"):
        suite_module.prepare_wae_gan_ablation_suite(
            comparison_config=str(comparison_config), manifest=str(manifest),
            train_gene_panels=str(panels), output_root=str(tmp_path / "suite"),
            gpus=(0, 1, 2),
        )


def test_matched_except_declared_rejects_an_undeclared_divergence():
    control = {"model": {"regularizer": "gan"}, "loss": {"pcc_weight": 0.1}}
    other = {"model": {"regularizer": "mmd"}, "loss": {"pcc_weight": 0.1}}
    with pytest.raises(ValueError, match="undeclared divergence"):
        suite_module._assert_matched_except_declared(control, other, "some_arm")


def test_matched_except_declared_allows_only_declared_fields():
    control = {"training": {"lr": 1e-4, "seed": 0}}
    other = {"training": {"lr": 3e-5, "seed": 0}}
    suite_module._assert_matched_except_declared(control, other, "wae_he_gan_lr3e5")
