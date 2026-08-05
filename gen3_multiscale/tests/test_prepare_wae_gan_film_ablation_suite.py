import json

import pytest
import yaml

from gen3_multiscale.scripts import prepare_wae_gan_film_ablation_suite as suite_module


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


def test_prepare_writes_four_film_arms_with_the_right_layer_and_sharing_choices(tmp_path):
    comparison_config = _write_comparison_config(tmp_path)
    manifest = _write_manifest(tmp_path)
    panels = tmp_path / "panels.json"
    panels.write_text("{}")
    output_root = tmp_path / "suite"

    plan = suite_module.prepare_wae_gan_film_ablation_suite(
        comparison_config=str(comparison_config), manifest=str(manifest),
        train_gene_panels=str(panels), output_root=str(output_root),
        hours=8.0, gpus=(0, 2, 3, 5), cpu_threads=12,
    )

    assert set(plan["arms"]) == set(suite_module.ARM_ORDER)
    configs = {
        arm: yaml.safe_load((output_root / "configs" / f"{arm}.yaml").read_text())
        for arm in suite_module.ARM_ORDER
    }

    control = configs["wae_he_gan_film_control"]
    assert control["model"]["params"]["encoder_conditioning"] == "film"
    assert control["model"]["params"]["film_layers"] == ["first", "second"]
    assert control["model"]["params"]["film_shared_generator"] is False
    assert control["model"]["params"]["latent_dim"] == 256
    assert control["model"]["params"]["autoencoder_hidden_dim"] == 1024
    assert control["training"]["lr"] == 1e-4
    assert control["training"]["gradient_accumulation_steps"] == 1
    assert control["training"]["device"] == "cuda:0"

    first_only = configs["wae_he_gan_film_first_only"]
    assert first_only["model"]["params"]["film_layers"] == ["first"]
    assert first_only["model"]["params"]["film_shared_generator"] is False
    assert first_only["training"]["device"] == "cuda:2"
    assert first_only["training"]["lr"] == 1e-4  # training hyperparams held fixed

    last_only = configs["wae_he_gan_film_last_only"]
    assert last_only["model"]["params"]["film_layers"] == ["second"]
    assert last_only["training"]["device"] == "cuda:3"

    shared = configs["wae_he_gan_film_shared"]
    assert shared["model"]["params"]["film_layers"] == ["first", "second"]
    assert shared["model"]["params"]["film_shared_generator"] is True
    assert shared["training"]["device"] == "cuda:5"

    checkpoint_dirs = {config["training"]["checkpoint_dir"] for config in configs.values()}
    log_dirs = {config["evaluation"]["tensorboard"]["log_dir"] for config in configs.values()}
    assert len(checkpoint_dirs) == 4
    assert len(log_dirs) == 4

    pointer = tmp_path / "LATEST_WAE_GAN_FILM_ABLATION_SUITE_ROOT.txt"
    assert pointer.read_text().strip() == str(output_root.resolve())


def test_prepare_refuses_to_overwrite_an_existing_suite_root(tmp_path):
    comparison_config = _write_comparison_config(tmp_path)
    manifest = _write_manifest(tmp_path)
    panels = tmp_path / "panels.json"
    panels.write_text("{}")
    output_root = tmp_path / "suite"
    output_root.mkdir()
    with pytest.raises(FileExistsError):
        suite_module.prepare_wae_gan_film_ablation_suite(
            comparison_config=str(comparison_config), manifest=str(manifest),
            train_gene_panels=str(panels), output_root=str(output_root),
        )


def test_prepare_rejects_wrong_gpu_count(tmp_path):
    comparison_config = _write_comparison_config(tmp_path)
    manifest = _write_manifest(tmp_path)
    panels = tmp_path / "panels.json"
    panels.write_text("{}")
    with pytest.raises(ValueError, match="gpus"):
        suite_module.prepare_wae_gan_film_ablation_suite(
            comparison_config=str(comparison_config), manifest=str(manifest),
            train_gene_panels=str(panels), output_root=str(tmp_path / "suite"),
            gpus=(0, 1, 2),
        )


def test_matched_except_declared_rejects_an_undeclared_divergence():
    control = {"model": {"regularizer": "gan"}, "loss": {"pcc_weight": 0.1}}
    other = {"model": {"regularizer": "mmd"}, "loss": {"pcc_weight": 0.1}}
    with pytest.raises(ValueError, match="undeclared divergence"):
        suite_module._assert_matched_except_declared(control, other, "some_arm")


def test_matched_except_declared_allows_only_declared_film_fields():
    control = {"model": {"params": {"film_layers": ["first", "second"], "latent_dim": 256}}}
    other = {"model": {"params": {"film_layers": ["first"], "latent_dim": 256}}}
    suite_module._assert_matched_except_declared(control, other, "wae_he_gan_film_first_only")
