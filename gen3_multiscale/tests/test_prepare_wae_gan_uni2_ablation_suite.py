import json

import pytest
import yaml

from gen3_multiscale.scripts import prepare_wae_gan_uni2_ablation_suite as suite_module

_VALID_REVISION = "d072f48609bec7ec4d2c43889262b3029bb1279f"


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


def test_prepare_writes_a_gigapath_control_and_a_uni2_arm_with_the_right_overrides(tmp_path):
    comparison_config = _write_comparison_config(tmp_path)
    manifest = _write_manifest(tmp_path)
    panels = tmp_path / "panels.json"
    panels.write_text("{}")
    output_root = tmp_path / "suite"

    plan = suite_module.prepare_wae_gan_uni2_ablation_suite(
        comparison_config=str(comparison_config), manifest=str(manifest),
        train_gene_panels=str(panels), output_root=str(output_root),
        uni2_pinned_revision=_VALID_REVISION, hours=8.0, gpus=(0, 2), cpu_threads=12,
    )

    assert set(plan["arms"]) == set(suite_module.ARM_ORDER)
    configs = {
        arm: yaml.safe_load((output_root / "configs" / f"{arm}.yaml").read_text())
        for arm in suite_module.ARM_ORDER
    }

    control = configs["wae_he_gan_uni2_control"]
    assert control["data"]["image_encoder"] == "gigapath"
    assert control["data"]["tile_encoder_revision"] == "abc123"
    assert "uni2_pinned_revision" not in control["data"]
    assert control["model"]["params"]["image_feature_dim"] == 1536
    assert control["training"]["lr"] == 1e-4
    assert control["training"]["device"] == "cuda:0"

    uni2 = configs["wae_he_gan_uni2"]
    assert uni2["data"]["image_encoder"] == "uni2"
    assert uni2["data"]["uni2_pinned_revision"] == _VALID_REVISION
    assert uni2["model"]["params"]["image_feature_dim"] == 1536  # same dim, no dimension change needed
    assert uni2["training"]["lr"] == 1e-4  # training hyperparams held fixed
    assert uni2["training"]["device"] == "cuda:2"

    checkpoint_dirs = {config["training"]["checkpoint_dir"] for config in configs.values()}
    log_dirs = {config["evaluation"]["tensorboard"]["log_dir"] for config in configs.values()}
    assert len(checkpoint_dirs) == 2
    assert len(log_dirs) == 2

    pointer = tmp_path / "LATEST_WAE_GAN_UNI2_ABLATION_SUITE_ROOT.txt"
    assert pointer.read_text().strip() == str(output_root.resolve())


def test_prepare_applies_an_optional_uni2_cache_dir_override(tmp_path):
    comparison_config = _write_comparison_config(tmp_path)
    manifest = _write_manifest(tmp_path)
    panels = tmp_path / "panels.json"
    panels.write_text("{}")
    output_root = tmp_path / "suite"
    cache_dir = tmp_path / "shared_uni2_cache"

    suite_module.prepare_wae_gan_uni2_ablation_suite(
        comparison_config=str(comparison_config), manifest=str(manifest),
        train_gene_panels=str(panels), output_root=str(output_root),
        uni2_pinned_revision=_VALID_REVISION, gpus=(0, 2),
        uni2_spot_feature_cache_dir=str(cache_dir),
    )
    uni2_config = yaml.safe_load((output_root / "configs" / "wae_he_gan_uni2.yaml").read_text())
    assert uni2_config["data"]["gen3_uni2_spot_feature_cache_dir"] == str(cache_dir)
    control_config = yaml.safe_load((output_root / "configs" / "wae_he_gan_uni2_control.yaml").read_text())
    assert "gen3_uni2_spot_feature_cache_dir" not in control_config["data"]


def test_prepare_refuses_to_overwrite_an_existing_suite_root(tmp_path):
    comparison_config = _write_comparison_config(tmp_path)
    manifest = _write_manifest(tmp_path)
    panels = tmp_path / "panels.json"
    panels.write_text("{}")
    output_root = tmp_path / "suite"
    output_root.mkdir()
    with pytest.raises(FileExistsError):
        suite_module.prepare_wae_gan_uni2_ablation_suite(
            comparison_config=str(comparison_config), manifest=str(manifest),
            train_gene_panels=str(panels), output_root=str(output_root),
            uni2_pinned_revision=_VALID_REVISION,
        )


def test_prepare_rejects_wrong_gpu_count(tmp_path):
    comparison_config = _write_comparison_config(tmp_path)
    manifest = _write_manifest(tmp_path)
    panels = tmp_path / "panels.json"
    panels.write_text("{}")
    with pytest.raises(ValueError, match="gpus"):
        suite_module.prepare_wae_gan_uni2_ablation_suite(
            comparison_config=str(comparison_config), manifest=str(manifest),
            train_gene_panels=str(panels), output_root=str(tmp_path / "suite"),
            uni2_pinned_revision=_VALID_REVISION, gpus=(0, 1, 2),
        )


def test_prepare_rejects_a_blank_uni2_pinned_revision(tmp_path):
    comparison_config = _write_comparison_config(tmp_path)
    manifest = _write_manifest(tmp_path)
    panels = tmp_path / "panels.json"
    panels.write_text("{}")
    with pytest.raises(ValueError, match="uni2_pinned_revision"):
        suite_module.prepare_wae_gan_uni2_ablation_suite(
            comparison_config=str(comparison_config), manifest=str(manifest),
            train_gene_panels=str(panels), output_root=str(tmp_path / "suite"),
            uni2_pinned_revision="",
        )


def test_matched_except_declared_rejects_an_undeclared_divergence():
    control = {"model": {"regularizer": "gan"}, "loss": {"pcc_weight": 0.1}}
    other = {"model": {"regularizer": "mmd"}, "loss": {"pcc_weight": 0.1}}
    with pytest.raises(ValueError, match="undeclared divergence"):
        suite_module._assert_matched_except_declared(control, other, "wae_he_gan_uni2")


def test_matched_except_declared_allows_only_declared_image_encoder_fields():
    control = {"data": {"image_encoder": "gigapath", "tile_encoder_revision": "abc"}}
    other = {"data": {"image_encoder": "uni2", "uni2_pinned_revision": "def", "tile_encoder_revision": "abc"}}
    suite_module._assert_matched_except_declared(control, other, "wae_he_gan_uni2")
