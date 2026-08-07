import json

import pytest
import yaml

from gen3_multiscale.scripts import prepare_wae_mmd_geneencoder_ablation_suite as suite_module

UNI2_REVISION = "d517a8dd47902dd7c308b3c36f63bce47e7b9a43"


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
    manifest_path.write_text(json.dumps({
        "gene_panel": ["G0", "G1"], "train_sample_ids": ["S0"], "validation_sample_ids": ["V0", "V1"],
    }))
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


def _scfoundation_basis_path(tmp_path):
    scfoundation_basis_path = tmp_path / "basis_scfoundation.pt"
    scfoundation_basis_path.write_text("stub")
    return scfoundation_basis_path


def test_prepare_writes_the_four_factorial_arms_with_the_right_overrides(tmp_path):
    comparison_config = _write_comparison_config(tmp_path)
    manifest = _write_manifest(tmp_path)
    panels = tmp_path / "panels.json"
    panels.write_text("{}")
    output_root = tmp_path / "suite"
    scfoundation_basis_path = _scfoundation_basis_path(tmp_path)

    plan = suite_module.prepare_wae_mmd_geneencoder_ablation_suite(
        comparison_config=str(comparison_config), manifest=str(manifest),
        train_gene_panels=str(panels), output_root=str(output_root),
        scfoundation_basis_path=str(scfoundation_basis_path), uni2_pinned_revision=UNI2_REVISION,
        latent_dim=64, hours=8.0, gpus=(0, 1, 2, 3), cpu_threads=12,
    )

    assert set(plan["arms"]) == set(suite_module.ARM_ORDER)
    assert len(suite_module.ARM_ORDER) == 4
    configs = {
        arm: yaml.safe_load((output_root / "configs" / f"{arm}.yaml").read_text())
        for arm in suite_module.ARM_ORDER
    }

    for arm_config in configs.values():
        assert arm_config["model"]["regularizer"] == "mmd"
        assert arm_config["model"]["params"]["latent_dim"] == 64
        assert arm_config["data"]["image_encoder"] == "uni2"
        assert arm_config["data"]["uni2_pinned_revision"] == UNI2_REVISION

    scf_film = configs["wae_he_mmd_geneencoder_scfoundation_film"]
    assert scf_film["model"]["params"]["gene_encoder_source"] == "frozen_table"
    assert scf_film["model"]["params"]["encoder_conditioning"] == "film"
    assert scf_film["data"]["gene_encoder_table_path"] == str(scfoundation_basis_path)

    scf_nofilm = configs["wae_he_mmd_geneencoder_scfoundation_nofilm"]
    assert scf_nofilm["model"]["params"]["gene_encoder_source"] == "frozen_table"
    assert scf_nofilm["model"]["params"]["encoder_conditioning"] == "none"
    assert "film_layers" not in scf_nofilm["model"]["params"]
    assert scf_nofilm["data"]["gene_encoder_table_path"] == str(scfoundation_basis_path)

    mlp_film = configs["wae_he_mmd_geneencoder_mlp_film"]
    assert mlp_film["model"]["params"]["gene_encoder_source"] == "linear"
    assert mlp_film["model"]["params"]["encoder_conditioning"] == "film"
    assert "gene_encoder_table_path" not in mlp_film["data"]

    mlp_nofilm = configs["wae_he_mmd_geneencoder_mlp_nofilm"]
    assert mlp_nofilm["model"]["params"]["gene_encoder_source"] == "linear"
    assert mlp_nofilm["model"]["params"]["encoder_conditioning"] == "none"
    assert "gene_encoder_table_path" not in mlp_nofilm["data"]

    for arm_config in configs.values():
        whole_slide = arm_config["evaluation"]["whole_slide_validation"]
        assert whole_slide["enabled"] is True
        assert whole_slide["max_slides"] == 2

    checkpoint_dirs = {config["training"]["checkpoint_dir"] for config in configs.values()}
    log_dirs = {config["evaluation"]["tensorboard"]["log_dir"] for config in configs.values()}
    assert len(checkpoint_dirs) == 4
    assert len(log_dirs) == 4

    pointer = tmp_path / "LATEST_WAE_MMD_GENEENCODER_ABLATION_SUITE_ROOT.txt"
    assert pointer.read_text().strip() == str(output_root.resolve())


def test_prepare_honors_uni2_spot_feature_cache_dir_override(tmp_path):
    comparison_config = _write_comparison_config(tmp_path)
    manifest = _write_manifest(tmp_path)
    panels = tmp_path / "panels.json"
    panels.write_text("{}")
    scfoundation_basis_path = _scfoundation_basis_path(tmp_path)
    output_root = tmp_path / "suite"

    suite_module.prepare_wae_mmd_geneencoder_ablation_suite(
        comparison_config=str(comparison_config), manifest=str(manifest),
        train_gene_panels=str(panels), output_root=str(output_root),
        scfoundation_basis_path=str(scfoundation_basis_path), uni2_pinned_revision=UNI2_REVISION,
        uni2_spot_feature_cache_dir="/data/custom_uni2_cache",
    )
    config = yaml.safe_load(
        (output_root / "configs" / "wae_he_mmd_geneencoder_mlp_nofilm.yaml").read_text()
    )
    assert config["data"]["gen3_uni2_spot_feature_cache_dir"] == "/data/custom_uni2_cache"


def test_prepare_refuses_to_overwrite_an_existing_suite_root(tmp_path):
    comparison_config = _write_comparison_config(tmp_path)
    manifest = _write_manifest(tmp_path)
    panels = tmp_path / "panels.json"
    panels.write_text("{}")
    output_root = tmp_path / "suite"
    output_root.mkdir()
    scfoundation_basis_path = _scfoundation_basis_path(tmp_path)
    with pytest.raises(FileExistsError):
        suite_module.prepare_wae_mmd_geneencoder_ablation_suite(
            comparison_config=str(comparison_config), manifest=str(manifest),
            train_gene_panels=str(panels), output_root=str(output_root),
            scfoundation_basis_path=str(scfoundation_basis_path), uni2_pinned_revision=UNI2_REVISION,
        )


def test_prepare_rejects_wrong_gpu_count(tmp_path):
    comparison_config = _write_comparison_config(tmp_path)
    manifest = _write_manifest(tmp_path)
    panels = tmp_path / "panels.json"
    panels.write_text("{}")
    scfoundation_basis_path = _scfoundation_basis_path(tmp_path)
    with pytest.raises(ValueError, match="gpus"):
        suite_module.prepare_wae_mmd_geneencoder_ablation_suite(
            comparison_config=str(comparison_config), manifest=str(manifest),
            train_gene_panels=str(panels), output_root=str(tmp_path / "suite"),
            scfoundation_basis_path=str(scfoundation_basis_path), uni2_pinned_revision=UNI2_REVISION,
            gpus=(0, 1),
        )


def test_prepare_rejects_a_blank_scfoundation_basis_path(tmp_path):
    comparison_config = _write_comparison_config(tmp_path)
    manifest = _write_manifest(tmp_path)
    panels = tmp_path / "panels.json"
    panels.write_text("{}")
    with pytest.raises(ValueError, match="scfoundation_basis_path"):
        suite_module.prepare_wae_mmd_geneencoder_ablation_suite(
            comparison_config=str(comparison_config), manifest=str(manifest),
            train_gene_panels=str(panels), output_root=str(tmp_path / "suite"),
            scfoundation_basis_path="", uni2_pinned_revision=UNI2_REVISION,
        )


def test_prepare_rejects_a_blank_uni2_pinned_revision(tmp_path):
    comparison_config = _write_comparison_config(tmp_path)
    manifest = _write_manifest(tmp_path)
    panels = tmp_path / "panels.json"
    panels.write_text("{}")
    scfoundation_basis_path = _scfoundation_basis_path(tmp_path)
    with pytest.raises(ValueError, match="uni2_pinned_revision"):
        suite_module.prepare_wae_mmd_geneencoder_ablation_suite(
            comparison_config=str(comparison_config), manifest=str(manifest),
            train_gene_panels=str(panels), output_root=str(tmp_path / "suite"),
            scfoundation_basis_path=str(scfoundation_basis_path), uni2_pinned_revision="",
        )


def test_prepare_rejects_non_positive_latent_dim(tmp_path):
    comparison_config = _write_comparison_config(tmp_path)
    manifest = _write_manifest(tmp_path)
    panels = tmp_path / "panels.json"
    panels.write_text("{}")
    scfoundation_basis_path = _scfoundation_basis_path(tmp_path)
    with pytest.raises(ValueError, match="latent_dim"):
        suite_module.prepare_wae_mmd_geneencoder_ablation_suite(
            comparison_config=str(comparison_config), manifest=str(manifest),
            train_gene_panels=str(panels), output_root=str(tmp_path / "suite"),
            scfoundation_basis_path=str(scfoundation_basis_path), uni2_pinned_revision=UNI2_REVISION,
            latent_dim=0,
        )


def test_allowed_divergent_keys_include_film_keys_only_across_a_film_boundary():
    same_film_status = suite_module._allowed_divergent_keys_between(
        "wae_he_mmd_geneencoder_scfoundation_film", "wae_he_mmd_geneencoder_mlp_film",
    )
    assert ("model", "params", "encoder_conditioning") not in same_film_status

    crossing_film_status = suite_module._allowed_divergent_keys_between(
        "wae_he_mmd_geneencoder_scfoundation_film", "wae_he_mmd_geneencoder_scfoundation_nofilm",
    )
    assert ("model", "params", "encoder_conditioning") in crossing_film_status
    assert ("model", "params", "film_layers") in crossing_film_status


def test_matched_except_declared_rejects_an_undeclared_divergence():
    control = {"model": {"regularizer": "gan"}, "loss": {"pcc_weight": 0.1}}
    other = {"model": {"regularizer": "mmd"}, "loss": {"pcc_weight": 0.1}}
    with pytest.raises(ValueError, match="undeclared divergence"):
        suite_module._assert_matched_except_declared(control, other, "wae_he_mmd_geneencoder_mlp_film", set())
