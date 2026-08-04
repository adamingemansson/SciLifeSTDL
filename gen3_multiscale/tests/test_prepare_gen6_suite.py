from pathlib import Path

import pytest
import yaml

from gen3_multiscale.scripts.prepare_gen6_suite import prepare_gen6_suite
from gen3_multiscale.scripts.run_gen6_queues import run_queues


def test_prepare_gen6_suite_writes_ten_matched_configs_without_launching(tmp_path, monkeypatch):
    base = {
        "experiment_name": "base",
        "model": {"arm": "gen4c", "kind": "conditioner", "params": {
            "image_feature_dim": 1536, "gex_context_embedding_dim": 3072,
            "hidden_dim": 64, "n_heads": 4, "n_blocks": 1,
        }},
        "masking": {"strata": [{"name": "small", "shape": "circle", "radius_range": [3, 5]}]},
        "data": {"hest_data_dir": "data", "hest_cache_dir": "cache",
                 "gex_feature_dim": 32, "tile_encoder_revision": "a" * 40,
                 "gen3_manifest_path": None},
        "evaluation": {"patient_level_aggregation": True},
        "required_fingerprints": {},
        "training": {"seed": 0, "checkpoint_dir": "old", "total_steps": 1,
                     "max_wall_clock_hours": 1, "lr": 1e-4},
    }
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base))
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    panels = tmp_path / "train_gene_panels.json"
    panels.write_text("{}")
    monkeypatch.setattr(
        "gen3_multiscale.scripts.prepare_gen6_suite.load_dataset_manifest",
        lambda _path: {"gene_panel": ["g1", "g2"]},
    )
    monkeypatch.setattr(
        "gen3_multiscale.scripts.prepare_gen6_suite.load_train_derived_gene_panels",
        lambda _path, _manifest: {
            "artifact_sha256": "panel-sha256",
            "panels": {
                "train_log1p_variance_top50": ["g1"],
                "train_log1p_variance_top200": ["g1", "g2"],
            },
        },
    )
    fingerprints = {
        "uni2_checkpoint": "uni2.bin", "uni2_revision": "b" * 40,
        "uni2_package_version": "1", "uni2_preprocessing_spec": "spec",
        "scfoundation_checkpoint": "scf.ckpt", "scfoundation_vocab": "vocab.tsv",
        "scfoundation_package_version": "1", "scfoundation_preprocessing_spec": "spec",
        "gigapath_checkpoint": "slide.pth", "stpath_checkpoint": "stpath.pth",
        "stpath_gene_vocab": "genes.json",
    }
    root = tmp_path / "suite"
    plan = prepare_gen6_suite(
        comparison_config=str(base_path), manifest=str(manifest),
        train_gene_panels=str(panels), output_root=str(root),
        fingerprints=fingerprints, hours=8,
    )
    assert set(plan["arms"]) == {f"gen6{x}" for x in "abcdefghij"}
    assert not list(root.rglob("*.pid"))
    for arm in plan["arms"]:
        config = yaml.safe_load((root / "configs" / f"{arm}.yaml").read_text())
        assert config["training"]["max_wall_clock_hours"] == 8
        assert config["data"]["gen3_manifest_path"] == str(manifest)
        assert config["evaluation"]["train_gene_panel_artifact"] == str(panels.resolve())
        assert config["loss"]["primary_mode"] == "rmse_pcc"
    assert plan["train_gene_panels"] == {
        "path": str(panels.resolve()),
        "artifact_sha256": "panel-sha256",
        "panels": ["train_log1p_variance_top200", "train_log1p_variance_top50"],
    }
    queue_plan = run_queues(str(root), ["0", "2", "3", "5"], list(plan["arms"]), dry_run=True)
    assert sum(len(queue) for queue in queue_plan["queues"].values()) == 10
    assert not list(root.glob("*.pid"))


def test_prepare_gen6_suite_rejects_a_staged_comparison_config(tmp_path):
    base = {
        "model": {"arm": "gen5c", "kind": "latent_flow", "params": {}},
        "masking": {"strata": [{"name": "small"}]},
        "data": {}, "evaluation": {}, "required_fingerprints": {}, "training": {},
    }
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base))
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="deterministic conditioner"):
        prepare_gen6_suite(
            comparison_config=str(base_path), manifest=str(manifest),
            train_gene_panels=str(tmp_path / "panels.json"),
            output_root=str(tmp_path / "suite"), fingerprints={}, hours=8,
        )


def test_prepare_gen6_suite_rejects_artifact_without_required_hvg_panels(tmp_path, monkeypatch):
    base = {
        "model": {"arm": "gen4c", "kind": "conditioner", "params": {}},
        "data": {}, "evaluation": {}, "required_fingerprints": {}, "training": {},
    }
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base))
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    panels = tmp_path / "panels.json"
    panels.write_text("{}")
    monkeypatch.setattr(
        "gen3_multiscale.scripts.prepare_gen6_suite.load_dataset_manifest",
        lambda _path: {},
    )
    monkeypatch.setattr(
        "gen3_multiscale.scripts.prepare_gen6_suite.load_train_derived_gene_panels",
        lambda _path, _manifest: {
            "artifact_sha256": "panel-sha256",
            "panels": {"train_log1p_variance_top50": ["g1"]},
        },
    )

    with pytest.raises(ValueError, match="HVG-50 and HVG-200"):
        prepare_gen6_suite(
            comparison_config=str(base_path), manifest=str(manifest),
            train_gene_panels=str(panels), output_root=str(tmp_path / "suite"),
            fingerprints={}, hours=8,
        )
