import json
from pathlib import Path

import pytest
import yaml

from gen3_multiscale.scripts.prepare_gen6b_stability_suite import prepare_gen6b_stability_suite


def _write_reference_config(path: Path, checkpoint_dir: Path) -> None:
    config = {
        "model": {
            "arm": "gen6b", "kind": "conditioner",
            "params": {"hidden_dim": 16, "n_heads": 2, "gene_encoder_type": "weighted_linear"},
        },
        "loss": {"primary_mode": "rmse_pcc", "pcc_weight": 0.1},
        "masking": {"strata": ["a", "b"]},
        "data": {"gen3_manifest_path": "/fake/manifest.json", "tile_encoder_revision": "0" * 40},
        "required_fingerprints": {"uni2_checkpoint": "/fake/uni2.pt"},
        "training": {
            "device": "cpu", "seed": 0, "lr": 1e-3, "total_steps": 100_000_000,
            "max_wall_clock_hours": 8.0, "checkpoint_dir": str(checkpoint_dir),
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def test_prepare_writes_three_matched_configs_differing_only_in_allowed_fields(tmp_path):
    reference_path = tmp_path / "gen6b.yaml"
    _write_reference_config(reference_path, tmp_path / "reference_ckpt")
    output_root = tmp_path / "gen6b_stability_20260805T000000Z"

    plan = prepare_gen6b_stability_suite(
        reference_config=str(reference_path), seeds=[0, 1, 2], output_root=str(output_root),
        hours=8.0, patience_validations=8, min_delta=0.0001,
    )

    assert plan["arm"] == "gen6b"
    assert plan["seeds"] == [0, 1, 2]
    assert set(plan["runs"]) == {"0", "1", "2"}

    configs = {
        seed: yaml.safe_load(Path(info["config"]).read_text())
        for seed, info in plan["runs"].items()
    }
    # Model/loss/masking/data sections are byte-identical across all three.
    for seed in ("1", "2"):
        assert configs[seed]["model"] == configs["0"]["model"]
        assert configs[seed]["loss"] == configs["0"]["loss"]
        assert configs[seed]["masking"] == configs["0"]["masking"]
        assert configs[seed]["data"] == configs["0"]["data"]
        assert configs[seed]["required_fingerprints"] == configs["0"]["required_fingerprints"]

    # Only the documented fields differ.
    assert configs["0"]["training"]["seed"] == 0
    assert configs["1"]["training"]["seed"] == 1
    assert configs["2"]["training"]["seed"] == 2
    assert configs["0"]["training"]["checkpoint_dir"] != configs["1"]["training"]["checkpoint_dir"]
    for seed in ("0", "1", "2"):
        assert configs[seed]["training"]["max_wall_clock_hours"] == 8.0
        assert configs[seed]["training"]["early_stopping"] == {
            "monitor": "validation_total", "mode": "min",
            "patience_validations": 8, "min_delta": 0.0001,
        }

    run_plan_on_disk = json.loads((output_root / "run_plan.json").read_text())
    assert run_plan_on_disk == plan
    pointer = tmp_path / "LATEST_GEN6B_STABILITY_SUITE_ROOT.txt"
    assert pointer.read_text().strip() == str(output_root.resolve())


def test_prepare_refuses_a_non_gen6b_reference_config(tmp_path):
    reference_path = tmp_path / "gen6a.yaml"
    reference_path.write_text(yaml.safe_dump({
        "model": {"arm": "gen6a"}, "training": {"checkpoint_dir": str(tmp_path / "ckpt")},
    }))
    with pytest.raises(ValueError, match="gen6b"):
        prepare_gen6b_stability_suite(
            reference_config=str(reference_path), seeds=[0, 1, 2],
            output_root=str(tmp_path / "out"),
        )


def test_prepare_refuses_fewer_than_two_distinct_seeds(tmp_path):
    reference_path = tmp_path / "gen6b.yaml"
    _write_reference_config(reference_path, tmp_path / "reference_ckpt")
    with pytest.raises(ValueError, match="distinct"):
        prepare_gen6b_stability_suite(
            reference_config=str(reference_path), seeds=[0, 0],
            output_root=str(tmp_path / "out"),
        )


def test_prepare_refuses_an_existing_output_root(tmp_path):
    reference_path = tmp_path / "gen6b.yaml"
    _write_reference_config(reference_path, tmp_path / "reference_ckpt")
    output_root = tmp_path / "already_exists"
    output_root.mkdir()
    with pytest.raises(FileExistsError):
        prepare_gen6b_stability_suite(
            reference_config=str(reference_path), seeds=[0, 1, 2], output_root=str(output_root),
        )


def test_assert_matched_except_catches_an_undeclared_divergence(tmp_path):
    """Adversarial: a stray per-seed model.params override (a bug the
    real suite-building loop must never introduce) must be caught by the
    same invariant check the suite runs on every pair of written configs,
    not silently pass."""
    from gen3_multiscale.scripts import prepare_gen6b_stability_suite as module

    reference_path = tmp_path / "gen6b.yaml"
    _write_reference_config(reference_path, tmp_path / "reference_ckpt")
    output_root = tmp_path / "gen6b_stability_test"
    plan = module.prepare_gen6b_stability_suite(
        reference_config=str(reference_path), seeds=[0, 1], output_root=str(output_root),
    )
    config_0 = yaml.safe_load(Path(plan["runs"]["0"]["config"]).read_text())
    config_0["model"]["params"]["hidden_dim"] = 999  # undeclared divergence
    config_1 = yaml.safe_load(Path(plan["runs"]["1"]["config"]).read_text())
    with pytest.raises(ValueError, match="model.params.hidden_dim"):
        module._assert_matched_except(
            config_0, config_1, allowed_diff_paths=module._ALLOWED_DIFF_PATHS, context="seed0 vs seed1",
        )
