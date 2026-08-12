import json
from pathlib import Path

from gen3_multiscale.scripts import prepare_mk_16_arm_suite as prepare_module
from gen3_multiscale.scripts.run_mk_16_arm_suite import run_master_suite
from gen3_multiscale.scripts.monitor_mk_16_arm_suite import monitor_snapshot
from gen3_multiscale.scripts.start_mk_16_tensorboards import start_tensorboards


GPU_ORDER = (0, 2, 3, 5)


def _fake_plan(output_root, arms, gpus, family=None):
    root = Path(output_root)
    (root / "configs").mkdir(parents=True)
    (root / "checkpoints").mkdir()
    (root / "logs").mkdir()
    (root / "tensorboard").mkdir()
    records = {}
    for arm, gpu in zip(arms, gpus):
        config = root / "configs" / f"{arm}.yaml"
        config.write_text("model: {}\n")
        (root / "tensorboard" / arm).mkdir()
        records[arm] = {
            "config": str(config), "gpu": gpu,
            "checkpoint_dir": str(root / "checkpoints" / arm),
            "tensorboard_log_dir": str(root / "tensorboard" / arm),
        }
    plan = {"kind": "fake", "family": family, "arm_order": list(arms), "arms": records}
    (root / "suite_plan.json").write_text(json.dumps(plan))
    return plan


def test_master_suite_has_two_waves_two_processes_per_gpu_and_four_tensorboards(
    tmp_path, monkeypatch,
):
    def fake_architecture(**kwargs):
        return _fake_plan(
            kwargs["output_root"],
            ("mk_local_wae_mmd", "mk_spatial_deterministic",
             "mk_local_conditional_wae_mmd", "mk_spatial_conditional_wae_mmd"),
            kwargs["gpus"],
        )

    def fake_structured(**kwargs):
        prefixes = {"deterministic": "det", "standard_wae": "wae", "conditional_wae": "cwae"}
        prefix = prefixes[kwargs["family"]]
        arms = tuple(f"{prefix}_{suffix}" for suffix in ("within", "between", "gradient", "combined"))
        return _fake_plan(kwargs["output_root"], arms, kwargs["gpus"], kwargs["family"])

    monkeypatch.setattr(prepare_module, "prepare_mk_architecture_suite", fake_architecture)
    monkeypatch.setattr(prepare_module, "prepare_mk_structured_field_suite", fake_structured)
    root = tmp_path / "master"
    plan = prepare_module.prepare_mk_16_arm_suite(
        comparison_config="comparison.yaml", manifest="manifest.json",
        train_gene_panels="panels.json", centered_gene_structure="structure.pt",
        output_root=str(root), uni2_pinned_revision="pinned",
        uni2_spot_feature_cache_dir="cache", gpus=GPU_ORDER,
    )
    assert len(plan["waves"]) == 2
    assert len(plan["tensorboard_groups"]) == 4
    assert sum(len(wave["arms"]) for wave in plan["waves"]) == 16
    arm_names = [record["arm"] for wave in plan["waves"] for record in wave["arms"]]
    assert len(set(arm_names)) == 16
    assert all(len(group["arms"]) == 4 for group in plan["tensorboard_groups"].values())
    for wave in plan["waves"]:
        assert len(wave["arms"]) == 8
        assert wave["gpu_process_counts"] == {str(gpu): 2 for gpu in GPU_ORDER}
    dry = run_master_suite(str(root), dry_run=True)
    assert [len(wave["arms"]) for wave in dry["waves"]] == [8, 8]
    tensorboards = start_tensorboards(
        str(root), ports=(55002, 55003, 55004, 55005), dry_run=True,
    )
    assert set(tensorboards["groups"]) == set(plan["tensorboard_groups"])
    assert [
        tensorboards["groups"][batch]["port"]
        for batch in sorted(tensorboards["groups"])
    ] == [55002, 55003, 55004, 55005]
    snapshot = monitor_snapshot(str(root))
    assert "wave_1" in snapshot and "wave_2" in snapshot
    assert snapshot.count("GPU=0") == 4


def test_launcher_refuses_duplicate_status_without_explicit_resume(
    tmp_path, monkeypatch,
):
    def fake_architecture(**kwargs):
        return _fake_plan(
            kwargs["output_root"],
            ("mk_local_wae_mmd", "mk_spatial_deterministic",
             "mk_local_conditional_wae_mmd", "mk_spatial_conditional_wae_mmd"),
            kwargs["gpus"],
        )

    def fake_structured(**kwargs):
        prefix = {"deterministic": "det", "standard_wae": "wae", "conditional_wae": "cwae"}[
            kwargs["family"]
        ]
        return _fake_plan(
            kwargs["output_root"],
            tuple(f"{prefix}_{suffix}" for suffix in ("within", "between", "gradient", "combined")),
            kwargs["gpus"], kwargs["family"],
        )

    monkeypatch.setattr(prepare_module, "prepare_mk_architecture_suite", fake_architecture)
    monkeypatch.setattr(prepare_module, "prepare_mk_structured_field_suite", fake_structured)
    root = tmp_path / "master"
    prepare_module.prepare_mk_16_arm_suite(
        comparison_config="comparison.yaml", manifest="manifest.json",
        train_gene_panels="panels.json", centered_gene_structure="structure.pt",
        output_root=str(root), uni2_pinned_revision="pinned",
        uni2_spot_feature_cache_dir="cache", gpus=GPU_ORDER,
    )
    status = root / "control" / "training_status.json"
    status.write_text("{}")
    try:
        run_master_suite(str(root))
    except FileExistsError as exc:
        assert "duplicate launch" in str(exc)
    else:
        raise AssertionError("duplicate status should fail closed")
