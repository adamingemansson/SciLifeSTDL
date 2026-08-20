import json
import sys

import pytest

from gen3_multiscale.scripts import discover_mk_last_unevaluated as discovery
from gen3_multiscale.scripts import run_mk_16_pending_evaluations as launcher


def _jobs(count=16):
    return [
        {
            "arm": f"arm_{index:02d}",
            "suite_kind": "mk_gene_field_suite" if index >= 8 else "mk_residual_wae_suite",
            "suite_root": "/suite",
            "config": f"/suite/configs/arm_{index:02d}.yaml",
            "checkpoint_dir": f"/suite/checkpoints/arm_{index:02d}",
            "checkpoint_step": 100,
            "diagnose_latent": index < 8,
        }
        for index in range(count)
    ]


def _complete_report(*, diagnose_latent):
    arms = {"model": {}, "conditional_mean": {}}
    if diagnose_latent:
        arms.update({
            "posterior_reconstruction": {}, "zero_latent": {},
            "shuffled_posterior": {},
        })
    return {
        "n_items": 448,
        "n_samples": 14,
        "query_gex_visible": False,
        "diagnose_latent": diagnose_latent,
        "checkpoint_step": 100,
        "prediction_roles": {"primary_point_prediction": "model"},
        "per_arm_patient_aggregated_metrics": arms,
        "whole_slide_structured_field_evaluation": {
            "target_gex_visible_to_model": False,
            "per_slide_records": [{"sample_id": f"s{i}"} for i in range(14)],
        },
    }


def test_report_audit_requires_complete_fixed_and_whole_slide_scopes(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_complete_report(diagnose_latent=True)))
    result = launcher._audit_report(path, diagnose_latent=True)
    assert result["fixed_items"] == 448
    assert result["whole_slides"] == 14

    broken = _complete_report(diagnose_latent=True)
    broken["whole_slide_structured_field_evaluation"]["per_slide_records"].pop()
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="whole slides=13"):
        launcher._audit_report(path, diagnose_latent=True)


def test_report_audit_requires_all_latent_diagnostic_paths(tmp_path):
    report = _complete_report(diagnose_latent=True)
    report["per_arm_patient_aggregated_metrics"].pop("shuffled_posterior")
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="shuffled_posterior"):
        launcher._audit_report(path, diagnose_latent=True)


def test_dry_run_distributes_four_jobs_to_each_gpu(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "_load_jobs", lambda *_args, **_kwargs: _jobs())
    output = tmp_path / "evaluation"
    result = launcher.run_evaluations(
        suite_roots=["/suite/a", "/suite/b", "/suite/c"],
        job_manifest=None,
        output_root=str(output), gpus=(0, 2, 3, 5), dry_run=True,
    )
    assert result["dry_run"] is True
    assignments = [row["gpu"] for row in result["jobs"].values()]
    assert {gpu: assignments.count(gpu) for gpu in (0, 2, 3, 5)} == {
        0: 4, 2: 4, 3: 4, 5: 4,
    }
    contract = json.loads((output / "evaluation_contract.json").read_text())
    assert contract["expected_fixed_items_per_arm"] == 448
    assert contract["expected_whole_slides_per_arm"] == 14


def test_failed_arm_does_not_block_later_arm_on_same_gpu(monkeypatch, tmp_path):
    jobs = _jobs(2)
    jobs[0]["diagnose_latent"] = False
    jobs[1]["diagnose_latent"] = False
    monkeypatch.setattr(launcher, "_load_jobs", lambda *_args, **_kwargs: jobs)

    def command(job, **_kwargs):
        code = 7 if job["arm"] == "arm_00" else 0
        return [sys.executable, "-c", f"raise SystemExit({code})"]

    monkeypatch.setattr(launcher, "_command", command)
    monkeypatch.setattr(
        launcher, "_audit_report",
        lambda *_args, **_kwargs: {"fixed_items": 448, "whole_slides": 14},
    )
    output = tmp_path / "evaluation"
    with pytest.raises(RuntimeError, match="arm_00"):
        launcher.run_evaluations(
            suite_roots=["/suite"], job_manifest=None, output_root=str(output),
            gpus=(0,), expected_arms=2,
        )
    status = json.loads((output / "evaluation_status.json").read_text())
    assert status["arm_00"]["status"] == "failed"
    assert status["arm_01"]["status"] == "finished"


def test_selected_manifest_refuses_checkpoint_changed_after_discovery(
    monkeypatch, tmp_path,
):
    suite = tmp_path / "suite"
    jobs = _jobs(2)
    for job in jobs:
        job["suite_root"] = str(suite)
        job["config"] = str(suite / "configs" / f"{job['arm']}.yaml")
    manifest = tmp_path / "selected.json"
    manifest.write_text(json.dumps({
        "kind": "mk_pending_evaluation_job_manifest",
        "suite_roots": [str(suite)],
        "jobs": jobs,
    }))
    monkeypatch.setattr(launcher, "_load_jobs", lambda *_args, **_kwargs: jobs)
    selected, roots = launcher._load_selected_jobs(manifest, expected_arms=2)
    assert [job["arm"] for job in selected] == ["arm_00", "arm_01"]
    assert roots == [str(suite)]

    changed = [dict(job) for job in jobs]
    changed[0]["checkpoint_step"] = 200
    monkeypatch.setattr(launcher, "_load_jobs", lambda *_args, **_kwargs: changed)
    with pytest.raises(ValueError, match="best checkpoint changed"):
        launcher._load_selected_jobs(manifest, expected_arms=2)


def test_discovery_selects_newest_exact_unevaluated_checkpoints(
    monkeypatch, tmp_path,
):
    results = tmp_path / "results"
    suites = [results / f"suite_{index}" for index in range(3)]
    for suite in suites:
        suite.mkdir(parents=True)
        (suite / "suite_plan.json").write_text("{}")

    jobs = []
    times = {}
    for index, suite in enumerate(suites):
        job = _jobs(1)[0]
        job["arm"] = f"arm_{index}"
        job["suite_root"] = str(suite.resolve())
        job["config"] = str((suite / "configs" / f"arm_{index}.yaml").resolve())
        job["checkpoint_dir"] = str(suite / "checkpoints" / f"arm_{index}")
        job["checkpoint_step"] = 100 + index
        jobs.append(job)
        times[job["checkpoint_dir"]] = 10.0 + index

    def load(suite_roots, **_kwargs):
        suite = str(suite_roots[0].resolve())
        return [job for job in jobs if job["suite_root"] == suite]

    evaluated = {(jobs[2]["config"], jobs[2]["checkpoint_step"])}
    monkeypatch.setattr(discovery, "_load_jobs", load)
    monkeypatch.setattr(discovery, "_complete_evaluations", lambda _root: evaluated)
    monkeypatch.setattr(
        discovery, "_checkpoint_time", lambda path: times[path],
    )
    output = tmp_path / "selected.json"
    payload = discovery.discover(
        results_root=str(results), output=str(output), count=2,
    )
    assert [job["arm"] for job in payload["jobs"]] == ["arm_1", "arm_0"]
    assert json.loads(output.read_text())["count"] == 2
