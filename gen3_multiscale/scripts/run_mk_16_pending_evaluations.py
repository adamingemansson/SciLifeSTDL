#!/usr/bin/env python3
"""Evaluate the 16 pending MK arms in four fail-independent GPU queues.

The controller accepts any collection of immutable MK suite roots, but is
deliberately fail-closed by default: exactly 16 unique arms must be present.
Each GPU owns one sequential queue, so evaluation never launches more than one
process per requested GPU.  A failed arm does not block later arms on that GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from gen3_multiscale.conditional_wae.contract import static_audit_conditional_wae_config
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.training import checkpoint as checkpoint_module


ALLOWED_SUITE_KINDS = {
    "mk_decoder_width_suite",
    "mk_four_architecture_suite",
    "mk_gene_axis_suite",
    "mk_residual_wae_suite",
    "mk_specialist_wae_suite",
    "mk_gene_field_suite",
    "mk_structured_field_suite",
}
EXPECTED_FIXED_ITEMS = 448
EXPECTED_WHOLE_SLIDES = 14


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _load_jobs(suite_roots: list[Path], *, expected_arms: int) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw_root in suite_roots:
        root = raw_root.expanduser().resolve()
        plan_path = root / "suite_plan.json"
        if not plan_path.is_file():
            raise FileNotFoundError(f"suite plan is missing: {plan_path}")
        plan = json.loads(plan_path.read_text())
        kind = str(plan.get("kind"))
        if kind not in ALLOWED_SUITE_KINDS:
            raise ValueError(f"{plan_path}: unsupported suite kind {kind!r}")
        order = list(plan.get("arm_order") or [])
        if not order or set(order) != set((plan.get("arms") or {}).keys()):
            raise ValueError(f"{plan_path}: arm_order and arms do not match exactly")
        for arm in order:
            if arm in names:
                raise ValueError(f"duplicate arm across suites: {arm}")
            names.add(arm)
            item = plan["arms"][arm]
            config_path = Path(item["config"]).expanduser().resolve()
            checkpoint_dir = Path(item["checkpoint_dir"]).expanduser().resolve()
            if not config_path.is_file():
                raise FileNotFoundError(f"{arm}: config is missing: {config_path}")
            config = yaml.safe_load(config_path.read_text())
            if (config.get("model") or {}).get("arm") != arm:
                raise ValueError(f"{arm}: config model.arm does not match the suite plan")
            if (config.get("model") or {}).get("task") != "he_to_st":
                raise ValueError(f"{arm}: final MK evaluation requires task=he_to_st")
            if bool((config.get("model") or {}).get("include_observed_gex", False)):
                raise ValueError(f"{arm}: query GEX must be hidden")
            static_audit_conditional_wae_config(config)
            manifest = load_dataset_manifest(config["data"]["gen3_manifest_path"])
            if len(manifest.get("validation_sample_ids", [])) != EXPECTED_WHOLE_SLIDES:
                raise ValueError(
                    f"{arm}: expected {EXPECTED_WHOLE_SLIDES} validation slides, got "
                    f"{len(manifest.get('validation_sample_ids', []))}"
                )
            identity = checkpoint_module.resolve_checkpoint_identity(checkpoint_dir / "best")
            state = checkpoint_module.load_training_state(identity.resolved_dir)
            if int(state.get("step", 0)) < 1:
                raise ValueError(f"{arm}: best checkpoint has no positive training step")
            deterministic = bool((config["model"].get("params") or {}).get(
                "deterministic_only", False,
            ))
            jobs.append({
                "arm": arm,
                "suite_kind": kind,
                "suite_root": str(root),
                "config": str(config_path),
                "checkpoint_dir": str(checkpoint_dir),
                "checkpoint_step": int(state["step"]),
                "diagnose_latent": not deterministic,
            })
    if len(jobs) != expected_arms:
        raise ValueError(
            f"expected exactly {expected_arms} arms, discovered {len(jobs)}: "
            f"{[job['arm'] for job in jobs]}"
        )
    return jobs


def _command(
    job: dict[str, Any], *, output_root: Path, noise_ceiling: Path | None,
    allow_code_drift: bool,
) -> list[str]:
    arm = job["arm"]
    command = [
        sys.executable, "-u", "-m",
        "gen3_multiscale.evaluation.conditional_wae_evaluator",
        "--config", job["config"],
        "--checkpoint-dir", job["checkpoint_dir"],
        "--output", str(output_root / f"{arm}_validation.json"),
        "--split", "validation",
        "--n-masks-per-stratum-per-sample", "8",
        "--device", "cuda",
        "--per-gene-diagnostics-output", str(output_root / f"{arm}.per_gene.npz"),
    ]
    if job["diagnose_latent"]:
        command.append("--diagnose-latent")
    if noise_ceiling is not None:
        command += ["--noise-ceiling", str(noise_ceiling)]
    if allow_code_drift:
        command.append("--allow-code-drift")
    return command


def _audit_report(path: Path, *, diagnose_latent: bool) -> dict[str, Any]:
    report = json.loads(path.read_text())
    failures = []
    if int(report.get("n_items", -1)) != EXPECTED_FIXED_ITEMS:
        failures.append(f"fixed items={report.get('n_items')} != {EXPECTED_FIXED_ITEMS}")
    if int(report.get("n_samples", -1)) != EXPECTED_WHOLE_SLIDES:
        failures.append(f"fixed samples={report.get('n_samples')} != {EXPECTED_WHOLE_SLIDES}")
    whole = report.get("whole_slide_structured_field_evaluation") or {}
    records = whole.get("per_slide_records") or []
    if len(records) != EXPECTED_WHOLE_SLIDES:
        failures.append(f"whole slides={len(records)} != {EXPECTED_WHOLE_SLIDES}")
    if report.get("query_gex_visible") is not False:
        failures.append("query_gex_visible is not false")
    if whole.get("target_gex_visible_to_model") is not False:
        failures.append("whole-slide target GEX was not explicitly hidden")
    if bool(report.get("diagnose_latent")) != bool(diagnose_latent):
        failures.append("latent diagnostic state differs from the launch contract")
    if diagnose_latent:
        required = {"posterior_reconstruction", "zero_latent", "shuffled_posterior"}
        available = set((report.get("per_arm_patient_aggregated_metrics") or {}).keys())
        missing = required - available
        if missing:
            failures.append(f"missing latent diagnostic paths: {sorted(missing)}")
    if failures:
        raise ValueError(f"{path}: incomplete evaluation: {'; '.join(failures)}")
    return {
        "fixed_items": int(report["n_items"]),
        "whole_slides": len(records),
        "checkpoint_step": report.get("checkpoint_step"),
        "primary_prediction": (report.get("prediction_roles") or {}).get(
            "primary_point_prediction"
        ),
    }


def run_evaluations(
    *, suite_roots: list[str], output_root: str, gpus: tuple[int, ...],
    expected_arms: int = 16, cpu_threads: int = 8,
    noise_ceiling: str | None = None, allow_code_drift: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("gpus must be a non-empty list of distinct ids")
    if cpu_threads < 1 or expected_arms < 1:
        raise ValueError("cpu_threads and expected_arms must be positive")
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"evaluation root already exists: {output}")
    ceiling = Path(noise_ceiling).expanduser().resolve() if noise_ceiling else None
    if ceiling is not None and not ceiling.is_file():
        raise FileNotFoundError(f"noise-ceiling artifact is missing: {ceiling}")
    jobs = _load_jobs(
        [Path(value) for value in suite_roots], expected_arms=expected_arms,
    )
    for index, job in enumerate(jobs):
        job["gpu"] = int(gpus[index % len(gpus)])
    output.mkdir(parents=True)
    (output / "logs").mkdir()
    contract = {
        "kind": "mk_16_pending_evaluation",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "suite_roots": [str(Path(value).expanduser().resolve()) for value in suite_roots],
        "expected_arms": expected_arms,
        "expected_fixed_items_per_arm": EXPECTED_FIXED_ITEMS,
        "expected_whole_slides_per_arm": EXPECTED_WHOLE_SLIDES,
        "gpus": list(gpus),
        "cpu_threads_per_evaluation": cpu_threads,
        "noise_ceiling": str(ceiling) if ceiling else None,
        "jobs": jobs,
    }
    _atomic_json(output / "evaluation_contract.json", contract)
    commands = {
        job["arm"]: _command(
            job, output_root=output, noise_ceiling=ceiling,
            allow_code_drift=allow_code_drift,
        )
        for job in jobs
    }
    if dry_run:
        result = {job["arm"]: {"gpu": job["gpu"], "command": commands[job["arm"]]}
                  for job in jobs}
        _atomic_json(output / "dry_run.json", result)
        return {"dry_run": True, "output_root": str(output), "jobs": result}

    status_path = output / "evaluation_status.json"
    status = {
        job["arm"]: {
            "gpu": job["gpu"], "suite_kind": job["suite_kind"],
            "diagnose_latent": job["diagnose_latent"], "status": "queued",
        }
        for job in jobs
    }
    status_lock = threading.Lock()
    _atomic_json(status_path, status)
    queues: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        queues[int(job["gpu"])].append(job)

    def update(arm: str, **values: Any) -> None:
        with status_lock:
            status[arm].update(values)
            _atomic_json(status_path, status)

    def worker(gpu: int, queue: list[dict[str, Any]]) -> None:
        for job in queue:
            arm = job["arm"]
            log = output / "logs" / f"{arm}_validation.log"
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment["SCILIFESTDL_CPU_THREADS"] = str(cpu_threads)
            for variable in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            ):
                environment[variable] = str(cpu_threads)
            with log.open("a", buffering=1) as handle:
                process = subprocess.Popen(
                    commands[arm], cwd=Path(__file__).resolve().parents[2],
                    env=environment, stdout=handle, stderr=subprocess.STDOUT,
                )
                update(
                    arm, status="running", pid=process.pid, log=str(log),
                    started=time.time(), command=commands[arm],
                )
                returncode = process.wait()
            if returncode:
                update(
                    arm, status="failed", returncode=returncode,
                    finished=time.time(),
                )
                continue
            report_path = output / f"{arm}_validation.json"
            try:
                audit = _audit_report(
                    report_path, diagnose_latent=bool(job["diagnose_latent"]),
                )
            except Exception as error:  # recorded, then the queue continues
                update(
                    arm, status="failed_audit", returncode=0,
                    audit_error=f"{type(error).__name__}: {error}",
                    finished=time.time(),
                )
                continue
            update(
                arm, status="finished", returncode=0, audit=audit,
                report=str(report_path), finished=time.time(),
            )

    threads = [
        threading.Thread(target=worker, args=(gpu, queue), daemon=False)
        for gpu, queue in sorted(queues.items())
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failures = [arm for arm, row in status.items() if row["status"] != "finished"]
    result = {
        "ok": not failures,
        "output_root": str(output),
        "n_finished": len(status) - len(failures),
        "n_failed": len(failures),
        "failures": failures,
        "evaluation_status": str(status_path),
    }
    _atomic_json(output / "evaluation_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if failures:
        raise RuntimeError(f"MK evaluations failed or failed audit: {failures}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gpus", default="0,2,3,5")
    parser.add_argument("--expected-arms", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--noise-ceiling")
    parser.add_argument("--allow-code-drift", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_evaluations(
        suite_roots=args.suite_root, output_root=args.output_root,
        gpus=tuple(int(value) for value in args.gpus.split(",") if value.strip()),
        expected_arms=args.expected_arms, cpu_threads=args.cpu_threads,
        noise_ceiling=args.noise_ceiling,
        allow_code_drift=args.allow_code_drift, dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
