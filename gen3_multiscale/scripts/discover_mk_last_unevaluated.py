#!/usr/bin/env python3
"""Select the newest completed MK checkpoints lacking a complete current evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gen3_multiscale.scripts.run_mk_16_pending_evaluations import (
    EXPECTED_FIXED_ITEMS,
    EXPECTED_WHOLE_SLIDES,
    _load_jobs,
)
from gen3_multiscale.training import checkpoint as checkpoint_module


def _complete_evaluations(results_root: Path) -> set[tuple[str, int]]:
    """Return (resolved config, checkpoint step) pairs with both final scopes."""
    complete: set[tuple[str, int]] = set()
    for path in results_root.rglob("*_validation.json"):
        try:
            report = json.loads(path.read_text())
        except Exception:
            continue
        if report.get("kind") != "conditional_wae_supervisor_evaluation":
            continue
        whole = report.get("whole_slide_structured_field_evaluation") or {}
        if int(report.get("n_items", -1)) != EXPECTED_FIXED_ITEMS:
            continue
        if len(whole.get("per_slide_records") or []) != EXPECTED_WHOLE_SLIDES:
            continue
        if report.get("query_gex_visible") is not False:
            continue
        config = report.get("config_path")
        step = report.get("checkpoint_step")
        if config and step is not None:
            complete.add((str(Path(config).expanduser().resolve()), int(step)))
    return complete


def _checkpoint_time(checkpoint_dir: str) -> float:
    identity = checkpoint_module.resolve_checkpoint_identity(Path(checkpoint_dir) / "best")
    candidates = [identity.resolved_dir]
    candidates.extend(identity.resolved_dir.glob("*"))
    return max(path.stat().st_mtime for path in candidates)


def discover(
    *, results_root: str, output: str, count: int = 16,
) -> dict[str, Any]:
    if count < 1:
        raise ValueError("count must be positive")
    root = Path(results_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"results root is missing: {root}")
    suite_roots = sorted({path.parent.resolve() for path in root.rglob("suite_plan.json")})
    all_jobs = []
    rejected_suites = []
    for suite_root in suite_roots:
        try:
            all_jobs.extend(_load_jobs([suite_root], expected_arms=None))
        except Exception as error:
            rejected_suites.append({
                "suite_root": str(suite_root),
                "reason": f"{type(error).__name__}: {error}",
            })
    evaluated = _complete_evaluations(root)
    candidates = []
    seen = set()
    for job in all_jobs:
        identity = (str(Path(job["config"]).resolve()), int(job["checkpoint_step"]))
        if identity in evaluated or identity in seen:
            continue
        seen.add(identity)
        row = dict(job)
        row["checkpoint_completed_mtime"] = _checkpoint_time(job["checkpoint_dir"])
        candidates.append(row)
    candidates.sort(
        key=lambda row: (row["checkpoint_completed_mtime"], row["arm"]),
        reverse=True,
    )
    if len(candidates) < count:
        raise ValueError(
            f"only {len(candidates)} completed, currently unevaluated MK checkpoints found; "
            f"cannot select {count}"
        )
    selected = []
    selected_arms = set()
    for row in candidates:
        if row["arm"] in selected_arms:
            continue
        selected.append(row)
        selected_arms.add(row["arm"])
        if len(selected) == count:
            break
    if len(selected) < count:
        raise ValueError(
            f"only {len(selected)} uniquely named, completed, currently unevaluated "
            f"MK checkpoints found; cannot select {count}"
        )
    selected_suite_roots = sorted({row["suite_root"] for row in selected})
    payload = {
        "kind": "mk_pending_evaluation_job_manifest",
        "version": 1,
        "results_root": str(root),
        "selection": "newest_best_checkpoint_mtime_without_complete_matching_evaluation",
        "count": count,
        "suite_roots": selected_suite_roots,
        "jobs": selected,
        "rejected_suites": rejected_suites,
    }
    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()
    payload = discover(
        results_root=args.results_root, output=args.output, count=args.count,
    )
    for index, job in enumerate(payload["jobs"], start=1):
        print(
            f"{index:>2}. {job['arm']:<40} step={job['checkpoint_step']:<8} "
            f"suite={job['suite_kind']}"
        )
    print(f"\nManifest: {Path(args.output).expanduser().resolve()}")


if __name__ == "__main__":
    main()
