#!/usr/bin/env python3
"""Step 6 deliverable: four-GPU SHORT DIAGNOSTIC launcher.

A thin wrapper around `training/launch_four_gpu_suite.py::launch_suite`,
called with `smoke_only=True` ALWAYS -- never `run_suite_with_smoke_gate`
(that function auto-starts a full run immediately after a passing smoke
gate, which is exactly what Adam's Step 6 instructions forbid: "Do not
start any training automatically"). This script structurally cannot start
a real multi-hour run: `smoke_only=True` is hardcoded below, not a CLI
flag, so there is no argument combination that promotes to a full run.

Runs one `--smoke` (one training step + one validation step) subprocess
per architecture config, one config per GPU, exactly like
launch_four_gpu_suite.py's own launch_suite -- reusing its already-audited
config-audit/fingerprint-check/process-group-cleanup machinery rather than
re-implementing any of it.

    python -m gen3_multiscale.scripts.step6_four_gpu_diagnostic \\
        --configs gen3_multiscale/configs/architecture1.yaml ... architecture4.yaml \\
        --gpus 0 1 2 3 --log-root /path/to/logs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

from gen3_multiscale.training.launch_four_gpu_suite import launch_suite


def run_four_gpu_smoke_diagnostic(
    config_paths: dict[str, Path], gpu_list: list[str], log_root: Path,
    threads_per_job: int = 4, skip_fingerprint_check: bool = False,
):
    named_configs = {
        name: OmegaConf.to_container(OmegaConf.load(path), resolve=True) for name, path in config_paths.items()
    }
    return launch_suite(
        named_configs, config_paths, gpu_list, Path(log_root), threads_per_job,
        smoke_only=True, skip_fingerprint_check=skip_fingerprint_check,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs=4, required=True, metavar="PATH",
                         help="Exactly four config YAML paths (architecture1.yaml .. architecture4.yaml).")
    parser.add_argument("--gpus", nargs=4, required=True, metavar="GPU_ID",
                         help="Exactly four GPU ids, one per --configs entry in order.")
    parser.add_argument("--threads-per-job", type=int, default=4)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--skip-fingerprint-check", action="store_true",
                         help="Only for local dry runs against a stub entrypoint -- never for a real launch.")
    args = parser.parse_args()

    if len(set(args.gpus)) != 4:
        raise ValueError(f"--gpus must name four DIFFERENT GPU ids, got {args.gpus!r}")
    config_paths = {Path(p).stem: Path(p) for p in args.configs}
    if len(config_paths) != 4:
        raise ValueError("--configs must name four DIFFERENT files (by stem)")

    result = run_four_gpu_smoke_diagnostic(
        config_paths, list(args.gpus), Path(args.log_root),
        threads_per_job=args.threads_per_job, skip_fingerprint_check=args.skip_fingerprint_check,
    )
    if not result.ok:
        print(f"FOUR-GPU SMOKE DIAGNOSTIC FAILED. Summary: {result.summary_path}", file=sys.stderr)
        sys.exit(1)
    print(f"four-GPU smoke diagnostic ok (one step per architecture, no full run started). Summary: {result.summary_path}")


if __name__ == "__main__":
    main()
