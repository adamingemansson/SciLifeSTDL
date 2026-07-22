#!/usr/bin/env python3
"""Resolve and run one gene-aware suite entry with auditable logging."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from omegaconf import OmegaConf

from scripts.gene_aware_suite_lib import (
    DEFAULT_MATRIX, final_artifact_path, load_matrix, resolve_run, smoke_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    matrix = load_matrix(args.matrix)
    entry, cfg = resolve_run(matrix, args.index)
    if args.smoke:
        cfg = smoke_config(cfg, args.run_id)
    artifact = final_artifact_path(entry, cfg)
    if artifact.is_file() and not args.fresh:
        print(f"SKIP: {entry.name} ({artifact} exists)", flush=True)
        return

    log_root = Path(args.log_root)
    generated = log_root / "generated_configs"
    generated.mkdir(parents=True, exist_ok=True)
    config_path = generated / f"{entry.name}.yaml"
    OmegaConf.save(cfg, config_path)
    log_path = log_root / f"{entry.name}.log"
    command = [args.python_bin, "-m", "src.training.train", "--config", str(config_path)]
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "OMP_NUM_THREADS": str(args.threads),
        "MKL_NUM_THREADS": str(args.threads),
        "OPENBLAS_NUM_THREADS": str(args.threads),
        "NUMEXPR_NUM_THREADS": str(args.threads),
        "PYTHONUNBUFFERED": "1",
    })
    started = time.time()
    with log_path.open("w") as log:
        log.write("matrix_entry: " + json.dumps(OmegaConf.to_container(entry, resolve=True)) + "\n")
        log.write("command: " + " ".join(shlex.quote(part) for part in command) + "\n")
        log.write(f"started_at_utc: {dt.datetime.now(dt.timezone.utc).isoformat()}\n")
        log.flush()
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"finished_at_utc: {dt.datetime.now(dt.timezone.utc).isoformat()}\n")
        log.write(f"wall_seconds: {int(time.time() - started)}\n")
        log.write(f"exit_status: {result.returncode}\n")
    if result.returncode:
        print(f"FAILED: {entry.name}; log: {log_path}", file=sys.stderr, flush=True)
        raise SystemExit(result.returncode)
    if not artifact.is_file():
        print(f"FAILED: {entry.name} exited zero but did not create {artifact}", file=sys.stderr)
        raise SystemExit(3)
    print(f"DONE: {entry.name}", flush=True)


if __name__ == "__main__":
    main()
