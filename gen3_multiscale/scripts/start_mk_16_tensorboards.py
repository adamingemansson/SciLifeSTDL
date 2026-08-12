#!/usr/bin/env python3
"""Start one TensorBoard server for each four-arm batch in an MK master suite."""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def _free_port(host: str) -> int:
    with socket.socket() as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _port_is_available(host: str, port: int) -> bool:
    with socket.socket() as sock:
        try:
            sock.bind((host, int(port)))
        except OSError:
            return False
    return True


def _pid_is_alive(pid: object) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def start_tensorboards(suite_root: str, *, host: str = "127.0.0.1",
                       ports: tuple[int, ...] | None = None,
                       dry_run: bool = False) -> dict:
    root = Path(suite_root).expanduser().resolve()
    plan = json.loads((root / "master_plan.json").read_text())
    if plan.get("kind") != "mk_16_arm_two_wave_suite":
        raise ValueError("master_plan.json is not an MK 16-arm suite")
    control = root / "control" / "tensorboard"
    control.mkdir(parents=True, exist_ok=True)
    status_path = control / "servers.json"
    if status_path.exists() and not dry_run:
        previous = json.loads(status_path.read_text())
        live = {
            batch: record.get("pid") for batch, record in previous.items()
            if _pid_is_alive(record.get("pid"))
        }
        if live:
            raise RuntimeError(
                f"TensorBoard servers are already running for this suite: {live}"
            )
    records = {}
    batch_names = sorted(plan["tensorboard_groups"])
    if ports is not None:
        if len(ports) != len(batch_names) or len(set(ports)) != len(ports):
            raise ValueError(
                f"ports must contain {len(batch_names)} distinct values, one per batch"
            )
        if any(not 1 <= int(port) <= 65535 for port in ports):
            raise ValueError("TensorBoard ports must be in [1, 65535]")
    for batch in batch_names:
        group = plan["tensorboard_groups"][batch]
        logdir = Path(group["logdir"])
        if not logdir.is_dir():
            raise FileNotFoundError(logdir)
        records[batch] = {"logdir": str(logdir), "arms": group["arms"]}
    assigned_ports = dict(zip(batch_names, ports)) if ports is not None else {}
    for batch in batch_names:
        port = int(assigned_ports[batch]) if batch in assigned_ports else _free_port(host)
        if not dry_run and not _port_is_available(host, port):
            raise RuntimeError(f"requested TensorBoard port {port} for {batch} is unavailable")
        records[batch]["port"] = port
    if dry_run:
        return {"dry_run": True, "suite_root": str(root), "groups": records}

    processes = []
    for batch, record in records.items():
        port = int(record["port"])
        log = control / f"{batch}.log"
        handle = open(log, "a", buffering=1)
        process = subprocess.Popen(
            [sys.executable, "-m", "tensorboard.main", "--logdir", record["logdir"],
             "--host", host, "--port", str(port), "--load_fast=false"],
            stdout=handle, stderr=subprocess.STDOUT,
        )
        processes.append((batch, process, handle, log, port))
    time.sleep(3)
    failed = []
    for batch, process, handle, log, port in processes:
        handle.close()
        if process.poll() is not None:
            failed.append(f"{batch} (see {log})")
            continue
        records[batch].update({
            "pid": process.pid, "port": port, "host": host,
            "url": f"http://{host}:{port}/", "log": str(log),
        })
    if failed:
        for _batch, process, _handle, _log, _port in processes:
            if process.poll() is None:
                process.terminate()
        raise RuntimeError(f"TensorBoard startup failed: {failed}")
    temporary = status_path.with_name(f"{status_path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(records, indent=2, sort_keys=True))
    os.replace(temporary, status_path)
    return {"suite_root": str(root), "groups": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--ports",
        help="comma-separated fixed ports in Batch A,B,C,D order; otherwise choose free ports",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    ports = (
        tuple(int(value.strip()) for value in args.ports.split(",") if value.strip())
        if args.ports else None
    )
    print(json.dumps(start_tensorboards(
        args.suite_root, host=args.host, ports=ports, dry_run=args.dry_run,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
