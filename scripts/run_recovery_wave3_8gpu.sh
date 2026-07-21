#!/usr/bin/env bash
# Fail-closed Wave 3 launcher: one-step smoke check, then the eight full runs.
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-4}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

if pgrep -f -- 'src.training.train.*recovery_wave3_' >/dev/null; then
  echo "ERROR: Wave 3 training processes are already running." >&2
  pgrep -af -- 'src.training.train.*recovery_wave3_' >&2 || true
  exit 2
fi

echo "===== Wave 3 one-step smoke check ====="
SMOKETEST=1 \
SMOKE_STEPS=1 \
STAGE=wave3 \
RUN_ID="${RUN_ID}_smoke" \
LOG_ROOT="logs/recovery_suite/smoke_wave3_${RUN_ID}" \
GPU_IDS=0,1,2,3 \
CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/run_recovery_suite_8gpu.sh

# set -e prevents full training after any smoke failure.
echo "===== Wave 3 full runs ====="
STAGE=wave3 \
RUN_ID="${RUN_ID}_full" \
LOG_ROOT="logs/recovery_suite/wave3_${RUN_ID}" \
GPU_IDS=0,1,2,3 \
CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/run_recovery_suite_8gpu.sh

echo "Wave 3 completed successfully."
echo "Report: reports/recovery_suite/wave3_${RUN_ID}_full/summary.csv"
