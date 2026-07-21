#!/usr/bin/env bash
# Eight distinct 40k component-isolation runs on a separately allocated
# eight-GPU server. This must not be used on the shared st-a100 allocation.
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-2}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

if pgrep -f -- 'src.training.train.*recovery_component40k_' >/dev/null; then
  echo "ERROR: component40k training processes are already running." >&2
  pgrep -af -- 'src.training.train.*recovery_component40k_' >&2 || true
  exit 2
fi

echo "===== component40k one-step smoke check ====="
SMOKETEST=1 \
SMOKE_STEPS=1 \
STAGE=component40k \
SERVER_PROFILE=dedicated8 \
RUN_ID="${RUN_ID}_smoke" \
LOG_ROOT="logs/recovery_suite/smoke_component40k_${RUN_ID}" \
GPU_IDS=0,1,2,3,4,5,6,7 \
CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/run_recovery_suite_8gpu.sh

# set -e prevents the full stage from starting after any smoke failure.
echo "===== component40k full 40k runs ====="
STAGE=component40k \
SERVER_PROFILE=dedicated8 \
RUN_ID="${RUN_ID}_full" \
LOG_ROOT="logs/recovery_suite/component40k_${RUN_ID}" \
GPU_IDS=0,1,2,3,4,5,6,7 \
CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/run_recovery_suite_8gpu.sh

echo "component40k completed successfully."
echo "Report: reports/recovery_suite/component40k_${RUN_ID}_full/summary.csv"
