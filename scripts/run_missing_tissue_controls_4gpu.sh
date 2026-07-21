#!/usr/bin/env bash
# Four independent corrected-task controls on the shared st-a100 allocation.
# Uses GPUs 0-3 only and two CPU threads per process.
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-2}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
SERVER_PROFILE_NAME="${SERVER_PROFILE:-shared4}"

if pgrep -f -- 'src.training.train.*missing_tissue_control_' >/dev/null; then
  echo "ERROR: missing-tissue control processes are already running." >&2
  pgrep -af -- 'src.training.train.*missing_tissue_control_' >&2 || true
  exit 2
fi

echo "===== missing-tissue controls one-step smoke check ====="
SMOKETEST=1 \
SMOKE_STEPS=1 \
STAGE=missing_tissue_controls \
SERVER_PROFILE="$SERVER_PROFILE_NAME" \
RUN_ID="${RUN_ID}_smoke" \
LOG_ROOT="logs/recovery_suite/smoke_missing_tissue_controls_${RUN_ID}" \
GPU_IDS="$GPU_IDS_CSV" \
CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/run_recovery_suite_8gpu.sh

echo "===== missing-tissue controls full 10k runs ====="
STAGE=missing_tissue_controls \
SERVER_PROFILE="$SERVER_PROFILE_NAME" \
RUN_ID="${RUN_ID}_full" \
LOG_ROOT="logs/recovery_suite/missing_tissue_controls_${RUN_ID}" \
GPU_IDS="$GPU_IDS_CSV" \
CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/run_recovery_suite_8gpu.sh

echo "Missing-tissue controls completed successfully."
echo "Report: reports/recovery_suite/missing_tissue_controls_${RUN_ID}_full/summary.csv"
