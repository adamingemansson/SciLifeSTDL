#!/usr/bin/env bash
# Four matched FM capacity tests on the shared st-a100 allocation. GPU_IDS and
# SERVER_PROFILE explicitly select the GPUs allocated for this invocation.
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-2}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
SERVER_PROFILE_NAME="${SERVER_PROFILE:-shared4}"

if pgrep -f -- 'src.training.train.*missing_tissue_wave4_fm_' >/dev/null; then
  echo "ERROR: Wave 4 FM processes are already running." >&2
  pgrep -af -- 'src.training.train.*missing_tissue_wave4_fm_' >&2 || true
  exit 2
fi

echo "===== Wave 4 matched FM one-step smoke check ====="
SMOKETEST=1 \
SMOKE_STEPS=1 \
STAGE=wave4_fm \
SERVER_PROFILE="$SERVER_PROFILE_NAME" \
RUN_ID="${RUN_ID}_smoke" \
LOG_ROOT="logs/recovery_suite/smoke_wave4_fm_${RUN_ID}" \
GPU_IDS="$GPU_IDS_CSV" \
CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/run_recovery_suite_8gpu.sh

echo "===== Wave 4 matched FM full 10k screen ====="
STAGE=wave4_fm \
SERVER_PROFILE="$SERVER_PROFILE_NAME" \
RUN_ID="${RUN_ID}_full" \
LOG_ROOT="logs/recovery_suite/wave4_fm_${RUN_ID}" \
GPU_IDS="$GPU_IDS_CSV" \
CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/run_recovery_suite_8gpu.sh

echo "Wave 4 matched FM screen completed successfully."
echo "Report: reports/recovery_suite/wave4_fm_${RUN_ID}_full/summary.csv"
