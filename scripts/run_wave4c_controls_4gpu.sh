#!/usr/bin/env bash
# Four same-server controls on the explicitly allocated st-a100 GPUs:
# FM reference, deterministic k=16, and pure harmonic k=8/k=16 anchors.
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-2}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
GPU_IDS_CSV="${GPU_IDS:-1,2,3,5}"
SERVER_PROFILE_NAME="${SERVER_PROFILE:-explicit_subset}"

if pgrep -f -- 'src.training.train.*missing_tissue_wave4c_' >/dev/null; then
  echo "ERROR: Wave 4C control processes are already running." >&2
  pgrep -af -- 'src.training.train.*missing_tissue_wave4c_' >&2 || true
  exit 2
fi

echo "===== Wave 4C same-server controls one-step smoke check ====="
SMOKETEST=1 \
SMOKE_STEPS=1 \
STAGE=wave4c_controls \
SERVER_PROFILE="$SERVER_PROFILE_NAME" \
RUN_ID="${RUN_ID}_smoke" \
LOG_ROOT="logs/recovery_suite/smoke_wave4c_controls_${RUN_ID}" \
GPU_IDS="$GPU_IDS_CSV" \
CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/run_recovery_suite_8gpu.sh

echo "===== Wave 4C same-server controls full runs ====="
STAGE=wave4c_controls \
SERVER_PROFILE="$SERVER_PROFILE_NAME" \
RUN_ID="${RUN_ID}_full" \
LOG_ROOT="logs/recovery_suite/wave4c_controls_${RUN_ID}" \
GPU_IDS="$GPU_IDS_CSV" \
CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/run_recovery_suite_8gpu.sh

echo "Wave 4C controls completed successfully."
echo "Report: reports/recovery_suite/wave4c_controls_${RUN_ID}_full/summary.csv"
