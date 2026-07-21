#!/usr/bin/env bash
# Eight distinct deterministic StormLite component tests on a dedicated
# eight-GPU machine. A one-step smoke stage gates the full 10k screen.
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-2}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

if pgrep -f -- 'src.training.train.*missing_tissue_wave4_hr_' >/dev/null; then
  echo "ERROR: Wave 4 harmonic-residual processes are already running." >&2
  pgrep -af -- 'src.training.train.*missing_tissue_wave4_hr_' >&2 || true
  exit 2
fi

echo "===== Wave 4 deterministic StormLite one-step smoke check ====="
SMOKETEST=1 \
SMOKE_STEPS=1 \
STAGE=wave4_hr \
SERVER_PROFILE=dedicated8 \
RUN_ID="${RUN_ID}_smoke" \
LOG_ROOT="logs/recovery_suite/smoke_wave4_hr_${RUN_ID}" \
GPU_IDS=0,1,2,3,4,5,6,7 \
CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/run_recovery_suite_8gpu.sh

echo "===== Wave 4 deterministic StormLite full 10k screen ====="
STAGE=wave4_hr \
SERVER_PROFILE=dedicated8 \
RUN_ID="${RUN_ID}_full" \
LOG_ROOT="logs/recovery_suite/wave4_hr_${RUN_ID}" \
GPU_IDS=0,1,2,3,4,5,6,7 \
CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/run_recovery_suite_8gpu.sh

echo "Wave 4 deterministic StormLite screen completed successfully."
echo "Report: reports/recovery_suite/wave4_hr_${RUN_ID}_full/summary.csv"
