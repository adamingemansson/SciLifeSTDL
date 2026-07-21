#!/usr/bin/env bash
# Eight distinct 10k screens for the actual missing-tissue task on a fully
# allocated eight-GPU machine. Smoke failure prevents the full batch.
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-2}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

if pgrep -f -- 'src.training.train.*missing_tissue_' >/dev/null; then
  echo "ERROR: missing-tissue training processes are already running." >&2
  pgrep -af -- 'src.training.train.*missing_tissue_' >&2 || true
  exit 2
fi

echo "===== missing-tissue one-step smoke check ====="
SMOKETEST=1 \
SMOKE_STEPS=1 \
STAGE=missing_tissue \
SERVER_PROFILE=dedicated8 \
RUN_ID="${RUN_ID}_smoke" \
LOG_ROOT="logs/recovery_suite/smoke_missing_tissue_${RUN_ID}" \
GPU_IDS=0,1,2,3,4,5,6,7 \
CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/run_recovery_suite_8gpu.sh

echo "===== missing-tissue full 10k screen ====="
STAGE=missing_tissue \
SERVER_PROFILE=dedicated8 \
RUN_ID="${RUN_ID}_full" \
LOG_ROOT="logs/recovery_suite/missing_tissue_${RUN_ID}" \
GPU_IDS=0,1,2,3,4,5,6,7 \
CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/run_recovery_suite_8gpu.sh

echo "Missing-tissue screen completed successfully."
echo "Report: reports/recovery_suite/missing_tissue_${RUN_ID}_full/summary.csv"
