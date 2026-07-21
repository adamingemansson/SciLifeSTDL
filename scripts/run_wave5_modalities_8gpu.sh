#!/usr/bin/env bash
# Eight-run tkdgx1 development screen: missing-tissue modalities, robust
# modality dropout, k=128 GEX branches/spatial bias, and the pure anchor.
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-2}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

if pgrep -f -- 'src.training.train.*missing_tissue_wave5_tk_' >/dev/null; then
  echo "ERROR: tkdgx1 Wave 5 processes are already running." >&2
  pgrep -af -- 'src.training.train.*missing_tissue_wave5_tk_' >&2 || true
  exit 2
fi

echo "===== tkdgx1 Wave 5 one-step smoke check ====="
SMOKETEST=1 \
SMOKE_STEPS=1 \
STAGE=wave5_modalities_tk \
SERVER_PROFILE=dedicated8 \
RUN_ID="${RUN_ID}_smoke" \
LOG_ROOT="logs/recovery_suite/smoke_wave5_modalities_tk_${RUN_ID}" \
GPU_IDS=0,1,2,3,4,5,6,7 \
CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/run_recovery_suite_8gpu.sh

echo "===== tkdgx1 Wave 5 full 10k runs ====="
STAGE=wave5_modalities_tk \
SERVER_PROFILE=dedicated8 \
RUN_ID="${RUN_ID}_full" \
LOG_ROOT="logs/recovery_suite/wave5_modalities_tk_${RUN_ID}" \
GPU_IDS=0,1,2,3,4,5,6,7 \
CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/run_recovery_suite_8gpu.sh

echo "tkdgx1 Wave 5 completed successfully."
echo "Report: reports/recovery_suite/wave5_modalities_tk_${RUN_ID}_full/summary.csv"
