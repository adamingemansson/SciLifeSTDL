#!/usr/bin/env bash
# Four matched st-a100 replications on Adam's allocated GPUs. These repeat
# full, GEX-only, surrounding-H&E-only, and neither on the st gene panel.
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-2}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
GPU_IDS_CSV="${GPU_IDS:-1,2,3,5}"

if pgrep -f -- 'src.training.train.*missing_tissue_wave5_st_' >/dev/null; then
  echo "ERROR: st-a100 Wave 5 processes are already running." >&2
  pgrep -af -- 'src.training.train.*missing_tissue_wave5_st_' >&2 || true
  exit 2
fi

echo "===== st-a100 Wave 5 one-step smoke check ====="
SMOKETEST=1 \
SMOKE_STEPS=1 \
STAGE=wave5_modalities_st \
SERVER_PROFILE=explicit_subset \
RUN_ID="${RUN_ID}_smoke" \
LOG_ROOT="logs/recovery_suite/smoke_wave5_modalities_st_${RUN_ID}" \
GPU_IDS="$GPU_IDS_CSV" \
CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/run_recovery_suite_8gpu.sh

echo "===== st-a100 Wave 5 full 10k runs ====="
STAGE=wave5_modalities_st \
SERVER_PROFILE=explicit_subset \
RUN_ID="${RUN_ID}_full" \
LOG_ROOT="logs/recovery_suite/wave5_modalities_st_${RUN_ID}" \
GPU_IDS="$GPU_IDS_CSV" \
CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/run_recovery_suite_8gpu.sh

echo "st-a100 Wave 5 completed successfully."
echo "Report: reports/recovery_suite/wave5_modalities_st_${RUN_ID}_full/summary.csv"
