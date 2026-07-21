#!/usr/bin/env bash
# Corrected st-a100 replication on Adam's allocated GPUs 1,2,3,5.
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-2}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
GPU_IDS_CSV="${GPU_IDS:-1,2,3,5}"

if pgrep -f -- 'src.training.train.*missing_tissue_wave6_st_heldout_' >/dev/null; then
  echo "ERROR: st-a100 Wave 6 held-out processes are already running." >&2
  pgrep -af -- 'src.training.train.*missing_tissue_wave6_st_heldout_' >&2 || true
  exit 2
fi

export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="$CPU_THREADS_PER_JOB"
export MKL_NUM_THREADS="$CPU_THREADS_PER_JOB"
export OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB"
export NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB"

echo "===== Preflight: cache frozen GigaPath H&E embeddings once ====="
FIRST_GPU="${GPU_IDS_CSV%%,*}"
CUDA_VISIBLE_DEVICES="$FIRST_GPU" "$PYTHON_BIN" scripts/precompute_gigapath_samples.py \
  --config configs/recovery_suite/92_wave6_st_heldout_full_k128.yaml

echo "===== st-a100 Wave 6 one-step held-out smoke check ====="
SMOKETEST=1 \
SMOKE_STEPS=1 \
STAGE=wave6_heldout_st \
SERVER_PROFILE=explicit_subset \
RUN_ID="${RUN_ID}_smoke" \
LOG_ROOT="logs/recovery_suite/smoke_wave6_heldout_st_${RUN_ID}" \
GPU_IDS="$GPU_IDS_CSV" \
CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/run_recovery_suite_8gpu.sh

echo "===== st-a100 Wave 6 full 10k held-out runs ====="
STAGE=wave6_heldout_st \
SERVER_PROFILE=explicit_subset \
RUN_ID="${RUN_ID}_full" \
LOG_ROOT="logs/recovery_suite/wave6_heldout_st_${RUN_ID}" \
GPU_IDS="$GPU_IDS_CSV" \
CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/run_recovery_suite_8gpu.sh

echo "st-a100 Wave 6 held-out replication completed successfully."
echo "Report: reports/recovery_suite/wave6_heldout_st_${RUN_ID}_full/summary.csv"
