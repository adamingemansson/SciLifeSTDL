#!/usr/bin/env bash
# Corrected tkdgx1 screen: eight learned jobs on GPUs 0-7, followed by a
# short CPU-only harmonic control. Whole samples, not mask seeds, define
# train/validation/test separation.
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-2}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

if pgrep -f -- 'src.training.train.*missing_tissue_wave6_tk_heldout_' >/dev/null; then
  echo "ERROR: tkdgx1 Wave 6 held-out processes are already running." >&2
  pgrep -af -- 'src.training.train.*missing_tissue_wave6_tk_heldout_' >&2 || true
  exit 2
fi

export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="$CPU_THREADS_PER_JOB"
export MKL_NUM_THREADS="$CPU_THREADS_PER_JOB"
export OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB"
export NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB"

echo "===== Preflight: cache frozen GigaPath H&E embeddings once ====="
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" scripts/precompute_gigapath_samples.py \
  --config configs/recovery_suite/84_wave6_tk_heldout_full_k128.yaml

echo "===== tkdgx1 Wave 6 one-step held-out smoke check ====="
SMOKETEST=1 \
SMOKE_STEPS=1 \
STAGE=wave6_heldout_tk \
SERVER_PROFILE=dedicated8 \
RUN_ID="${RUN_ID}_smoke" \
LOG_ROOT="logs/recovery_suite/smoke_wave6_heldout_tk_${RUN_ID}" \
GPU_IDS=0,1,2,3,4,5,6,7 \
CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/run_recovery_suite_8gpu.sh

echo "===== tkdgx1 Wave 6 full 10k held-out runs ====="
STAGE=wave6_heldout_tk \
SERVER_PROFILE=dedicated8 \
RUN_ID="${RUN_ID}_full" \
LOG_ROOT="logs/recovery_suite/wave6_heldout_tk_${RUN_ID}" \
GPU_IDS=0,1,2,3,4,5,6,7 \
CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/run_recovery_suite_8gpu.sh

echo "tkdgx1 Wave 6 held-out screen completed successfully."
echo "Report: reports/recovery_suite/wave6_heldout_tk_${RUN_ID}_full/summary.csv"
