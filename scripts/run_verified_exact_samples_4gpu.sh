#!/usr/bin/env bash
# Minimal post-fix rerun on st-a100 GPUs 1,2,3,5. Exact sample resolution and
# patch-content cache fingerprints must pass before these results are trusted.
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-2}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"

if pgrep -f -- 'src.training.train.*missing_tissue_verified_exact_' >/dev/null; then
  echo "ERROR: verified exact-sample processes are already running." >&2
  pgrep -af -- 'src.training.train.*missing_tissue_verified_exact_' >&2 || true
  exit 2
fi

export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="$CPU_THREADS_PER_JOB"
export MKL_NUM_THREADS="$CPU_THREADS_PER_JOB"
export OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB"
export NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB"

echo "===== Resolve exact HEST files and rebuild mismatched GigaPath caches ====="
CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" scripts/precompute_gigapath_samples.py \
  --config configs/recovery_suite/126_verified_exact_full_k32.yaml

if [[ "$SKIP_SMOKE" != "1" ]]; then
  echo "===== One-step verified exact-sample smoke check ====="
  SMOKETEST=1 SMOKE_STEPS=1 \
  STAGE=verified_exact_st SERVER_PROFILE=explicit_subset \
  RUN_ID="${RUN_ID}_smoke" \
  LOG_ROOT="logs/recovery_suite/smoke_verified_exact_${RUN_ID}" \
  GPU_IDS=1,2,3,5 CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
  PYTHON_BIN="$PYTHON_BIN" bash scripts/run_recovery_suite_8gpu.sh
fi

if [[ "$SMOKE_ONLY" == "1" ]]; then
  echo "Verified exact-sample smoke check passed; full rerun was not started."
  exit 0
fi

echo "===== Full verified exact-sample rerun ====="
STAGE=verified_exact_st SERVER_PROFILE=explicit_subset \
RUN_ID="${RUN_ID}_full" \
LOG_ROOT="logs/recovery_suite/verified_exact_${RUN_ID}" \
GPU_IDS=1,2,3,5 CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
PYTHON_BIN="$PYTHON_BIN" bash scripts/run_recovery_suite_8gpu.sh

echo "Verified rerun complete."
echo "Report: reports/recovery_suite/verified_exact_st_${RUN_ID}_full/summary.csv"
