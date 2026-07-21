#!/usr/bin/env bash
# Eight consecutive four-job batches on Adam's st-a100 allocation (GPUs
# 1,2,3,5). CPU use is capped at two threads per job. By default this runs a
# short representative smoke check and continues automatically into the full
# two-hour marathon; SMOKE_ONLY=1 or SKIP_SMOKE=1 separates those phases.
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-2}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
GPU_IDS_CSV="${GPU_IDS:-1,2,3,5}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"

if [[ "$GPU_IDS_CSV" != "1,2,3,5" ]]; then
  echo "ERROR: this st-a100 runner is fixed to GPU_IDS=1,2,3,5." >&2
  exit 2
fi
if pgrep -f -- 'src.training.train.*missing_tissue_wave7_st_' >/dev/null; then
  echo "ERROR: Wave 7 st-a100 processes are already running." >&2
  pgrep -af -- 'src.training.train.*missing_tissue_wave7_st_' >&2 || true
  exit 2
fi

export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="$CPU_THREADS_PER_JOB"
export MKL_NUM_THREADS="$CPU_THREADS_PER_JOB"
export OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB"
export NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB"

STPATH_ROOT="${STPATH_ROOT:-$(dirname "$(pwd -P)")/STPath}"
export STPATH_GENE_VOC_PATH="${STPATH_GENE_VOC_PATH:-$STPATH_ROOT/utils_data/symbol2ensembl.json}"
export STPATH_MODEL_WEIGHT_PATH="${STPATH_MODEL_WEIGHT_PATH:-$STPATH_ROOT/stfm.pth}"
if [[ ! -f "$STPATH_GENE_VOC_PATH" || ! -f "$STPATH_MODEL_WEIGHT_PATH" ]]; then
  echo "ERROR: STPath vocabulary or weights are missing." >&2
  echo "Vocabulary: $STPATH_GENE_VOC_PATH" >&2
  echo "Weights:    $STPATH_MODEL_WEIGHT_PATH" >&2
  exit 2
fi

echo "===== Preflight: cache frozen GigaPath features for all eight samples ====="
CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" scripts/precompute_gigapath_samples.py \
  --config configs/recovery_suite/92_wave6_st_heldout_full_k128.yaml

if [[ "$SKIP_SMOKE" != "1" ]]; then
  echo "===== Representative one-step smoke check (three batches) ====="
  SMOKETEST=1 \
  SMOKE_STEPS=1 \
  STAGE=wave7_st_smoke \
  SERVER_PROFILE=explicit_subset \
  RUN_ID="${RUN_ID}_smoke" \
  LOG_ROOT="logs/recovery_suite/smoke_wave7_st_${RUN_ID}" \
  GPU_IDS="$GPU_IDS_CSV" \
  CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
  PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/run_recovery_suite_8gpu.sh
fi

if [[ "$SMOKE_ONLY" == "1" ]]; then
  echo "Wave 7 representative smoke check passed; full marathon was not started."
  exit 0
fi

echo "===== Full Wave 7 marathon: 8 batches x 4 jobs x 10k steps ====="
STAGE=wave7_st_marathon \
SERVER_PROFILE=explicit_subset \
RUN_ID="${RUN_ID}_full" \
LOG_ROOT="logs/recovery_suite/wave7_st_marathon_${RUN_ID}" \
GPU_IDS="$GPU_IDS_CSV" \
CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/run_recovery_suite_8gpu.sh

echo "Wave 7 st-a100 marathon completed successfully."
echo "Report: reports/recovery_suite/wave7_st_marathon_${RUN_ID}_full/summary.csv"
