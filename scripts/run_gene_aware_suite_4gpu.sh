#!/usr/bin/env bash
# Four consecutive jobs per st-a100 GPU (1,2,3,5), with a capacity gate
# between the first and remaining three jobs.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-2}"
GPU_IDS_CSV="${GPU_IDS:-1,2,3,5}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
MATRIX="${MATRIX:-configs/recovery_suite/gene_aware_transport_16.yaml}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
SKIP_PRECOMPUTE="${SKIP_PRECOMPUTE:-0}"
FRESH="${FRESH:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"
LOG_ROOT="${LOG_ROOT:-logs/recovery_suite/gene_aware_${RUN_ID}}"
REPORT_ROOT="${REPORT_ROOT:-reports/recovery_suite/gene_aware_${RUN_ID}}"

if [[ "$GPU_IDS_CSV" != "1,2,3,5" ]]; then
  echo "ERROR: this runner is fixed to GPU_IDS=1,2,3,5." >&2
  exit 2
fi
if ! [[ "$CPU_THREADS_PER_JOB" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: CPU_THREADS_PER_JOB must be a positive integer." >&2
  exit 2
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$CPU_THREADS_PER_JOB"
export MKL_NUM_THREADS="$CPU_THREADS_PER_JOB"
export OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB"
export NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB"
mkdir -p "$LOG_ROOT" "$REPORT_ROOT"
exec > >(tee -a "$LOG_ROOT/orchestrator.log") 2>&1

echo "===== Static fail-closed audit of all 16 configs ====="
"$PYTHON_BIN" scripts/check_gene_aware_suite.py --matrix "$MATRIX"
if pgrep -af -- 'src.training.train.*geneaware_' >"$LOG_ROOT/existing_processes.txt"; then
  echo "ERROR: gene-aware suite processes are already running:" >&2
  cat "$LOG_ROOT/existing_processes.txt" >&2
  exit 2
fi
available_kb="$(df -Pk "$REPO_ROOT" | awk 'NR==2 {print $4}')"
required_kb=$((MIN_FREE_GB * 1024 * 1024))
if (( available_kb < required_kb )); then
  echo "ERROR: less than ${MIN_FREE_GB} GiB is free." >&2
  df -h "$REPO_ROOT" >&2
  exit 2
fi

GPUS=(1 2 3 5)
fresh_arg=()
if [[ "$FRESH" == "1" ]]; then fresh_arg+=(--fresh); fi

run_parallel_indices() {
  local label="$1"
  shift
  local indices=("$@")
  if (( ${#indices[@]} != 4 )); then
    echo "ERROR: $label expected four indices, got ${#indices[@]}." >&2
    return 2
  fi
  local pids=() i
  for i in 0 1 2 3; do
    echo "GPU ${GPUS[$i]} -> matrix index ${indices[$i]} ($label)"
    "$PYTHON_BIN" scripts/run_gene_aware_job.py \
      --matrix "$MATRIX" --index "${indices[$i]}" --gpu "${GPUS[$i]}" \
      --threads "$CPU_THREADS_PER_JOB" --log-root "$LOG_ROOT/$label" \
      --run-id "$RUN_ID" --python-bin "$PYTHON_BIN" "${fresh_arg[@]}" &
    pids+=("$!")
  done
  local failed=0 pid
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  (( failed == 0 ))
}

if [[ "$SKIP_SMOKE" != "1" ]]; then
  echo "===== Representative one-step smoke tests ====="
  mapfile -t smoke_indices < <(
    "$PYTHON_BIN" scripts/check_gene_aware_suite.py --matrix "$MATRIX" --smoke-indices
  )
  pids=()
  for i in 0 1 2 3; do
    "$PYTHON_BIN" scripts/run_gene_aware_job.py \
      --matrix "$MATRIX" --index "${smoke_indices[$i]}" --gpu "${GPUS[$i]}" \
      --threads "$CPU_THREADS_PER_JOB" --log-root "$LOG_ROOT/smoke" \
      --run-id "${RUN_ID}_smoke" --python-bin "$PYTHON_BIN" --smoke \
      "${fresh_arg[@]}" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  if (( failed )); then
    echo "ERROR: smoke tests failed; no full jobs started." >&2
    exit 1
  fi
fi
if [[ "$SMOKE_ONLY" == "1" ]]; then
  echo "Gene-aware smoke tests passed; full suite was not started."
  exit 0
fi

precompute_novae() {
  local config="$1" label="$2"
  local pids=() shard failed=0
  echo "===== Novae precompute: $label ====="
  for shard in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" \
      "$PYTHON_BIN" scripts/precompute_context_novae_cache.py \
      --config "$config" --shard-index "$shard" --num-shards 4 \
      >"$LOG_ROOT/precompute_${label}_shard${shard}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  if (( failed )); then
    echo "ERROR: Novae precompute failed for $label." >&2
    return 1
  fi
}

if [[ "$SKIP_PRECOMPUTE" != "1" ]]; then
  echo "===== Cache exact frozen GigaPath features ====="
  CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" scripts/precompute_gigapath_samples.py \
    --config configs/recovery_suite/148_gene_aware_cross_10k.yaml
  precompute_novae configs/recovery_suite/151_gene_aware_overfit_int7.yaml overfit_int7
  precompute_novae configs/recovery_suite/148_gene_aware_cross_10k.yaml cross
  precompute_novae configs/recovery_suite/149_gene_aware_single_int7_10k.yaml single_int7
  precompute_novae configs/recovery_suite/150_gene_aware_single_int8_10k.yaml single_int8
fi

echo "===== Phase 1: four 3k-step capacity gates ====="
mapfile -t overfit_indices < <(
  "$PYTHON_BIN" scripts/check_gene_aware_suite.py --matrix "$MATRIX" --stage overfit
)
run_parallel_indices overfit "${overfit_indices[@]}" || {
  echo "ERROR: a capacity run failed technically; held-out jobs were not started." >&2
  exit 1
}
"$PYTHON_BIN" scripts/check_gene_aware_overfit_gate.py \
  --matrix "$MATRIX" --output "$REPORT_ROOT/overfit_decision.json"

echo "===== Phase 2: three consecutive held-out jobs on each GPU ====="
worker_pids=()
run_heldout_slot() {
  local slot="$1" gpu="$2" index
  mapfile -t indices < <(
    "$PYTHON_BIN" scripts/check_gene_aware_suite.py \
      --matrix "$MATRIX" --stage heldout --slot "$slot"
  )
  if (( ${#indices[@]} != 3 )); then
    echo "ERROR: slot $slot has ${#indices[@]} held-out jobs, expected three." >&2
    return 2
  fi
  for index in "${indices[@]}"; do
    "$PYTHON_BIN" scripts/run_gene_aware_job.py \
      --matrix "$MATRIX" --index "$index" --gpu "$gpu" \
      --threads "$CPU_THREADS_PER_JOB" --log-root "$LOG_ROOT/heldout" \
      --run-id "$RUN_ID" --python-bin "$PYTHON_BIN" "${fresh_arg[@]}" || return 1
  done
}
for slot in 0 1 2 3; do
  run_heldout_slot "$slot" "${GPUS[$slot]}" &
  worker_pids+=("$!")
done
failed=0
for pid in "${worker_pids[@]}"; do wait "$pid" || failed=1; done
if (( failed )); then
  "$PYTHON_BIN" scripts/summarize_gene_aware_suite.py \
    --matrix "$MATRIX" --output-dir "$REPORT_ROOT" --allow-incomplete || true
  echo "ERROR: at least one held-out GPU queue failed; rerun to resume." >&2
  exit 1
fi

"$PYTHON_BIN" scripts/summarize_gene_aware_suite.py \
  --matrix "$MATRIX" --output-dir "$REPORT_ROOT"
echo "Gene-aware suite complete. Report: $REPORT_ROOT/summary.csv"
