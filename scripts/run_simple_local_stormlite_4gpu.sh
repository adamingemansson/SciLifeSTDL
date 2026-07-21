#!/usr/bin/env bash
# Four matched, whole-sample-held-out diagnostics on Adam's st-a100 GPUs.
# The learned path is intentionally local/deterministic; two arms retain
# leakage-safe context-only Novae and one removes it as the matched ablation.
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-2}"
GPU_IDS_CSV="${GPU_IDS:-1,2,3,5}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
FRESH="${FRESH:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-5}"

if [[ "$GPU_IDS_CSV" != "1,2,3,5" ]]; then
  echo "ERROR: this st-a100 runner is fixed to GPU_IDS=1,2,3,5." >&2
  exit 2
fi
if pgrep -f -- 'src.training.train.*missing_tissue_simple_local_' >/dev/null; then
  echo "ERROR: simple-local processes are already running." >&2
  pgrep -af -- 'src.training.train.*missing_tissue_simple_local_' >&2 || true
  exit 2
fi

available_kb="$(df -Pk . | awk 'NR==2 {print $4}')"
required_kb="$((MIN_FREE_GB * 1024 * 1024))"
if (( available_kb < required_kb )); then
  echo "ERROR: fewer than ${MIN_FREE_GB} GiB are free on $(pwd -P)." >&2
  df -h . >&2
  exit 2
fi

export PYTHONPATH="$(pwd -P)${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="$CPU_THREADS_PER_JOB"
export MKL_NUM_THREADS="$CPU_THREADS_PER_JOB"
export OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB"
export NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB"

"$PYTHON_BIN" scripts/check_simple_local_configs.py

if [[ "$SKIP_SMOKE" != "1" ]]; then
  echo "===== Simple-local one-step smoke check ====="
  SMOKETEST=1 SMOKE_STEPS=1 STAGE=simple_local \
    SERVER_PROFILE=explicit_subset RUN_ID="${RUN_ID}_smoke" \
    LOG_ROOT="logs/recovery_suite/simple_local_${RUN_ID}_smoke" \
    GPU_IDS="$GPU_IDS_CSV" CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
    PYTHON_BIN="$PYTHON_BIN" \
    bash scripts/run_recovery_suite_8gpu.sh
fi

if [[ "$SMOKE_ONLY" == "1" ]]; then
  echo "Simple-local smoke check passed; full run was not started."
  exit 0
fi

echo "===== Cache exact-sample frozen GigaPath context features ====="
CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" scripts/precompute_gigapath_samples.py \
  --config configs/recovery_suite/131_simple_local_full_novae.yaml

echo "===== Cache leakage-safe context-only Novae training/validation graphs ====="
IFS=',' read -r -a gpu_ids <<< "$GPU_IDS_CSV"
cache_pids=()
cache_log_root="logs/recovery_suite/simple_local_${RUN_ID}_novae_precompute"
mkdir -p "$cache_log_root"
for slot in "${!gpu_ids[@]}"; do
  CUDA_VISIBLE_DEVICES="${gpu_ids[$slot]}" \
    "$PYTHON_BIN" scripts/precompute_context_novae_cache.py \
      --config configs/recovery_suite/131_simple_local_full_novae.yaml \
      --validation-only --shard-index "$slot" --num-shards "${#gpu_ids[@]}" \
      > "$cache_log_root/shard${slot}.log" 2>&1 &
  cache_pids+=("$!")
done
cache_failed=0
for slot in "${!cache_pids[@]}"; do
  if ! wait "${cache_pids[$slot]}"; then
    cache_failed=1
    echo "FAILED: Novae precompute shard $slot" >&2
    tail -80 "$cache_log_root/shard${slot}.log" >&2 || true
  fi
done
if (( cache_failed )); then
  echo "Novae precompute failed; training was not started." >&2
  exit 1
fi

echo "===== Four matched simple-local full runs (5k steps; 256 unique holes) ====="
STAGE=simple_local SERVER_PROFILE=explicit_subset RUN_ID="${RUN_ID}_full" \
  LOG_ROOT="logs/recovery_suite/simple_local_${RUN_ID}_full" \
  GPU_IDS="$GPU_IDS_CSV" CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
  PYTHON_BIN="$PYTHON_BIN" FRESH="$FRESH" \
  bash scripts/run_recovery_suite_8gpu.sh

"$PYTHON_BIN" scripts/summarize_simple_local_results.py

echo "Simple-local diagnostic completed."
echo "Report: reports/recovery_suite/simple_local_${RUN_ID}_full/summary.csv"
