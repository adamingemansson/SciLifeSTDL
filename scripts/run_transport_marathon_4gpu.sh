#!/usr/bin/env bash
# Leakage-safe 40-run overnight diagnostic on st-a100 GPUs 1,2,3,5.
# Each GPU owns one fixed ten-job queue. No job migrates to another GPU.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-2}"
GPU_IDS_CSV="${GPU_IDS:-1,2,3,5}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
MATRIX="${MATRIX:-configs/recovery_suite/transport_marathon_40.yaml}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
SKIP_PRECOMPUTE="${SKIP_PRECOMPUTE:-0}"
FRESH="${FRESH:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"
LOG_ROOT="${LOG_ROOT:-logs/recovery_suite/transport_marathon_${RUN_ID}}"
REPORT_ROOT="${REPORT_ROOT:-reports/recovery_suite/transport_marathon_${RUN_ID}}"

if [[ "$GPU_IDS_CSV" != "1,2,3,5" ]]; then
  echo "ERROR: this runner is intentionally fixed to GPU_IDS=1,2,3,5." >&2
  exit 2
fi
if ! [[ "$CPU_THREADS_PER_JOB" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: CPU_THREADS_PER_JOB must be a positive integer." >&2
  exit 2
fi
if [[ ! -f "$MATRIX" ]]; then
  echo "ERROR: matrix not found: $MATRIX" >&2
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

echo "===== Static fail-closed audit of all 40 resolved configs ====="
"$PYTHON_BIN" scripts/check_transport_marathon.py --matrix "$MATRIX"

if pgrep -af -- 'src.training.train.*transport20k_' >"$LOG_ROOT/existing_processes.txt"; then
  echo "ERROR: transport-marathon training processes are already running:" >&2
  cat "$LOG_ROOT/existing_processes.txt" >&2
  exit 2
fi

available_kb="$(df -Pk "$REPO_ROOT" | awk 'NR==2 {print $4}')"
required_kb=$((MIN_FREE_GB * 1024 * 1024))
if (( available_kb < required_kb )); then
  echo "ERROR: less than ${MIN_FREE_GB} GiB is free on the repository filesystem." >&2
  df -h "$REPO_ROOT" >&2
  exit 2
fi

fresh_arg=()
if [[ "$FRESH" == "1" ]]; then
  fresh_arg+=(--fresh)
fi
GPUS=(1 2 3 5)

run_smoke_batch() {
  local batch="$1"
  shift
  local indices=("$@")
  local pids=()
  local i index
  echo "Smoke batch $batch: matrix indices ${indices[*]}"
  for i in "${!indices[@]}"; do
    index="${indices[$i]}"
    "$PYTHON_BIN" scripts/run_transport_marathon_job.py \
      --matrix "$MATRIX" --index "$index" --gpu "${GPUS[$i]}" \
      --threads "$CPU_THREADS_PER_JOB" \
      --log-root "$LOG_ROOT/smoke" --run-id "${RUN_ID}_smoke" \
      --python-bin "$PYTHON_BIN" --smoke "${fresh_arg[@]}" &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  if (( failed )); then
    echo "ERROR: smoke batch $batch failed; full jobs were not started." >&2
    exit 1
  fi
}

if [[ "$SKIP_SMOKE" != "1" ]]; then
  echo "===== Eight representative one-step smoke tests ====="
  mapfile -t smoke_indices < <(
    "$PYTHON_BIN" scripts/check_transport_marathon.py --matrix "$MATRIX" --smoke-indices
  )
  if (( ${#smoke_indices[@]} != 8 )); then
    echo "ERROR: expected exactly eight smoke indices." >&2
    exit 2
  fi
  run_smoke_batch 1 "${smoke_indices[@]:0:4}"
  run_smoke_batch 2 "${smoke_indices[@]:4:4}"
fi

if [[ "$SMOKE_ONLY" == "1" ]]; then
  echo "Transport marathon smoke tests passed; full run was not started."
  exit 0
fi

precompute_novae_config() {
  local config="$1"
  local label="$2"
  local pids=()
  local shard
  echo "===== Novae precompute: $label ====="
  for shard in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" \
      "$PYTHON_BIN" scripts/precompute_context_novae_cache.py \
      --config "$config" --shard-index "$shard" --num-shards 4 \
      >"$LOG_ROOT/precompute_novae_${label}_shard${shard}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  if (( failed )); then
    echo "ERROR: Novae precompute failed for $label; training was not started." >&2
    return 1
  fi
}

if [[ "$SKIP_PRECOMPUTE" != "1" ]]; then
  echo "===== Cache exact-sample frozen GigaPath features ====="
  CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" scripts/precompute_gigapath_samples.py \
    --config configs/recovery_suite/145_transport_marathon_cross.yaml
  precompute_novae_config configs/recovery_suite/145_transport_marathon_cross.yaml cross
  precompute_novae_config configs/recovery_suite/146_transport_marathon_single_int7.yaml single_int7
  precompute_novae_config configs/recovery_suite/147_transport_marathon_single_int8.yaml single_int8
fi

run_slot() {
  local slot="$1"
  local gpu="$2"
  local index
  mapfile -t indices < <(
    "$PYTHON_BIN" scripts/check_transport_marathon.py \
      --matrix "$MATRIX" --indices-for-slot "$slot"
  )
  if (( ${#indices[@]} != 10 )); then
    echo "ERROR: slot $slot resolved to ${#indices[@]} jobs, expected 10." >&2
    return 2
  fi
  echo "GPU $gpu: starting ten-job queue for slot $slot (${indices[*]})"
  for index in "${indices[@]}"; do
    "$PYTHON_BIN" scripts/run_transport_marathon_job.py \
      --matrix "$MATRIX" --index "$index" --gpu "$gpu" \
      --threads "$CPU_THREADS_PER_JOB" \
      --log-root "$LOG_ROOT/full" --run-id "$RUN_ID" \
      --python-bin "$PYTHON_BIN" "${fresh_arg[@]}" || {
        echo "ERROR: slot $slot failed at matrix index $index; later jobs in this slot were not started." >&2
        return 1
      }
  done
  echo "GPU $gpu: slot $slot complete"
}

echo "===== Full transport marathon: 4 GPUs x 10 sequential jobs x 20k steps ====="
worker_pids=()
for slot in 0 1 2 3; do
  run_slot "$slot" "${GPUS[$slot]}" &
  worker_pids+=("$!")
done
failed=0
for pid in "${worker_pids[@]}"; do
  wait "$pid" || failed=1
done
if (( failed )); then
  echo "ERROR: at least one GPU queue failed. Completed metrics are preserved; rerun to resume." >&2
  "$PYTHON_BIN" scripts/summarize_transport_marathon.py \
    --matrix "$MATRIX" --output-dir "$REPORT_ROOT" --allow-incomplete || true
  exit 1
fi

"$PYTHON_BIN" scripts/summarize_transport_marathon.py \
  --matrix "$MATRIX" --output-dir "$REPORT_ROOT"
echo "Transport marathon completed successfully."
echo "Report: $REPORT_ROOT/summary.csv"
