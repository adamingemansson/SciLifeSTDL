#!/usr/bin/env bash
# Build dense WSI and leakage-safe Novae caches on the allocated st-a100 GPUs.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-2}"
CONFIG="configs/recovery_suite/152_hierarchical_slide_full_20k.yaml"
GPUS=(1 2 3 5)
SAMPLES=(INT1 INT2 INT3 INT4 INT5 INT6 INT7 INT8)
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_ROOT="logs/recovery_suite/hierarchical_precompute_${RUN_ID}"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="$CPU_THREADS_PER_JOB"
export MKL_NUM_THREADS="$CPU_THREADS_PER_JOB"
export OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB"
export NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB"
mkdir -p "$LOG_ROOT"

if [[ -z "${GIGAPATH_SLIDE_CHECKPOINT:-}" || ! -f "$GIGAPATH_SLIDE_CHECKPOINT" ]]; then
  echo "ERROR: export GIGAPATH_SLIDE_CHECKPOINT=/absolute/path/to/slide_encoder.pth" >&2
  exit 2
fi

echo "===== Dense mask-aware WSI tile caches ====="
pids=()
for slot in 0 1 2 3; do
  first="${SAMPLES[$((2 * slot))]}"
  second="${SAMPLES[$((2 * slot + 1))]}"
  CUDA_VISIBLE_DEVICES="${GPUS[$slot]}" "$PYTHON_BIN" \
    scripts/precompute_gigapath_wsi_tiles.py --config "$CONFIG" \
    --sample-id "$first" --sample-id "$second" --device cuda \
    >"$LOG_ROOT/wsi_gpu${GPUS[$slot]}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if (( failed )); then
  echo "ERROR: dense WSI precompute failed; inspect $LOG_ROOT/wsi_gpu*.log" >&2
  exit 1
fi

echo "===== Spot-aligned frozen GigaPath caches ====="
CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" scripts/precompute_gigapath_samples.py \
  --config "$CONFIG" >"$LOG_ROOT/spot_gigapath.log" 2>&1

echo "===== Context-only Novae caches (queries physically absent from graphs) ====="
pids=()
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" "$PYTHON_BIN" \
    scripts/precompute_context_novae_cache.py --config "$CONFIG" \
    --shard-index "$shard" --num-shards 4 \
    >"$LOG_ROOT/novae_shard${shard}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if (( failed )); then
  echo "ERROR: Novae precompute failed; inspect $LOG_ROOT/novae_shard*.log" >&2
  exit 1
fi
echo "Hierarchical caches ready. Logs: $LOG_ROOT"
