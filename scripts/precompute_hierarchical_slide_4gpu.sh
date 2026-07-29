#!/usr/bin/env bash
# Build dense WSI and leakage-safe Novae caches on configurable GPUs.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-2}"
CONFIG="configs/recovery_suite/152_hierarchical_slide_full_20k.yaml"
GPU_IDS_CSV="${GPU_IDS:-1,2,3,5}"
IFS=',' read -r -a GPUS <<< "$GPU_IDS_CSV"
SAMPLES=(INT1 INT2 INT3 INT4 INT5 INT6 INT7 INT8)
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_ROOT="logs/recovery_suite/hierarchical_precompute_${RUN_ID}"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="$CPU_THREADS_PER_JOB"
export MKL_NUM_THREADS="$CPU_THREADS_PER_JOB"
export OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB"
export NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB"
mkdir -p "$LOG_ROOT"

if (( ${#GPUS[@]} < 1 || ${#GPUS[@]} > 8 )); then
  echo "ERROR: GPU_IDS must contain between 1 and 8 comma-separated devices." >&2
  exit 2
fi
declare -A seen_gpus=()
for gpu in "${GPUS[@]}"; do
  if ! [[ "$gpu" =~ ^[0-9]+$ ]] || [[ -n "${seen_gpus[$gpu]:-}" ]]; then
    echo "ERROR: GPU_IDS contains an invalid or duplicate device: $GPU_IDS_CSV" >&2
    exit 2
  fi
  seen_gpus[$gpu]=1
done

if [[ -z "${GIGAPATH_SLIDE_CHECKPOINT:-}" || ! -f "$GIGAPATH_SLIDE_CHECKPOINT" ]]; then
  echo "ERROR: export GIGAPATH_SLIDE_CHECKPOINT=/absolute/path/to/slide_encoder.pth" >&2
  exit 2
fi
# 19th Codex re-audit (Step 5 Part 2, remaining launch blocker #5):
# scripts/precompute_gigapath_wsi_tiles.py now REQUIRES --tile-encoder-revision
# (a resolved, immutable Hugging Face commit SHA) -- this launcher must
# require and forward it, exactly like it already does for
# GIGAPATH_SLIDE_CHECKPOINT, or every dense WSI cache this script builds
# would silently use whatever the tile encoder's "main" happens to be.
if [[ -z "${GIGAPATH_TILE_ENCODER_REVISION:-}" ]]; then
  echo "ERROR: export GIGAPATH_TILE_ENCODER_REVISION=<immutable-40-hex-commit-sha>" >&2
  echo "  Resolve prov-gigapath/prov-gigapath's current commit SHA yourself first" >&2
  echo "  (e.g. via the HuggingFace web UI or huggingface_hub.HfApi().model_info(...).sha)" >&2
  echo "  -- see scripts/precompute_gigapath_wsi_tiles.py --help." >&2
  exit 2
fi
if ! [[ "$GIGAPATH_TILE_ENCODER_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: GIGAPATH_TILE_ENCODER_REVISION must be a full 40-character lowercase" >&2
  echo "  hex Hugging Face commit SHA, not a branch/tag like 'main'." >&2
  exit 2
fi
if ! "$PYTHON_BIN" -c 'import openslide; print("OpenSlide WSI backend ready")'; then
  echo "ERROR: HEST pyramidal TIFF reading requires OpenSlide." >&2
  echo "Install it in this environment with:" >&2
  echo "  python3 -m pip install openslide-bin openslide-python" >&2
  exit 2
fi
"$PYTHON_BIN" scripts/precompute_gigapath_wsi_tiles.py --config "$CONFIG" \
  --sample-id "${SAMPLES[0]}" --probe-only \
  --tile-encoder-revision "$GIGAPATH_TILE_ENCODER_REVISION"

echo "===== Dense mask-aware WSI tile caches ====="
pids=()
run_wsi_slot() {
  local slot="$1" gpu="$2" sample_index
  for ((sample_index=slot; sample_index<${#SAMPLES[@]}; sample_index+=${#GPUS[@]})); do
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" \
      scripts/precompute_gigapath_wsi_tiles.py --config "$CONFIG" \
      --sample-id "${SAMPLES[$sample_index]}" --device cuda \
      --tile-encoder-revision "$GIGAPATH_TILE_ENCODER_REVISION"
  done
}
for ((slot=0; slot<${#GPUS[@]}; slot++)); do
  run_wsi_slot "$slot" "${GPUS[$slot]}" \
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
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PYTHON_BIN" scripts/precompute_gigapath_samples.py \
  --config "$CONFIG" >"$LOG_ROOT/spot_gigapath.log" 2>&1

echo "===== Context-only Novae caches (queries physically absent from graphs) ====="
pids=()
for ((shard=0; shard<${#GPUS[@]}; shard++)); do
  CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" "$PYTHON_BIN" \
    scripts/precompute_context_novae_cache.py --config "$CONFIG" \
    --shard-index "$shard" --num-shards "${#GPUS[@]}" \
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
