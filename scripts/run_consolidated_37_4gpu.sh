#!/usr/bin/env bash
# Consolidated single run (2026-07-23) -- everything from the 207-251
# recovery_suite range that ISN'T already completed at heldout evaluation,
# PLUS 230/231/232 (originally completed at 10k, rerun here at 20k for a
# like-for-like comparison against every other config's schedule -- see
# their own renamed _v2_20k experiment_name/checkpoint_dir, distinct from
# the existing 10k results, which are left untouched for later reference),
# PLUS 252/253 (new DINOv2 x gene-encoder cross configs, closing a gap --
# every other new axis this round got crossed with the best gene encoders,
# DINOv2 hadn't been yet).
#
# ALL 37 configs standardized to the ORIGINAL 20k-step schedule (epochs=
# 20000, checkpoint_every_n_steps=10000, early_stopping_min_steps=20000) --
# the earlier 10k "trade breadth for depth" round-4/5 schedule was reverted
# for this consolidated run, per explicit instruction. One consecutive run,
# 4 GPUs (1,2,3,5), 9-10 jobs each, smoke-tested up front before any full
# run starts.
#
#   GPU[0] (default 1): 216 -> 217 -> 218 -> 220 -> 221 -> 222 -> 223 -> 224 -> 225 -> 226
#   GPU[1] (default 2): 227 -> 228 -> 229 -> 230 -> 231 -> 232 -> 233 -> 234 -> 235
#   GPU[2] (default 3): 236 -> 237 -> 238 -> 239 -> 240 -> 241 -> 242 -> 243 -> 244
#   GPU[3] (default 5): 245 -> 246 -> 247 -> 248 -> 249 -> 250 -> 251 -> 252 -> 253
#
# Usage:
#   bash scripts/run_consolidated_37_4gpu.sh
#   GPU_IDS=1,2,3,5 SMOKE_ONLY=1 bash scripts/run_consolidated_37_4gpu.sh
#   SKIP_SMOKE=1 bash scripts/run_consolidated_37_4gpu.sh
#   ALLOW_EXISTING_TRANSPORT_JOBS=1 bash scripts/run_consolidated_37_4gpu.sh  # share GPUs with already-running light jobs
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-2}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"
GPU_IDS_CSV="${GPU_IDS:-1,2,3,5}"
IFS=',' read -r -a GPUS <<< "$GPU_IDS_CSV"
ALLOW_EXISTING_TRANSPORT_JOBS="${ALLOW_EXISTING_TRANSPORT_JOBS:-0}"

QUEUE_1=(216 217 218 220 221 222 223 224 225 226)
QUEUE_2=(227 228 229 230 231 232 233 234 235)
QUEUE_3=(236 237 238 239 240 241 242 243 244)
QUEUE_4=(245 246 247 248 249 250 251 252 253)
QUEUES=(QUEUE_1 QUEUE_2 QUEUE_3 QUEUE_4)
ALL_NUMBERS=(216 217 218 220 221 222 223 224 225 226 227 228 229 230 231 232
             233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248
             249 250 251 252 253)

declare -A CONFIG_PATH
declare -A CONFIG_NAME
for f in configs/recovery_suite/2[0-5][0-9]_transport_*.yaml; do
  number="$(basename "$f" | cut -d_ -f1)"
  name="$(awk -F': ' '/^experiment_name:/ {print $2; exit}' "$f")"
  if [[ -z "$name" ]]; then
    echo "ERROR: could not read experiment_name from $f" >&2
    exit 2
  fi
  CONFIG_PATH["$number"]="$f"
  CONFIG_NAME["$number"]="$name"
done
for number in "${ALL_NUMBERS[@]}"; do
  if [[ -z "${CONFIG_PATH[$number]:-}" ]]; then
    echo "ERROR: config $number not found under configs/recovery_suite/" >&2
    exit 2
  fi
done

LOG_ROOT="logs/recovery_suite/consolidated37_${RUN_ID}"
REPORT_ROOT="reports/recovery_suite/consolidated37_${RUN_ID}"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$CPU_THREADS_PER_JOB"
export MKL_NUM_THREADS="$CPU_THREADS_PER_JOB"
export OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB"
export NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB"
mkdir -p "$LOG_ROOT" "$REPORT_ROOT"

echo "===== YAML sanity check (all 37 configs parse, distinct experiment_name, all at 20k steps) ====="
"$PYTHON_BIN" - <<PYEOF
import sys
from pathlib import Path
import yaml

numbers = [str(n) for n in [216,217,218,220,221,222,223,224,225,226,227,228,229,
                              230,231,232,233,234,235,236,237,238,239,240,241,242,
                              243,244,245,246,247,248,249,250,251,252,253]]
seen_names = {}
for n in numbers:
    matches = list(Path("configs/recovery_suite").glob(f"{n}_transport_*.yaml"))
    if len(matches) != 1:
        print(f"ERROR: expected exactly one config for {n}, found {matches}", file=sys.stderr)
        sys.exit(2)
    cfg = yaml.safe_load(matches[0].read_text())
    name = cfg["experiment_name"]
    if name in seen_names:
        print(f"ERROR: experiment_name {name!r} used by both {seen_names[name]} and {n}", file=sys.stderr)
        sys.exit(2)
    seen_names[name] = n
    if int(cfg["training"]["epochs"]) != 20000:
        print(f"ERROR: {n} does not have training.epochs=20000", file=sys.stderr)
        sys.exit(2)
print(f"OK: {len(numbers)} configs parse, {len(seen_names)} distinct experiment_names, all at 20k steps.")
PYEOF

echo "===== GPU_IDS check ====="
if (( ${#GPUS[@]} != 4 )); then
  echo "ERROR: GPU_IDS must provide exactly four comma-separated devices (got '$GPU_IDS_CSV')." >&2
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

echo "===== Exact GPU existence check ====="
mapfile -t VISIBLE_GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)
if [ "${#VISIBLE_GPUS[@]}" -eq 0 ]; then
  echo "ERROR: nvidia-smi reported no visible GPUs." >&2
  exit 2
fi
for gpu in "${GPUS[@]}"; do
  found=0
  for visible in "${VISIBLE_GPUS[@]}"; do
    [[ "$visible" == "$gpu" ]] && found=1 && break
  done
  if [[ "$found" -ne 1 ]]; then
    echo "ERROR: requested GPU $gpu is not among nvidia-smi's visible devices: ${VISIBLE_GPUS[*]}" >&2
    exit 2
  fi
done

echo "===== Disk-space check (>= ${MIN_FREE_GB}GiB free) ====="
avail_kb="$(df -Pk "$REPO_ROOT" | awk 'NR==2 {print $4}')"
avail_gb=$((avail_kb / 1024 / 1024))
if (( avail_gb < MIN_FREE_GB )); then
  echo "ERROR: only ${avail_gb}GiB free on $REPO_ROOT's filesystem, need >= ${MIN_FREE_GB}GiB." >&2
  exit 2
fi
echo "OK: ${avail_gb}GiB free."

if [[ -z "${GIGAPATH_SLIDE_CHECKPOINT:-}" || ! -f "$GIGAPATH_SLIDE_CHECKPOINT" ]]; then
  echo "ERROR: GIGAPATH_SLIDE_CHECKPOINT is missing or not a file." >&2
  exit 2
fi
if ! CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PYTHON_BIN" -c \
  'import torch; from gigapath.torchscale.component.flash_attention import flash_attn_func; assert torch.cuda.is_available() and flash_attn_func is not None'; then
  echo "ERROR: GigaPath LongNet's FlashAttention CUDA kernel is unavailable." >&2
  echo "  MAX_JOBS=4 python3 -m pip install flash-attn==2.5.8 --no-build-isolation" >&2
  exit 2
fi
for sid in INT1 INT2 INT3 INT4 INT5 INT6 INT7 INT8; do
  test -f "data/cache/hest1k/gigapath_slide_cache/${sid}.npz" || {
    echo "ERROR: missing dense WSI cache for $sid; run a hierarchical precompute launcher" >&2
    exit 2
  }
done
if pgrep -af -- 'src.training.train.*transport_' >"$LOG_ROOT/existing_processes.txt"; then
  if [[ "$ALLOW_EXISTING_TRANSPORT_JOBS" == "1" ]]; then
    echo "WARNING: transport jobs already exist, proceeding anyway (ALLOW_EXISTING_TRANSPORT_JOBS=1):" >&2
    cat "$LOG_ROOT/existing_processes.txt" >&2
  else
    echo "ERROR: transport jobs already exist -- this run is meant to start clean:" >&2
    cat "$LOG_ROOT/existing_processes.txt" >&2
    echo "  If these are known-light jobs you intend to keep, rerun with ALLOW_EXISTING_TRANSPORT_JOBS=1." >&2
    exit 2
  fi
fi

job_is_done() {
  local number="$1" name="${CONFIG_NAME[$1]}"
  test -f "results/checkpoints/recovery_suite/${name}/heldout_sample_summary.json"
}

run_one() {
  local number="$1" gpu="$2" smoke="$3"
  local name="${CONFIG_NAME[$number]}" path="${CONFIG_PATH[$number]}"
  local command=("$PYTHON_BIN" -m src.training.train --config "$path")
  if [[ "$smoke" == "1" ]]; then
    command+=(--override
      "experiment_name=${name}_smoke_${RUN_ID}"
      training.epochs=1 training.unique_mask_count=1
      training.checkpoint_every_n_steps=0 training.log_print_every_n_steps=1
      "training.checkpoint_dir=results/checkpoints/recovery_suite/smoke/${RUN_ID}/${name}"
      validation.every_n_steps=1 validation.early_stopping_min_steps=1
      validation.patience_checks=1000 validation.require_anchor_improvement=false
      "evaluation.training_mask_bank_path=results/mask_banks/training/recovery_suite/smoke_${RUN_ID}_${name}.json"
      evaluation.n_validation_masks=1 evaluation.n_test_masks=1 evaluation.n_samples=1
      "evaluation.mask_bank_dir=results/mask_banks/recovery_suite/smoke_consolidated37_${RUN_ID}_${name}"
    )
    echo "GPU $gpu -> $name (smoke)"
    CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}" >"$LOG_ROOT/smoke_${name}.log" 2>&1
    return $?
  fi
  if job_is_done "$number"; then
    echo "GPU $gpu -> $name SKIP (already completed, found heldout_sample_summary.json)"
    return 0
  fi
  echo "GPU $gpu -> $name START"
  CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}" >"$LOG_ROOT/full_${name}.log" 2>&1
}

run_queue() {
  local queue_name="$1" gpu="$2"
  local -n queue_ref="$queue_name"
  local number
  for number in "${queue_ref[@]}"; do
    run_one "$number" "$gpu" 0 || {
      echo "ERROR: ${CONFIG_NAME[$number]} failed; see $LOG_ROOT/full_${CONFIG_NAME[$number]}.log" >&2
      return 1
    }
  done
}

smoke_queue() {
  local queue_name="$1" gpu="$2"
  local -n queue_ref="$queue_name"
  local number
  for number in "${queue_ref[@]}"; do
    run_one "$number" "$gpu" 1 || return 1
  done
}

if [[ "$SKIP_SMOKE" != "1" ]]; then
  echo "===== One-step fail-closed smoke test (all 37 configs, up to 10 sequential per GPU) ====="
  smoke_pids=()
  for idx in 0 1 2 3; do
    smoke_queue "${QUEUES[$idx]}" "${GPUS[$idx]}" &
    smoke_pids+=("$!")
  done
  smoke_failed=0
  for pid in "${smoke_pids[@]}"; do wait "$pid" || smoke_failed=1; done
  if (( smoke_failed )); then
    echo "ERROR: smoke test failed; inspect $LOG_ROOT/smoke_*.log" >&2
    exit 1
  fi
  echo "Smoke tests passed (all 37 configs)."
fi
if [[ "$SMOKE_ONLY" == "1" ]]; then
  echo "Smoke tests passed; full runs were not started."
  exit 0
fi

echo "===== Four parallel queues (35 jobs total, no inter-config dependencies) ====="
queue_pids=()
for idx in 0 1 2 3; do
  run_queue "${QUEUES[$idx]}" "${GPUS[$idx]}" &
  queue_pids+=("$!")
done
queue_failed=0
for pid in "${queue_pids[@]}"; do wait "$pid" || queue_failed=1; done

if (( queue_failed )); then
  echo "WARNING: at least one job failed; inspect $LOG_ROOT/*.log" >&2
fi
"$PYTHON_BIN" scripts/summarize_transport_round4.py --output "$REPORT_ROOT/summary.csv" || true
echo "Consolidated run finished (RUN_ID=$RUN_ID). Report: $REPORT_ROOT/summary.csv"
echo "Logs: $LOG_ROOT"
