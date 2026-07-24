#!/usr/bin/env bash
# Lung round (2026-07-24) -- 4 GPU-training conditions, 1 config per GPU,
# smoke-tested up front before any full run starts (same pattern as
# scripts/run_consolidated_37_4gpu.sh, trimmed since there's no queueing
# needed with exactly 4 configs on 4 GPUs). Smallest model (zero attention
# parameters) deliberately placed on GPU 1 per explicit instruction:
#
#   GPU[0] (default 1): 302_lung_simple_fusion                (smallest -- GigaPath+gene sum, mean-pooled, no attention)
#   GPU[1] (default 2): 303_lung_simple_cross_attn             (same tokens, 2-layer learned cross-attn stack)
#   GPU[2] (default 3): 301_lung_stpath_scratch                 (STPath's real architecture, no pretraining)
#   GPU[3] (default 5): 306_lung_simple_stpath_transformer      (our tokens + STPath's real SpatialTransformer backbone, no organ/tech)
#
# 304_lung_harmonic.yaml (closed-form, no training) and
# 305_lung_stpath_pretrained_eval.yaml (frozen weights, eval only) need no
# GPU-training slot -- run them separately, e.g. on whichever GPU frees up
# first:
#   CUDA_VISIBLE_DEVICES=1 python3 -m src.training.train --config configs/lung_round/304_lung_harmonic.yaml
#   CUDA_VISIBLE_DEVICES=1 python3 -m src.training.train --config configs/lung_round/305_lung_stpath_pretrained_eval.yaml
#
# 305 additionally needs STPATH_MODEL_WEIGHT_PATH (real released weight,
# huggingface.co/tlhuang/STPath) set -- 301/302/303/306 don't (301 trains
# from a random init; 302/303 don't use the stpath package at all; 306
# only needs stpath's SpatialTransformer/ModelConfig classes, no weights).
# All configs except 302/303 need STPATH_GENE_VOC_PATH (301/305's gene
# vocabulary resource -- a fixed lookup table, not a model weight).
#
# 306 has only been verified by reading STPath's real source, never
# executed end-to-end (the `stpath` package wasn't installed anywhere
# this session had shell access) -- its smoke test below is the actual
# first real verification, not a formality. Watch its smoke log closely.
#
# Usage:
#   bash scripts/run_lung_round_4gpu.sh
#   GPU_IDS=1,2,3,5 SMOKE_ONLY=1 bash scripts/run_lung_round_4gpu.sh
#   SKIP_SMOKE=1 bash scripts/run_lung_round_4gpu.sh
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

NUMBERS=(302 303 301 306)
declare -A CONFIG_PATH
declare -A CONFIG_NAME
for number in "${NUMBERS[@]}"; do
  f="configs/lung_round/${number}_lung_"*.yaml
  matches=( $f )
  if [[ ${#matches[@]} -ne 1 || ! -f "${matches[0]}" ]]; then
    echo "ERROR: expected exactly one config for $number, found: ${matches[*]}" >&2
    exit 2
  fi
  CONFIG_PATH["$number"]="${matches[0]}"
  name="$(awk -F': ' '/^experiment_name:/ {print $2; exit}' "${matches[0]}")"
  if [[ -z "$name" ]]; then
    echo "ERROR: could not read experiment_name from ${matches[0]}" >&2
    exit 2
  fi
  CONFIG_NAME["$number"]="$name"
done

LOG_ROOT="logs/lung_round/run_${RUN_ID}"
mkdir -p "$LOG_ROOT"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$CPU_THREADS_PER_JOB"
export MKL_NUM_THREADS="$CPU_THREADS_PER_JOB"
export OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB"
export NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB"

echo "===== YAML sanity check (4 configs parse, distinct experiment_name, all at 10k steps, matching sample lists) ====="
"$PYTHON_BIN" - <<PYEOF
import sys
from pathlib import Path
import yaml

numbers = ["302", "303", "301", "306"]
seen_names = {}
expected_samples = None
for n in numbers:
    matches = list(Path("configs/lung_round").glob(f"{n}_lung_*.yaml"))
    if len(matches) != 1:
        print(f"ERROR: expected exactly one config for {n}, found {matches}", file=sys.stderr)
        sys.exit(2)
    cfg = yaml.safe_load(matches[0].read_text())
    name = cfg["experiment_name"]
    if name in seen_names:
        print(f"ERROR: experiment_name {name!r} used by both {seen_names[name]} and {n}", file=sys.stderr)
        sys.exit(2)
    seen_names[name] = n
    if int(cfg["training"]["epochs"]) != 10000:
        print(f"ERROR: {n} does not have training.epochs=10000", file=sys.stderr)
        sys.exit(2)
    samples = tuple(cfg["data"]["sample_ids"])
    if expected_samples is None:
        expected_samples = samples
    elif samples != expected_samples:
        print(f"ERROR: {n} sample_ids {samples} differ from {numbers[0]}'s {expected_samples}", file=sys.stderr)
        sys.exit(2)
print(f"OK: {len(numbers)} configs parse, {len(seen_names)} distinct experiment_names, all at 10k steps, "
      f"same {len(expected_samples)} Lung samples.")
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

if [[ -z "${STPATH_GENE_VOC_PATH:-}" || ! -f "$STPATH_GENE_VOC_PATH" ]]; then
  echo "ERROR: STPATH_GENE_VOC_PATH is missing or not a file (needed by 301 directly; 302/303/306" >&2
  echo "  only via the shared config-schema check above, harmless if unused)." >&2
  exit 2
fi

echo "===== stpath package presence check (needed by 301 and 306) ====="
if ! "$PYTHON_BIN" -c 'import stpath' 2>/dev/null; then
  echo "ERROR: the external stpath package is not importable -- 301 and 306 both need it" >&2
  echo "  (git clone Graph-and-Geometric-Learning/STPath + pip install -e .)." >&2
  exit 2
fi

echo "===== Sample presence check (st + patches on disk for all 8 Lung samples) ====="
for sid in TENX72 TENX62 MEND90 MEND89 MEND88 MEND87 MEND86 MEND85; do
  test -f "data/raw/hest1k/st/${sid}.h5ad" || echo "WARNING: data/raw/hest1k/st/${sid}.h5ad not found (symlink set up?)" >&2
  test -f "data/raw/hest1k/patches/${sid}.h5" || echo "WARNING: data/raw/hest1k/patches/${sid}.h5 not found (symlink set up?)" >&2
done

job_is_done() {
  local name="${CONFIG_NAME[$1]}"
  test -f "results/checkpoints/lung_round/${name}/heldout_sample_summary.json"
}

run_one() {
  # Same --override smoke-test mechanism as run_consolidated_37_4gpu.sh's
  # run_one: 1 real step, tiny mask counts, redirected to smoke-only
  # checkpoint/mask-bank paths so it never collides with (or corrupts) the
  # real run's own paths.
  local number="$1" gpu="$2" smoke="$3"
  local name="${CONFIG_NAME[$number]}" path="${CONFIG_PATH[$number]}"
  local command=("$PYTHON_BIN" -m src.training.train --config "$path")
  if [[ "$smoke" == "1" ]]; then
    command+=(--override
      "experiment_name=${name}_smoke_${RUN_ID}"
      training.epochs=1 training.unique_mask_count=1
      training.checkpoint_every_n_steps=0 training.log_print_every_n_steps=1
      "training.checkpoint_dir=results/checkpoints/lung_round/smoke/${RUN_ID}/${name}"
      validation.every_n_steps=1 validation.early_stopping_min_steps=1
      validation.patience_checks=1000 validation.require_anchor_improvement=false
      "evaluation.training_mask_bank_path=results/mask_banks/training/lung_round/smoke_${RUN_ID}_${name}.json"
      evaluation.n_validation_masks=1 evaluation.n_test_masks=1 evaluation.n_samples=1
      "evaluation.mask_bank_dir=results/mask_banks/lung_round/smoke_${RUN_ID}_${name}"
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

if [[ "$SKIP_SMOKE" != "1" ]]; then
  echo "===== One-step fail-closed smoke test (4 configs in parallel) ====="
  smoke_pids=()
  for i in 0 1 2 3; do
    run_one "${NUMBERS[$i]}" "${GPUS[$i]}" 1 &
    smoke_pids+=("$!")
  done
  smoke_failed=0
  for pid in "${smoke_pids[@]}"; do wait "$pid" || smoke_failed=1; done
  if (( smoke_failed )); then
    echo "ERROR: smoke test failed; inspect $LOG_ROOT/smoke_*.log" >&2
    exit 1
  fi
  echo "OK: all 4 smoke tests passed."
fi

if [[ "$SMOKE_ONLY" == "1" ]]; then
  echo "SMOKE_ONLY=1 set, stopping before the full run."
  exit 0
fi

echo "===== Full run (4 configs, 1 per GPU, 10k steps each) ====="
run_pids=()
for i in 0 1 2 3; do
  run_one "${NUMBERS[$i]}" "${GPUS[$i]}" 0 &
  run_pids+=("$!")
done

failed=0
for i in 0 1 2 3; do
  number="${NUMBERS[$i]}"
  if ! wait "${run_pids[$i]}"; then
    echo "ERROR: ${CONFIG_NAME[$number]} failed; see $LOG_ROOT/full_${CONFIG_NAME[$number]}.log" >&2
    failed=1
  else
    echo "OK: ${CONFIG_NAME[$number]} finished."
  fi
done

if [[ "$failed" != "0" ]]; then
  echo "WARNING: at least one job failed -- check logs above." >&2
  exit 1
fi
echo "Lung round finished: all 4 training conditions completed."
