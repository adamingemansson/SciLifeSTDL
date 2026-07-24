#!/usr/bin/env bash
# Lung round PLACEMENT/SAMPLE diagnostic (2026-07-24): tries to explain
# the real, still-open gap between our lung_round STPath/harmonic numbers
# and a reference STPath benchmark notebook's much higher ones (harmonic:
# notebook ~0.29-0.33 vs our holefix 304 ~0.008; STPath in-context+
# image-ablated: notebook ~0.146-0.159 vs our holefix 305 ~0.02-0.05).
# Two prior explanations were RULED OUT: our STPathContextEncoder already
# injects real context-spot expression as ge_tokens (verified against the
# real STPathInference.inference() source, matches exactly), and hole
# size/shape/single-hole-per-eval already matched what was intended
# (n_patches=1, shape=mixed, radius fixed to [5.0,8.0] spot_spacing this
# same session). Also fixed along the way: GigaPath preprocessing was
# doing an extra bicubic resize-to-256 before center-cropping to 224 that
# STPath's own real pipeline never does (HEST-1k patches are already
# 224x224, so STPath's real CenterCrop(224) is a no-op there) --
# src/models/conditioning.py's _gigapath_preprocess_and_encode, affects
# every image-conditioned config in this project, not just STPath ones.
#
# This script tests the two remaining candidate explanations, in
# parallel, one config per GPU:
#   GPU[0] (default 1): 312 harmonic,  MEND85, masking.center_mode=geometric_median (placement effect)
#   GPU[1] (default 2): 313 stpath,    MEND85, masking.center_mode=geometric_median (placement effect)
#   GPU[2] (default 3): 314 harmonic,  TENX118, unchanged random placement (sample effect)
#   GPU[3] (default 5): 315 stpath,    TENX118, unchanged random placement (sample effect)
#
# 312/313 hold the sample fixed (MEND85) and only change WHERE the hole
# is placed (always the slide's own geometric-median spot, matching the
# notebook's `choose_central_window`/`make_central_mask`, instead of our
# usual uniformly-random placement). 314/315 hold placement fixed
# (unchanged random) and only change the SAMPLE (TENX118, the exact
# sample the reference notebook used, instead of our own MEND85).
# Comparing each pair's PCC against 304/305's already-known holefix
# results tells us how much of the gap is placement vs. sample choice.
#
# 314/315 need TENX118 present under data/raw/hest1k/st/TENX118.h5ad +
# patches/TENX118.h5 -- if your local HEST-1k mirror doesn't have it,
# this script SKIPS those two (not a hard failure) and still runs 312/313.
#
# Usage:
#   STPATH_GENE_VOC_PATH=... STPATH_MODEL_WEIGHT_PATH=... \
#     bash scripts/run_lung_round_placement_diagnostic.sh
#   GPU_IDS=1,2,3,5 bash scripts/run_lung_round_placement_diagnostic.sh
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-2}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
GPU_IDS_CSV="${GPU_IDS:-1,2,3,5}"
IFS=',' read -r -a GPUS <<< "$GPU_IDS_CSV"

ALL_NUMBERS=(312 313 314 315)
declare -A CONFIG_PATH
declare -A CONFIG_NAME
for number in "${ALL_NUMBERS[@]}"; do
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

echo "===== YAML sanity check (4 configs parse, distinct experiment_name, eval-only) ====="
"$PYTHON_BIN" - <<PYEOF
import sys
from pathlib import Path
import yaml

numbers = ["312", "313", "314", "315"]
seen_names = {}
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
    if int(cfg["training"]["epochs"]) != 1:
        print(f"ERROR: {n} is meant to be eval-only (epochs=1), got {cfg['training']['epochs']}", file=sys.stderr)
        sys.exit(2)
print(f"OK: {len(numbers)} configs parse, {len(seen_names)} distinct experiment_names, all eval-only.")
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

if [[ -z "${STPATH_GENE_VOC_PATH:-}" || ! -f "$STPATH_GENE_VOC_PATH" ]]; then
  echo "ERROR: STPATH_GENE_VOC_PATH is missing or not a file (needed by 313/315)." >&2
  exit 2
fi
if [[ -z "${STPATH_MODEL_WEIGHT_PATH:-}" || ! -f "$STPATH_MODEL_WEIGHT_PATH" ]]; then
  echo "ERROR: STPATH_MODEL_WEIGHT_PATH is missing or not a file (needed by 313/315)." >&2
  exit 2
fi

echo "===== stpath package presence check (needed by 313, 315) ====="
if ! "$PYTHON_BIN" -c 'import stpath' 2>/dev/null; then
  echo "ERROR: the external stpath package is not importable -- 313 and 315 need it." >&2
  exit 2
fi

echo "===== MEND85 presence check (needed by 312, 313) ====="
for sid in TENX72 TENX62 MEND90 MEND89 MEND88 MEND87 MEND86 MEND85; do
  test -f "data/raw/hest1k/st/${sid}.h5ad" || echo "WARNING: data/raw/hest1k/st/${sid}.h5ad not found" >&2
  test -f "data/raw/hest1k/patches/${sid}.h5" || echo "WARNING: data/raw/hest1k/patches/${sid}.h5 not found" >&2
done

echo "===== TENX118 presence check (needed by 314, 315 -- SOFT, skips those two if missing) ====="
TENX118_AVAILABLE=1
if [[ ! -f "data/raw/hest1k/st/TENX118.h5ad" || ! -f "data/raw/hest1k/patches/TENX118.h5" ]]; then
  TENX118_AVAILABLE=0
  echo "TENX118 not found locally (data/raw/hest1k/st/TENX118.h5ad and/or patches/TENX118.h5 missing)." >&2
  echo "  314 and 315 will be SKIPPED. To include them, download TENX118 the same way" >&2
  echo "  docs/dataset_notes.md documents for the other HEST-1k samples, then rerun." >&2
else
  echo "OK: TENX118 present."
fi

job_is_done() {
  local number="$1"
  local name="${CONFIG_NAME[$number]}"
  test -f "results/checkpoints/lung_round/${name}/heldout_sample_summary.json"
}

run_one() {
  local number="$1" gpu="$2"
  local name="${CONFIG_NAME[$number]}" path="${CONFIG_PATH[$number]}"
  if job_is_done "$number"; then
    echo "GPU $gpu -> $name SKIP (already completed, found heldout_sample_summary.json)"
    return 0
  fi
  echo "GPU $gpu -> $name START"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m src.training.train --config "$path" \
    >"$LOG_ROOT/full_${name}.log" 2>&1
}

echo "===== Running diagnostic configs (eval-only, no smoke test needed) ====="
run_pids=()
run_numbers=()

run_one 312 "${GPUS[0]}" & run_pids+=("$!"); run_numbers+=(312)
run_one 313 "${GPUS[1]}" & run_pids+=("$!"); run_numbers+=(313)
if [[ "$TENX118_AVAILABLE" == "1" ]]; then
  run_one 314 "${GPUS[2]}" & run_pids+=("$!"); run_numbers+=(314)
  run_one 315 "${GPUS[3]}" & run_pids+=("$!"); run_numbers+=(315)
else
  echo "Skipping 314/315 (TENX118 unavailable) -- only GPUs ${GPUS[0]} and ${GPUS[1]} are used this run."
fi

failed=0
for i in "${!run_pids[@]}"; do
  number="${run_numbers[$i]}"
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
echo "Placement/sample diagnostic finished."
