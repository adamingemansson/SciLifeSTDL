#!/usr/bin/env bash
# Lung round HOLEFIX (2026-07-24): reruns all 11 lung_round configs
# (301-311) with the real masking bug fixed. Every prior lung_round
# result was computed on masking.params.radius_range: [0.5, 1.0]
# (spot_spacing units) -- a radius that mathematically can almost never
# reach past the center spot, so 15 of every 16 test masks had exactly
# n_query=1 (a single point has zero variance, so every gene scores NaN
# and gets silently dropped) and the 16th had n_query=2 (Pearson r on 2
# points is forced to exactly +-1, per gene). The entire reported PCC for
# every 301-311 config was therefore coming from ONE single 2-spot mask,
# not a real measure of imputation quality. See git history: 69dd12a
# introduced [1.0,2.0] as a genuine small-hole diagnostic tier (for
# recovery_suite 207-210), then 1283426 shrunk it further to [0.5,1.0]
# before any run had launched -- that over-shrunk value is what got
# copied forward as the STANDING default for the whole lung_round series,
# never intended as such.
#
# Fixed to radius_range: [5.0, 8.0] spot_spacing (~90-230 query spots per
# hole, using the ~3.63*R^2 hex-packing density estimate) -- bigger than
# recovery_suite's own "normal" [3.0,6.0] tier for physical plausibility
# (real tissue damage: folds/tears/bubbles), while staying under ~40% of
# MEND85's (the Lung round's test sample) 604 post-QC spots so evaluation
# masks stay non-overlapping and context stays informative. Every
# experiment_name/checkpoint_dir/training_mask_bank_path got a "_holefix"
# suffix (fresh paths, no collision with the old broken results) and all
# 11 configs share one new eval mask bank:
# results/mask_banks/lung_round/transport_lung_v2_10k_holefix.
#
# All 11 configs run SEQUENTIALLY within 4 GPU lanes (not 1 config per
# GPU like earlier lung_round launchers -- there are more configs than
# GPUs this time), balanced by training cost (9 real 10k-step training
# configs + 2 eval-only epochs=1 baselines):
#   GPU[0] (default 1): 301 (stpath_scratch) -> 304 (harmonic) -> 305 (stpath pretrained eval)
#   GPU[1] (default 2): 302 (simple_fusion) -> 306 (simple_stpath_transformer) -> 310 (spatial_transformer_stpath_gene)
#   GPU[2] (default 3): 303 (simple_cross_attn) -> 307 (simple_cross_attn_decoder) -> 311 (cross_attn_stpath_gene)
#   GPU[3] (default 5): 308 (cross_attn_universal_gene) -> 309 (spatial_transformer_universal_gene)
#
# 301/305/306/309/310 need the `stpath` package (SpatialTransformer/
# STPathContextEncoder backbone); 301/305/308/309/310/311 need
# STPATH_GENE_VOC_PATH (universal gene-identity vocabulary).
#
# Usage:
#   STPATH_GENE_VOC_PATH=... bash scripts/run_lung_round_holefix_4gpu.sh
#   GPU_IDS=1,2,3,5 SMOKE_ONLY=1 bash scripts/run_lung_round_holefix_4gpu.sh
#   SKIP_SMOKE=1 bash scripts/run_lung_round_holefix_4gpu.sh
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

ALL_NUMBERS=(301 302 303 304 305 306 307 308 309 310 311)
STPATH_PACKAGE_NUMBERS=(301 305 306 309 310)
GENE_VOC_NUMBERS=(301 305 308 309 310 311)
TRAIN_NUMBERS=(301 302 303 306 307 308 309 310 311)
EVAL_ONLY_NUMBERS=(304 305)

# GPU lane assignment (see header for the balancing rationale).
LANE_0=(301 304 305)
LANE_1=(302 306 310)
LANE_2=(303 307 311)
LANE_3=(308 309)

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

echo "===== YAML sanity check (11 configs parse, distinct experiment_name, correct epochs, matching sample lists, fixed radius) ====="
"$PYTHON_BIN" - <<PYEOF
import sys
from pathlib import Path
import yaml

numbers = ["301","302","303","304","305","306","307","308","309","310","311"]
eval_only = {"304", "305"}
seen_names = {}
expected_samples = None
for n in numbers:
    matches = list(Path("configs/lung_round").glob(f"{n}_lung_*.yaml"))
    if len(matches) != 1:
        print(f"ERROR: expected exactly one config for {n}, found {matches}", file=sys.stderr)
        sys.exit(2)
    cfg = yaml.safe_load(matches[0].read_text())
    name = cfg["experiment_name"]
    if not name.endswith("_holefix"):
        print(f"ERROR: {n}'s experiment_name {name!r} is missing the _holefix suffix", file=sys.stderr)
        sys.exit(2)
    if name in seen_names:
        print(f"ERROR: experiment_name {name!r} used by both {seen_names[name]} and {n}", file=sys.stderr)
        sys.exit(2)
    seen_names[name] = n
    expected_epochs = 1 if n in eval_only else 10000
    if int(cfg["training"]["epochs"]) != expected_epochs:
        print(f"ERROR: {n} expected training.epochs={expected_epochs}, got {cfg['training']['epochs']}", file=sys.stderr)
        sys.exit(2)
    if int(cfg["training"]["unique_mask_count"]) > int(cfg["training"]["epochs"]):
        print(f"ERROR: {n} has unique_mask_count > epochs", file=sys.stderr)
        sys.exit(2)
    radius = list(cfg["masking"]["params"]["radius_range"])
    if radius != [5.0, 8.0]:
        print(f"ERROR: {n} has radius_range {radius}, expected [5.0, 8.0]", file=sys.stderr)
        sys.exit(2)
    if cfg["evaluation"]["mask_bank_dir"] != "results/mask_banks/lung_round/transport_lung_v2_10k_holefix":
        print(f"ERROR: {n} mask_bank_dir does not point at the new shared holefix bank", file=sys.stderr)
        sys.exit(2)
    samples = tuple(cfg["data"]["sample_ids"])
    if expected_samples is None:
        expected_samples = samples
    elif samples != expected_samples:
        print(f"ERROR: {n} sample_ids {samples} differ from {numbers[0]}'s {expected_samples}", file=sys.stderr)
        sys.exit(2)
print(f"OK: {len(numbers)} configs parse, {len(seen_names)} distinct experiment_names, "
      f"radius_range=[5.0,8.0] and shared holefix mask bank everywhere, "
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
  echo "ERROR: STPATH_GENE_VOC_PATH is missing or not a file (needed by 301/305/308/309/310/311)." >&2
  exit 2
fi

echo "===== stpath package presence check (needed by 301, 305, 306, 309, 310) ====="
if ! "$PYTHON_BIN" -c 'import stpath' 2>/dev/null; then
  echo "ERROR: the external stpath package is not importable -- 301, 305, 306, 309, and 310 all need it" >&2
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

run_lane() {
  local gpu="$1" smoke="$2"; shift 2
  local number
  for number in "$@"; do
    if ! run_one "$number" "$gpu" "$smoke"; then
      echo "ERROR: lane on GPU $gpu failed at config $number (${CONFIG_NAME[$number]})" >&2
      return 1
    fi
  done
}

if [[ "$SKIP_SMOKE" != "1" ]]; then
  echo "===== One-step fail-closed smoke test (11 configs, sequential per lane, 4 lanes in parallel) ====="
  smoke_pids=()
  run_lane "${GPUS[0]}" 1 "${LANE_0[@]}" & smoke_pids+=("$!")
  run_lane "${GPUS[1]}" 1 "${LANE_1[@]}" & smoke_pids+=("$!")
  run_lane "${GPUS[2]}" 1 "${LANE_2[@]}" & smoke_pids+=("$!")
  run_lane "${GPUS[3]}" 1 "${LANE_3[@]}" & smoke_pids+=("$!")
  smoke_failed=0
  for pid in "${smoke_pids[@]}"; do wait "$pid" || smoke_failed=1; done
  if (( smoke_failed )); then
    echo "ERROR: smoke test failed; inspect $LOG_ROOT/smoke_*.log" >&2
    exit 1
  fi
  echo "OK: all 11 smoke tests passed."
fi

if [[ "$SMOKE_ONLY" == "1" ]]; then
  echo "SMOKE_ONLY=1 set, stopping before the full run."
  exit 0
fi

echo "===== Full run (11 configs, 4 GPU lanes, sequential within each lane) ====="
run_pids=()
run_lane "${GPUS[0]}" 0 "${LANE_0[@]}" & run_pids+=("$!")
run_lane "${GPUS[1]}" 0 "${LANE_1[@]}" & run_pids+=("$!")
run_lane "${GPUS[2]}" 0 "${LANE_2[@]}" & run_pids+=("$!")
run_lane "${GPUS[3]}" 0 "${LANE_3[@]}" & run_pids+=("$!")

failed=0
for pid in "${run_pids[@]}"; do
  wait "$pid" || failed=1
done

if [[ "$failed" != "0" ]]; then
  echo "WARNING: at least one lane failed -- check logs under $LOG_ROOT and rerun (already-completed" >&2
  echo "  configs are skipped automatically via job_is_done)." >&2
  exit 1
fi
echo "Lung round HOLEFIX finished: all 11 conditions completed across 4 GPU lanes."