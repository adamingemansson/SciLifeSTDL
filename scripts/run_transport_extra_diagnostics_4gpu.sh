#!/usr/bin/env bash
# Two independent diagnostic axes for hierarchical_gene_transport_regressor,
# run together across 4 GPUs (2026-07-23):
#   - smallhole (207-210): radius_range [3.0,6.0] -> [0.5,1.0], tests whether
#     hole size itself caps performance.
#   - local_k sweep (211-214): local_k 128 -> 256/512 at the ORIGINAL hole
#     size, tests whether the model is neighbor-starved rather than
#     architecture-limited. Kept separate from the smallhole axis so the two
#     don't get conflated.
# Neither axis has O0X-style capacity gates -- every config here is a direct
# held-out run (INT1-6 train / INT7 val / INT8 test), so this script is
# simpler than run_transport_suite_v2_4gpu.sh: no gate-pass/fail branching.
#
# Each GPU's queue is exactly the one dependency that exists in this batch:
# a harmonic control's evaluation.mask_bank_dir points at its own learned
# sibling's mask bank, so the learned run must go first on the same GPU
# (same reasoning as v2 runner's post-queue harmonic-control step, just
# co-located per-GPU here instead of deferred to the very end since there's
# no capacity-gate barrier to wait on).
#
#   GPU[0] (default 1): 207 (C01 smallhole)          -> 208 (C07 smallhole)
#   GPU[1] (default 2): 209 (C05 smallhole)          -> 210 (harmonic smallhole control)
#   GPU[2] (default 3): 211 (C05 k256)               -> 212 (harmonic k256 control)
#   GPU[3] (default 5): 213 (C05 k512)               -> 214 (harmonic k512 control)
#
# Usage:
#   bash scripts/run_transport_extra_diagnostics_4gpu.sh
#   GPU_IDS=1,2,3,5 SMOKE_ONLY=1 bash scripts/run_transport_extra_diagnostics_4gpu.sh
#   SKIP_SMOKE=1 bash scripts/run_transport_extra_diagnostics_4gpu.sh
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

QUEUE_1=(207 208)  # C01 smallhole -> C07 smallhole (independent of each other)
QUEUE_2=(209 210)  # C05 smallhole -> harmonic smallhole control (210 depends on 209's mask bank)
QUEUE_3=(211 212)  # C05 k256 -> harmonic k256 control (212 depends on 211's mask bank)
QUEUE_4=(213 214)  # C05 k512 -> harmonic k512 control (214 depends on 213's mask bank)
QUEUES=(QUEUE_1 QUEUE_2 QUEUE_3 QUEUE_4)

declare -A CONFIG_PATH
declare -A CONFIG_NAME
for f in configs/recovery_suite/2[01][0-9]_transport_*.yaml; do
  number="$(basename "$f" | cut -d_ -f1)"
  [[ "$number" =~ ^(207|208|209|210|211|212|213|214)$ ]] || continue
  name="$(awk -F': ' '/^experiment_name:/ {print $2; exit}' "$f")"
  if [[ -z "$name" ]]; then
    echo "ERROR: could not read experiment_name from $f" >&2
    exit 2
  fi
  CONFIG_PATH["$number"]="$f"
  CONFIG_NAME["$number"]="$name"
done
for number in 207 208 209 210 211 212 213 214; do
  if [[ -z "${CONFIG_PATH[$number]:-}" ]]; then
    echo "ERROR: config $number not found under configs/recovery_suite/" >&2
    exit 2
  fi
done

LOG_ROOT="logs/recovery_suite/transport_extra_diagnostics_${RUN_ID}"
REPORT_ROOT="reports/recovery_suite/transport_extra_diagnostics_${RUN_ID}"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$CPU_THREADS_PER_JOB"
export MKL_NUM_THREADS="$CPU_THREADS_PER_JOB"
export OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB"
export NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB"
mkdir -p "$LOG_ROOT" "$REPORT_ROOT"

echo "===== YAML sanity check (all 8 configs parse and have distinct experiment_name/checkpoint paths) ====="
"$PYTHON_BIN" - <<'PYEOF'
import sys
from pathlib import Path
import yaml

numbers = ["207", "208", "209", "210", "211", "212", "213", "214"]
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
print(f"OK: {len(numbers)} configs parse, {len(seen_names)} distinct experiment_names.")
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
  echo "ERROR: transport jobs already exist (some other suite/diagnostic is running):" >&2
  cat "$LOG_ROOT/existing_processes.txt" >&2
  exit 2
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
      "evaluation.mask_bank_dir=results/mask_banks/recovery_suite/smoke_extra_diagnostics_${RUN_ID}_${name}"
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

if [[ "$SKIP_SMOKE" != "1" ]]; then
  echo "===== One-step fail-closed smoke test (first job of each queue) ====="
  smoke_pids=()
  for idx in 0 1 2 3; do
    queue_name="${QUEUES[$idx]}"
    declare -n q_ref="$queue_name"
    run_one "${q_ref[0]}" "${GPUS[$idx]}" 1 &
    smoke_pids+=("$!")
  done
  smoke_failed=0
  for pid in "${smoke_pids[@]}"; do wait "$pid" || smoke_failed=1; done
  if (( smoke_failed )); then
    echo "ERROR: smoke test failed; inspect $LOG_ROOT/smoke_*.log" >&2
    exit 1
  fi
  echo "Smoke tests passed."
fi
if [[ "$SMOKE_ONLY" == "1" ]]; then
  echo "Smoke tests passed; full runs were not started."
  exit 0
fi

echo "===== Four parallel queues (2 jobs each) ====="
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
"$PYTHON_BIN" scripts/summarize_transport_extra_diagnostics.py --output "$REPORT_ROOT/summary.csv" || true
echo "Extra diagnostics run finished (RUN_ID=$RUN_ID). Report: $REPORT_ROOT/summary.csv"
echo "Logs: $LOG_ROOT"
