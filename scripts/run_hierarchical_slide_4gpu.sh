#!/usr/bin/env bash
# Four matched 20k-step jobs on the first four configured GPUs.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-2}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
GPU_IDS_CSV="${GPU_IDS:-1,2,3,5}"
IFS=',' read -r -a GPUS <<< "$GPU_IDS_CSV"
CONFIGS=(
  configs/recovery_suite/152_hierarchical_slide_full_20k.yaml
  configs/recovery_suite/153_hierarchical_no_slide_20k.yaml
  configs/recovery_suite/154_hierarchical_no_novae_20k.yaml
  configs/recovery_suite/155_hierarchical_he_only_20k.yaml
)
NAMES=(
  hierarchical_slide_full_seed10
  hierarchical_no_slide_seed10
  hierarchical_no_novae_seed10
  hierarchical_he_only_seed10
)
LOG_ROOT="logs/recovery_suite/hierarchical_slide_${RUN_ID}"
REPORT_ROOT="reports/recovery_suite/hierarchical_slide_${RUN_ID}"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$CPU_THREADS_PER_JOB"
export MKL_NUM_THREADS="$CPU_THREADS_PER_JOB"
export OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB"
export NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB"
mkdir -p "$LOG_ROOT" "$REPORT_ROOT"

if (( ${#GPUS[@]} < 4 || ${#GPUS[@]} > 8 )); then
  echo "ERROR: GPU_IDS must provide between 4 and 8 comma-separated devices." >&2
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
  echo "ERROR: GIGAPATH_SLIDE_CHECKPOINT is missing or not a file." >&2
  exit 2
fi
for sid in INT1 INT2 INT3 INT4 INT5 INT6 INT7 INT8; do
  test -f "data/cache/hest1k/gigapath_slide_cache/${sid}.npz" || {
    echo "ERROR: missing dense WSI cache for $sid; run a hierarchical precompute launcher" >&2
    exit 2
  }
done
if pgrep -af -- 'src.training.train.*hierarchical_' >"$LOG_ROOT/existing_processes.txt"; then
  echo "ERROR: hierarchical jobs already exist:" >&2
  cat "$LOG_ROOT/existing_processes.txt" >&2
  exit 2
fi

run_batch() {
  local phase="$1" smoke="$2"
  local pids=() i
  for i in 0 1 2 3; do
    local command=(
      "$PYTHON_BIN" -m src.training.train --config "${CONFIGS[$i]}"
    )
    if [[ "$smoke" == "1" ]]; then
      command+=(--override
        "experiment_name=${NAMES[$i]}_smoke_${RUN_ID}"
        training.epochs=1 training.unique_mask_count=1
        training.checkpoint_every_n_steps=0 training.log_print_every_n_steps=1
        "training.checkpoint_dir=results/checkpoints/recovery_suite/smoke/${RUN_ID}/${NAMES[$i]}"
        validation.every_n_steps=1 validation.early_stopping_min_steps=1
        validation.patience_checks=1000 validation.require_anchor_improvement=false
        evaluation.n_validation_masks=1 evaluation.n_test_masks=1 evaluation.n_samples=1
        "evaluation.mask_bank_dir=results/mask_banks/recovery_suite/smoke_hierarchical_${RUN_ID}"
        "evaluation.training_mask_bank_path=results/mask_banks/training/recovery_suite/smoke_${RUN_ID}_${NAMES[$i]}.json"
      )
    fi
    echo "GPU ${GPUS[$i]} -> ${NAMES[$i]} ($phase)"
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "${command[@]}" \
      >"$LOG_ROOT/${phase}_${NAMES[$i]}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0 pid
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  if (( failed )); then
    echo "ERROR: $phase failed; inspect $LOG_ROOT/${phase}_*.log" >&2
    return 1
  fi
}

if [[ "$SKIP_SMOKE" != "1" ]]; then
  echo "===== One-step fail-closed smoke test ====="
  run_batch smoke 1
fi
if [[ "$SMOKE_ONLY" == "1" ]]; then
  echo "Smoke tests passed; full runs were not started."
  exit 0
fi

echo "===== Four matched 20k held-out-sample runs ====="
failed=0
harmonic_pid=""
if (( ${#GPUS[@]} >= 5 )); then
  echo "GPU ${GPUS[4]} -> hierarchical_harmonic_k128 (full control)"
  CUDA_VISIBLE_DEVICES="${GPUS[4]}" "$PYTHON_BIN" -m src.training.train \
    --config configs/recovery_suite/156_hierarchical_harmonic_control.yaml \
    >"$LOG_ROOT/full_hierarchical_harmonic_k128.log" 2>&1 &
  harmonic_pid="$!"
fi
run_batch full 0 || failed=1
if [[ -n "$harmonic_pid" ]]; then
  wait "$harmonic_pid" || failed=1
else
  echo "===== Exact-mask non-learned harmonic control ====="
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PYTHON_BIN" -m src.training.train \
    --config configs/recovery_suite/156_hierarchical_harmonic_control.yaml \
    >"$LOG_ROOT/full_hierarchical_harmonic_k128.log" 2>&1 || failed=1
fi
if (( failed )); then
  echo "ERROR: at least one full job failed; inspect $LOG_ROOT/full_*.log" >&2
  exit 1
fi
"$PYTHON_BIN" scripts/summarize_hierarchical_slide.py \
  --output "$REPORT_ROOT/summary.csv"
echo "Hierarchical suite complete: $REPORT_ROOT/summary.csv"
