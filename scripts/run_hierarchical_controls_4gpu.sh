#!/usr/bin/env bash
# Four orthogonal exact-mask controls on the shared st-a100 GPU allocation.
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
  configs/recovery_suite/161_hierarchical_coordinate_only_20k.yaml
  configs/recovery_suite/162_hierarchical_local_he_raw_gex_20k.yaml
  configs/recovery_suite/163_hierarchical_global_he_raw_gex_20k.yaml
  configs/recovery_suite/164_hierarchical_official_stpath_control.yaml
)
NAMES=(
  hierarchical_control_coordinate_only_seed10
  hierarchical_control_local_he_raw_gex_seed10
  hierarchical_control_global_he_raw_gex_seed10
  hierarchical_control_official_stpath
)
LOG_ROOT="logs/recovery_suite/hierarchical_controls_${RUN_ID}"
REPORT_ROOT="reports/recovery_suite/hierarchical_controls_${RUN_ID}"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$CPU_THREADS_PER_JOB"
export MKL_NUM_THREADS="$CPU_THREADS_PER_JOB"
export OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB"
export NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB"
STPATH_ROOT="${STPATH_ROOT:-$REPO_ROOT/STPath}"
export STPATH_GENE_VOC_PATH="${STPATH_GENE_VOC_PATH:-$STPATH_ROOT/utils_data/symbol2ensembl.json}"
export STPATH_MODEL_WEIGHT_PATH="${STPATH_MODEL_WEIGHT_PATH:-$STPATH_ROOT/stfm.pth}"
export GIGAPATH_SLIDE_CHECKPOINT="${GIGAPATH_SLIDE_CHECKPOINT:-}"
mkdir -p "$LOG_ROOT" "$REPORT_ROOT"

echo "===== Static fail-closed audit of four parallel controls ====="
"$PYTHON_BIN" scripts/check_hierarchical_control_configs.py

if (( ${#GPUS[@]} != 4 )); then
  echo "ERROR: GPU_IDS must provide exactly four comma-separated devices." >&2
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
for required in \
  "${GIGAPATH_SLIDE_CHECKPOINT:-}" \
  "$STPATH_GENE_VOC_PATH" \
  "$STPATH_MODEL_WEIGHT_PATH"; do
  if [[ -z "$required" || ! -f "$required" ]]; then
    echo "ERROR: required checkpoint/vocabulary is missing: $required" >&2
    exit 2
  fi
done
if ! CUDA_VISIBLE_DEVICES="${GPUS[2]}" "$PYTHON_BIN" -c \
  'import torch; from gigapath.torchscale.component.flash_attention import flash_attn_func; assert torch.cuda.is_available() and flash_attn_func is not None'; then
  echo "ERROR: GigaPath LongNet FlashAttention is unavailable." >&2
  exit 2
fi
for sid in INT1 INT2 INT3 INT4 INT5 INT6 INT7 INT8; do
  test -f "data/cache/hest1k/gigapath_slide_cache/${sid}.npz" || {
    echo "ERROR: missing dense WSI cache for $sid" >&2
    exit 2
  }
done
if pgrep -af -- 'src.training.train.*hierarchical_control_' >"$LOG_ROOT/existing_processes.txt"; then
  echo "ERROR: hierarchical control jobs already exist:" >&2
  cat "$LOG_ROOT/existing_processes.txt" >&2
  exit 2
fi

run_phase() {
  local phase="$1" smoke="$2" i pid failed
  local pids=()
  for i in 0 1 2 3; do
    local command=("$PYTHON_BIN" -m src.training.train --config "${CONFIGS[$i]}")
    if [[ "$smoke" == "1" ]]; then
      command+=(--override
        "experiment_name=${NAMES[$i]}_smoke_${RUN_ID}"
        training.epochs=1 training.unique_mask_count=1
        training.checkpoint_every_n_steps=0 training.log_print_every_n_steps=1
        "training.checkpoint_dir=results/checkpoints/recovery_suite/smoke/${RUN_ID}/${NAMES[$i]}"
        validation.every_n_steps=1 validation.early_stopping_min_steps=1
        validation.patience_checks=1000 validation.require_anchor_improvement=false
        evaluation.n_validation_masks=1 evaluation.n_test_masks=1 evaluation.n_samples=1
        "evaluation.mask_bank_dir=results/mask_banks/recovery_suite/smoke_hierarchical_controls_${RUN_ID}"
        "evaluation.training_mask_bank_path=results/mask_banks/training/recovery_suite/smoke_${RUN_ID}_${NAMES[$i]}.json"
      )
    fi
    echo "GPU ${GPUS[$i]} -> ${NAMES[$i]} ($phase)"
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "${command[@]}" \
      >"$LOG_ROOT/${phase}_${NAMES[$i]}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  if (( failed )); then
    echo "ERROR: $phase failed; inspect $LOG_ROOT/${phase}_*.log" >&2
    return 1
  fi
}

if [[ "$SKIP_SMOKE" != "1" ]]; then
  echo "===== One-step fail-closed smoke test ====="
  run_phase smoke 1
fi
if [[ "$SMOKE_ONLY" == "1" ]]; then
  echo "Parallel-control smoke tests passed; full runs were not started."
  exit 0
fi

echo "===== Four exact-mask parallel controls ====="
run_phase full 0
"$PYTHON_BIN" scripts/summarize_hierarchical_controls.py \
  --output "$REPORT_ROOT/summary.csv"
echo "Hierarchical controls complete: $REPORT_ROOT/summary.csv"
