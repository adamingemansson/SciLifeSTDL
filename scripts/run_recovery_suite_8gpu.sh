#!/usr/bin/env bash
# Staged recovery/ablation runner for one machine with eight GPUs.
set -Eeuo pipefail

STAGE="${STAGE:-repair}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_IDS_ARR <<< "$GPU_IDS_CSV"
if [[ "${#GPU_IDS_ARR[@]}" -ne 8 ]]; then
  echo "ERROR: GPU_IDS must contain exactly eight comma-separated ids." >&2
  exit 2
fi

FRESH="${FRESH:-0}"
SMOKETEST="${SMOKETEST:-0}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_ROOT="${LOG_ROOT:-logs/recovery_suite/${STAGE}_${RUN_ID}}"
mkdir -p "$LOG_ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

N_CORES="$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
THREADS_PER_JOB=$((N_CORES / 8))
(( THREADS_PER_JOB > 0 )) || THREADS_PER_JOB=1

declare -a CONFIGS NAMES SEEDS
case "$STAGE" in
  repair)
    CONFIGS=(
      configs/recovery_suite/00_harmonic_anchor.yaml
      configs/recovery_suite/01_harmonic_residual_builtin.yaml
      configs/recovery_suite/01_harmonic_residual_builtin.yaml
      configs/recovery_suite/01_harmonic_residual_builtin.yaml
      configs/recovery_suite/02_harmonic_residual_stormlite_concat.yaml
      configs/recovery_suite/02_harmonic_residual_stormlite_concat.yaml
      configs/recovery_suite/02_harmonic_residual_stormlite_concat.yaml
      configs/recovery_suite/03_fixed_novae_flagship_regression.yaml
    )
    NAMES=(
      recovery_harmonic_anchor
      recovery_hr_builtin_seed0
      recovery_hr_builtin_seed1
      recovery_hr_builtin_seed2
      recovery_hr_concat_seed0
      recovery_hr_concat_seed1
      recovery_hr_concat_seed2
      recovery_fixed_novae_flagship_seed10
    )
    SEEDS=(0 0 1 2 0 1 2 10)
    ;;
  controls)
    : "${STPATH_GENE_VOC_PATH:?Set STPATH_GENE_VOC_PATH for the STPath controls}"
    : "${STPATH_MODEL_WEIGHT_PATH:?Set STPATH_MODEL_WEIGHT_PATH for pretrained STPath}"
    CONFIGS=(
      configs/recovery_suite/04_control_fm_builtin.yaml
      configs/recovery_suite/05_control_fm_stormlite.yaml
      configs/recovery_suite/03_fixed_novae_flagship_regression.yaml
      configs/recovery_suite/03_fixed_novae_flagship_regression.yaml
      configs/recovery_suite/06_stpath_pretrained_benchmark.yaml
      configs/recovery_suite/06_stpath_pretrained_benchmark.yaml
      configs/recovery_suite/07_stpath_unfrozen_benchmark.yaml
      configs/recovery_suite/07_stpath_unfrozen_benchmark.yaml
    )
    NAMES=(
      recovery_control_fm_builtin_seed0
      recovery_control_fm_stormlite_seed0
      recovery_fixed_novae_flagship_seed11
      recovery_fixed_novae_flagship_seed12
      recovery_stpath_pretrained_seed10
      recovery_stpath_pretrained_seed11
      recovery_stpath_unfrozen_seed10
      recovery_stpath_unfrozen_seed11
    )
    SEEDS=(0 0 11 12 10 11 10 11)
    ;;
  *)
    echo "ERROR: STAGE must be repair or controls" >&2
    exit 2
    ;;
esac

if [[ "$STAGE" == "controls" && "$SMOKETEST" != "1" ]]; then
  "$PYTHON_BIN" scripts/check_recovery_gate.py
fi

# All context-only Novae jobs consume the same immutable 64-mask schedule.
# Populate it once before concurrent readers start. The cache is reused safely.
if [[ "$SMOKETEST" != "1" ]] && [[ "$STAGE" == "repair" || "$STAGE" == "controls" ]]; then
  echo "Precomputing/reusing the shared context-only Novae mask cache on 8 GPUs..."
  precompute_pids=()
  for slot in "${!GPU_IDS_ARR[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU_IDS_ARR[$slot]}" \
      OMP_NUM_THREADS="$THREADS_PER_JOB" MKL_NUM_THREADS="$THREADS_PER_JOB" \
      "$PYTHON_BIN" scripts/precompute_context_novae_cache.py \
        --config configs/recovery_suite/03_fixed_novae_flagship_regression.yaml \
        --shard-index "$slot" --num-shards 8 \
        > "$LOG_ROOT/precompute_context_novae_shard${slot}.log" 2>&1 &
    precompute_pids+=("$!")
  done
  precompute_failed=0
  for slot in "${!precompute_pids[@]}"; do
    if ! wait "${precompute_pids[$slot]}"; then
      precompute_failed=1
      echo "FAILED: context-only Novae precompute shard $slot" >&2
      tail -80 "$LOG_ROOT/precompute_context_novae_shard${slot}.log" >&2 || true
    fi
  done
  if (( precompute_failed )); then
    echo "Novae precomputation failed. Training was not started." >&2
    exit 1
  fi
fi

pids=()
for slot in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$slot]}"
  name="${NAMES[$slot]}"
  seed="${SEEDS[$slot]}"
  gpu="${GPU_IDS_ARR[$slot]}"
  checkpoint="results/checkpoints/recovery_suite/$name"
  metrics="$checkpoint/audit_test_metrics.json"
  log="$LOG_ROOT/$name.log"

  if [[ "$FRESH" != "1" && -f "$metrics" ]]; then
    echo "GPU $gpu -> $name: SKIP (completed metrics found)"
    pids+=("")
    continue
  fi

  overrides=(
    "experiment_name=$name"
    "training.seed=$seed"
    "training.checkpoint_dir=$checkpoint"
  )
  if [[ "$SMOKETEST" == "1" ]]; then
    overrides+=(
      "training.epochs=20"
      "training.unique_mask_count=20"
      "training.checkpoint_every_n_steps=0"
      "training.log_print_every_n_steps=1"
      "validation.every_n_steps=10"
      "validation.early_stopping_min_steps=20"
      "validation.patience_checks=1000"
      "validation.require_anchor_improvement=false"
      "evaluation.n_samples=2"
      "evaluation.training_mask_bank_path=results/mask_banks/training/recovery_suite/smoke_${name}.json"
    )
  fi

  echo "GPU $gpu -> $name"
  (
    printf 'command:' > "$log"
    printf ' %q' env "CUDA_VISIBLE_DEVICES=$gpu" "$PYTHON_BIN" -m src.training.train \
      --config "$config" --override "${overrides[@]}" >> "$log"
    printf '\n' >> "$log"
    CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="$THREADS_PER_JOB" \
      MKL_NUM_THREADS="$THREADS_PER_JOB" \
      "$PYTHON_BIN" -m src.training.train --config "$config" \
      --override "${overrides[@]}" >> "$log" 2>&1
  ) &
  pids+=("$!")
done

failed=0
for slot in "${!pids[@]}"; do
  [[ -n "${pids[$slot]}" ]] || continue
  if wait "${pids[$slot]}"; then
    echo "DONE: ${NAMES[$slot]}"
  else
    failed=1
    echo "FAILED: ${NAMES[$slot]}" >&2
    tail -80 "$LOG_ROOT/${NAMES[$slot]}.log" >&2 || true
  fi
done

if (( failed )); then
  echo "Stage $STAGE failed. Do not promote. Logs: $LOG_ROOT" >&2
  exit 1
fi

if [[ "$STAGE" == "repair" && "$SMOKETEST" != "1" ]]; then
  "$PYTHON_BIN" scripts/check_recovery_gate.py
fi

"$PYTHON_BIN" scripts/collect_audit_results.py \
  --checkpoint-root results/checkpoints/recovery_suite \
  --output-dir "reports/recovery_suite/${STAGE}_${RUN_ID}" \
  --log-root "$LOG_ROOT"

echo "Stage $STAGE completed. Review reports/recovery_suite/${STAGE}_${RUN_ID}/summary.csv"
