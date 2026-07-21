#!/usr/bin/env bash
# Staged recovery/ablation runner. The legacy filename is retained for
# compatibility. The default shared-server profile enforces GPUs 0-3; the
# explicit dedicated8 profile is only for a separate machine allocated in full.
set -Eeuo pipefail

STAGE="${STAGE:-repair}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPU_IDS_ARR <<< "$GPU_IDS_CSV"
SERVER_PROFILE="${SERVER_PROFILE:-shared4}"
case "$SERVER_PROFILE" in
  shared4)
    if [[ "$GPU_IDS_CSV" != "0,1,2,3" ]]; then
      echo "ERROR: shared4 allocation is fixed to GPU_IDS=0,1,2,3." >&2
      exit 2
    fi
    ;;
  dedicated8)
    if [[ "$GPU_IDS_CSV" != "0,1,2,3,4,5,6,7" ]]; then
      echo "ERROR: dedicated8 requires GPU_IDS=0,1,2,3,4,5,6,7." >&2
      exit 2
    fi
    ;;
  *)
    echo "ERROR: SERVER_PROFILE must be shared4 or dedicated8." >&2
    exit 2
    ;;
esac
GPU_COUNT="${#GPU_IDS_ARR[@]}"

CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-4}"
if [[ ! "$CPU_THREADS_PER_JOB" =~ ^[0-9]+$ ]] \
    || (( CPU_THREADS_PER_JOB < 1 || CPU_THREADS_PER_JOB > 4 )); then
  echo "ERROR: CPU_THREADS_PER_JOB must be an integer from 1 to 4 (default: 4)." >&2
  exit 2
fi

FRESH="${FRESH:-0}"
SMOKETEST="${SMOKETEST:-0}"
SMOKE_STEPS="${SMOKE_STEPS:-20}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_ROOT="${LOG_ROOT:-logs/recovery_suite/${STAGE}_${RUN_ID}}"
mkdir -p "$LOG_ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Directly executed helper scripts live under scripts/, so Python otherwise
# places that directory (not the repository root) on sys.path and cannot
# import the sibling src package.
REPO_ROOT="$(pwd -P)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
STPATH_ROOT="${STPATH_ROOT:-$(dirname "$REPO_ROOT")/STPath}"
if [[ -z "${STPATH_GENE_VOC_PATH:-}" || "${STPATH_GENE_VOC_PATH:-}" == /absolute/path/* ]]; then
  export STPATH_GENE_VOC_PATH="$STPATH_ROOT/utils_data/symbol2ensembl.json"
fi
if [[ -z "${STPATH_MODEL_WEIGHT_PATH:-}" || "${STPATH_MODEL_WEIGHT_PATH:-}" == /absolute/path/* ]]; then
  export STPATH_MODEL_WEIGHT_PATH="$STPATH_ROOT/stfm.pth"
fi

THREADS_PER_JOB="$CPU_THREADS_PER_JOB"

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
    if [[ "$STPATH_GENE_VOC_PATH" == /absolute/path/* || ! -f "$STPATH_GENE_VOC_PATH" ]]; then
      echo "ERROR: STPATH_GENE_VOC_PATH is not a real file: $STPATH_GENE_VOC_PATH" >&2
      echo "Expected the STPath symbol2ensembl.json vocabulary file." >&2
      exit 2
    fi
    if [[ "$STPATH_MODEL_WEIGHT_PATH" == /absolute/path/* || ! -f "$STPATH_MODEL_WEIGHT_PATH" ]]; then
      echo "ERROR: STPATH_MODEL_WEIGHT_PATH is not a real file: $STPATH_MODEL_WEIGHT_PATH" >&2
      echo "Expected the downloaded STPath stfm.pth checkpoint." >&2
      exit 2
    fi
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
  ablations)
    CONFIGS=(
      configs/recovery_suite/08_ablation_mlp_only.yaml
      configs/recovery_suite/09_ablation_novae_only.yaml
      configs/recovery_suite/10_ablation_no_he.yaml
      configs/recovery_suite/11_ablation_query_image_dropout.yaml
      configs/recovery_suite/12_ablation_dense_decoder.yaml
      configs/recovery_suite/13_ablation_fusion_sum.yaml
      configs/recovery_suite/14_ablation_no_spatial_bias.yaml
      configs/recovery_suite/15_ablation_qk_norm.yaml
    )
    NAMES=(
      recovery_ablation_mlp_only_seed10
      recovery_ablation_novae_only_seed10
      recovery_ablation_no_he_seed10
      recovery_ablation_query_image_dropout_seed10
      recovery_ablation_dense_decoder_seed10
      recovery_ablation_fusion_sum_seed10
      recovery_ablation_no_spatial_bias_seed10
      recovery_ablation_qk_norm_seed10
    )
    SEEDS=(10 10 10 10 10 10 10 10)
    ;;
  wave3)
    : "${STPATH_GENE_VOC_PATH:?Set STPATH_GENE_VOC_PATH for the Wave 3 STPath benchmark}"
    : "${STPATH_MODEL_WEIGHT_PATH:?Set STPATH_MODEL_WEIGHT_PATH for the Wave 3 STPath benchmark}"
    if [[ ! -f "$STPATH_GENE_VOC_PATH" || ! -f "$STPATH_MODEL_WEIGHT_PATH" ]]; then
      echo "ERROR: Wave 3 requires real STPath vocabulary and checkpoint files." >&2
      echo "vocabulary: $STPATH_GENE_VOC_PATH" >&2
      echo "checkpoint: $STPATH_MODEL_WEIGHT_PATH" >&2
      exit 2
    fi
    CONFIGS=(
      configs/recovery_suite/05_control_fm_stormlite.yaml
      configs/recovery_suite/05_control_fm_stormlite.yaml
      configs/recovery_suite/16_wave3_stormlite_control_cap3000.yaml
      configs/recovery_suite/06_stpath_pretrained_benchmark.yaml
      configs/recovery_suite/17_wave3_flagship_modern_transport.yaml
      configs/recovery_suite/18_wave3_flagship_adaln_warmup.yaml
      configs/recovery_suite/19_wave3_flagship_modern_recipe.yaml
      configs/recovery_suite/19_wave3_flagship_modern_recipe.yaml
    )
    NAMES=(
      recovery_wave3_stormlite_control_seed1
      recovery_wave3_stormlite_control_seed2
      recovery_wave3_stormlite_matched_cap3000_seed0
      recovery_wave3_stpath_pretrained_seed12
      recovery_wave3_flagship_modern_transport_seed10
      recovery_wave3_flagship_adaln_warmup_seed10
      recovery_wave3_flagship_modern_recipe_seed10
      recovery_wave3_flagship_modern_recipe_seed11
    )
    SEEDS=(1 2 0 12 10 10 10 11)
    ;;
  component40k)
    CONFIGS=(
      configs/recovery_suite/20_component40k_warmup_only.yaml
      configs/recovery_suite/21_component40k_adaln_only.yaml
      configs/recovery_suite/22_component40k_coord_augment_only.yaml
      configs/recovery_suite/23_component40k_logit_time_only.yaml
      configs/recovery_suite/24_component40k_minibatch_ot_only.yaml
      configs/recovery_suite/25_component40k_lower_lr_only.yaml
      configs/recovery_suite/26_component40k_boundary_only.yaml
      configs/recovery_suite/27_component40k_logit_ot_pair.yaml
    )
    NAMES=(
      recovery_component40k_warmup_only_seed10
      recovery_component40k_adaln_only_seed10
      recovery_component40k_coord_augment_only_seed10
      recovery_component40k_logit_time_only_seed10
      recovery_component40k_minibatch_ot_only_seed10
      recovery_component40k_lower_lr_only_seed10
      recovery_component40k_boundary_only_seed10
      recovery_component40k_logit_ot_pair_seed10
    )
    SEEDS=(10 10 10 10 10 10 10 10)
    ;;
  *)
    echo "ERROR: STAGE must be repair, controls, ablations, wave3 or component40k" >&2
    exit 2
    ;;
esac

if [[ ( "$STAGE" == "controls" || "$STAGE" == "ablations" || "$STAGE" == "wave3" ) && "$SMOKETEST" != "1" ]]; then
  # The direct harmonic-residual branch is an independent rejected
  # diagnostic. Wave 1 tests the historically working FM family and only
  # depends on the clean FM flagship being finite/noncollapsed.
  "$PYTHON_BIN" scripts/check_recovery_gate.py --mode flagship
fi

# All context-only Novae jobs consume the same immutable 64-mask schedule.
# Populate it once before concurrent readers start. The cache is reused safely.
if [[ "$SMOKETEST" != "1" ]] && [[ "$STAGE" == "repair" || "$STAGE" == "controls" || "$STAGE" == "ablations" || "$STAGE" == "wave3" || "$STAGE" == "component40k" ]]; then
  echo "Precomputing/reusing the shared context-only Novae mask cache on $GPU_COUNT GPUs..."
  precompute_pids=()
  for slot in "${!GPU_IDS_ARR[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU_IDS_ARR[$slot]}" \
      OMP_NUM_THREADS="$THREADS_PER_JOB" MKL_NUM_THREADS="$THREADS_PER_JOB" \
      OPENBLAS_NUM_THREADS="$THREADS_PER_JOB" NUMEXPR_NUM_THREADS="$THREADS_PER_JOB" \
      "$PYTHON_BIN" scripts/precompute_context_novae_cache.py \
        --config configs/recovery_suite/03_fixed_novae_flagship_regression.yaml \
        --shard-index "$slot" --num-shards "$GPU_COUNT" \
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

failed=0
JOB_COUNT="${#CONFIGS[@]}"
for (( batch_start=0; batch_start<JOB_COUNT; batch_start+=GPU_COUNT )); do
  batch_number=$((batch_start / GPU_COUNT + 1))
  batch_total=$(((JOB_COUNT + GPU_COUNT - 1) / GPU_COUNT))
  echo "Starting batch $batch_number/$batch_total with at most $GPU_COUNT jobs " \
       "and $THREADS_PER_JOB CPU threads per job."
  pids=()
  batch_names=()

  for (( local_slot=0; local_slot<GPU_COUNT; local_slot++ )); do
    job_index=$((batch_start + local_slot))
    (( job_index < JOB_COUNT )) || break
    config="${CONFIGS[$job_index]}"
    name="${NAMES[$job_index]}"
    seed="${SEEDS[$job_index]}"
    gpu="${GPU_IDS_ARR[$local_slot]}"
    checkpoint="results/checkpoints/recovery_suite/$name"
    metrics="$checkpoint/audit_test_metrics.json"
    log="$LOG_ROOT/$name.log"

    if [[ "$FRESH" != "1" && -f "$metrics" ]]; then
      echo "GPU $gpu -> $name: SKIP (completed metrics found)"
      continue
    fi

    overrides=(
      "experiment_name=$name"
      "training.seed=$seed"
      "training.checkpoint_dir=$checkpoint"
    )
    if [[ "$SMOKETEST" == "1" ]]; then
      overrides+=(
        "training.epochs=$SMOKE_STEPS"
        "training.unique_mask_count=$SMOKE_STEPS"
        "training.checkpoint_every_n_steps=0"
        "training.checkpoint_dir=results/checkpoints/recovery_suite/smoke/${RUN_ID}/$name"
        "training.log_print_every_n_steps=1"
        "validation.every_n_steps=$SMOKE_STEPS"
        "validation.early_stopping_min_steps=$SMOKE_STEPS"
        "validation.patience_checks=1000"
        "validation.require_anchor_improvement=false"
        "validation.n_samples=1"
        "evaluation.n_validation_masks=1"
        "evaluation.n_test_masks=1"
        "evaluation.n_samples=1"
        "evaluation.mask_bank_path=results/mask_banks/recovery_suite/smoke_${name}.json"
        "evaluation.training_mask_bank_path=results/mask_banks/training/recovery_suite/smoke_${name}.json"
      )
    fi

    echo "GPU $gpu -> $name"
    (
      started_epoch="$(date +%s)"
      printf 'command:' > "$log"
      printf ' %q' env "CUDA_VISIBLE_DEVICES=$gpu" \
        "OMP_NUM_THREADS=$THREADS_PER_JOB" "MKL_NUM_THREADS=$THREADS_PER_JOB" \
        "OPENBLAS_NUM_THREADS=$THREADS_PER_JOB" "NUMEXPR_NUM_THREADS=$THREADS_PER_JOB" \
        "$PYTHON_BIN" -m src.training.train --config "$config" \
        --override "${overrides[@]}" >> "$log"
      printf '\n' >> "$log"
      printf 'started_at_utc: %s\n' "$(date -u +%FT%TZ)" >> "$log"
      if CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="$THREADS_PER_JOB" \
          MKL_NUM_THREADS="$THREADS_PER_JOB" \
          OPENBLAS_NUM_THREADS="$THREADS_PER_JOB" \
          NUMEXPR_NUM_THREADS="$THREADS_PER_JOB" \
          "$PYTHON_BIN" -m src.training.train --config "$config" \
          --override "${overrides[@]}" >> "$log" 2>&1; then
        run_status=0
      else
        run_status=$?
      fi
      finished_epoch="$(date +%s)"
      printf 'finished_at_utc: %s\n' "$(date -u +%FT%TZ)" >> "$log"
      printf 'wall_seconds: %d\n' "$((finished_epoch - started_epoch))" >> "$log"
      printf 'exit_status: %d\n' "$run_status" >> "$log"
      exit "$run_status"
    ) &
    pids+=("$!")
    batch_names+=("$name")
  done

  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      echo "DONE: ${batch_names[$i]}"
    else
      failed=1
      echo "FAILED: ${batch_names[$i]}" >&2
      tail -80 "$LOG_ROOT/${batch_names[$i]}.log" >&2 || true
    fi
  done

  if (( failed )); then
    echo "Stage $STAGE failed in batch $batch_number. Later batches were not started." >&2
    break
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
