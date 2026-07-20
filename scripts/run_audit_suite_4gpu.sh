#!/usr/bin/env bash
# Run the validity audit as consecutive waves, with at most one process per GPU.
set -Eeuo pipefail

DEVICES_STRING="${DEVICES:-0 1 2 3}"
read -r -a GPU_IDS <<< "$DEVICES_STRING"
if (( ${#GPU_IDS[@]} != 4 )); then
  echo "ERROR: DEVICES must contain exactly four device ids, e.g. DEVICES='0 1 2 3'" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
SKIP_STPATH="${SKIP_STPATH:-0}"
SKIP_HELDOUT="${SKIP_HELDOUT:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_ROOT="${LOG_ROOT:-logs/audit_suite/$RUN_ID}"
mkdir -p "$LOG_ROOT"

# Force every launched training process to see exactly one GPU. Lightning then
# uses that one accelerator and does not spawn DDP copies that race on caches.
export PYTHONUNBUFFERED=1

run_command() {
  local gpu="$1" name="$2"; shift 2
  local log="$LOG_ROOT/${name}.log"
  printf '[%s] GPU %s -> %s\n' "$(date -u +%FT%TZ)" "$gpu" "$name"
  printf 'command:' > "$log"
  printf ' %q' "$@" >> "$log"
  printf '\n' >> "$log"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY RUN: CUDA_VISIBLE_DEVICES=%q' "$gpu" | tee -a "$log"
    printf ' %q' "$@" | tee -a "$log"
    printf '\n' | tee -a "$log"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$@" >> "$log" 2>&1
}

run_config() {
  local gpu="$1" config="$2"
  local name
  name="$(basename "$config" .yaml)"
  run_command "$gpu" "$name" "$PYTHON_BIN" -m src.training.train --config "$config"
}

run_autoencoder() {
  local gpu="$1" config="$2"
  local name
  name="$(basename "$config" .yaml)"
  run_command "$gpu" "$name" "$PYTHON_BIN" scripts/pretrain_expression_autoencoder.py --config "$config"
}

# run_wave NAME kind config...  kind is train or autoencoder.
run_wave() {
  local wave="$1" kind="$2"; shift 2
  local configs=("$@")
  local total="${#configs[@]}"
  (( total > 0 )) || return 0
  printf '\n===== %s (%d jobs) =====\n' "$wave" "$total"

  local offset=0 failed=0
  while (( offset < total )); do
    local pids=() names=()
    for slot in 0 1 2 3; do
      local idx=$((offset + slot))
      (( idx < total )) || break
      local cfg="${configs[$idx]}" gpu="${GPU_IDS[$slot]}"
      local name="$(basename "$cfg" .yaml)"
      if [[ "$kind" == "autoencoder" ]]; then
        run_autoencoder "$gpu" "$cfg" &
      else
        run_config "$gpu" "$cfg" &
      fi
      pids+=("$!")
      names+=("$name")
    done

    for i in "${!pids[@]}"; do
      if ! wait "${pids[$i]}"; then
        failed=1
        echo "FAILED: ${names[$i]} (tail follows)" >&2
        tail -80 "$LOG_ROOT/${names[$i]}.log" >&2 || true
      else
        echo "DONE: ${names[$i]}"
      fi
    done
    if (( failed )) && [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      echo "Stopping after failed wave '$wave'. Logs: $LOG_ROOT" >&2
      exit 1
    fi
    offset=$((offset + 4))
  done
  printf '===== completed %s =====\n' "$wave"
}

CFG=configs/audit_suite

# The residual-flow jobs depend on these validated/frozen autoencoders, so this
# wave always completes before any residual-flow process starts.
run_wave "wave-0-expression-autoencoders" autoencoder \
  "$CFG/pretrain_autoencoder_INT1.yaml" \
  "$CFG/pretrain_autoencoder_heldout_shared_panel.yaml"

run_wave "wave-1-nonparametric-and-set-summary-baselines" train \
  "$CFG/baseline_global_mean.yaml" \
  "$CFG/baseline_nearest.yaml" \
  "$CFG/baseline_local_mean.yaml" \
  "$CFG/baseline_idw.yaml" \
  "$CFG/baseline_harmonic.yaml" \
  "$CFG/baseline_learned_mean.yaml" \
  "$CFG/baseline_learned_sum.yaml"

controls=(
  "$CFG/control_fm_ot_builtin_mlp.yaml"
  "$CFG/control_fm_ot_stormlite_mlp.yaml"
  "$CFG/control_fm_ot_stpath_scratch_mlp_residual.yaml"
)
if [[ "$SKIP_STPATH" != "1" ]]; then
  controls+=("$CFG/control_fm_ot_stpath_frozen_backbone_custom_head.yaml")
fi
run_wave "wave-2-clean-controls" train "${controls[@]}"

run_wave "wave-3-deterministic-harmonic-residual" train \
  "$CFG/harmonic_residual_builtin_mlp.yaml" \
  "$CFG/harmonic_residual_gigapath_mlp.yaml" \
  "$CFG/harmonic_residual_stormlite_mlp.yaml" \
  "$CFG/harmonic_residual_stormlite_modality_dropout.yaml"

run_wave "wave-4-residual-flow-matching" train \
  "$CFG/residual_fm_ot_builtin_mlp.yaml" \
  "$CFG/residual_fm_ot_gigapath_mlp.yaml" \
  "$CFG/residual_fm_ot_stormlite_mlp.yaml" \
  "$CFG/residual_fm_ot_stormlite_modality_dropout.yaml"

run_wave "wave-5-matched-seeds" train \
  "$CFG/harmonic_residual_stormlite_seed1.yaml" \
  "$CFG/harmonic_residual_stormlite_seed2.yaml" \
  "$CFG/residual_fm_ot_stormlite_seed1.yaml" \
  "$CFG/residual_fm_ot_stormlite_seed2.yaml"

if [[ "$SKIP_HELDOUT" != "1" ]]; then
  run_wave "wave-6-heldout-samples" train \
    "$CFG/heldout_harmonic_residual_stormlite.yaml" \
    "$CFG/heldout_residual_fm_ot_stormlite.yaml"
fi

if [[ "$DRY_RUN" != "1" ]]; then
  "$PYTHON_BIN" scripts/collect_audit_results.py \
    --checkpoint-root results/checkpoints/audit_suite \
    --output-dir "reports/audit_suite/$RUN_ID" \
    --log-root "$LOG_ROOT"
fi

printf '\nAudit suite finished.\nLogs: %s\nCompact report: reports/audit_suite/%s\n' "$LOG_ROOT" "$RUN_ID"
