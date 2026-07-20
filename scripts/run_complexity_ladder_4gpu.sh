#!/usr/bin/env bash
# Run the post-audit complexity ladder in dependency-ordered stages.
# Despite the historical filename, DEVICES may contain any positive number
# of single-job GPU ids; use the 8-GPU wrapper for the current machine.
set -Eeuo pipefail

DEVICES_STRING="${DEVICES:-0 1 2 3}"
read -r -a GPU_IDS <<< "$DEVICES_STRING"
if (( ${#GPU_IDS[@]} < 1 )); then
  echo "ERROR: DEVICES must contain at least one GPU id" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
STAGE="${STAGE:-screen}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
FRESH="${FRESH:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_ROOT="${LOG_ROOT:-logs/complexity_ladder/${STAGE}_${RUN_ID}}"
CFG="configs/complexity_ladder"
mkdir -p "$LOG_ROOT"
export PYTHONUNBUFFERED=1
# Direct helper-script entry points must be able to import the sibling src
# package when the runner is launched from the repository root.
REPO_ROOT="$(pwd -P)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

checkpoint_dir_for_config() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg.get("training", {}).get("checkpoint_dir", f"results/checkpoints/{cfg['experiment_name']}"))
PY
}

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
  local name checkpoint
  name="$(basename "$config" .yaml)"
  checkpoint="$(checkpoint_dir_for_config "$config")"
  if [[ "$FRESH" != "1" && -f "$checkpoint/audit_test_metrics.json" ]]; then
    echo "SKIP completed: $name ($checkpoint/audit_test_metrics.json exists)"
    return 0
  fi
  local cmd=("$PYTHON_BIN" -m src.training.train --config "$config")
  if [[ "$SMOKE" == "1" ]]; then
    cmd+=(--override
      training.epochs=300
      training.unique_mask_count=64
      training.checkpoint_every_n_steps=0
      training.checkpoint_dir="results/checkpoints/complexity_ladder/smoke/$name"
      evaluation.training_mask_bank_path="results/mask_banks/training/complexity_ladder/smoke_${name}.json"
      validation.every_n_steps=100
      validation.patience_checks=2
      validation.early_stopping_min_steps=300
      validation.require_anchor_improvement=false
      evaluation.n_samples=2)
  fi
  run_command "$gpu" "$name" "${cmd[@]}"
}

run_autoencoder() {
  local gpu="$1" config="$2"
  local name checkpoint
  name="$(basename "$config" .yaml)"
  checkpoint="$(checkpoint_dir_for_config "$config")"
  if [[ "$FRESH" != "1" && -f "$checkpoint/best_autoencoder.pt" && "$SMOKE" != "1" ]]; then
    echo "SKIP completed: $name ($checkpoint/best_autoencoder.pt exists)"
    return 0
  fi
  local cmd=("$PYTHON_BIN" scripts/pretrain_expression_autoencoder.py --config "$config")
  if [[ "$SMOKE" == "1" ]]; then
    cmd+=(--override
      training.steps=300
      training.validate_every_n_steps=100
      training.patience_checks=2
      training.required_max_validation_rmse=999
      training.required_min_validation_pcc=-1
      training.checkpoint_dir=results/checkpoints/complexity_ladder/smoke/expression_autoencoder_latent32)
  fi
  run_command "$gpu" "$name" "${cmd[@]}"
}

run_precompute_novae() {
  local gpu="$1" config="$2"
  local name="precompute_context_novae"
  run_command "$gpu" "$name" "$PYTHON_BIN" scripts/precompute_context_novae_cache.py --config "$config"
}

run_wave() {
  local wave="$1" kind="$2"; shift 2
  local configs=("$@")
  (( ${#configs[@]} > 0 )) || return 0
  printf '\n===== %s (%d jobs) =====\n' "$wave" "${#configs[@]}"
  local offset=0
  while (( offset < ${#configs[@]} )); do
    local pids=() names=() failed=0
    for slot in "${!GPU_IDS[@]}"; do
      local idx=$((offset + slot))
      (( idx < ${#configs[@]} )) || break
      local config="${configs[$idx]}" gpu="${GPU_IDS[$slot]}"
      local name="$(basename "$config" .yaml)"
      if [[ "$kind" == "autoencoder" ]]; then
        run_autoencoder "$gpu" "$config" &
      else
        run_config "$gpu" "$config" &
      fi
      pids+=("$!")
      names+=("$name")
    done
    for i in "${!pids[@]}"; do
      if wait "${pids[$i]}"; then
        echo "DONE: ${names[$i]}"
      else
        failed=1
        echo "FAILED: ${names[$i]}" >&2
        tail -80 "$LOG_ROOT/${names[$i]}.log" >&2 || true
      fi
    done
    if (( failed )) && [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      echo "Stopping after failed wave '$wave'. Logs: $LOG_ROOT" >&2
      exit 1
    fi
    offset=$((offset + ${#GPU_IDS[@]}))
  done
}

run_baselines() {
  local A=configs/audit_suite
  run_wave "baselines" train \
    "$A/baseline_global_mean.yaml" \
    "$A/baseline_nearest.yaml" \
    "$A/baseline_local_mean.yaml" \
    "$A/baseline_idw.yaml" \
    "$A/baseline_harmonic.yaml" \
    "$A/baseline_learned_mean.yaml" \
    "$A/baseline_learned_sum.yaml"
}

run_deterministic() {
  run_wave "deterministic-backbone-and-fusion" train \
    "$CFG/01_hr_builtin_mlp.yaml" \
    "$CFG/02_hr_gigapath_mlp.yaml" \
    "$CFG/03_hr_stormlite_sum_mlp.yaml" \
    "$CFG/04_hr_stormlite_concat_mlp.yaml" \
    "$CFG/05_hr_stormlite_mome_mlp.yaml" \
    "$CFG/06_hr_stormlite_mome_no_bias_mlp.yaml" \
    "$CFG/07_hr_stormlite_mome_relpos_mlp.yaml" \
    "$CFG/08_hr_stormlite_gnn_mlp.yaml"
}

run_robustness() {
  run_wave "missing-image-training" train \
    "$CFG/09_hr_stormlite_concat_dropout.yaml" \
    "$CFG/10_hr_stormlite_mome_dropout.yaml"
}

run_generative() {
  run_wave "plain-flow-controls" train \
    "$CFG/11_fm_builtin_mlp.yaml" \
    "$CFG/12_fm_stormlite_concat_mlp.yaml" \
    "$CFG/13_fm_stormlite_mome_mlp.yaml"
  run_wave "residual-autoencoder" autoencoder \
    "$CFG/14_pretrain_residual_autoencoder_latent32.yaml"
  run_wave "residual-flow" train \
    "$CFG/15_residual_fm_builtin_mlp.yaml" \
    "$CFG/16_residual_fm_stormlite_concat_mlp.yaml" \
    "$CFG/17_residual_fm_stormlite_mome_mlp.yaml" \
    "$CFG/18_residual_fm_stormlite_mome_dropout.yaml"
}

run_novae() {
  if [[ "$SMOKE" == "1" ]]; then
    echo "ERROR: clean Novae is intentionally excluded from SMOKE; run STAGE=novae after smoke passes" >&2
    exit 2
  fi
  run_precompute_novae "${GPU_IDS[0]}" "$CFG/20_clean_novae_hr_novae.yaml"
  run_wave "matched-clean-novae" train \
    "$CFG/19_clean_novae_hr_mlp_matched.yaml" \
    "$CFG/20_clean_novae_hr_novae.yaml" \
    "$CFG/21_clean_novae_hr_both.yaml"
  run_wave "clean-flagship-capacity" train \
    "$CFG/22_clean_flagship_small.yaml" \
    "$CFG/23_clean_flagship_bigger.yaml"
}

config_uses_clean_novae() {
  "$PYTHON_BIN" - "$1" <<'PYCONF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print("1" if cfg.get("data", {}).get("novae_mode") == "context_only" else "0")
PYCONF
}

run_confirm() {
  mapfile -t configs < <(find "$CFG/confirm" -maxdepth 1 -name '*.yaml' -print | sort)
  if (( ${#configs[@]} == 0 )); then
    echo "ERROR: no promoted configs found. Run scripts/promote_complexity_winners.py first." >&2
    exit 2
  fi
  for config in "${configs[@]}"; do
    if [[ "$(config_uses_clean_novae "$config")" == "1" ]]; then
      run_precompute_novae "${GPU_IDS[0]}" "$config"
    fi
  done
  run_wave "three-seed-confirmation" train "${configs[@]}"
}

case "$STAGE" in
  smoke)
    SMOKE=1
    run_deterministic
    run_wave "smoke-plain-flow" train \
      "$CFG/11_fm_builtin_mlp.yaml" \
      "$CFG/12_fm_stormlite_concat_mlp.yaml" \
      "$CFG/13_fm_stormlite_mome_mlp.yaml"
    ;;
  baselines) run_baselines ;;
  deterministic) run_deterministic ;;
  robustness) run_robustness ;;
  generative) run_generative ;;
  novae) run_novae ;;
  confirm) run_confirm ;;
  heldout)
    if [[ ! -d "$CFG/heldout" ]]; then
      echo "ERROR: no held-out configs. Run scripts/promote_heldout_winners.py first." >&2
      exit 2
    fi
    mapfile -t heldout_configs < <(find "$CFG/heldout" -maxdepth 1 -name '*.yaml' -print | sort)
    if (( ${#heldout_configs[@]} == 0 )); then
      echo "ERROR: no held-out configs. Run scripts/promote_heldout_winners.py first." >&2
      exit 2
    fi
    if grep -q 'name: residual_fm_ot' "${heldout_configs[@]}"; then
      run_wave "heldout-residual-autoencoder" autoencoder \
        configs/audit_suite/pretrain_autoencoder_heldout_shared_panel.yaml
    fi
    run_wave "heldout-finalists" train "${heldout_configs[@]}"
    ;;
  screen)
    run_baselines
    run_deterministic
    run_robustness
    run_generative
    run_novae
    ;;
  *)
    echo "ERROR: STAGE must be smoke|baselines|deterministic|robustness|generative|novae|confirm|heldout|screen" >&2
    exit 2
    ;;
esac

if [[ "$DRY_RUN" != "1" && "$SMOKE" != "1" ]]; then
  "$PYTHON_BIN" scripts/collect_audit_results.py \
    --checkpoint-root results/checkpoints/complexity_ladder \
    --output-dir "reports/complexity_ladder/${STAGE}_${RUN_ID}" \
    --log-root "$LOG_ROOT"
fi

printf '\nFinished stage %s. Logs: %s\n' "$STAGE" "$LOG_ROOT"
