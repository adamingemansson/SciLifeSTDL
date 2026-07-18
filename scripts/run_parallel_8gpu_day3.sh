#!/bin/bash
# Day 3 batch (2026-07-19) — follow-up to day2 (see docs/results_log.md).
# Two threads driving this batch:
#
#   1. The user's direct request: a real, fair, no-pretraining-advantage
#      test of StormLite's own architecture vs. STPath's own architecture
#      (StormLite's flagship decoder held constant, STPath with
#      stpath_pretrained: false — see exp_hest1k_fm_ot_stpath_unfrozen_
#      bothresidual_paneldecoder_add.yaml's own header for the full
#      reasoning).
#   2. Two day2 findings need more data before they mean anything
#      definitive: (a) StormLite+decoder's 5 seeds spanned PCC 0.366-0.493
#      — is that reducible with EMA, and (b) STPath(pretrained)+decoder
#      only has 2 seeds (0.4874, 0.5061) vs. StormLite's 5 — an unfair
#      sample-size comparison. This batch adds more seeds on BOTH sides
#      and tests whether EMA (added today, src/training/train.py
#      EMACallback) actually reduces the seed-to-seed spread it was
#      built to address.
#
# 8 jobs, GPUs 0-7:
#   0. STPath UNFROZEN + winning decoder (single, 40k, NEW) — the direct
#      request: real architecture-only comparison, no pretraining
#      advantage on either side, same decoder as StormLite's flagship.
#   1. StormLite "bigger" capacity + winning decoder, WITH the actual fix
#      (lower lr + warmup_steps + EMA — not more clipping, which already
#      failed twice) — does this resolve the mode-collapse?
#   2-3. StormLite + winning decoder, seed=0 and seed=5, BOTH with EMA
#      added — direct test of whether EMA reduces the 0.366-0.493 spread
#      seen without it (seed=0 here is a re-run of the SAME seed that
#      gave 0.4933 without EMA, so it's directly comparable).
#   4-5. STPath (pretrained) + winning decoder, seed=2 and seed=3 — more
#      seed data toward a real seed-averaged StormLite-vs-STPath
#      comparison (was n=2, now n=4).
#   6. multisample StormLite + winning decoder, WITH EMA (40k, not 80k —
#      80k regressed yesterday) — does EMA help the multi-sample number
#      the way it might help single-sample?
#   7. multisample STPath bothresidual (pretrained, OWN dense decoder,
#      NOT the panel_invariant swap — that already hurt STPath's
#      multi-sample number yesterday) + EMA — does EMA help STPath's own
#      best multi-sample arm (0.1887 without it)?
#
# IMPORTANT: job 6-7 are multi-sample (cfg.data.sample_ids) and go through
# train.py's OWN main() directly, NOT run_comparison.py — same
# reasoning/completion-marker distinction as every earlier parallel script.
#
# sample_ids in the multi-sample configs assume INT1-INT8 are downloaded
# locally — same caveat as every earlier script.
#
# Usage: bash scripts/run_parallel_8gpu_day3.sh
# Smoke test: SMOKETEST=1 bash scripts/run_parallel_8gpu_day3.sh
# Logs: logs/parallel_run_day3/<name>.log

set -u

SMOKETEST="${SMOKETEST:-0}"

SINGLE_CONFIGS=(
    "configs/exp_hest1k_fm_ot_stpath_unfrozen_bothresidual_paneldecoder_add.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_mome_both_bigger_paneldecoder_add_warmupema.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder_add.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder_add.yaml"
    "configs/exp_hest1k_fm_ot_stpath_bothresidual_paneldecoder_add.yaml"
    "configs/exp_hest1k_fm_ot_stpath_bothresidual_paneldecoder_add.yaml"
)
SINGLE_EXTRA_OVERRIDES=(
    ""
    ""
    "training.epochs=40000 training.seed=0 training.checkpoint_every_n_steps=10000 training.log_print_every_n_steps=1000 training.ema_decay=0.999"
    "training.epochs=40000 training.seed=5 training.checkpoint_every_n_steps=10000 training.log_print_every_n_steps=1000 training.ema_decay=0.999"
    "training.seed=2"
    "training.seed=3"
)
SINGLE_NAMES=(
    "stpath_unfrozen_bothresidual_paneldecoder_add"
    "stormlite_mome_both_bigger_paneldecoder_add_warmupema"
    "stormlite_mome_both_paneldecoder_add_seed0_ema"
    "stormlite_mome_both_paneldecoder_add_seed5_ema"
    "stpath_bothresidual_paneldecoder_add_seed2"
    "stpath_bothresidual_paneldecoder_add_seed3"
)

MULTI_CONFIGS=(
    "configs/exp_multisample_fm_ot_stormlite_mome_both_paneldecoder_add.yaml"
    "configs/exp_multisample_fm_ot_stpath_bothresidual.yaml"
)
MULTI_EXTRA_OVERRIDES=(
    "training.epochs=40000 training.checkpoint_every_n_steps=10000 training.log_print_every_n_steps=1000 training.ema_decay=0.999"
    "training.epochs=40000 training.checkpoint_every_n_steps=10000 training.log_print_every_n_steps=1000 training.ema_decay=0.999"
)
MULTI_NAMES=(
    "multisample_stormlite_mome_both_paneldecoder_add_ema"
    "multisample_stpath_bothresidual_ema"
)

LOG_DIR="logs/parallel_run_day3"
EXTRA_ARGS="--shuffle-diagnostic"
SMOKE_OVERRIDE=""

if [ "$SMOKETEST" = "1" ]; then
    LOG_DIR="logs/parallel_run_day3_smoketest"
    SMOKE_OVERRIDE="training.epochs=10 training.checkpoint_every_n_steps=5 training.log_print_every_n_steps=3"
    echo "*** SMOKETEST=1 -- running tiny versions of all 8 jobs (epochs=10) to catch config/crash issues before the real run. Logs: $LOG_DIR ***"
fi

N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
TOTAL_JOBS=$((${#SINGLE_CONFIGS[@]} + ${#MULTI_CONFIGS[@]}))
N_CORES=$(nproc)
THREADS_PER_JOB=$((N_CORES / TOTAL_JOBS))
[ "$THREADS_PER_JOB" -lt 1 ] && THREADS_PER_JOB=1

if [ "$TOTAL_JOBS" -gt "$N_GPUS" ]; then
    echo "ERROR: $TOTAL_JOBS configs but only $N_GPUS GPUs visible — trim CONFIGS or this will double up on a GPU."
    exit 1
fi

mkdir -p "$LOG_DIR"
echo "Launching $TOTAL_JOBS jobs across GPUs 0-$((TOTAL_JOBS - 1)), ${THREADS_PER_JOB} CPU threads each, logs in $LOG_DIR..."

gpu=0
for i in "${!SINGLE_CONFIGS[@]}"; do
    cfg="${SINGLE_CONFIGS[$i]}"
    extra="${SINGLE_EXTRA_OVERRIDES[$i]}"
    name="${SINGLE_NAMES[$i]}"
    logfile="$LOG_DIR/${name}.log"
    if [ -f "$logfile" ] && grep -q "^model " "$logfile"; then
        echo "  GPU $gpu: $cfg ($name) -> SKIPPING, already completed (see $logfile)"
        gpu=$((gpu + 1))
        continue
    fi
    echo "  GPU $gpu: $cfg ($name, single-sample, run_comparison.py) -> $logfile"
    CUDA_VISIBLE_DEVICES=$gpu \
    OMP_NUM_THREADS=$THREADS_PER_JOB \
    MKL_NUM_THREADS=$THREADS_PER_JOB \
    python -m src.evaluation.run_comparison "$cfg" \
        --override ${extra} ${SMOKE_OVERRIDE} \
        ${EXTRA_ARGS} \
        > "$logfile" 2>&1 &
    gpu=$((gpu + 1))
done

for i in "${!MULTI_CONFIGS[@]}"; do
    cfg="${MULTI_CONFIGS[$i]}"
    extra="${MULTI_EXTRA_OVERRIDES[$i]}"
    name="${MULTI_NAMES[$i]}"
    logfile="$LOG_DIR/${name}.log"
    if [ -f "$logfile" ] && grep -q "^mean PCC:" "$logfile"; then
        echo "  GPU $gpu: $cfg ($name) -> SKIPPING, already completed (see $logfile)"
        gpu=$((gpu + 1))
        continue
    fi
    echo "  GPU $gpu: $cfg ($name, multi-sample, train.py directly) -> $logfile"
    CUDA_VISIBLE_DEVICES=$gpu \
    OMP_NUM_THREADS=$THREADS_PER_JOB \
    MKL_NUM_THREADS=$THREADS_PER_JOB \
    python -m src.training.train --config "$cfg" \
        --override ${extra} ${SMOKE_OVERRIDE} \
        > "$logfile" 2>&1 &
    gpu=$((gpu + 1))
done

n_launched=$(jobs -p | wc -l)
echo "$n_launched job(s) launched (PIDs: $(jobs -p | tr '\n' ' ')), $((TOTAL_JOBS - n_launched)) skipped as already-completed. Waiting for completion..."
wait
echo "All jobs finished. Check $LOG_DIR/*.log for each run's output."
echo ""
echo "Quick summary — single-sample (real result only, grep -m1, not the shuffle-diagnostic row):"
for name in "${SINGLE_NAMES[@]}"; do
    echo "--- $name ---"
    grep -m1 -A2 "^model " "$LOG_DIR/${name}.log" | tail -2
done
echo ""
echo "Quick summary — multi-sample:"
for name in "${MULTI_NAMES[@]}"; do
    echo "--- $name ---"
    grep -A1 "^mean PCC:" "$LOG_DIR/${name}.log"
done
