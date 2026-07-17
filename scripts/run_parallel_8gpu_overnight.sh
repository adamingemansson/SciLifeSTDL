#!/bin/bash
# Overnight batch (2026-07-17) — StormLite-focused. Real open question:
# STPath's from-scratch (unfrozen) bothresidual (PCC 0.3857 @ 40k) now
# beats StormLite's best arm (mome_both, 0.3461 @ 40k) even with NO
# pretraining advantage on either side — the gap at this comparison point
# is architectural, not about pretraining. Tonight tests the four most
# likely real levers: training length, capacity, position-bias choice,
# and data diversity (see docs/results_log.md's most recent entries for
# the full reasoning).
#
# 8 jobs, GPUs 0-7:
#   0. mome_both @ 80000 epochs (single-sample) — is the still-climbing
#      trend (0.28@10k -> 0.35@40k) going to catch up to STPath?
#   1. mome_both_bigger @ 40000 epochs (single-sample) — 4 layers/8 heads/
#      512-dim tokens instead of 2/4/256 (StormLite's capacity is much
#      smaller than STPath's real architecture)
#   2. mome_both_relpos @ 40000 epochs (single-sample) — relative_position
#      bias instead of frame_averaging, first real test at MoME scale
#      (every earlier bias_type sweep predates the MoME-FFN fix)
#   3-6. the 4 multi-sample configs (StormLite mome_both/mome_novae,
#      STPath bothresidual anchor, WAE-GAN novaeresidual), INT1-INT8,
#      40000 epochs
#   7. multisample_fm_ot_stormlite_mome_both_bigger — the flagship
#      "capacity + data diversity together" attempt, the actual best shot
#      at beating STPath tonight
#
# IMPORTANT: multi-sample configs (3-7) go through train.py's OWN main()
# directly, NOT run_comparison.py (which explicitly doesn't support
# cfg.data.sample_ids — see _train_model's own docstring note). Their
# completion is detected differently: "mean PCC:" in the log, not the
# "^model " table row run_comparison.py prints for the single-sample jobs.
#
# sample_ids in the multi-sample configs assume INT1-INT8 are downloaded
# locally — check `ls data/raw/hest1k/` first; trim those configs'
# sample_ids lists if fewer are actually present (a missing sample raises
# a clear FileNotFoundError naming it, not a silent skip).
#
# All 8 jobs write a checkpoint every 10000 steps
# (training.checkpoint_every_n_steps, PeriodicCheckpointCallback,
# train.py) — safe to kill any job partway through and still evaluate its
# last checkpoint (run_comparison.py --skip-training for jobs 0-2,
# load_trained_model directly for the multi-sample jobs 3-7, since
# run_comparison.py doesn't support cfg.data.sample_ids at all).
#
# Usage: bash scripts/run_parallel_8gpu_overnight.sh
# Logs: logs/parallel_run_overnight/<name>.log

set -u

SINGLE_CONFIGS=(
    "configs/exp_hest1k_fm_ot_stormlite_mome_both.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_mome_both_bigger.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_mome_both_relpos.yaml"
)
SINGLE_EXTRA_OVERRIDES=(
    "training.epochs=80000 training.checkpoint_every_n_steps=10000"
    ""
    ""
)
MULTI_CONFIGS=(
    "configs/exp_multisample_fm_ot_stormlite_mome_both.yaml"
    "configs/exp_multisample_fm_ot_stormlite_mome_novae.yaml"
    "configs/exp_multisample_fm_ot_stpath_bothresidual.yaml"
    "configs/exp_multisample_wae_gan_stpath_novaeresidual.yaml"
    "configs/exp_multisample_fm_ot_stormlite_mome_both_bigger.yaml"
)

LOG_DIR="logs/parallel_run_overnight"
EXTRA_ARGS="--shuffle-diagnostic"

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
    name=$(basename "$cfg" .yaml)
    logfile="$LOG_DIR/${name}.log"
    if [ -f "$logfile" ] && grep -q "^model " "$logfile"; then
        echo "  GPU $gpu: $cfg -> SKIPPING, already completed (see $logfile)"
        gpu=$((gpu + 1))
        continue
    fi
    echo "  GPU $gpu: $cfg (single-sample, run_comparison.py) -> $logfile"
    CUDA_VISIBLE_DEVICES=$gpu \
    OMP_NUM_THREADS=$THREADS_PER_JOB \
    MKL_NUM_THREADS=$THREADS_PER_JOB \
    python -m src.evaluation.run_comparison "$cfg" \
        --override ${extra} \
        ${EXTRA_ARGS} \
        > "$logfile" 2>&1 &
    gpu=$((gpu + 1))
done

for cfg in "${MULTI_CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    logfile="$LOG_DIR/${name}.log"
    if [ -f "$logfile" ] && grep -q "^mean PCC:" "$logfile"; then
        echo "  GPU $gpu: $cfg -> SKIPPING, already completed (see $logfile)"
        gpu=$((gpu + 1))
        continue
    fi
    echo "  GPU $gpu: $cfg (multi-sample, train.py directly) -> $logfile"
    CUDA_VISIBLE_DEVICES=$gpu \
    OMP_NUM_THREADS=$THREADS_PER_JOB \
    MKL_NUM_THREADS=$THREADS_PER_JOB \
    python -m src.training.train --config "$cfg" \
        > "$logfile" 2>&1 &
    gpu=$((gpu + 1))
done

n_launched=$(jobs -p | wc -l)
echo "$n_launched job(s) launched (PIDs: $(jobs -p | tr '\n' ' ')), $((TOTAL_JOBS - n_launched)) skipped as already-completed. Waiting for completion..."
wait
echo "All jobs finished. Check $LOG_DIR/*.log for each run's output."
echo ""
echo "Quick summary — single-sample (real result only, grep -m1, not the shuffle-diagnostic row):"
for cfg in "${SINGLE_CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    echo "--- $name ---"
    grep -m1 -A2 "^model " "$LOG_DIR/${name}.log" | tail -2
done
echo ""
echo "Quick summary — multi-sample:"
for cfg in "${MULTI_CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    echo "--- $name ---"
    grep -A1 "^mean PCC:" "$LOG_DIR/${name}.log"
done
