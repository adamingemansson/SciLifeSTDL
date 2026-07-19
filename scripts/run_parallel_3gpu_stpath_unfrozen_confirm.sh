#!/bin/bash
# 3-config confirmation batch (2026-07-19) — the "full retrain" STPath
# comparison, requested directly. Every STPath number so far either used
# PRETRAINED weights (a real external-data advantage, ~4-seed mean 0.503)
# or the OLD dense decoder (unfrozen/bothresidual, PCC 0.3857 @ 40k,
# single seed, no EMA) — this batch is the one comparison that's never
# actually been run: STPath UNFROZEN (stpath_pretrained: false, fully
# from-scratch, same as "full retrain") WITH the winning decoder
# (panel_invariant/add) that both StormLite and pretrained-STPath already
# use, seed-averaged, +EMA. Isolates architecture from pretraining on both
# sides at once.
#
# Config (configs/exp_hest1k_fm_ot_stpath_unfrozen_bothresidual_paneldecoder_add.yaml)
# was written 2026-07-19 but never actually run until now.
#
# Comparison targets:
#   StormLite (small) + decoder, 8-seed mean (EMA+non-EMA) ~0.462
#   STPath (pretrained) + decoder, 4-seed mean ~0.503
#   STPath (unfrozen) + OLD dense decoder, 1 seed, no EMA: 0.3857
#
# Deliberately small: 3 GPUs, 3 seeds, no pairing. Each job gets its own
# checkpoint_dir (checkpoint-collision-bug fix pattern).
#
# Usage: bash scripts/run_parallel_3gpu_stpath_unfrozen_confirm.sh
# Smoke test: SMOKETEST=1 bash scripts/run_parallel_3gpu_stpath_unfrozen_confirm.sh
# Logs: logs/parallel_run_stpath_unfrozen_confirm/<name>.log

set -u

SMOKETEST="${SMOKETEST:-0}"

CFG="configs/exp_hest1k_fm_ot_stpath_unfrozen_bothresidual_paneldecoder_add.yaml"

SEEDS=(10 11 12)
JOB_NAME=(
    "stpath_bothresidual_paneldecoder_unfrozen_ema_seed10"
    "stpath_bothresidual_paneldecoder_unfrozen_ema_seed11"
    "stpath_bothresidual_paneldecoder_unfrozen_ema_seed12"
)

LOG_DIR="logs/parallel_run_stpath_unfrozen_confirm"
SMOKE_OVERRIDE=""
if [ "$SMOKETEST" = "1" ]; then
    LOG_DIR="logs/parallel_run_stpath_unfrozen_confirm_smoketest"
    SMOKE_OVERRIDE="training.epochs=10 training.checkpoint_every_n_steps=5 training.log_print_every_n_steps=3"
    echo "*** SMOKETEST=1 -- tiny versions of all 3 jobs (epochs=10). Logs: $LOG_DIR ***"
fi
mkdir -p "$LOG_DIR"

N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
N_JOBS=${#SEEDS[@]}
if [ "$N_JOBS" -gt "$N_GPUS" ]; then
    echo "ERROR: need $N_JOBS GPUs (1 job each) but only $N_GPUS visible."
    exit 1
fi
N_CORES=$(nproc)
THREADS_PER_JOB=$(( N_CORES / N_JOBS ))
[ "$THREADS_PER_JOB" -lt 1 ] && THREADS_PER_JOB=1

echo "Launching $N_JOBS jobs (1 per GPU, GPUs 0-$((N_JOBS-1))), ${THREADS_PER_JOB} CPU threads each, logs in $LOG_DIR..."

run_one() {
    local gpu="$1" idx="$2"
    local seed="${SEEDS[$idx]}" name="${JOB_NAME[$idx]}"
    local logfile="$LOG_DIR/${name}.log"
    if [ -f "$logfile" ] && grep -q "^model " "$logfile"; then
        echo "  GPU $gpu: [$name] SKIP (already completed)"
        return
    fi
    echo "  GPU $gpu: [$name] START"
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=$THREADS_PER_JOB MKL_NUM_THREADS=$THREADS_PER_JOB \
    python -m src.evaluation.run_comparison "$CFG" \
        --override training.seed=${seed} \
                    training.ema_decay=0.999 \
                    training.checkpoint_dir=results/checkpoints/${name} \
                    training.checkpoint_every_n_steps=10000 \
                    training.log_print_every_n_steps=1000 \
                    ${SMOKE_OVERRIDE} \
        --shuffle-diagnostic > "$logfile" 2>&1
    echo "  GPU $gpu: [$name] DONE"
}

for i in "${!SEEDS[@]}"; do
    run_one "$i" "$i" &
done
wait

echo ""
echo "=== All 3 jobs finished. Results: ==="
for name in "${JOB_NAME[@]}"; do
    echo "--- $name ---"
    grep -m1 -A2 "^model " "$LOG_DIR/${name}.log" 2>/dev/null | tail -2
done

echo ""
echo "Compare the 3-seed mean above against:"
echo "  StormLite (small) + decoder, 8-seed mean (EMA+non-EMA): ~0.462"
echo "  STPath (pretrained) + decoder, 4-seed mean: ~0.503"
echo "  STPath (unfrozen) + OLD dense decoder, 1 seed, no EMA: 0.3857"
echo "  If this mean lands well below both of the above, the earlier"
echo "  'StormLite beats unfrozen STPath' finding holds even with the"
echo "  decoder swap. If it lands close to StormLite's mean, the decoder"
echo "  swap helps STPath's architecture too, not just StormLite's."
