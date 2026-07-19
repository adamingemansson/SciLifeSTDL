#!/bin/bash
# 8-config confirmation batch for tkdgx1 (2026-07-19) -- pivoted back from
# st-a100 (shared node, Markus + another labmate's "STHLM" jobs both
# landed on our GPUs mid-run, no longer worth waiting on there). Picks up
# the most important open questions with real GPU-hours, not stacked with
# lower-priority exploratory items -- "mainly StormLite and unfrozen
# STPath, max 2 seeds per model" per direct instruction.
#
# *** MEMORY WARNING ***: the "bigger" StormLite config's real memory
# usage GREW during actual training on the 80GB st-a100 card -- from
# ~33.8GB to ~40.2GB on GPU 0, ~32.2GB to ~38.6GB on GPU 6 (observed via
# nvidia-smi mid-run, not just at start). That's already AT or ABOVE the
# 33GB/40GB safe ceiling established for this project's 40GB cards, with
# growth still happening -- a real OOM risk here, not just theoretical.
# Watch `nvidia-smi` during this run; if any "bigger" job OOMs, that's
# confirmed, not a fluke -- kill it rather than letting it take down a
# labmate's job sharing the same GPU.
#
# Already have ONE real completed seed for bigger_qknorm_warmup (seed11,
# PCC 0.1877, RMSE 0.2258, AUC 0.8947, ST-FID 0.7956 -- real, not
# collapsed, just a weaker seed than earlier confounded estimates
# suggested) -- see docs/results_log.md. This batch adds 2 MORE seeds
# (10, 12) to that, capped at 2 new per instruction, for a 3-seed mean
# once combined.
#
# 8 jobs, GPUs 0-7:
#   0-1. bigger_qknorm_warmup, seeds 10/12 (80k) -- 2 NEW seeds (+ the
#        already-completed seed11 = 3-seed mean once combined). The core
#        "does bigger capacity + QK-norm reliably close the gap to
#        STPath" question -- see docs/results_log.md's 16-config-batch
#        entry for why this needs an unconfounded multi-seed mean.
#   2-3. stpath_unfrozen (full retrain, no pretrained weights), seeds
#        10/11 (40k) -- 2 NEW seeds. The "full retrain" STPath comparison
#        this project has never actually run for real before tonight.
#   4.   bigger_qknormonly_flatlr, seed10 (40k) -- diagnostic: isolates
#        QK-norm from the warmup/lower-LR change it's always been
#        bundled with (see registry.py's velocity_net docstring... no,
#        see the BIGGER config's own header -- this is the QK-norm-vs-
#        warmup 2x2 factorial, job 7 from the original 14-job plan).
#   5.   bigger_warmuponly_noqknorm, seed10 (40k) -- the mirror
#        diagnostic: warmup+lower-LR alone, no QK-norm. Completes the 2x2.
#   6-7. AdaLN-residual velocity_net (velocity_net_type="adaln_residual"),
#        seeds 10/11 (40k) -- the architecture-audit finding from
#        docs/results_log.md (velocity_net was >100x smaller than the
#        context_encoder feeding it, no residual connections, one-time
#        input concatenation only for conditioning). Isolated cleanly
#        against the flagship (small) StormLite, not stacked with
#        anything else -- a third StormLite variant, fits "mainly
#        StormLite" without violating the "max 2 seeds per model" cap on
#        the two PRIMARY models above (this is a genuinely different
#        model, not a 3rd seed of either).
#
# Usage: bash scripts/run_parallel_8gpu_tkdgx1_confirm.sh
# Smoke test: SMOKETEST=1 bash scripts/run_parallel_8gpu_tkdgx1_confirm.sh
# Logs: logs/parallel_run_tkdgx1_confirm/<name>.log

set -u

SMOKETEST="${SMOKETEST:-0}"
SMOKETEST_EPOCHS="${SMOKETEST_EPOCHS:-2}"

BIGGER_CFG="configs/exp_hest1k_fm_ot_stormlite_mome_both_bigger_paneldecoder_add_warmupema.yaml"
STPATH_CFG="configs/exp_hest1k_fm_ot_stpath_unfrozen_bothresidual_paneldecoder_add.yaml"
FLAG_CFG="configs/exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder_add.yaml"

CONFIGS=(
    "$BIGGER_CFG" "$BIGGER_CFG"
    "$STPATH_CFG" "$STPATH_CFG"
    "$BIGGER_CFG" "$BIGGER_CFG"
    "$FLAG_CFG" "$FLAG_CFG"
)
NAMES=(
    "stormlite_mome_both_paneldecoder_bigger_qknorm_warmup_ema_seed10"
    "stormlite_mome_both_paneldecoder_bigger_qknorm_warmup_ema_seed12"
    "stpath_bothresidual_paneldecoder_unfrozen_ema_seed10"
    "stpath_bothresidual_paneldecoder_unfrozen_ema_seed11"
    "stormlite_mome_both_paneldecoder_bigger_qknormonly_flatlr_ema_seed10"
    "stormlite_mome_both_paneldecoder_bigger_warmuponly_noqknorm_ema_seed10"
    "stormlite_mome_both_paneldecoder_adalnvelocity_ema_seed10"
    "stormlite_mome_both_paneldecoder_adalnvelocity_ema_seed11"
)
SEEDS=(10 12 10 11 10 10 10 11)
EPOCHS=(80000 80000 40000 40000 40000 40000 40000 40000)
EXTRA_OVERRIDE=(
    "model.params.storm_lite_qk_norm=true"
    "model.params.storm_lite_qk_norm=true"
    ""
    ""
    "model.params.storm_lite_qk_norm=true model.params.warmup_steps=0 model.params.lr=1.0e-3"
    "model.params.storm_lite_qk_norm=false"
    "model.params.velocity_net_type=adaln_residual model.params.velocity_net_n_layers=3"
    "model.params.velocity_net_type=adaln_residual model.params.velocity_net_n_layers=3"
)

LOG_DIR="logs/parallel_run_tkdgx1_confirm"
EXTRA_ARGS="--shuffle-diagnostic"
SMOKE_OVERRIDE=""
if [ "$SMOKETEST" = "1" ]; then
    LOG_DIR="logs/parallel_run_tkdgx1_confirm_smoketest"
    SMOKE_OVERRIDE="training.epochs=${SMOKETEST_EPOCHS} training.checkpoint_every_n_steps=1 training.log_print_every_n_steps=1"
    echo "*** SMOKETEST=1 -- tiny versions of all 8 jobs (epochs=${SMOKETEST_EPOCHS}). Logs: $LOG_DIR ***"
fi
mkdir -p "$LOG_DIR"

N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
TOTAL_JOBS=${#CONFIGS[@]}
if [ "$TOTAL_JOBS" -gt "$N_GPUS" ]; then
    echo "ERROR: $TOTAL_JOBS configs but only $N_GPUS GPUs visible."
    exit 1
fi
N_CORES=$(nproc)
THREADS_PER_JOB=$((N_CORES / TOTAL_JOBS))
[ "$THREADS_PER_JOB" -lt 1 ] && THREADS_PER_JOB=1

echo "Launching $TOTAL_JOBS jobs across GPUs 0-$((TOTAL_JOBS - 1)), ${THREADS_PER_JOB} CPU threads each, logs in $LOG_DIR..."
echo "*** WATCH nvidia-smi for OOM on the 'bigger' jobs (0,1,4,5) -- see header, real memory growth risk on 40GB cards ***"

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    name="${NAMES[$i]}"
    seed="${SEEDS[$i]}"
    epochs="${EPOCHS[$i]}"
    extra="${EXTRA_OVERRIDE[$i]}"
    logfile="$LOG_DIR/${name}.log"
    if [ -f "$logfile" ] && grep -q "^model " "$logfile"; then
        echo "  GPU $i: [$name] SKIP (already completed)"
        continue
    fi
    echo "  GPU $i: [$name] START"
    epoch_override=""
    [ "$SMOKETEST" != "1" ] && epoch_override="training.epochs=${epochs}"
    CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=$THREADS_PER_JOB MKL_NUM_THREADS=$THREADS_PER_JOB \
    python -m src.evaluation.run_comparison "$cfg" \
        --override ${extra} \
                    training.seed=${seed} \
                    training.ema_decay=0.999 \
                    ${epoch_override} \
                    training.checkpoint_dir=results/checkpoints/${name} \
                    training.checkpoint_every_n_steps=10000 \
                    training.log_print_every_n_steps=1000 \
                    ${SMOKE_OVERRIDE} \
        ${EXTRA_ARGS} > "$logfile" 2>&1 &
done

wait
echo ""
echo "=== All jobs finished. Results: ==="
for name in "${NAMES[@]}"; do
    echo "--- $name ---"
    grep -m1 -A2 "^model " "$LOG_DIR/${name}.log" 2>/dev/null | tail -2
done

echo ""
echo "Compare against:"
echo "  StormLite (small) + decoder, 8-seed mean (EMA+non-EMA): ~0.462"
echo "  STPath (pretrained) + decoder, 4-seed mean: ~0.503"
echo "  STPath (unfrozen) + OLD dense decoder, 1 seed, no EMA: 0.3857"
echo "  bigger_qknorm_warmup seed11 (already completed, st-a100): PCC 0.1877"
echo "  Combine jobs 0,1 above with seed11 for a real 3-seed bigger+QK-norm mean."
