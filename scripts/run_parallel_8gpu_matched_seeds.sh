#!/bin/bash
# Matched-seed-count confirmation batch (2026-07-19) -- the deliberately
# SIMPLE next step after the tkdgx1 8-config batch corrected two earlier
# narratives (see docs/results_log.md): StormLite-small (flagship) has an
# 8-seed mean (~0.462), but STPath-unfrozen (full retrain, same decoder)
# only had 2 seeds behind its 0.464 mean when we called it a "tie" --
# that comparison rests on a much thinner STPath sample. No new
# architecture, no new code -- just running configs we already have
# (the plain, non-stacked versions of each -- no logit-normal, no
# no-log1p, no AdaLN velocity net), more times, to firm up the numbers.
#
# 8 jobs, all single-sample, 40k epochs, all parallel on 8 GPUs:
#   3x STPath-unfrozen (plain), seeds 12-14 -- brings it from 2 seeds
#       (10, 11) to 5 total.
#   3x StormLite-small (plain flagship), seeds 12-14 -- adds to the
#       already-solid 8-seed base, for 11 total.
#   2x StormLite-bigger + QK-norm + warmup (the ONE confirmed-working
#       "bigger" recipe, nothing else stacked on top), seeds 13-14 --
#       adds to the existing 3 seeds (seed10=0.5079, seed11=0.1877,
#       seed12=0.5019 -- see docs/results_log.md), for 5 total.
#
# Usage: bash scripts/run_parallel_8gpu_matched_seeds.sh
# Smoke test: SMOKETEST=1 bash scripts/run_parallel_8gpu_matched_seeds.sh
# Logs: logs/parallel_run_matched_seeds/<name>.log

set -u

SMOKETEST="${SMOKETEST:-0}"
SMOKETEST_EPOCHS="${SMOKETEST_EPOCHS:-2}"

STPATH_CFG="configs/exp_hest1k_fm_ot_stpath_unfrozen_bothresidual_paneldecoder_add.yaml"
FLAG_CFG="configs/exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder_add.yaml"
BIGGER_CFG="configs/exp_hest1k_fm_ot_stormlite_mome_both_bigger_paneldecoder_add_warmupema.yaml"

CONFIGS=(
    "$STPATH_CFG" "$STPATH_CFG" "$STPATH_CFG"
    "$FLAG_CFG" "$FLAG_CFG" "$FLAG_CFG"
    "$BIGGER_CFG" "$BIGGER_CFG"
)
NAMES=(
    "stpath_bothresidual_paneldecoder_unfrozen_ema_seed12"
    "stpath_bothresidual_paneldecoder_unfrozen_ema_seed13"
    "stpath_bothresidual_paneldecoder_unfrozen_ema_seed14"
    "stormlite_mome_both_paneldecoder_ema_seed12"
    "stormlite_mome_both_paneldecoder_ema_seed13"
    "stormlite_mome_both_paneldecoder_ema_seed14"
    "stormlite_mome_both_paneldecoder_bigger_qknorm_warmup_ema_seed13"
    "stormlite_mome_both_paneldecoder_bigger_qknorm_warmup_ema_seed14"
)
SEEDS=(12 13 14 12 13 14 13 14)
EXTRA_OVERRIDE=("" "" "" "" "" "" "model.params.storm_lite_qk_norm=true" "model.params.storm_lite_qk_norm=true")
EPOCHS=(40000 40000 40000 40000 40000 40000 80000 80000)

LOG_DIR="logs/parallel_run_matched_seeds"
EXTRA_ARGS="--shuffle-diagnostic"
SMOKE_OVERRIDE=""
if [ "$SMOKETEST" = "1" ]; then
    LOG_DIR="logs/parallel_run_matched_seeds_smoketest"
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

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    name="${NAMES[$i]}"
    seed="${SEEDS[$i]}"
    logfile="$LOG_DIR/${name}.log"
    if [ -f "$logfile" ] && grep -q "^model " "$logfile"; then
        echo "  GPU $i: [$name] SKIP (already completed)"
        continue
    fi
    echo "  GPU $i: [$name] START"
    extra="${EXTRA_OVERRIDE[$i]}"
    epoch_override=""
    [ "$SMOKETEST" != "1" ] && epoch_override="training.epochs=${EPOCHS[$i]}"
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
echo "  StormLite (small) + decoder, 8-seed mean (before this batch): ~0.462"
echo "  STPath (unfrozen) + decoder, 2-seed mean (before this batch): ~0.464 (seeds 10=0.4125, 11=0.5160)"
echo "  StormLite bigger+QK-norm+warmup, 3-seed mean (before this batch): ~0.399 (seeds 10=0.5079, 11=0.1877, 12=0.5019)"
echo "  Combine the 3 stpath_unfrozen results above with seeds 10/11 for a real 5-seed mean."
echo "  Combine the 3 stormlite(small) results above with the existing 8 seeds for an 11-seed mean."
echo "  Combine the 2 stormlite(bigger) results above with the existing 3 seeds for a 5-seed mean."
