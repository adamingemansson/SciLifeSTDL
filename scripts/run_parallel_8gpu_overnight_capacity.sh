#!/bin/bash
# Overnight capacity/variance batch (2026-07-19) -- combines resolving
# the one big remaining statistical uncertainty (StormLite-bigger's seed
# variance) with a genuinely NEW, isolated test of the original
# architecture-audit finding: velocity_net is >100x smaller than the
# context_encoder feeding it (see docs/results_log.md). The AdaLN
# residual change tested that at similar WIDTH (hidden_dim unchanged) --
# this batch separately tests raw CAPACITY (hidden_dim doubled, still the
# plain "mlp" architecture) as its own isolated lever, not stacked with
# AdaLN or anything else. Zero new code -- hidden_dim only feeds
# velocity_net in FlowMatchingOT (encoder/decoder use the separate
# ae_hidden_dim), confirmed via grep before writing this.
#
# 8 jobs, all single-sample, all parallel on 8 GPUs:
#   0-2. StormLite bigger+QK-norm+warmup, seeds 15/16/17 (80k) -- 3 more
#        seeds, bringing this arm from 5 to 8. Historically the noisiest
#        arm (range 0.19-0.51 over 5 seeds) but also the highest mean in
#        the most recent same-batch comparison (0.498, 2 seeds) --
#        resolving whether that's signal or luck.
#   3-4. AdaLN-residual velocity_net, seeds 12/13 (40k) -- 2 more seeds,
#        bringing this arm from 2 to 4. Two seeds (mean 0.29, one with
#        elevated ST-FID) isn't enough to conclude the architecture
#        change doesn't help.
#   5-6. Bigger velocity_net CAPACITY (plain "mlp" type unchanged,
#        hidden_dim doubled 512->1024), seeds 10/11 (40k) -- NEW arm,
#        isolated from the AdaLN architecture question. Directly tests
#        "does the flow-matching core need more raw capacity" without
#        also changing HOW conditioning is injected.
#   7.   STPath pretrained, one more seed (40k) -- still the project's
#        ceiling (existing 4-seed mean 0.503); one more seed toward a
#        more robust estimate of the actual target StormLite needs to beat.
#
# Usage: bash scripts/run_parallel_8gpu_overnight_capacity.sh
# Smoke test: SMOKETEST=1 bash scripts/run_parallel_8gpu_overnight_capacity.sh
# Logs: logs/parallel_run_overnight_capacity/<name>.log

set -u

SMOKETEST="${SMOKETEST:-0}"
SMOKETEST_EPOCHS="${SMOKETEST_EPOCHS:-2}"

BIGGER_CFG="configs/exp_hest1k_fm_ot_stormlite_mome_both_bigger_paneldecoder_add_warmupema.yaml"
FLAG_CFG="configs/exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder_add.yaml"
STPATH_PRETRAINED_CFG="configs/exp_hest1k_fm_ot_stpath_bothresidual_paneldecoder_add.yaml"

CONFIGS=(
    "$BIGGER_CFG" "$BIGGER_CFG" "$BIGGER_CFG"
    "$FLAG_CFG" "$FLAG_CFG"
    "$FLAG_CFG" "$FLAG_CFG"
    "$STPATH_PRETRAINED_CFG"
)
NAMES=(
    "stormlite_mome_both_paneldecoder_bigger_qknorm_warmup_ema_seed15"
    "stormlite_mome_both_paneldecoder_bigger_qknorm_warmup_ema_seed16"
    "stormlite_mome_both_paneldecoder_bigger_qknorm_warmup_ema_seed17"
    "stormlite_mome_both_paneldecoder_adalnvelocity_ema_seed12"
    "stormlite_mome_both_paneldecoder_adalnvelocity_ema_seed13"
    "stormlite_mome_both_paneldecoder_biggervelocitynet_ema_seed10"
    "stormlite_mome_both_paneldecoder_biggervelocitynet_ema_seed11"
    "stpath_bothresidual_paneldecoder_pretrained_ema_seed10"
)
SEEDS=(15 16 17 12 13 10 11 10)
EPOCHS=(80000 80000 80000 40000 40000 40000 40000 40000)
EXTRA_OVERRIDE=(
    "model.params.storm_lite_qk_norm=true"
    "model.params.storm_lite_qk_norm=true"
    "model.params.storm_lite_qk_norm=true"
    "model.params.velocity_net_type=adaln_residual model.params.velocity_net_n_layers=3"
    "model.params.velocity_net_type=adaln_residual model.params.velocity_net_n_layers=3"
    "model.params.hidden_dim=1024"
    "model.params.hidden_dim=1024"
    ""
)

LOG_DIR="logs/parallel_run_overnight_capacity"
EXTRA_ARGS="--shuffle-diagnostic"
SMOKE_OVERRIDE=""
if [ "$SMOKETEST" = "1" ]; then
    LOG_DIR="logs/parallel_run_overnight_capacity_smoketest"
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
    extra="${EXTRA_OVERRIDE[$i]}"
    logfile="$LOG_DIR/${name}.log"
    if [ -f "$logfile" ] && grep -q "^model " "$logfile"; then
        echo "  GPU $i: [$name] SKIP (already completed)"
        continue
    fi
    echo "  GPU $i: [$name] START"
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
echo "  StormLite bigger+QK-norm+warmup, 5-seed mean (before this batch): 0.4388 (range 0.19-0.51)"
echo "  AdaLN velocity net, 2-seed mean (before this batch): 0.2895"
echo "  StormLite small (flagship, plain velocity_net, hidden_dim=512): ~0.45-0.47 depending on framing"
echo "  STPath pretrained, 4-seed mean (before this batch): 0.503"
echo "  Combine jobs 0-2 with the existing 5 bigger_qknorm_warmup seeds for an 8-seed mean."
echo "  Combine jobs 3-4 with the existing 2 adalnvelocity seeds for a 4-seed mean."
echo "  Jobs 5-6 (biggervelocitynet, hidden_dim=1024) are a brand new arm -- compare directly"
echo "  against the flagship's plain hidden_dim=512 mean to see if raw velocity_net capacity helps."
