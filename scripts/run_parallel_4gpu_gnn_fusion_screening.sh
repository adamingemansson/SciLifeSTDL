#!/bin/bash
# GNN fusion_mode screening batch (2026-07-20) — cheap "rule out or
# continue" pass for storm_lite_fusion_mode="gnn" (stMCDI-inspired sparse
# k-NN message passing, see _GNNBlock's own docstring in
# storm_lite_encoder.py for the honest grounding note). NOT a full
# seed-averaged comparison — 2 seeds per capacity tier, and epochs cut
# from the usual 40k to 15k specifically because the goal here is a fast
# go/no-go signal before committing more compute, not a final number.
# If this looks promising, a proper 40k/multi-seed follow-up is the next
# step, not this batch's own numbers.
#
# 4 jobs: 2 seeds on the small flagship, 2 seeds on the bigger
# (+QK-norm+warmup) flagship — tests whether the GNN fusion mode helps at
# either capacity tier, still isolated (only fusion_mode changes, nothing
# else stacked on top).
#
# Comparison targets (all at their OWN usual epoch counts, NOT 15k — this
# batch's numbers are only meaningful relative to EACH OTHER at 15k, not
# directly against these):
#   StormLite small (fusion_mode="mome", flagship), 11-seed mean: 0.4546 (40k)
#   StormLite bigger+QK-norm+warmup (fusion_mode="mome"), 8-seed mean: 0.4637
#     (0.5031 excl. seed11 outlier) (80k)
#
# Usage: bash scripts/run_parallel_4gpu_gnn_fusion_screening.sh
# Smoke test: SMOKETEST=1 bash scripts/run_parallel_4gpu_gnn_fusion_screening.sh
# Logs: logs/parallel_run_gnn_fusion_screening/<name>.log

set -u

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SMOKETEST="${SMOKETEST:-0}"
SMOKETEST_EPOCHS="${SMOKETEST_EPOCHS:-2}"
SCREEN_EPOCHS="${SCREEN_EPOCHS:-15000}"

FLAG_CFG="configs/exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder_add.yaml"
BIGGER_CFG="configs/exp_hest1k_fm_ot_stormlite_mome_both_bigger_paneldecoder_add_warmupema.yaml"

CONFIGS=(
    "$FLAG_CFG" "$FLAG_CFG"
    "$BIGGER_CFG" "$BIGGER_CFG"
)
NAMES=(
    "stormlite_gnn_paneldecoder_ema_seed10"
    "stormlite_gnn_paneldecoder_ema_seed11"
    "stormlite_gnn_paneldecoder_bigger_qknorm_warmup_ema_seed10"
    "stormlite_gnn_paneldecoder_bigger_qknorm_warmup_ema_seed11"
)
SEEDS=(10 11 10 11)
EXTRA_OVERRIDE=(
    "model.params.storm_lite_fusion_mode=gnn"
    "model.params.storm_lite_fusion_mode=gnn"
    "model.params.storm_lite_fusion_mode=gnn model.params.storm_lite_qk_norm=true"
    "model.params.storm_lite_fusion_mode=gnn model.params.storm_lite_qk_norm=true"
)

LOG_DIR="logs/parallel_run_gnn_fusion_screening"
EXTRA_ARGS="--shuffle-diagnostic"
SMOKE_OVERRIDE=""
if [ "$SMOKETEST" = "1" ]; then
    LOG_DIR="logs/parallel_run_gnn_fusion_screening_smoketest"
    SMOKE_OVERRIDE="training.epochs=${SMOKETEST_EPOCHS} training.checkpoint_every_n_steps=1 training.log_print_every_n_steps=1"
    echo "*** SMOKETEST=1 -- tiny versions of all 4 jobs (epochs=${SMOKETEST_EPOCHS}). Logs: $LOG_DIR ***"
    rm -rf "$LOG_DIR"   # always fresh -- see 2026-07-20 gene-tokenizer-batch fix
fi
mkdir -p "$LOG_DIR"

TOTAL_JOBS=${#CONFIGS[@]}

# GPU_IDS (2026-07-20): comma-separated physical device indices, one per
# job, e.g. GPU_IDS=4,5,6,7 to use GPUs 4-7 instead of the default
# 0..(N-1). Wrapping the whole script in CUDA_VISIBLE_DEVICES=4,5,6,7
# does NOT work -- each job below sets its own per-process
# CUDA_VISIBLE_DEVICES, which overrides rather than combines with an
# outer value. Default reproduces the original 0..(N-1) behavior exactly.
if [ -n "${GPU_IDS:-}" ]; then
    IFS=',' read -ra GPU_ID_ARR <<< "$GPU_IDS"
else
    GPU_ID_ARR=()
    for ((j = 0; j < TOTAL_JOBS; j++)); do GPU_ID_ARR+=("$j"); done
fi
if [ "${#GPU_ID_ARR[@]}" -ne "$TOTAL_JOBS" ]; then
    echo "ERROR: GPU_IDS has ${#GPU_ID_ARR[@]} entries but there are $TOTAL_JOBS jobs -- must match exactly."
    exit 1
fi

N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$TOTAL_JOBS" -gt "$N_GPUS" ]; then
    echo "ERROR: $TOTAL_JOBS configs but only $N_GPUS GPUs visible."
    exit 1
fi
N_CORES=$(nproc)
THREADS_PER_JOB=$((N_CORES / TOTAL_JOBS))
[ "$THREADS_PER_JOB" -lt 1 ] && THREADS_PER_JOB=1

echo "Launching $TOTAL_JOBS jobs on physical GPUs [${GPU_ID_ARR[*]}], ${THREADS_PER_JOB} CPU threads each, logs in $LOG_DIR..."

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    name="${NAMES[$i]}"
    seed="${SEEDS[$i]}"
    extra="${EXTRA_OVERRIDE[$i]}"
    gpu="${GPU_ID_ARR[$i]}"
    logfile="$LOG_DIR/${name}.log"
    if [ -f "$logfile" ] && grep -q "^model " "$logfile"; then
        echo "  GPU $gpu: [$name] SKIP (already completed)"
        continue
    fi
    echo "  GPU $gpu: [$name] START"
    epoch_override=""
    [ "$SMOKETEST" != "1" ] && epoch_override="training.epochs=${SCREEN_EPOCHS}"
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=$THREADS_PER_JOB MKL_NUM_THREADS=$THREADS_PER_JOB \
    python3 -m src.evaluation.run_comparison "$cfg" \
        --override ${extra} \
                    training.seed=${seed} \
                    training.ema_decay=0.999 \
                    ${epoch_override} \
                    training.checkpoint_dir=results/checkpoints/${name} \
                    training.checkpoint_every_n_steps=5000 \
                    training.log_print_every_n_steps=500 \
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
echo "This is a SCREENING pass (${SCREEN_EPOCHS} epochs, not the usual 40k/80k) --"
echo "read it as 'does GNN fusion show ANY signal of working at all', not as a"
echo "final number. A clear win here means a proper full-epoch, more-seeds"
echo "follow-up is worth it; a clear loss or no-signal result means drop it."
