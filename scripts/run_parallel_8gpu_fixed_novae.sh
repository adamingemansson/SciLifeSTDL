#!/bin/bash
# Leak-free Novae confirmation batch: 8 jobs on 8 devices.
#
# Phase 1 precomputes a shared 64-mask context-only Novae bank. Mask 0 is
# warmed serially first so image/GigaPath caches are established without a
# concurrent first-writer race; all 8 GPUs then precompute eight masks each.
# Phase 2 launches one full training/evaluation job per GPU. Every job sees the
# same masking bank, so differences are architectural/initialization changes,
# not different leaked or randomly selected Novae graphs.
#
# Job matrix:
#   0-1 flagship small MoME+OT baseline, seeds 10/11
#   2-3 bigger MoME+OT + QK norm + lower LR/warmup, seeds 10/11
#   4   flagship, skip the historical double-log1p
#   5   flagship, real minibatch-OT coupling
#   6   flagship, k-NN-restricted spatial attention
#   7   GNN fusion exploratory arm (same encoders/decoder/FM-OT core)
#
# Usage:
#   bash scripts/run_parallel_8gpu_fixed_novae.sh
# Select physical devices:
#   GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/run_parallel_8gpu_fixed_novae.sh
# Smoke test (2 training draws and 8 bank masks):
#   SMOKETEST=1 bash scripts/run_parallel_8gpu_fixed_novae.sh
# Force a clean rerun of logs/checkpoints (cached feature bank is retained):
#   FRESH=1 bash scripts/run_parallel_8gpu_fixed_novae.sh

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_IDS_ARR <<< "$GPU_IDS_CSV"
if [ "${#GPU_IDS_ARR[@]}" -ne 8 ]; then
    echo "ERROR: GPU_IDS must contain exactly 8 comma-separated device IDs; got: $GPU_IDS_CSV"
    exit 1
fi

SMOKETEST="${SMOKETEST:-0}"
SMOKETEST_EPOCHS="${SMOKETEST_EPOCHS:-2}"
FRESH="${FRESH:-0}"

CONFIG="configs/exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder_add_fixed_novae.yaml"
LOG_DIR="logs/parallel_run_fixed_novae_8gpu"
BANK_SIZE=64
BANK_SEED=700000
EPOCHS=40000
if [ "$SMOKETEST" = "1" ]; then
    LOG_DIR="logs/parallel_run_fixed_novae_8gpu_smoketest"
    BANK_SIZE=8
    EPOCHS="$SMOKETEST_EPOCHS"
fi

if [ "$FRESH" = "1" ]; then
    rm -rf "$LOG_DIR"
fi
mkdir -p "$LOG_DIR"

N_CORES=$(nproc)
THREADS_PER_JOB=$(( N_CORES / 8 ))
[ "$THREADS_PER_JOB" -lt 1 ] && THREADS_PER_JOB=1

COMMON_OVERRIDE="training.novae_mask_bank_size=${BANK_SIZE} training.novae_mask_bank_seed=${BANK_SEED} masking.max_context_points=3000"

# Warm one mask serially. This also establishes image/GigaPath caches before
# eight precompute processes begin reading the same sample.
WARM_LOG="$LOG_DIR/precompute_warm_mask0.log"
echo "Warming fixed-Novae bank mask 0 on physical GPU ${GPU_IDS_ARR[0]}..."
CUDA_VISIBLE_DEVICES="${GPU_IDS_ARR[0]}" OMP_NUM_THREADS="$THREADS_PER_JOB" MKL_NUM_THREADS="$THREADS_PER_JOB" \
"$PYTHON_BIN" -m src.evaluation.run_comparison_fixed_novae "$CONFIG" \
    --precompute-only --precompute-start 0 --precompute-count 1 \
    --override $COMMON_OVERRIDE > "$WARM_LOG" 2>&1

# Precompute equal contiguous shards. In smoke mode BANK_SIZE=8, so each GPU
# computes one mask. In the real run each GPU computes eight masks.
SHARD_SIZE=$(( BANK_SIZE / 8 ))
if [ $(( SHARD_SIZE * 8 )) -ne "$BANK_SIZE" ]; then
    echo "ERROR: BANK_SIZE=$BANK_SIZE must be divisible by 8"
    exit 1
fi

echo "Precomputing ${BANK_SIZE}-mask context-only Novae bank across GPUs: $GPU_IDS_CSV"
for slot in $(seq 0 7); do
    start=$(( slot * SHARD_SIZE ))
    gpu="${GPU_IDS_ARR[$slot]}"
    logfile="$LOG_DIR/precompute_shard_${slot}.log"
    (
        CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="$THREADS_PER_JOB" MKL_NUM_THREADS="$THREADS_PER_JOB" \
        "$PYTHON_BIN" -m src.evaluation.run_comparison_fixed_novae "$CONFIG" \
            --precompute-only --precompute-start "$start" --precompute-count "$SHARD_SIZE" \
            --override $COMMON_OVERRIDE > "$logfile" 2>&1
    ) &
done
wait

echo "Novae mask bank complete. Launching eight matched training jobs."

JOB_NAME=(
    "fixednovae_flagship_seed10"
    "fixednovae_flagship_seed11"
    "fixednovae_bigger_mome_ot_seed10"
    "fixednovae_bigger_mome_ot_seed11"
    "fixednovae_flagship_nolog1p_seed10"
    "fixednovae_flagship_minibatchot_seed10"
    "fixednovae_flagship_knn32_seed10"
    "fixednovae_gnn_fusion_seed10"
)
JOB_SEED=(10 11 10 11 10 10 10 10)
JOB_OVERRIDE=(
    ""
    ""
    "model.params.storm_lite_n_layers=4 model.params.storm_lite_n_heads=8 model.params.cond_hidden_dim=512 model.params.storm_lite_qk_norm=true model.params.lr=3.0e-4 model.params.warmup_steps=1000"
    "model.params.storm_lite_n_layers=4 model.params.storm_lite_n_heads=8 model.params.cond_hidden_dim=512 model.params.storm_lite_qk_norm=true model.params.lr=3.0e-4 model.params.warmup_steps=1000"
    "model.params.storm_lite_input_already_log1p=true"
    "model.params.fm_coupling=minibatch_ot"
    "model.params.storm_lite_knn_k=32"
    "model.params.storm_lite_fusion_mode=gnn model.params.storm_lite_gnn_k=8"
)

run_one() {
    local slot="$1"
    local gpu="${GPU_IDS_ARR[$slot]}"
    local name="${JOB_NAME[$slot]}"
    local seed="${JOB_SEED[$slot]}"
    local extra="${JOB_OVERRIDE[$slot]}"
    local logfile="$LOG_DIR/${name}.log"
    local checkpoint="results/checkpoints/${name}"

    if [ "$FRESH" = "1" ]; then
        rm -rf "$checkpoint"
    elif [ -f "$logfile" ] && grep -q "^model " "$logfile"; then
        echo "  GPU $gpu: [$name] SKIP (completed result already present)"
        return
    fi

    echo "  GPU $gpu: [$name] START"
    CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="$THREADS_PER_JOB" MKL_NUM_THREADS="$THREADS_PER_JOB" \
    "$PYTHON_BIN" -m src.evaluation.run_comparison_fixed_novae "$CONFIG" \
        --override $COMMON_OVERRIDE \
                   $extra \
                   experiment_name="$name" \
                   training.epochs="$EPOCHS" \
                   training.seed="$seed" \
                   training.ema_decay=0.999 \
                   training.checkpoint_dir="$checkpoint" \
                   training.checkpoint_every_n_steps=10000 \
                   training.log_print_every_n_steps=1000 \
        --shuffle-diagnostic > "$logfile" 2>&1
    echo "  GPU $gpu: [$name] DONE"
}

for slot in $(seq 0 7); do
    run_one "$slot" &
done
wait

echo ""
echo "=== Fixed-Novae 8-GPU results ==="
for name in "${JOB_NAME[@]}"; do
    echo "--- $name ---"
    grep -m1 -A2 "^model " "$LOG_DIR/${name}.log" 2>/dev/null | tail -2 || true
done

echo ""
echo "All reported rows above use context-only Novae for BOTH the fixed training mask bank and shared evaluation draw."
