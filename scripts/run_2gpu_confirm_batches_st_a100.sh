#!/bin/bash
# Merged confirmation batch (2026-07-19), rewritten for st-a100 specifically
# -- that node is SHARED (markus.+ has real, active jobs on GPUs 1,2,3,4,5,7
# as of this writing, confirmed via `ps -p <pid> -o pid,user,etime,cmd` on
# every process, not just an nvidia-smi snapshot) -- only GPU 0 and GPU 6
# are actually free. Do NOT widen GPU_IDS below without re-checking
# `nvidia-smi` + `ps` on any newly-idle-looking device first.
#
# Combines both of the smaller confirmation batches
# (run_parallel_4gpu_bigger_qknorm_confirm.sh,
# run_parallel_3gpu_stpath_unfrozen_confirm.sh) into 7 jobs queued
# sequentially across the 2 available GPUs instead of running each in
# parallel across 4/3 dedicated GPUs -- see each job's own header comment
# in this file for what it tests and why.
#
# Load split (epoch-cost balanced, not just job-count balanced):
#   GPU 0: bigger_seed10 (80k) + bigger_seed12 (80k) + stpath_seed10 (40k) + stpath_seed12 (40k)  = 240k epoch-units, 4 jobs
#   GPU 6: bigger_seed11 (80k) + bigger_seed13 (80k) + stpath_seed11 (40k)                          = 200k epoch-units, 3 jobs
#
# Usage: bash scripts/run_2gpu_confirm_batches_st_a100.sh
# Smoke test: SMOKETEST=1 bash scripts/run_2gpu_confirm_batches_st_a100.sh
# Logs: logs/parallel_run_2gpu_confirm/<name>.log

set -u

SMOKETEST="${SMOKETEST:-0}"

BIGGER_CFG="configs/exp_hest1k_fm_ot_stormlite_mome_both_bigger_paneldecoder_add_warmupema.yaml"
STPATH_CFG="configs/exp_hest1k_fm_ot_stpath_unfrozen_bothresidual_paneldecoder_add.yaml"

# job index -> (config, name, override extras beyond seed/checkpoint_dir)
JOB_CONFIG=(
    "$BIGGER_CFG" "$BIGGER_CFG" "$BIGGER_CFG" "$BIGGER_CFG"
    "$STPATH_CFG" "$STPATH_CFG" "$STPATH_CFG"
)
JOB_NAME=(
    "stormlite_mome_both_paneldecoder_bigger_qknorm_warmup_ema_seed10"
    "stormlite_mome_both_paneldecoder_bigger_qknorm_warmup_ema_seed11"
    "stormlite_mome_both_paneldecoder_bigger_qknorm_warmup_ema_seed12"
    "stormlite_mome_both_paneldecoder_bigger_qknorm_warmup_ema_seed13"
    "stpath_bothresidual_paneldecoder_unfrozen_ema_seed10"
    "stpath_bothresidual_paneldecoder_unfrozen_ema_seed11"
    "stpath_bothresidual_paneldecoder_unfrozen_ema_seed12"
)
JOB_SEED=(10 11 12 13 10 11 12)
JOB_EPOCHS=(80000 80000 80000 80000 40000 40000 40000)
# bigger jobs need storm_lite_qk_norm=true; stpath jobs already have their
# architecture fixed in the config, only seed/epochs/checkpoint vary
JOB_EXTRA_OVERRIDE=(
    "model.params.storm_lite_qk_norm=true"
    "model.params.storm_lite_qk_norm=true"
    "model.params.storm_lite_qk_norm=true"
    "model.params.storm_lite_qk_norm=true"
    ""
    ""
    ""
)

# GPU 0 <- jobs 0,2,4,6 ; GPU 6 <- jobs 1,3,5 (epoch-cost-balanced, see header)
GPU0_JOBS=(0 2 4 6)
GPU6_JOBS=(1 3 5)

LOG_DIR="logs/parallel_run_2gpu_confirm"
SMOKE_OVERRIDE=""
if [ "$SMOKETEST" = "1" ]; then
    LOG_DIR="logs/parallel_run_2gpu_confirm_smoketest"
    SMOKE_OVERRIDE="training.epochs=10 training.checkpoint_every_n_steps=5 training.log_print_every_n_steps=3"
    echo "*** SMOKETEST=1 -- tiny versions of all 7 jobs (epochs=10). Logs: $LOG_DIR ***"
fi
mkdir -p "$LOG_DIR"

N_CORES=$(nproc)
THREADS_PER_JOB=$(( N_CORES / 2 ))   # 2 GPUs run concurrently
[ "$THREADS_PER_JOB" -lt 1 ] && THREADS_PER_JOB=1

run_one() {
    local gpu="$1" idx="$2"
    local cfg="${JOB_CONFIG[$idx]}" name="${JOB_NAME[$idx]}" seed="${JOB_SEED[$idx]}"
    local epochs="${JOB_EPOCHS[$idx]}" extra="${JOB_EXTRA_OVERRIDE[$idx]}"
    local logfile="$LOG_DIR/${name}.log"
    if [ -f "$logfile" ] && grep -q "^model " "$logfile"; then
        echo "  GPU $gpu: [$name] SKIP (already completed)"
        return
    fi
    echo "  GPU $gpu: [$name] START"
    local epoch_override=""
    [ "$SMOKETEST" != "1" ] && epoch_override="training.epochs=${epochs}"
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=$THREADS_PER_JOB MKL_NUM_THREADS=$THREADS_PER_JOB \
    python -m src.evaluation.run_comparison "$cfg" \
        --override ${extra} \
                    training.seed=${seed} \
                    training.ema_decay=0.999 \
                    ${epoch_override} \
                    training.checkpoint_dir=results/checkpoints/${name} \
                    training.checkpoint_every_n_steps=10000 \
                    training.log_print_every_n_steps=1000 \
                    ${SMOKE_OVERRIDE} \
        --shuffle-diagnostic > "$logfile" 2>&1
    echo "  GPU $gpu: [$name] DONE"
}

run_gpu_queue() {
    local gpu="$1"; shift
    local indices=("$@")
    for idx in "${indices[@]}"; do
        run_one "$gpu" "$idx"
    done
}

echo "Launching 7 jobs across 2 free GPUs (0, 6) on st-a100, ${THREADS_PER_JOB} CPU threads each."
echo "GPUs 1,2,3,4,5,7 are occupied by markus.+'s jobs -- not touched."
echo "Logs: $LOG_DIR"

run_gpu_queue 0 "${GPU0_JOBS[@]}" &
run_gpu_queue 6 "${GPU6_JOBS[@]}" &
wait

echo ""
echo "=== All 7 jobs finished. Results: ==="
for name in "${JOB_NAME[@]}"; do
    echo "--- $name ---"
    grep -m1 -A2 "^model " "$LOG_DIR/${name}.log" 2>/dev/null | tail -2
done

echo ""
echo "Compare against:"
echo "  StormLite (small) + decoder, 8-seed mean (EMA+non-EMA): ~0.462"
echo "  STPath (pretrained) + decoder, 4-seed mean: ~0.503"
echo "  STPath (unfrozen) + OLD dense decoder, 1 seed, no EMA: 0.3857"
echo "  Earlier confounded bigger+QK-norm points: 0.3886 (40k), 0.5028 (80k, all3-stacked)"
