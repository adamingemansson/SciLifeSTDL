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
# run_parallel_3gpu_stpath_unfrozen_confirm.sh) PLUS one new diagnostic
# job into 8 jobs queued sequentially across the 2 available GPUs (4 per
# GPU) -- see each job's own header comment in this file for what it
# tests and why. Reasoning summary (docs/results_log.md has the full
# numbers each of these compares against):
#
#   1-4. bigger + QK-norm + warmup + EMA, seeds 10/11/12/13, 80k epochs --
#        the core open question from the 16-config batch: is bigger-
#        capacity StormLite a genuinely reliable win, or was the single
#        0.5028 data point luck/confound (it also had logit-normal +
#        no-log1p stacked in)? This gives a clean, unconfounded 4-seed
#        mean, comparable to StormLite-small's 0.462 (8-seed) and
#        STPath's 0.503 (4-seed).
#   5-7. STPath UNFROZEN + winning decoder + EMA, seeds 10/11/12, 40k
#        epochs -- the "full retrain" comparison: STPath with zero
#        pretrained weights, same decoder everyone else uses. Only one
#        old data point exists (0.3857, BEFORE the decoder swap) --
#        tells us whether StormLite's edge over from-scratch STPath is
#        real architecture, or whether the decoder swap helps STPath's
#        architecture just as much.
#   8.   bigger + QK-norm ONLY, flat lr=1.0e-3, warmup_steps=0, seed 10,
#        40k epochs -- isolates QK-norm from the LR/warmup change it's
#        always been bundled with so far. Every "bigger+QK-norm works"
#        result to date also had a lowered LR (3e-4) and 1000-step warmup
#        baked into the same config edit -- never tested separately. If
#        this collapses back to nan/AUC~0.5, warmup/lower-LR were doing
#        real work; if it trains fine, QK-norm alone was sufficient.
#
# Load split (epoch-cost balanced, 4 jobs/GPU):
#   GPU 0: bigger_seed10 (80k) + bigger_seed12 (80k) + stpath_seed10 (40k) + stpath_seed12 (40k)              = 240k epoch-units
#   GPU 6: bigger_seed11 (80k) + bigger_seed13 (80k) + stpath_seed11 (40k) + bigger_qknorm_only_flatlr (40k)  = 240k epoch-units
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
    "$BIGGER_CFG"
)
JOB_NAME=(
    "stormlite_mome_both_paneldecoder_bigger_qknorm_warmup_ema_seed10"
    "stormlite_mome_both_paneldecoder_bigger_qknorm_warmup_ema_seed11"
    "stormlite_mome_both_paneldecoder_bigger_qknorm_warmup_ema_seed12"
    "stormlite_mome_both_paneldecoder_bigger_qknorm_warmup_ema_seed13"
    "stpath_bothresidual_paneldecoder_unfrozen_ema_seed10"
    "stpath_bothresidual_paneldecoder_unfrozen_ema_seed11"
    "stpath_bothresidual_paneldecoder_unfrozen_ema_seed12"
    "stormlite_mome_both_paneldecoder_bigger_qknormonly_flatlr_ema_seed10"
)
JOB_SEED=(10 11 12 13 10 11 12 10)
JOB_EPOCHS=(80000 80000 80000 80000 40000 40000 40000 40000)
# bigger jobs need storm_lite_qk_norm=true; stpath jobs already have their
# architecture fixed in the config, only seed/epochs/checkpoint vary; job 8
# is QK-norm alone -- explicitly resets warmup_steps/lr back to the
# ORIGINAL (pre-"bigger fix") values to isolate QK-norm's own effect
JOB_EXTRA_OVERRIDE=(
    "model.params.storm_lite_qk_norm=true"
    "model.params.storm_lite_qk_norm=true"
    "model.params.storm_lite_qk_norm=true"
    "model.params.storm_lite_qk_norm=true"
    ""
    ""
    ""
    "model.params.storm_lite_qk_norm=true model.params.warmup_steps=0 model.params.lr=1.0e-3"
)

# GPU 0 <- jobs 0,2,4,6 ; GPU 6 <- jobs 1,3,5,7 (epoch-cost-balanced, see header)
GPU0_JOBS=(0 2 4 6)
GPU6_JOBS=(1 3 5 7)

LOG_DIR="logs/parallel_run_2gpu_confirm"
SMOKE_OVERRIDE=""
if [ "$SMOKETEST" = "1" ]; then
    LOG_DIR="logs/parallel_run_2gpu_confirm_smoketest"
    SMOKE_OVERRIDE="training.epochs=10 training.checkpoint_every_n_steps=5 training.log_print_every_n_steps=3"
    echo "*** SMOKETEST=1 -- tiny versions of all 8 jobs (epochs=10). Logs: $LOG_DIR ***"
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

echo "Launching 8 jobs across 2 free GPUs (0, 6) on st-a100, ${THREADS_PER_JOB} CPU threads each."
echo "GPUs 1,2,3,4,5,7 are occupied by markus.+'s jobs -- not touched."
echo "Logs: $LOG_DIR"

run_gpu_queue 0 "${GPU0_JOBS[@]}" &
run_gpu_queue 6 "${GPU6_JOBS[@]}" &
wait

echo ""
echo "=== All 8 jobs finished. Results: ==="
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
echo "  Job 8 (qknormonly_flatlr) is a standalone diagnostic -- collapse (nan/AUC~0.5)"
echo "  means warmup+lower-LR were doing real work; a real number means QK-norm alone sufficed."
