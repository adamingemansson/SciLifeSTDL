#!/bin/bash
# AdaLN-residual velocity_net confirmation batch (2026-07-19) — see
# docs/results_log.md's architecture audit and
# src/models/registry.py's _AdaLNVelocityNet docstring for the full
# reasoning: velocity_net (the actual flow-matching generative core) was
# a plain 2-hidden-layer MLP, >100x smaller than the context_encoder
# feeding it, with no residual connections and conditioning injected via
# one-time input concatenation only — unlike every comparable published
# flow-matching/diffusion architecture. This tests the fix in isolation,
# NOT stacked with anything else (bigger capacity, QK-norm, etc.) — one
# new lever, cleanly measured against the already-known flagship
# baseline, matching this project's established practice after the
# "all3" stacking collapse taught us not to bundle untested changes.
#
# Comparison target: StormLite flagship (small) + decoder, 8-seed mean
# (EMA+non-EMA combined) ~0.462 — see docs/results_log.md. This batch
# adds 3 more seeds with velocity_net_type="adaln_residual", everything
# else IDENTICAL to the flagship config (same decoder, same capacity,
# same EMA), so any real gain (or loss) is attributable to the velocity
# net change alone.
#
# Run this on whichever GPUs are free once the current batch finishes —
# hardcoded to GPU 0/6 like the other st-a100 scripts in this project;
# edit GPU_IDS below if different GPUs are free by the time this runs.
#
# Usage: bash scripts/run_parallel_3gpu_adaln_velocity_confirm.sh
# Smoke test: SMOKETEST=1 bash scripts/run_parallel_3gpu_adaln_velocity_confirm.sh
# Logs: logs/parallel_run_adaln_velocity_confirm/<name>.log

set -u

SMOKETEST="${SMOKETEST:-0}"
SMOKETEST_EPOCHS="${SMOKETEST_EPOCHS:-2}"

CFG="configs/exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder_add.yaml"

SEEDS=(10 11 12)
JOB_NAME=(
    "stormlite_mome_both_paneldecoder_adalnvelocity_ema_seed10"
    "stormlite_mome_both_paneldecoder_adalnvelocity_ema_seed11"
    "stormlite_mome_both_paneldecoder_adalnvelocity_ema_seed12"
)
GPU_IDS=(0 6 0)   # 3 jobs on 2 GPUs, GPU 0 gets 2 sequential, GPU 6 gets 1

LOG_DIR="logs/parallel_run_adaln_velocity_confirm"
SMOKE_OVERRIDE=""
if [ "$SMOKETEST" = "1" ]; then
    LOG_DIR="logs/parallel_run_adaln_velocity_confirm_smoketest"
    SMOKE_OVERRIDE="training.epochs=${SMOKETEST_EPOCHS} training.checkpoint_every_n_steps=1 training.log_print_every_n_steps=1"
    echo "*** SMOKETEST=1 -- tiny versions of all 3 jobs (epochs=${SMOKETEST_EPOCHS}). Logs: $LOG_DIR ***"
fi
mkdir -p "$LOG_DIR"

N_CORES=$(nproc)
N_UNIQUE_GPUS=2
THREADS_PER_JOB=$(( N_CORES / N_UNIQUE_GPUS ))
[ "$THREADS_PER_JOB" -lt 1 ] && THREADS_PER_JOB=1

run_one() {
    local gpu="$1" idx="$2"
    local seed="${SEEDS[$idx]}" name="${JOB_NAME[$idx]}"
    local logfile="$LOG_DIR/${name}.log"
    if [ -f "$logfile" ] && grep -q "^model " "$logfile"; then
        echo "  GPU $gpu: [$name] SKIP (already completed)"
        return
    fi
    echo "  GPU $gpu: [$name] START"
    local epoch_override=""
    [ "$SMOKETEST" != "1" ] && epoch_override="training.epochs=40000"
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=$THREADS_PER_JOB MKL_NUM_THREADS=$THREADS_PER_JOB \
    python -m src.evaluation.run_comparison "$CFG" \
        --override model.params.velocity_net_type=adaln_residual \
                    model.params.velocity_net_n_layers=3 \
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

run_gpu0_queue() { run_one 0 0; run_one 0 2; }
run_gpu6_queue() { run_one 6 1; }

echo "Launching 3 jobs across GPUs 0/6, ${THREADS_PER_JOB} CPU threads each. Logs: $LOG_DIR"
run_gpu0_queue &
run_gpu6_queue &
wait

echo ""
echo "=== All 3 jobs finished. Results: ==="
for name in "${JOB_NAME[@]}"; do
    echo "--- $name ---"
    grep -m1 -A2 "^model " "$LOG_DIR/${name}.log" 2>/dev/null | tail -2
done

echo ""
echo "Compare the 3-seed mean above against:"
echo "  StormLite (small, plain-mlp velocity_net) + decoder, 8-seed mean (EMA+non-EMA): ~0.462"
echo "  A real win means the AdaLN-residual velocity_net's mean clearly beats 0.462 --"
echo "  if it's flat or worse, the velocity_net wasn't the bottleneck and the extra"
echo "  residual/AdaLN complexity isn't worth keeping as a default."
