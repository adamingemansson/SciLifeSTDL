#!/bin/bash
# 4-config confirmation batch (2026-07-19) — the ONE deliberately cheap,
# targeted follow-up to the 16-config overnight batch (see
# docs/results_log.md's "16-config batch — full results" entry).
#
# Motivation: the 16-config batch's two "bigger" StormLite data points are
# CONFOUNDED and can't answer "does bigger capacity + QK-norm reliably
# close the gap to STPath" on their own:
#   - bigger_qknorm_warmup_40k_ema = 0.3886 (40k epochs, QK-norm+warmup ONLY)
#   - bigger_all3_warmup_80k_ema   = 0.5028 (80k epochs, QK-norm+warmup
#     PLUS logit-normal timestep sampling PLUS input_already_log1p stacked
#     in — and that same "all3" stack collapsed hard on a different seed
#     elsewhere in the same batch, PCC 0.0582, ST-FID 6.44 — see the
#     "real negative result" note in that entry)
# So the higher number (0.5028) can't be attributed to bigger+QK-norm
# specifically — it might be the longer 80k schedule, or the extra
# (risky) stacked changes, or just seed luck. This batch isolates the ONE
# lever worth paying for: bigger capacity + QK-norm + warmup + EMA, alone,
# at a fixed 80k schedule, across 4 seeds — no logit-normal, no
# input_already_log1p, deliberately avoiding the stacking that's already
# shown to be fragile.
#
# Deliberately small: 4 GPUs, 4 jobs, one seed each, no pairing needed —
# per-run cost awareness (33/40GB per device, and GPU-hours cost money).
# Comparison target for the resulting 4-seed mean:
#   STPath (pretrained) + decoder, 4-seed mean ~0.503
#   StormLite (small) + decoder, 8-seed mean (EMA+non-EMA combined) ~0.462
#
# Each job gets its own UNIQUE checkpoint_dir (2026-07-19 checkpoint-
# collision-bug fix pattern, see docs/results_log.md) since all 4 jobs
# share the same base config file concurrently.
#
# Usage: bash scripts/run_parallel_4gpu_bigger_qknorm_confirm.sh
# Smoke test: SMOKETEST=1 bash scripts/run_parallel_4gpu_bigger_qknorm_confirm.sh
# Logs: logs/parallel_run_bigger_qknorm_confirm/<name>.log

set -u

SMOKETEST="${SMOKETEST:-0}"

CFG="configs/exp_hest1k_fm_ot_stormlite_mome_both_bigger_paneldecoder_add_warmupema.yaml"

SEEDS=(10 11 12 13)
JOB_NAME=(
    "stormlite_mome_both_paneldecoder_bigger_qknorm_warmup_ema_seed10"
    "stormlite_mome_both_paneldecoder_bigger_qknorm_warmup_ema_seed11"
    "stormlite_mome_both_paneldecoder_bigger_qknorm_warmup_ema_seed12"
    "stormlite_mome_both_paneldecoder_bigger_qknorm_warmup_ema_seed13"
)

LOG_DIR="logs/parallel_run_bigger_qknorm_confirm"
SMOKE_OVERRIDE=""
if [ "$SMOKETEST" = "1" ]; then
    LOG_DIR="logs/parallel_run_bigger_qknorm_confirm_smoketest"
    SMOKE_OVERRIDE="training.epochs=10 training.checkpoint_every_n_steps=5 training.log_print_every_n_steps=3"
    echo "*** SMOKETEST=1 -- tiny versions of all 4 jobs (epochs=10). Logs: $LOG_DIR ***"
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
        --override model.params.storm_lite_qk_norm=true \
                    training.epochs=80000 \
                    training.seed=${seed} \
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
echo "=== All 4 jobs finished. Results: ==="
for name in "${JOB_NAME[@]}"; do
    echo "--- $name ---"
    grep -m1 -A2 "^model " "$LOG_DIR/${name}.log" 2>/dev/null | tail -2
done

echo ""
echo "Compare the 4-seed mean above against:"
echo "  STPath (pretrained) + decoder, 4-seed mean ~0.503"
echo "  StormLite (small) + decoder, 8-seed mean (EMA+non-EMA) ~0.462"
echo "  Single earlier (confounded) bigger+QK-norm points: 0.3886 (40k, no EMA-mix),"
echo "  0.5028 (80k, but with logit-normal+nolog1p also stacked in -- not a clean read)"
