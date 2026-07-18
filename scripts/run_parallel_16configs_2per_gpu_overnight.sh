#!/bin/bash
# 16-config OVERNIGHT batch (2026-07-19), TWO jobs per GPU run
# CONSECUTIVELY — each of the 8 GPUs runs its first job, then a second job
# on the same device once the first finishes. 8 pairs run in parallel;
# within a pair the two jobs are sequential. Total finishes when the
# slowest GPU's pair is done. Prior batches took ~2h for 8 parallel jobs,
# so 16 in 2-deep pairs is ~4h — comfortably overnight.
#
# The point of 16 slots: SEED-AVERAGE the best configs (turning day3's
# single-seed wins into real 3-seed averages) instead of one seed each,
# AND fully A/B every 2026-07-19 improvement (QK-norm, logit-normal FM
# sampling, input_already_log1p) — see docs/results_log.md's 2026-07-19
# "audit + improvements" entry and each feature's own docstring.
#
# Current bar: StormLite+decoder+EMA seed5 = PCC 0.5391 (day3 best).
# STPath(pretrained)+decoder ~0.503 over 4 seeds.
#
# 16 jobs (index -> GPU: job i and job i+8 share GPU i):
#   SINGLE-SAMPLE (run_comparison.py; flagship = stormlite mome_both +
#   panel_invariant/add decoder; all +EMA + telemetry; 40k unless noted):
#     0. flagship baseline, seed10        } EMA-baseline average
#     1. flagship baseline, seed11        }  (+ day3 seed5 = 3 seeds)
#     2. flagship +QK-norm, seed10        } QK-norm small-model average
#     3. flagship +QK-norm, seed11        }  (+ day4 seed6)
#     4. flagship +logit-normal, seed10   — logit-normal A/B
#     5. flagship +input_already_log1p, seed10 — double-log1p A/B
#     6. flagship +ALL-3-stacked, seed10  } "best model" candidate average
#     7. flagship +ALL-3-stacked, seed11  }  (+ day4 seed9)
#     8. BIGGER +QK-norm+warmup, 40k      — does QK-norm make bigger work?
#     9. BIGGER +ALL-3+warmup, 80k        — bigger, everything, longer
#   MULTI-SAMPLE (train.py directly, INT1-INT8; all +EMA; 40k):
#    10. StormLite mome_both +decoder +QK-norm
#    11. StormLite mome_both +decoder +ALL-3-stacked
#    12. STPath UNFROZEN +decoder (no pretraining advantage)
#    13. STPath FROZEN (pretrained) +decoder
#    14. StormLite BIGGER +decoder +QK-norm+warmup
#    15. StormLite mome_both +decoder (EMA-only baseline, vs day3's 0.1371)
#
# Pairing keeps the two heaviest (job 9 = bigger 80k; job 14 = multisample
# bigger) on different GPUs and as the SECOND job of their pair, so no GPU
# runs two heavy jobs back to back.
#
# Usage: bash scripts/run_parallel_16configs_2per_gpu_overnight.sh
# Smoke test: SMOKETEST=1 bash scripts/run_parallel_16configs_2per_gpu_overnight.sh
# Logs: logs/parallel_run_16/<name>.log

set -u

SMOKETEST="${SMOKETEST:-0}"

# reusable override fragments
DEC="model.params.decoder_type=panel_invariant model.params.decoder_gene_embed_dim=256 model.params.decoder_combine_mode=add model.params.decoder_mlp_depth=2"
BIG="model.params.storm_lite_n_layers=4 model.params.storm_lite_n_heads=8 model.params.cond_hidden_dim=512 model.params.lr=3.0e-4 model.params.warmup_steps=1000"
ALL3="model.params.storm_lite_qk_norm=true model.params.fm_time_sampling=logit_normal model.params.storm_lite_input_already_log1p=true"
CK="training.checkpoint_every_n_steps=10000 training.log_print_every_n_steps=1000"
EMA="training.ema_decay=0.999"

FLAG="configs/exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder_add.yaml"
BIGGER="configs/exp_hest1k_fm_ot_stormlite_mome_both_bigger_paneldecoder_add_warmupema.yaml"
MS_SL="configs/exp_multisample_fm_ot_stormlite_mome_both_paneldecoder_add.yaml"
MS_STPATH="configs/exp_multisample_fm_ot_stpath_bothresidual.yaml"

# 16 parallel arrays (index 0-15). JOB_TYPE: "single" (run_comparison.py,
# completion marker "^model ") or "multi" (train.py, marker "^mean PCC:").
JOB_CONFIG=(
    "$FLAG" "$FLAG" "$FLAG" "$FLAG" "$FLAG" "$FLAG" "$FLAG" "$FLAG" "$BIGGER" "$BIGGER"
    "$MS_SL" "$MS_SL" "$MS_STPATH" "$MS_STPATH" "$MS_SL" "$MS_SL"
)
JOB_TYPE=(
    single single single single single single single single single single
    multi multi multi multi multi multi
)
JOB_OVERRIDE=(
    "training.epochs=40000 training.seed=10 ${CK} ${EMA}"
    "training.epochs=40000 training.seed=11 ${CK} ${EMA}"
    "model.params.storm_lite_qk_norm=true training.epochs=40000 training.seed=10 ${CK} ${EMA}"
    "model.params.storm_lite_qk_norm=true training.epochs=40000 training.seed=11 ${CK} ${EMA}"
    "model.params.fm_time_sampling=logit_normal training.epochs=40000 training.seed=10 ${CK} ${EMA}"
    "model.params.storm_lite_input_already_log1p=true training.epochs=40000 training.seed=10 ${CK} ${EMA}"
    "${ALL3} training.epochs=40000 training.seed=10 ${CK} ${EMA}"
    "${ALL3} training.epochs=40000 training.seed=11 ${CK} ${EMA}"
    "model.params.storm_lite_qk_norm=true training.epochs=40000 ${CK} ${EMA}"
    "${ALL3} training.epochs=80000 ${CK} ${EMA}"
    "model.params.storm_lite_qk_norm=true training.epochs=40000 ${CK} ${EMA}"
    "${ALL3} training.epochs=40000 ${CK} ${EMA}"
    "model.params.stpath_pretrained=false ${DEC} training.epochs=40000 ${CK} ${EMA}"
    "${DEC} training.epochs=40000 ${CK} ${EMA}"
    "${BIG} model.params.storm_lite_qk_norm=true training.epochs=40000 ${CK} ${EMA}"
    "training.epochs=40000 ${CK} ${EMA}"
)
JOB_NAME=(
    "flagship_baseline_seed10_ema"
    "flagship_baseline_seed11_ema"
    "flagship_qknorm_seed10_ema"
    "flagship_qknorm_seed11_ema"
    "flagship_logitnormal_seed10_ema"
    "flagship_nolog1p_seed10_ema"
    "flagship_all3_seed10_ema"
    "flagship_all3_seed11_ema"
    "bigger_qknorm_warmup_40k_ema"
    "bigger_all3_warmup_80k_ema"
    "ms_stormlite_qknorm_ema"
    "ms_stormlite_all3_ema"
    "ms_stpath_unfrozen_decoder_ema"
    "ms_stpath_frozen_decoder_ema"
    "ms_stormlite_bigger_qknorm_ema"
    "ms_stormlite_baseline_ema"
)

LOG_DIR="logs/parallel_run_16"
EXTRA_ARGS="--shuffle-diagnostic"
SMOKE_OVERRIDE=""
if [ "$SMOKETEST" = "1" ]; then
    LOG_DIR="logs/parallel_run_16_smoketest"
    SMOKE_OVERRIDE="training.epochs=10 training.checkpoint_every_n_steps=5 training.log_print_every_n_steps=3"
    echo "*** SMOKETEST=1 -- tiny versions of all 16 jobs (epochs=10). Logs: $LOG_DIR ***"
fi

N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
N_JOBS=${#JOB_CONFIG[@]}
PAIRS=$(( (N_JOBS + 1) / 2 ))            # GPUs actually used (= 8 for 16 jobs)
N_CORES=$(nproc)
# only PAIRS jobs run concurrently (one per GPU at a time), so divide cores
# by the number of concurrent jobs, not the total job count
THREADS_PER_JOB=$(( N_CORES / PAIRS ))
[ "$THREADS_PER_JOB" -lt 1 ] && THREADS_PER_JOB=1

if [ "$PAIRS" -gt "$N_GPUS" ]; then
    echo "ERROR: need $PAIRS GPUs (2 jobs each for $N_JOBS jobs) but only $N_GPUS visible."
    exit 1
fi

mkdir -p "$LOG_DIR"
echo "Launching $N_JOBS jobs as $PAIRS consecutive pairs across GPUs 0-$((PAIRS-1)), ${THREADS_PER_JOB} CPU threads each, logs in $LOG_DIR..."

# run one job (blocking) on a given GPU; skips if already completed
run_one() {
    local gpu="$1" idx="$2"
    local cfg="${JOB_CONFIG[$idx]}" extra="${JOB_OVERRIDE[$idx]}"
    local name="${JOB_NAME[$idx]}" jtype="${JOB_TYPE[$idx]}"
    local logfile="$LOG_DIR/${name}.log"
    local marker
    [ "$jtype" = "single" ] && marker="^model " || marker="^mean PCC:"
    if [ -f "$logfile" ] && grep -q "$marker" "$logfile"; then
        echo "  GPU $gpu: [$name] SKIP (already completed)"
        return
    fi
    echo "  GPU $gpu: [$name] START ($jtype)"
    if [ "$jtype" = "single" ]; then
        CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=$THREADS_PER_JOB MKL_NUM_THREADS=$THREADS_PER_JOB \
        python -m src.evaluation.run_comparison "$cfg" \
            --override ${extra} ${SMOKE_OVERRIDE} ${EXTRA_ARGS} > "$logfile" 2>&1
    else
        CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=$THREADS_PER_JOB MKL_NUM_THREADS=$THREADS_PER_JOB \
        python -m src.training.train --config "$cfg" \
            --override ${extra} ${SMOKE_OVERRIDE} > "$logfile" 2>&1
    fi
    echo "  GPU $gpu: [$name] DONE"
}

# each GPU runs job[gpu] then job[gpu+PAIRS], consecutively
run_gpu_pair() {
    local gpu="$1"
    run_one "$gpu" "$gpu"
    local second=$(( gpu + PAIRS ))
    [ "$second" -lt "$N_JOBS" ] && run_one "$gpu" "$second"
}

for gpu in $(seq 0 $((PAIRS - 1))); do
    run_gpu_pair "$gpu" &
done
wait
echo ""
echo "All 16 jobs finished. Summary:"
echo ""
echo "=== SINGLE-SAMPLE (real result row, not the shuffle-diagnostic) ==="
for i in "${!JOB_NAME[@]}"; do
    [ "${JOB_TYPE[$i]}" = "single" ] || continue
    name="${JOB_NAME[$i]}"
    echo "--- $name ---"
    grep -m1 -A2 "^model " "$LOG_DIR/${name}.log" 2>/dev/null | tail -2
done
echo ""
echo "=== MULTI-SAMPLE ==="
for i in "${!JOB_NAME[@]}"; do
    [ "${JOB_TYPE[$i]}" = "multi" ] || continue
    name="${JOB_NAME[$i]}"
    echo "--- $name ---"
    grep -A1 "^mean PCC:" "$LOG_DIR/${name}.log" 2>/dev/null
done
