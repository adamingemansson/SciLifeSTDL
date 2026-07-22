#!/bin/bash
# Held-out-spot generalization diagnostic (2026-07-20) -- tests a real,
# mechanistically plausible concern raised via independent review: does
# the model just recall spots it saw as a SUPERVISED TRAINING TARGET many
# times over a training run (its real coordinates/image are always given
# as input, and it's trained via MSE to output that spot's real
# expression -- across thousands of random masking draws on the SAME
# slide, most spots become a training target repeatedly), rather than
# genuinely reconstructing from context? See masking.held_out_mask's own
# docstring and docs/results_log.md's 2026-07-20 entry for the full
# mechanism and reasoning.
#
# Mechanism: masking.heldout_fraction=0.15 fixes 15% of INT1's spots,
# ONCE, as a held-out pool that make_context_query_split's heldout_mask
# param GUARANTEES is never placed in a training query set for the whole
# run (see make_context_query_split's heldout_mask docstring in
# train.py). At the end of training, evaluate_heldout_generalization
# evaluates ONLY on those held-out spots -- genuinely never seen as a
# training target -- and prints it alongside the STANDARD eval number
# (also heldout-excluded, for a fair "seen-eligible" comparison) so the
# gap between them is directly visible in each job's own log. Uses
# get_novae_features_context_only (the 2026-07-20 Novae leak fix) for
# any config using Novae, so this diagnostic isn't ALSO confounded by
# graph-level leakage on top of the memorization question it exists to
# isolate -- per direct instruction to use the fixed Novae path.
#
# Read each job's printed "Standard-vs-held-out PCC gap": near-zero means
# the model genuinely generalizes to unseen spots (memorization concern
# NOT supported); a large positive gap means the standard eval number was
# substantially inflated by recalling spots it had already seen as
# training targets (memorization concern CONFIRMED).
#
# 8 jobs, covering the project's key architectures + one clean isolation
# of Novae from the memorization question itself:
#   1-2. StormLite flagship ("both" gene encoder), 2 seeds
#   3.   StormLite flagship, gene_encoder_type="mlp" (no Novae at all --
#        isolates memorization from the separate Novae-leak question)
#   4-5. STPath unfrozen flagship (no pretraining advantage), 2 seeds
#   6.   STPath PRETRAINED hybrid -- the project's current highest-scoring
#        arm (~0.506), most Novae-dependent, most important to check
#   7.   StormLite bigger+QK-norm+warmup -- does more capacity memorize MORE?
#   8.   STPath pretrained hybrid, stpath_new_gene_encoder_type=none --
#        clean Novae-free STPath-pretrained data point
#
# Epochs deliberately reduced to 20k (not the usual 40k/80k) -- this is a
# DIAGNOSTIC pass to detect whether the effect exists at all, not a final
# result. If the effect shows up, it will already show up at 20k (it's a
# function of REPEATED exposure across steps, not needing full convergence).
#
# Usage: bash scripts/run_parallel_8gpu_heldout_generalization_test.sh
# Smoke test: SMOKETEST=1 bash scripts/run_parallel_8gpu_heldout_generalization_test.sh
# Logs: logs/parallel_run_heldout_generalization_test/<name>.log

set -u

SMOKETEST="${SMOKETEST:-0}"
SMOKETEST_EPOCHS="${SMOKETEST_EPOCHS:-2}"
TEST_EPOCHS="${TEST_EPOCHS:-20000}"
HELDOUT_FRACTION="${HELDOUT_FRACTION:-0.15}"

FLAG_CFG="configs/exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder_add.yaml"
STPATH_UNFROZEN_CFG="configs/exp_hest1k_fm_ot_stpath_unfrozen_bothresidual_paneldecoder_add.yaml"
STPATH_PRETRAINED_CFG="configs/exp_hest1k_fm_ot_stpath_bothresidual_paneldecoder_add.yaml"
BIGGER_CFG="configs/exp_hest1k_fm_ot_stormlite_mome_both_bigger_paneldecoder_add_warmupema.yaml"

CONFIGS=(
    "$FLAG_CFG" "$FLAG_CFG" "$FLAG_CFG"
    "$STPATH_UNFROZEN_CFG" "$STPATH_UNFROZEN_CFG"
    "$STPATH_PRETRAINED_CFG"
    "$BIGGER_CFG"
    "$STPATH_PRETRAINED_CFG"
)
NAMES=(
    "heldouttest_stormlite_both_seed10"
    "heldouttest_stormlite_both_seed11"
    "heldouttest_stormlite_mlp_seed10"
    "heldouttest_stpath_unfrozen_seed10"
    "heldouttest_stpath_unfrozen_seed11"
    "heldouttest_stpath_pretrained_both_seed10"
    "heldouttest_stormlite_bigger_qknorm_warmup_seed10"
    "heldouttest_stpath_pretrained_none_seed10"
)
SEEDS=(10 11 10 10 11 10 10 10)
EXTRA_OVERRIDE=(
    ""
    ""
    "model.params.gene_encoder_type=mlp"
    ""
    ""
    ""
    "model.params.storm_lite_qk_norm=true"
    "model.params.stpath_new_gene_encoder_type=none"
)

LOG_DIR="logs/parallel_run_heldout_generalization_test"
SMOKE_OVERRIDE=""
if [ "$SMOKETEST" = "1" ]; then
    LOG_DIR="logs/parallel_run_heldout_generalization_test_smoketest"
    SMOKE_OVERRIDE="training.epochs=${SMOKETEST_EPOCHS} training.checkpoint_every_n_steps=1 training.log_print_every_n_steps=1"
    echo "*** SMOKETEST=1 -- tiny versions of all 8 jobs (epochs=${SMOKETEST_EPOCHS}). Logs: $LOG_DIR ***"
    rm -rf "$LOG_DIR"
fi
mkdir -p "$LOG_DIR"

TOTAL_JOBS=${#CONFIGS[@]}
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
echo "heldout_fraction=${HELDOUT_FRACTION}, epochs=${TEST_EPOCHS} (diagnostic, not final)"

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    name="${NAMES[$i]}"
    seed="${SEEDS[$i]}"
    extra="${EXTRA_OVERRIDE[$i]}"
    gpu="${GPU_ID_ARR[$i]}"
    logfile="$LOG_DIR/${name}.log"
    if [ -f "$logfile" ] && grep -q "^HELD-OUT mean PCC:" "$logfile"; then
        echo "  GPU $gpu: [$name] SKIP (already completed)"
        continue
    fi
    echo "  GPU $gpu: [$name] START"
    epoch_override=""
    [ "$SMOKETEST" != "1" ] && epoch_override="training.epochs=${TEST_EPOCHS}"
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=$THREADS_PER_JOB MKL_NUM_THREADS=$THREADS_PER_JOB \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 -m src.training.train --config "$cfg" \
        --override ${extra} \
                    training.seed=${seed} \
                    training.ema_decay=0.999 \
                    masking.heldout_fraction=${HELDOUT_FRACTION} \
                    masking.heldout_seed=20260720 \
                    ${epoch_override} \
                    training.checkpoint_dir=results/checkpoints/${name} \
                    training.checkpoint_every_n_steps=10000 \
                    training.log_print_every_n_steps=1000 \
                    ${SMOKE_OVERRIDE} > "$logfile" 2>&1 &
done

wait
echo ""
echo "=== All jobs finished. Standard vs. held-out PCC: ==="
for name in "${NAMES[@]}"; do
    echo "--- $name ---"
    grep -E "^mean PCC:|^HELD-OUT mean PCC:|^Standard-vs-held-out PCC gap:" "$LOG_DIR/${name}.log" 2>/dev/null
done

echo ""
echo "How to read this: a near-zero (or negative) gap means the model genuinely"
echo "generalizes to spots it never saw as a training target -- the memorization"
echo "concern is NOT supported for that config. A large positive gap (standard"
echo "PCC >> held-out PCC) means the standard eval number was substantially"
echo "inflated by recalling already-seen spots -- confirms the concern for that"
echo "config. Compare job3 (mlp, no Novae) against jobs 1-2 (both) to see whether"
echo "Novae's leak (already fixed for eval, still present in training) changes the"
echo "picture; compare job6/8 (STPath pretrained with/without Novae) the same way."
