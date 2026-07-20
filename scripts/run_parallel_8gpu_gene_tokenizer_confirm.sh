#!/bin/bash
# Gene-tokenizer confirmation batch (2026-07-19) — the "do the gene-
# tokenization, and combinations with Novae AND bigger capacity" follow-up
# to the second architecture research pass (see docs/results_log.md and
# TokenizedGeneEncoder's own docstring in conditioning.py).
#
# 8 jobs: a clean 2x2x2 design -- {tokenizer, tokenizer_novae} x
# {small flagship, bigger+QK-norm+warmup} x {seed10, seed11}. Each cell
# isolated (no other untested lever stacked in) so results are cleanly
# attributable: does per-gene identity-aware tokenization help at all,
# does Novae add anything on top of it (orthogonal axis -- Novae's
# spatial awareness is external/pretrained, unrelated to gene identity),
# and does the already-promising "bigger" capacity arm (see
# docs/results_log.md's overnight-capacity-batch entry, 7/8 seeds
# clustered ~0.50) compound with the new gene encoder or not.
#
# Comparison targets, already known (docs/results_log.md):
#   gene_encoder_type="both" (mlp+novae), small flagship: 11-seed mean 0.4546
#   bigger+QK-norm+warmup (existing gene_encoder_type="both"): 8-seed mean
#     0.4637 (0.5031 over 7 seeds excl. the seed11 outlier)
#   STPath unfrozen (the real per-gene-tokenization architecture this new
#     encoder is modeled on): 5-seed mean 0.4706
#   STPath pretrained (the actual ceiling): 5-seed mean 0.5060
#
# Usage: bash scripts/run_parallel_4gpu_gene_tokenizer_confirm.sh
# Smoke test: SMOKETEST=1 bash scripts/run_parallel_4gpu_gene_tokenizer_confirm.sh
# Logs: logs/parallel_run_gene_tokenizer_confirm/<name>.log

set -u

SMOKETEST="${SMOKETEST:-0}"
SMOKETEST_EPOCHS="${SMOKETEST_EPOCHS:-2}"

FLAG_CFG="configs/exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder_add.yaml"
BIGGER_CFG="configs/exp_hest1k_fm_ot_stormlite_mome_both_bigger_paneldecoder_add_warmupema.yaml"

CONFIGS=(
    "$FLAG_CFG" "$FLAG_CFG"
    "$FLAG_CFG" "$FLAG_CFG"
    "$BIGGER_CFG" "$BIGGER_CFG"
    "$BIGGER_CFG" "$BIGGER_CFG"
)
NAMES=(
    "stormlite_mome_tokenizer_paneldecoder_ema_seed10"
    "stormlite_mome_tokenizer_paneldecoder_ema_seed11"
    "stormlite_mome_tokenizernovae_paneldecoder_ema_seed10"
    "stormlite_mome_tokenizernovae_paneldecoder_ema_seed11"
    "stormlite_mome_tokenizer_paneldecoder_bigger_qknorm_warmup_ema_seed10"
    "stormlite_mome_tokenizer_paneldecoder_bigger_qknorm_warmup_ema_seed11"
    "stormlite_mome_tokenizernovae_paneldecoder_bigger_qknorm_warmup_ema_seed10"
    "stormlite_mome_tokenizernovae_paneldecoder_bigger_qknorm_warmup_ema_seed11"
)
SEEDS=(10 11 10 11 10 11 10 11)
# 2026-07-19: ALL 8 jobs at 40k -- deliberately matched, not the "bigger"
# arm's usual 80k. Big-vs-small used to be confounded with training
# length (bigger always got 2x the epochs) in every prior comparison in
# this project; matching epochs here means capacity is the ONLY thing
# that differs between the small and bigger halves of this batch, so any
# gap is cleanly attributable to capacity, not training time. NOTE: this
# means the "bigger" jobs below are NOT directly comparable to the
# established 80k-epoch bigger_qknorm_warmup baseline (0.4637/0.5031,
# docs/results_log.md) -- this batch needs its own within-batch big-vs-
# small read, not a cross-batch one against that different-epoch-budget number.
EPOCHS=(40000 40000 40000 40000 40000 40000 40000 40000)
EXTRA_OVERRIDE=(
    "model.params.gene_encoder_type=tokenizer"
    "model.params.gene_encoder_type=tokenizer"
    "model.params.gene_encoder_type=tokenizer_novae"
    "model.params.gene_encoder_type=tokenizer_novae"
    "model.params.gene_encoder_type=tokenizer model.params.storm_lite_qk_norm=true"
    "model.params.gene_encoder_type=tokenizer model.params.storm_lite_qk_norm=true"
    "model.params.gene_encoder_type=tokenizer_novae model.params.storm_lite_qk_norm=true"
    "model.params.gene_encoder_type=tokenizer_novae model.params.storm_lite_qk_norm=true"
)

LOG_DIR="logs/parallel_run_gene_tokenizer_confirm"
EXTRA_ARGS="--shuffle-diagnostic"
SMOKE_OVERRIDE=""
if [ "$SMOKETEST" = "1" ]; then
    LOG_DIR="logs/parallel_run_gene_tokenizer_confirm_smoketest"
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
echo "  StormLite small (flagship, gene_encoder_type='both'), 11-seed mean: 0.4546 (40k epochs)"
echo "  StormLite bigger+QK-norm+warmup (gene_encoder_type='both'), 8-seed mean: 0.4637, or 0.5031"
echo "    over 7 seeds excl. the seed11 outlier -- BUT that's at 80k epochs, not this batch's 40k,"
echo "    so it's NOT a clean cross-batch comparison for the bigger jobs below (jobs 4-7)."
echo "    The clean read for THIS batch is small-vs-bigger WITHIN it (both at 40k) --"
echo "    that isolates capacity as the only difference, not training length."
echo "  STPath unfrozen (the real per-gene-tokenization architecture this is modeled on), 5-seed mean: 0.4706"
echo "  STPath pretrained (the actual ceiling), 5-seed mean: 0.5060"
echo "  A real win on the small-flagship tokenizer jobs means it beats 0.4546 outright."
echo "  A real win on the bigger tokenizer jobs means it beats 0.4637/0.5031 -- i.e. gene"
echo "  identity AND capacity are both real, separate, additive levers, not redundant ones."
