#!/bin/bash
# Gene-tokenizer confirmation batch (2026-07-19) — the "do the gene-
# tokenization, and combinations with Novae" follow-up to the second
# architecture research pass (see docs/results_log.md and
# TokenizedGeneEncoder's own docstring in conditioning.py).
#
# Deliberately SMALL given "we cannot run as much now": 4 jobs, not 8 —
# 2 seeds each of the 2 NEW gene_encoder_type options ("tokenizer",
# "tokenizer_novae"), on the small flagship (NOT stacked with "bigger"
# capacity or the AdaLN velocity net) so this cleanly isolates the ONE
# new question -- does per-gene identity-aware tokenization beat the
# existing dense-MLP/Novae-whole-profile gene encoders, and does Novae
# add anything on top of it (orthogonal axis, see TokenizedGeneEncoder's
# docstring: Novae's spatial awareness is external/pretrained, unrelated
# to per-gene identity).
#
# Comparison targets, already known (docs/results_log.md):
#   gene_encoder_type="mlp"+novae ("both"): StormLite small flagship,
#     11-seed mean 0.4546
#   STPath unfrozen (the actual target this new encoder is motivated by
#     -- STPath's real per-gene tokenization is the one structural
#     difference identified that we've never tried closing): 5-seed mean
#     0.4706
#
# Usage: bash scripts/run_parallel_4gpu_gene_tokenizer_confirm.sh
# Smoke test: SMOKETEST=1 bash scripts/run_parallel_4gpu_gene_tokenizer_confirm.sh
# Logs: logs/parallel_run_gene_tokenizer_confirm/<name>.log

set -u

SMOKETEST="${SMOKETEST:-0}"
SMOKETEST_EPOCHS="${SMOKETEST_EPOCHS:-2}"

FLAG_CFG="configs/exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder_add.yaml"

CONFIGS=("$FLAG_CFG" "$FLAG_CFG" "$FLAG_CFG" "$FLAG_CFG")
NAMES=(
    "stormlite_mome_tokenizer_paneldecoder_ema_seed10"
    "stormlite_mome_tokenizer_paneldecoder_ema_seed11"
    "stormlite_mome_tokenizernovae_paneldecoder_ema_seed10"
    "stormlite_mome_tokenizernovae_paneldecoder_ema_seed11"
)
SEEDS=(10 11 10 11)
EXTRA_OVERRIDE=(
    "model.params.gene_encoder_type=tokenizer"
    "model.params.gene_encoder_type=tokenizer"
    "model.params.gene_encoder_type=tokenizer_novae"
    "model.params.gene_encoder_type=tokenizer_novae"
)

LOG_DIR="logs/parallel_run_gene_tokenizer_confirm"
EXTRA_ARGS="--shuffle-diagnostic"
SMOKE_OVERRIDE=""
if [ "$SMOKETEST" = "1" ]; then
    LOG_DIR="logs/parallel_run_gene_tokenizer_confirm_smoketest"
    SMOKE_OVERRIDE="training.epochs=${SMOKETEST_EPOCHS} training.checkpoint_every_n_steps=1 training.log_print_every_n_steps=1"
    echo "*** SMOKETEST=1 -- tiny versions of all 4 jobs (epochs=${SMOKETEST_EPOCHS}). Logs: $LOG_DIR ***"
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
    [ "$SMOKETEST" != "1" ] && epoch_override="training.epochs=40000"
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
echo "  StormLite small (flagship, gene_encoder_type='both'), 11-seed mean: 0.4546"
echo "  STPath unfrozen (the real per-gene-tokenization architecture this is modeled on), 5-seed mean: 0.4706"
echo "  A real win means tokenizer/tokenizer_novae's mean clearly beats 0.4546 -- and if it"
echo "  approaches or beats 0.4706, that's real evidence the gene-tokenization gap was the"
echo "  actual structural difference driving STPath's edge."
