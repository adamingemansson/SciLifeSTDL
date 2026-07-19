#!/bin/bash
# Sequential pretrain -> finetune driver (2026-07-19) — see
# configs/exp_pretrain_multisample_stormlite_mome_both_paneldecoder_add.yaml
# and exp_finetune_hest1k_stormlite_mome_both_paneldecoder_add.yaml's own
# headers for the full recipe/reasoning. Genuinely sequential (finetune
# needs the pretrain checkpoint to exist) — NOT parallel like every other
# script in this project, so run this on its own GPU alongside whatever
# else is going, or standalone.
#
# Also prints the from-scratch baseline number for direct comparison
# (exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder_add.yaml's own
# 40k-epoch numbers, already in docs/results_log.md — not re-run here,
# just referenced) so the finetune result's real gain (or lack of one) is
# immediately visible, not something you have to go dig up separately.
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run_pretrain_then_finetune_stormlite.sh
# Smoke test: SMOKETEST=1 bash scripts/run_pretrain_then_finetune_stormlite.sh
# Logs: logs/pretrain_finetune/<name>.log

set -u

SMOKETEST="${SMOKETEST:-0}"
SMOKETEST_EPOCHS="${SMOKETEST_EPOCHS:-10}"
LOG_DIR="logs/pretrain_finetune"
SMOKE_OVERRIDE=""
if [ "$SMOKETEST" = "1" ]; then
    LOG_DIR="logs/pretrain_finetune_smoketest"
    SMOKE_OVERRIDE="training.epochs=${SMOKETEST_EPOCHS} training.checkpoint_every_n_steps=1 training.log_print_every_n_steps=1"
    echo "*** SMOKETEST=1 -- tiny pretrain (epochs=${SMOKETEST_EPOCHS}) then tiny finetune. Logs: $LOG_DIR ***"
fi
mkdir -p "$LOG_DIR"

PRETRAIN_CFG="configs/exp_pretrain_multisample_stormlite_mome_both_paneldecoder_add.yaml"
FINETUNE_CFG="configs/exp_finetune_hest1k_stormlite_mome_both_paneldecoder_add.yaml"
PRETRAIN_LOG="$LOG_DIR/pretrain_multisample_stormlite_mome_both_paneldecoder_add.log"
FINETUNE_LOG="$LOG_DIR/finetune_hest1k_stormlite_mome_both_paneldecoder_add.log"

if [ -f "$PRETRAIN_LOG" ] && grep -q "^mean PCC:" "$PRETRAIN_LOG"; then
    echo "Pretrain stage already completed (see $PRETRAIN_LOG), skipping."
else
    echo "=== Stage 1/2: PRETRAIN (multi-sample INT1-INT8) -> $PRETRAIN_LOG ==="
    python -m src.training.train --config "$PRETRAIN_CFG" --override ${SMOKE_OVERRIDE} > "$PRETRAIN_LOG" 2>&1
    echo "Pretrain stage finished. Result:"
    grep -A1 "^mean PCC:" "$PRETRAIN_LOG"
fi

echo ""
if [ -f "$FINETUNE_LOG" ] && grep -q "^model " "$FINETUNE_LOG"; then
    echo "Finetune stage already completed (see $FINETUNE_LOG), skipping."
else
    echo "=== Stage 2/2: FINETUNE (single-sample INT1, warm-started) -> $FINETUNE_LOG ==="
    python -m src.evaluation.run_comparison "$FINETUNE_CFG" --override ${SMOKE_OVERRIDE} --shuffle-diagnostic > "$FINETUNE_LOG" 2>&1
    echo "Finetune stage finished."
fi
echo ""
echo "=== Warm-start summary (what actually transferred) ==="
grep "load_pretrained_weights_into" "$FINETUNE_LOG"
echo ""
echo "=== Finetune result ==="
grep -m1 -A2 "^model " "$FINETUNE_LOG" | tail -2
echo ""
echo "Compare against the from-scratch baseline (docs/results_log.md,"
echo "exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder_add.yaml @ 40k):"
echo "  seed0 (best) = 0.4933, seed1 = 0.4501, seed2 = 0.3657, seed3 = 0.3896,"
echo "  seed4 = 0.4782, seed5+EMA = 0.5391 -- mean ~0.435 without pretraining."
echo "  This finetune run trained only 10k epochs (vs. 40k from-scratch) --"
echo "  a real win means matching/beating the from-scratch mean at 1/4 the epochs."
