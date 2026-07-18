#!/bin/bash
# Day 4 OVERNIGHT batch (2026-07-19) — first batch built on the three new
# improvements from today's deep code-audit + literature research (see
# docs/results_log.md's 2026-07-19 "audit + improvements" entry and each
# feature's own docstring in src/):
#
#   A. QK-normalization (storm_lite_qk_norm, _QKNormAttention) — the real
#      STRUCTURAL fix for the "bigger" StormLite attention-entropy collapse
#      (Henry et al. 2020 EMNLP; ViT-22B / SD3 variant). A stress test
#      (tests/test_qknorm_timesampling_log1p.py) confirmed it bounds
#      attention logits to ~2.8 where the un-normed path explodes to
#      ~4200 under blown-up Q/K weights — exactly the softmax-saturation
#      failure gradient clipping alone couldn't fix (and made worse).
#   B. Logit-normal flow-matching timestep sampling (fm_time_sampling) —
#      SD3 / Esser et al. 2024 (arXiv 2403.03206): concentrates training
#      supervision on the informative middle of the noise->data trajectory.
#   C. input_already_log1p — opt-out of a real double-log1p the audit
#      found (basic_qc_and_normalize log1p's adata.X, the gene encoder
#      logged it AGAIN, squashing dynamic range a second time).
#
# All three are opt-in with prior-behavior-preserving defaults (verified:
# 18/18 test files pass, default StormLite params byte-identical).
#
# Best result so far: StormLite+decoder+EMA seed5 = PCC 0.5391 (day3),
# now the bar to beat. STPath(pretrained)+decoder averages ~0.503 over 4
# seeds. This batch honors the user's requested structure (multisample
# StormLite / STPath-unfrozen / STPath-frozen + bigger StormLite + single-
# sample variants) while A/B-testing each new improvement against that bar.
#
# 8 jobs, GPUs 0-7:
#   MULTI-SAMPLE (INT1-INT8, decoder held constant = clean encoder
#   comparison, all +EMA, 40k):
#     0. StormLite mome_both + decoder + QK-norm
#     1. STPath UNFROZEN + decoder (no pretraining advantage)
#     2. STPath FROZEN (pretrained) + decoder
#   SINGLE-SAMPLE (all +EMA):
#     3. StormLite BIGGER + decoder + QK-norm + warmup + lower-lr, 80k —
#        THE test of whether QK-norm finally makes bigger capacity work
#        (gradient clipping alone failed twice; QK-norm targets the root
#        cause). Longer run since it's a capacity+length question.
#     4. StormLite flagship + decoder + QK-norm (seed 6, 40k) — does
#        QK-norm help even the SMALL model, or push the 0.5391 bar higher?
#     5. StormLite flagship + decoder + logit-normal t sampling (seed 7)
#     6. StormLite flagship + decoder + input_already_log1p (seed 8) —
#        the double-log1p A/B
#     7. StormLite flagship + decoder + QK-norm + logit-normal +
#        input_already_log1p, ALL STACKED (seed 9) — the "best model"
#        candidate combining every improvement that helps
#
# IMPORTANT: jobs 0-2 are multi-sample (cfg.data.sample_ids) via train.py's
# main() directly; jobs 3-7 single-sample via run_comparison.py — same
# completion-marker distinction as every earlier parallel script.
#
# Usage: bash scripts/run_parallel_8gpu_day4_overnight.sh
# Smoke test: SMOKETEST=1 bash scripts/run_parallel_8gpu_day4_overnight.sh
# Logs: logs/parallel_run_day4/<name>.log

set -u

SMOKETEST="${SMOKETEST:-0}"

# shared decoder + EMA + telemetry override fragments (the winning arm)
DEC="model.params.decoder_type=panel_invariant model.params.decoder_gene_embed_dim=256 model.params.decoder_combine_mode=add model.params.decoder_mlp_depth=2"
CK="training.checkpoint_every_n_steps=10000 training.log_print_every_n_steps=1000"
EMA="training.ema_decay=0.999"

MULTI_CONFIGS=(
    "configs/exp_multisample_fm_ot_stormlite_mome_both_paneldecoder_add.yaml"
    "configs/exp_multisample_fm_ot_stpath_bothresidual.yaml"
    "configs/exp_multisample_fm_ot_stpath_bothresidual.yaml"
)
MULTI_EXTRA_OVERRIDES=(
    "model.params.storm_lite_qk_norm=true training.epochs=40000 ${CK} ${EMA}"
    "model.params.stpath_pretrained=false ${DEC} training.epochs=40000 ${CK} ${EMA}"
    "${DEC} training.epochs=40000 ${CK} ${EMA}"
)
MULTI_NAMES=(
    "multisample_stormlite_mome_both_paneldecoder_add_qknorm_ema"
    "multisample_stpath_unfrozen_paneldecoder_add_ema"
    "multisample_stpath_frozen_paneldecoder_add_ema"
)

SINGLE_CONFIGS=(
    "configs/exp_hest1k_fm_ot_stormlite_mome_both_bigger_paneldecoder_add_warmupema.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder_add.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder_add.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder_add.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder_add.yaml"
)
SINGLE_EXTRA_OVERRIDES=(
    "model.params.storm_lite_qk_norm=true training.epochs=80000 ${CK} ${EMA}"
    "model.params.storm_lite_qk_norm=true training.epochs=40000 training.seed=6 ${CK} ${EMA}"
    "model.params.fm_time_sampling=logit_normal training.epochs=40000 training.seed=7 ${CK} ${EMA}"
    "model.params.storm_lite_input_already_log1p=true training.epochs=40000 training.seed=8 ${CK} ${EMA}"
    "model.params.storm_lite_qk_norm=true model.params.fm_time_sampling=logit_normal model.params.storm_lite_input_already_log1p=true training.epochs=40000 training.seed=9 ${CK} ${EMA}"
)
SINGLE_NAMES=(
    "stormlite_bigger_paneldecoder_add_qknorm_warmup_ema_80k"
    "stormlite_mome_both_paneldecoder_add_qknorm_seed6_ema"
    "stormlite_mome_both_paneldecoder_add_logitnormal_seed7_ema"
    "stormlite_mome_both_paneldecoder_add_nolog1p_seed8_ema"
    "stormlite_mome_both_paneldecoder_add_allimprovements_seed9_ema"
)

LOG_DIR="logs/parallel_run_day4"
EXTRA_ARGS="--shuffle-diagnostic"
SMOKE_OVERRIDE=""

if [ "$SMOKETEST" = "1" ]; then
    LOG_DIR="logs/parallel_run_day4_smoketest"
    SMOKE_OVERRIDE="training.epochs=10 training.checkpoint_every_n_steps=5 training.log_print_every_n_steps=3"
    echo "*** SMOKETEST=1 -- running tiny versions of all 8 jobs (epochs=10) to catch config/crash issues before the real run. Logs: $LOG_DIR ***"
fi

N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
TOTAL_JOBS=$((${#SINGLE_CONFIGS[@]} + ${#MULTI_CONFIGS[@]}))
N_CORES=$(nproc)
THREADS_PER_JOB=$((N_CORES / TOTAL_JOBS))
[ "$THREADS_PER_JOB" -lt 1 ] && THREADS_PER_JOB=1

if [ "$TOTAL_JOBS" -gt "$N_GPUS" ]; then
    echo "ERROR: $TOTAL_JOBS configs but only $N_GPUS GPUs visible — trim CONFIGS or this will double up on a GPU."
    exit 1
fi

mkdir -p "$LOG_DIR"
echo "Launching $TOTAL_JOBS jobs across GPUs 0-$((TOTAL_JOBS - 1)), ${THREADS_PER_JOB} CPU threads each, logs in $LOG_DIR..."

gpu=0
for i in "${!MULTI_CONFIGS[@]}"; do
    cfg="${MULTI_CONFIGS[$i]}"
    extra="${MULTI_EXTRA_OVERRIDES[$i]}"
    name="${MULTI_NAMES[$i]}"
    logfile="$LOG_DIR/${name}.log"
    if [ -f "$logfile" ] && grep -q "^mean PCC:" "$logfile"; then
        echo "  GPU $gpu: $cfg ($name) -> SKIPPING, already completed (see $logfile)"
        gpu=$((gpu + 1))
        continue
    fi
    echo "  GPU $gpu: $cfg ($name, multi-sample, train.py directly) -> $logfile"
    CUDA_VISIBLE_DEVICES=$gpu \
    OMP_NUM_THREADS=$THREADS_PER_JOB \
    MKL_NUM_THREADS=$THREADS_PER_JOB \
    python -m src.training.train --config "$cfg" \
        --override ${extra} ${SMOKE_OVERRIDE} \
        > "$logfile" 2>&1 &
    gpu=$((gpu + 1))
done

for i in "${!SINGLE_CONFIGS[@]}"; do
    cfg="${SINGLE_CONFIGS[$i]}"
    extra="${SINGLE_EXTRA_OVERRIDES[$i]}"
    name="${SINGLE_NAMES[$i]}"
    logfile="$LOG_DIR/${name}.log"
    if [ -f "$logfile" ] && grep -q "^model " "$logfile"; then
        echo "  GPU $gpu: $cfg ($name) -> SKIPPING, already completed (see $logfile)"
        gpu=$((gpu + 1))
        continue
    fi
    echo "  GPU $gpu: $cfg ($name, single-sample, run_comparison.py) -> $logfile"
    CUDA_VISIBLE_DEVICES=$gpu \
    OMP_NUM_THREADS=$THREADS_PER_JOB \
    MKL_NUM_THREADS=$THREADS_PER_JOB \
    python -m src.evaluation.run_comparison "$cfg" \
        --override ${extra} ${SMOKE_OVERRIDE} \
        ${EXTRA_ARGS} \
        > "$logfile" 2>&1 &
    gpu=$((gpu + 1))
done

n_launched=$(jobs -p | wc -l)
echo "$n_launched job(s) launched (PIDs: $(jobs -p | tr '\n' ' ')), $((TOTAL_JOBS - n_launched)) skipped as already-completed. Waiting for completion..."
wait
echo "All jobs finished. Check $LOG_DIR/*.log for each run's output."
echo ""
echo "Quick summary — multi-sample:"
for name in "${MULTI_NAMES[@]}"; do
    echo "--- $name ---"
    grep -A1 "^mean PCC:" "$LOG_DIR/${name}.log"
done
echo ""
echo "Quick summary — single-sample (real result only, grep -m1, not the shuffle-diagnostic row):"
for name in "${SINGLE_NAMES[@]}"; do
    echo "--- $name ---"
    grep -m1 -A2 "^model " "$LOG_DIR/${name}.log" | tail -2
done
