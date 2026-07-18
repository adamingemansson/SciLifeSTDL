#!/bin/bash
# Day 2 batch (2026-07-19) — follow-up to today's day1 results (see
# docs/results_log.md for the full writeup once added). Two findings drove
# this batch:
#
#   1. The panel_invariant/add decoder swap is the ONLY lever that's
#      actually worked so far. 4 data points at various seeds/epochs all
#      landed in PCC 0.45-0.49 (0.4933 @ seed0/40k, 0.4681 @ 80k, 0.4501 @
#      seed1, 0.4557 novae-only) — a huge, reliable win over StormLite's
#      own dense-decoder baseline (~0.28-0.35), but NOT a clean, confident
#      "beats STPath" claim yet: only seed0/40k clearly beat STPath's
#      0.4717, the others landed at-or-below it. Needs more seeds to average
#      over before that claim is safe.
#   2. "Bigger" StormLite capacity (4L/8H/512d) is DROPPED from today's
#      batch — it mode-collapsed (PCC=nan) in single-sample AND
#      multi-sample overnight, and gradient_clip_val=1.0 did NOT fix the
#      multi-sample case today (RMSE got WORSE, 0.5247 vs 0.2745) — not
#      worth another GPU-slot chasing it blind. The decoder swap DID
#      separately rescue multi-sample StormLite's dense-decoder collapse
#      (nan -> real PCC 0.1361) — a different, more promising thread.
#
# 8 jobs, GPUs 0-7, ALL built around the decoder swap:
#   0. STPath + panel_invariant/add decoder (single, 40k) — the direct,
#      decoder-held-constant comparison: does the SAME decoder swap that
#      helped StormLite also help STPath, isolating the encoder-choice
#      question from the decoder-choice question? (config already existed,
#      just bumped to 40k earlier today)
#   1-3. StormLite + decoder, seeds 2/3/4 (single, 40k, via --override on
#      the base winning config) — building toward a real seed-averaged PCC
#      (have seed0=0.4933, seed1=0.4501 already; this adds 3 more) before
#      claiming StormLite reliably beats STPath.
#   4. STPath + decoder, seed=1 (single, 40k, via --override) — matched
#      seed-variance check on STPath's OWN decoder-swap number, so the
#      StormLite-vs-STPath comparison can be an apples-to-apples averaged
#      comparison, not single-seed noise vs single-seed noise.
#   5. multisample StormLite mome_both + decoder, pushed to 80000 epochs
#      (was 40k -> PCC 0.1361, no longer collapsed) — does more training
#      help now that the decoder swap fixed the collapse?
#   6. multisample STPath bothresidual + decoder swapped in (via
#      --override, no dedicated file existed yet) — does the SAME decoder
#      that rescued multi-sample StormLite also help multi-sample STPath
#      (currently PCC 0.1887 with the dense decoder)?
#   7. multisample StormLite mome_novae (novae-only, not "both") + decoder
#      swapped in (via --override) — isolates the decoder-rescue effect
#      from the "both" gene encoder specifically.
#
# IMPORTANT: jobs 5-7 are multi-sample (cfg.data.sample_ids) and go through
# train.py's OWN main() directly, NOT run_comparison.py — same
# reasoning/completion-marker distinction as every earlier parallel script.
#
# sample_ids in the multi-sample configs assume INT1-INT8 are downloaded
# locally — same caveat as every earlier script.
#
# All 8 jobs checkpoint every 10000 steps AND print real loss telemetry
# every 1000 steps (training.checkpoint_every_n_steps /
# training.log_print_every_n_steps).
#
# Usage: bash scripts/run_parallel_8gpu_day2.sh
# Smoke test: SMOKETEST=1 bash scripts/run_parallel_8gpu_day2.sh
# Logs: logs/parallel_run_day2/<name>.log

set -u

SMOKETEST="${SMOKETEST:-0}"

SINGLE_CONFIGS=(
    "configs/exp_hest1k_fm_ot_stpath_bothresidual_paneldecoder_add.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder_add.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder_add.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder_add.yaml"
    "configs/exp_hest1k_fm_ot_stpath_bothresidual_paneldecoder_add.yaml"
)
SINGLE_EXTRA_OVERRIDES=(
    ""
    "training.epochs=40000 training.seed=2 training.checkpoint_every_n_steps=10000 training.log_print_every_n_steps=1000"
    "training.epochs=40000 training.seed=3 training.checkpoint_every_n_steps=10000 training.log_print_every_n_steps=1000"
    "training.epochs=40000 training.seed=4 training.checkpoint_every_n_steps=10000 training.log_print_every_n_steps=1000"
    "training.seed=1"
)
# distinct log/checkpoint names for the 3 repeated-config jobs above (same
# cfg path used more than once with different overrides — basename alone
# would collide on the same logfile/skip-check otherwise)
SINGLE_NAMES=(
    "stpath_bothresidual_paneldecoder_add_seed0"
    "stormlite_mome_both_paneldecoder_add_seed2"
    "stormlite_mome_both_paneldecoder_add_seed3"
    "stormlite_mome_both_paneldecoder_add_seed4"
    "stpath_bothresidual_paneldecoder_add_seed1"
)

MULTI_CONFIGS=(
    "configs/exp_multisample_fm_ot_stormlite_mome_both_paneldecoder_add.yaml"
    "configs/exp_multisample_fm_ot_stpath_bothresidual.yaml"
    "configs/exp_multisample_fm_ot_stormlite_mome_novae.yaml"
)
MULTI_EXTRA_OVERRIDES=(
    "training.epochs=80000 training.checkpoint_every_n_steps=10000 training.log_print_every_n_steps=1000"
    "model.params.decoder_type=panel_invariant model.params.decoder_gene_embed_dim=256 model.params.decoder_combine_mode=add model.params.decoder_mlp_depth=2 training.epochs=40000 training.checkpoint_every_n_steps=10000 training.log_print_every_n_steps=1000"
    "model.params.decoder_type=panel_invariant model.params.decoder_gene_embed_dim=256 model.params.decoder_combine_mode=add model.params.decoder_mlp_depth=2 training.epochs=40000 training.checkpoint_every_n_steps=10000 training.log_print_every_n_steps=1000"
)
MULTI_NAMES=(
    "multisample_stormlite_mome_both_paneldecoder_add_80k"
    "multisample_stpath_bothresidual_paneldecoder_add"
    "multisample_stormlite_mome_novae_paneldecoder_add"
)

LOG_DIR="logs/parallel_run_day2"
EXTRA_ARGS="--shuffle-diagnostic"
SMOKE_OVERRIDE=""

if [ "$SMOKETEST" = "1" ]; then
    LOG_DIR="logs/parallel_run_day2_smoketest"
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

n_launched=$(jobs -p | wc -l)
echo "$n_launched job(s) launched (PIDs: $(jobs -p | tr '\n' ' ')), $((TOTAL_JOBS - n_launched)) skipped as already-completed. Waiting for completion..."
wait
echo "All jobs finished. Check $LOG_DIR/*.log for each run's output."
echo ""
echo "Quick summary — single-sample (real result only, grep -m1, not the shuffle-diagnostic row):"
for name in "${SINGLE_NAMES[@]}"; do
    echo "--- $name ---"
    grep -m1 -A2 "^model " "$LOG_DIR/${name}.log" | tail -2
done
echo ""
echo "Quick summary — multi-sample:"
for name in "${MULTI_NAMES[@]}"; do
    echo "--- $name ---"
    grep -A1 "^mean PCC:" "$LOG_DIR/${name}.log"
done
