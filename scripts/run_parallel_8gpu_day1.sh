#!/bin/bash
# Weekend daytime batch (2026-07-19) — follow-up to last night's overnight
# batch. Two real findings from that run drive today's jobs (see
# docs/results_log.md's 2026-07-18/19 entry for the full writeup):
#
#   1. StormLite + panel_invariant/add decoder is the new project best,
#      beating STPath outright: PCC 0.4933 vs STPath's 0.4717
#      (exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder_add.yaml @ 40k).
#      Training length (80k) and raw capacity (bigger StormLite) alone did
#      NOT close the gap — the decoder swap is the real lever.
#   2. Bigger StormLite capacity (4 layers/8 heads/512-dim) mode-collapsed
#      to a constant output (PCC=nan, ConstantInputWarning from pearsonr)
#      in BOTH single- and all 3 multi-sample settings, while the smaller
#      default StormLite trained fine. Root cause: no trainer in this
#      codebase clipped gradients before today. Fixed via
#      gradient_clip_val=1.0 (src/training/train.py, all 3 Trainer()
#      sites) — today's batch verifies the fix actually resolves it.
#
# 8 jobs, GPUs 0-7:
#   0. mome_both_bigger (single) — re-run with the gradient-clip fix,
#      verify the single-sample collapse is actually resolved
#   1. multisample mome_both_bigger (flagship) — same fix, verify at
#      multi-sample scale
#   2. mome_both_bigger_paneldecoder_add (single, NEW) — combines both
#      real wins (bigger capacity + the winning decoder) — the actual
#      best shot at beating tonight's 0.4933 further
#   3. mome_both_paneldecoder_add pushed to 80k epochs (single, NEW) —
#      does the winning decoder keep climbing, unlike plain mome_both
#      (which regressed 0.3461->0.3329 from 40k->80k)?
#   4. multisample mome_both_paneldecoder_add (NEW) — does the winning
#      decoder rescue multi-sample StormLite the way it helped
#      single-sample, or was multi-sample StormLite's collapse a
#      SEPARATE issue from the decoder gap?
#   5. multisample stpath bothresidual — re-run with gradient clipping,
#      CONTROL test: is STPath's own multi-sample degradation (PCC 0.12
#      vs single-sample 0.47) partly an optimization-instability issue
#      too, or purely architectural/data-heterogeneity?
#   6. mome_novae_paneldecoder_add (single, NEW) — winning decoder with
#      gene_encoder_type="novae" instead of "both" — isolates whether the
#      decoder swap is the dominant lever regardless of gene-encoder choice
#   7. mome_both_paneldecoder_add_seed1 (single, NEW) — same winning
#      config, only training.seed differs (0->1) — is PCC 0.4933
#      reproducible, or a lucky draw?
#
# IMPORTANT: jobs 1, 4, 5 are multi-sample (cfg.data.sample_ids) and go
# through train.py's OWN main() directly, NOT run_comparison.py — same
# reasoning/completion-marker distinction as the overnight script (see
# its own header comment).
#
# sample_ids in the multi-sample configs assume INT1-INT8 are downloaded
# locally — same caveat as the overnight script.
#
# All 8 jobs checkpoint every 10000 steps AND print real loss telemetry
# every 1000 steps (training.log_print_every_n_steps, PeriodicPrintCallback
# — added today specifically because last night's collapsed run left NO
# per-step loss values in its log at all, making it impossible to tell
# WHEN training degenerated; logger=False disables Lightning's logger and
# tqdm's progress bar auto-disables on non-tty redirected output).
#
# Usage: bash scripts/run_parallel_8gpu_day1.sh
# Smoke test: SMOKETEST=1 bash scripts/run_parallel_8gpu_day1.sh
# Logs: logs/parallel_run_day1/<name>.log

set -u

SMOKETEST="${SMOKETEST:-0}"

SINGLE_CONFIGS=(
    "configs/exp_hest1k_fm_ot_stormlite_mome_both_bigger.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_mome_both_bigger_paneldecoder_add.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder_add_80k.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_mome_novae_paneldecoder_add.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder_add_seed1.yaml"
)
MULTI_CONFIGS=(
    "configs/exp_multisample_fm_ot_stormlite_mome_both_bigger.yaml"
    "configs/exp_multisample_fm_ot_stormlite_mome_both_paneldecoder_add.yaml"
    "configs/exp_multisample_fm_ot_stpath_bothresidual.yaml"
)

LOG_DIR="logs/parallel_run_day1"
EXTRA_ARGS="--shuffle-diagnostic"
SMOKE_OVERRIDE=""

if [ "$SMOKETEST" = "1" ]; then
    LOG_DIR="logs/parallel_run_day1_smoketest"
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
for cfg in "${SINGLE_CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    logfile="$LOG_DIR/${name}.log"
    if [ -f "$logfile" ] && grep -q "^model " "$logfile"; then
        echo "  GPU $gpu: $cfg -> SKIPPING, already completed (see $logfile)"
        gpu=$((gpu + 1))
        continue
    fi
    echo "  GPU $gpu: $cfg (single-sample, run_comparison.py) -> $logfile"
    CUDA_VISIBLE_DEVICES=$gpu \
    OMP_NUM_THREADS=$THREADS_PER_JOB \
    MKL_NUM_THREADS=$THREADS_PER_JOB \
    python -m src.evaluation.run_comparison "$cfg" \
        --override ${SMOKE_OVERRIDE} \
        ${EXTRA_ARGS} \
        > "$logfile" 2>&1 &
    gpu=$((gpu + 1))
done

for cfg in "${MULTI_CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    logfile="$LOG_DIR/${name}.log"
    if [ -f "$logfile" ] && grep -q "^mean PCC:" "$logfile"; then
        echo "  GPU $gpu: $cfg -> SKIPPING, already completed (see $logfile)"
        gpu=$((gpu + 1))
        continue
    fi
    echo "  GPU $gpu: $cfg (multi-sample, train.py directly) -> $logfile"
    CUDA_VISIBLE_DEVICES=$gpu \
    OMP_NUM_THREADS=$THREADS_PER_JOB \
    MKL_NUM_THREADS=$THREADS_PER_JOB \
    python -m src.training.train --config "$cfg" \
        --override ${SMOKE_OVERRIDE} \
        > "$logfile" 2>&1 &
    gpu=$((gpu + 1))
done

n_launched=$(jobs -p | wc -l)
echo "$n_launched job(s) launched (PIDs: $(jobs -p | tr '\n' ' ')), $((TOTAL_JOBS - n_launched)) skipped as already-completed. Waiting for completion..."
wait
echo "All jobs finished. Check $LOG_DIR/*.log for each run's output."
echo ""
echo "Quick summary — single-sample (real result only, grep -m1, not the shuffle-diagnostic row):"
for cfg in "${SINGLE_CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    echo "--- $name ---"
    grep -m1 -A2 "^model " "$LOG_DIR/${name}.log" | tail -2
done
echo ""
echo "Quick summary — multi-sample:"
for cfg in "${MULTI_CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    echo "--- $name ---"
    grep -A1 "^mean PCC:" "$LOG_DIR/${name}.log"
done
