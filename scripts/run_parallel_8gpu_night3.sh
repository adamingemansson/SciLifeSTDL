#!/bin/bash
# Third batch (2026-07-17) — same CUDA_VISIBLE_DEVICES-pinning / CPU-
# thread-capping pattern as run_parallel_8gpu.sh. Covers two things from
# today's investigation:
#
# 1. The "does a better gene encoder help even WITHOUT pretrained
#    weights?" factorial (user question, 2026-07-17) — the unfrozen
#    baseline plus all three Route-B residual modes applied ON TOP OF
#    stpath_pretrained: false, direct counterparts to the existing
#    mlpresidual/novaeresidual/bothresidual configs (which apply the same
#    residuals to the PRETRAINED fusion). See
#    exp_hest1k_fm_ot_stpath_unfrozen_mlpresidual.yaml's header for the
#    full reasoning.
#
# 2. A re-run of all three StormLite arms, now that three real bugs found
#    from yesterday's actual results are fixed: RelativePositionBias and
#    RandomFourierFeatures both silently produced near-random output on
#    real HEST-1k pixel-scale coordinates, AND CombinedGeneEncoder's
#    "both" mode had the same sum-vs-concat bug already fixed once for
#    STPath's residual (see conditioning.py/storm_lite_encoder.py's own
#    docstrings, commits 32c3e2b/1727b32/45ec5b7 — the last of those also
#    added FrameAveragingBias, STPath's real verified relative-position
#    mechanism, as StormLite's new default bias). Yesterday's run scored
#    every StormLite arm WORSE than a trivial interp_baseline; this checks
#    whether the fixes actually restore competitive performance.
#
# 3. The pretrained bothresidual config (this project's current best
#    result, PCC 0.4717 at 40000 epochs) as an 8th reference-anchor slot
#    — every unfrozen/StormLite arm above can be compared against it
#    directly within the same run's output.
#
# Usage: bash scripts/run_parallel_8gpu_night3.sh
# Logs: logs/parallel_run_night3_<epochs>ep/<name>.log — the directory
# name includes EPOCHS (2026-07-17) so re-running at a different epoch
# count (e.g. investigating the training-length-dependent instability
# found on 2026-07-17 — the bothresidual reference anchor scored 0.4717
# @ 40000 epochs, 0.3694 @ 10000, but only 0.0364 @ 20000, non-monotonic
# and unexplained by any code change traced so far) gets its OWN log
# directory instead of either colliding with the resume/skip logic below
# or silently overwriting a previous epoch count's results that are
# still useful to keep for comparison.
#
# Resumable: safe to re-run after an interrupted invocation (lost
# connection, killed job, etc.) — a config is SKIPPED if its log (for
# THIS EPOCHS value) already contains the final "model ..." metrics
# table line (proof run_comparison.py reached the end successfully), and
# (re-)RUN, from scratch, otherwise (covers both "never started" and
# "started but got cut off mid-training" — there's no partial-checkpoint
# resume in this codebase, so an interrupted job's only real option is a
# clean restart, which just means letting its log file get overwritten).
# To force a specific config to rerun even though it already completed
# at this epoch count, delete its log file first.

set -u

CONFIGS=(
    "configs/exp_hest1k_fm_ot_stpath_unfrozen.yaml"
    "configs/exp_hest1k_fm_ot_stpath_unfrozen_mlpresidual.yaml"
    "configs/exp_hest1k_fm_ot_stpath_unfrozen_novaeresidual.yaml"
    "configs/exp_hest1k_fm_ot_stpath_unfrozen_bothresidual.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_mlp.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_novae.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_both.yaml"
    "configs/exp_hest1k_fm_ot_stpath_bothresidual.yaml"   # PRETRAINED reference anchor — known-good result at THIS epoch count (0.3694 @ 10000, 0.4717 @ 40000) is the trust check for the whole run
)
EPOCHS=10000
LOG_DIR="logs/parallel_run_night3_${EPOCHS}ep"
EXTRA_ARGS="--shuffle-diagnostic"

N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
N_CORES=$(nproc)
THREADS_PER_JOB=$((N_CORES / ${#CONFIGS[@]}))
[ "$THREADS_PER_JOB" -lt 1 ] && THREADS_PER_JOB=1

if [ "${#CONFIGS[@]}" -gt "$N_GPUS" ]; then
    echo "ERROR: ${#CONFIGS[@]} configs but only $N_GPUS GPUs visible — trim CONFIGS or this will double up on a GPU."
    exit 1
fi

mkdir -p "$LOG_DIR"
echo "Launching ${#CONFIGS[@]} jobs across GPUs 0-$((${#CONFIGS[@]}-1)), ${THREADS_PER_JOB} CPU threads each, logs in $LOG_DIR..."

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    name=$(basename "$cfg" .yaml)
    logfile="$LOG_DIR/${name}.log"
    if [ -f "$logfile" ] && grep -q "^model " "$logfile"; then
        echo "  GPU $i: $cfg -> SKIPPING, already completed (see $logfile)"
        continue
    fi
    echo "  GPU $i: $cfg -> $logfile"
    CUDA_VISIBLE_DEVICES=$i \
    OMP_NUM_THREADS=$THREADS_PER_JOB \
    MKL_NUM_THREADS=$THREADS_PER_JOB \
    python -m src.evaluation.run_comparison "$cfg" \
        --override "training.epochs=${EPOCHS}" \
        ${EXTRA_ARGS} \
        > "$logfile" 2>&1 &
done

n_launched=$(jobs -p | wc -l)
echo "$n_launched job(s) launched (PIDs: $(jobs -p | tr '\n' ' ')), $((${#CONFIGS[@]} - n_launched)) skipped as already-completed. Waiting for completion..."
wait
echo "All jobs finished. Check $LOG_DIR/*.log for each model's table."
echo "Quick summary (last comparison table line per job):"
for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    echo "--- $name ---"
    grep -A2 "^model " "$LOG_DIR/${name}.log" | tail -2
done
