#!/bin/bash
# Second night's batch (2026-07-16) — same mechanism as
# run_parallel_8gpu.sh (see that script's own comments for the
# CUDA_VISIBLE_DEVICES-pinning / CPU-thread-oversubscription bugs this
# avoids), covering the features implemented AFTER the first overnight
# run was kicked off: organ/tech conditioning, relative-position
# attention bias (already inside stormlite configs), varied mask
# geometries, and coordinate augmentation.
#
# NOT included here: exp_hest1k_fm_ot_multisample.yaml — it uses
# cfg.data.sample_ids (a list), which src/evaluation/run_comparison.py's
# _train_model does NOT support yet (see that function's own docstring
# note, 2026-07-16). Run it directly instead:
#   python -m src.training.train --config configs/exp_hest1k_fm_ot_multisample.yaml
# (see run_multisample_demo.sh in this directory for a ready-to-go wrapper,
# and that config's own header for the real caveat: INT1-4 are all-Visium/
# same-cohort, so organ/tech conditioning is exercised as a mechanism
# smoke test here, not yet validated against real cross-organ signal).
#
# Usage: bash scripts/run_parallel_8gpu_night2.sh
# Logs: logs/parallel_run_night2/<name>.log

set -u

CONFIGS=(
    "configs/exp_hest1k_fm_ot_stpath_bothresidual.yaml"                # baseline (this project's best arm so far) — direct comparison anchor for the two below
    "configs/exp_hest1k_fm_ot_stpath_bothresidual_mixedmask.yaml"      # same model, mixed_dropout masking (item 3: varied hole shapes + sparse dropout)
    "configs/exp_hest1k_fm_ot_stpath_bothresidual_augment.yaml"        # same model, training.augment_coords: true (rotation/reflection augmentation)
)
EPOCHS=10000
EXTRA_ARGS="--shuffle-diagnostic"

N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
N_CORES=$(nproc)
THREADS_PER_JOB=$((N_CORES / ${#CONFIGS[@]}))
[ "$THREADS_PER_JOB" -lt 1 ] && THREADS_PER_JOB=1

if [ "${#CONFIGS[@]}" -gt "$N_GPUS" ]; then
    echo "ERROR: ${#CONFIGS[@]} configs but only $N_GPUS GPUs visible — trim CONFIGS or this will double up on a GPU."
    exit 1
fi

mkdir -p logs/parallel_run_night2
echo "Launching ${#CONFIGS[@]} jobs across GPUs 0-$((${#CONFIGS[@]}-1)), ${THREADS_PER_JOB} CPU threads each..."

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    name=$(basename "$cfg" .yaml)
    logfile="logs/parallel_run_night2/${name}.log"
    echo "  GPU $i: $cfg -> $logfile"
    CUDA_VISIBLE_DEVICES=$i \
    OMP_NUM_THREADS=$THREADS_PER_JOB \
    MKL_NUM_THREADS=$THREADS_PER_JOB \
    python -m src.evaluation.run_comparison "$cfg" \
        --override "training.epochs=${EPOCHS}" \
        ${EXTRA_ARGS} \
        > "$logfile" 2>&1 &
done

echo "All ${#CONFIGS[@]} jobs launched (PIDs: $(jobs -p | tr '\n' ' ')). Waiting for completion..."
wait
echo "All jobs finished. Check logs/parallel_run_night2/*.log for each model's table."
echo "Quick summary (last comparison table line per job):"
for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    echo "--- $name ---"
    grep -A2 "^model " "logs/parallel_run_night2/${name}.log" | tail -2
done
