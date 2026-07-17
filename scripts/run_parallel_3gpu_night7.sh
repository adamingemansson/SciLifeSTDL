#!/bin/bash
# Seventh batch (2026-07-17) — GPUs 0-2 just freed up (mome_40k finished).
# Closes a real gap: STPath's Route-B "both" residual (MLPGeneEncoder +
# frozen Novae, concat-mode) has only ever been tested on FM-OT, where
# it's this project's best confirmed result (bothresidual: PCC 0.41-0.47
# across 10k/20k/40k epochs). Never checked whether it helps WAE-GAN or
# VQ-VAE+AR the same way.
#
# Also includes a control: a plain (no-residual, no StormLite)
# wae_gan_stpath rerun. Night5/night6 are separately investigating a real
# WAE-GAN+StormLite mode collapse (PCC=nan, constant output) — this tells
# us whether WAE-GAN is unstable with pretrained STPath too (a WAE-GAN-
# family issue) or only with StormLite's added complexity (a StormLite-
# specific issue), independent of whatever night6 finds.
#
# Same CUDA_VISIBLE_DEVICES-pinning / CPU-thread-capping / resumable /
# corrected-grep pattern as every other parallel script in this directory.
#
# Usage: bash scripts/run_parallel_3gpu_night7.sh
# Logs: logs/parallel_run_night7_10000ep/<name>.log

set -u

CONFIGS=(
    "configs/exp_hest1k_wae_gan_stpath_bothresidual.yaml"
    "configs/exp_hest1k_vqvae_ar_stpath_bothresidual.yaml"
    "configs/exp_hest1k_wae_gan_stpath.yaml"
)
EPOCHS=10000
LOG_DIR="logs/parallel_run_night7_${EPOCHS}ep"
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
echo "Quick summary (real result only — grep -m1, not the shuffle-diagnostic row):"
for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    echo "--- $name ---"
    grep -m1 -A2 "^model " "$LOG_DIR/${name}.log" | tail -2
done
