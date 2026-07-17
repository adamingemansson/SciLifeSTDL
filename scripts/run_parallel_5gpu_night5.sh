#!/bin/bash
# Fifth batch (2026-07-17) — fills real gaps left over from tonight's work,
# on the 5 GPUs NOT already busy with run_parallel_3gpu_mome_40k.sh (GPUs
# 0-2). Pins explicitly to GPUs 3-7 rather than the usual 0-indexed loop,
# so this can run alongside the 40k job without colliding.
#
# Same CUDA_VISIBLE_DEVICES-pinning / CPU-thread-capping / resumable /
# corrected-grep pattern as every other parallel script in this directory
# (see commit 5d519fc for why -m1 matters — every log has TWO "model "
# tables when --shuffle-diagnostic is on, and only the FIRST is real).
#
# Five configs, five different open questions:
#   1. stormlite_mome_both_paneldecoder — does swapping in the new
#      PanelInvariantGeneDecoder (panel-invariant, diagram-5's "target
#      platform space" box) cost anything vs. the dense decoder, on the
#      same-panel sanity case? Direct comparison: stormlite_mome_both.
#   2. wae_gan_stormlite_mome_both   } does the MoME-FFN fix that rescued
#   3. vqvae_ar_stormlite_mome_both  } StormLite for FM-OT generalize to
#      the other two generator families, or is it FM-OT-specific?
#   4. stpath_unfrozen_mlpresidual   } fill in the from-scratch STPath +
#   5. stpath_unfrozen_novaeresidual } Route-B residual sweep — only
#      bothresidual has a confirmed 10k/20k/40k result so far
#      (run_parallel_8gpu_night3.sh/_night4.sh); these two single-residual
#      arms complete the same 4-way comparison already run for the
#      PRETRAINED stpath_bothresidual family.
#
# Usage: bash scripts/run_parallel_5gpu_night5.sh
# Logs: logs/parallel_run_night5_10000ep/<name>.log

set -u

CONFIGS=(
    "configs/exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder.yaml"
    "configs/exp_hest1k_wae_gan_stormlite_mome_both.yaml"
    "configs/exp_hest1k_vqvae_ar_stormlite_mome_both.yaml"
    "configs/exp_hest1k_fm_ot_stpath_unfrozen_mlpresidual.yaml"
    "configs/exp_hest1k_fm_ot_stpath_unfrozen_novaeresidual.yaml"
)
GPU_IDS=(3 4 5 6 7)
EPOCHS=10000
LOG_DIR="logs/parallel_run_night5_${EPOCHS}ep"
EXTRA_ARGS="--shuffle-diagnostic"

N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
N_CORES=$(nproc)
THREADS_PER_JOB=$((N_CORES / ${#CONFIGS[@]}))
[ "$THREADS_PER_JOB" -lt 1 ] && THREADS_PER_JOB=1

if [ "${#CONFIGS[@]}" -ne "${#GPU_IDS[@]}" ]; then
    echo "ERROR: ${#CONFIGS[@]} configs but ${#GPU_IDS[@]} GPU_IDS entries — these must match 1:1."
    exit 1
fi
for gid in "${GPU_IDS[@]}"; do
    if [ "$gid" -ge "$N_GPUS" ]; then
        echo "ERROR: GPU index $gid requested but only $N_GPUS GPUs visible."
        exit 1
    fi
done

mkdir -p "$LOG_DIR"
echo "Launching ${#CONFIGS[@]} jobs on GPUs ${GPU_IDS[*]}, ${THREADS_PER_JOB} CPU threads each, logs in $LOG_DIR..."

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    gid="${GPU_IDS[$i]}"
    name=$(basename "$cfg" .yaml)
    logfile="$LOG_DIR/${name}.log"
    if [ -f "$logfile" ] && grep -q "^model " "$logfile"; then
        echo "  GPU $gid: $cfg -> SKIPPING, already completed (see $logfile)"
        continue
    fi
    echo "  GPU $gid: $cfg -> $logfile"
    CUDA_VISIBLE_DEVICES=$gid \
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
