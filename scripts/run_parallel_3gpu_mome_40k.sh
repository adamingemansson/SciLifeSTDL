#!/bin/bash
# Follow-up to run_parallel_8gpu_night4.sh (2026-07-17) — the three
# fusion_mode="mome" StormLite arms at 40000 epochs, on 3 GPUs.
#
# Motivation: at 10000 epochs, MoME-FFN rescued StormLite from
# near-zero/negative PCC (every sum-mode arm) to genuinely competitive
# results (mlp=0.1694, novae=0.2962, both=0.2798), with mome_both already
# beating the pretrained stpath_bothresidual anchor on ST-FID (1.30 vs
# 3.63) despite trailing it on PCC (0.28-0.30 vs 0.41). Separately,
# stpath_bothresidual itself went from 0.37-0.41 at 10-20k epochs to
# 0.4717 at 40k — this checks whether the MoME arms show the same
# training-length-dependent improvement, which would meaningfully close
# (or close) the gap with the pretrained anchor.
#
# Same CUDA_VISIBLE_DEVICES-pinning / CPU-thread-capping / resumable
# pattern as every other parallel script in this directory.
#
# Usage: bash scripts/run_parallel_3gpu_mome_40k.sh
# Logs: logs/parallel_run_mome40k/<name>.log

set -u

CONFIGS=(
    "configs/exp_hest1k_fm_ot_stormlite_mome_mlp.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_mome_novae.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_mome_both.yaml"
)
EPOCHS=40000
LOG_DIR="logs/parallel_run_mome40k"
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
