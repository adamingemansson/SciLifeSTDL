#!/bin/bash
# Fourth batch (2026-07-17) — decisive StormLite architecture comparison.
# Same CUDA_VISIBLE_DEVICES-pinning / CPU-thread-capping pattern as every
# other parallel script in this directory.
#
# Purpose: every StormLite arm has scored at/below interp_baseline so far
# (PCC roughly -0.01 to +0.01) even after fixing two real coordinate-
# scale bugs (RelativePositionBias, RandomFourierFeatures — commits
# 32c3e2b/1727b32) and the CombinedGeneEncoder sum-vs-concat issue
# (45ec5b7) — this run tests today's two further changes side-by-side
# against the un-fixed baseline and against this project's actual best
# result, all in one comparison:
#
#   - fusion_mode="sum" (original, now also gets per-branch LayerNorm
#     before combining, applied unconditionally) x 3 gene_encoder_type
#   - fusion_mode="mome" (real STORM detail: shared attention, per-
#     modality FFN experts — see _MoMETransformerBlock's own docstring)
#     x 3 gene_encoder_type
#   - stpath_bothresidual (pretrained) as the reference anchor — this
#     project's actual best result (PCC 0.42-0.47 across 10k/20k/40k
#     epochs, confirmed reproducible after fixing the summary-script bug
#     that made it LOOK like it had regressed — see commit 5d519fc)
#   - stpath_unfrozen_bothresidual as a second anchor — same residual,
#     but from-scratch, the fairer "no pretrained weights" comparison
#     point for StormLite (which is also fully from-scratch)
#
# IMPORTANT: the summary printed by THIS script correctly reads only the
# real (first) "model ..." table now — see commit 5d519fc for the bug
# that made every earlier parallel script's summary print the shuffle-
# diagnostic row instead. Still worth spot-checking the full per-config
# log if a number looks surprising.
#
# Usage: bash scripts/run_parallel_8gpu_night4.sh
# Logs: logs/parallel_run_night4/<name>.log
#
# Resumable: safe to re-run after an interrupted invocation — a config is
# skipped if its log already contains the final "model ..." table line.

set -u

CONFIGS=(
    "configs/exp_hest1k_fm_ot_stormlite_mlp.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_novae.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_both.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_mome_mlp.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_mome_novae.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_mome_both.yaml"
    "configs/exp_hest1k_fm_ot_stpath_bothresidual.yaml"
    "configs/exp_hest1k_fm_ot_stpath_unfrozen_bothresidual.yaml"
)
EPOCHS=10000
LOG_DIR="logs/parallel_run_night4_${EPOCHS}ep"
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
echo "Quick summary (real result only — NOT the shuffle-diagnostic row, see header):"
for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    echo "--- $name ---"
    grep -m1 -A2 "^model " "$LOG_DIR/${name}.log" | tail -2
done
