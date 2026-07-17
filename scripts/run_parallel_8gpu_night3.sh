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
# 2. A re-run of all three StormLite arms, now that two real bugs found
#    from yesterday's actual results are fixed (RelativePositionBias and
#    RandomFourierFeatures both silently produced near-random output on
#    real HEST-1k pixel-scale coordinates — see conditioning.py's own
#    docstrings on each, commits 32c3e2b/1727b32). Yesterday's run scored
#    every StormLite arm WORSE than a trivial interp_baseline; this checks
#    whether the fix actually restores competitive performance.
#
# Usage: bash scripts/run_parallel_8gpu_night3.sh
# Logs: logs/parallel_run_night3/<name>.log

set -u

CONFIGS=(
    "configs/exp_hest1k_fm_ot_stpath_unfrozen.yaml"
    "configs/exp_hest1k_fm_ot_stpath_unfrozen_mlpresidual.yaml"
    "configs/exp_hest1k_fm_ot_stpath_unfrozen_novaeresidual.yaml"
    "configs/exp_hest1k_fm_ot_stpath_unfrozen_bothresidual.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_mlp.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_novae.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_both.yaml"
)
EPOCHS=40000
EXTRA_ARGS="--shuffle-diagnostic"

N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
N_CORES=$(nproc)
THREADS_PER_JOB=$((N_CORES / ${#CONFIGS[@]}))
[ "$THREADS_PER_JOB" -lt 1 ] && THREADS_PER_JOB=1

if [ "${#CONFIGS[@]}" -gt "$N_GPUS" ]; then
    echo "ERROR: ${#CONFIGS[@]} configs but only $N_GPUS GPUs visible — trim CONFIGS or this will double up on a GPU."
    exit 1
fi

mkdir -p logs/parallel_run_night3
echo "Launching ${#CONFIGS[@]} jobs across GPUs 0-$((${#CONFIGS[@]}-1)), ${THREADS_PER_JOB} CPU threads each..."

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    name=$(basename "$cfg" .yaml)
    logfile="logs/parallel_run_night3/${name}.log"
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
echo "All jobs finished. Check logs/parallel_run_night3/*.log for each model's table."
echo "Quick summary (last comparison table line per job):"
for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    echo "--- $name ---"
    grep -A2 "^model " "logs/parallel_run_night3/${name}.log" | tail -2
done
