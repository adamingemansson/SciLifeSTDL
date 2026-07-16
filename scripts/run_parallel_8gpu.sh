#!/bin/bash
# Launch up to 8 configs simultaneously, one per GPU, all in one terminal.
#
# Real bug this avoids (2026-07-16, this session): running with all GPUs
# visible and no CUDA_VISIBLE_DEVICES pin makes PyTorch Lightning
# auto-launch DDP, which re-executes the whole script per GPU — a real
# 8-GPU run crashed (SIGSEGV) from concurrent processes racing to write
# the same feature-cache file. Pinning each process to exactly ONE GPU
# (CUDA_VISIBLE_DEVICES=$i) avoids DDP entirely — each process is fully
# independent, no shared launcher, no race on that front (the cache
# writes themselves are now also atomic — see train.py _atomic_savez —
# so this is safe even if multiple processes are the FIRST to populate
# a shared cache simultaneously, just wastes some redundant compute).
#
# Second real bug this avoids: PyTorch's default CPU threading (OMP/MKL)
# tries to use ALL cores per process by default — with 8 simultaneous
# processes on a shared server, that's severe oversubscription (this
# session hit the same problem with just 3 processes on a 256-core
# machine). OMP_NUM_THREADS/MKL_NUM_THREADS below caps each process to a
# fair share of the machine's cores.
#
# Usage: edit CONFIGS below (up to 8 entries, one per GPU), then:
#   bash scripts/run_parallel_8gpu.sh
# Each job's full stdout/stderr goes to logs/parallel_run/<name>.log —
# tail -f logs/parallel_run/*.log to watch progress, or just wait for
# this script to print "all jobs finished".

set -u

CONFIGS=(
    "configs/exp_hest1k_fm_ot_stpath.yaml"
    "configs/exp_hest1k_fm_ot_stpath_mlpresidual.yaml"
    "configs/exp_hest1k_fm_ot_stpath_novaeresidual.yaml"
    "configs/exp_hest1k_fm_ot_stpath_bothresidual.yaml"
    "configs/exp_hest1k_fm_ot_stpath_unfrozen.yaml"
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

mkdir -p logs/parallel_run
echo "Launching ${#CONFIGS[@]} jobs across GPUs 0-$((${#CONFIGS[@]}-1)), ${THREADS_PER_JOB} CPU threads each..."

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    name=$(basename "$cfg" .yaml)
    logfile="logs/parallel_run/${name}.log"
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
echo "All jobs finished. Check logs/parallel_run/*.log for each model's table."
echo "Quick summary (last comparison table line per job):"
for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    echo "--- $name ---"
    grep -A2 "^model " "logs/parallel_run/${name}.log" | tail -2
done
