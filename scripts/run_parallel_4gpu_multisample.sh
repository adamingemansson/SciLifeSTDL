#!/bin/bash
# First real multi-sample overnight run (2026-07-17). Uses
# `python -m src.training.train`, NOT run_comparison.py — multi-sample
# training (_main_multi_sample, cfg.data.sample_ids) is a separate code
# path run_comparison.py explicitly doesn't support yet (see its own
# _train_model docstring). Different completion/output format too:
# _main_multi_sample prints "mean PCC: X.XXXX" / "RMSE: X.XXXX", not
# run_comparison.py's "model  pcc  rmse  ..." table row — the resumable
# skip-check below matches THAT format, not the usual "^model " pattern
# every other script in this directory uses.
#
# 4 configs, GPUs 0-3, weighted toward StormLite per current research
# priority:
#   1. multisample_fm_ot_stormlite_mome_both — flagship StormLite arm
#   2. multisample_fm_ot_stormlite_mome_novae — StormLite's Novae-only branch
#   3. multisample_fm_ot_stpath_bothresidual — comparison anchor (best
#      single-sample result overall)
#   4. multisample_wae_gan_stpath_novaeresidual — best single-sample
#      WAE-GAN result, checking whether it holds with more data diversity
#
# Real gap this run depends on being fixed (see docs/results_log.md's
# 2026-07-17 entries): multi-sample training previously could not use
# images/Novae/STPath at all (load_multi_sample_data always passed
# images=None and never computed Novae features) — now fixed, this is the
# first real exercise of that fix on real data.
#
# sample_ids in every config below assumes INT1-INT8 are downloaded
# locally — check with `ls data/raw/hest1k/` BEFORE launching and edit
# each config's sample_ids list to match what's actually present. A
# missing sample raises a clear FileNotFoundError naming it, not a silent
# skip or a wrong result — if that happens, trim the list and rerun.
#
# Usage: bash scripts/run_parallel_4gpu_multisample.sh
# Logs: logs/parallel_run_multisample/<name>.log
#
# Resumable: safe to re-run after an interrupted invocation — a config is
# skipped if its log already contains the final "mean PCC:" line.

set -u

CONFIGS=(
    "configs/exp_multisample_fm_ot_stormlite_mome_both.yaml"
    "configs/exp_multisample_fm_ot_stormlite_mome_novae.yaml"
    "configs/exp_multisample_fm_ot_stpath_bothresidual.yaml"
    "configs/exp_multisample_wae_gan_stpath_novaeresidual.yaml"
)

N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
N_CORES=$(nproc)
THREADS_PER_JOB=$((N_CORES / ${#CONFIGS[@]}))
[ "$THREADS_PER_JOB" -lt 1 ] && THREADS_PER_JOB=1

if [ "${#CONFIGS[@]}" -gt "$N_GPUS" ]; then
    echo "ERROR: ${#CONFIGS[@]} configs but only $N_GPUS GPUs visible — trim CONFIGS or this will double up on a GPU."
    exit 1
fi

LOG_DIR="logs/parallel_run_multisample"
mkdir -p "$LOG_DIR"
echo "Launching ${#CONFIGS[@]} jobs across GPUs 0-$((${#CONFIGS[@]}-1)), ${THREADS_PER_JOB} CPU threads each, logs in $LOG_DIR..."

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    name=$(basename "$cfg" .yaml)
    logfile="$LOG_DIR/${name}.log"
    if [ -f "$logfile" ] && grep -q "^mean PCC:" "$logfile"; then
        echo "  GPU $i: $cfg -> SKIPPING, already completed (see $logfile)"
        continue
    fi
    echo "  GPU $i: $cfg -> $logfile"
    CUDA_VISIBLE_DEVICES=$i \
    OMP_NUM_THREADS=$THREADS_PER_JOB \
    MKL_NUM_THREADS=$THREADS_PER_JOB \
    python -m src.training.train --config "$cfg" \
        > "$logfile" 2>&1 &
done

n_launched=$(jobs -p | wc -l)
echo "$n_launched job(s) launched (PIDs: $(jobs -p | tr '\n' ' ')), $((${#CONFIGS[@]} - n_launched)) skipped as already-completed. Waiting for completion..."
wait
echo "All jobs finished. Check $LOG_DIR/*.log for each model's mean PCC/RMSE."
echo "Quick summary:"
for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    echo "--- $name ---"
    grep -E "^mean PCC:|^RMSE:|load_multi_sample:|Traceback" "$LOG_DIR/${name}.log" | tail -5
done
