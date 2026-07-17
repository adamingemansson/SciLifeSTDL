#!/bin/bash
# Sixth batch (2026-07-17) — follow-up to night5's results. Five jobs, five
# open questions raised by that run, maximizing information extracted per
# GPU-hour rather than re-running anything that already produced a clean
# result:
#
#   1. stormlite_mome_both_paneldecoder — RERUN. Night5 crashed with
#      AssertionError (decoder_gene_names missing) because
#      inject_decoder_gene_names was only wired into train.py's own
#      main()/_main_multi_sample(), not run_comparison.py's separate
#      _train_model() (fixed, commit 75f5313). First real result for this
#      config.
#   2. wae_gan_stormlite_both (NEW config, fusion_mode="sum" default) —
#      night5's wae_gan_stormlite_mome_both collapsed (PCC=nan,
#      ConstantInputWarning — model output the same expression vector for
#      every query). This isolates whether MoME specifically destabilized
#      WAE-GAN's adversarial training, or whether StormLite+WAE-GAN
#      collapses regardless of fusion_mode.
#   3. wae_gan_stormlite_mome_both, reseeded (seed=1, separate
#      checkpoint_dir so the seed=0 checkpoint isn't overwritten) — checks
#      whether the collapse is a one-off (stochastic GAN training
#      instability) or reproducible/systematic.
#   4-5. stpath_unfrozen_mlpresidual / novaeresidual, reseeded (seed=1,
#      separate checkpoint_dirs) — night5 gave mlpresidual=0.0336,
#      novaeresidual=0.0999, apparently reversing the earlier night3
#      finding (MLP clearly beat baseline, Novae was worst). A second seed
#      tells us whether that ordering is stable or within noise before
#      treating it as a real reversal — eval sets here are only ~15-45
#      query points per masking draw, small enough that this could easily
#      be noise rather than a regression.
#
# Same CUDA_VISIBLE_DEVICES-pinning / CPU-thread-capping / resumable /
# corrected-grep pattern as every other parallel script in this directory.
#
# Usage: bash scripts/run_parallel_5gpu_night6.sh
# Logs: logs/parallel_run_night6_10000ep/<name>.log

set -u

CONFIGS=(
    "configs/exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder.yaml"
    "configs/exp_hest1k_wae_gan_stormlite_both.yaml"
    "configs/exp_hest1k_wae_gan_stormlite_mome_both.yaml"
    "configs/exp_hest1k_fm_ot_stpath_unfrozen_mlpresidual.yaml"
    "configs/exp_hest1k_fm_ot_stpath_unfrozen_novaeresidual.yaml"
)
# per-config extra --override args beyond epochs (empty string = none).
# The two reseed jobs get seed=1 + a distinct checkpoint_dir so they don't
# clobber the seed=0 checkpoints already saved from night5.
EXTRA_OVERRIDES=(
    ""
    ""
    "training.seed=1 training.checkpoint_dir=results/checkpoints/wae_gan_hest1k_pilot_stormlite_mome_both_seed1"
    "training.seed=1 training.checkpoint_dir=results/checkpoints/fm_ot_hest1k_pilot_stpath_unfrozen_mlpresidual_seed1"
    "training.seed=1 training.checkpoint_dir=results/checkpoints/fm_ot_hest1k_pilot_stpath_unfrozen_novaeresidual_seed1"
)
GPU_IDS=(3 4 5 6 7)
EPOCHS=10000
LOG_DIR="logs/parallel_run_night6_${EPOCHS}ep"
EXTRA_ARGS="--shuffle-diagnostic"

N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
N_CORES=$(nproc)
THREADS_PER_JOB=$((N_CORES / ${#CONFIGS[@]}))
[ "$THREADS_PER_JOB" -lt 1 ] && THREADS_PER_JOB=1

if [ "${#CONFIGS[@]}" -ne "${#GPU_IDS[@]}" ] || [ "${#CONFIGS[@]}" -ne "${#EXTRA_OVERRIDES[@]}" ]; then
    echo "ERROR: CONFIGS/GPU_IDS/EXTRA_OVERRIDES must all have the same length."
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
    extra="${EXTRA_OVERRIDES[$i]}"
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
        --override "training.epochs=${EPOCHS}" ${extra} \
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
