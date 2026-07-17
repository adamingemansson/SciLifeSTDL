#!/bin/bash
# Eighth batch (2026-07-17) — lunch-break batch, mixed epoch counts. Five
# jobs, one at 40000 epochs and four at 10000:
#
#   1. stpath_unfrozen_bothresidual @ 40000 — the pretrained bothresidual
#      anchor (this project's best result) has 10k/20k/40k numbers
#      (0.4154/0.4077/0.4717); its unfrozen (no pretrained weights, "STPath
#      architecture trained from scratch") counterpart only has a 10k
#      number so far (0.2549). This fills in the matching 40k point so the
#      pretrained-vs-unfrozen gap can be checked at the SAME epoch count
#      the pretrained anchor peaks at, not just at 10k.
#   2. stormlite_mome_both_paneldecoder_bigger @ 10000 — capacity
#      follow-up to the first paneldecoder result (PCC 0.1308 vs. the
#      dense decoder's 0.2798 at the same epoch count): decoder_gene_embed_dim
#      bumped 64->256. Tests whether the gap is a capacity problem (fixable)
#      or something more structural about the panel-invariant mechanism.
#   3-4. vqvae_ar_stpath_mlpresidual / novaeresidual @ 10000 — only "both"
#      residual has been tested on VQ-VAE+AR so far
#      (exp_hest1k_vqvae_ar_stpath_bothresidual.yaml, night7); these
#      isolate the individual MLP-only/Novae-only contributions, mirroring
#      the FM-OT ablation that already exists.
#   5. wae_gan_stpath_novaeresidual @ 10000 — WAE-GAN+StormLite is
#      confirmed broken (reproducible mode collapse across 2 seeds); this
#      checks whether WAE-GAN is fine with STPath's real pretrained fusion
#      plus a single (smaller) residual addition, picking Novae since it
#      reproducibly beat MLP residual in night6's reseeded unfrozen-STPath
#      results.
#
# Same CUDA_VISIBLE_DEVICES-pinning / CPU-thread-capping / resumable /
# corrected-grep pattern as every other parallel script in this directory.
# NOTE: pins to GPUs 3-7 — night7 (GPUs 0-2) was still running when this
# was queued up, so this deliberately avoids that range. Check
# `nvidia-smi` first regardless before launching.
#
# Usage: bash scripts/run_parallel_5gpu_night8.sh
# Logs: logs/parallel_run_night8/<name>.log

set -u

CONFIGS=(
    "configs/exp_hest1k_fm_ot_stpath_unfrozen_bothresidual.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder_bigger.yaml"
    "configs/exp_hest1k_vqvae_ar_stpath_mlpresidual.yaml"
    "configs/exp_hest1k_vqvae_ar_stpath_novaeresidual.yaml"
    "configs/exp_hest1k_wae_gan_stpath_novaeresidual.yaml"
)
EPOCHS_PER_CONFIG=(40000 10000 10000 10000 10000)
GPU_IDS=(3 4 5 6 7)
LOG_DIR="logs/parallel_run_night8"
EXTRA_ARGS="--shuffle-diagnostic"

N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
N_CORES=$(nproc)
THREADS_PER_JOB=$((N_CORES / ${#CONFIGS[@]}))
[ "$THREADS_PER_JOB" -lt 1 ] && THREADS_PER_JOB=1

if [ "${#CONFIGS[@]}" -ne "${#GPU_IDS[@]}" ] || [ "${#CONFIGS[@]}" -ne "${#EPOCHS_PER_CONFIG[@]}" ]; then
    echo "ERROR: CONFIGS/GPU_IDS/EPOCHS_PER_CONFIG must all have the same length."
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
    epochs="${EPOCHS_PER_CONFIG[$i]}"
    name=$(basename "$cfg" .yaml)
    logfile="$LOG_DIR/${name}.log"
    if [ -f "$logfile" ] && grep -q "^model " "$logfile"; then
        echo "  GPU $gid: $cfg -> SKIPPING, already completed (see $logfile)"
        continue
    fi
    echo "  GPU $gid: $cfg (${epochs} epochs) -> $logfile"
    CUDA_VISIBLE_DEVICES=$gid \
    OMP_NUM_THREADS=$THREADS_PER_JOB \
    MKL_NUM_THREADS=$THREADS_PER_JOB \
    python -m src.evaluation.run_comparison "$cfg" \
        --override "training.epochs=${epochs}" \
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
