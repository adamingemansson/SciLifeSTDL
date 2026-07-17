#!/bin/bash
# Ninth batch (2026-07-17) — last run before the overnight gap. 7 configs,
# 10000 epochs each, GPUs 0,1,2,4,5,6,7 (skips GPU 3 -- still busy with
# the 40k stpath_unfrozen_bothresidual run from night8). Weighted toward StormLite (the priority
# context encoder for this project's actual research direction — GigaPath
# + Novae, STPath-inspired fusion, built from scratch rather than relying
# on STPath's external pretrained weights) rather than an even split:
#
#   1. stormlite_mome_both_paneldecoder_add — queued from earlier: our own
#      updated PanelInvariantGeneDecoder (combine_mode="add", scGPT's real
#      combination rule; gene_embed_dim=256; mlp_depth=2) on StormLite.
#   2. wae_gan_stpath_mlpresidual — queued from earlier: completes the
#      WAE-GAN residual-source ablation (plain/both/novae already run,
#      see docs/results_log.md). STPath, but a pre-existing queued gap,
#      not a new-decoder test.
#   3. stormlite_mome_both_lloki — NEW decoder_type="lloki" (faithfully
#      ports LLOKI-CAE's real, verified mechanism — see LLOKIStyleDecoder's
#      own docstring) on this project's leading from-scratch context
#      encoder.
#   4. stormlite_mome_novae_lloki — same new decoder, StormLite's
#      Novae-only gene branch (diversifies beyond "both").
#   5. stormlite_mome_mlp_paneldecoder_add — our updated
#      PanelInvariantGeneDecoder, StormLite's MLP-only gene branch
#      (diversifies beyond "both").
#   6. stpath_bothresidual_lloki — the new decoder on this project's
#      overall best context encoder (pretrained STPath+bothresidual) —
#      one STPath data point kept for comparison, not the focus.
#   7. wae_gan_stormlite_mome_both_lloki — diagnostic: does a different
#      decoder architecture change WAE-GAN+StormLite's confirmed
#      reproducible mode collapse at all, or is it entirely upstream
#      (context-encoder/adversarial-loss interaction)? Also the
#      regression-test target for a real bug found and fixed while wiring
#      "lloki" in: WAEGAN.training_step was calling self.decoder(...)
#      directly instead of through self._decode(...), which would have
#      crashed outright for decoder_type="lloki" (fixed, see
#      tests/test_alternative_decoders.py::test_wae_gan_lloki_end_to_end).
#
# NOT included: decoder_type="gene_attention" (Geneformer-inspired) —
# fully implemented and unit-tested (see tests/test_alternative_decoders.py),
# training-loss integration fixed, but run_comparison.py's shared FID/MMD
# machinery (PCA fit on the FULL training-panel width) doesn't yet handle
# gene_attention's restricted output panel correctly — deferred rather
# than risk a silently-wrong or crashing unattended run.
#
# Same CUDA_VISIBLE_DEVICES-pinning / CPU-thread-capping / resumable /
# corrected-grep pattern as every other parallel script in this directory.
#
# Usage: bash scripts/run_parallel_7gpu_night9.sh
# Logs: logs/parallel_run_night9_10000ep/<name>.log

set -u

CONFIGS=(
    "configs/exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder_add.yaml"
    "configs/exp_hest1k_wae_gan_stpath_mlpresidual.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_mome_both_lloki.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_mome_novae_lloki.yaml"
    "configs/exp_hest1k_fm_ot_stormlite_mome_mlp_paneldecoder_add.yaml"
    "configs/exp_hest1k_fm_ot_stpath_bothresidual_lloki.yaml"
    "configs/exp_hest1k_wae_gan_stormlite_mome_both_lloki.yaml"
)
GPU_IDS=(0 1 2 4 5 6 7)  # skips GPU 3 -- still busy with the 40k
                          # stpath_unfrozen_bothresidual run (night8)
EPOCHS=10000
LOG_DIR="logs/parallel_run_night9_${EPOCHS}ep"
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
