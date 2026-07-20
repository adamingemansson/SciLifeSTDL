#!/bin/bash
# st-a100 batch (2026-07-20), GPUs 0-3 — the node freed up (no more
# labmate contention), used for two DIFFERENT questions at once, each
# isolated from the other:
#
#   1. Two NEW single-lever architecture tests on the StormLite flagship
#      (single-sample, INT1) -- each tested ALONE, not stacked:
#        - storm_lite_knn_k: k-NN-restricted attention (ports STFlow's
#          real design choice, Huang et al. 2025, arXiv 2506.05361 --
#          verified via direct PDF read, 2026-07-20). An inductive-bias
#          change, NOT a memory fix (see _knn_additive_mask's own
#          docstring in storm_lite_encoder.py) -- memory is already
#          handled separately by masking.max_context_points.
#        - boundary_consistency_weight: adapted from DISCO's ablation-
#          proven-most-important component (Duan et al. 2025, "DISCO: A
#          Diffusion Model for Spatial Transcriptomics Data Completion" --
#          verified via direct PDF read, 2026-07-20). See
#          _boundary_consistency_loss's own docstring in registry.py for
#          the full (honest, non-literal) adaptation from DISCO's real
#          per-step re-noise-and-splice mechanism to a training-time
#          auxiliary loss.
#      Values below (knn_k=32, boundary_consistency_weight=0.1,
#      bandwidth=100.0) are FIRST ESTIMATES, not empirically tuned --
#      flagged explicitly, same as masking.max_context_points=3000 was.
#
#   2. Multi-sample generalization check on BOTH established flagships
#      (StormLite and STPath-unfrozen, same panel_invariant/add decoder,
#      no new levers) -- addresses the "everything tested so far is one
#      sample (INT1) only" gap flagged 2026-07-20, using the newly-freed
#      80GB cards to finally afford full INT1-INT8 concurrently.
#
# Job 1 and 2 are deliberately single-sample+isolated-lever; job 3 and 4
# are deliberately multi-sample+NO new lever -- keeping the two questions
# ("does this new architecture idea help?" vs "does the flagship
# generalize across samples?") cleanly unconfounded from each other.
#
# Usage: bash scripts/run_parallel_4gpu_sta100_knn_boundary_multisample.sh
# Smoke test: SMOKETEST=1 bash scripts/run_parallel_4gpu_sta100_knn_boundary_multisample.sh
# Logs: logs/parallel_run_sta100_knn_boundary_multisample/<name>.log

set -u

SMOKETEST="${SMOKETEST:-0}"
SMOKETEST_EPOCHS="${SMOKETEST_EPOCHS:-2}"

FLAG_CFG="configs/exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder_add.yaml"
MULTISAMPLE_STORMLITE_CFG="configs/exp_multisample_fm_ot_stormlite_mome_both_paneldecoder_add.yaml"
MULTISAMPLE_STPATH_CFG="configs/exp_multisample_fm_ot_stpath_unfrozen_bothresidual_paneldecoder_add.yaml"

CONFIGS=(
    "$FLAG_CFG"
    "$FLAG_CFG"
    "$MULTISAMPLE_STPATH_CFG"
    "$MULTISAMPLE_STORMLITE_CFG"
)
NAMES=(
    "stormlite_mome_both_paneldecoder_knnk32_ema_seed10"
    "stormlite_mome_both_paneldecoder_boundaryconsist_ema_seed10"
    "multisample_stpath_unfrozen_bothresidual_paneldecoder_ema_seed10"
    "multisample_stormlite_mome_both_paneldecoder_ema_seed10"
)
SEEDS=(10 10 10 10)
EPOCHS=(40000 40000 40000 40000)
EXTRA_OVERRIDE=(
    "model.params.storm_lite_knn_k=32"
    "model.params.boundary_consistency_weight=0.1 model.params.boundary_consistency_bandwidth=100.0"
    ""
    ""
)

LOG_DIR="logs/parallel_run_sta100_knn_boundary_multisample"
EXTRA_ARGS="--shuffle-diagnostic"
SMOKE_OVERRIDE=""
if [ "$SMOKETEST" = "1" ]; then
    LOG_DIR="logs/parallel_run_sta100_knn_boundary_multisample_smoketest"
    SMOKE_OVERRIDE="training.epochs=${SMOKETEST_EPOCHS} training.checkpoint_every_n_steps=1 training.log_print_every_n_steps=1"
    echo "*** SMOKETEST=1 -- tiny versions of all 4 jobs (epochs=${SMOKETEST_EPOCHS}). Logs: $LOG_DIR ***"
    rm -rf "$LOG_DIR"   # always fresh -- see 2026-07-20 gene-tokenizer-batch fix, same reasoning
fi
mkdir -p "$LOG_DIR"

N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
TOTAL_JOBS=${#CONFIGS[@]}
if [ "$TOTAL_JOBS" -gt "$N_GPUS" ]; then
    echo "ERROR: $TOTAL_JOBS configs but only $N_GPUS GPUs visible."
    exit 1
fi
N_CORES=$(nproc)
THREADS_PER_JOB=$((N_CORES / TOTAL_JOBS))
[ "$THREADS_PER_JOB" -lt 1 ] && THREADS_PER_JOB=1

echo "Launching $TOTAL_JOBS jobs on GPUs 0-$((TOTAL_JOBS - 1)), ${THREADS_PER_JOB} CPU threads each, logs in $LOG_DIR..."

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    name="${NAMES[$i]}"
    seed="${SEEDS[$i]}"
    extra="${EXTRA_OVERRIDE[$i]}"
    logfile="$LOG_DIR/${name}.log"
    if [ -f "$logfile" ] && grep -q "^model " "$logfile"; then
        echo "  GPU $i: [$name] SKIP (already completed)"
        continue
    fi
    echo "  GPU $i: [$name] START"
    epoch_override=""
    [ "$SMOKETEST" != "1" ] && epoch_override="training.epochs=${EPOCHS[$i]}"
    CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=$THREADS_PER_JOB MKL_NUM_THREADS=$THREADS_PER_JOB \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 -m src.evaluation.run_comparison "$cfg" \
        --override ${extra} \
                    training.seed=${seed} \
                    training.ema_decay=0.999 \
                    ${epoch_override} \
                    training.checkpoint_dir=results/checkpoints/${name} \
                    training.checkpoint_every_n_steps=10000 \
                    training.log_print_every_n_steps=1000 \
                    ${SMOKE_OVERRIDE} \
        ${EXTRA_ARGS} > "$logfile" 2>&1 &
done

wait
echo ""
echo "=== All jobs finished. Results: ==="
for name in "${NAMES[@]}"; do
    echo "--- $name ---"
    grep -m1 -A2 "^model " "$LOG_DIR/${name}.log" 2>/dev/null | tail -2
done

echo ""
echo "Compare against:"
echo "  Job 1 (knn_k=32) vs StormLite small (flagship, no knn_k), 11-seed mean: 0.4546 (40k epochs)"
echo "  Job 2 (boundary_consistency_weight=0.1) vs the same 0.4546 baseline"
echo "  Job 3 (multisample STPath-unfrozen) vs single-sample flagship STPath-unfrozen (see"
echo "    exp_hest1k_fm_ot_stpath_unfrozen_bothresidual_paneldecoder_add.yaml's own header)"
echo "  Job 4 (multisample StormLite) vs single-sample StormLite flagship (0.4546) --"
echo "    also directly comparable to job 3 for a multi-sample StormLite-vs-STPath read"
