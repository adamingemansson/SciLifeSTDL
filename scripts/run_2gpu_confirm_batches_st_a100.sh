#!/bin/bash
# FULL-DAY confirmation batch for st-a100 (2026-07-19). That node is
# SHARED -- markus.+ has real, active jobs on GPUs 1,2,3,4,5,7 (confirmed
# via `ps -p <pid> -o pid,user,etime,cmd` on every process, not just an
# nvidia-smi snapshot) -- only GPU 0 and GPU 6 are actually free. Do NOT
# widen GPU_IDS below without re-checking `nvidia-smi` + `ps` on any
# newly-idle-looking device first.
#
# 14 jobs total, queued sequentially, 7 per GPU (~a full day, not the
# earlier 8-job/~5h version). Every job's reasoning:
#
#   0-3. bigger + QK-norm + warmup + EMA, seeds 10/11/12/13, 80k epochs --
#        clean, unconfounded 4-seed mean for "does bigger capacity +
#        QK-norm reliably close the gap to STPath" (the single earlier
#        data point, 0.5028, also had logit-normal + no-log1p stacked in).
#   4-6. STPath UNFROZEN + winning decoder + EMA, seeds 10/11/12, 40k
#        epochs -- the "full retrain" comparison: STPath with zero
#        pretrained weights, same decoder everyone else uses. Only one
#        old data point exists (0.3857, BEFORE the decoder swap).
#   7.   bigger + QK-norm ONLY, flat lr=1.0e-3, warmup_steps=0, seed 10,
#        40k epochs -- isolates QK-norm from the LR/warmup change it's
#        always been bundled with. Collapse = warmup/lower-LR were doing
#        real work; a real number = QK-norm alone sufficed.
#   8.   bigger + warmup+lower-LR ONLY, storm_lite_qk_norm=false, seed 10,
#        40k epochs -- the MIRROR of job 7: completes the 2x2 factorial
#        (baseline collapsed; +QK-norm only = job7; +warmup only = this;
#        +both = jobs 0-3/prior 0.3886-0.5028). Tells us whether warmup
#        alone (no QK-norm) is also enough to prevent the collapse, or
#        whether QK-norm is the necessary ingredient either way.
#   9-10. flagship (small) + QK-norm + logit-normal, seeds 10/11, 40k
#        epochs, EMA -- disambiguates the "all3" stacking collapse (one
#        seed of QK-norm+logit-normal+no-log1p together collapsed to PCC
#        0.058). Tests this SPECIFIC pair alone, 2 seeds (not 1, since
#        the whole reason we're here is a single-seed collapse).
#   11-12. flagship (small) + QK-norm + no-log1p, seeds 10/11, 40k epochs,
#        EMA -- the OTHER candidate pairing from the same "all3" collapse,
#        same reasoning as jobs 9-10.
#   13.  pretrain -> finetune (scripts/run_pretrain_then_finetune_stormlite.sh)
#        -- HIGHEST PRIORITY item not yet run at all. Motivated by a real
#        finding: STPath's own pretrained-vs-unfrozen ablation is worth
#        ~0.086 PCC, architecture held constant -- StormLite has never had
#        a pretraining stage. Genuinely sequential internally (multi-
#        sample pretrain @ 80k epochs, then single-sample finetune warm-
#        started @ 10k epochs) -- placed FIRST on GPU 0 so it starts
#        immediately and has the most time to complete during the day.
#
# Load split (job-count balanced, 7/GPU; job 13's own runtime is uncertain
# since it's never been timed, so GPU 0 intentionally carries fewer of the
# well-understood single-sample jobs to make room for it):
#   GPU 0: job13(pretrain->finetune) + job0(80k) + job2(80k) + job4(40k) + job6(40k) + job9(40k) + job11(40k)
#   GPU 6: job1(80k) + job3(80k) + job5(40k) + job7(40k) + job8(40k) + job10(40k) + job12(40k)
#
# Usage: bash scripts/run_2gpu_confirm_batches_st_a100.sh
# Smoke test (2 epochs by default): SMOKETEST=1 bash scripts/run_2gpu_confirm_batches_st_a100.sh
# Smoke test, custom epoch count: SMOKETEST=1 SMOKETEST_EPOCHS=5 bash scripts/run_2gpu_confirm_batches_st_a100.sh
# Logs: logs/parallel_run_2gpu_confirm/<name>.log (job 13 also writes its
# own logs/pretrain_finetune/*.log via the driver script it calls)

set -u

SMOKETEST="${SMOKETEST:-0}"
SMOKETEST_EPOCHS="${SMOKETEST_EPOCHS:-2}"

BIGGER_CFG="configs/exp_hest1k_fm_ot_stormlite_mome_both_bigger_paneldecoder_add_warmupema.yaml"
STPATH_CFG="configs/exp_hest1k_fm_ot_stpath_unfrozen_bothresidual_paneldecoder_add.yaml"
FLAG_CFG="configs/exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder_add.yaml"

# job index -> config/name/seed/epochs/extra-override ("single" jobs, 0-12)
JOB_CONFIG=(
    "$BIGGER_CFG" "$BIGGER_CFG" "$BIGGER_CFG" "$BIGGER_CFG"
    "$STPATH_CFG" "$STPATH_CFG" "$STPATH_CFG"
    "$BIGGER_CFG" "$BIGGER_CFG"
    "$FLAG_CFG" "$FLAG_CFG"
    "$FLAG_CFG" "$FLAG_CFG"
)
JOB_NAME=(
    "stormlite_mome_both_paneldecoder_bigger_qknorm_warmup_ema_seed10"
    "stormlite_mome_both_paneldecoder_bigger_qknorm_warmup_ema_seed11"
    "stormlite_mome_both_paneldecoder_bigger_qknorm_warmup_ema_seed12"
    "stormlite_mome_both_paneldecoder_bigger_qknorm_warmup_ema_seed13"
    "stpath_bothresidual_paneldecoder_unfrozen_ema_seed10"
    "stpath_bothresidual_paneldecoder_unfrozen_ema_seed11"
    "stpath_bothresidual_paneldecoder_unfrozen_ema_seed12"
    "stormlite_mome_both_paneldecoder_bigger_qknormonly_flatlr_ema_seed10"
    "stormlite_mome_both_paneldecoder_bigger_warmuponly_noqknorm_ema_seed10"
    "stormlite_mome_both_paneldecoder_qknorm_logitnormal_ema_seed10"
    "stormlite_mome_both_paneldecoder_qknorm_logitnormal_ema_seed11"
    "stormlite_mome_both_paneldecoder_qknorm_nolog1p_ema_seed10"
    "stormlite_mome_both_paneldecoder_qknorm_nolog1p_ema_seed11"
)
JOB_SEED=(10 11 12 13 10 11 12 10 10 10 11 10 11)
JOB_EPOCHS=(80000 80000 80000 80000 40000 40000 40000 40000 40000 40000 40000 40000 40000)
JOB_EXTRA_OVERRIDE=(
    "model.params.storm_lite_qk_norm=true"
    "model.params.storm_lite_qk_norm=true"
    "model.params.storm_lite_qk_norm=true"
    "model.params.storm_lite_qk_norm=true"
    ""
    ""
    ""
    "model.params.storm_lite_qk_norm=true model.params.warmup_steps=0 model.params.lr=1.0e-3"
    "model.params.storm_lite_qk_norm=false"
    "model.params.storm_lite_qk_norm=true model.params.fm_time_sampling=logit_normal"
    "model.params.storm_lite_qk_norm=true model.params.fm_time_sampling=logit_normal"
    "model.params.storm_lite_qk_norm=true model.params.storm_lite_input_already_log1p=true"
    "model.params.storm_lite_qk_norm=true model.params.storm_lite_input_already_log1p=true"
)

GPU0_JOBS=(0 2 4 6 9 11)      # + job 13 (pretrain->finetune), handled separately, first
GPU6_JOBS=(1 3 5 7 8 10 12)

LOG_DIR="logs/parallel_run_2gpu_confirm"
SMOKE_OVERRIDE=""
if [ "$SMOKETEST" = "1" ]; then
    LOG_DIR="logs/parallel_run_2gpu_confirm_smoketest"
    SMOKE_OVERRIDE="training.epochs=${SMOKETEST_EPOCHS} training.checkpoint_every_n_steps=1 training.log_print_every_n_steps=1"
    echo "*** SMOKETEST=1 -- tiny versions of all 14 jobs (epochs=${SMOKETEST_EPOCHS}). Logs: $LOG_DIR ***"
fi
mkdir -p "$LOG_DIR"

N_CORES=$(nproc)
THREADS_PER_JOB=$(( N_CORES / 2 ))   # 2 GPUs run concurrently
[ "$THREADS_PER_JOB" -lt 1 ] && THREADS_PER_JOB=1

run_one() {
    local gpu="$1" idx="$2"
    local cfg="${JOB_CONFIG[$idx]}" name="${JOB_NAME[$idx]}" seed="${JOB_SEED[$idx]}"
    local epochs="${JOB_EPOCHS[$idx]}" extra="${JOB_EXTRA_OVERRIDE[$idx]}"
    local logfile="$LOG_DIR/${name}.log"
    if [ -f "$logfile" ] && grep -q "^model " "$logfile"; then
        echo "  GPU $gpu: [$name] SKIP (already completed)"
        return
    fi
    echo "  GPU $gpu: [$name] START"
    local epoch_override=""
    [ "$SMOKETEST" != "1" ] && epoch_override="training.epochs=${epochs}"
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=$THREADS_PER_JOB MKL_NUM_THREADS=$THREADS_PER_JOB \
    python -m src.evaluation.run_comparison "$cfg" \
        --override ${extra} \
                    training.seed=${seed} \
                    training.ema_decay=0.999 \
                    ${epoch_override} \
                    training.checkpoint_dir=results/checkpoints/${name} \
                    training.checkpoint_every_n_steps=10000 \
                    training.log_print_every_n_steps=1000 \
                    ${SMOKE_OVERRIDE} \
        --shuffle-diagnostic > "$logfile" 2>&1
    echo "  GPU $gpu: [$name] DONE"
}

run_pretrain_finetune_job() {
    local gpu="$1"
    local logfile="$LOG_DIR/pretrain_finetune_stormlite.log"
    if [ -f "$logfile" ] && grep -q "Finetune stage finished\|Finetune stage already completed" "$logfile"; then
        echo "  GPU $gpu: [pretrain_finetune_stormlite] SKIP (already completed)"
        return
    fi
    echo "  GPU $gpu: [pretrain_finetune_stormlite] START (sequential pretrain->finetune driver)"
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=$THREADS_PER_JOB MKL_NUM_THREADS=$THREADS_PER_JOB \
    SMOKETEST=$SMOKETEST SMOKETEST_EPOCHS=$SMOKETEST_EPOCHS \
    bash scripts/run_pretrain_then_finetune_stormlite.sh > "$logfile" 2>&1
    echo "  GPU $gpu: [pretrain_finetune_stormlite] DONE"
}

run_gpu0_queue() {
    run_pretrain_finetune_job 0
    for idx in "${GPU0_JOBS[@]}"; do
        run_one 0 "$idx"
    done
}
run_gpu6_queue() {
    for idx in "${GPU6_JOBS[@]}"; do
        run_one 6 "$idx"
    done
}

echo "Launching 14 jobs across 2 free GPUs (0, 6) on st-a100, ${THREADS_PER_JOB} CPU threads each."
echo "GPUs 1,2,3,4,5,7 are occupied by markus.+'s jobs -- not touched."
echo "GPU 0 runs pretrain->finetune FIRST (highest priority, longest, starts immediately), then 6 more jobs."
echo "GPU 6 runs 7 jobs."
echo "Logs: $LOG_DIR"

run_gpu0_queue &
run_gpu6_queue &
wait

echo ""
echo "=== All jobs finished. Results: ==="
for name in "${JOB_NAME[@]}"; do
    echo "--- $name ---"
    grep -m1 -A2 "^model " "$LOG_DIR/${name}.log" 2>/dev/null | tail -2
done
echo "--- pretrain_finetune_stormlite ---"
grep -A2 "=== Finetune result ===" "$LOG_DIR/pretrain_finetune_stormlite.log" 2>/dev/null | tail -2

echo ""
echo "Compare against:"
echo "  StormLite (small) + decoder, 8-seed mean (EMA+non-EMA): ~0.462"
echo "  STPath (pretrained) + decoder, 4-seed mean: ~0.503"
echo "  STPath (unfrozen) + OLD dense decoder, 1 seed, no EMA: 0.3857"
echo "  Earlier confounded bigger+QK-norm points: 0.3886 (40k), 0.5028 (80k, all3-stacked)"
echo "  Jobs 7/8 (qknormonly_flatlr / warmuponly_noqknorm) are standalone diagnostics --"
echo "  compare both against the collapsed original (nan/AUC~0.5) and the working"
echo "  QK-norm+warmup-together jobs (0-3) to see which ingredient is load-bearing."
echo "  Jobs 9-12 (qknorm_logitnormal / qknorm_nolog1p, 2 seeds each) tell us whether"
echo "  either pairing is safe to adopt as default, or whether the 'all3' collapse"
echo "  risk lives in one of these pairs specifically."
