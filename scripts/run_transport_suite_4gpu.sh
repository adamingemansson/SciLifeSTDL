#!/usr/bin/env bash
# hierarchical_gene_transport_regressor 20-run suite (2026-07-22 handoff,
# "Twenty-run suite"): 4 capacity gates (O01-O04) each followed by a
# sequential chain of 4 held-out matched runs (C01-C16), one queue per GPU,
# all four queues running in parallel. Mirrors run_hierarchical_slide_4gpu.sh
# / run_hierarchical_controls_4gpu.sh's own preflight conventions, plus two
# checks the handoff asks for that neither sibling script has yet: a
# disk-space check and resumable-artifact detection (skip a job whose
# checkpoint_dir already has a completed marker from a prior run).
#
# Queues (GPU_IDS default matches the handoff exactly: 1,2,3,5):
#   GPU[0] (default 1): O01 -> C01 -> C05 -> C09 -> C13
#   GPU[1] (default 2): O02 -> C02 -> C06 -> C10 -> C14
#   GPU[2] (default 3): O03 -> C03 -> C07 -> C11 -> C15
#   GPU[3] (default 5): O04 -> C04 -> C08 -> C12 -> C16
#
# Capacity gate (O0X): 3k-step repeated-training-hole run: finite metrics,
# >=0.002 RMSE improvement over IDW, correction RMS >=0.005 (checked via
# quality_gate.json, NOT enforced in-process -- require_anchor_improvement
# stays false in the config, same as the existing gene_aware_overfit_int7
# precedent, so a failed gate is a clean skip here, not a mid-training
# crash). If a GPU's own O0X gate fails, that GPU's C-chain is skipped;
# the other three GPUs' queues are unaffected.
#
# Usage:
#   bash scripts/run_transport_suite_4gpu.sh
#   GPU_IDS=1,2,3,5 SMOKE_ONLY=1 bash scripts/run_transport_suite_4gpu.sh
#   SKIP_SMOKE=1 bash scripts/run_transport_suite_4gpu.sh
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-2}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"
GPU_IDS_CSV="${GPU_IDS:-1,2,3,5}"
IFS=',' read -r -a GPUS <<< "$GPU_IDS_CSV"

# Config numbers per queue slot: [O0X, C-a, C-b, C-c, C-d]
QUEUE_1=(165 169 173 177 181)  # O01 C01 C05 C09 C13
QUEUE_2=(166 170 174 178 182)  # O02 C02 C06 C10 C14
QUEUE_3=(167 171 175 179 183)  # O03 C03 C07 C11 C15
QUEUE_4=(168 172 176 180 184)  # O04 C04 C08 C12 C16
QUEUES=(QUEUE_1 QUEUE_2 QUEUE_3 QUEUE_4)

declare -A CONFIG_PATH
declare -A CONFIG_NAME
for f in configs/recovery_suite/1[6-8][0-9]_transport_*.yaml; do
  number="$(basename "$f" | cut -d_ -f1)"
  name="$(basename "$f" .yaml)"
  CONFIG_PATH["$number"]="$f"
  CONFIG_NAME["$number"]="$name"
done

LOG_ROOT="logs/recovery_suite/transport_suite_${RUN_ID}"
REPORT_ROOT="reports/recovery_suite/transport_suite_${RUN_ID}"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$CPU_THREADS_PER_JOB"
export MKL_NUM_THREADS="$CPU_THREADS_PER_JOB"
export OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB"
export NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB"
mkdir -p "$LOG_ROOT" "$REPORT_ROOT"

echo "===== Static fail-closed audit of all 20 transport-suite configs ====="
"$PYTHON_BIN" scripts/check_transport_suite_configs.py

echo "===== GPU_IDS check ====="
if (( ${#GPUS[@]} != 4 )); then
  echo "ERROR: GPU_IDS must provide exactly four comma-separated devices (got '$GPU_IDS_CSV')." >&2
  exit 2
fi
declare -A seen_gpus=()
for gpu in "${GPUS[@]}"; do
  if ! [[ "$gpu" =~ ^[0-9]+$ ]] || [[ -n "${seen_gpus[$gpu]:-}" ]]; then
    echo "ERROR: GPU_IDS contains an invalid or duplicate device: $GPU_IDS_CSV" >&2
    exit 2
  fi
  seen_gpus[$gpu]=1
done

echo "===== Exact GPU existence check ====="
mapfile -t VISIBLE_GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)
if [ "${#VISIBLE_GPUS[@]}" -eq 0 ]; then
  echo "ERROR: nvidia-smi reported no visible GPUs." >&2
  exit 2
fi
for gpu in "${GPUS[@]}"; do
  found=0
  for visible in "${VISIBLE_GPUS[@]}"; do
    [[ "$visible" == "$gpu" ]] && found=1 && break
  done
  if [[ "$found" -ne 1 ]]; then
    echo "ERROR: requested GPU $gpu is not among nvidia-smi's visible devices: ${VISIBLE_GPUS[*]}" >&2
    exit 2
  fi
done

echo "===== Disk-space check (>= ${MIN_FREE_GB}GiB free) ====="
avail_kb="$(df -Pk "$REPO_ROOT" | awk 'NR==2 {print $4}')"
avail_gb=$((avail_kb / 1024 / 1024))
if (( avail_gb < MIN_FREE_GB )); then
  echo "ERROR: only ${avail_gb}GiB free on $REPO_ROOT's filesystem, need >= ${MIN_FREE_GB}GiB." >&2
  echo "Checkpoints/mask banks for 20 jobs need real headroom; free space or lower MIN_FREE_GB." >&2
  exit 2
fi
echo "OK: ${avail_gb}GiB free."

if [[ -z "${GIGAPATH_SLIDE_CHECKPOINT:-}" || ! -f "$GIGAPATH_SLIDE_CHECKPOINT" ]]; then
  echo "ERROR: GIGAPATH_SLIDE_CHECKPOINT is missing or not a file." >&2
  exit 2
fi
if ! CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PYTHON_BIN" -c \
  'import torch; from gigapath.torchscale.component.flash_attention import flash_attn_func; assert torch.cuda.is_available() and flash_attn_func is not None'; then
  echo "ERROR: GigaPath LongNet's FlashAttention CUDA kernel is unavailable." >&2
  echo "  MAX_JOBS=4 python3 -m pip install flash-attn==2.5.8 --no-build-isolation" >&2
  exit 2
fi
for sid in INT1 INT2 INT3 INT4 INT5 INT6 INT7 INT8; do
  test -f "data/cache/hest1k/gigapath_slide_cache/${sid}.npz" || {
    echo "ERROR: missing dense WSI cache for $sid; run a hierarchical precompute launcher" >&2
    exit 2
  }
done
if pgrep -af -- 'src.training.train.*transport_' >"$LOG_ROOT/existing_processes.txt"; then
  echo "ERROR: transport-suite jobs already exist:" >&2
  cat "$LOG_ROOT/existing_processes.txt" >&2
  exit 2
fi

# Resumable-artifact detection: a job is "already done" if its checkpoint_dir
# already carries the completion marker a real (non-smoke) run of that job
# writes at the very end -- quality_gate.json for capacity gates (evaluation
# is disabled, that's the only final artifact), heldout_sample_summary.json
# for held-out runs (train.py's own final report, same file
# summarize_transport_suite.py reads). Smoke runs use their own timestamped
# results/checkpoints/recovery_suite/smoke/... path so they are never
# affected by this check and always re-run fresh.
job_is_done() {
  local number="$1" name="${CONFIG_NAME[$1]}"
  local ckpt="results/checkpoints/recovery_suite/${name}"
  if [[ "$number" -lt 169 ]]; then
    test -f "$ckpt/quality_gate.json"
  else
    test -f "$ckpt/heldout_sample_summary.json"
  fi
}

run_one() {
  local number="$1" gpu="$2" smoke="$3"
  local name="${CONFIG_NAME[$number]}" path="${CONFIG_PATH[$number]}"
  local command=("$PYTHON_BIN" -m src.training.train --config "$path")
  if [[ "$smoke" == "1" ]]; then
    command+=(--override
      "experiment_name=${name}_smoke_${RUN_ID}"
      training.epochs=1 training.unique_mask_count=1
      training.checkpoint_every_n_steps=0 training.log_print_every_n_steps=1
      "training.checkpoint_dir=results/checkpoints/recovery_suite/smoke/${RUN_ID}/${name}"
      validation.every_n_steps=1 validation.early_stopping_min_steps=1
      validation.patience_checks=1000 validation.require_anchor_improvement=false
      # training_mask_bank_path is keyed by n_items=cfg.training.epochs
      # (see ensure_training_seed_bank / _training_seed_bank_for_config).
      # Smoke overrides epochs to 1; without its own bank path here, a
      # capacity-gate smoke run (O01-O04, evaluation disabled, no other
      # bank-path override below) writes a real 1-item bank to the SAME
      # path the real 3000-step run then tries to reuse -- a genuine
      # n_items mismatch the bank's own staleness guard correctly rejects.
      # Applies to every smoke run, not just held-out ones.
      "evaluation.training_mask_bank_path=results/mask_banks/training/recovery_suite/smoke_${RUN_ID}_${name}.json"
    )
    if [[ "$number" -ge 169 ]]; then
      command+=(evaluation.n_validation_masks=1 evaluation.n_test_masks=1 evaluation.n_samples=1
        "evaluation.mask_bank_dir=results/mask_banks/recovery_suite/smoke_transport_${RUN_ID}")
    fi
    echo "GPU $gpu -> $name (smoke)"
    CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}" >"$LOG_ROOT/smoke_${name}.log" 2>&1
    return $?
  fi
  if job_is_done "$number"; then
    echo "GPU $gpu -> $name SKIP (already completed, found prior completion marker)"
    return 0
  fi
  echo "GPU $gpu -> $name START"
  CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}" >"$LOG_ROOT/full_${name}.log" 2>&1
}

run_queue() {
  local queue_name="$1" gpu="$2"
  local -n queue_ref="$queue_name"
  local o_number="${queue_ref[0]}" o_name="${CONFIG_NAME[${queue_ref[0]}]}"

  if ! job_is_done "$o_number"; then
    run_one "$o_number" "$gpu" 0 || {
      echo "ERROR: capacity gate $o_name failed to run (job crashed); see $LOG_ROOT/full_${o_name}.log" >&2
      return 1
    }
  else
    echo "GPU $gpu -> $o_name SKIP (already completed)"
  fi

  local gate_file="results/checkpoints/recovery_suite/${o_name}/quality_gate.json"
  local passed
  passed="$("$PYTHON_BIN" -c "
import json, sys
try:
    print(json.load(open('$gate_file')).get('passed'))
except FileNotFoundError:
    print('None')
")"
  if [[ "$passed" != "True" ]]; then
    echo "CAPACITY GATE FAILED for $o_name (passed=$passed) -- wiring or optimization is broken." >&2
    echo "Per the handoff's decision rules: stop. Skipping this GPU's held-out chain (${queue_ref[*]:1})." >&2
    return 1
  fi
  echo "Capacity gate $o_name PASSED -- proceeding to this GPU's held-out chain."

  local i number
  for i in 1 2 3 4; do
    number="${queue_ref[$i]}"
    run_one "$number" "$gpu" 0 || {
      echo "ERROR: ${CONFIG_NAME[$number]} failed; see $LOG_ROOT/full_${CONFIG_NAME[$number]}.log" >&2
      return 1
    }
  done
}

if [[ "$SKIP_SMOKE" != "1" ]]; then
  # Representative, not exhaustive: the model's own math (residual on/off,
  # geometry- vs hierarchical-conditioning, shared vs per-gene gate,
  # transport_heads=1..16, local_k=32..128) is already covered by
  # tests/test_hierarchical_gene_transport.py at the PyTorch level without
  # needing GPU/data/WSI caches. What a real smoke test can additionally
  # catch is CONFIG WIRING (does this exact YAML parse and run end to end on
  # real data) -- one smoke per GPU queue's capacity-gate config (O01-O04)
  # already exercises that for every distinct base setup (all four modality
  # combinations, both fusion modes, both residual on/off), so this is a
  # complete-enough smoke pass without spending 20x the GPU time.
  echo "===== One-step fail-closed smoke test (O01-O04, one per queue) ====="
  smoke_pids=()
  for idx in 0 1 2 3; do
    queue_name="${QUEUES[$idx]}"
    declare -n q_ref="$queue_name"
    run_one "${q_ref[0]}" "${GPUS[$idx]}" 1 &
    smoke_pids+=("$!")
  done
  smoke_failed=0
  for pid in "${smoke_pids[@]}"; do wait "$pid" || smoke_failed=1; done
  if (( smoke_failed )); then
    echo "ERROR: smoke test failed; inspect $LOG_ROOT/smoke_*.log" >&2
    exit 1
  fi
  echo "Smoke tests passed."
fi
if [[ "$SMOKE_ONLY" == "1" ]]; then
  echo "Smoke tests passed; full runs were not started."
  exit 0
fi

echo "===== Four parallel queues (capacity gate -> 4 held-out runs each) ====="
queue_pids=()
for idx in 0 1 2 3; do
  run_queue "${QUEUES[$idx]}" "${GPUS[$idx]}" &
  queue_pids+=("$!")
done
queue_failed=0
for pid in "${queue_pids[@]}"; do wait "$pid" || queue_failed=1; done

echo "===== Exact-mask non-learned harmonic control (same test masks as C05) ====="
CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PYTHON_BIN" -m src.training.train \
  --config configs/recovery_suite/185_transport_harmonic_k128_control.yaml \
  >"$LOG_ROOT/full_transport_harmonic_k128.log" 2>&1 || queue_failed=1

if (( queue_failed )); then
  echo "WARNING: at least one queue/job failed or a capacity gate was not passed; inspect $LOG_ROOT/*.log" >&2
fi
"$PYTHON_BIN" scripts/summarize_transport_suite.py --output "$REPORT_ROOT/summary.csv" || true
echo "Transport suite run finished (RUN_ID=$RUN_ID). Report: $REPORT_ROOT/summary.csv"
echo "Logs: $LOG_ROOT"
