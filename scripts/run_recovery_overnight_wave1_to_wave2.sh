#!/usr/bin/env bash
# One unattended, fail-closed handoff: full Wave 1, then Wave 2.
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-4}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
REPO_ROOT="$(pwd -P)"
STPATH_ROOT="${STPATH_ROOT:-$(dirname "$REPO_ROOT")/STPath}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -z "${STPATH_GENE_VOC_PATH:-}" || "${STPATH_GENE_VOC_PATH:-}" == /absolute/path/* ]]; then
  export STPATH_GENE_VOC_PATH="$STPATH_ROOT/utils_data/symbol2ensembl.json"
fi
if [[ -z "${STPATH_MODEL_WEIGHT_PATH:-}" || "${STPATH_MODEL_WEIGHT_PATH:-}" == /absolute/path/* ]]; then
  export STPATH_MODEL_WEIGHT_PATH="$STPATH_ROOT/stfm.pth"
fi

if [[ ! -f "$STPATH_GENE_VOC_PATH" ]]; then
  echo "ERROR: STPath vocabulary not found: $STPATH_GENE_VOC_PATH" >&2
  exit 2
fi

STPATH_URL="https://huggingface.co/tlhuang/STPath/resolve/main/stfm.pth"
STPATH_SHA256="03d49af98103c22eaee064632a366ad6ba2c1e627adcf47e00b13746a3b348fe"
if [[ ! -f "$STPATH_MODEL_WEIGHT_PATH" ]]; then
  echo "Downloading the official STPath checkpoint to $STPATH_MODEL_WEIGHT_PATH"
  mkdir -p "$(dirname "$STPATH_MODEL_WEIGHT_PATH")"
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 3 --output "$STPATH_MODEL_WEIGHT_PATH.part" "$STPATH_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=3 --output-document="$STPATH_MODEL_WEIGHT_PATH.part" "$STPATH_URL"
  else
    echo "ERROR: curl or wget is required to download stfm.pth" >&2
    exit 2
  fi
  mv "$STPATH_MODEL_WEIGHT_PATH.part" "$STPATH_MODEL_WEIGHT_PATH"
fi

actual_sha256="$(sha256sum "$STPATH_MODEL_WEIGHT_PATH" | awk '{print $1}')"
if [[ "$actual_sha256" != "$STPATH_SHA256" ]]; then
  echo "ERROR: STPath checkpoint checksum mismatch: $STPATH_MODEL_WEIGHT_PATH" >&2
  echo "expected: $STPATH_SHA256" >&2
  echo "actual:   $actual_sha256" >&2
  exit 2
fi

if pgrep -f -- 'src.training.train.*recovery_(control|fixed_novae_flagship_seed1[12]|stpath_|ablation_)' >/dev/null; then
  echo "ERROR: recovery Wave 1/2 training processes are already running." >&2
  echo "Refusing to launch duplicate jobs." >&2
  pgrep -af -- 'src.training.train.*recovery_(control|fixed_novae_flagship_seed1[12]|stpath_|ablation_)' >&2 || true
  exit 2
fi

echo "===== One-step smoke check: all Wave 1 configs ====="
SMOKETEST=1 \
SMOKE_STEPS=1 \
STAGE=controls \
RUN_ID="${RUN_ID}_smoke_wave1" \
LOG_ROOT="logs/recovery_suite/smoke_controls_${RUN_ID}" \
GPU_IDS=0,1,2,3 \
CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/run_recovery_suite_8gpu.sh

echo "===== One-step smoke check: all Wave 2 configs ====="
SMOKETEST=1 \
SMOKE_STEPS=1 \
STAGE=ablations \
RUN_ID="${RUN_ID}_smoke_wave2" \
LOG_ROOT="logs/recovery_suite/smoke_ablations_${RUN_ID}" \
GPU_IDS=0,1,2,3 \
CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/run_recovery_suite_8gpu.sh

# set -e makes both smoke waves fail closed. Full training starts only after
# all sixteen configs build, train for one step and complete a minimal audit.
echo "===== Wave 1: matched FM and STPath controls ====="
STAGE=controls \
RUN_ID="${RUN_ID}_wave1" \
LOG_ROOT="logs/recovery_suite/controls_${RUN_ID}" \
GPU_IDS=0,1,2,3 \
CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/run_recovery_suite_8gpu.sh

# set -e makes this handoff fail closed: this line is reachable only when all
# eight Wave 1 jobs produced successful audit outputs.
echo "===== Wave 1 complete; starting Wave 2 component ablations ====="
STAGE=ablations \
RUN_ID="${RUN_ID}_wave2" \
LOG_ROOT="logs/recovery_suite/ablations_${RUN_ID}" \
GPU_IDS=0,1,2,3 \
CPU_THREADS_PER_JOB="$CPU_THREADS_PER_JOB" \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/run_recovery_suite_8gpu.sh

echo "Wave 1 and Wave 2 completed successfully."
echo "Wave 1 report: reports/recovery_suite/controls_${RUN_ID}_wave1/summary.csv"
echo "Wave 2 report: reports/recovery_suite/ablations_${RUN_ID}_wave2/summary.csv"
