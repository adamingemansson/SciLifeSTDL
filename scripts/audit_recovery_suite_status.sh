#!/usr/bin/env bash
# One-shot status audit for the whole 207-251 recovery_suite run (smallhole/
# k-sweep diagnostics through today's gene-encoder/candidate-mechanism/
# image-encoder matrix). For each config, reports exactly one of:
#   COMPLETED  -- results/checkpoints/recovery_suite/<name>/heldout_sample_summary.json exists
#   RUNNING    -- a live process's command line references this config file
#   MISSING    -- neither -- needs a launch
#
# Run this ON THE MACHINE THAT ACTUALLY TRAINED THESE (st-a100), from the
# repo root, after `git pull`. Read-only: never starts, stops, or modifies
# anything, only reports.
#
# Usage: bash scripts/audit_recovery_suite_status.sh
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUNNING_PROCS="$(ps -eo pid,args | grep -E 'src\.training\.train' | grep -v grep || true)"

declare -a NUMBERS
for f in configs/recovery_suite/2[0-4][0-9]_transport_*.yaml configs/recovery_suite/25[01]_transport_*.yaml; do
  [ -e "$f" ] || continue
  number="$(basename "$f" | cut -d_ -f1)"
  case "$number" in
    20[0-6]) continue ;;  # keep this audit scoped to 207+
  esac
  NUMBERS+=("$number")
done

completed=0
running=0
missing=0
missing_list=()

printf "%-5s %-58s %-11s %s\n" "NUM" "EXPERIMENT_NAME" "STATUS" "DETAIL"
printf '%.0s-' {1..110}; echo

for number in $(printf '%s\n' "${NUMBERS[@]}" | sort -n); do
  matches=(configs/recovery_suite/${number}_transport_*.yaml)
  config_path="${matches[0]}"
  name="$(awk -F': ' '/^experiment_name:/ {print $2; exit}' "$config_path")"
  if [[ -z "$name" ]]; then
    printf "%-5s %-58s %-11s %s\n" "$number" "???" "ERROR" "could not read experiment_name from $config_path"
    continue
  fi

  summary_path="results/checkpoints/recovery_suite/${name}/heldout_sample_summary.json"
  proc_line="$(printf '%s\n' "$RUNNING_PROCS" | grep -F "$config_path" || true)"

  if [[ -f "$summary_path" ]]; then
    # Real schema (see summarize_transport_suite_v2.py's _heldout_row):
    # payload["image_modes"][payload["primary_image_mode"]]["pcc_mean"] --
    # nested by image mode, not a flat top-level key.
    pcc="$(python3 -c "import json
try:
    d = json.load(open('$summary_path'))
    mode = d['primary_image_mode']
    print(d['image_modes'][mode]['pcc_mean'])
except Exception as e:
    print(f'n/a ({e})')
" 2>/dev/null || echo "n/a")"
    printf "%-5s %-58s %-11s pcc=%s\n" "$number" "$name" "COMPLETED" "$pcc"
    completed=$((completed + 1))
  elif [[ -n "$proc_line" ]]; then
    pid="$(echo "$proc_line" | awk '{print $1}' | head -1)"
    printf "%-5s %-58s %-11s pid=%s\n" "$number" "$name" "RUNNING" "$pid"
    running=$((running + 1))
  else
    printf "%-5s %-58s %-11s %s\n" "$number" "$name" "MISSING" "$config_path"
    missing=$((missing + 1))
    missing_list+=("$number:$config_path")
  fi
done

echo
echo "Summary: $completed completed, $running running, $missing missing (out of ${#NUMBERS[@]} total, 207-251)."
if (( missing > 0 )); then
  echo
  echo "Missing configs (copy/paste-ready, fill in a free GPU index for each --"
  echo "check 'nvidia-smi' first, do not just default to 1,2,3,5 blindly):"
  for entry in "${missing_list[@]}"; do
    number="${entry%%:*}"
    path="${entry#*:}"
    echo "CUDA_VISIBLE_DEVICES=<gpu> nohup python3 -m src.training.train --config $path > logs/manual_rerun_${number}.log 2>&1 &"
  done
fi
