#!/usr/bin/env bash
# Frees disk space by deleting the (re-trainable) model weight files inside
# results/checkpoints/**, while keeping every actual result artifact and
# every config. Rerunning a config from scratch reproduces the deleted
# weights; it does NOT reproduce a lost heldout_sample_summary.json or a
# training_history/logs curve, so those are never touched here.
#
# Per checkpoint_dir, deletes ONLY:
#   trainable_weights.pt, trainable_weights.pt.tmp*   (the actual weights —
#     tens of MB per run; excludes frozen backbones like Gigapath/STPath by
#     construction, see save_trainable_state_dict's docstring in train.py)
#   model_cfg.json, gene_names.json                    (only useful for
#     reloading the weights above — dead weight once the weights are gone)
#
# Everything else in checkpoint_dir is left alone:
#   heldout_sample_summary.json, per_gene_results.csv, per_spot_results.csv,
#   audit_test_metrics*.json, validation_history.json, quality_gate.json,
#   sample_split.json, evaluation_gene_panels.json, training_exclusion.json,
#   overfit_training_mask.json, logs/ (Lightning CSVLogger curves)
#
# Skips any checkpoint_dir whose OWNING CONFIG is currently being trained
# (i.e. python3 -m src.training.train --config <that file> is a live
# process) -- matched by config file path, the same signal
# scripts/audit_recovery_suite_status.sh uses, since checkpoint_dir itself
# never actually appears in a training process's argv (only --config does).
# So this is safe to run while other configs are still training.
#
# Usage:
#   ./scripts/cleanup_checkpoint_weights.sh --dry-run   # preview, no deletes
#   ./scripts/cleanup_checkpoint_weights.sh              # actually delete
set -euo pipefail

ROOT="results/checkpoints"
DRY_RUN=0
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    DRY_RUN=1
  else
    ROOT="$arg"
  fi
done

if [[ ! -d "$ROOT" ]]; then
  echo "No such directory: $ROOT (nothing to clean up)"
  exit 0
fi

# Build checkpoint_dir -> owning config file map, so "is this still
# running" can check for the config path in `ps` output (what actually
# shows up in a launch command's argv) instead of the checkpoint_dir path
# (which never does).
declare -A CKPT_TO_CONFIG
while IFS= read -r -d '' cfg; do
  ckpt_line="$(grep -m1 '^\s*checkpoint_dir:' "$cfg" || true)"
  [[ -z "$ckpt_line" ]] && continue
  ckpt_path="${ckpt_line#*checkpoint_dir:}"
  ckpt_path="$(echo "$ckpt_path" | tr -d '[:space:]')"
  CKPT_TO_CONFIG["$ckpt_path"]="$cfg"
done < <(find configs -name "*.yaml" -print0 2>/dev/null)

total_bytes=0
n_deleted=0
n_skipped_running=0
n_skipped_no_weights=0

while IFS= read -r -d '' weights_file; do
  ckpt_dir="$(dirname "$weights_file")"

  # Skip if the config that owns this checkpoint_dir is currently being
  # trained. Require BOTH "src.training.train" (real launch commands are
  # `python3 -m src.training.train --config <path>`) AND the config path
  # in the same ps line -- a plain single-substring grep against the
  # config path alone is not safe: it can false-positive on any unrelated
  # process whose argv happens to mention that path for other reasons
  # (verified in testing -- a shell wrapper that echoes/logs the full
  # command line is enough to trigger it). Neither this script nor any
  # other non-training tool in this repo has "src.training.train" in its
  # own invocation, so the two-substring AND is specific to an actual
  # training process.
  config_file="${CKPT_TO_CONFIG[$ckpt_dir]:-}"
  if [[ -n "$config_file" ]]; then
    if ps -eo args= | grep -F "src.training.train" | grep -qF -- "$config_file"; then
      echo "SKIP (running, config=$config_file): $ckpt_dir"
      n_skipped_running=$((n_skipped_running + 1))
      continue
    fi
  fi

  dir_bytes=0
  files_to_remove=()
  for f in "$ckpt_dir"/trainable_weights.pt "$ckpt_dir"/trainable_weights.pt.tmp* \
           "$ckpt_dir"/model_cfg.json "$ckpt_dir"/gene_names.json; do
    if [[ -f "$f" ]]; then
      files_to_remove+=("$f")
      dir_bytes=$((dir_bytes + $(stat -c%s "$f" 2>/dev/null || stat -f%z "$f")))
    fi
  done

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "WOULD DELETE ($(numfmt --to=iec "$dir_bytes" 2>/dev/null || echo "${dir_bytes}B")): $ckpt_dir"
  else
    rm -f "${files_to_remove[@]}"
    echo "DELETED ($(numfmt --to=iec "$dir_bytes" 2>/dev/null || echo "${dir_bytes}B")):     $ckpt_dir"
  fi
  total_bytes=$((total_bytes + dir_bytes))
  n_deleted=$((n_deleted + 1))
done < <(find "$ROOT" -type f -name "trainable_weights.pt" -print0)

echo
echo "===== summary ====="
if [[ "$DRY_RUN" == "1" ]]; then
  echo "Would free: $(numfmt --to=iec "$total_bytes" 2>/dev/null || echo "${total_bytes} bytes") across $n_deleted checkpoint dirs"
else
  echo "Freed: $(numfmt --to=iec "$total_bytes" 2>/dev/null || echo "${total_bytes} bytes") across $n_deleted checkpoint dirs"
fi
echo "Skipped (still running): $n_skipped_running"
echo "Configs and all result JSON/CSV files (heldout_sample_summary.json etc.) were left untouched."
