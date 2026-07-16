#!/bin/bash
# Runs the multi-sample training + organ/tech conditioning demo config
# (configs/exp_hest1k_fm_ot_multisample.yaml, 2026-07-16) via
# src.training.train directly — NOT run_comparison.py, which doesn't
# support cfg.data.sample_ids yet (see that config's own header, and
# src/evaluation/run_comparison.py's _train_model docstring note).
#
# IMPORTANT before running: this config's model.params.n_genes (16570) is
# a PLACEHOLDER copied from the single-sample INT1 config — the real
# shared-gene-panel size across INT1/INT2/INT3/INT4 will differ (see the
# config's own header). Run once, note the real
# "load_multi_sample: ... keeps N/M genes" line each sample prints to
# stdout, then edit n_genes in the config to match before trusting a full
# run's results (a wrong value won't crash, it'll just silently misalign
# columns — see the config header for why this isn't auto-injected yet).
#
# Usage: bash scripts/run_multisample_demo.sh [extra --override args...]

set -eu

mkdir -p logs
LOGFILE="logs/multisample_demo_$(date +%s).log"
echo "Running configs/exp_hest1k_fm_ot_multisample.yaml -> ${LOGFILE}"
echo "(watch for the 'load_multi_sample: ... keeps N/M genes' lines — verify n_genes against them)"

python -m src.training.train --config configs/exp_hest1k_fm_ot_multisample.yaml "$@" 2>&1 | tee "${LOGFILE}"
