#!/usr/bin/env bash
# Eight-device entry point for the existing dependency-ordered ladder.
set -Eeuo pipefail
export DEVICES="${DEVICES:-0 1 2 3 4 5 6 7}"
exec bash scripts/run_complexity_ladder_4gpu.sh
