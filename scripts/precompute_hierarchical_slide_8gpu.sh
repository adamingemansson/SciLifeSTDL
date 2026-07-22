#!/usr/bin/env bash
# DGX launcher: one INT sample per GPU, then eight Novae shards.
set -Eeuo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
exec bash scripts/precompute_hierarchical_slide_4gpu.sh
