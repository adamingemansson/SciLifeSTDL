#!/usr/bin/env bash
# DGX launcher: eight distinct learned comparisons, one per GPU. The short
# exact harmonic control runs after them without duplicating a learned seed.
set -Eeuo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
exec bash scripts/run_hierarchical_slide_4gpu.sh
