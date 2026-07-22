#!/usr/bin/env bash
# DGX launcher. Four learned comparisons use GPUs 0-3; the exact harmonic
# control uses GPU 4. GPUs 5-7 remain free rather than duplicating seeds.
set -Eeuo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
exec bash scripts/run_hierarchical_slide_4gpu.sh
