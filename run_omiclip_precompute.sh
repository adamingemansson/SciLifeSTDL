#!/usr/bin/env bash
set -euo pipefail

MK_ROOT="/data/adam.ingemansson/SciLifeSTDL-MK"
MANIFEST="$MK_ROOT/data/cache/hest1k/gen3_multiorgan_80_qc2_manifest.json"
OMICLIP_CHECKPOINT="/data/adam.ingemansson/checkpoints/omiclip/checkpoint.pt"

cd "$MK_ROOT"

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest not found at $MANIFEST" >&2
    find "$MK_ROOT" -iname "*manifest*.json" 2>/dev/null
    exit 1
fi
echo "Using manifest: $MANIFEST"

if [ ! -f "$OMICLIP_CHECKPOINT" ]; then
    echo "ERROR: OmiCLIP checkpoint not found at $OMICLIP_CHECKPOINT" >&2
    exit 1
fi
echo "Using OmiCLIP checkpoint: $OMICLIP_CHECKPOINT"

OMICLIP_REVISION=$(python3 -c "
from huggingface_hub import HfApi
print(HfApi().model_info('WangGuangyuLab/Loki').sha)
")
echo "Using OmiCLIP pinned revision: $OMICLIP_REVISION"

CONFIG=$(find "$MK_ROOT/gen3_multiscale/results" -name "config.yaml" -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | cut -d' ' -f2-)
if [ -z "${CONFIG:-}" ]; then
    echo "ERROR: no config.yaml found under $MK_ROOT/gen3_multiscale/results -- pass one explicitly." >&2
    exit 1
fi
echo "Using config: $CONFIG"

python scripts/precompute_gen3_omiclip_spot_features.py \
    --config "$CONFIG" \
    --manifest "$MANIFEST" \
    --omiclip-checkpoint-path "$OMICLIP_CHECKPOINT" \
    --omiclip-pinned-revision "$OMICLIP_REVISION" \
    --device cuda \
    --batch-size 32
