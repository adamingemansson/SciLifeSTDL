#!/usr/bin/env bash
UNI2_ROOT=/data/adam.ingemansson/SciLifeSTDL-MK/gen3_multiscale/results/wae_mmd_geneencoder_ablation_uni2_zfix_v1
OMICLIP_ROOT=/data/adam.ingemansson/SciLifeSTDL-MK/gen3_multiscale/results/wae_mmd_geneencoder_ablation_omiclip_zfix_v1

while true; do
    clear
    echo "=== z-noise-augmentation validation progress -- $(date) ==="
    echo
    for root in "$UNI2_ROOT" "$OMICLIP_ROOT"; do
        for f in "$root"/logs/*_eval.log; do
            arm=$(basename "${f%_eval.log}")
            last=$(grep "evaluation progress" "$f" | tail -n 1)
            if [ -n "$last" ]; then
                echo "$arm: $(echo "$last" | grep -oE '[0-9]+/[0-9]+')"
            elif [ -f "${f%_eval.log}_eval.json" ]; then
                echo "$arm: DONE"
            else
                echo "$arm: starting up / preflight"
            fi
        done
    done
    sleep 10
done
