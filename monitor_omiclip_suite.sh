#!/usr/bin/env bash
SUITE_ROOT=/data/adam.ingemansson/SciLifeSTDL-MK/gen3_multiscale/results/wae_mmd_geneencoder_ablation_omiclip_v1
ARMS=(wae_he_mmd_geneencoder_omiclip_scfoundation_film wae_he_mmd_geneencoder_omiclip_scfoundation_nofilm wae_he_mmd_geneencoder_omiclip_mlp_film wae_he_mmd_geneencoder_omiclip_mlp_nofilm)

while true; do
    clear
    echo "=== OmiCLIP geneencoder suite monitor -- $(date) ==="
    echo
    for arm in "${ARMS[@]}"; do
        pid=$(pgrep -f "train_conditional_wae.*${arm}\.yaml" | head -1)
        echo "--- $arm ---"
        if [ -n "$pid" ]; then
            echo "  PID $pid  $(ps -o etime=,time=,pcpu=,pmem= -p "$pid")"
        else
            echo "  NOT RUNNING"
        fi
        log="$SUITE_ROOT/logs/${arm}.log"
        if [ -f "$log" ]; then
            tail -n 3 "$log" | sed 's/^/  | /'
        fi
        echo
    done
    sleep 30
done
