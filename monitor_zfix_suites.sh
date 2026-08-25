#!/usr/bin/env bash
UNI2_ROOT=/data/adam.ingemansson/SciLifeSTDL-MK/gen3_multiscale/results/wae_mmd_geneencoder_ablation_uni2_zfix_v1
OMICLIP_ROOT=/data/adam.ingemansson/SciLifeSTDL-MK/gen3_multiscale/results/wae_mmd_geneencoder_ablation_omiclip_zfix_v1
UNI2_ARMS=(wae_he_mmd_geneencoder_scfoundation_film wae_he_mmd_geneencoder_scfoundation_nofilm wae_he_mmd_geneencoder_mlp_film wae_he_mmd_geneencoder_mlp_nofilm)
OMICLIP_ARMS=(wae_he_mmd_geneencoder_omiclip_scfoundation_film wae_he_mmd_geneencoder_omiclip_scfoundation_nofilm wae_he_mmd_geneencoder_omiclip_mlp_film wae_he_mmd_geneencoder_omiclip_mlp_nofilm)

while true; do
    clear
    echo "=== z-noise-augmentation retrain monitor -- $(date) ==="
    echo
    echo "----- UNI2 suite -----"
    for arm in "${UNI2_ARMS[@]}"; do
        pid=$(pgrep -f "train_conditional_wae.*${arm}\.yaml" | head -1)
        echo "--- $arm ---"
        if [ -n "$pid" ]; then
            echo "  PID $pid  $(ps -o etime=,time=,pcpu=,pmem= -p "$pid")"
        else
            echo "  NOT RUNNING"
        fi
        log="$UNI2_ROOT/logs/${arm}.log"
        [ -f "$log" ] && tail -n 3 "$log" | sed 's/^/  | /'
        echo
    done
    echo "----- OmiCLIP suite -----"
    for arm in "${OMICLIP_ARMS[@]}"; do
        pid=$(pgrep -f "train_conditional_wae.*${arm}\.yaml" | head -1)
        echo "--- $arm ---"
        if [ -n "$pid" ]; then
            echo "  PID $pid  $(ps -o etime=,time=,pcpu=,pmem= -p "$pid")"
        else
            echo "  NOT RUNNING"
        fi
        log="$OMICLIP_ROOT/logs/${arm}.log"
        [ -f "$log" ] && tail -n 3 "$log" | sed 's/^/  | /'
        echo
    done
    sleep 30
done
