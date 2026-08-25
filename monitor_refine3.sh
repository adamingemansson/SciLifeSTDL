#!/usr/bin/env bash
SUITE_ROOT=/data/adam.ingemansson/SciLifeSTDL-MK/gen3_multiscale/results/wae_mmd_geneencoder_uni2_refine3_v1
ARMS=(wae_he_mmd_geneencoder_scfoundation_film wae_he_mmd_geneencoder_scfoundation_nofilm \
      wae_he_mmd_geneencoder_mlp_film wae_he_mmd_geneencoder_mlp_nofilm)
GPUS="0,2,3,5"

while true; do
    clear
    echo "=== refine3 suite -- $(date) ==="
    echo

    # ---- system RAM ----
    read -r _ TOT USED FREE _ _ AVAIL <<< "$(free -g | awk '/^Mem:/')"
    echo "SYSTEM RAM:  total=${TOT}G  used=${USED}G  available=${AVAIL}G   <- 'available' is what matters"
    echo

    # ---- GPUs ----
    echo "GPU        VRAM used / total     util"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
               --format=csv,noheader,nounits -i "$GPUS" \
    | awk -F', ' '{printf "  cuda:%-6s %6d / %6d MiB   %3d%%\n", $1, $2, $3, $4}'
    echo

    # ---- our processes ----
    echo "ARM                                    PID     ELAPSED     CPU%    RSS      MEM%"
    total_rss=0
    for arm in "${ARMS[@]}"; do
        pid=$(pgrep -f "train_conditional_wae.*${arm}\.yaml" | head -1)
        if [ -n "$pid" ]; then
            read -r et pcpu rss pmem <<< "$(ps -o etime=,pcpu=,rss=,pmem= -p "$pid")"
            printf "  %-36s %-7s %-11s %-7s %6.1fG  %5s%%\n" \
                   "${arm#wae_he_mmd_geneencoder_}" "$pid" "$et" "$pcpu" "$(echo "$rss/1048576" | bc -l)" "$pmem"
            total_rss=$((total_rss + rss))
        else
            printf "  %-36s %s\n" "${arm#wae_he_mmd_geneencoder_}" "NOT RUNNING"
        fi
    done
    printf "\n  TOTAL across our 4 arms: %.1f GB\n" "$(echo "$total_rss/1048576" | bc -l)"
    echo "  reference: BEFORE the patch fix each arm held ~7.2% (~150 GB); expect roughly a third now"
    echo

    # ---- progress ----
    echo "LAST LOG LINE PER ARM"
    for arm in "${ARMS[@]}"; do
        log="$SUITE_ROOT/logs/${arm}.log"
        printf "  --- %s\n" "${arm#wae_he_mmd_geneencoder_}"
        [ -f "$log" ] && tail -n 2 "$log" | sed 's/^/      /' || echo "      (no log yet)"
    done
    sleep 30
done
