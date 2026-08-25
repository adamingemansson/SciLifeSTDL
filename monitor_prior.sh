#!/bin/bash
ROOT=/data/adam.ingemansson/SciLifeSTDL-MK/gen3_multiscale/results/wae_mmd_geneencoder_uni2_refine3_prior_v1
while true; do
  clear
  echo "=== $(date +%H:%M:%S)  arch4 prior arms ==="
  free -g | awk 'NR==2{printf "RAM  used %sG / %sG   available %sG\n",$3,$2,$7}'
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
             --format=csv,noheader | sed 's/^/GPU /'
  echo
  for f in $ROOT/logs/*.log; do
    arm=$(basename $f .log)
    last=$(grep -E "^step " $f | tail -1)
    printf "%-45s %s\n" "$arm" "${last:-initialising...}"
  done
  sleep 30
done
