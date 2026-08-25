#!/usr/bin/env bash
CX=/data/adam.ingemansson/SciLifeSTDL-MK/gen3_multiscale/results/mk_wae_gan_coexpression_ablation_suite_20260807
U2=/data/adam.ingemansson/SciLifeSTDL-MK/gen3_multiscale/results/mk_wae_gan_uni2_ablation_suite_20260807
HIST=/data/adam.ingemansson/SciLifeSTDL-MK/gen3_multiscale/results/mk_wae_gan_histology_ablation_suite_20260807

while true; do
  clear
  date
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv -i 0,2,3,5
  for log in \
    "$CX/logs/wae_he_gan_coexpression_control.log" \
    "$CX/logs/wae_he_gan_coexpression.log" \
    "$CX/logs/wae_he_gan_coexpression_scfoundation.log" \
    "$U2/logs/wae_he_gan_uni2_control.log" \
    "$U2/logs/wae_he_gan_uni2.log" \
    "$HIST/logs/wae_he_gan_histology_control.log" \
    "$HIST/logs/wae_he_gan_histology.log" \
  ; do
    echo
    echo "== $log =="
    tail -5 "$log" 2>/dev/null
  done
  sleep 30
done
