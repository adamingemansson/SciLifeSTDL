#!/usr/bin/env bash
# Staged recovery/ablation runner. The legacy filename is retained for
# compatibility. The default shared-server profile enforces GPUs 0-3; the
# explicit dedicated8 profile is only for a separate machine allocated in full.
set -Eeuo pipefail

STAGE="${STAGE:-repair}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPU_IDS_ARR <<< "$GPU_IDS_CSV"
SERVER_PROFILE="${SERVER_PROFILE:-shared4}"
case "$SERVER_PROFILE" in
  shared4)
    if [[ "$GPU_IDS_CSV" != "0,1,2,3" ]]; then
      echo "ERROR: shared4 allocation is fixed to GPU_IDS=0,1,2,3." >&2
      exit 2
    fi
    ;;
  shared_subset)
    declare -A seen_shared_gpu=()
    for gpu in "${GPU_IDS_ARR[@]}"; do
      if [[ ! "$gpu" =~ ^[0-3]$ ]]; then
        echo "ERROR: shared_subset GPU IDs must be drawn from 0,1,2,3; got $gpu." >&2
        exit 2
      fi
      if [[ -n "${seen_shared_gpu[$gpu]:-}" ]]; then
        echo "ERROR: shared_subset contains duplicate GPU ID $gpu." >&2
        exit 2
      fi
      seen_shared_gpu[$gpu]=1
    done
    ;;
  explicit_subset)
    declare -A seen_explicit_gpu=()
    for gpu in "${GPU_IDS_ARR[@]}"; do
      if [[ ! "$gpu" =~ ^[0-7]$ ]]; then
        echo "ERROR: explicit_subset GPU IDs must be drawn from 0-7; got $gpu." >&2
        exit 2
      fi
      if [[ -n "${seen_explicit_gpu[$gpu]:-}" ]]; then
        echo "ERROR: explicit_subset contains duplicate GPU ID $gpu." >&2
        exit 2
      fi
      seen_explicit_gpu[$gpu]=1
    done
    ;;
  dedicated8)
    if [[ "$GPU_IDS_CSV" != "0,1,2,3,4,5,6,7" ]]; then
      echo "ERROR: dedicated8 requires GPU_IDS=0,1,2,3,4,5,6,7." >&2
      exit 2
    fi
    ;;
  *)
    echo "ERROR: SERVER_PROFILE must be shared4, shared_subset, explicit_subset or dedicated8." >&2
    exit 2
    ;;
esac
GPU_COUNT="${#GPU_IDS_ARR[@]}"

CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-4}"
if [[ ! "$CPU_THREADS_PER_JOB" =~ ^[0-9]+$ ]] \
    || (( CPU_THREADS_PER_JOB < 1 || CPU_THREADS_PER_JOB > 4 )); then
  echo "ERROR: CPU_THREADS_PER_JOB must be an integer from 1 to 4 (default: 4)." >&2
  exit 2
fi

FRESH="${FRESH:-0}"
SMOKETEST="${SMOKETEST:-0}"
SMOKE_STEPS="${SMOKE_STEPS:-20}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_ROOT="${LOG_ROOT:-logs/recovery_suite/${STAGE}_${RUN_ID}}"
mkdir -p "$LOG_ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Directly executed helper scripts live under scripts/, so Python otherwise
# places that directory (not the repository root) on sys.path and cannot
# import the sibling src package.
REPO_ROOT="$(pwd -P)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
STPATH_ROOT="${STPATH_ROOT:-$(dirname "$REPO_ROOT")/STPath}"
if [[ -z "${STPATH_GENE_VOC_PATH:-}" || "${STPATH_GENE_VOC_PATH:-}" == /absolute/path/* ]]; then
  export STPATH_GENE_VOC_PATH="$STPATH_ROOT/utils_data/symbol2ensembl.json"
fi
if [[ -z "${STPATH_MODEL_WEIGHT_PATH:-}" || "${STPATH_MODEL_WEIGHT_PATH:-}" == /absolute/path/* ]]; then
  export STPATH_MODEL_WEIGHT_PATH="$STPATH_ROOT/stfm.pth"
fi

THREADS_PER_JOB="$CPU_THREADS_PER_JOB"

declare -a CONFIGS NAMES SEEDS
case "$STAGE" in
  repair)
    CONFIGS=(
      configs/recovery_suite/00_harmonic_anchor.yaml
      configs/recovery_suite/01_harmonic_residual_builtin.yaml
      configs/recovery_suite/01_harmonic_residual_builtin.yaml
      configs/recovery_suite/01_harmonic_residual_builtin.yaml
      configs/recovery_suite/02_harmonic_residual_stormlite_concat.yaml
      configs/recovery_suite/02_harmonic_residual_stormlite_concat.yaml
      configs/recovery_suite/02_harmonic_residual_stormlite_concat.yaml
      configs/recovery_suite/03_fixed_novae_flagship_regression.yaml
    )
    NAMES=(
      recovery_harmonic_anchor
      recovery_hr_builtin_seed0
      recovery_hr_builtin_seed1
      recovery_hr_builtin_seed2
      recovery_hr_concat_seed0
      recovery_hr_concat_seed1
      recovery_hr_concat_seed2
      recovery_fixed_novae_flagship_seed10
    )
    SEEDS=(0 0 1 2 0 1 2 10)
    ;;
  controls)
    : "${STPATH_GENE_VOC_PATH:?Set STPATH_GENE_VOC_PATH for the STPath controls}"
    : "${STPATH_MODEL_WEIGHT_PATH:?Set STPATH_MODEL_WEIGHT_PATH for pretrained STPath}"
    if [[ "$STPATH_GENE_VOC_PATH" == /absolute/path/* || ! -f "$STPATH_GENE_VOC_PATH" ]]; then
      echo "ERROR: STPATH_GENE_VOC_PATH is not a real file: $STPATH_GENE_VOC_PATH" >&2
      echo "Expected the STPath symbol2ensembl.json vocabulary file." >&2
      exit 2
    fi
    if [[ "$STPATH_MODEL_WEIGHT_PATH" == /absolute/path/* || ! -f "$STPATH_MODEL_WEIGHT_PATH" ]]; then
      echo "ERROR: STPATH_MODEL_WEIGHT_PATH is not a real file: $STPATH_MODEL_WEIGHT_PATH" >&2
      echo "Expected the downloaded STPath stfm.pth checkpoint." >&2
      exit 2
    fi
    CONFIGS=(
      configs/recovery_suite/04_control_fm_builtin.yaml
      configs/recovery_suite/05_control_fm_stormlite.yaml
      configs/recovery_suite/03_fixed_novae_flagship_regression.yaml
      configs/recovery_suite/03_fixed_novae_flagship_regression.yaml
      configs/recovery_suite/06_stpath_pretrained_benchmark.yaml
      configs/recovery_suite/06_stpath_pretrained_benchmark.yaml
      configs/recovery_suite/07_stpath_unfrozen_benchmark.yaml
      configs/recovery_suite/07_stpath_unfrozen_benchmark.yaml
    )
    NAMES=(
      recovery_control_fm_builtin_seed0
      recovery_control_fm_stormlite_seed0
      recovery_fixed_novae_flagship_seed11
      recovery_fixed_novae_flagship_seed12
      recovery_stpath_pretrained_seed10
      recovery_stpath_pretrained_seed11
      recovery_stpath_unfrozen_seed10
      recovery_stpath_unfrozen_seed11
    )
    SEEDS=(0 0 11 12 10 11 10 11)
    ;;
  ablations)
    CONFIGS=(
      configs/recovery_suite/08_ablation_mlp_only.yaml
      configs/recovery_suite/09_ablation_novae_only.yaml
      configs/recovery_suite/10_ablation_no_he.yaml
      configs/recovery_suite/11_ablation_query_image_dropout.yaml
      configs/recovery_suite/12_ablation_dense_decoder.yaml
      configs/recovery_suite/13_ablation_fusion_sum.yaml
      configs/recovery_suite/14_ablation_no_spatial_bias.yaml
      configs/recovery_suite/15_ablation_qk_norm.yaml
    )
    NAMES=(
      recovery_ablation_mlp_only_seed10
      recovery_ablation_novae_only_seed10
      recovery_ablation_no_he_seed10
      recovery_ablation_query_image_dropout_seed10
      recovery_ablation_dense_decoder_seed10
      recovery_ablation_fusion_sum_seed10
      recovery_ablation_no_spatial_bias_seed10
      recovery_ablation_qk_norm_seed10
    )
    SEEDS=(10 10 10 10 10 10 10 10)
    ;;
  wave3)
    : "${STPATH_GENE_VOC_PATH:?Set STPATH_GENE_VOC_PATH for the Wave 3 STPath benchmark}"
    : "${STPATH_MODEL_WEIGHT_PATH:?Set STPATH_MODEL_WEIGHT_PATH for the Wave 3 STPath benchmark}"
    if [[ ! -f "$STPATH_GENE_VOC_PATH" || ! -f "$STPATH_MODEL_WEIGHT_PATH" ]]; then
      echo "ERROR: Wave 3 requires real STPath vocabulary and checkpoint files." >&2
      echo "vocabulary: $STPATH_GENE_VOC_PATH" >&2
      echo "checkpoint: $STPATH_MODEL_WEIGHT_PATH" >&2
      exit 2
    fi
    CONFIGS=(
      configs/recovery_suite/05_control_fm_stormlite.yaml
      configs/recovery_suite/05_control_fm_stormlite.yaml
      configs/recovery_suite/16_wave3_stormlite_control_cap3000.yaml
      configs/recovery_suite/06_stpath_pretrained_benchmark.yaml
      configs/recovery_suite/17_wave3_flagship_modern_transport.yaml
      configs/recovery_suite/18_wave3_flagship_adaln_warmup.yaml
      configs/recovery_suite/19_wave3_flagship_modern_recipe.yaml
      configs/recovery_suite/19_wave3_flagship_modern_recipe.yaml
    )
    NAMES=(
      recovery_wave3_stormlite_control_seed1
      recovery_wave3_stormlite_control_seed2
      recovery_wave3_stormlite_matched_cap3000_seed0
      recovery_wave3_stpath_pretrained_seed12
      recovery_wave3_flagship_modern_transport_seed10
      recovery_wave3_flagship_adaln_warmup_seed10
      recovery_wave3_flagship_modern_recipe_seed10
      recovery_wave3_flagship_modern_recipe_seed11
    )
    SEEDS=(1 2 0 12 10 10 10 11)
    ;;
  component40k)
    CONFIGS=(
      configs/recovery_suite/20_component40k_warmup_only.yaml
      configs/recovery_suite/21_component40k_adaln_only.yaml
      configs/recovery_suite/22_component40k_coord_augment_only.yaml
      configs/recovery_suite/23_component40k_logit_time_only.yaml
      configs/recovery_suite/24_component40k_minibatch_ot_only.yaml
      configs/recovery_suite/25_component40k_lower_lr_only.yaml
      configs/recovery_suite/26_component40k_boundary_only.yaml
      configs/recovery_suite/27_component40k_logit_ot_pair.yaml
    )
    NAMES=(
      recovery_component40k_warmup_only_seed10
      recovery_component40k_adaln_only_seed10
      recovery_component40k_coord_augment_only_seed10
      recovery_component40k_logit_time_only_seed10
      recovery_component40k_minibatch_ot_only_seed10
      recovery_component40k_lower_lr_only_seed10
      recovery_component40k_boundary_only_seed10
      recovery_component40k_logit_ot_pair_seed10
    )
    SEEDS=(10 10 10 10 10 10 10 10)
    ;;
  missing_tissue)
    CONFIGS=(
      configs/recovery_suite/28_missing_tissue_reference.yaml
      configs/recovery_suite/29_missing_tissue_optimized.yaml
      configs/recovery_suite/30_missing_tissue_latent64.yaml
      configs/recovery_suite/31_missing_tissue_latent128.yaml
      configs/recovery_suite/32_missing_tissue_transformer_large.yaml
      configs/recovery_suite/33_missing_tissue_flow_large.yaml
      configs/recovery_suite/34_missing_tissue_decoder_large.yaml
      configs/recovery_suite/35_missing_tissue_knn32.yaml
    )
    NAMES=(
      missing_tissue_reference_seed10
      missing_tissue_optimized_seed10
      missing_tissue_latent64_seed10
      missing_tissue_latent128_seed10
      missing_tissue_transformer_large_seed10
      missing_tissue_flow_large_seed10
      missing_tissue_decoder_large_seed10
      missing_tissue_knn32_seed10
    )
    SEEDS=(10 10 10 10 10 10 10 10)
    ;;
  missing_tissue_controls)
    : "${STPATH_GENE_VOC_PATH:?Set STPATH_GENE_VOC_PATH for missing-tissue controls}"
    : "${STPATH_MODEL_WEIGHT_PATH:?Set STPATH_MODEL_WEIGHT_PATH for official/pretrained STPath}"
    if [[ ! -f "$STPATH_GENE_VOC_PATH" || ! -f "$STPATH_MODEL_WEIGHT_PATH" ]]; then
      echo "ERROR: missing-tissue controls require real STPath vocabulary and checkpoint files." >&2
      echo "vocabulary: $STPATH_GENE_VOC_PATH" >&2
      echo "checkpoint: $STPATH_MODEL_WEIGHT_PATH" >&2
      exit 2
    fi
    CONFIGS=(
      configs/recovery_suite/36_missing_tissue_official_stpath.yaml
      configs/recovery_suite/37_missing_tissue_stpath_fm_pretrained.yaml
      configs/recovery_suite/38_missing_tissue_stpath_fm_scratch.yaml
      configs/recovery_suite/39_missing_tissue_stormlite_deterministic.yaml
    )
    NAMES=(
      missing_tissue_control_official_stpath
      missing_tissue_control_stpath_fm_pretrained_seed10
      missing_tissue_control_stpath_fm_scratch_seed10
      missing_tissue_control_stormlite_deterministic_seed10
    )
    SEEDS=(10 10 10 10)
    ;;
  wave4_hr)
    CONFIGS=(
      configs/recovery_suite/40_wave4_hr_mlp_only.yaml
      configs/recovery_suite/41_wave4_hr_novae_only.yaml
      configs/recovery_suite/42_wave4_hr_fusion_sum.yaml
      configs/recovery_suite/43_wave4_hr_no_spatial_bias.yaml
      configs/recovery_suite/44_wave4_hr_relative_position.yaml
      configs/recovery_suite/45_wave4_hr_gnn_knn8.yaml
      configs/recovery_suite/46_wave4_hr_transformer_large.yaml
      configs/recovery_suite/47_wave4_hr_harmonic_k16.yaml
    )
    NAMES=(
      missing_tissue_wave4_hr_mlp_only_seed10
      missing_tissue_wave4_hr_novae_only_seed10
      missing_tissue_wave4_hr_fusion_sum_seed10
      missing_tissue_wave4_hr_no_spatial_bias_seed10
      missing_tissue_wave4_hr_relative_position_seed10
      missing_tissue_wave4_hr_gnn_knn8_seed10
      missing_tissue_wave4_hr_transformer_large_seed10
      missing_tissue_wave4_hr_harmonic_k16_seed10
    )
    SEEDS=(10 10 10 10 10 10 10 10)
    ;;
  wave4_fm)
    CONFIGS=(
      configs/recovery_suite/48_wave4_fm_latent128_refopt.yaml
      configs/recovery_suite/49_wave4_fm_decoder512_refopt.yaml
      configs/recovery_suite/50_wave4_fm_latent128_decoder512_refopt.yaml
      configs/recovery_suite/51_wave4_fm_latent128_decoder512_flow1024_refopt.yaml
    )
    NAMES=(
      missing_tissue_wave4_fm_latent128_refopt_seed10
      missing_tissue_wave4_fm_decoder512_refopt_seed10
      missing_tissue_wave4_fm_latent128_decoder512_refopt_seed10
      missing_tissue_wave4_fm_latent128_decoder512_flow1024_refopt_seed10
    )
    SEEDS=(10 10 10 10)
    ;;
  wave4b_hr)
    CONFIGS=(
      configs/recovery_suite/52_wave4b_hr_harmonic_k4.yaml
      configs/recovery_suite/53_wave4b_hr_reference_k8.yaml
      configs/recovery_suite/54_wave4b_hr_harmonic_k12.yaml
      configs/recovery_suite/55_wave4b_hr_harmonic_k24.yaml
      configs/recovery_suite/56_wave4b_hr_harmonic_k32.yaml
      configs/recovery_suite/57_wave4b_hr_k16_mlp_only.yaml
      configs/recovery_suite/58_wave4b_hr_k16_novae_only.yaml
      configs/recovery_suite/59_wave4b_hr_k16_no_spatial_bias.yaml
    )
    NAMES=(
      missing_tissue_wave4b_hr_harmonic_k4_seed10
      missing_tissue_wave4b_hr_reference_k8_seed10
      missing_tissue_wave4b_hr_harmonic_k12_seed10
      missing_tissue_wave4b_hr_harmonic_k24_seed10
      missing_tissue_wave4b_hr_harmonic_k32_seed10
      missing_tissue_wave4b_hr_k16_mlp_only_seed10
      missing_tissue_wave4b_hr_k16_novae_only_seed10
      missing_tissue_wave4b_hr_k16_no_spatial_bias_seed10
    )
    SEEDS=(10 10 10 10 10 10 10 10)
    ;;
  wave4c_controls)
    CONFIGS=(
      configs/recovery_suite/60_wave4c_fm_reference.yaml
      configs/recovery_suite/61_wave4c_hr_harmonic_k16.yaml
      configs/recovery_suite/62_wave4c_harmonic_anchor_k8.yaml
      configs/recovery_suite/63_wave4c_harmonic_anchor_k16.yaml
    )
    NAMES=(
      missing_tissue_wave4c_fm_reference_seed10
      missing_tissue_wave4c_hr_harmonic_k16_seed10
      missing_tissue_wave4c_harmonic_anchor_k8
      missing_tissue_wave4c_harmonic_anchor_k16
    )
    SEEDS=(10 10 10 10)
    ;;
  wave4d_hr)
    CONFIGS=(
      configs/recovery_suite/64_wave4d_hr_harmonic_k48.yaml
      configs/recovery_suite/65_wave4d_hr_harmonic_k64.yaml
      configs/recovery_suite/66_wave4d_hr_harmonic_k96.yaml
      configs/recovery_suite/67_wave4d_hr_harmonic_k128.yaml
      configs/recovery_suite/68_wave4d_hr_k32_mlp_only.yaml
      configs/recovery_suite/69_wave4d_hr_k32_novae_only.yaml
      configs/recovery_suite/70_wave4d_hr_k32_no_spatial_bias.yaml
      configs/recovery_suite/71_wave4d_hr_k32_mlp_no_spatial_bias.yaml
    )
    NAMES=(
      missing_tissue_wave4d_hr_harmonic_k48_seed10
      missing_tissue_wave4d_hr_harmonic_k64_seed10
      missing_tissue_wave4d_hr_harmonic_k96_seed10
      missing_tissue_wave4d_hr_harmonic_k128_seed10
      missing_tissue_wave4d_hr_k32_mlp_only_seed10
      missing_tissue_wave4d_hr_k32_novae_only_seed10
      missing_tissue_wave4d_hr_k32_no_spatial_bias_seed10
      missing_tissue_wave4d_hr_k32_mlp_no_spatial_bias_seed10
    )
    SEEDS=(10 10 10 10 10 10 10 10)
    ;;
  wave5_modalities_tk)
    CONFIGS=(
      configs/recovery_suite/72_wave5_tk_gex_only.yaml
      configs/recovery_suite/73_wave5_tk_he_only.yaml
      configs/recovery_suite/74_wave5_tk_neither.yaml
      configs/recovery_suite/75_wave5_tk_modality_dropout.yaml
      configs/recovery_suite/76_wave5_tk_k128_mlp_only.yaml
      configs/recovery_suite/77_wave5_tk_k128_novae_only.yaml
      configs/recovery_suite/78_wave5_tk_k128_no_spatial_bias.yaml
      configs/recovery_suite/79_wave5_tk_harmonic_anchor_k128.yaml
    )
    NAMES=(
      missing_tissue_wave5_tk_gex_only_seed10
      missing_tissue_wave5_tk_he_only_seed10
      missing_tissue_wave5_tk_neither_seed10
      missing_tissue_wave5_tk_modality_dropout_seed10
      missing_tissue_wave5_tk_k128_mlp_only_seed10
      missing_tissue_wave5_tk_k128_novae_only_seed10
      missing_tissue_wave5_tk_k128_no_spatial_bias_seed10
      missing_tissue_wave5_tk_harmonic_anchor_k128
    )
    SEEDS=(10 10 10 10 10 10 10 10)
    ;;
  wave5_modalities_st)
    CONFIGS=(
      configs/recovery_suite/80_wave5_st_full.yaml
      configs/recovery_suite/81_wave5_st_gex_only.yaml
      configs/recovery_suite/82_wave5_st_he_only.yaml
      configs/recovery_suite/83_wave5_st_neither.yaml
    )
    NAMES=(
      missing_tissue_wave5_st_full_seed10
      missing_tissue_wave5_st_gex_only_seed10
      missing_tissue_wave5_st_he_only_seed10
      missing_tissue_wave5_st_neither_seed10
    )
    SEEDS=(10 10 10 10)
    ;;
  wave6_heldout_tk)
    CONFIGS=(
      configs/recovery_suite/84_wave6_tk_heldout_full_k128.yaml
      configs/recovery_suite/85_wave6_tk_heldout_gex_only_k128.yaml
      configs/recovery_suite/86_wave6_tk_heldout_he_only_k128.yaml
      configs/recovery_suite/87_wave6_tk_heldout_neither_k128.yaml
      configs/recovery_suite/88_wave6_tk_heldout_full_k16.yaml
      configs/recovery_suite/89_wave6_tk_heldout_full_k32.yaml
      configs/recovery_suite/90_wave6_tk_heldout_full_k64.yaml
      configs/recovery_suite/91_wave6_tk_heldout_full_k128_no_spatial_bias.yaml
      configs/recovery_suite/96_wave6_tk_heldout_harmonic_anchor_k128.yaml
    )
    NAMES=(
      missing_tissue_wave6_tk_heldout_full_k128_seed10
      missing_tissue_wave6_tk_heldout_gex_only_k128_seed10
      missing_tissue_wave6_tk_heldout_he_only_k128_seed10
      missing_tissue_wave6_tk_heldout_neither_k128_seed10
      missing_tissue_wave6_tk_heldout_full_k16_seed10
      missing_tissue_wave6_tk_heldout_full_k32_seed10
      missing_tissue_wave6_tk_heldout_full_k64_seed10
      missing_tissue_wave6_tk_heldout_full_k128_no_spatial_bias_seed10
      missing_tissue_wave6_tk_heldout_harmonic_anchor_k128
    )
    SEEDS=(10 10 10 10 10 10 10 10 10)
    ;;
  wave6_heldout_st)
    CONFIGS=(
      configs/recovery_suite/92_wave6_st_heldout_full_k128.yaml
      configs/recovery_suite/93_wave6_st_heldout_gex_only_k128.yaml
      configs/recovery_suite/94_wave6_st_heldout_he_only_k128.yaml
      configs/recovery_suite/95_wave6_st_heldout_neither_k128.yaml
    )
    NAMES=(
      missing_tissue_wave6_st_heldout_full_k128_seed10
      missing_tissue_wave6_st_heldout_gex_only_k128_seed10
      missing_tissue_wave6_st_heldout_he_only_k128_seed10
      missing_tissue_wave6_st_heldout_neither_k128_seed10
    )
    SEEDS=(10 10 10 10)
    ;;
  wave7_st_smoke)
    # Three quick batches cover every distinct implementation path. Scalar-only
    # variants are exercised by the same constructors in the full marathon.
    CONFIGS=(
      configs/recovery_suite/92_wave6_st_heldout_full_k128.yaml
      configs/recovery_suite/116_wave7_st_modality_dropout.yaml
      configs/recovery_suite/117_wave7_st_foldb_full.yaml
      configs/recovery_suite/104_wave7_st_gnn_k8.yaml
      configs/recovery_suite/102_wave7_st_relative_position.yaml
      configs/recovery_suite/108_wave7_st_fusion_concat.yaml
      configs/recovery_suite/107_wave7_st_qk_norm.yaml
      configs/recovery_suite/106_wave7_st_transformer_large.yaml
      configs/recovery_suite/121_wave7_st_fm_builtin.yaml
      configs/recovery_suite/122_wave7_st_fm_stormlite.yaml
      configs/recovery_suite/123_wave7_st_fm_stpath_pretrained.yaml
      configs/recovery_suite/124_wave7_st_fm_stpath_scratch.yaml
    )
    NAMES=(
      missing_tissue_wave7_st_core_full_k128_seed10
      missing_tissue_wave7_st_modality_dropout_seed10
      missing_tissue_wave7_st_foldb_full_seed10
      missing_tissue_wave7_st_gnn_k8_seed10
      missing_tissue_wave7_st_relative_position_seed10
      missing_tissue_wave7_st_fusion_concat_seed10
      missing_tissue_wave7_st_qk_norm_seed10
      missing_tissue_wave7_st_transformer_large_seed10
      missing_tissue_wave7_st_fm_builtin_seed10
      missing_tissue_wave7_st_fm_stormlite_seed10
      missing_tissue_wave7_st_fm_stpath_pretrained_seed10
      missing_tissue_wave7_st_fm_stpath_scratch_seed10
    )
    SEEDS=(10 10 10 10 10 10 10 10 10 10 10 10)
    ;;
  wave7_st_marathon)
    # Eight sequential batches on four GPUs: core modalities, neighbourhood,
    # spatial/fusion, capacity, context budget, optimization robustness, a
    # second held-out patient direction, and matched FM/context encoders.
    CONFIGS=(
      configs/recovery_suite/92_wave6_st_heldout_full_k128.yaml
      configs/recovery_suite/93_wave6_st_heldout_gex_only_k128.yaml
      configs/recovery_suite/94_wave6_st_heldout_he_only_k128.yaml
      configs/recovery_suite/95_wave6_st_heldout_neither_k128.yaml
      configs/recovery_suite/97_wave7_st_full_k8.yaml
      configs/recovery_suite/98_wave7_st_full_k16.yaml
      configs/recovery_suite/99_wave7_st_full_k32.yaml
      configs/recovery_suite/100_wave7_st_full_k64.yaml
      configs/recovery_suite/101_wave7_st_no_spatial_bias.yaml
      configs/recovery_suite/102_wave7_st_relative_position.yaml
      configs/recovery_suite/103_wave7_st_fusion_sum.yaml
      configs/recovery_suite/104_wave7_st_gnn_k8.yaml
      configs/recovery_suite/105_wave7_st_transformer_1layer.yaml
      configs/recovery_suite/106_wave7_st_transformer_large.yaml
      configs/recovery_suite/107_wave7_st_qk_norm.yaml
      configs/recovery_suite/108_wave7_st_fusion_concat.yaml
      configs/recovery_suite/109_wave7_st_context384.yaml
      configs/recovery_suite/110_wave7_st_context512.yaml
      configs/recovery_suite/111_wave7_st_context1024.yaml
      configs/recovery_suite/112_wave7_st_context1536.yaml
      configs/recovery_suite/113_wave7_st_no_coord_augment.yaml
      configs/recovery_suite/114_wave7_st_lr1e4.yaml
      configs/recovery_suite/115_wave7_st_lr1e3.yaml
      configs/recovery_suite/116_wave7_st_modality_dropout.yaml
      configs/recovery_suite/117_wave7_st_foldb_full.yaml
      configs/recovery_suite/118_wave7_st_foldb_gex_only.yaml
      configs/recovery_suite/119_wave7_st_foldb_he_only.yaml
      configs/recovery_suite/120_wave7_st_foldb_neither.yaml
      configs/recovery_suite/121_wave7_st_fm_builtin.yaml
      configs/recovery_suite/122_wave7_st_fm_stormlite.yaml
      configs/recovery_suite/123_wave7_st_fm_stpath_pretrained.yaml
      configs/recovery_suite/124_wave7_st_fm_stpath_scratch.yaml
    )
    NAMES=(
      missing_tissue_wave7_st_core_full_k128_seed10
      missing_tissue_wave7_st_core_gex_only_k128_seed10
      missing_tissue_wave7_st_core_he_only_k128_seed10
      missing_tissue_wave7_st_core_neither_k128_seed10
      missing_tissue_wave7_st_full_k8_seed10
      missing_tissue_wave7_st_full_k16_seed10
      missing_tissue_wave7_st_full_k32_seed10
      missing_tissue_wave7_st_full_k64_seed10
      missing_tissue_wave7_st_no_spatial_bias_seed10
      missing_tissue_wave7_st_relative_position_seed10
      missing_tissue_wave7_st_fusion_sum_seed10
      missing_tissue_wave7_st_gnn_k8_seed10
      missing_tissue_wave7_st_transformer_1layer_seed10
      missing_tissue_wave7_st_transformer_large_seed10
      missing_tissue_wave7_st_qk_norm_seed10
      missing_tissue_wave7_st_fusion_concat_seed10
      missing_tissue_wave7_st_context384_seed10
      missing_tissue_wave7_st_context512_seed10
      missing_tissue_wave7_st_context1024_seed10
      missing_tissue_wave7_st_context1536_seed10
      missing_tissue_wave7_st_no_coord_augment_seed10
      missing_tissue_wave7_st_lr1e4_seed10
      missing_tissue_wave7_st_lr1e3_seed10
      missing_tissue_wave7_st_modality_dropout_seed10
      missing_tissue_wave7_st_foldb_full_seed10
      missing_tissue_wave7_st_foldb_gex_only_seed10
      missing_tissue_wave7_st_foldb_he_only_seed10
      missing_tissue_wave7_st_foldb_neither_seed10
      missing_tissue_wave7_st_fm_builtin_seed10
      missing_tissue_wave7_st_fm_stormlite_seed10
      missing_tissue_wave7_st_fm_stpath_pretrained_seed10
      missing_tissue_wave7_st_fm_stpath_scratch_seed10
    )
    SEEDS=(10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10)
    ;;
  verified_exact_st)
    CONFIGS=(
      configs/recovery_suite/125_verified_exact_harmonic_k32.yaml
      configs/recovery_suite/126_verified_exact_full_k32.yaml
      configs/recovery_suite/127_verified_exact_gex_only_k32.yaml
      configs/recovery_suite/128_verified_exact_neither_k32.yaml
    )
    NAMES=(
      missing_tissue_verified_exact_harmonic_k32
      missing_tissue_verified_exact_full_k32_seed10
      missing_tissue_verified_exact_gex_only_k32_seed10
      missing_tissue_verified_exact_neither_k32_seed10
    )
    SEEDS=(10 10 10 10)
    ;;
  simple_local)
    CONFIGS=(
      configs/recovery_suite/129_simple_local_harmonic_k32.yaml
      configs/recovery_suite/130_simple_local_gex_novae.yaml
      configs/recovery_suite/131_simple_local_full_novae.yaml
      configs/recovery_suite/132_simple_local_full_mlp.yaml
    )
    NAMES=(
      missing_tissue_simple_local_harmonic_k32
      missing_tissue_simple_local_gex_novae_seed10
      missing_tissue_simple_local_full_novae_seed10
      missing_tissue_simple_local_full_mlp_seed10
    )
    SEEDS=(10 10 10 10)
    ;;
  local_transformer)
    CONFIGS=(
      configs/recovery_suite/133_local_transformer_neither.yaml
      configs/recovery_suite/134_local_transformer_gex_novae.yaml
      configs/recovery_suite/135_local_transformer_full_novae.yaml
      configs/recovery_suite/136_local_mome_full_novae.yaml
    )
    NAMES=(
      missing_tissue_local_transformer_neither_seed10
      missing_tissue_local_transformer_gex_novae_seed10
      missing_tissue_local_transformer_full_novae_seed10
      missing_tissue_local_mome_full_novae_seed10
    )
    SEEDS=(10 10 10 10)
    ;;
  direct_stormlite)
    CONFIGS=(
      configs/recovery_suite/137_direct_transformer_neither.yaml
      configs/recovery_suite/138_direct_transformer_gex_novae.yaml
      configs/recovery_suite/139_direct_transformer_full_novae.yaml
      configs/recovery_suite/140_direct_mome_full_novae.yaml
    )
    NAMES=(
      missing_tissue_direct_transformer_neither_seed10
      missing_tissue_direct_transformer_gex_novae_seed10
      missing_tissue_direct_transformer_full_novae_seed10
      missing_tissue_direct_mome_full_novae_seed10
    )
    SEEDS=(10 10 10 10)
    ;;
  *)
    echo "ERROR: unknown recovery STAGE: $STAGE" >&2
    exit 2
    ;;
esac

if [[ ( "$STAGE" == "controls" || "$STAGE" == "ablations" || "$STAGE" == "wave3" ) && "$SMOKETEST" != "1" ]]; then
  # The direct harmonic-residual branch is an independent rejected
  # diagnostic. Wave 1 tests the historically working FM family and only
  # depends on the clean FM flagship being finite/noncollapsed.
  "$PYTHON_BIN" scripts/check_recovery_gate.py --mode flagship
fi

# All context-only Novae jobs consume the same immutable 64-mask schedule.
# Populate it once before concurrent readers start. The cache is reused safely.
if [[ "$SMOKETEST" != "1" ]] && [[ "$STAGE" == "repair" || "$STAGE" == "controls" || "$STAGE" == "ablations" || "$STAGE" == "wave3" || "$STAGE" == "component40k" || "$STAGE" == "missing_tissue" || "$STAGE" == "missing_tissue_controls" || "$STAGE" == "wave4_hr" || "$STAGE" == "wave4_fm" || "$STAGE" == "wave4b_hr" || "$STAGE" == "wave4c_controls" || "$STAGE" == "wave4d_hr" || "$STAGE" == "wave5_modalities_tk" || "$STAGE" == "wave5_modalities_st" ]]; then
  echo "Precomputing/reusing the shared context-only Novae mask cache on $GPU_COUNT GPUs..."
  precompute_pids=()
  for slot in "${!GPU_IDS_ARR[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU_IDS_ARR[$slot]}" \
      OMP_NUM_THREADS="$THREADS_PER_JOB" MKL_NUM_THREADS="$THREADS_PER_JOB" \
      OPENBLAS_NUM_THREADS="$THREADS_PER_JOB" NUMEXPR_NUM_THREADS="$THREADS_PER_JOB" \
      "$PYTHON_BIN" scripts/precompute_context_novae_cache.py \
        --config configs/recovery_suite/03_fixed_novae_flagship_regression.yaml \
        --shard-index "$slot" --num-shards "$GPU_COUNT" \
        > "$LOG_ROOT/precompute_context_novae_shard${slot}.log" 2>&1 &
    precompute_pids+=("$!")
  done
  precompute_failed=0
  for slot in "${!precompute_pids[@]}"; do
    if ! wait "${precompute_pids[$slot]}"; then
      precompute_failed=1
      echo "FAILED: context-only Novae precompute shard $slot" >&2
      tail -80 "$LOG_ROOT/precompute_context_novae_shard${slot}.log" >&2 || true
    fi
  done
  if (( precompute_failed )); then
    echo "Novae precomputation failed. Training was not started." >&2
    exit 1
  fi
fi

failed=0
JOB_COUNT="${#CONFIGS[@]}"
for (( batch_start=0; batch_start<JOB_COUNT; batch_start+=GPU_COUNT )); do
  batch_number=$((batch_start / GPU_COUNT + 1))
  batch_total=$(((JOB_COUNT + GPU_COUNT - 1) / GPU_COUNT))
  echo "Starting batch $batch_number/$batch_total with at most $GPU_COUNT jobs " \
       "and $THREADS_PER_JOB CPU threads per job."
  pids=()
  batch_names=()

  for (( local_slot=0; local_slot<GPU_COUNT; local_slot++ )); do
    job_index=$((batch_start + local_slot))
    (( job_index < JOB_COUNT )) || break
    config="${CONFIGS[$job_index]}"
    name="${NAMES[$job_index]}"
    seed="${SEEDS[$job_index]}"
    gpu="${GPU_IDS_ARR[$local_slot]}"
    checkpoint="results/checkpoints/recovery_suite/$name"
    metrics="$checkpoint/audit_test_metrics.json"
    if [[ "$STAGE" == "wave6_heldout_tk" || "$STAGE" == "wave6_heldout_st" || "$STAGE" == "wave7_st_smoke" || "$STAGE" == "wave7_st_marathon" || "$STAGE" == "verified_exact_st" || "$STAGE" == "simple_local" || "$STAGE" == "local_transformer" || "$STAGE" == "direct_stormlite" ]]; then
      metrics="$checkpoint/heldout_sample_summary.json"
    fi
    log="$LOG_ROOT/$name.log"

    if [[ "$FRESH" != "1" && -f "$metrics" ]]; then
      echo "GPU $gpu -> $name: SKIP (completed metrics found)"
      continue
    fi

    overrides=(
      "experiment_name=$name"
      "training.seed=$seed"
      "training.checkpoint_dir=$checkpoint"
    )
    if [[ "$SMOKETEST" == "1" ]]; then
      overrides+=(
        "training.epochs=$SMOKE_STEPS"
        "training.unique_mask_count=$SMOKE_STEPS"
        "training.checkpoint_every_n_steps=0"
        "training.checkpoint_dir=results/checkpoints/recovery_suite/smoke/${RUN_ID}/$name"
        "training.log_print_every_n_steps=1"
        "validation.every_n_steps=$SMOKE_STEPS"
        "validation.early_stopping_min_steps=$SMOKE_STEPS"
        "validation.patience_checks=1000"
        "validation.require_anchor_improvement=false"
        "validation.n_samples=1"
        "evaluation.n_validation_masks=1"
        "evaluation.n_test_masks=1"
        "evaluation.n_samples=1"
        "evaluation.mask_bank_path=results/mask_banks/recovery_suite/smoke_${name}.json"
        "evaluation.mask_bank_dir=results/mask_banks/recovery_suite/smoke_${name}"
        "evaluation.training_mask_bank_path=results/mask_banks/training/recovery_suite/smoke_${name}.json"
      )
    fi

    echo "GPU $gpu -> $name"
    (
      started_epoch="$(date +%s)"
      printf 'command:' > "$log"
      printf ' %q' env "CUDA_VISIBLE_DEVICES=$gpu" \
        "OMP_NUM_THREADS=$THREADS_PER_JOB" "MKL_NUM_THREADS=$THREADS_PER_JOB" \
        "OPENBLAS_NUM_THREADS=$THREADS_PER_JOB" "NUMEXPR_NUM_THREADS=$THREADS_PER_JOB" \
        "$PYTHON_BIN" -m src.training.train --config "$config" \
        --override "${overrides[@]}" >> "$log"
      printf '\n' >> "$log"
      printf 'started_at_utc: %s\n' "$(date -u +%FT%TZ)" >> "$log"
      if CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="$THREADS_PER_JOB" \
          MKL_NUM_THREADS="$THREADS_PER_JOB" \
          OPENBLAS_NUM_THREADS="$THREADS_PER_JOB" \
          NUMEXPR_NUM_THREADS="$THREADS_PER_JOB" \
          "$PYTHON_BIN" -m src.training.train --config "$config" \
          --override "${overrides[@]}" >> "$log" 2>&1; then
        run_status=0
      else
        run_status=$?
      fi
      finished_epoch="$(date +%s)"
      printf 'finished_at_utc: %s\n' "$(date -u +%FT%TZ)" >> "$log"
      printf 'wall_seconds: %d\n' "$((finished_epoch - started_epoch))" >> "$log"
      printf 'exit_status: %d\n' "$run_status" >> "$log"
      exit "$run_status"
    ) &
    pids+=("$!")
    batch_names+=("$name")
  done

  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      echo "DONE: ${batch_names[$i]}"
    else
      failed=1
      echo "FAILED: ${batch_names[$i]}" >&2
      tail -80 "$LOG_ROOT/${batch_names[$i]}.log" >&2 || true
    fi
  done

  if (( failed )); then
    echo "Stage $STAGE failed in batch $batch_number. Later batches were not started." >&2
    break
  fi
done

if (( failed )); then
  echo "Stage $STAGE failed. Do not promote. Logs: $LOG_ROOT" >&2
  exit 1
fi

if [[ "$STAGE" == "repair" && "$SMOKETEST" != "1" ]]; then
  "$PYTHON_BIN" scripts/check_recovery_gate.py
fi

collect_args=(
  --checkpoint-root results/checkpoints/recovery_suite
  --output-dir "reports/recovery_suite/${STAGE}_${RUN_ID}"
  --log-root "$LOG_ROOT"
)
if [[ "$STAGE" == "missing_tissue" ]]; then
  collect_args+=(--experiment-prefix missing_tissue_)
elif [[ "$STAGE" == "missing_tissue_controls" ]]; then
  collect_args+=(--experiment-prefix missing_tissue_control_)
elif [[ "$STAGE" == "wave4_hr" ]]; then
  collect_args+=(--experiment-prefix missing_tissue_wave4_hr_)
elif [[ "$STAGE" == "wave4_fm" ]]; then
  collect_args+=(--experiment-prefix missing_tissue_wave4_fm_)
elif [[ "$STAGE" == "wave4b_hr" ]]; then
  collect_args+=(--experiment-prefix missing_tissue_wave4b_hr_)
elif [[ "$STAGE" == "wave4c_controls" ]]; then
  collect_args+=(--experiment-prefix missing_tissue_wave4c_)
elif [[ "$STAGE" == "wave4d_hr" ]]; then
  collect_args+=(--experiment-prefix missing_tissue_wave4d_hr_)
elif [[ "$STAGE" == "wave5_modalities_tk" ]]; then
  collect_args+=(--experiment-prefix missing_tissue_wave5_tk_)
elif [[ "$STAGE" == "wave5_modalities_st" ]]; then
  collect_args+=(--experiment-prefix missing_tissue_wave5_st_)
elif [[ "$STAGE" == "wave6_heldout_tk" ]]; then
  collect_args+=(--experiment-prefix missing_tissue_wave6_tk_heldout_)
elif [[ "$STAGE" == "wave6_heldout_st" ]]; then
  collect_args+=(--experiment-prefix missing_tissue_wave6_st_heldout_)
elif [[ "$STAGE" == "wave7_st_smoke" || "$STAGE" == "wave7_st_marathon" ]]; then
  collect_args+=(--experiment-prefix missing_tissue_wave7_st_)
elif [[ "$STAGE" == "verified_exact_st" ]]; then
  collect_args+=(--experiment-prefix missing_tissue_verified_exact_)
elif [[ "$STAGE" == "simple_local" ]]; then
  collect_args+=(--experiment-prefix missing_tissue_simple_local_)
elif [[ "$STAGE" == "local_transformer" ]]; then
  collect_args+=(--experiment-prefix missing_tissue_local_)
elif [[ "$STAGE" == "direct_stormlite" ]]; then
  collect_args+=(--experiment-prefix missing_tissue_direct_)
fi
"$PYTHON_BIN" scripts/collect_audit_results.py "${collect_args[@]}"

echo "Stage $STAGE completed. Review reports/recovery_suite/${STAGE}_${RUN_ID}/summary.csv"
