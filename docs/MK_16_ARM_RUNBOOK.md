# MK 16-arm, two-wave runbook

This screen contains four matched four-arm batches. Nothing is launched by the
preparation command.

| Batch | Arms | Question |
|---|---|---|
| A: architecture controls | `mk_local_wae_mmd`, `mk_spatial_deterministic`, `mk_local_conditional_wae_mmd`, `mk_spatial_conditional_wae_mmd` | local vs spatial; deterministic vs standard/conditional WAE |
| B: deterministic structure | `mk_field_within`, `mk_field_between`, `mk_field_gradient`, `mk_field_combined` | within-spot, between-spot and gradient mechanisms without a latent sampler |
| C: standard WAE structure | `wae_within`, `wae_between`, `wae_gradient`, `wae_combined` | the same structured mechanisms with WAE-MMD and a standard prior |
| D: conditional WAE structure | `cwae_within`, `cwae_between`, `cwae_gradient`, `cwae_combined` | the same mechanisms with an image-conditioned prior |

All arms are H&E-only at inference. Held-out query GEX is never visible. All use
the same expanded manifest, full gene panel, frozen/pinned UNI2 cache, masks,
optimizer, seed, budget, TensorBoard/evaluation schema and train-only centered
gene-structure artifact.

Raw H&E patch tensors are retained only while their UNI2-cache provenance is
verified and are then released (`data.retain_patches_in_memory: false`). The
compact UNI2 features remain resident. This is the expanded-cohort RAM fix and
is set explicitly in every generated arm.

## Scheduling

- Wave 1 runs Batch A + B: eight processes concurrently, two on each of GPUs
  `0,2,3,5`.
- Wave 2 begins only after every Wave 1 process exits successfully, and runs
  Batch C + D with the same placement.
- Each process is capped at six CPU threads, so a wave requests at most 48 CPU
  threads.
- Every four-arm batch has its own TensorBoard log root and server.
- A failed Wave 1 blocks Wave 2. Accidental duplicate controllers and duplicate
  TensorBoard servers fail closed.

## Prepare only

The expanded cohort currently has 242 training slides, 14 validation slides and
17,068 genes. Fit its missing shared structure once before suite preparation.
The fitter deterministically samples at most 512 spots per training slide,
records the source/selected row counts and selection hash, and releases every
slide's raw H&E patches immediately. This bounds the randomized-SVD matrix at
at most 123,904 rows instead of pooling every spot from all 242 slides.

```bash
cd /data/adam.ingemansson/SciLifeSTDL-MK16
conda activate st3d

PYTHON=/data/adam.ingemansson/miniforge3/envs/st3d/bin/python3
MANIFEST=/data/adam.ingemansson/SciLifeSTDL/data/cache/hest1k/gen3_multiorgan7_expanded_manifest_seed1.json
STRUCTURE_ROOT=/data/adam.ingemansson/SciLifeSTDL-MK16/gen3_multiscale/results/mk_centered_structure_expanded_rank64
STRUCTURE="$STRUCTURE_ROOT/centered_gene_structure_rank64.pt"
mkdir -p "$STRUCTURE_ROOT"
set -o pipefail

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u \
  -m gen3_multiscale.scripts.fit_mk_centered_gene_structure \
  --manifest "$MANIFEST" \
  --output "$STRUCTURE" \
  --rank 64 \
  --seed 0 \
  --max-spots-per-slide 512 \
  --svd-device cuda:0 \
  2>&1 | tee "$STRUCTURE_ROOT/fit.log"

"$PYTHON" -m gen3_multiscale.scripts.discover_mk_16_inputs
```

Run the preparation command below only after that command saves and validates
the structure artifact. It does not start model training.

```bash
cd /data/adam.ingemansson/SciLifeSTDL-MK16

PYTHON=/data/adam.ingemansson/miniforge3/envs/st3d/bin/python3
COMPARISON=/data/adam.ingemansson/SciLifeSTDL/gen3_multiscale/results/gen3_full_harmonicfix_20260730T150307Z/config_arch1/config.yaml
MANIFEST=/data/adam.ingemansson/SciLifeSTDL/data/cache/hest1k/gen3_multiorgan7_expanded_manifest_seed1.json
PANELS=/data/adam.ingemansson/SciLifeSTDL-MK/gen3_multiscale/results/train_gene_panels_v3.json
STRUCTURE=/data/adam.ingemansson/SciLifeSTDL-MK16/gen3_multiscale/results/mk_centered_structure_expanded_rank64/centered_gene_structure_rank64.pt
UNI2_CACHE=/data/adam.ingemansson/SciLifeSTDL/data/cache/hest1k/uni2_gen3_spot_cache
UNI2_REV=d517a8dd47902dd7c308b3c36f63bce47e7b9a43
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=/data/adam.ingemansson/SciLifeSTDL-MK16/gen3_multiscale/results/mk_16_arm_${STAMP}

for path in "$PYTHON" "$COMPARISON" "$MANIFEST" "$PANELS" "$STRUCTURE" "$UNI2_CACHE"; do
  [[ -e "$path" ]] || { echo "MISSING: $path"; exit 1; }
done

"$PYTHON" -u -m gen3_multiscale.scripts.prepare_mk_16_arm_suite \
  --comparison-config "$COMPARISON" \
  --manifest "$MANIFEST" \
  --train-gene-panels "$PANELS" \
  --centered-gene-structure "$STRUCTURE" \
  --output-root "$ROOT" \
  --uni2-pinned-revision "$UNI2_REV" \
  --uni2-spot-feature-cache-dir "$UNI2_CACHE" \
  --gpus 0,2,3,5 \
  --hours 8 \
  --cpu-threads-per-arm 6 \
  --latent-dim 64

"$PYTHON" -m gen3_multiscale.scripts.run_mk_16_arm_suite \
  --suite-root "$ROOT" --dry-run

echo "$ROOT"
```

Inspect `master_plan.json`. It must contain 16 unique arms, two waves of eight,
and `gpu_process_counts` equal to two for every listed GPU.

## Smoke, TensorBoard and full launch (later)

Do not run these until the currently active experiment is finished.

```bash
cd /data/adam.ingemansson/SciLifeSTDL-MK16
PYTHON=/data/adam.ingemansson/miniforge3/envs/st3d/bin/python3
ROOT=$(cat gen3_multiscale/results/LATEST_MK_16_ARM_SUITE_ROOT.txt)

# All 16 real-data one-step smokes, in the same two-wave placement.
"$PYTHON" -u -m gen3_multiscale.scripts.run_mk_16_arm_suite \
  --suite-root "$ROOT" --smoke

# Four independent TensorBoard servers; prints all four ports.
"$PYTHON" -u -m gen3_multiscale.scripts.start_mk_16_tensorboards \
  --suite-root "$ROOT" --ports 55002,55003,55004,55005

# Full two-wave training controller. Run under nohup only after smoke passes.
nohup "$PYTHON" -u -m gen3_multiscale.scripts.run_mk_16_arm_suite \
  --suite-root "$ROOT" \
  > "$ROOT/control/training_controller.log" 2>&1 &
echo $! > "$ROOT/control/training_controller.pid"
```

If a controller genuinely ended after a failure, inspect the status/logs and
resume only unfinished arms explicitly:

```bash
"$PYTHON" -u -m gen3_multiscale.scripts.run_mk_16_arm_suite \
  --suite-root "$ROOT" --resume-controller
```

## One-shot monitor

```bash
"$PYTHON" -m gen3_multiscale.scripts.monitor_mk_16_arm_suite \
  --suite-root "$ROOT" --wave 1 --follow --interval 10 --tail-lines 3
```

Use `--wave 2` in a second terminal when Wave 2 starts. The fixed TensorBoard
mapping is Batch A=`55002`, Batch B=`55003`, Batch C=`55004`, Batch D=`55005`.

Final fixed-mask evaluation is separate from training-time validation and uses
the expanded validation cohort: 14 samples x 4 strata x 8 masks = 448 items per
arm. Whole-slide evaluation predicts every tissue spot in every held-out slide.
