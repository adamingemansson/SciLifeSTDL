# Gen3 manual release runbook

This is the deliberately small release path. Full-gene PCC/RMSE remain the
primary outcome. The train-derived top-50/top-200 panels are secondary views.
Architecture 4 is **not** independent: train/select Architecture 3 and fit its
training-only residual basis before resolving or starting Architecture 4.

## 1. One-time artifacts

Run from a clean checkout of the exact commit used for training:

```bash
conda activate st3d

export MANIFEST=/absolute/path/to/gen3_dataset_manifest.json
export TILE_REVISION=<40-lowercase-hex-pinned-HuggingFace-commit>
export GIGAPATH_SLIDE_CHECKPOINT="$PWD/STPath_weights/gigapath/slide_encoder.pth"
export RELEASE_ROOT="$PWD/gen3_multiscale/results/manual_$(date -u +%Y%m%dT%H%M%SZ)"
export PANEL_ARTIFACT="$RELEASE_ROOT/train_gene_panels.json"
export SYNC_INIT="$RELEASE_ROOT/synchronized_initialization"
mkdir -p "$RELEASE_ROOT"

python -m gen3_multiscale.scripts.build_train_gene_panels \
  --manifest "$MANIFEST" --output "$PANEL_ARTIFACT"

CUDA_VISIBLE_DEVICES=1 python -m gen3_multiscale.scripts.prepare_manual_initializations \
  --manifest "$MANIFEST" \
  --gigapath-slide-checkpoint "$GIGAPATH_SLIDE_CHECKPOINT" \
  --output-dir "$SYNC_INIT"
```

## 2. Resolve immutable Architecture 1-3 configs

Choose a short diagnostic (for example 10,000 steps / 2 hours) before the
24-hour run. Use the same values for all arms.

```bash
for A in 1 2 3; do
  extra=()
  if [[ "$A" == 3 ]]; then
    extra+=(--gigapath-checkpoint "$GIGAPATH_SLIDE_CHECKPOINT")
  fi
  python -m gen3_multiscale.scripts.resolve_experiment_config \
    --base-config "gen3_multiscale/configs/architecture${A}.yaml" \
    --output-dir "$RELEASE_ROOT/config_arch${A}" \
    --gen3-manifest-path "$MANIFEST" \
    --tile-encoder-revision "$TILE_REVISION" \
    --synchronized-init-dir "$SYNC_INIT" \
    --train-gene-panel-artifact "$PANEL_ARTIFACT" \
    --checkpoint-dir "$RELEASE_ROOT/checkpoints_arch${A}" \
    --total-steps 10000 --max-wall-clock-hours 2 --seed 10 \
    "${extra[@]}"
done
```

First run construction/one-step smoke tests (sequentially is easiest to read):

```bash
for A in 1 2 3; do
  CUDA_VISIBLE_DEVICES=$A OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python -m gen3_multiscale.training.train \
      --config "$RELEASE_ROOT/config_arch${A}/config.yaml" --smoke
done
```

Then start the three independent learned models in separate tmux windows:

```bash
tmux new-session -d -s gen3_arch1 \
  "cd '$PWD' && conda run --no-capture-output -n st3d env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m gen3_multiscale.training.train --config '$RELEASE_ROOT/config_arch1/config.yaml' 2>&1 | tee '$RELEASE_ROOT/arch1.log'"
tmux new-session -d -s gen3_arch2 \
  "cd '$PWD' && conda run --no-capture-output -n st3d env CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m gen3_multiscale.training.train --config '$RELEASE_ROOT/config_arch2/config.yaml' 2>&1 | tee '$RELEASE_ROOT/arch2.log'"
tmux new-session -d -s gen3_arch3 \
  "cd '$PWD' && conda run --no-capture-output -n st3d env CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m gen3_multiscale.training.train --config '$RELEASE_ROOT/config_arch3/config.yaml' 2>&1 | tee '$RELEASE_ROOT/arch3.log'"
```

GPU 5 should remain free at this stage. Starting Architecture 4 there now
would invalidate the intended Architecture-3-conditioned comparison.

## 3. Select Architecture 3 and fit Architecture 4's basis

Use validation only—never test—for selection and basis construction:

```bash
CUDA_VISIBLE_DEVICES=5 python -m gen3_multiscale.evaluation.gen3_evaluator \
  --config "$RELEASE_ROOT/config_arch3/config.yaml" \
  --checkpoint-dir "$RELEASE_ROOT/checkpoints_arch3" \
  --output "$RELEASE_ROOT/arch3_validation.json" \
  --split validation --use-best --device cuda

CUDA_VISIBLE_DEVICES=5 python -m gen3_multiscale.scripts.fit_architecture4_residual_basis \
  --config "$RELEASE_ROOT/config_arch3/config.yaml" \
  --architecture3-checkpoint-dir "$RELEASE_ROOT/checkpoints_arch3/best" \
  --output-basis-path "$RELEASE_ROOT/architecture4_basis.pt" \
  --n-masks-per-sample 20 --rank 64 --device cuda

python -m gen3_multiscale.scripts.resolve_experiment_config \
  --base-config gen3_multiscale/configs/architecture4.yaml \
  --output-dir "$RELEASE_ROOT/config_arch4" \
  --gen3-manifest-path "$MANIFEST" \
  --tile-encoder-revision "$TILE_REVISION" \
  --synchronized-init-dir "$SYNC_INIT" \
  --train-gene-panel-artifact "$PANEL_ARTIFACT" \
  --checkpoint-dir "$RELEASE_ROOT/checkpoints_arch4" \
  --gigapath-checkpoint "$GIGAPATH_SLIDE_CHECKPOINT" \
  --gene-residual-basis "$RELEASE_ROOT/architecture4_basis.pt" \
  --architecture3-conditioner-checkpoint "$RELEASE_ROOT/checkpoints_arch3/best" \
  --total-steps 10000 --max-wall-clock-hours 2 --seed 10

CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m gen3_multiscale.training.train \
    --config "$RELEASE_ROOT/config_arch4/config.yaml" --smoke --staged-smoke

tmux new-session -d -s gen3_arch4 \
  "cd '$PWD' && conda run --no-capture-output -n st3d env CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m gen3_multiscale.training.train --config '$RELEASE_ROOT/config_arch4/config.yaml' 2>&1 | tee '$RELEASE_ROOT/arch4.log'"
```

## 4. Monitor and evaluate

```bash
for A in 1 2 3 4; do
  python -m gen3_multiscale.scripts.step6_progress \
    --checkpoint-dir "$RELEASE_ROOT/checkpoints_arch${A}"
done

for A in 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=1 python -m gen3_multiscale.evaluation.gen3_evaluator \
    --config "$RELEASE_ROOT/config_arch${A}/config.yaml" \
    --checkpoint-dir "$RELEASE_ROOT/checkpoints_arch${A}" \
    --output "$RELEASE_ROOT/arch${A}_validation.json" \
    --split validation --use-best --device cuda
done
```

Promote to 24 hours only if the capacity/smoke gates pass, losses are finite,
and validation beats the matched mean/nearest-neighbour/harmonic baselines.
After all decisions are frozen, run the test split exactly once with
`--split test --allow-test`.
