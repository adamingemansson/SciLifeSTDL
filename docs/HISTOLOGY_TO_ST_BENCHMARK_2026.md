# H&E-to-ST benchmark plan (2026)

The project now has two benchmark tracks. Their numbers must never be mixed.

## Track A — official HEST

Purpose: compare against published work. Use the official HEST tasks, crops,
folds, top-50 HVGs and PCC implementation without project-specific changes.
The initial required methods are H-Optimus-1 PCA/ridge, UNI2-h PCA/ridge,
GigaPath PCA/ridge, STFlow, STPath and TRIPLEX.

The official HEST repository's 2026-04-03 table is the clean external anchor:
PCA-256 plus ridge on official top-50 HVGs gives average PCC 0.4229 for
H-Optimus-1, 0.4197 for GenBio-PathFM, 0.4150 for H-Optimus-0, 0.4141 for
UNI2-h and 0.3875 for GigaPath. Those values are **not** thresholds for Track B.
They use different samples, panels and preprocessing.

## Track B — expanded exact split

Purpose: compare methods on the real project objective. Use the immutable
242-train/14-validation manifest, its 17,068-gene target vocabulary, the same
training-only panels, the same normalized-log1p target space and every held-out
spot exactly once. The primary score is patient-macro whole-slide gene-wise
PCC. The 448-item fixed-mask result remains a separately labelled stress test.

### Required point metrics

- gene-wise PCC and Spearman across spots;
- RMSE, MSE and MAE in normalized-log1p space;
- mean/median per-gene R2, retaining negative values rather than clipping;
- median/IQR gene PCC and fractions above 0, 0.1, 0.2 and 0.3;
- patient-macro estimates, paired slide deltas and 95% confidence intervals;
- all genes, train HVG-50/200 and within-slide-variance 50/200.

### Required structured metrics

- spot-profile PCC;
- gene-gene coexpression-matrix PCC/MAE;
- per-gene SSIM after the shared masked Visium rasterization;
- Moran's-I PCC/MAE;
- local/wide signed-gradient PCC, sign agreement and energy ratio;
- count-splitting noise-ceiling-adjusted PCC where available.

SSIM uses one method-independent rasterization: the median spot-neighbour
distance defines two raster pixels, spot values are Gaussian-splatted, local
moments are normalized by tissue support, and SSIM is averaged only at real
spot centers. Empty background is therefore never included in the mean.

## Method tiers

1. **Required, directly actionable:** frozen UNI2 and GigaPath PCA/ridge on
   Track B; the official HEST PCA/ridge leaderboard on Track A; the existing
   pretrained STPath evaluator on Track B.
2. **Required upstream reproductions:** STFlow and TRIPLEX. Their native paper
   tables are useful context, but only reruns under one of the two contracts
   enter our comparison table.
3. **Structure-focused audit set:** FLAG (explicit gene/spatial structural
   metrics), MERGE (hierarchical spot graph), M2TGLGO (gene-prior graph), and
   recent HiST/HistoGPA/DriftST/HyperST preprints. These are scientifically
   relevant, but their reported scores are not automatically comparable.

## First decision gate

Run the frozen UNI2 PCA-256/ridge control before further architecture work.
If MK does not improve its point metrics, continued complexity requires a
clear paired improvement in coexpression, Moran or gradient preservation. If
it improves neither, stop tuning that architecture family.

## Commands

```bash
python -u -m gen3_multiscale.evaluation.frozen_feature_ridge_evaluator \
  --config "$COMPARISON_CONFIG" \
  --manifest "$MANIFEST" \
  --train-gene-panels "$PANELS" \
  --image-encoder uni2 \
  --cache-dir "$UNI2_CACHE" \
  --pca-components 256 \
  --pca-spots-per-slide 128 \
  --ridge-spots-per-slide 2048 \
  --ridge-alpha 1.0 \
  --missing-image-policy zero \
  --linear-algebra-device cuda:0 \
  --output "$BENCHMARK_ROOT/uni2_pca256_ridge_validation.json"
```

`zero` is the primary MK-comparable policy: the verified cache's explicit
zero feature is used for an edge spot whose H&E patch is missing. The report
also records image coverage. Use `exclude` only as a labelled sensitivity
analysis.

Repeat with `--image-encoder gigapath` and its cache directory. Consolidate
reports without comparing incompatible tracks:

The nonlinear frozen-UNI2 control uses the same train-only PCA and exact
whole-slide evaluator:

```bash
python -u -m gen3_multiscale.evaluation.frozen_feature_mlp_evaluator \
  --config "$COMPARISON_CONFIG" \
  --manifest "$MANIFEST" \
  --train-gene-panels "$PANELS" \
  --image-encoder uni2 \
  --cache-dir "$UNI2_CACHE" \
  --pca-components 256 \
  --hidden-dim 512 \
  --device cuda \
  --linear-algebra-device cuda:0 \
  --output "$BENCHMARK_ROOT/uni2_pca256_mlp_validation.json"
```

If the ridge fit is already complete and only the metric implementation has
changed, reuse its immutable fit instead of reloading the 242 training slides:

```bash
python -u -m gen3_multiscale.evaluation.frozen_feature_ridge_artifact_evaluator \
  --config "$COMPARISON_CONFIG" \
  --manifest "$MANIFEST" \
  --train-gene-panels "$PANELS" \
  --image-encoder uni2 \
  --cache-dir "$UNI2_CACHE" \
  --fit-report "$BENCHMARK_ROOT/uni2_pca256_ridge_validation.json" \
  --fit-artifact "$BENCHMARK_ROOT/uni2_pca256_ridge_validation.model.npz" \
  --output "$BENCHMARK_ROOT/uni2_pca256_ridge_validation_rescored.json"
```

This path fails closed on manifest path, gene order, encoder, missing-image
policy, artifact dimensions, and feature-cache provenance. It evaluates only
the held-out slides.

Consolidate reports without comparing incompatible tracks:

```bash
python -m gen3_multiscale.scripts.summarize_hest_mk_benchmarks \
  "$MK_EVALUATION_ROOT" "$STPATH_EVALUATION_ROOT" "$BENCHMARK_ROOT" \
  --output "$BENCHMARK_ROOT/comparable_results.csv"
```

The versioned machine-readable contract and method registry are in
`configs/benchmarks/hest_mk_2026.yaml`.
