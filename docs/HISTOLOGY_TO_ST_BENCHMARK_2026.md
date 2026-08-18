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

The official UNI2/GigaPath reproduction is prepared, but intentionally not
launched, in
`configs/benchmarks/hest_official_track_a_uni2_gigapath.yaml`. It pins the
official nine tasks, official top-50 gene lists, HEST normalization, PCA-256
and ridge. HEST downloads its benchmark data and extracts encoder embeddings
on the first fold; UNI2 and GigaPath model access must therefore be available
before launch. Record the HEST source commit and environment lock alongside
the output. The launch command, when explicitly approved, is:

```bash
cd /data/adam.ingemansson/SciLifeSTDL-MK16/HEST
python -u -m hest.bench.benchmark \
  --config ../configs/benchmarks/hest_official_track_a_uni2_gigapath.yaml
```

Do not run this command merely to update Track B: it is a separate external
benchmark contract and can download data and encoder weights.

## Track B — expanded exact split

Purpose: compare methods on the real project objective. Use the immutable
242-train/14-validation manifest, its 17,068-gene target vocabulary, the same
training-only panels, the same normalized-log1p target space and every held-out
spot exactly once. The primary score is patient-macro whole-slide gene-wise
PCC. The 448-item fixed-mask result remains a separately labelled stress test.

### Required point metrics

- mean/median gene-wise PCC and Spearman across spots (spatial localization);
- mean/median spot-profile PCC across genes (within-spot composition);
- slide mean-expression-profile PCC/Spearman across genes (global tissue
  composition, explicitly not raw-count pseudobulk);
- pooled flattened PCC as a labelled diagnostic only, never a ranking metric,
  because abundant genes and between-gene means can dominate it;
- RMSE, MSE and MAE in normalized-log1p space;
- mean/median per-gene R2, retaining negative values rather than clipping;
- median/IQR gene PCC and fractions above 0, 0.1, 0.2 and 0.3;
- patient-macro estimates, paired slide deltas and 95% confidence intervals;
- all genes, train HVG-50/200 and within-slide-variance 50/200.

The shared evaluator also emits formula-compatible versions of the six point
diagnostics used by HEtoSGEBench: per-gene PCC, normalized mutual information,
Jensen-Shannon divergence, range/standard-deviation-normalized RMSE, coordinate-free vector
SSIM and zero/nonzero AUC. Their mean and median are reported. The benchmark
paper's additional raw-count AUC thresholds (counts >1, >2, >5, and so on)
are not claimed here because Track B evaluates normalized-log1p targets and
predictions; zero/nonzero status is the only threshold preserved exactly by
that transform. Track B retains its stricter patient-macro aggregation and
fixed training-derived panels, so these fields must not be presented as a
numerical reproduction of that paper's cohort or cross-validation protocol.

Interpretation is directional: higher is better for PCC, Spearman, NMI, SSIM,
AUC and R2; lower is better for RMSE, MAE, JS divergence and NRMSE. Gradient
energy ratios are best near 1 rather than simply high. Pooled PCC and the
mean-expression-profile metrics describe calibration/composition but cannot
demonstrate that spatial gene patterns are localized correctly.

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
This coordinate-aware spatial SSIM remains distinct from the paper-compatible
`benchmark_gene_ssim`, which treats each gene as a one-dimensional vector and
does not use spot coordinates.

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
  --max-wall-clock-hours 8 \
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
  --output "$BENCHMARK_ROOT/uni2_pca256_ridge_validation_rescored.json" \
  --per-gene-diagnostics-output \
    "$BENCHMARK_ROOT/uni2_pca256_ridge_validation_rescored.per_gene.npz"
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

For the actual Track-B decision table, use the stricter auditor. It admits
only exact 14-slide/every-spot normalized-log1p H&E-only reports and writes all
rejection reasons. STPath remains visible but carries an explicit HEST-1k
pretraining-overlap caveat.

```bash
python -m gen3_multiscale.scripts.build_hest_mk_track_b_leaderboard \
  --ridge-report "$BENCHMARK_ROOT/uni2_pca256_ridge_validation_rescored.json" \
  --roots "$MK_EVALUATION_ROOT" "$STPATH_EVALUATION_ROOT" "$BENCHMARK_ROOT" \
  --output "$BENCHMARK_ROOT/track_b_whole_slide_leaderboard.tsv"
```

Add `--per-gene-diagnostics-output /path/to/METHOD.per_gene.npz` when
re-scoring each MK, ridge/MLP or H&E-only STPath report. Then explain which
genes drive each score and whether success tracks abundance, spatial
autocorrelation, gradient energy or the count-split ceiling:

```bash
python -m gen3_multiscale.scripts.analyze_hest_mk_per_gene \
  "$BENCHMARK_ROOT"/*.per_gene.npz \
  --noise-ceiling "$NOISE_CEILING" \
  --output-dir "$BENCHMARK_ROOT/per_gene_analysis"
```

The sidecar comparison fails if gene order, slide/patient identity, or any
target-derived attribute differs between methods. It reports breadth
(fractions of genes above PCC 0/0.1/0.2/0.3), top/bottom genes, and rank
associations between performance and held-out gene properties. These are
diagnostics; they must not be used to select training features.

After the finalist seed replications have been summarized, build the
predictability atlas directly from the saved seed-by-gene table (no model or
GPU is loaded):

```bash
python -m gen3_multiscale.scripts.build_mk_gene_predictability_atlas \
  --seed-gene-table "$GENE_ROBUSTNESS/per_gene_seed_robustness.tsv" \
  --architectures mk_wb_parallel_gated mk_wbw_sandwich \
  --ceiling-threshold 0.1 \
  --meaningful-delta 0.01 \
  --output-dir "$GENE_ROBUSTNESS/predictability_atlas"
```

The atlas ranks genes by their worst seed rather than their luckiest run,
reports an explicitly unbounded descriptive PCC/noise-ceiling ratio only for
eligible genes, and quantifies the per-gene oracle gain between the two
architectures. A small oracle gain plus high gene-PCC rank agreement means the
architectures are effectively redundant; it is not evidence for an ensemble.
Because abundance, variance, detection frequency, spatial autocorrelation,
gradient energy and the count-split ceiling are correlated, the atlas also
reports partial rank associations, standardized multivariable rank
coefficients and variance-inflation factors. These remain descriptive rather
than causal. `high_headroom_genes.tsv` prioritizes measurable genes whose
count-split ceiling is high relative to current reproducible PCC; this is a
held-out diagnostic and must not be converted into a training-panel choice.

The versioned machine-readable contract and method registry are in
`configs/benchmarks/hest_mk_2026.yaml`.
