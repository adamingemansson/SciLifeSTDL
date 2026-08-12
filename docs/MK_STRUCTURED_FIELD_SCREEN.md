# MK deterministic structured-field screen

This is a four-arm **H&E-only at inference** experiment. It is not a masked-hole
experiment and it is not a WAE experiment. All arms predict the same complete
normalized-log1p gene panel and differ only in the structured component being
tested.

## Arms

| Arm | Within-spot gene programs | Whole-slide H&E/coordinate attention + predicted-GEX graph refinement | Signed gradient loss |
|---|---:|---:|---:|
| `mk_field_within` | yes | no | no |
| `mk_field_between` | no | yes | no |
| `mk_field_gradient` | no | no | local + wider scale |
| `mk_field_combined` | yes | yes | local + wider scale |

Shared controls: frozen UNI2 cache and pinned revision, full gene panel, manifest
and patient-disjoint splits, masks, optimizer, dimensions, seed, training budget,
RMSE+PCC value objective, panels, whole-slide validation, and evaluator.

Comparison compatibility is fail-closed: established evaluator JSON fields,
TensorBoard scalar names, representative whole-slide images, fixed slides/genes,
colour scales and target/prediction/error layout remain unchanged. New
structured-field diagnostics are additive and live under separate namespaces;
they never replace the earlier PCC/RMSE/AUC or images.

The shared gene artifact is fitted from **training slides only**. Every slide is
gene-wise centered before fitting. Rows are weighted to give equal total weight
to each organ and equal weight to each slide within an organ. The artifact also
contains training-only within-slide per-gene scales for the gradient losses.
Its gene order, basis contents, scales, training samples, organs, and fitting
rules are hashed or recorded and checked at load time.

## Preparation order

Use the resolved paths from the intended MK cohort. Do not substitute an older
raw coexpression basis.

```bash
cd /data/adam.ingemansson/SciLifeSTDL-MK

PYTHON=/data/adam.ingemansson/miniforge3/envs/st3d/bin/python3
MANIFEST=/absolute/path/to/dataset_manifest.json
PANELS=/absolute/path/to/train_gene_panels.json
COMPARISON=/absolute/path/to/resolved_gen3_arch1_config.yaml
UNI2_CACHE=/data/adam.ingemansson/SciLifeSTDL/data/cache/hest1k/uni2_gen3_spot_cache
UNI2_REV=d517a8dd47902dd7c308b3c36f63bce47e7b9a43
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ROOT=/data/adam.ingemansson/SciLifeSTDL-MK/gen3_multiscale/results/mk_structured_field_${STAMP}
STRUCTURE=${ROOT}.centered_gene_structure_rank64.pt

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u \
  -m gen3_multiscale.scripts.fit_mk_centered_gene_structure \
  --manifest "$MANIFEST" \
  --output "$STRUCTURE" \
  --rank 64 \
  --seed 0 \
  --svd-device cuda

"$PYTHON" -u -m gen3_multiscale.scripts.prepare_mk_structured_field_suite \
  --comparison-config "$COMPARISON" \
  --manifest "$MANIFEST" \
  --train-gene-panels "$PANELS" \
  --centered-gene-structure "$STRUCTURE" \
  --output-root "$ROOT" \
  --uni2-pinned-revision "$UNI2_REV" \
  --uni2-spot-feature-cache-dir "$UNI2_CACHE" \
  --gpus 0,2,3,5 \
  --hours 8 \
  --cpu-threads 12

"$PYTHON" -m gen3_multiscale.scripts.run_mk_architecture_suite \
  --suite-root "$ROOT" --smoke --dry-run
```

After the dry-run and real-data smoke pass, remove `--smoke --dry-run` to launch.
Do not launch this screen until the currently running `refine0` measurement has
been preserved and its exact checkpoint/results root recorded.

## Interpretation

- Within > local baseline: explicit train-only gene programs help.
- Between > local baseline: joint spatial-field prediction helps.
- Gradient > local baseline: signed multiscale supervision preserves expression transitions.
- Combined > its individual arms: the mechanisms are complementary.
- Combined no better: retain the simplest winning component; do not add a WAE.

Select checkpoints using the same deterministic point-prediction validation
metric across arms, and report all genes, HVG-50, HVG-200, per-organ/per-slide
results, plus whole-slide maps. Gradient losses are training terms only; final
claims must use the shared evaluator.

## Final evaluation contract

The active expanded cohort's **final fixed-mask evaluation on the validation
split** currently resolves to **448 items** (14 validation samples x 4 strata x
8 masks per stratum). This is distinct from training-time masked validation
(currently 224 items: 14 x 4 x 4) and periodic whole-slide TensorBoard
validation. It replaces the older 9-sample/288-item final evaluation cohort.
PCC, RMSE and AUC are comparable only when resolved sample IDs and masks match.
The evaluator derives and records `n_samples`, `n_mask_strata`,
`n_masks_per_stratum_per_sample`, and `n_items` from the manifest/schedule. The
four arms also enable a separate deterministic H&E-only final evaluation over
every spot in every held-out validation slide:

- per-gene PCC/RMSE/AUC on the full panel and fixed training-derived panels;
- expression-profile PCC across genes within each spot;
- gene-gene correlation-matrix agreement on the fixed panels;
- per-gene Moran's-I agreement on the same local spatial graph;
- signed local (k=6) and wider (k=18) edge-gradient PCC, error, energy ratio,
  and direction agreement in training-only per-gene standard-deviation units;
- optional count-split noise-ceiling-normalized PCC, while retaining ordinary
  PCC as the primary comparable score.

Results are stored per slide, patient-macro aggregated with 95% confidence
intervals, and stratified by organ.  The evaluator never exposes held-out GEX
to the model, never selects genes from held-out targets, and does not use cell
type, ontology or pathology annotations.

```bash
PYTHON=/data/adam.ingemansson/miniforge3/envs/st3d/bin/python3

# Optional CPU-only measurability artifact, once per validation cohort.
"$PYTHON" -m gen3_multiscale.scripts.measure_gene_noise_ceiling \
  --config "$ROOT/configs/mk_field_within.yaml" \
  --split validation \
  --output "$ROOT/validation_noise_ceiling.json"

# Run once per arm; omit --noise-ceiling if the artifact was not measured.
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u \
  -m gen3_multiscale.evaluation.conditional_wae_evaluator \
  --config "$ROOT/configs/mk_field_within.yaml" \
  --checkpoint-dir "$ROOT/checkpoints/mk_field_within" \
  --output "$ROOT/evaluation/mk_field_within_validation.json" \
  --split validation --n-masks-per-stratum-per-sample 8 --device cuda \
  --noise-ceiling "$ROOT/validation_noise_ceiling.json"
```
