# Validity fixes and four-device audit suite

This document describes the clean benchmark path added after the repository
review. Historical configurations are retained for reproducibility, but their
scores must not be mixed with the audit suite unless they satisfy the same data,
masking, image-availability and sampling contracts.

## Scientific validity contract

### Expression preprocessing

The loader owns expression preprocessing. Every audit config explicitly uses:

```yaml
data:
  expression_transform: normalize_log1p
  expression_target_sum: 10000
```

The applied transformation is recorded in `adata.uns`. Reapplying the same
contract is an idempotent no-op; requesting a conflicting transform raises.
StormLite and STPath are told that their input is already log-transformed, so
there is no second `log1p` inside the model.

### Graph-derived gene features

Novae features computed on a complete slide are invalid for missing-expression
evaluation: graph message passing can carry hidden query expression into the
context embeddings. The default is therefore:

```yaml
data:
  novae_mode: disabled
```

`novae_mode: context_only` computes features on `adata[context_mask]` after the
query rows have been physically removed. It is expensive for fresh random masks
and is intended mainly for fixed-mask studies. Historical full-graph
reproduction requires both:

```yaml
data:
  novae_mode: unsafe_full_graph
  allow_unsafe_novae: true
```

Results from that mode are contaminated and must be labelled as such.

### Cohort and image availability

Expression-only controls can set `require_image_coverage: true`, which filters
them to the same barcode cohort that has H&E patches without loading image
pixels. This prevents image models from being compared on an easier or different
spot set.

H&E availability is an explicit experimental condition:

- `full`: context and query H&E are available;
- `target_zero`: H&E is unavailable only inside the missing target region;
- `all_zero`: no H&E is available anywhere;
- `shuffled`: query H&E is permuted as a shortcut diagnostic.

Unavailable images are accompanied by Boolean availability masks. Encoders use
a learned missing-image token rather than interpreting a zero tensor as tissue.
Training configs can apply query-only and all-image modality dropout.

### Validation and test separation

Mask banks are saved by barcode, not row number, and include fingerprints of
the coordinates, slice IDs, masking parameters, split counts and seeds. A stale
bank therefore raises instead of being silently reused. Validation and test
masks use separate seed ranges. Validation uses the same Monte-Carlo sampling
seeds at every checkpoint, so random draw difficulty cannot decide which
checkpoint wins. The callback selects the best validation state and can stop
after repeated non-improvement. The untouched test bank is evaluated only after
training.

Repeated experimentation on one slide still makes that slide a development
set. The final configs therefore use explicit sample groups:

- training: `INT1`–`INT8`;
- validation: `INT9`–`INT10`;
- test: `INT11`–`INT12`.

The shared output vocabulary is intersected across training slides only.
Validation and test slides are subsequently aligned to that immutable ordered
panel; a missing gene raises rather than allowing held-out data to shrink the
vocabulary. Test expression values are not used for preprocessing choices,
optimization or model selection.

### Stochastic evaluation

Generative models are sampled repeatedly for each immutable test mask. Metrics
are computed from the predictive mean, while predictive standard deviation,
90% interval coverage and interval width are recorded separately. ST-FID uses
one PCA basis and one effective dimensionality across all masks and image modes
within an experiment.

The configured domain-label diagnostic is named
`spatial_domain_plausibility`. Its default labels come from deterministic
PCA + MiniBatchKMeans; optional Leiden labels and genuinely curated labels are
recorded by source. Unsupervised-domain agreement is not curated cell-type
accuracy.

## Model naming

`control_fm_ot_stpath_frozen_backbone_custom_head` is a frozen STPath backbone
inside this repository's trainable completion system. It is not official
zero-shot STPath. `control_fm_ot_stpath_scratch_mlp_residual` is the comparable
scratch control.

The old decoder name `panel_invariant` remains as a deprecated alias. The honest
name is `gene_conditioned_vocabulary`: it decodes a fixed training vocabulary
and cannot predict unseen genes.

## New baselines and candidates

The suite starts with global mean, nearest-neighbour, local mean, inverse-distance
weighted and graph-harmonic interpolation. It also includes strict learned mean
and sum set summaries without attention, images or graph message passing.

`harmonic_residual` predicts a zero-initialized correction around the harmonic
anchor. Its initial output is exactly the deterministic anchor.

`residual_fm_ot` models expression residuals around the same anchor. Before any
residual-flow job runs, a separate autoencoder is trained specifically on
`query_expression - harmonic_anchor` targets constructed from training masks;
it is not trained on absolute expression. Single-slide validation uses the
fixed validation mask bank, while multi-slide pretraining uses training slides
for the gene panel and held-out validation slides for its quality gate. Test
slides are never loaded by the pretraining script. The residual-flow model
rejects checkpoints with the wrong target type, harmonic parameters, gene
order, width, latent size or RMSE/PCC gate, then freezes the validated encoder
and base decoder. A trainable conditional decoder starts as a zero correction.

## Running the suite on four devices

From the repository root:

```bash
DEVICES="0 1 2 3" bash scripts/run_audit_suite_4gpu.sh
```

Each subprocess sees one GPU through `CUDA_VISIBLE_DEVICES`; no job uses DDP.
At most four jobs run concurrently, and the script waits for an entire wave
before starting the next one:

| Wave | Purpose |
|---|---|
| 0 | single-slide and held-out-sample expression autoencoders |
| 1 | deterministic and set-summary baselines |
| 2 | clean FM-OT / StormLite / STPath controls |
| 3 | deterministic harmonic-residual candidates |
| 4 | pretrained residual flow-matching candidates |
| 5 | matched additional seeds |
| 6 | held-out-sample deterministic and residual-flow tests |

Useful controls:

```bash
DRY_RUN=1 bash scripts/run_audit_suite_4gpu.sh
SKIP_STPATH=1 bash scripts/run_audit_suite_4gpu.sh
SKIP_HELDOUT=1 bash scripts/run_audit_suite_4gpu.sh
CONTINUE_ON_ERROR=1 bash scripts/run_audit_suite_4gpu.sh
```

Logs are written under `logs/audit_suite/<UTC run id>/`. Checkpoints and full
metric JSON remain under `results/checkpoints/audit_suite/`. At the end,
`collect_audit_results.py` copies compact configs, manifests, validation
histories and metrics—never model weights or raw logs—to
`reports/audit_suite/<run id>/` so those summaries can be reviewed and committed.

## Interpreting results

Compare methods first on held-out samples and report all configured seeds. Do
not drop a seed because it underperforms. Image-model claims should include all
four image modes; a gain that disappears under shuffled or target-missing H&E
has a different interpretation from a robust expression-context gain.

Raw Novae-based historical scores, one-draw scores, and repeatedly reused
single masks are exploratory evidence only. They should not be placed in the
same headline table as this suite without rerunning under the validity contract.

## Reproducibility details

Each checkpoint directory contains the resolved configuration, configuration
SHA-256, Git revision/dirty state, Python and package versions, CUDA/runtime
metadata, device names, mask-bank paths and the exact training seed schedule.
Context-only Novae caches are keyed by context barcodes plus the gene panel,
preprocessing contract, expression checksum and feature-function identity.
Context-only Novae also enforces `training.num_workers: 0` to avoid copying
AnnData/model state into worker processes.

ST-FID values are comparable only when `effective_pca_components` is the same.
The evaluator fixes one PCA basis and one dimensionality across every mask and
image mode inside a run and writes that dimension into the result JSON.
