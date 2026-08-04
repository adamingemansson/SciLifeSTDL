# Conditional H&E-to-GEX WAE project

This is separate from the existing objective, where both H&E and GEX disappear
inside a hole. The supervisor tasks keep target-region H&E visible:

- Task I, `H&E -> ST`: full H&E is visible; no GEX context is supplied.
- Task II, `H&E + ST -> ST`: full H&E is visible everywhere, GEX is zeroed in
  the query region, and the verified boundary/local surrounding GEX context
  remains available.

## Matched first comparison

- `regularizer=mmd`: conditional WAE with IMQ-MMD aggregate-posterior matching.
- `regularizer=gan`: the same model with an adversarial latent discriminator.

Everything else is shared: Architecture-1-derived image tokenization and
spatial attention, expression encoder, deterministic conditional-mean head,
conditional residual decoder, RMSE+PCC reconstruction loss, data split and evaluation.

The input schema can contain compact observed surrounding-GEX rows for Task II,
but validates that their indices exactly exclude every query row. During training only,
real query GEX is encoded to `z` and used as the reconstruction target. At
inference, `z` is sampled from `N(0,I)` and combined with the legal context. The
conditional-mean head is trained explicitly to reduce the risk that the decoder
ignores its legal conditioning inputs and simply copies information through the
latent code. For Task I this branch is H&E-only; for Task II it uses H&E plus
the permitted surrounding ST context.

Architecture 1's weighted GEX token branch is active only for Task II's legal
surrounding ST context. Its gene-transport head is replaced by the conditional
WAE decoder. Architecture 2's harmonic anchor is excluded. The project reuses
the leakage-safe image cache, Fourier coordinates, relative-geometry spatial
attention, manifest splits, full frozen gene panel and shared metrics.

## Executable gates

```bash
python -m gen3_multiscale.scripts.conditional_wae_smoke_launcher
```

Prepare the four matched resolved configs without launching anything:

```bash
python -m gen3_multiscale.scripts.prepare_conditional_wae_suite \
  --comparison-config /path/to/resolved_architecture1.yaml \
  --manifest /path/to/dataset_manifest.json \
  --train-gene-panels /path/to/train_gene_panels.json \
  --output-root /path/to/new_conditional_wae_suite
```

Then run a real-data one-step gate for one generated config:

```bash
python -m gen3_multiscale.training.train_conditional_wae \
  --config /path/to/new_conditional_wae_suite/configs/wae_he_mmd.yaml --smoke
```

Or smoke all four simultaneously on four GPUs (the launcher does not run
unless invoked explicitly):

```bash
python -m gen3_multiscale.scripts.run_conditional_wae_suite \
  --suite-root /path/to/new_conditional_wae_suite \
  --gpus 0,1,2,3 --smoke
```

The evaluator always predicts from legal inference inputs and sampled prior
latents; it never selects a checkpoint using target-encoded reconstruction.
It reports the sampled conditional model and the explicit deterministic
conditional-mean head separately, including configured HVG-50/HVG-200 panels.

## TensorBoard diagnostics

Generated suite configs enable direct TensorBoard logging under
`SUITE_ROOT/tensorboard/ARM`. Scalars are written at the existing training and
validation boundaries. Every fifth validation reuses that same validation pass
to record a fixed, bounded cohort (at most 5,000 query spots):

- conditional-context and held-out target-posterior embeddings for Projector;
- sample, patient, organ, mask stratum, barcode and raw coordinate metadata;
- a 512-point subset with aligned, downsampled H&E patch thumbnails;
- per-slide context/posterior PCA maps, predictive-RMSE maps, and target versus
  prediction maps for two train-derived HVG genes.

The `posterior_z_target_diagnostic` embedding deliberately uses held-out target
GEX to inspect representation structure. It is diagnostic only: it is never an
inference input and cannot affect validation loss or checkpoint selection.
Projector computes PCA/UMAP interactively; the spatial PCA maps preserve the
original spot coordinates so clusters can be related back to tissue regions.

Launch TensorBoard on the server with any free port, for example:

```bash
python -m tensorboard.main --logdir /path/to/SUITE_ROOT/tensorboard \
  --host 127.0.0.1 --port 38435 --load_fast=false
```
