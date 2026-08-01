# Gen5 runbook

Gen5 uses the same four conditioning systems as Gen4, but the flow
generates a complete gene-expression vector in the latent space of one
shared expression autoencoder:

| Scientific arm | Config key | Conditioning |
|---|---|---|
| 1 | `gen5c` | UNI2 + scFoundation |
| 2 | `gen5b` | GigaPath + scFoundation |
| 3 | `gen5d` | STPath joint conditioner |
| 4 | `gen5e` | STPath + UNI2 + scFoundation |

`gen5a` is the optional UNI2 + weighted-linear-GEX baseline.

## Real run order

1. Build and validate the caches described in
   `gen3_multiscale/gen4/RUNBOOK.md`.

2. Train the shared autoencoder once, from training-split expression only:

   ```bash
   python -m gen3_multiscale.scripts.train_gen5_autoencoder \
     --manifest <manifest.json> \
     --output-checkpoint <run>/shared_autoencoder.pt \
     --train-gene-panel-artifact <train_gene_panels.json> \
     --latent-dim 256 --hidden-dim 1024 \
     --epochs 50 --batch-size 64 --device cuda
   ```

   The implementation uses a disk-backed expression matrix and transfers
   only one mini-batch to the GPU. Inspect the emitted reconstruction
   report before proceeding; this is the ceiling imposed on every Gen5
   arm by the shared decoder.

3. Train and validation-select the matching Gen4 deterministic conditioner
   for each arm. Gen5 must consume that exact verified immutable bundle,
   not a freshly initialized or differently selected conditioner.

4. Prepare all four primary configs together. This verifies the manifest,
   train-derived panels, shared autoencoder, selected conditioners, frozen
   model files, and every required per-sample cache before publishing one
   immutable run root. Each mutable Gen4 `best/` input is resolved once;
   the generated config records its exact immutable bundle.

   ```bash
   python -m gen3_multiscale.scripts.prepare_gen5_suite \
     --manifest <manifest.json> \
     --cache-root <hest-cache-root> \
     --train-gene-panels <train_gene_panels.json> \
     --autoencoder-checkpoint <autoencoder-run>/shared_autoencoder.pt \
     --conditioner gen5c=<gen4c-checkpoints>/best \
     --conditioner gen5b=<gen4b-checkpoints>/best \
     --conditioner gen5d=<gen4d-checkpoints>/best \
     --conditioner gen5e=<gen4e-checkpoints>/best \
     --output-root <new-gen5-run-root> \
     --uni2-checkpoint <UNI2-weights> \
     --scfoundation-checkpoint <scFoundation-weights> \
     --scfoundation-vocab <scFoundation-vocab> \
     --stpath-checkpoint <STPath-weights> \
     --stpath-gene-vocab <STPath-gene-vocab> \
     --gigapath-checkpoint <GigaPath-LongNet-weights> \
     --total-steps 100000000 --max-wall-clock-hours 24 --device cuda
   ```

5. Run a real-artifact staged smoke and fixed-mask capacity gate, then the
   full latent-flow run:

   ```bash
   python -m gen3_multiscale.training.train \
     --config <gen5-run-root>/configs/gen5c.yaml --smoke --staged-smoke

   python -m gen3_multiscale.scripts.step6_overfit_test \
     --config <gen5-run-root>/configs/gen5c.yaml \
     --sample-id <training-sample> \
     --checkpoint-dir <overfit-output>

   python -m gen3_multiscale.training.train \
     --config <gen5-run-root>/configs/gen5c.yaml
   ```

   Repeat the gate for `gen5b`, `gen5d`, and `gen5e`. Do not launch a
   long run for an arm that fails either gate.

6. Evaluate on validation using the common evaluator. Compare the Gen5
   latent flow with its matched Gen4 residual flow, its deterministic
   conditioner, and the same mean/nearest-neighbor/harmonic baselines.
   Use the test split once only after selection is frozen.

## Reliability gates

- Gen5 reuses the same arm-specific real-data adapter, query exclusion,
  cache coverage/provenance checks, mask schedule, and metric pipeline as
  Gen4/Gen3.
- The autoencoder checkpoint is bound to the frozen gene order, dataset,
  code state, and numeric checkpoint SHA256.
- The frozen Gen4 conditioner is resolved once, identity-verified, and
  pinned for the run.
- Only the latent velocity network trains; both conditioner and
  autoencoder remain frozen and in evaluation mode.
- Validation predictions use stable per-item generators. Reports retain
  exact checkpoint identity and full-gene, train-derived top-50/top-200,
  per-stratum, nonzero-AUC, paired-baseline, and uncertainty metrics where
  applicable.

## Required hardware validation

Before long training, run the staged smoke with the real external weights,
then verify autoencoder reconstruction, one real optimizer step, finite
gradients, memory use, and a fixed-mask overfit improvement. The local
test suite cannot replace this target-GPU real-weight gate.
