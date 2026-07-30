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

4. Resolve each Gen5 config with the manifest, frozen Gen4 conditioner,
   shared encoder checkpoint, shared decoder checkpoint, and arm-specific
   encoder/cache identities:

   ```bash
   python -m gen3_multiscale.scripts.resolve_gen45_config \
     --base-config gen3_multiscale/configs/gen5/gen5c.yaml \
     --output-config <run>/gen5c.yaml \
     --manifest <manifest.json> \
     --checkpoint-dir <run>/gen5c_checkpoints \
     --fingerprint gen4_conditioner_checkpoint=<selected-gen4-bundle> \
     --fingerprint expression_autoencoder_checkpoint=<run>/shared_autoencoder.pt \
     <arm-specific --fingerprint arguments>
   ```

5. Run a real-artifact staged smoke and fixed-mask capacity gate, then the
   full latent-flow run:

   ```bash
   python -m gen3_multiscale.training.train \
     --config <resolved_gen5.yaml> --smoke --staged-smoke

   python -m gen3_multiscale.scripts.step6_overfit_test \
     --config <resolved_gen5.yaml> \
     --sample-id <training-sample> \
     --checkpoint-dir <overfit-output>

   python -m gen3_multiscale.training.train \
     --config <resolved_gen5.yaml>
   ```

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
