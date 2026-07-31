# Gen4 runbook

Gen4 compares four conditioning systems under the same deterministic
conditioner and residual-flow training/evaluation pipeline:

| Scientific arm | Config key | Conditioning |
|---|---|---|
| 1 | `gen4c` | UNI2 + scFoundation |
| 2 | `gen4b` | GigaPath + scFoundation |
| 3 | `gen4d` | STPath joint conditioner |
| 4 | `gen4e` | STPath + UNI2 + scFoundation |

`gen4a` (UNI2 + weighted-linear GEX) is an optional baseline, not one of
the four primary arms.

## Real run order

Use one immutable dataset manifest and the same mask bank, gene panel,
split, optimizer settings, and evaluation protocol for all arms.

1. Precompute the required frozen features once:

   ```bash
   python -m gen3_multiscale.scripts.precompute_gen45_features \
     --manifest <manifest.json> \
     --modalities both \
     --uni2-checkpoint <UNI2-h.bin> \
     --uni2-revision <40-hex-commit> \
     --scfoundation-checkpoint <scfoundation.ckpt> \
     --scfoundation-vocab <scFoundation/OS_scRNA_gene_index.19264.tsv> \
     --scfoundation-repo <scFoundation-repository> \
     --scfoundation-revision <40-hex-commit> \
     --device cuda \
     --report <cache-report.json>
   ```

   Use `--shard-index I --n-shards N` to split cache construction across
   GPUs. The command loads each encoder once and writes provenance-bound
   caches. The scFoundation repository must be the official
   `biomap-research/scFoundation` checkout at exactly the declared commit;
   the loader uses its real `model/load.py` API rather than assuming an
   installable package that the official release does not provide. The
   report records the exact checkpoint/vocabulary/code/preprocessing
   identities to use when resolving configs. Arms using GigaPath/STPath
   additionally consume the existing provenance-bound GigaPath caches.

2. Resolve each conditioner config. Supply only the artifacts required by
   that arm; resolution fails if required values remain unset:

   ```bash
   python -m gen3_multiscale.scripts.resolve_gen45_config \
     --base-config gen3_multiscale/configs/gen4/gen4c_conditioner.yaml \
     --output-config <run>/gen4c_conditioner.yaml \
     --manifest <manifest.json> \
     --checkpoint-dir <run>/gen4c_conditioner_checkpoints \
     --fingerprint uni2_checkpoint=<UNI2-h.bin> \
     --fingerprint uni2_revision=<40-hex-commit> \
     --fingerprint scfoundation_checkpoint=<scfoundation.ckpt> \
     --fingerprint scfoundation_vocab=<vocab.json> \
     --fingerprint uni2_package_version=<timm-version> \
     --fingerprint uni2_preprocessing_spec=<spec> \
     --fingerprint scfoundation_package_version=<package-version> \
     --fingerprint scfoundation_preprocessing_spec=<spec>
   ```

3. Run the real-artifact staged smoke, then the fixed-mask capacity gate:

   ```bash
   python -m gen3_multiscale.training.train \
     --config <resolved_conditioner.yaml> --smoke --staged-smoke

   python -m gen3_multiscale.scripts.step6_overfit_test \
     --config <resolved_conditioner.yaml> \
     --sample-id <training-sample> \
     --checkpoint-dir <overfit-output>
   ```

4. Train the conditioner and select its best checkpoint using validation
   only:

   ```bash
   python -m gen3_multiscale.training.train \
     --config <resolved_conditioner.yaml>

   python -m gen3_multiscale.evaluation.gen3_evaluator \
     --config <resolved_conditioner.yaml> \
     --checkpoint-dir <conditioner-checkpoints> \
     --output <validation-report.json> \
     --split validation
   ```

5. Fit the rank-64 residual basis from training masks only:

   ```bash
   python -m gen3_multiscale.scripts.fit_gen4_residual_basis \
     --config <resolved_conditioner.yaml> \
     --conditioner-checkpoint-dir <selected-conditioner-bundle> \
     --output-basis-path <run>/basis.pt \
     --n-masks-per-sample 20 --rank 64 --device cuda
   ```

6. Resolve the matching flow config with the exact selected conditioner
   bundle and basis (`gen4_conditioner_checkpoint` and
   `gene_residual_basis`), then repeat staged smoke, capacity gate, full
   training, and validation evaluation.

7. Compare arms on validation. Touch the test split once, only after the
   model-selection rule is frozen.

## Reliability gates

- The real trainer/evaluator run arm-specific cache coverage and provenance
  checks before constructing a model, optimizer, or DataLoader.
- Query spots and H&E-overlapping rows are absent/zeroed in every context
  modality.
- Flow runs verify and freeze the exact selected conditioner bundle and
  verify the residual-basis sidecar against its dataset, gene panel,
  conditioner, mask schedule, and numeric basis content.
- Run manifests and checkpoints bind code state, dataset/gene/mask
  fingerprints, per-sample cache contents, and every configured model
  artifact SHA256.
- Validation model selection uses deterministic reconstruction; flow
  diagnostic sampling uses stable per-item common random numbers.

## Required hardware validation

Before a long run, execute one staged smoke with the real UNI2,
scFoundation, and STPath installations/checkpoints on the target GPU.
This repository's automated tests use faithful stubs because those gated
weights/packages are not available in the local development environment.
Do not start a long run if any real-weight construction, cache provenance,
capacity, finiteness, or memory gate fails.
