# Gen4 runbook

See `GEN4_CONTRACT.md` for the full design. This is the operational
dependency order and an honest list of what still needs real weights/GPU
validation before any real run.

## 0. Scope reminder

Nothing in this suite has downloaded a checkpoint, loaded real UNI2/
scFoundation/STPath weights, or started a training run. Every test in
`tests/test_gen4_*.py` and the smoke launcher below run on tiny synthetic
CPU data with deterministic stub encoders
(`tests/_gen4_fixtures.py::StubUNI2Encoder`/`StubSCFoundationEncoder`/
`Gen4STPathStub`).

## 1. Dependency order, per arm

```
1. Precompute caches (skip for arm D's image slot -- see step 3):
   - Arm A: gen4/uni2_spot_cache.py (UNI2 features)
   - Arm B: existing GigaPath spot cache (unchanged) + gen4/scfoundation_cache.py
   - Arm C: gen4/uni2_spot_cache.py + gen4/scfoundation_cache.py
   - Arm D: existing GigaPath spot cache only (STPath's own image tokenizer input)

2. gen4/preflight.py --config configs/gen4/<arm>_conditioner.yaml
   (static schema/provenance checks; also run audit_uni2_cache_matches_config /
   audit_scfoundation_cache_matches_config against the caches built in step 1)

3. Build Gen4SpatialFieldInputs per training/validation/test mask:
   - Arms A-C: gen4/inputs.py::build_gen4_spatial_field_example
     (precomputed_spot_features from step 1's image cache;
      gex_context_embedding from step 1's scFoundation cache, arms B/C only)
   - Arm D: gen4/stpath_example.py::build_gen4_stpath_example
     (runs Gen4STPathContextEncoder.encode_context_only PER MASK -- see
      GEN4_CONTRACT.md section 8 for why this cannot be precomputed once)

4. Train the deterministic conditioner:
   gen4/model_factory.py::build_gen4_conditioner(config, ...) + an ordinary
   training loop reusing training/checkpoint.py's save_checkpoint/
   load_trainable_state directly (both are already architecture-agnostic).

5. Select the conditioner's best checkpoint using VALIDATION ONLY.

6. Freeze that checkpoint; fit the rank-64 residual basis:
   gen4/basis_fit.py::fit_gen4_residual_basis(frozen_conditioner,
   TRAINING-only examples, gene_names, rank=64, output_basis_path=...)

7. Train the flow model:
   gen4/model_factory.py::build_gen4_flow(config, ..., gene_basis=<step 6>,
   gene_names=...) -- load the frozen conditioner checkpoint's weights
   into model.conditioner before training, matching Gen3 Architecture 4's
   own "freeze_conditioner_initially" discipline.

8. Evaluate (validation, then test exactly once) -- see section 4 below
   for the current evaluator-reuse boundary.
```

## 2. Smoke check (run this first, always)

```bash
python -m gen3_multiscale.scripts.gen4_smoke_launcher
```

Constructs all four arms' conditioner + flow model at tiny dims with stub
encoders, runs one real optimizer step each, and one
`sample_predictive_distribution` call. No GPU, no real weights. Exits
non-zero on any exception; prints one report dict per arm on success.

## 3. Parameter counts / frozen vs. trainable

Run `gen4/param_report.py::report_parameters(model)` on any constructed
model for a live table (per GEN4_CONTRACT.md section 3's summary). Example
from the smoke launcher's tiny dims (real deployment dims will differ, but
the frozen-vs-trainable SHAPE per arm is the same):

| Arm | Conditioner params | Flow params | `gene_encoder` (WeightedGeneExpressionEncoder) | `slide_encoder` |
|---|---|---|---|---|
| A (`gen4a`) | 146,888 | 167,763 | trainable | trainable (`MaskAwareCoordinateAttentionPool`) |
| B (`gen4b`) | 138,474 | 159,349 | constructed, frozen (never called) | frozen (`FrozenGigaPathSlideEncoder`) |
| C (`gen4c`) | 146,922 | 167,797 | constructed, frozen (never called) | trainable (`MaskAwareCoordinateAttentionPool`) |
| D (`gen4d`) | 126,782 | 147,657 | trainable | n/a (no global branch) |

(`gex_context_proj`, present only for arms B/C, is always trainable.)

## 4. Honest gaps -- what still needs real weights/GPU validation

- **UNI2**: `gen4/uni2_encoder.py::FrozenUNI2TileEncoder` has never run
  against a real UNI2 checkpoint in this environment. Its `timm.create_model`
  call, `load_state_dict` strictness, and real `output_dim` (assumed 1536
  as a placeholder in `configs/gen4/gen4a_conditioner.yaml`/`gen4c_conditioner.yaml`
  -- verify against the real checkpoint) are all unverified against the
  actual pretrained weights.
- **scFoundation**: `gen4/scfoundation_encoder.py::FrozenSCFoundationEncoder`
  depends on an `import scfoundation` package and a
  `scfoundation.build_model_from_state_dict` call that has never been
  exercised against a real installation -- the real package's public API
  was not available to verify in this environment. `output_dim` (assumed
  3072 as a placeholder in `configs/gen4/gen4b_conditioner.yaml`/
  `gen4c_conditioner.yaml`) must be confirmed against the real checkpoint.
- **STPath**: `gen4/stpath_context.py::Gen4STPathContextEncoder.encode_context_only`
  has never run against the real `stpath` package/weights. It is built by
  close reading of the already-verified `src/models/stpath_encoder.py`
  (same repo, same tokenizer/model call shape) but the context-only token
  assembly (all rows real, no query concatenation) has not been run
  end-to-end against `stpath.model.model.STFM.prediction_head`.
- **Full manifest-driven mask-schedule integration for basis fitting**:
  `gen4/basis_fit.py` takes an already-materialized list of
  `(inputs, targets)` pairs, not a manifest + mask-schedule spec. Wiring a
  real training-mask iterator (mirroring
  `training.gen3_dataset.Gen3SpatialFieldDataset`/`build_gen3_mask_schedule`,
  extended to attach each arm's `context_gex_embedding`/STPath per-mask
  step) is real-data-dependent integration work not attempted this round.
- **No full trainer/evaluator CLI**: this suite reuses
  `training/checkpoint.py` directly (architecture-agnostic, verified) but
  does not wire Gen4 arms into `training/train.py`'s or
  `evaluation/gen3_evaluator.py`'s own CLIs (both are coupled to
  `models/model_factory.py`'s "1"-"4" architecture dispatch, which this
  suite deliberately never modifies -- GEN4_CONTRACT.md section 13).
- **Dense-WSI UNI2 tile cache**: arms A/C's regional/global branches
  assume a UNI2-encoded dense-WSI tile cache analogous to
  `data/slide_context.py`'s existing GigaPath one; no such cache builder
  exists yet (the per-spot cache in `gen4/uni2_spot_cache.py` is built,
  the dense-tile analog is not).
- **Real HEST-1k end-to-end run for any arm**: every test and the smoke
  launcher use tiny synthetic (non-HEST) or synthetic-HEST-shaped data;
  no arm has been run against real HEST-1k samples.

## 5. Static preflight/audit

```bash
python -m gen3_multiscale.gen4.preflight --config configs/gen4/gen4a_conditioner.yaml
```

Prints a JSON report (arm, kind, which `required_fingerprints` are still
unset, `ready_for_real_training`). Every committed config in
`configs/gen4/` currently reports `ready_for_real_training: false` --
correct, since no real checkpoint paths have been filled in.
