# Gen5 runbook

See `GEN5_CONTRACT.md` for the full design; `gen3_multiscale/gen4/RUNBOOK.md`
for the shared conditioning/cache dependency order (unchanged, reused).

## 1. Dependency order

```
1. Complete Gen4's own dependency order (gen4/RUNBOOK.md section 1) for
   each arm through step 5: a real, validation-selected, frozen Gen4
   conditioner checkpoint per arm.

2. Fit the shared expression autoencoder (ONCE, training samples only):
   gen5/autoencoder_training.py::train_expression_autoencoder(
       train_expression, gene_names, latent_dim=256, ...)
   gen5/autoencoder.py::save_expression_autoencoder_checkpoint(...)

3. Run the autoencoder capacity gate (gen5/autoencoder.py::
   evaluate_autoencoder_reconstruction) on held-out training spots AND
   real validation spots. Report the reconstruction ceiling before
   proceeding -- no flow arm's decoded output can exceed it.

4. Per arm, train the latent flow:
   gen5/model_factory.py::build_gen5_model(config, ..., autoencoder=<step 2's
   frozen checkpoint>) -- load the arm's frozen Gen4 conditioner checkpoint
   into model.conditioner, call model.freeze_conditioner(), then train
   model.velocity_network only (Adam over
   [p for p in model.parameters() if p.requires_grad]).

5. Evaluate (validation, then test exactly once):
   gen5/evaluator.py::compare_gen5_arm(...) against the matched Gen4
   residual-flow arm, the deterministic conditioner, and the
   mean/nearest-neighbor/harmonic baselines.
```

## 2. Smoke check (run this first, always)

```bash
python -m gen3_multiscale.scripts.gen5_smoke_launcher
```

Trains a tiny autoencoder on synthetic data, then constructs and runs one
real optimizer step + one real sampling call for all four arms. No GPU,
no real weights, no full training loop. Reports peak CPU RSS as a
stand-in for the CUDA memory smoke test (§4).

## 3. Static preflight

```bash
python -m gen3_multiscale.gen5.preflight --config configs/gen5/gen5a.yaml
```

Every committed config currently reports `ready_for_real_training: false`
-- correct, no real checkpoint paths have been filled in yet.

## 4. Honest gaps -- what still needs real weights/GPU validation

Everything already listed in `gen4/RUNBOOK.md` section 4 (UNI2/scFoundation/
STPath real weights, dense-WSI UNI2 cache, full manifest-driven mask
iterator, real HEST-1k run) applies identically to Gen5's conditioning
side, plus:

- **No real Gen4 conditioner checkpoint exists.** Gen5's own tests and
  smoke launcher only ever construct fresh (random-init) `Gen4Conditioner`
  instances -- the "matched conditioning systems" guarantee (loading the
  SAME trained weights Gen4 used) has never been exercised end-to-end.
- **CUDA memory smoke test (required gate 9) has not run on real
  hardware.** This environment has no GPU. The smoke launcher's peak-RSS
  report is a CPU proxy only, not a substitute for
  `torch.cuda.max_memory_allocated()` under a real batch/hole size.
- **`gen5/evaluator.py::compare_gen5_arm` has never run against real
  model outputs** -- only against hand-constructed synthetic arrays in
  its own (implicit, via the metrics module's existing tests) numerical
  correctness; no test in this round calls it with genuine Gen4/Gen5
  model predictions end to end, since no trained checkpoint exists to
  produce them.
- **Autoencoder latent_dim=256 is untested at that width** -- every test
  and the smoke launcher use tiny latent dims (8-12) for CPU speed; the
  real 256-dim configuration has been constructed (`configs/gen5/*.yaml`
  parse and pass the static audit) but never trained or evaluated.
