# MK conditional latent-flow project

These four arms are controlled counterparts to the four conditional-WAE arms.
They reuse the same full-visible H&E input, optional leakage-safe surrounding
ST context, Architecture-1 conditioner, learned target-expression encoder,
conditional-mean head, full-expression decoder, masks, splits and metrics.

- `flow_he`: H&E → ST with ordinary conditional rectified flow.
- `flow_he_ot`: H&E → ST with minibatch Sinkhorn OT flow matching.
- `flow_he_st`: H&E + surrounding ST → masked ST with ordinary flow.
- `flow_he_st_ot`: H&E + surrounding ST → masked ST with OT flow.

Training encodes true query expression to an informative latent using the same
reconstruction objective as WAE. The flow target is stop-gradient, preventing
the transport objective from collapsing the learned encoder. A spatial
velocity transformer learns the path from Gaussian noise to that latent while
conditioned on the legal H&E ± surrounding-ST representation. OT arms only
change how prior noise rows are paired to fixed spatial target rows.

At inference, target GEX is unavailable. Gaussian latent noise is integrated
from t=0 to t=1 through the conditional velocity field, then the same
conditional expression decoder produces full-panel GEX. The evaluator reports
the sampled flow and deterministic conditional-mean baseline separately.

## Commands

Synthetic CPU gate:

```bash
python -m gen3_multiscale.scripts.conditional_flow_smoke_launcher
```

Prepare immutable configs:

```bash
python -m gen3_multiscale.scripts.prepare_conditional_flow_suite \
  --comparison-config /path/to/resolved_architecture1.yaml \
  --manifest /path/to/dataset_manifest.json \
  --train-gene-panels /path/to/train_gene_panels.json \
  --output-root /path/to/new_conditional_flow_suite \
  --hours 8
```

Real-data one-step smoke on four distinct GPUs:

```bash
python -m gen3_multiscale.scripts.run_conditional_flow_suite \
  --suite-root /path/to/new_conditional_flow_suite \
  --gpus 0,1,2,3 --smoke
```

Remove `--smoke` for the configured full runs. Generated configs log scalars,
Projector cohorts, H&E thumbnails and spatial diagnostics beneath
`SUITE_ROOT/tensorboard/ARM` using the same bounded MK TensorBoard contract.

Evaluate a completed arm:

```bash
python -m gen3_multiscale.evaluation.conditional_flow_evaluator \
  --config /path/to/configs/flow_he.yaml \
  --checkpoint-dir /path/to/checkpoints/flow_he \
  --output /path/to/evaluation/flow_he_validation.json \
  --split validation --n-masks-per-sample 8 --device cuda --allow-code-drift
```
