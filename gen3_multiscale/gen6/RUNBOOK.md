# Gen6 matched component screen

Gen6 reuses the Gen3 masking, manifest, data adapter, trainer, transactional
checkpoints and evaluator.  It does not introduce a second experiment
harness.  Every deterministic arm predicts the full manifest gene panel and
uses the same full-panel RMSE + PCC objective plus the existing weak spatial
gradient term.

## Arms

| Key | Deliberate component change |
|---|---|
| gen6a | released STPath, loaded pretrained then fully unfrozen; STPath's own head |
| gen6b | weighted-linear GEX + UNI2 + simple fusion |
| gen6c | scFoundation + UNI2 + simple fusion |
| gen6d | weighted-linear GEX + GigaPath/LongNet + simple fusion |
| gen6e | scFoundation + GigaPath/LongNet + simple fusion |
| gen6f | scFoundation + UNI2 + MoME + frame averaging |
| gen6g | same encoders/MoME + normalized learned relative bias |
| gen6h | same encoders/MoME + normalized Fourier relative attention |
| gen6i | same encoders + bidirectional image/GEX cross-attention |
| gen6j | MoME + full local/boundary/regional/global spatial field |
| gen6k | selected frozen Gen6 conditioner + minibatch-OT residual flow |
| gen6l | selected frozen Gen6 conditioner + conditional WAE-GAN |

G6-B:E form the 2x2 encoder comparison.  G6-F:J hold scFoundation+UNI2
fixed so fusion/geometry changes are attributable.  G6-K:L are staged only
after a deterministic conditioner is validation-selected.

## Package smoke

```bash
python -m gen3_multiscale.scripts.gen6_smoke_launcher
```

This runs one CPU optimizer step for G6-B:J.  G6-A additionally needs the
real STPath package/checkpoint.  G6-K:L need staged artifacts.

## Prepare deterministic configs (no training)

Use one resolved prior config from the same manifest as the comparison base.
Pass the exact identities already used to build the caches:

```bash
python -m gen3_multiscale.scripts.prepare_gen6_suite \
  --comparison-config /path/resolved_comparison.yaml \
  --manifest /path/dataset_manifest.json \
  --output-root gen3_multiscale/results/gen6_screen_$(date -u +%Y%m%dT%H%M%SZ) \
  --hours 8 \
  --fingerprint uni2_checkpoint=/path/pytorch_model.bin \
  --fingerprint uni2_revision=PINNED_40_HEX \
  --fingerprint uni2_package_version=VERSION \
  --fingerprint uni2_preprocessing_spec=EXACT_SPEC \
  --fingerprint scfoundation_checkpoint=/path/models.ckpt \
  --fingerprint scfoundation_vocab=/path/vocab.tsv \
  --fingerprint scfoundation_package_version=VERSION \
  --fingerprint scfoundation_preprocessing_spec=EXACT_SPEC \
  --fingerprint gigapath_checkpoint=/path/slide_encoder.pth \
  --fingerprint stpath_checkpoint=/path/stfm.pth \
  --fingerprint stpath_gene_vocab=/path/symbol2ensembl.json
```

Preflight fails before model construction if any required cache, fingerprint,
shape or provenance is missing.

## Launch four concurrent GPU queues

First inspect the assignment:

```bash
ROOT=$(cat gen3_multiscale/results/LATEST_GEN6_SUITE_ROOT.txt)
python -m gen3_multiscale.scripts.run_gen6_queues --suite-root "$ROOT" --gpus 0,2,3,5 --dry-run
```

Launch detached from the terminal without tmux:

```bash
nohup python -u -m gen3_multiscale.scripts.run_gen6_queues \
  --suite-root "$ROOT" --gpus 0,2,3,5 \
  > "$ROOT/logs/queue_master.log" 2>&1 &
echo $! > "$ROOT/queue_master.pid"
```

`queue_status.json` and per-arm logs show live state.  A failed arm blocks
only later arms on that GPU queue; it never silently proceeds on the same GPU.

## Prepare G6-K/L

Fit a basis from the exact validation-selected deterministic checkpoint:

```bash
python -m gen3_multiscale.scripts.fit_gen4_residual_basis \
  --config /path/selected_gen6_config.yaml \
  --conditioner-checkpoint-dir /path/checkpoints/gen6X/best \
  --output-basis-path /path/gen6_residual_basis_rank64.pt \
  --n-masks-per-sample 20 --rank 64 --device cpu --svd-device cuda
```

Then prepare both staged generators (no training):

```bash
python -m gen3_multiscale.scripts.prepare_gen6_generators \
  --conditioner-checkpoint /path/checkpoints/gen6X/best \
  --basis /path/gen6_residual_basis_rank64.pt \
  --output-root /path/gen6_generators --hours 8
```

Launch those two configs with `run_gen6_queues --arms gen6k,gen6l`.

## Evaluation

Use the unchanged evaluator for every completed arm:

```bash
python -m gen3_multiscale.evaluation.gen3_evaluator \
  --config /path/gen6X.yaml --checkpoint-dir /path/checkpoints/gen6X \
  --output /path/gen6X_validation.json --split validation --use-best --device cuda
```

Compare full-panel, train-derived top-50/top-200, RMSE, PCC, AUC, patient
aggregation and the existing baselines.  G6-K/L additionally report sampled
uncertainty.  No long run is started by any preparation or smoke command.
