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
| gen6k | frozen Gen6-C + shared expression autoencoder + minibatch-OT latent flow |
| gen6l | the same frozen Gen6-C + conditional WAE-GAN |
| gen6m | Gen6-C + 2 rounds of spatial refinement over predicted expression (k=6) |
| gen6n | Gen6-C + 4 rounds of spatial refinement over a wider neighbourhood (k=12) |
| gen6o | Gen6-L + a supervised conditional-mean head, latent correlatable at sampling |
| gen6p | Gen6-K with hard OT assignment instead of barycentric averaging |

G6-B:E form the 2x2 encoder comparison.  G6-F:J hold scFoundation+UNI2
fixed so fusion/geometry changes are attributable. G6-K:L deliberately use
the exact same frozen Gen6-C checkpoint, making Gen6-C the deterministic
control for both generative additions.

G6-M:P are a second screen, each changing exactly one thing relative to an
arm that has already been trained (M/N vs C, O vs L, P vs K), so no control
needs re-running:

- **M, N.** Nothing in G6-B:J propagates PREDICTED expression between query
  spots -- each is decoded independently from its own conditioning. Yet on
  this manifest's validation slides, simply averaging a spot's OBSERVED
  neighbours scores 0.392 on the top-200 panel against a measured count-split
  noise ceiling of 0.767. Two settings rather than one so "refinement helps"
  can be separated from "that one depth happened to help". The refiner's
  update head is zero-initialised and it is built under a forked RNG, so at
  step 0 a refinement arm is bit-identical to gen6c.
- **O.** Adds a conditioning-only prediction supervised against the same
  target, which is what makes `prediction = conditional_mean + residual`
  measurable; without that decomposition an uncertainty number can be
  reported but not diagnosed. Sampling can then couple the query spots of one
  draw (rho), so a sample expresses a region-level alternative rather than
  spatially white noise. rho is inference-only, so the same checkpoint scores
  at rho=0 too.
- **P.** Barycentric OT pairing averages plan rows and contracts the flow's
  source (measured std ratio 0.78-0.86 at target latent scale 0.1) while
  sampling always integrates from a full N(0, I). Hard assignment returns
  actual noise draws, so train and inference share one source. G6-K stays
  pinned to barycentric so it remains a valid control.

## Package smoke

```bash
python -m gen3_multiscale.scripts.gen6_smoke_launcher
```

This runs one CPU optimizer step for G6-B:J and G6-M:N.  G6-A additionally
needs the real STPath package/checkpoint.  G6-K:L and G6-O:P need staged
artifacts.  The smoke also prints G6-C, G6-M and G6-N losses, which must be
identical: that is the refiner's zero-initialised identity plus its forked
RNG, and a divergence there means a refinement arm is no longer comparable to
its control.

## Prepare deterministic configs (no training)

Use one resolved prior config from the same manifest as the comparison base.
Pass the exact identities already used to build the caches:

```bash
python -m gen3_multiscale.scripts.prepare_gen6_suite \
  --comparison-config /path/resolved_comparison.yaml \
  --manifest /path/dataset_manifest.json \
  --train-gene-panels /path/train_gene_panels.json \
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

Preparation verifies that the train-derived panel artifact matches the manifest
and contains both HVG-50 and HVG-200. Preflight fails before model construction
if any required cache, fingerprint, shape or provenance is missing.

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

Train and validate one shared full-expression autoencoder using the existing
Gen5 command (this is a separate stage and its checkpoint remains frozen in
Gen6-K):

```bash
python -m gen3_multiscale.scripts.train_gen5_autoencoder \
  --manifest /path/dataset_manifest.json \
  --output-checkpoint /path/shared_autoencoder.pt \
  --latent-dim 256 --hidden-dim 1024 --epochs 50 \
  --batch-size 64 --lr 1e-3 --device cuda --seed 0
```

Then prepare both staged generators from the exact Gen6-C checkpoint (no
training is started by this command):

```bash
python -m gen3_multiscale.scripts.prepare_gen6_generators \
  --conditioner-checkpoint /path/checkpoints/gen6c/best \
  --autoencoder-checkpoint /path/shared_autoencoder.pt \
  --output-root /path/gen6_generators --hours 8
```

Launch those configs with `run_gen6_queues --arms gen6k,gen6l`.

By default this writes all four staged arms.  When G6-K and G6-L have already
been trained, narrow it so only the new ones are prepared:

```bash
python -m gen3_multiscale.scripts.prepare_gen6_generators \
  --conditioner-checkpoint /path/checkpoints/gen6c/best \
  --autoencoder-checkpoint /path/shared_autoencoder.pt \
  --output-root /path/gen6_generators_v2 --hours 8 \
  --arms gen6o gen6p
```

The autoencoder is still required for the argument check even when only
G6-O is prepared; only G6-K/G6-P bind it as a fingerprint.

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
