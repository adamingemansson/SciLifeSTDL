# Architecture Plan: Model-Agnostic Generative Pipeline

Last updated: 2026-07-13. Captures the design discussed for building one
architecture that can swap between VAE / WAE-GAN / diffusion backbones
(and later, different conditioning strategies and metrics) without
rewriting data, training, or evaluation code.

## The core problem

Swapping VAE ↔ GAN ↔ diffusion isn't just swapping a network — each family
has a genuinely different **training loop** and **sampling procedure**:
- VAE: one forward pass, ELBO loss, one optimizer.
- GAN: alternating generator/discriminator updates, two optimizers.
- Diffusion: training step is noise-prediction on a random timestep;
  sampling is a separate iterative loop (dozens–thousands of steps), not a
  forward pass.

A single shared `model.loss()` call (the original design) works for VAE but
breaks for the other two. The architecture has to abstract over training
*procedure*, not just network shape.

## Four-layer decomposition

1. **Conditioning encoder (shared, family-agnostic)** — turns `context`
   (neighboring cells/spatial graph) into a fixed representation, e.g. a
   GNN/transformer over a k-NN spatial graph → a set of tokens. Built once,
   consumed identically by every model family. This is the actual reusable
   "general architecture" contribution — not yet implemented (see Status
   below), current models are unconditioned placeholders.
2. **Backbone registry** — small swappable network pieces (denoiser,
   encoder+decoder, generator+discriminator) that plug into layer 3.
3. **Model family wrapper (`BaseGenerativeModel`)** — every family
   implements the same interface, differently internally:
   ```
   BaseGenerativeModel(pytorch_lightning.LightningModule):
     sample(context, query) -> output      # the ONE entry point for generation,
                                            # regardless of family internals
     training_step(batch, batch_idx)       # Lightning entry point; VAE: ELBO.
                                            # WAE-GAN: alternates encoder/decoder
                                            # vs. discriminator internally.
     configure_optimizers()                # 1 optimizer (VAE), or a list
                                            # (WAE-GAN: [opt_ae, opt_disc])
   ```
   Built on `pytorch_lightning.LightningModule` specifically for its
   multi-optimizer "manual optimization" mode (`automatic_optimization =
   False`), which is what makes GAN-style alternating updates possible
   without hand-rolling optimizer bookkeeping.
4. **Metrics — fully decoupled.** Every family's `sample()` returns the same
   output shape, so `src/evaluation/metrics.py` never needs to know which
   family produced it. No changes needed here for new model families.

## Build vs. reuse

Most infrastructure is either already built or nearly free via existing
dependencies — the real work is the conditioning encoder and each family's
generator/loss logic.

| Component | Status |
|---|---|
| Data loading, masking simulators | Already built (`src/data/`) |
| Pointwise metrics (PCC, RMSE, AUC) | Already built (`src/evaluation/metrics.py`) |
| FID/MMD skeleton | Already built, needs a real embedding once chosen |
| ARI/NMI | One-liner via `sklearn.metrics` |
| Moran's I / Geary's C | Free via `squidpy.gr.spatial_autocorr` |
| Training loop plumbing (checkpointing, device mgmt, multi-optimizer) | Free via `pytorch_lightning` |
| Diffusion noise scheduling + sampling loop math | Free via `diffusers` schedulers, once we add a diffusion family |
| VAE encoder/decoder | Built (`registry.py` `VAEBaseline`) |
| WAE-GAN encoder/decoder/discriminator | Built (`registry.py` `WAEGAN`) |
| **Conditioning encoder** | **Not yet built — real work, next priority** |
| Dataset-specific adapter for the chosen pilot dataset | Not yet built — blocked on Phase 1 (`docs/project_outline.md`) |

## WAE-GAN (first GAN-family entry)

Chosen over a vanilla conditional GAN for lower implementation/training
risk. From Tolstikhin et al. 2017 (`docs/literature_review.md`,
`docs/metrics_notes.md` §4) — keeps a plain deterministic encoder/decoder
with a normal reconstruction loss, and replaces the VAE's KL term with a
discriminator that only regularizes the *encoder's aggregated latent
distribution* toward the prior. The adversarial part is a smaller,
better-behaved sub-problem than a vanilla GAN that generates expression
directly through the discriminator signal — appropriate given how sparse
and zero-inflated gene expression data is, which vanilla GANs tend to
handle poorly.

Training step (`WAEGAN.training_step`, manual optimization):
1. Encode real expression → `z_fake`. Sample `z_real ~ N(0, I)` (the prior).
2. **Discriminator step**: binary classify `z_real` vs. `z_fake` (detached).
3. **Encoder/decoder step**: reconstruction loss (`decoder(z_fake)` vs.
   input) + adversarial loss (encourage `z_fake` to fool the discriminator).

Generation (`sample()`) never touches the encoder or discriminator — decode
`z ~ prior`, identical to the VAE path. This means WAE-GAN and VAE are
interchangeable at inference time from the pipeline's point of view, which
is exactly the point of the shared interface.

## Prioritization (unchanged from the original plan)

1. **VAE** — done, proves the plumbing end-to-end.
2. **WAE-GAN** — done (this update). Chosen GAN entry point.
3. **Diffusion / Flow Matching** — next priority. Every close prior-art
   paper reviewed (Mimyr, isoST, stDiff, LGDiST) is diffusion-based, so
   this is where real comparability against the literature lives. Flow
   Matching (Lipman et al. 2022, `docs/literature_review.md`) is the
   preferred entry over classic DDPM — simpler, more stable
   simulation-free training, subsumes diffusion as a special case. `diffusers`
   schedulers handle the noise/sampling math; we only write the denoiser
   network.
4. **Normalizing flows** — deprioritized; invertibility constraints on the
   network are restrictive and there's no prior-art pull toward it here.

## Known gaps / next steps
- All current models (`interp_baseline`, `vae_baseline`, `wae_gan`) are
  **unconditioned placeholders** — they ignore `context`/`query['coords']`
  beyond trivial use. Building the conditioning encoder (layer 1) is the
  next real architecture task, and should happen before or alongside
  picking a pilot dataset, since the encoder design depends on data
  representation (point cloud/graph vs. fixed patches — see
  `docs/project_outline.md` Phase 4).
- `src/training/train.py` now drives training through
  `pytorch_lightning.Trainer`, wrapping the current single-batch debug
  setup in a placeholder `_SingleBatchDataset`. Replace with a real
  per-cell/mini-batch `Dataset` once a pilot dataset is chosen — the
  model interface does not need to change when that happens.
