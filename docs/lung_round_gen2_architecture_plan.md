# Gen-2 Architecture Plan (2026-07-24): 4 full-HEST-1k candidates

Planning document only — nothing here is implemented yet. Purpose: hand this
to GPT for review before building anything, per the plan agreed in-session.
Goal: 4 genuinely different architectures, each trainable in ~1-2 days on one
A100 (4 GPUs in parallel), on the full HEST-1k corpus (not the ~8-sample Lung
pilot this repo has been using for architecture search so far).

Every "REUSED" component below names a real file/class that already exists
and has been exercised on real hardware this project. Every "NEW" component
is flagged explicitly — nothing is silently assumed to exist. Sizes follow
GPT's own PDF sizing advice (`Gene_Expression_Prediction_Model.pdf`):
~35-70M trainable parameters is enough for this task; frozen pretrained
backbones sit outside that budget and are not counted against it.

## 0. What we already know, going in

From real runs on this project (lung_round, 11 configs, GigaPath-preprocessing
bug fixed and confirmed live):

- **STPath's real pretrained backbone (frozen weights) is our best performer
  so far**, both from-scratch-with-STPath's-architecture (`stpath_scratch`)
  and truly-frozen (`stpath_pretrained_eval`) — consistently ahead of every
  from-scratch alternative, but still far below the reference notebook's own
  STPath numbers (~0.05 vs. ~0.15 hest50 PCC).
- **A verified, not-yet-fixed cause of part of that gap**: our pipeline
  library-size-normalizes expression before log1p everywhere; STPath's real
  pretrained weights (and the reference notebook) were only ever exposed to
  raw-count log1p. Every architecture below that touches STPath's frozen
  weights or scFoundation must feed them their own native preprocessing, not
  ours — this is a known, real bug class now, not a guess.
- **A trained-from-scratch model with NO image input at all (harmonic
  baseline) ties or beats most image-conditioned architectures** on the Lung
  pilot. That's a red flag about how much signal our from-scratch image
  conditioning paths currently extract — motivates leaning harder on frozen
  pretrained backbones (image AND gene) rather than learning representations
  from scratch on a comparatively small dataset.
- **`context_selection: nearest_query` + `max_context_points`
  (`src/data/mask_bank.py::cap_context_mask`) already implements GPT's "small
  neighborhood transformer" idea** — real k-NN-to-query capping via a KD-tree,
  not a crude grid window. Nothing new needed to get a fixed local
  neighborhood.
- **A Pearson-correlation loss term does not exist yet** anywhere in
  training (`src/evaluation/metrics.py::pearson_per_gene` is eval-only). GPT's
  suggested `loss = 0.7·MSE(log1p) + 0.3·(1 - pearson)` is new work.
- **A masking curriculum (single-spot → contiguous → irregular holes, with
  growing mask fraction) does not exist.** `src/data/masking.py` draws a
  fixed distribution every step; there is no step-dependent schedule. New
  work if any architecture wants GPT's exact 3-stage curriculum.

## 1. Shared infrastructure (all 4 architectures use this, unchanged)

- **Data**: HEST-1k, loaded via `src/data/loaders.py::load_multi_sample` /
  `load_hest_sample`. Full corpus, not organ-restricted this time — mixing
  organs requires the shared-gene-panel intersection
  (`load_multi_sample`'s `shared_genes`) to stay non-trivial; if the shared
  panel collapses too far when mixing very different organs/platforms, fall
  back to a same-platform (all-Visium) subset first pass rather than the
  full heterogeneous corpus. Decide the exact sample list once we know how
  many HEST-1k slides are actually staged on the server.
- **Expression preprocessing**: `normalize_total(target_sum=1e4)` + `log1p`
  (`src/data/loaders.py::basic_qc_and_normalize`, our project default) for
  the target and for any from-scratch gene encoder input. Frozen pretrained
  components (STPath, scFoundation) get their OWN native preprocessing fed
  from the newly-added `adata.layers["raw_counts"]` /
  `adata.obs["_scilifestdl_raw_library_size"]` (added this session,
  `src/data/loaders.py`) — never our normalized value reused for them.
- **Masking**: `src/data/masking.py::random_dropout_patches`, single hole,
  mixed non-round shapes, `radius_range` swept per architecture (see below).
  `center_mode="random"` (default) unless an architecture specifically wants
  `"geometric_median"` (already disproved as a fix for the STPath gap on
  Lung, kept available as a knob, not a default).
- **Context cap**: `max_context_points` + `context_selection: nearest_query`
  (`src/data/mask_bank.py::cap_context_mask`) — every architecture caps
  context to a local neighborhood around the query region rather than the
  whole slide, both for GPU memory and because GPT's own analysis says
  spatial transcriptomic signal is dominated by local structure.
- **Held-out protocol**: sample-level train/validation/test split
  (`data.train_sample_ids` / `validation_sample_ids` / `test_sample_ids`),
  fixed mask banks (`src/data/mask_bank.py`), full audit evaluation
  (`src/evaluation/audit_evaluation.py::evaluate_model_on_mask_bank`) —
  PCC/RMSE on the full gene panel, `lung_hest_bench_50`-style fixed panels
  where available for the organs actually trained on, plus the new
  `pcc_raw_log1p` notebook-comparable metric (added this session) for any
  architecture that touches STPath or scFoundation.
- **task_contract**: `missing_tissue`, `image_mode: target_zero` as the
  primary/training mode (query H&E withheld — this is the actual deployment
  task: reconstruct a damaged/missing region using only surrounding tissue +
  surrounding GEX). `all_zero`/`full` stay as evaluation-only diagnostic
  modes, never training targets.

## 2. Architecture 1 — "GPT-v1 faithful": small-neighborhood transformer, MLP gene encoder

**Hypothesis being tested**: does GPT's own concrete, most conservative
recommendation (build this first, ~40M params, no foundation-model gene
encoder) already close a meaningful chunk of the gap, with nothing more
exotic than what we already have wired up?

**Flow**:
```
H&E patch (224x224x3) -> frozen GigaPath tile encoder -> image_feat [1536]
  (REUSED: src/models/conditioning.py::GigapathPatchEncoder / precompute_gigapath_features)
gene expression (normalized log1p, shared panel) -> MLP gene encoder -> gene_feat [256]
  (REUSED: gene_encoder_type="mlp" path, src/models/simple_fusion_encoder.py::_build_gene_encoder)
relative (dx, dy) per context spot -> learned Fourier positional encoding -> coord_feat [64]
  (NEW: small MLP over Fourier features; nothing in this repo currently does
   this exact "Fourier-then-MLP" relative-position embedding — closest
   existing piece is the coordinate handling inside SimpleCrossAttention/
   SpatialTransformer encoders, which is not quite the same construction)
[image_feat(1536); gene_feat(256); coord_feat(64)] -> concat -> Linear(1856, 512) -> spot_token [512]
context spots capped to nearest ~80 (context_selection="nearest_query", max_context_points=80)
  + 1 learnable missing-spot query token [512]
  -> Transformer encoder, 8 layers, 512 hidden, 8 heads, MLP ratio 4x, dropout 0.1  (~35M params)
  (REUSED core mechanism: SimpleFusionSpatialTransformerContextEncoder /
   STPathBackboneSimpleGene's own transformer stack, src/models/
   simple_fusion_encoder.py + registry.py — reconfigured to this exact
   width/depth/neighborhood size rather than a new implementation)
-> query token's final hidden state [512]
-> Gene decoder MLP: Linear(512,1024) -> GELU -> LayerNorm -> Linear(1024, n_genes)
  (REUSED: dense decoder path already used by context_transport_regressor /
   simple_cross_attn_dense_decoder)
-> predicted expression [n_genes]
```
**Loss** (NEW — Pearson term does not exist in training yet):
`loss = 0.7 * MSE(log1p_pred, log1p_target) + 0.3 * (1 - mean_gene_pearson(pred, target))`,
computed per training batch across the batch's spots (gene-wise Pearson needs
>1 spot per batch to be meaningful — batch size must stay well above 1;
see training config below).

**Trainable params**: ~2.7M (gene encoder) + ~0.1M (coord encoder) + ~0.6M
(fusion projection) + ~35M (transformer) + ~13M (decoder, scales with
n_genes) ≈ **45-55M**, in GPT's recommended range. Frozen GigaPath (~1.1B)
outside that budget.

**What's genuinely new here**: the Fourier positional-embedding module, and
the Pearson loss term. Everything else is a reconfiguration of existing
classes, not new code.

## 3. Architecture 2 — scFoundation as the gene encoder (GPT's #1-ranked pretrained option)

**Hypothesis being tested**: GPT's top recommendation across the whole
consultation — swap the from-scratch MLP gene encoder for a real pretrained
single-cell foundation model — using the scFoundation integration already
built this session (`ScFoundationGeneEncoder`,
`precompute_scfoundation_features`, both in `src/models/conditioning.py`,
verified against scFoundation's real cloned source, not yet numerically
verified against a live checkpoint on real hardware — this run is that
verification).

**Flow**:
```
H&E patch -> frozen GigaPath -> image_feat [1536]                    (REUSED, same as Arch 1)
context spot's own expression (raw-count log1p, NOT our library-size-
  normalized value -- scFoundation's real preprocessing contract)
  -> frozen scFoundation encoder (100M params, 19,264-gene vocabulary)
  -> 4-way pool-concat -> gene_embed [scfoundation_dim, auto-detected at
     runtime, typically a few thousand-d -- see conditioning.py docstring]
  -> LayerNorm + Linear -> gene_feat [256]
  (REUSED: ScFoundationGeneEncoder + precompute_scfoundation_features,
   src/models/conditioning.py; ContextOnlyFeatureProvider caching,
   src/data/context_features.py; prepare_scfoundation_inputs,
   src/training/train.py -- ALL already built and unit-tested this session,
   never run on live CUDA hardware with the real checkpoint yet)
relative coords -> Fourier + MLP -> coord_feat [64]                   (NEW, same module as Arch 1)
[image_feat; gene_feat; coord_feat] -> concat -> Linear -> spot_token [512]
context capped to nearest ~80 + query token
  -> Transformer, 8 layers, 512 hidden, 8 heads                       (REUSED, same stack as Arch 1)
-> query hidden state -> PanelInvariantGeneDecoder(gene_names=shared_panel)
  (REUSED: src/models/conditioning.py::PanelInvariantGeneDecoder --
   gene-identity lookup decoder, chosen over the plain dense decoder here
   specifically because a foundation-model gene encoder is naturally
   panel-agnostic and it is wasteful to throw that away at the output side;
   also directly tests the cross-platform-decoder machinery on real data
   for the first time at this scale)
-> predicted expression [n_genes]
```
**Loss**: same `0.7·MSE + 0.3·(1-Pearson)` as Architecture 1.

**Trainable params**: ~0.1M (coord) + ~0.6M (fusion) + ~35M (transformer) +
LayerNorm+Linear scFoundation projection head (~scfoundation_dim×256, a few
million) + PanelInvariantGeneDecoder's gene-embedding table + MLP head
(depends on shared vocabulary size, typically 5-15M) ≈ **45-60M**. Frozen
GigaPath (~1.1B) + frozen scFoundation (100M) both outside budget.

**Real open risk, flagged explicitly for GPT**: `precompute_scfoundation_features`
requires CUDA and has never been run against the real checkpoint end-to-end
— this architecture's FIRST training run is also its first real numerical
verification. If it fails or produces garbage, that is a legitimate outcome
to report back, not a sign the plan was wrong.

## 4. Architecture 3 — Self-supervised gene autoencoder + spatial transformer in latent space

**Hypothesis being tested**: GPT's own most distinctive, most-recommended
idea (`"My preferred solution"` / `"One idea that I think could make your
project novel"` in the PDF) — decompose the problem into (a) a
domain-specific transcriptome autoencoder trained on ALL available HEST-1k
expression profiles, independent of the spatial task, and (b) a spatial
transformer that only has to predict a LATENT code, decoded back to genes by
the SAME autoencoder. GPT explicitly argues this generalizes better than
asking one network to map neighboring images+genes directly to a raw
20,000-gene vector, and is architecturally distinct from Architectures 1/2
(the transformer never sees or predicts raw gene space).

This is the one architecture with a genuinely new PRETRAINING STAGE, not
just a new forward-pass module.

**Stage A (pretrain, NEW — nothing like this exists in this repo yet)**:
```
full gene expression (normalized log1p, shared panel, n_genes wide)
  -> Linear(n_genes, 4096) -> GELU -> LayerNorm
  -> Linear(4096, 1024)    -> GELU -> LayerNorm
  -> Linear(1024, 256)                                    latent [256]
  -> Linear(256, 1024)     -> GELU -> LayerNorm
  -> Linear(1024, 4096)    -> GELU -> LayerNorm
  -> Linear(4096, n_genes)                                 reconstruction
```
Trained with plain reconstruction loss (MSE(log1p) + Pearson term, same
weighting as above) on every observed spot from every training slide —
NOT spatial, no context/query split, no masking; this is a pure
representation-learning pretraining pass over the full expression matrix.
Cheap relative to the spatial stage (~20-40M trainable params, no images, no
transformer, likely converges in a few hours, not a full day) — plan to run
this FIRST, before the 1-2 day spatial run, on a fraction of a GPU-day.

**Stage B (spatial, reuses Stage A's frozen/lightly-fine-tuned encoder+decoder)**:
```
H&E patch -> frozen GigaPath -> image_feat [1536]                     (REUSED)
context spot expression -> Stage-A encoder (frozen, or fine-tuned at a
  much smaller LR than the transformer -- GPT's own suggestion) -> gene_latent [256]
relative coords -> Fourier + MLP -> coord_feat [64]                    (NEW, shared module)
[image_feat; gene_latent; coord_feat] -> concat -> Linear -> spot_token [512]
context capped to nearest ~80 + query token
  -> Transformer, 8 layers, 512 hidden, 8 heads                        (REUSED stack)
-> query hidden state -> Linear(512, 256)                         predicted_latent
-> Stage-A DECODER (frozen or lightly fine-tuned) -> predicted expression [n_genes]
```
**Loss**: two terms — a latent-space loss (`MSE(predicted_latent,
true_latent_of_the_masked_spot)`, teacher-forced from Stage A's own encoder
applied to the real hidden expression, available during training since this
is supervised masking not real missing data) PLUS the same gene-space
`0.7·MSE + 0.3·(1-Pearson)` term applied after decoding, so the model is
never purely optimizing an internal latent nobody checks against real genes.

**Trainable params, Stage B**: ~0.1M (coord) + ~0.6M (fusion) + ~35M
(transformer) + ~1M (latent projection head) + Stage-A encoder/decoder
fine-tuning (if unfrozen, ~20-30M more at a 10x-smaller LR) ≈ **35-40M
frozen-backbone / up to ~70M if Stage A is unfrozen**, matching GPT's own
"35-70M trainable" target band exactly.

**Real open risk, flagged for GPT**: this is the only architecture requiring
a full 2-stage training pipeline (pretrain-then-spatial) to be built end to
end — more moving parts, more places for a subtle bug (e.g. a latent-space
train/eval mismatch) to hide. Worth GPT's specific scrutiny on whether the
2-stage split is worth the added complexity at our current data scale.

## 5. Architecture 4 — STPath-grounded hybrid (leans on our own best real result)

**Hypothesis being tested**: rather than a GPT-only design, this is the one
architecture built explicitly around what our OWN experiments already show
works best (STPath's real frozen pretrained backbone), corrected for the
one concrete, verified bug found this session (library-size-normalization
mismatch), with scFoundation layered in as an ADDITIVE residual signal
rather than a replacement — testing whether frozen STPath + frozen
scFoundation together beat either alone.

**Flow**:
```
H&E patch -> STPath's OWN real image tokenizer (ImageTokenizer, frozen,
  loaded as part of the real STFM checkpoint)                          (REUSED)
context spot expression, fed as RAW-COUNT log1p (the bug fix: no
  library-size normalization for this specific channel -- new
  `raw_counts`/`_scilifestdl_raw_library_size` plumbing added this session,
  src/data/loaders.py, threaded to a new "stpath_native_log1p" input mode)
  -> STPath's real GeneExpTokenizer one-hot scatter -> STFM's frozen
     spatial_transformer backbone (real pretrained STPath weights)      (REUSED:
     src/models/stpath_encoder.py::STPathContextEncoder, pretrained=True)
  -> STPath hidden state [d_model=512]
ADDITIVE residual channel: same context spot's expression -> frozen
  scFoundation encoder -> ScFoundationGeneEncoder projection -> [64]
  -> residual_proj -> added into STPath's hidden state before its own
     prediction head (REUSED mechanism: new_gene_encoder_type="novae"
     residual-injection pattern already built in STPathContextEncoder,
     generalized this session's scFoundation work to plug into the SAME
     slot -- confirm at implementation time that the "novae" residual slot
     accepts scfoundation features already, or extend it, this is a small
     wiring change either way, not a new mechanism)
-> STPath's own real prediction head (frozen) -> predicted expression
   in STPath's OWN vocabulary -> mapped back onto our shared gene panel
   by name (REUSED: existing gene-name alignment already used by
   stpath_pretrained_eval)
```
**Loss**: STPath's frozen prediction head is not being trained here (mirrors
`stpath_pretrained_eval`'s zero-trainable-parameter design) — the ONLY
trainable parameters are the scFoundation residual projection + its
`residual_proj` into STPath's hidden state. Loss is the same
`0.7·MSE + 0.3·(1-Pearson)`, but gradients only reach the small residual
path (matches the RAE pattern already used throughout this codebase:
frozen big representation + small trainable head).

**Trainable params**: on the order of the existing `new_gene_encoder_type`
residual arms already shipped (LayerNorm+Linear scFoundation projection +
a `residual_proj` into `d_model=512`) — a few million at most. This is
DELIBERATELY the smallest-trainable-parameter architecture of the 4, since
its entire bet is "the frozen pretrained representations are already good,
our own training was never the bottleneck, the preprocessing bug was."

**Real open risk, flagged for GPT**: this is the architecture most likely to
be capped by STPath's own domain mismatch (pretrained on ccRCC-style
Visium data per this repo's own STPath integration notes, not necessarily
on whatever organs the full-HEST-1k run covers) — worth explicitly checking
per-organ performance rather than only an aggregate number, since GPT's own
PDF-documented "full" (observed-image) vs `masked_zero` diagnostic on Lung
already showed the frozen STPath path struggling more on out-of-distribution
tissue than the missing-image-fill mechanism alone would explain.

## 6. Cross-architecture comparison

| | Image encoder | Gene encoder | Spatial reasoning | Decoder | Trainable params | Genuinely new code |
|---|---|---|---|---|---|---|
| 1. GPT-v1 faithful | frozen GigaPath | from-scratch MLP | 8L/512h/8heads, ~80-token local neighborhood | dense | ~45-55M | Fourier coord module, Pearson loss |
| 2. scFoundation | frozen GigaPath | frozen scFoundation (100M) | same transformer stack | PanelInvariantGeneDecoder | ~45-60M | Pearson loss (shared w/ #1); first live scFoundation run |
| 3. Latent autoencoder | frozen GigaPath | Stage-A self-supervised autoencoder (own pretrain) | same transformer stack, predicts LATENT not genes | Stage-A decoder | ~35-70M | Stage-A pretraining pipeline, latent loss, Pearson loss |
| 4. STPath hybrid | STPath's own frozen image tokenizer | STPath frozen (primary) + scFoundation frozen (residual) | STPath's own frozen spatial_transformer | STPath's own frozen head | few M (residual only) | raw-log1p input plumbing (mostly done), residual-slot wiring, Pearson loss |

All 4 share: local-neighborhood context capping, `target_zero` training
mode, the same held-out evaluation harness, and the same
`0.7·MSE(log1p) + 0.3·(1-Pearson)` loss shape (weights are a starting point,
not fixed — worth asking GPT whether 0.7/0.3 still holds at full-HEST-1k
scale vs. the small-pilot scale it was proposed for).

## 7. Training/masking/data operational plan

- **Masking, all 4**: single hole per item, `shape: mixed`, `radius_range`
  swept so holes cover roughly the GPT-suggested 15-30% of a neighborhood's
  spots at `9x9`-equivalent local density — needs a real radius_range sweep
  against full-HEST-1k's actual spot density (varies by platform/organ),
  not assumed from the Lung pilot's tuned `[5.0, 8.0]` spot_spacing value.
  GPT's masking curriculum (single-spot → contiguous → irregular,
  growing 10%→30% coverage) is NOT implemented as an automatic schedule
  yet — flagged as new infrastructure; first pass should use a FIXED
  representative setting (the curriculum's END state) rather than block
  the whole plan on building a step-dependent scheduler, unless GPT thinks
  the curriculum itself is important enough to build first.
- **Context cap**: `max_context_points: 80` (Architectures 1/2/3, matching
  the "9x9 neighborhood" ~81-token target) vs. STPath's own historical
  `max_context_points: 3000` (Architecture 4, since STPath's real
  pretrained weights were calibrated against a much larger context set —
  shrinking it to 80 for consistency with the other 3 would itself be an
  unintended intervention on a component we are NOT trying to change).
- **Data scope**: full HEST-1k, exact sample list TBD once we confirm what's
  staged on the server — plan for same-platform (Visium) first pass across
  as many organs as the shared-gene-panel intersection tolerates, holding
  out full SAMPLES (not spots) for validation/test per the existing
  `holdout_unit: sample` convention.
- **Batch size**: large enough for the Pearson loss term to be meaningful
  (needs multiple query spots per step to compute a real gene-wise
  correlation) — start at 32-64 query spots per step, matching GPT's own
  v1 recommendation.
- **Optimizer/schedule**: AdamW, lr 1e-4, existing warmup+EMA machinery
  already in `src/training/train.py` (`EMACallback`, warmup ramp) — reused
  unchanged.
- **Budget**: 1-2 days per architecture, 1 A100 each, 4 in parallel. Given
  full HEST-1k is much larger than the Lung pilot, epoch count needs
  re-deriving from real step-time on a smoke run rather than reusing the
  pilot's `epochs: 1` / `unique_mask_count` settings verbatim — first action
  once this plan is approved should be a short smoke run per architecture to
  measure real steps/sec before committing to a fixed step budget for the
  full 1-2 day run.
- **Evaluation**: same audit harness as the Lung pilot, plus the new
  `pcc_raw_log1p` notebook-comparable metric wherever STPath/scFoundation
  are involved (Architectures 2 and 4), so this round's numbers are directly
  comparable to the reference notebook's own STPath benchmark, not just to
  each other.

## 8. Open questions for GPT

1. Is the 0.7/0.3 MSE/Pearson loss weighting likely to hold at full-corpus
   scale, or does it need re-tuning once the dataset is ~100x larger than
   the pilot it was suggested for?
2. For Architecture 3's latent-space teacher-forcing: is supervising the
   predicted latent directly against Stage A's encoder output (rather than
   only the decoded gene-space loss) likely to help or to over-constrain the
   spatial transformer to Stage A's specific latent geometry?
3. Given Architecture 4 leans entirely on STPath's frozen pretrained weights
   (which may be domain-mismatched to non-ccRCC organs), is it worth a
   fifth, even smaller arm that's just Architecture 1/2 but explicitly
   ablates image conditioning entirely (a stronger harmonic-style control),
   given how competitive the current harmonic baseline already is?
4. Any of these 4 GPT would actively discourage running for a full 1-2 days
   given what's already known from the Lung pilot's results?
