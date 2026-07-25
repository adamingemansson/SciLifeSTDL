# gen2_architectures

A self-contained folder implementing the 4 architectures from
`docs/lung_round_gen2_architecture_plan.md`, revised per ChatGPT's review
of that plan (curated subset of its suggestions — see "What changed from
the original plan" below). Everything needed to run these architectures
lives here: reused infrastructure is **copied in**, not imported from the
rest of the repo, so this folder can be reviewed and run on its own.

Status: implemented and unit/integration-tested against synthetic data
(41 tests, `gen2_architectures/tests/`). **Never run against real HEST-1k
data or a real GPU yet** — see section 7 for what to fill in before the
first real launch, and section 8 for a recommended smoke-test step before
committing to a full 1–2 day run.

## 1. The four architectures, at a glance

| | Image encoder | Gene encoder | Spatial reasoning | Decoder | Trainable params* | Compute budget |
|---|---|---|---|---|---|---|
| 1. GPT-v1 baseline | frozen GigaPath | from-scratch MLP | 8L/512h/8heads local transformer, 80-token neighborhood | dense MLP | ~28M | full |
| 2. scFoundation | frozen GigaPath | frozen scFoundation (100M) | same transformer stack | dense MLP | ~28M | full |
| 3. Latent autoencoder | frozen GigaPath | Stage-A self-supervised **denoising** autoencoder | same transformer stack, predicts a LATENT not genes | Stage-A's own decoder | ~13M (Stage A) + ~26M (Stage B) | full (A is cheap, run first) |
| 4. STPath hybrid | STPath's own frozen image tokenizer | STPath frozen (primary) + scFoundation frozen (residual) | STPath's own frozen spatial transformer | STPath's own frozen head | a few M (residual only) | **half** |

*Trainable-parameter counts scale with the real full-HEST-1k gene panel
width (measured above against a 500-gene placeholder) — expect these to
land in GPT's own recommended 35–70M band once the real gene count is
known; not re-tuned against a placeholder number.

Plus two same-architecture ablations of Architecture 1 (`arch1b`/`arch1c`)
completing GPT review's suggested image/gene/both decomposition.

## 2. What changed from the original plan (ChatGPT review, applied)

The user chose the **curated subset**: adopt the cheap, clearly-good
changes; defer the bigger scope additions.

**Adopted:**
- Staged MSE→Pearson loss schedule (not a fixed 0.7/0.3 from step one) —
  `models/components.py::StagedGeneLoss`.
- Stage A is a **denoising** autoencoder (masks 20% of genes per row, not
  a plain identity-risking reconstruction) — `models/arch3_stage_a_autoencoder.py`.
- Mask/confidence embedding on every spot token — `models/components.py::ConfidenceEmbedding`.
- Image-only and gene-only ablations completing the 3-way decomposition —
  `configs/arch1b_image_only_baseline.yaml`, `configs/arch1c_gene_only_baseline.yaml`.
- Architecture 4 given **half** the compute budget of the other 3 (its own
  review flagged this as the riskiest architecture).

**Deferred** (explicitly out of v1 scope, per the user's choice — GPT
itself called these more experimental/"nobody does this"):
- Uncertainty prediction (μ, σ per gene + Gaussian/NB NLL loss).
- Contrastive/InfoNCE auxiliary loss on Architecture 3's latent space.
- Attention-based ("gene token") decoder as an alternative to the dense MLP decoder.

**Kept from the original plan, unchanged:**
- Local-neighborhood context capping via k-NN per query spot (GPT's
  central design idea).
- The latent-space supervision term for Architecture 3 Stage B (GPT
  review's own answer: keep it, the decoder already constrains the latent
  to stay meaningful).

## 3. Shared infrastructure

**Copied verbatim or lightly adapted from the main repo** (not
reimplemented — see each file's own docstring for exactly what changed,
usually just import paths):

| gen2_architectures path | Source | Notes |
|---|---|---|
| `data/loaders.py` | `src/data/loaders.py` | HEST-1k loading, QC, normalization, the `raw_counts`/library-size stash |
| `data/masking.py` | `src/data/masking.py` | Hole-drawing strategies |
| `data/mask_bank.py` | `src/data/mask_bank.py` | Fixed validation/test mask persistence, `cap_context_mask` |
| `data/context_features.py` | `src/data/context_features.py` | `ContextOnlyFeatureProvider` generic caching engine, PLUS a NEW `PrecomputedSpotFeatureProvider` (2026-07-25) for leakage-free features like scFoundation's — see section 13's disk-cost writeup |
| `data/patch_overlap.py` | `src/data/slide_context.py` (partial) | Only `nonoverlapping_context_patch_mask` — the WSI-dense-tile mechanism was dropped, unused by any gen2 architecture |
| `data/masked_item.py` | `src/training/train.py::_build_masked_item` (trimmed) | Novae/niche channels dropped; a generic `context_extra_features` additive channel replaces them |
| `models/conditioning.py` | `src/models/conditioning.py` | `GigapathPatchEncoder`, `MLPGeneEncoder`, `ScFoundationGeneEncoder`, `RandomFourierFeatures`, `PanelInvariantGeneDecoder`, `OrganTechEmbedding`, etc. |
| `models/stpath_encoder.py` | `src/models/stpath_encoder.py` | `STPathContextEncoder`, extended with a `scfoundation` residual arm (see file docstring, 2026-07-25) |
| `evaluation/audit_evaluation.py`, `metrics.py`, `cell_type_classifier.py` | `src/evaluation/*.py` | Mask-bank evaluation harness, incl. the `pcc_raw_log1p` notebook-comparable metric |
| `training/validation.py` | `src/training/validation.py` | `move_to_device`, `predictive_samples` |

**Genuinely new code** (does not exist anywhere else in the repo):

| Path | What |
|---|---|
| `models/components.py` | `CoordEmbedding`, `ConfidenceEmbedding`, `build_local_transformer`, `StagedGeneLoss` |
| `models/local_neighborhood_transformer.py` | Shared Architecture 1/2 implementation + `nearest_context_neighbors` (the actual per-query-spot local-neighborhood mechanism) |
| `models/arch1_gpt_baseline.py`, `arch2_scfoundation.py` | Thin named factories over the shared implementation |
| `models/arch3_stage_a_autoencoder.py` | `DenoisingTranscriptomeAutoencoder`, `corrupt_expression` |
| `models/arch3_stage_b_latent_transformer.py` | `Architecture3StageB` |
| `models/arch4_stpath_hybrid.py` | `Architecture4` wrapper |
| `models/eval_compat.py` | Deterministic-model shim so the (stochastic-model-oriented) evaluation harness works for plain regressors |
| `training/data_prep.py`, `checkpoint.py`, `evaluate.py` | Data loading orchestration, checkpoint save/load, evaluation wiring, `apply_sample_selection` |
| `training/train_local_neighborhood.py`, `train_arch3_stage_a.py`, `train_arch3_stage_b.py` | Training entrypoints |
| `data/hest1k_catalog.py` | Real HEST-1k metadata query + local-inventory cross-check, `resolve_sample_selection` (config-driven train/validation/test split, section 9) |
| `scripts/inventory_hest1k.py` | Human-readable local-vs-catalog Visium coverage report by organ |

## 4. Architecture 1 — GPT-v1 baseline

**Hypothesis**: GPT's own most conservative, concrete recommendation
already recovers a meaningful chunk of the gap to the reference STPath
notebook.

```
H&E patch (precomputed GigaPath feature, 1536-d, frozen)
  -> GigapathPatchEncoder (LayerNorm + Linear)          -> image_feat [256]
context expression (normalized log1p, shared gene panel)
  -> MLPGeneEncoder (512 -> 256 -> 256, GELU + LayerNorm)-> gene_feat [256]
(context_xy - query_xy), per query spot's own k=80 nearest context spots
  -> RandomFourierFeatures(32) -> MLP(64 -> 64)          -> coord_feat [64]
context H&E patch's own availability (bool)
  -> ConfidenceEmbedding (lookup, 2 rows)                -> conf_feat [16]
[image_feat; gene_feat; coord_feat; conf_feat] (592-d)
  -> Linear(592, 512)                                     -> spot_token [512]
  (+ organ/tech embedding added, if organ_vocab/tech_vocab configured)
spot_tokens (<=80, one per k-NN neighbor) + 1 learnable query token
  -> nn.TransformerEncoder, 8 layers, 512 hidden, 8 heads,
     GELU, dropout 0.1, pre-norm                          -> [k+1, 512]
query token's final hidden state
  -> MLP decoder: Linear(512,1024) -> GELU -> LayerNorm -> Linear(1024, n_genes)
                                                            -> predicted expression [n_genes]
```

**Loss**: `StagedGeneLoss` — pure MSE for the first 20% of training, then
0.9·MSE + 0.1·(1−Pearson) until 60%, then 0.7·MSE + 0.3·(1−Pearson) for
the rest. Pearson is computed per-gene across the batch's query spots.

**Local neighborhood**: for EACH query spot individually (not one shared
context window for a whole hole), the k=80 nearest OBSERVED context spots
by Euclidean distance — `models/local_neighborhood_transformer.py::nearest_context_neighbors`,
a real k-NN computation, not `mask_bank.py`'s coarser joint "nearest to
any query point" cap (that cap is still applied UPSTREAM to bound the
total context set size sent to the model per training item, for memory —
see section 6).

## 5. Architecture 2 — scFoundation gene encoder

Identical to Architecture 1 except the gene encoder. `context["expression"]`
is populated with precomputed, frozen scFoundation cell embeddings (NOT
raw/normalized expression) by a `context_gene_feature_provider` — the
REPLACEMENT channel (`models/local_neighborhood_transformer.py::_build_gene_encoder`,
`gene_encoder_type="scfoundation"` branch).

```
context expression (RAW counts through scFoundation's own real
preprocessing — see below, NOT our normalized log1p)
  -> frozen scFoundation encoder (100M params, 19,264-gene vocabulary)
  -> 4-way pool-concat (scFoundation's own real pooling scheme)
  -> gene_latent [scfoundation_dim, auto-detected at runtime]
  -> ScFoundationGeneEncoder (LayerNorm + Linear)         -> gene_feat [256]
... rest identical to Architecture 1 ...
```

**Preprocessing note** (verified against scFoundation's real cloned
source, `models/conditioning.py::precompute_scfoundation_features`):
unlike STPath, scFoundation's own real preprocessing formula
(`log1p(x / x.sum() * 1e4)`) happens to be **identical** to this project's
own default expression transform. So Architecture 2 correctly feeds
scFoundation our regular already-normalized-log1p expression with
`already_normalized_log1p=True` — no raw-count feed needed here, unlike
Architecture 4. (An earlier draft of the architecture plan document
conflated this with STPath's genuinely different requirement; corrected
during implementation after re-reading scFoundation's real source.)

**Open risk**: this is the first time this project's scFoundation
integration runs against a real checkpoint on real CUDA hardware. Requires
`SCFOUNDATION_REPO_PATH`/`SCFOUNDATION_MODEL_PATH` env vars.

## 6. Architecture 3 — self-supervised latent autoencoder

GPT's own favorite ("I would actually bet on Architecture 3 winning").
Two stages, trained separately.

### Stage A — denoising transcriptome autoencoder (`train_arch3_stage_a.py`)

```
full gene expression [n_genes] (normalized log1p)
  -> corrupt_expression: zero 20% of genes PER ROW (independently per spot,
     not the same genes across the batch), optional Gaussian noise on top
  -> Linear(n_genes, 4096) -> GELU -> LayerNorm
  -> Linear(4096, 1024)    -> GELU -> LayerNorm
  -> Linear(1024, 256)                                     latent [256]
  -> Linear(256, 1024)     -> GELU -> LayerNorm
  -> Linear(1024, 4096)    -> GELU -> LayerNorm
  -> Linear(4096, n_genes)                                  reconstruction
```
Trained to reconstruct the CLEAN (uncorrupted) target from the corrupted
input, on every observed spot pooled across every training slide — no
context/query split, no images, no spatial structure at all. `StagedGeneLoss`
again. This is cheap relative to Stage B; run it first, expect it to finish
well within the combined 1-2 day budget for the pair.

### Stage B — spatial transformer predicting a latent (`train_arch3_stage_b.py`)

```
H&E patch -> frozen GigaPath                                -> image_feat [256]
context expression -> Stage-A encoder (frozen, or fine-tuned
  at 10x smaller LR than the transformer if finetune_autoencoder: true)
                                                              -> gene_latent [256]
(context_xy - query_xy) -> CoordEmbedding                    -> coord_feat [64]
context image availability -> ConfidenceEmbedding             -> conf_feat [16]
[image_feat; gene_latent; coord_feat; conf_feat] -> Linear -> spot_token [512]
per-query-spot k=80 nearest-neighbor local transformer (identical stack
  to Architectures 1/2)                                      -> [k+1, 512]
query token's final hidden state -> Linear(512, 256)          predicted_latent
predicted_latent -> Stage-A's OWN decoder (frozen or fine-tuned)
                                                              -> predicted expression [n_genes]
```

**Loss**: two terms.
1. `latent_loss = MSE(predicted_latent, true_latent)`, where `true_latent`
   is Stage-A's own encoder applied to the REAL (uncorrupted) expression
   of the query spot — teacher-forced (available since this is supervised
   masking, not genuinely missing data). `Architecture3StageB.true_latent()`,
   always computed with `no_grad` (a target, not something the loss should
   pull the autoencoder's own weights toward matching).
2. `gene_loss = StagedGeneLoss(predicted_expression, true_expression, progress)`
   — applied AFTER decoding, so the model is never purely optimizing an
   internal latent nobody checks against real genes (GPT review's own
   answer to "should Stage B supervise the latent directly?": yes, keep
   both).

Total loss = `latent_loss_weight * latent_loss + gene_loss`.

**Real bug caught during implementation and fixed** (covered by
`tests/test_arch3.py::test_stage_b_finetune_reenables_gradient_even_on_a_previously_frozen_instance`):
`finetune_autoencoder=True` must actively RE-ENABLE gradients on the
passed-in autoencoder, not merely skip disabling them — `requires_grad` is
not part of `state_dict`, so a Stage-A checkpoint loaded fresh could
arrive already frozen from a prior caller.

## 7. Architecture 4 — STPath frozen hybrid

Leans on this project's OWN real result: STPath's frozen pretrained
backbone consistently outperformed every from-scratch alternative tried on
the Lung pilot (`configs/lung_round/`, this repo's earlier architecture
search). Fixes the one concrete bug found this session (library-size
normalization mismatch) and adds scFoundation as a small ADDITIVE residual.

```
H&E patch -> STPath's own real frozen ImageTokenizer (part of the loaded
             STFM checkpoint)
context expression, fed as RAW-COUNT log1p (NOT our library-size-
  normalized log1p — STPath's real pretrained weights and the reference
  notebook were only ever exposed to log1p(raw counts); see
  data/loaders.py's raw_counts/_scilifestdl_raw_library_size stash)
  -> STPath's real GeneExpTokenizer one-hot scatter
  -> STFM's frozen spatial_transformer backbone (real pretrained weights)
ADDITIVE residual: same context expression -> frozen scFoundation encoder
  -> ScFoundationGeneEncoder projection [64] -> residual_proj
  -> added into STPath's hidden state BEFORE its own prediction head
     (models/stpath_encoder.py's new_gene_encoder_type="scfoundation" arm,
     extended 2026-07-25 for this architecture — mirrors the "novae"
     residual arm already built and tested in the main repo)
-> STPath's own real frozen prediction head
  -> predicted expression, restricted to genes STPath's released
     vocabulary actually covers, mapped back onto the shared training
     panel by name via `_decoder_target_col_idx` (the evaluation harness
     automatically slices the target to match — this is a real, expected
     restriction, not a bug)
```

**Trainable parameters**: only the scFoundation residual projection + its
injection into STPath's hidden state — a few million at most. STPath
itself stays fully frozen (`pretrained=True`), matching the RAE pattern
used throughout this project (frozen big representation + small trainable
head).

**Known limitation, not worked around**: `STPathContextEncoder`'s
`organ_type`/`tech_type` are FIXED at construction (STPath's own real
`IDTokenizer` vocabulary), not a per-sample runtime value the way
Architectures 1–3's `OrganTechEmbedding` is. This architecture is
restricted to a single organ/tech (Lung/Visium in the shipped config)
rather than the full multi-organ HEST-1k corpus the other 3 can use.

**Compute budget**: HALF of the other 3 architectures (GPT review's #6
concern — STPath's frozen weights may be domain-mismatched outside the
organs it was pretrained on; be prepared to stop early if a smoke run's
loss curve looks flat/stuck).

## 8. Ablation baselines (Architecture 1 variants)

GPT review suggestion #8: complete the image/gene/both decomposition.

- `configs/arch1_gpt_baseline.yaml` — both modalities (the real Architecture 1).
- `configs/arch1b_image_only_baseline.yaml` — `context_gex_mode: zero`
  (removes gene expression, keeps H&E).
- `configs/arch1c_gene_only_baseline.yaml` — `image_mode: all_zero`
  (removes H&E entirely, keeps gene expression).

All three share the exact same architecture, capacity, and (deliberately)
the same `mask_bank_dir` — the comparison isolates information content per
modality, not a different model or different held-out regions.

## 9. Training plan

**Data scope**: full HEST-1k, Visium only (mixing technologies would
collapse the shared gene panel to whatever the narrowest targeted panel
covers, e.g. Xenium's ~few-hundred genes — a real, verified constraint of
`data/loaders.py::load_multi_sample`'s strict intersection, not a
convenience choice), **human only** (`species: "Homo sapiens"` — see
`data/hest1k_catalog.py::load_visium_metadata`'s docstring for the real
zero-gene-intersection bug found on the first actual multi-organ server
run, 2026-07-25: HEST-1k genuinely mixes 421 human + 181 mouse Visium
samples, confirmed against the live CSV, and nothing filtered on species
before that fix). The ~515-usable-Visium-samples/~24-organs figure
originally reported here (`scripts/inventory_hest1k.py`, also 2026-07-25)
was measured BEFORE the species fix and therefore counted some
now-excluded mouse samples — re-run the inventory script (now
human-only by default) for the real corrected count before relying on
specific numbers; the per-organ resolution logic itself (`min_samples_per_organ`,
`max_samples_per_organ`, below) is unaffected and will simply resolve to
fewer samples for any organ that had real mouse-only or mixed coverage.

**Second real bug found on the same server run, after the species fix**:
even human-only, a ~170-sample multi-organ selection still hit "No genes
shared across all samples." Real cause, confirmed against actual server
data: `load_multi_sample` used to apply `sc.pp.filter_genes(min_cells=3)`
INDEPENDENTLY to each sample before intersecting var_names — requiring a
gene to survive that per-sample threshold in literally every one of ~170
samples, spanning multiple HEST-1k source studies that turn out to use
different underlying reference-genome gene panels (confirmed: real INT1
has 36,601 raw genes vs real MEND139's 33,538, though their raw pairwise
overlap is still 87%). Enough independent per-sample QC decisions
compounded to an empty intersection. Fixed: gene-level QC is no longer
applied per sample; the shared panel is now the raw intersection across
samples (reflecting genuine measurement overlap, not per-sample QC
noise), and `min_cells` is enforced POOLED — a gene needs `min_cells`
total detected spots summed across the whole training cohort, not
independently in each one (`data/loaders.py::_pooled_min_cells_filter`).
See `tests/test_loaders_multi_sample.py` for a synthetic reproduction of
both the original bug and the fix.

**Third real bug, same server run, immediately after the pooled-min_cells
fix**: the RAW (pre-QC) intersection across all ~170 samples was ALSO
exactly zero — confirmed directly (`load_multi_sample`'s error message
changed to explicitly say "even before any gene-level QC" once the
above fix landed, isolating this as a genuinely different cause). Real
diagnosis (grouping real local samples by their HEST-1k source-study id
prefix and computing pairwise/cumulative raw panel overlap): every group
(`INT`/`MEND`/`MISC`/`NCBI`/`SPA`/`ZEN`) is real whole-transcriptome
scale (14,808-36,601 raw genes, `ZEN`'s panel is byte-identical to
`INT`'s) EXCEPT `TENX`, whose real raw panel is only **541 genes** — a
small targeted panel despite carrying the same `st_technology ==
"Visium"` label as everything else, exactly the Visium-vs-Xenium scale
mismatch this project already knew to avoid (see this section's own
opening paragraph), just hiding inside "Visium." Excluding `TENX`,
the real cumulative 6-way intersection is 13,234 genes — a healthy panel
size. Fixed via `data/hest1k_catalog.py::load_visium_metadata`'s new
`min_nb_genes` parameter (default 5000, comfortably between TENX's 541
and every legitimate group's 14,808+): filters on HEST-1k's own real
`nb_genes` metadata column rather than hardcoding the "TENX" prefix
(which could miss some other future small-panel study bundled under the
same label). See `tests/test_hest1k_catalog.py`'s
`test_min_nb_genes_excludes_small_panel_samples` for a synthetic
reproduction.

**Fourth real bug, same server run, immediately after the min_nb_genes
fix**: TENX samples were STILL in the resolved training set. Real cause,
confirmed against the live metadata: `nb_genes` is NOT cleanly bimodal —
real values vary CONTINUOUSLY within every single source-study group,
including the ones already trusted as "whole-transcriptome": real
confirmed `TENX` values include 538, 541, 1056, ..., 5001, 10006, 10017
(some clear a 5000 threshold despite still being a genuinely different,
incompatible panel), and real `NCBI` — otherwise a clean
whole-transcriptome group — turned out to ALSO include a 541-gene
outlier. A numeric size proxy cannot reliably separate compatible from
incompatible samples here; only real measured gene-IDENTITY overlap can.
Fixed via `data/hest1k_catalog.py::resolve_compatible_sample_ids`
(enabled by default in `resolve_sample_selection` as
`check_gene_panel_compatibility=True`): after the per-organ split, reads
every resolved sample's REAL var_names directly (`anndata.read_h5ad(...,
backed="r")` — cheap, doesn't load the expression matrix), builds a
"core" panel from genes present in ≥90% of the cohort, and drops any
sample covering <90% of that core — printing exactly which samples got
excluded and why, rather than silently shrinking or crashing. The final
shared panel is the EXACT intersection over kept samples (never an
approximation, never zero-filled). Adds real I/O at the start of every
training run (reading ~170-240 files' headers) — expect this step to
take some real wall-clock time before the first training step logs,
proportional to cohort size; it's a one-time cost per run, not
per-step. `species`/`min_nb_genes`/`check_gene_panel_compatibility`/
`min_gene_coverage`/`min_sample_coverage`/`min_panel_size` are all
overridable from a config's `data.sample_selection` block if the
defaults ever need tuning for a specific run
(`training/data_prep.py::apply_sample_selection`). See
`tests/test_gene_panel_compatibility.py` for synthetic reproductions of
both the failure mode and the fix, including a deliberately-crafted
outlier sample whose total gene COUNT clears 5000 but whose real overlap
with the rest of the cohort does not.

**Fifth real bug, same server run, after data loading finally reached
the held-out (validation/test) samples**: `held-out sample 'NCBI1' is
missing 21 genes from the fit-derived reference panel`, crashing the
whole run. Not a new mechanism — the expected, previously-flagged edge
case of `resolve_compatible_sample_ids`' ~90%-coverage compatibility bar
(bug four, above) being genuinely looser than `load_multi_sample`'s own
downstream reference_genes path, which requires EXACT (100%, zero
tolerance) coverage of the train-derived panel by design (a held-out
sample must never be allowed to shrink or otherwise influence the
vocabulary — see that function's own docstring). A sample can pass the
coarser cohort-level check yet still miss a few genes from the specific
exact panel the TRAIN samples alone ultimately settled on. Fixed via
`training/data_prep.py::load_held_out_samples_with_images`: loads each
validation/test sample INDIVIDUALLY, catching just this specific
"fit-derived reference panel" error per sample and skipping it (with a
printed warning) instead of losing the whole run to one incompatible
held-out sample — any other `ValueError` still propagates and fails
loudly, not silently swallowed. Wired into `train_local_neighborhood.py`
and `train_arch3_stage_b.py` (Stage A has no held-out expression-panel
loading, unaffected); `validation_sample_ids`/`test_sample_ids` are
reassigned to the actually-kept subset immediately after loading, so
every later use (scFoundation provider construction, the eval loops)
sees only samples that genuinely loaded. See
`tests/test_load_held_out_samples.py`.

**Sixth real bug, same server run, at the first periodic checkpoint save
(training had actually reached real steps by this point — loss
decreasing, diagnostics sane)**: `TypeError: Object of type ListConfig
is not JSON serializable`. Real cause: `_model_config_dict`/`build_model`
(and the equivalent lines in `train_arch3_stage_a.py`/
`train_arch3_stage_b.py`) built model-config-to-save dicts via
`dict(cfg.model.get("params", {}))` — a SHALLOW conversion. Any
nested list-valued param (`organ_vocab`/`tech_vocab`, auto-injected by
`apply_sample_selection`; or a literal YAML list like Stage A's
`hidden_dims: [4096, 1024]`) stayed as an OmegaConf `ListConfig` object,
which `json.dump` cannot serialize — every one of the 7 shipped configs
would have hit this on its first periodic checkpoint save, not just
Architecture 1's. Fixed by replacing every `dict(cfg.model.get("params",
{}))` with `OmegaConf.to_container(cfg.model.get("params", {}),
resolve=True)`, which recursively converts every nested
`ListConfig`/`DictConfig` into a plain `list`/`dict`. (The project
already has a more general `_jsonable` helper in `data/mask_bank.py`
for the same class of problem elsewhere — `OmegaConf.to_container` is
the more direct fix here since `cfg.model.params` is always a plain
OmegaConf container, never numpy.) See
`tests/test_model_config_json_serializable.py`.

Sample selection is resolved at RUN TIME, not hardcoded — every shipped
config sets `data.sample_selection` (organs, per-organ sample caps,
validation/test counts, a split seed) and
`data/hest1k_catalog.py::resolve_sample_selection` deterministically
queries the real HEST-1k metadata CSV + local inventory to build
`train_sample_ids`/`validation_sample_ids`/`test_sample_ids`/
`organ_by_sample`/`tech_by_sample`/`organ_vocab`/`tech_vocab` from it
(`training/data_prep.py::apply_sample_selection`, called at the top of
every training entrypoint). Shipped defaults: `organs: all`,
`min_samples_per_organ: 5` (excludes organs too small for a real
train/val/test split), `max_samples_per_organ: 30` (caps the largest
organs — Brain: 121, Skin/Kidney: 67 each — so epoch size stays tractable
for a 1-2 day budget), `n_validation_per_organ: 2`, `n_test_per_organ: 2`
— roughly 17 organs, ~240 training samples with these settings. Architecture
4 uses `organs: [Lung]` only (its documented single-organ constraint,
section 7) against the real 38-sample local Lung inventory. Architectures
1/1b/1c/2/3-Stage-A/3-Stage-B all share the EXACT same `sample_selection`
block (same organs/caps/seed), which resolves deterministically to the
same sample split — keeping every comparison in section 8 and the
Arch1-vs-Arch2 comparison fair (same data, not just the same architecture
capacity). A literal `data.train_sample_ids` list still works unchanged
for anyone who wants to bypass this mechanism (`apply_sample_selection` is
a no-op when `data.sample_selection` is absent).

**Masking**: single hole per training item, `shape: mixed` (each hole
independently circle/ellipse/irregular), `radius_range` tuned for the Lung
pilot's spot density — needs re-verification against full-HEST-1k's real
per-platform spot density before the real run (a sweep, not assumed to
transfer). `image_mode: target_zero` (query H&E withheld — the actual
missing_tissue task), `all_zero`/`full` reserved for evaluation-only
diagnostic modes.

**Context cap**: `max_context_points: 80` for Architectures 1/2/3
(matching the ~80-neighbor "9×9" target GPT's review endorsed as
appropriately sized for full HEST-1k). `max_context_points: 3000` for
Architecture 4 — deliberately NOT shrunk to 80, since STPath's real
pretrained weights were calibrated against a much larger context set and
changing that would be an unintended intervention on a component this
architecture is explicitly NOT trying to modify.

**"Batch size"**: there is no separate DataLoader batch dimension — each
training step draws ONE masked item (one hole), and every query spot
within that hole is processed as a batch through the shared transformer in
a single forward call. The effective per-step batch size is therefore the
hole's spot count, which varies stochastically with `radius_range` rather
than being fixed — Pearson's per-gene correlation needs several spots per
step to be meaningful, so `radius_range` should be tuned to keep typical
holes well above single-digit spot counts (already true of the inherited
Lung-pilot `radius_range: [5.0, 8.0]` spot-spacing setting, which produced
tens-to-hundreds of spots per hole in that context).

**Loss schedule**: `StagedGeneLoss` for every architecture (Stage A
included) — pure MSE for the first 20% of steps, 0.9 MSE / 0.1 Pearson to
60%, 0.7 MSE / 0.3 Pearson after that. Architecture 3 Stage B adds the
teacher-forced latent-space MSE term on top (section 6).

**Optimizer**: AdamW, `lr=1e-4`, gradient clip 1.0. Architecture 3 Stage B
uses TWO param groups when `finetune_autoencoder: true` — the transformer
at the configured LR, the autoencoder at `autoencoder_lr_multiplier` (0.1)
times that.

**Checkpointing**: `training/checkpoint.py` saves ONLY trainable parameters
and non-frozen buffers (frozen GigaPath/STPath/scFoundation backbones are
reloaded fresh from their own real pretrained source every time, never
re-saved) — plus `model_config.json`/`gene_names.json` even when there are
literally zero trainable weights (a real bug fixed in the main repo this
session: skipping metadata whenever nothing was trainable made a
fully-frozen checkpoint impossible to reconstruct later; not repeated
here — see `tests/test_checkpoint.py`).

**Evaluation**: the same audit harness as the rest of this project — fixed
mask banks, predictive-mean PCC/RMSE/nonzero-AUC/ST-FID/ST-MMD, plus the
notebook-comparable `pcc_raw_log1p` metric for any architecture touching
STPath or scFoundation (2 and 4). Every gen2 model is a plain
DETERMINISTIC regressor (uncertainty prediction was deferred, section 2),
so `evaluation.n_samples: 1` and `interval90_coverage`/`predictive_std`
will read as trivial/uninformative (~1.0/~0.0) by construction — see
`models/eval_compat.py`'s own docstring. PCC/RMSE/ST-FID/`pcc_raw_log1p`
are the metrics that actually matter for comparing these 4 architectures.

## 10. How to run

```bash
# Architecture 1 (GPT-v1 baseline)
python3 -m gen2_architectures.training.train_local_neighborhood \
    --config gen2_architectures/configs/arch1_gpt_baseline.yaml

# Architecture 1 ablations
python3 -m gen2_architectures.training.train_local_neighborhood \
    --config gen2_architectures/configs/arch1b_image_only_baseline.yaml
python3 -m gen2_architectures.training.train_local_neighborhood \
    --config gen2_architectures/configs/arch1c_gene_only_baseline.yaml

# Architecture 2 (scFoundation) -- requires SCFOUNDATION_REPO_PATH/SCFOUNDATION_MODEL_PATH
python3 -m gen2_architectures.training.train_local_neighborhood \
    --config gen2_architectures/configs/arch2_scfoundation.yaml

# Architecture 3 -- Stage A MUST run before Stage B; train_arch3_combined.py
# (2026-07-25) runs BOTH as one unsupervised job, splitting a single wall-clock
# budget between them (default 20% Stage A / 80% Stage B -- see its own
# docstring), so there's no manual "wait for Stage A, then launch Stage B" step:
python3 -m gen2_architectures.training.train_arch3_combined \
    --stage_a_config gen2_architectures/configs/arch3_stage_a_pretrain.yaml \
    --stage_b_config gen2_architectures/configs/arch3_stage_b_spatial.yaml \
    --total_hours 36
# (the two stages can still be launched individually if you want more manual
# control -- see train_arch3_stage_a.py/train_arch3_stage_b.py's own --help;
# each falls back to its own config's standalone max_wall_clock_hours then)

# Architecture 4 -- requires STPATH_GENE_VOC_PATH/STPATH_MODEL_WEIGHT_PATH
#                   and SCFOUNDATION_REPO_PATH/SCFOUNDATION_MODEL_PATH
python3 -m gen2_architectures.training.train_local_neighborhood \
    --config gen2_architectures/configs/arch4_stpath_hybrid.yaml
```

Every script resumes automatically from `training.checkpoint_dir` if it
already contains a checkpoint (checks for `model_config.json`).

## 11. What you need to fill in before launching

1. ~~Real full-HEST-1k sample list~~ — **done.** Every config now resolves
   its sample scope from `data.sample_selection` at run time against the
   real HEST-1k metadata + local inventory (section 9) — no hardcoded IDs
   left to fill in. Re-run `scripts/inventory_hest1k.py` if you download
   more samples later and want to raise `max_samples_per_organ` or lower
   `min_samples_per_organ` to pull in more organs.
2. ~~`organ_vocab`/`tech_vocab`~~ — **done.** Auto-injected at run time from
   the resolved `sample_selection` (`training/data_prep.py::apply_sample_selection`).
   Set them explicitly in a config only to override the auto-detected
   vocabulary. `OrganTechEmbedding` still fails loudly (`KeyError`) if a
   later run somehow sees an organ outside whatever vocabulary was
   resolved at construction time — intentional, not a bug (see
   `tests/test_arch1_arch2.py::test_architecture1_unknown_organ_raises`).
3. ~~`coord_scale`~~ — **done.** `models/components.py::CoordEmbedding`
   wraps the SAME `RandomFourierFeatures` class that had a real,
   previously-fixed aliasing bug at the wrong coordinate scale (see that
   class's own docstring in `models/conditioning.py`); the earlier
   hardcoded `1000.0` guess is now auto-derived per run from the REAL
   training data's own coordinate spread
   (`training/data_prep.py::derive_coord_scale`/`apply_coord_scale`,
   porting the exact formula `src/training/train.py` already established
   and debugged: mean, across training samples, of each sample's per-spot
   (x, y) std). Set it explicitly in a config only to override the
   auto-detected value. Architecture 4 (STPathContextEncoder conditions
   on organ/tech directly, no `CoordEmbedding`) and Stage A (no spatial
   component) correctly never get this injected — see
   `tests/test_coord_scale_and_smoke_override.py`.
4. **`radius_range`** (every architecture's `masking.params`, currently
   the Lung-pilot-tuned `[5.0, 8.0]`) — lower-risk than it looks: every
   config already sets `radius_unit: spot_spacing`
   (`data/masking.py::random_dropout_patches`), which expresses hole
   radii in median-nearest-neighbor-distance units, not raw coordinates —
   by construction this already makes hole size comparable across
   slides/organs with different spot density or pixel scale. Still worth
   a visual sanity check on a non-Lung organ during the smoke pass (see
   section 12), but this is not a per-organ guess the way `coord_scale`
   was.
5. **`total_steps`** — every config has a placeholder (100000, or 50000 for
   Architecture 4). Use `--smoke_steps N` (section 12) to run a short
   real pass first and derive a real budget from its measured steps/sec,
   rather than launching a full 1-2 day run on an unverified guess.
6. **Environment variables**: `SCFOUNDATION_REPO_PATH`, `SCFOUNDATION_MODEL_PATH`
   (Architectures 2 and 4), `STPATH_GENE_VOC_PATH`, `STPATH_MODEL_WEIGHT_PATH`
   (Architecture 4) — same resources this project's existing STPath/
   scFoundation configs already use.
7. **Architecture 3 Stage A → Stage B handoff**: set
   `arch3_stage_b_spatial.yaml`'s `model.stage_a_checkpoint_dir` to Stage
   A's `training.checkpoint_dir` once that run completes. `train_arch3_stage_b.py`
   checks the gene COUNT matches and raises if not, but cannot verify gene
   IDENTITY/order — keep the two configs' `data.sample_selection` blocks
   identical by construction (they already are in the shipped configs, so
   they resolve to the exact same sample split and gene panel).

## 12. Recommended before the real 1-2 day runs

Budget a short verification pass per architecture (GPT review's own
suggestion) before committing real compute. Every training entrypoint
accepts `--smoke_steps N` for exactly this — it overrides
`total_steps` down to `N` and scales `checkpoint_every_n_steps`/
`eval_every_n_steps`/`log_every_n_steps` down to match, in memory only
(no config file edits, nothing to remember to revert before the real
run):

```bash
python3 -m gen2_architectures.training.train_local_neighborhood \
    --config gen2_architectures/configs/arch1_gpt_baseline.yaml --smoke_steps 500
```

- A few hundred to a thousand real steps on real data/hardware.
- Confirm loss actually decreases, no NaNs/Infs.
- Confirm gradients reach the expected modules (`sum(p.grad.abs().sum() for p in model.parameters() if p.requires_grad)` > 0
  for every trainable submodule — the unit tests already verify this on
  synthetic data; re-verify once real data/hardware are in the loop, since
  real gradient flow through GigaPath/STPath/scFoundation's frozen forward
  passes is untested here).
- Confirm GPU utilization stays high (the per-query-spot k-NN neighborhood
  construction is CPU-bound `torch.cdist`+`topk` — verify it isn't
  bottlenecking the GPU-bound transformer forward pass at real batch
  sizes).

**Setting the real training budget (2026-07-25, revised)**: every
config's `total_steps` is now a large hard safety cap (1,000,000) that
should never actually be reached — the real practical stopping point is
`training.max_wall_clock_hours`, checked every step
(`training/data_prep.py::resolve_wall_clock_deadline`). This is
deliberately NOT "measure steps/sec from the smoke run, back-derive
total_steps" — a short smoke run's early steps include one-time
GigaPath/scFoundation cache-population overhead that isn't representative
of steady-state throughput, and per-step cost can drift over a genuinely
multi-day run (disk I/O contention with other architectures training
concurrently, etc.). Just say how long you want a run to actually take:

- Architectures 1/1b/1c/2: `max_wall_clock_hours: 36` (~1.5 days) each.
- Architecture 4: `max_wall_clock_hours: 18` (HALF — see GPT review #6's
  own risk note in that config).
- Architecture 3 (Stage A + Stage B): see section 10's combined-run
  section below — one `--total_hours` flag covers both stages together.

Whichever of `total_steps` or `max_wall_clock_hours` is hit first stops
training; either way the final checkpoint save and held-out test
evaluation still run normally (the training loop just `break`s, it
doesn't exit early). See `tests/test_wall_clock_deadline.py`.

## 13. Checkpoints: history, rollback, and disk budget

Every `save_checkpoint()` call still writes the "latest" files
(`trainable_weights.pt`, `model_config.json`, `gene_names.json`,
`training_state.json`) directly under `training.checkpoint_dir` — that's
what every script's resume logic reads on restart, unchanged. What's new
(2026-07-25): each save now also preserves a step-numbered snapshot under
`checkpoint_dir/history/step_XXXXXXXX/`, so a run that diverges or
corrupts state after a later save can be rolled back to an earlier
known-good step instead of losing everything since the last manual
backup.

**Disk cost, and why it's controlled** — the training server's real
budget is 50 GB total, shared across however many of the 5 configs run
concurrently, so checkpoint history is deliberately cheap by default:

- History snapshots are **hard-linked**, not copied (`os.link`, falling
  back to a real copy only if the filesystem can't hard-link across
  devices) — this avoids a second physical write at save time, though the
  eventual disk cost of keeping N snapshots is still real (each snapshot's
  data survives independently once the root path is replaced by the next
  save; hard-linking saves write I/O, not steady-state space).
- `training.checkpoint_keep_last` (default **2** if unset) caps how many
  history snapshots are kept per run — older ones are pruned automatically
  on every save. Add it explicitly to a config to change it, e.g.
  `checkpoint_keep_last: 1` to minimize footprint further, or `0` to
  disable history entirely (root-only, the old always-overwrite
  behavior).
- `save_checkpoint()` prints the weights file size and total history size
  on every save — watch this early in a run (the first few checkpoints)
  to see the REAL per-architecture cost before it's spent 1-2 days
  compounding across all 5 configs. Rough expectation: Architectures 1/2/4
  only save trainable weights (tens of MB — frozen GigaPath/STPath/
  scFoundation backbones are never re-saved), so history is cheap.
  Architecture 3 **Stage A**'s full autoencoder (`encoder`+`decoder`, both
  sides of a `genes -> 4096 -> 1024 -> 256 -> ... -> genes` MLP) is the one
  checkpoint whose size scales with the shared gene panel width — on a
  wide multi-organ panel this can run into the hundreds of MB per
  snapshot, so keep an eye on its printed size specifically and lower its
  `checkpoint_keep_last` first if the 50 GB budget gets tight.

**Rolling back**, once you've decided (from the loss curve or the
diagnostics in section 14) that a later checkpoint is bad:

```bash
# see what's available without changing anything
python3 -m gen2_architectures.scripts.rollback_checkpoint \
    --checkpoint_dir gen2_architectures/results/arch1_gpt_baseline --list

# overwrite the root ("latest") checkpoint with an earlier snapshot --
# the NEXT time you launch that training script, it resumes from here
python3 -m gen2_architectures.scripts.rollback_checkpoint \
    --checkpoint_dir gen2_architectures/results/arch1_gpt_baseline --step 42000
```

Rollback only touches that one `checkpoint_dir` — it doesn't affect other
architectures' runs, and it doesn't delete any history it wasn't told to
roll back to. If you request a step with no surviving snapshot (already
pruned, or a typo), it raises immediately and lists the real available
steps rather than silently no-op'ing.

**Feature caches (GigaPath, scFoundation) — the other real disk cost,
answered directly (2026-07-25)**:

- **GigaPath** (`training/data_prep.py::get_gigapath_features`): cached
  ONE FILE PER SAMPLE under `<hest_cache_dir>/gigapath_cache/{sample_id}.npz`
  — 1536 raw floats/spot (6.1 KB/spot). A ~1000-2000-spot sample costs
  ~6-12 MB; the whole shared training cohort (Architectures 1/1b/1c/2/
  3-Stage-B all resolve to the SAME sample set by design, section 9) is
  computed and cached exactly ONCE and reused across every one of those
  configs, not duplicated per architecture. Real order of magnitude for
  a few hundred samples: low single-digit GB total. Not a real concern
  against the 50GB budget.
- **scFoundation** — real bug found and fixed the same day, directly
  from a question about GigaPath's footprint: `build_scfoundation_provider`
  used to route scFoundation through the SAME context-only caching engine
  built for Novae (`ContextOnlyNovaeProvider`), whose cache is keyed by a
  digest of the CURRENT masking draw's context set. Training masking
  uses `seed=step`, so a fresh, essentially never-repeated context mask
  gets drawn on almost every step — meaning a new small `.npz` file got
  written on nearly every training step, for the life of the run, with
  near-zero real cache-hit rate. For Architecture 4
  (`max_context_points: 3000`, 50,000-step budget) this could plausibly
  have accumulated into hundreds of GB before the 50GB budget was ever
  hit as anything other than a training crash. scFoundation's own
  `precompute_scfoundation_features` docstring already said "ONCE per
  sample" — it just wasn't actually being called that way. Fixed via
  `data/context_features.py::PrecomputedSpotFeatureProvider`: computes
  every spot's scFoundation embedding ONCE (scFoundation has no
  query-leakage risk to guard against — it's computed per-spot from that
  spot's own expression alone, unlike Novae's graph propagation, see
  `model_uses_scfoundation`'s own docstring), disk-cached ONE FILE PER
  SAMPLE under `<hest_cache_dir>/scfoundation_cache/{sample_id}.npz`
  (deliberately a new directory name, not reusing the old
  `scfoundation_context_cache/` — avoids the new provider ever
  misreading an old per-mask-format file left over from a prior run),
  and every subsequent call is a pure in-memory slice by the current
  context mask — no redundant scFoundation forward passes, no per-step
  file growth. Same order-of-magnitude cost as GigaPath now (one file
  per sample). See `tests/test_precomputed_spot_feature_provider.py`.

## 14. Monitoring a run: what to expect

Every `log_every_n_steps` line now ends with the diagnostics GPT's
second-round code audit suggested watching ("those five plots can catch
many silent failures long before validation metrics do"):
`gene_embedding_norm`, `query_token_norm`, `decoder_output_norm` (and
`latent_norm` for Architecture 3, where there's a real bottleneck),
plus `query_token_param_norm` (the learned initial query token's own
drift). Coverage genuinely differs by architecture — see
`training/diagnostics.py`'s module docstring for exactly which signal
means what per architecture; Architecture 4's only trainable component is
the scFoundation residual, so its diagnostics are reported under the same
`gene_embedding_norm`/`decoder_output_norm` names but refer to the
residual encoder/injection, not a full model pass.

**What healthy looks like:**
- `loss`/`mse` trend down over the first few hundred to few thousand
  steps, not perfectly monotonically (this is single-sample SGD, not
  full-batch — expect noise step to step, look at the trend over a
  moving window).
- `pearson_penalty` starts near its unpenalized value and shrinks as
  `pearson_weight` ramps up in stage 2/3 (`components.py::StagedGeneLoss`)
  — if it's still large/flat once `pearson_weight` is near its max, the
  model isn't learning gene-gene co-variation structure, only marginal
  scale.
- The norm diagnostics should settle into a roughly stable RANGE after
  an initial adjustment period (first few hundred steps) — some drift is
  fine, that's training; the failure mode is one of them either collapsing
  toward 0 (that submodule's output stopped carrying information — a
  dead/saturated layer) or growing without bound (a sign of an
  undamped feedback loop, usually preceding a NaN).
- `query_token_param_norm` (Architecture 1/2/4) grows slowly and smoothly
  from its small random-init value (`torch.randn(hidden_dim) * 0.02`) —
  a sudden jump usually means the optimizer just took a large corrective
  step after something upstream misbehaved.
- Architecture 4's `decoder_output_norm` (the scFoundation residual
  injection) starting at exactly 0 is CORRECT and expected —
  `residual_proj` is zero-initialized by design (`stpath_encoder.py`) so
  training starts as a safe no-op on top of STPath's frozen predictions;
  watch that it moves AWAY from 0 over the first checkpoints, not that it
  starts nonzero.

**What to actually act on:**
- **NaN/Inf anywhere** (loss or any diagnostic) — stop the run, don't
  wait for it to "recover." Roll back (section 13) to the last checkpoint
  before the diagnostics started drifting toward this, lower `lr`, and
  restart from there.
- **`decoder_output_norm` flat at (near) 0 for hundreds of steps past
  init** — the model is predicting a constant/near-zero output regardless
  of input; check that gradients are actually reaching that submodule
  (`sum(p.grad.abs().sum() for p in model.parameters() if p.requires_grad)`,
  same check section 12 already recommends before a full run).
- **`attention_entropy`** (printed once per `eval_every_n_steps`, not
  every log step — it's the one opt-in, higher-cost diagnostic, see
  `diagnostics.py::compute_attention_entropy`'s own docstring for why) —
  near-zero entropy means the local transformer has collapsed onto
  attending to a single neighbor for every query, which is a real failure
  mode for a k=80 neighborhood (the model is throwing away most of its
  available spatial context); near-`log(k+1)` (maximally uniform) for a
  long stretch can mean the opposite — attention isn't learning to
  discriminate neighbors at all yet. Neither is fatal on its own this
  early, but worth a closer look if validation PCC also stalls at the
  same time.
- **Validation PCC/RMSE not improving while training loss keeps
  dropping** — the usual overfitting signature; check `n_query`/sample
  counts are what you expect and consider the run may need
  regularization or simply doesn't need the full `total_steps` budget.
- **`pcc_raw_log1p`** (Architecture 4's test-eval only, when it appears)
  is the notebook-comparable metric this session's STPath investigation
  built — expect it to differ from the normalized-space `pcc` above it;
  that's real signal-affecting rescaling, not a bug (see this session's
  earlier STPath preprocessing investigation).

None of these diagnostics change what gets checkpointed or how the loss
is computed — they're read-only monitoring, safe to ignore entirely if a
run is behaving and you just want to watch PCC/RMSE at `eval_every_n_steps`.

## 15. File map

```
gen2_architectures/
  README.md                              <- this file
  models/
    components.py                        NEW  Fourier coord embed, confidence embed, staged loss, transformer builder
    conditioning.py                      COPIED  frozen encoders, decoders, RAE-pattern building blocks
    stpath_encoder.py                    COPIED+EXTENDED  STPathContextEncoder + scfoundation residual arm
    local_neighborhood_transformer.py    NEW  shared Architecture 1/2 implementation
    arch1_gpt_baseline.py                NEW  Architecture 1 factory
    arch2_scfoundation.py                NEW  Architecture 2 factory
    arch3_stage_a_autoencoder.py         NEW  denoising transcriptome autoencoder
    arch3_stage_b_latent_transformer.py  NEW  Architecture 3 Stage B
    arch4_stpath_hybrid.py               NEW  Architecture 4 wrapper
    eval_compat.py                       NEW  deterministic-model shim for the shared eval harness
  data/
    loaders.py, masking.py, mask_bank.py, context_features.py, augmentation.py   COPIED
    patch_overlap.py                     COPIED (partial, from slide_context.py)
    masked_item.py                       COPIED+TRIMMED  _build_masked_item, Novae/niche dropped
    hest1k_catalog.py                    NEW  real metadata + local-inventory query, resolve_sample_selection
  evaluation/
    audit_evaluation.py, metrics.py, cell_type_classifier.py                     COPIED
  training/
    data_prep.py                         NEW  data loading orchestration, GigaPath caching (ported, bug-fixed logic), scFoundation provider wiring, apply_sample_selection
    checkpoint.py                        NEW  trainable-only save/load + step-numbered history/rollback (section 13)
    diagnostics.py                       NEW  per-step monitoring hooks (section 14)
    evaluate.py                          NEW  evaluation harness glue
    train_local_neighborhood.py          NEW  training entrypoint for Architectures 1/2/4
    train_arch3_stage_a.py               NEW  Stage A pretraining entrypoint
    train_arch3_stage_b.py               NEW  Stage B training entrypoint
    train_arch3_combined.py              NEW  runs Stage A+B as one unsupervised job, time-split budget (section 10)
    validation.py                        COPIED  move_to_device, predictive_samples
  configs/
    arch1_gpt_baseline.yaml, arch1b_image_only_baseline.yaml, arch1c_gene_only_baseline.yaml
    arch2_scfoundation.yaml
    arch3_stage_a_pretrain.yaml, arch3_stage_b_spatial.yaml
    arch4_stpath_hybrid.yaml
  scripts/
    inventory_hest1k.py                  NEW  human-readable local-vs-catalog Visium coverage report by organ
    rollback_checkpoint.py               NEW  operator CLI for checkpoint history (section 13)
  tests/                                 102 tests, synthetic data only, no real HEST-1k/GPU required
    test_components.py, test_arch1_arch2.py, test_arch3.py, test_arch4.py,
    test_checkpoint.py, test_diagnostics.py, test_masked_item.py, test_train_local_neighborhood_integration.py,
    test_hest1k_catalog.py, test_apply_sample_selection.py, test_coord_scale_and_smoke_override.py,
    test_loaders_multi_sample.py, test_gene_panel_compatibility.py, test_atomic_savez.py,
    test_precomputed_spot_feature_provider.py, test_load_held_out_samples.py,
    test_model_config_json_serializable.py, test_wall_clock_deadline.py, test_train_arch3_combined.py
```
