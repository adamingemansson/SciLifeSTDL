# Results Log

Dated, report-facing notes on real-data findings that are worth citing
directly, as they're confirmed — separate from `docs/architecture_plan.md`
(design rationale/roadmap) and `docs/model_schematics.md` (build status).
Each entry: what was tested, the numbers, and the interpretation.

## 2026-07-17: WAE-GAN's residual-source ordering (Novae alone beats "both")

**Setup**: WAE-GAN + STPath's real pretrained fusion (`context_encoder_type:
stpath`), HEST-1k INT1, 10000 epochs, comparing STPath's Route-B residual
mechanism (`stpath_new_gene_encoder_type`) across its three options —
none (plain baseline), `both` (MLPGeneEncoder + frozen Novae, concat-mode),
and `novae` alone.

| config | PCC | ST-FID |
|---|---|---|
| `exp_hest1k_wae_gan_stpath.yaml` (plain, no residual) | 0.1975 | 12.66 |
| `exp_hest1k_wae_gan_stpath_bothresidual.yaml` (MLP+Novae) | 0.2901 | 3.03 |
| `exp_hest1k_wae_gan_stpath_novaeresidual.yaml` (Novae only) | **0.3453** | 3.44 |

**Finding**: Novae residual alone is WAE-GAN's best configuration found so
far — it beats the combined MLP+Novae ("both") residual, which in turn
beats the plain (no-residual) baseline. Adding MLP's signal into the mix
*hurts* relative to Novae alone for this generator family.

**Why this is likely real, not noise**: it echoes the same pattern already
seen for FM-OT (`docs/architecture_plan.md`'s Route-B ablation, 2026-07-16):
there, `mlpresidual` alone (PCC 0.4291) beat `bothresidual` (0.4154), which
in turn beat the plain baseline (0.3961) and `novaeresidual` alone (0.3904).
So across two structurally different generator families now, the
*combined* MLP+Novae residual underperforms whichever single source is
actually best for that family's training dynamics — FM-OT's best single
source is MLP, WAE-GAN's is Novae, but in both cases concatenating the two
sources dilutes rather than complements the stronger one. Reads as a real,
reportable result: naive concatenation of two residual signals isn't
free — it can cost accuracy relative to just using the better source alone.

**Not yet tested**: `wae_gan_stpath_mlpresidual` (MLP alone) — needed to
confirm Novae is genuinely WAE-GAN's optimum, not just better than MLP by
default. Queued as a follow-up, not yet run as of this entry.

**Related, same investigation**: WAE-GAN + StormLite (our own from-scratch
context encoder, not STPath) shows a reproducible mode collapse (PCC=nan,
constant output across 2 seeds — `exp_hest1k_wae_gan_stormlite_mome_both.yaml`,
seed 0 and seed 1) regardless of `fusion_mode` (`exp_hest1k_wae_gan_stormlite_both.yaml`,
sum mode, scored PCC -0.007 / ST-FID 47.9 — not collapsed to a constant, but
not learning anything useful either). Confirmed via the plain
`exp_hest1k_wae_gan_stpath.yaml` control above (finite, normal PCC=0.1975)
to be StormLite-specific instability, not a general WAE-GAN-family issue —
WAE-GAN is fine with a well-behaved context encoder, it's StormLite's added
complexity that destabilizes its adversarial training specifically.

## 2026-07-17: PanelInvariantGeneDecoder capacity + literature research

**Setup**: the panel-invariant decoder (`PanelInvariantGeneDecoder`,
`src/models/conditioning.py` — see its own docstring for the mechanism)
scored PCC 0.1308 on its first real run vs. the dense decoder's 0.2798 at
matched epochs (`exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder.yaml`
vs. `exp_hest1k_fm_ot_stormlite_mome_both.yaml`, both 10k epochs). Also
flagged separately: this decoder's real ~20GB training memory footprint
(vs. ~2GB for the dense decoder on the same run), traced to materializing
a `[N_query, n_genes, 2*hidden_dim]` tensor every forward pass (query
features concatenated with every gene's embedding, before the MLP).

**Capacity test**: bumping `decoder_gene_embed_dim` 64->256
(`..._paneldecoder_bigger.yaml`) improved PCC to 0.2039 — closes roughly
half the gap to the dense decoder, but ST-FID got slightly worse (3.30 ->
3.91). Capacity is a real, partial factor, not the whole story.

**Literature research** (2026-07-17, before assuming "just add more
capacity" was the right lever): checked how published single-cell/spatial
foundation models actually combine gene-identity signal with
expression/context signal, rather than guessing.
- **scGPT** (Cui et al. 2024, *Nature Methods*): combines a trainable
  per-gene identity embedding with an expression-value embedding via
  **element-wise addition**, not concatenation, to form each gene token.
  Directly actionable: our decoder used concatenation, which is both an
  arbitrary choice (not grounded in precedent) and the actual source of
  the ~20GB memory footprint (the `2*hidden_dim` channel doubling).
- **Geneformer** (Theodoris et al. 2023, *Nature*): genes are literal
  sequence positions, self-attended together, decoded via a shared
  per-position output head. Considered as a design for this decoder but
  **not adopted** — self-attention over our full ~16570-gene panel is
  `O(n_panel^2)` per query location, too expensive without the
  truncation/sparsity machinery Geneformer itself relies on (it typically
  processes ~2048 genes/cell, not the full panel).
- **LLOKI** (Levy et al. 2025, *Genome Research*) — directly relevant,
  not just structurally similar: a 2025 paper solving the exact stated
  problem this decoder targets ("integrating spatial transcriptomics
  across platforms without requiring shared gene panels"), via a
  conditional autoencoder conditioned on technology + gene panel (a
  learnable batch/panel token appended at both encoder and decoder
  input). A real, published alternative architecture for the same
  problem — noted here as a design point we're aware of and chose not to
  take, not adopted in this codebase. Worth revisiting if the
  gene-identity-lookup approach continues to underperform once real
  cross-platform data exists.

**Implemented**: `combine_mode` param on `PanelInvariantGeneDecoder`
(`"concat"` default, unchanged behavior for every existing config;
`"add"` — scGPT's real mechanism, roughly halves the decoder's memory
footprint) — plus `decoder_hidden_dim`/`decoder_mlp_depth`, which were
previously NOT independent knobs (the decoder's internal width silently
inherited whatever `ae_hidden_dim`/`cond_hidden_dim` the rest of the model
happened to use — an accidental coupling, not a deliberate choice).
`exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder_add.yaml` combines all
three (gene_embed_dim=256, combine_mode=add, mlp_depth=2) — queued, not
yet run as of this entry.

## 2026-07-17: GeneAttentionDecoder and LLOKIStyleDecoder implemented

Follow-up to the research above — both candidate designs that were
initially only *discussed* got implemented (`src/models/conditioning.py`),
not left as documentation:

**`LLOKIStyleDecoder`** faithfully ports LLOKI-CAE's real mechanism,
verified directly from source (`github.com/ma-compbio/LLOKI/lloki/cae/
conditional_autoencoder.py`, 2026-07-17) rather than from search-snippet
descriptions alone: encoder input = `concat(features, tech_embedding)`,
decoder input = `concat(latent, tech_embedding)`, multi-layer ReLU stack
(final layer unactivated), tech token concatenated once at input. Real
nuance found only by reading the source: LLOKI-CAE is only *half* of
LLOKI — its panel-invariance comes from a separate component (LLOKI-FP,
an external pretrained single-cell foundation model doing imputation
upstream) that isn't ported here, so `LLOKIStyleDecoder` is fixed-width
(same limitation as the original dense decoder), not panel-invariant.
What's genuinely useful and ported: technology-conditioned decoding
within a fixed panel — a different, complementary capability to
`PanelInvariantGeneDecoder`'s gene-identity lookup, not a competing
"better" version of it.

**`GeneAttentionDecoder`** implements the Geneformer-inspired design
(genes as self-attended tokens, shared per-position output head) that was
initially rejected for being too expensive over the full ~16570-gene
panel. Not dropped — implemented WITH a hard `MAX_SAFE_PANEL_SIZE=4096`
guard, since the design is completely tractable (and more expressive than
independent per-gene scoring, since attention lets gene predictions
depend on each other) for the actual realistic use case: a genuinely
smaller target panel (e.g. Xenium's ~300-500 genes), auto-derived here via
`sc.pp.highly_variable_genes` (scanpy's standard HVG selection, `n_top_genes=512`)
rather than the full training vocabulary.

**A real integration gap found and partially fixed while wiring this
in**: the training/eval pipeline assumed a decoder's output always covers
the FULL training panel (true for "dense" and "panel_invariant"'s default
use) — `GeneAttentionDecoder`'s deliberately restricted panel broke that
assumption. Fixed the TRAINING-loss path (`BaseGenerativeModel.
_slice_target_for_decoder`, an index buffer computed at construction from
`full_gene_names`, aligning the restricted output with the correct
columns of `target_expression`). **Not yet fixed**: `run_comparison.py`'s
shared FID/MMD machinery fits a PCA model on the full training-panel
width once per comparison run — incompatible with a restricted-panel
prediction. `decoder_type="gene_attention"` is therefore excluded from
real training runs until that's addressed; `decoder_type="lloki"` has no
such gap (fixed-width, drops straight into the existing pipeline).

**A real pre-existing bug found (unrelated to either new decoder, but
found while adding the second one)**: `WAEGAN.training_step` called
`self.decoder(...)` directly instead of through the `self._decode(...)`
dispatcher every other call site (including WAEGAN's own `sample()`)
already used — silently skipped `tech` conditioning during WAE-GAN
training for every `decoder_type`, and would have crashed outright for
`decoder_type="lloki"` specifically (its `forward()` requires `tech` as a
non-optional argument, no default). Fixed; regression-tested in
`tests/test_alternative_decoders.py::test_wae_gan_lloki_end_to_end`.

Queued for the 2026-07-17 night9 batch (`scripts/run_parallel_7gpu_night9.sh`,
10k epochs, weighted toward StormLite per current research priority):
`lloki` on StormLite (both/novae gene branches) and pretrained STPath
bothresidual; `panel_invariant` with `combine_mode="add"` on StormLite's
mlp branch; `lloki` on WAE-GAN+StormLite (diagnostic — does a different
decoder change the confirmed mode collapse, or is it entirely upstream).

## 2026-07-17: night9 real bug — tech/organ never populated, breaking `lloki`

**All 4 `lloki` configs crashed** on their first real run. Root cause:
`run_comparison.py` (the actual pipeline every parallel script uses) never
set `context["tech"]`/`query["tech"]` anywhere in the file — always
`None`, regardless of `adata.obs`'s real values. Harmless for every other
consumer (`OrganTechEmbedding` is gated on `organ_vocab`/`tech_vocab`
being set at construction, which no single-sample config does), but
`LLOKIStyleDecoder.forward()` requires a valid, non-`None` `tech` string
with no default — so it crashed immediately on `assert tech in
self.tech_to_id`. Fixed at the source (`_build_shared_eval`, `_evaluate`,
`_evaluate_shuffled_images`, and the training-dataset construction in
both `run_comparison.py` and `train.py`'s `main()`) rather than patched
around the symptom.

**The 3 non-`lloki` configs that DID complete**:

| config | PCC | ST-FID |
|---|---|---|
| `stormlite_mome_both_paneldecoder_add` (combine_mode=add) | 0.2622 | 2.26 |
| `stormlite_mome_mlp_paneldecoder_add` | 0.2215 | 2.11 |
| `wae_gan_stpath_mlpresidual` | **0.3603** | 4.11 |

`wae_gan_stpath_mlpresidual` completes the WAE-GAN residual-source
ablation: plain (0.1975) < both (0.2901) < mlp (0.3603) < **novae
(0.3453)**... wait — mlp (0.3603) actually edges out novae (0.3453) here,
reversing what looked like a clean "Novae is WAE-GAN's optimum" read from
the earlier single-arm result. Both are close (0.36 vs 0.35) and both
clearly beat "both" (0.29) — the real, robust finding is still "combining
MLP+Novae underperforms the better single source," just with the
identity of that single source now looking closer to a toss-up between
MLP and Novae for WAE-GAN specifically, unlike FM-OT where MLP alone
(0.4291) beat Novae alone (0.3904) by a clearer margin.

## 2026-07-17: 40k `stpath_unfrozen_bothresidual` vs. StormLite

| config | PCC (best available) |
|---|---|
| STPath pretrained `bothresidual` | **0.4717** (40k) |
| STPath unfrozen `bothresidual` | 0.3857 (40k) |
| StormLite `mome_both` | 0.3461 (40k) |
| StormLite `mome_both_paneldecoder_add` | 0.2622 (10k only) |
| StormLite `mome_novae` | 0.2968 (plateaued by 10k) |
| StormLite `mome_mlp_paneldecoder_add` | 0.2215 (10k only) |
| StormLite `mome_mlp` | 0.1708 (plateaued by 10k) |

The from-scratch, no-pretrained-weights STPath arm (0.3857) now beats
**every** StormLite arm run so far, including StormLite's own best
(`mome_both`, 0.3461). At this comparison point the gap isn't about
pretraining anymore — STPath's underlying architecture (its real spatial
transformer + tokenization scheme) is outperforming StormLite's from-
scratch fusion, independent of pretrained weights. Not a fully matched
comparison yet (the paneldecoder variants only have 10k numbers, `mome_both`
has 40k) — worth revisiting once every arm has a 40k number.

Separately: pretrained-vs-unfrozen `bothresidual`'s gap roughly HALVED
from 10k to 40k (0.161 → 0.086) — pretraining's advantage shrinks
substantially with more training rather than staying fixed, though it
never fully closes.

## 2026-07-17: multi-sample training extended to support images/Novae/STPath

Real gap closed: `_main_multi_sample`/`load_multi_sample_data`
(`src/training/train.py`) previously hardcoded `images=None` and never
computed Novae features — meaning multi-sample training (`cfg.data.sample_ids`
as a list, across INT1-INT8 rather than just INT1) could not exercise
StormLite+MoME+Novae or STPath+bothresidual at all, only `gene_encoder_type`
`"raw"`/`"mlp"`. Now loads images and precomputes Novae features per
sample (own Gigapath/Novae cache path each), mirroring `main()`'s
single-sample dispatch logic exactly. Also closed a separate real gap:
multi-sample `n_genes` had no auto-injector at all (previously a manual,
easy-to-get-wrong step — see `exp_hest1k_fm_ot_multisample.yaml`'s old
header) — now auto-derived from the real shared-gene-panel intersection
(`inject_multi_sample_n_genes`).

STPath itself is still not fully multi-sample-aware: `STPathContextEncoder`
uses one fixed `stpath_organ_type`/`stpath_tech_type` string across every
sample (its real IDTokenizer vocabulary, not per-sample-dynamic like
`OrganTechEmbedding`) — fine for the current all-ccRCC, all-Visium
INT1-INT24, a real limitation only if genuinely mixed-organ/platform
samples are used with STPath specifically.

Four configs queued for the first real multi-sample run
(`scripts/run_parallel_4gpu_multisample.sh`, INT1-INT8, weighted toward
StormLite): `fm_ot` + StormLite+MoME (both/novae gene branches), `fm_ot` +
STPath bothresidual (comparison anchor), `wae_gan` + STPath novaeresidual
(checks whether the single-sample result, PCC 0.3453, holds with more
data diversity). Bumped to 40k epochs and folded into the overnight batch
below rather than run standalone — see that entry.

## 2026-07-17: second lloki bug — tech_vocab assumed "Visium", real value is "unknown"

The `tech`/`organ` fix above unblocked the crash, but all 4 `lloki`
configs still failed with a different error: `AssertionError: tech
'unknown' not in decoder tech_vocab ['Visium']`. Real bug in my own
config assumption, not the pipeline: `load_adata()` never passes
`organ=`/`tech=` into `load_hest_sample()` for single-sample configs, so
`adata.obs["tech"]` is always the literal string `"unknown"` (that
function's own documented default), not `"Visium"` — the biological fact
that INT samples ARE Visium-sequenced was never true of what flows
through this specific pipeline path. Fixed all 5 `lloki` configs'
`tech_vocab` to `["unknown"]`. Since this is the only value that ever
appears for single-sample configs, `lloki`'s technology-conditioning is
exercised but not meaningfully differentiated (same "constant value,
contributes nothing but doesn't break" situation `OrganTechEmbedding`
already documents on single-organ data) — real technology conditioning
would need either genuinely multi-platform data, or `organ=`/`tech=`
wired from the config into `load_adata()` (a real, small follow-up,
not done here).

**One real, unexplained finding from the corrected run**:
`stormlite_mome_both_lloki` and `wae_gan_stormlite_mome_both_lloki` both
came back `PCC=nan` — but `stormlite_mome_novae_lloki` (0.2166) and
`stpath_bothresidual_lloki` (0.2240) both work fine. Looks specific to
StormLite's `gene_encoder_type="both"` (CombinedGeneEncoder) interacting
badly with `lloki` specifically, across both generator families tried —
not yet investigated further, flagged for follow-up.

## 2026-07-17: PeriodicCheckpointCallback

User request, motivated by tonight's overnight batch (several 40k-80k
epoch runs with no interactive supervision): a new opt-in
`training.checkpoint_every_n_steps` config field saves trainable weights
+ config + gene names every N steps, overwriting the same checkpoint path
each time (not versioned) — so a killed/crashed/disconnected job still
leaves a recent, loadable checkpoint behind instead of only ever saving
once at the very end. Reuses `save_trained_model` exactly (only trainable
params, not frozen backbones — see `save_trainable_state_dict`'s own
docstring on why: a STPath-conditioned model's full `state_dict()` is
~4.7GB, saving that every 10k steps for hours would be real, avoidable
cost). Wired into all 3 trainer-construction sites
(`run_comparison.py`'s `_train_model`, `train.py`'s `main()` and
`_main_multi_sample`). Unset by default — zero behavior change for every
existing config.

## 2026-07-17: overnight batch — 4 real levers to close the StormLite/STPath gap

The open question from the two entries above: STPath's from-scratch
(unfrozen) `bothresidual` (0.3857 @ 40k) beats StormLite's best arm
(`mome_both`, 0.3461 @ 40k) even with no pretraining advantage on either
side — the gap is architectural, not about pretraining. Since StormLite
is this project's priority architecture (GigaPath + Novae, built from
scratch rather than relying on STPath's external weights), tonight tests
the four most likely real causes, `scripts/run_parallel_8gpu_overnight.sh`:

1. **Training length**: `mome_both` pushed to 80k epochs (was still
   climbing at 40k, never plateaued — does it catch up?).
2. **Capacity**: `mome_both_bigger` — 4 transformer layers/8 heads/512-dim
   tokens instead of 2/4/256 (StormLite's current config is much smaller
   than STPath's real architecture).
3. **Position-bias choice**: `mome_both_relpos` — `relative_position`
   instead of the default `frame_averaging`, the first direct comparison
   at MoME scale (every earlier bias_type sweep predates the MoME-FFN
   fix, commit 00a188a).
4. **Data diversity**: the 4 multi-sample configs (INT1-INT8, now 40k
   epochs) plus a flagship 5th combining capacity + data diversity
   together (`multisample_fm_ot_stormlite_mome_both_bigger`) — the actual
   best shot at beating STPath tonight, since it addresses both
   hypotheses (architecture bottleneck, single-sample overfitting)
   simultaneously.

Not yet run as of this entry.

## 2026-07-18/19: overnight batch results — StormLite beats STPath (decoder lever), a real capacity collapse, and two real infra bugs

Results from the batch above (all real runs, `logs/parallel_run_overnight/`):

| job | PCC | RMSE | ST-FID | note |
|---|---|---|---|---|
| `mome_both` @ 80k (single) | 0.3329 | 0.3209 | 5.32 | **regressed** vs. 0.3461 @ 40k — training length is not the lever |
| `mome_both_bigger` (single, 4L/8H/512d) | **nan** | 0.3492 | 9.05 | mode-collapsed to a constant output (see below) |
| `mome_both_paneldecoder_add` (single) | **0.4933** | 0.2971 | 2.90 | **new best result of the project — beats STPath's 0.4717** |
| `multisample mome_both` | nan | 0.2745 | — | collapsed |
| `multisample mome_novae` | nan | 0.2745 | — | collapsed (identical RMSE to mome_both — see below) |
| `multisample mome_both_bigger` (flagship) | nan | 0.2745 | — | collapsed |
| `multisample stpath bothresidual` | 0.1244 | 0.2026 | — | far below single-sample STPath's 0.4717 |
| `multisample wae_gan stpath novaeresidual` | 0.1219 | 0.2096 | — | same multi-sample degradation |

**Headline: StormLite + `decoder_type="panel_invariant"` (`combine_mode="add"`) now beats STPath's best-ever result on every metric** (PCC 0.4933 vs. 0.4717, RMSE 0.2971 vs. 0.3053, ST-FID 2.90 vs. 1.95 — actually the one metric STPath still wins; PCC/RMSE/AUC all favor StormLite). The decoder swap is doing real work; training length (job 0) and raw capacity (job 1) are not the levers that close the gap.

**The `mome_both_bigger` collapse, root-caused**: `grep -i nan` on the raw log showed only `ConstantInputWarning: An input array is constant` from `pearsonr` — i.e. the model's predicted expression has ZERO variance per gene across the whole held-out set, for every gene. Not a literal NaN-weight divergence (RMSE stayed finite) — the decoder collapsed to predicting a constant (effectively the dataset mean), a classic MSE-only degenerate local minimum. This hit BOTH the single-sample and all 3 multi-sample "bigger" StormLite configs identically (the 3 multi-sample jobs even landed on the exact same RMSE, 0.2745 — strong evidence they all output the same constant against the same held-out draw), while the smaller default StormLite (2L/4H/256d) trained fine. The only architectural difference is capacity; no trainer in this codebase clipped gradients before this, and a deeper/wider fusion transformer (plus `FrameAveragingBias`'s attention bias) is meaningfully more prone to gradient-explosion-driven collapse early in training than the smaller default.

**Fix applied** (`src/training/train.py`, all 3 `pl.Trainer(...)` sites): `gradient_clip_val=1.0` added everywhere (safe, zero-cost if unneeded — standard default in the flow-matching/diffusion literature). Also added `PeriodicPrintCallback` (opt-in via `training.log_print_every_n_steps`, same `batch_idx`-based cadence as `PeriodicCheckpointCallback`) — found while investigating this that **none of these background/redirected runs left any per-step loss telemetry behind at all**: `logger=False` disables Lightning's own logger, and tqdm's progress bar auto-disables when stdout isn't a real terminal (true for every one of this project's `> logfile 2>&1` parallel launch scripts). The collapsed run's entire log had zero `train/loss` values — impossible to tell *when* it degenerated. `PeriodicPrintCallback` uses a plain `print()`, which always survives redirection.

**Separate, real infra bug found and fixed while re-running the test suite** (`tests/test_model_save_load.py` failed — not caused by the above changes, confirmed via `git stash`): `save_trainable_state_dict` filtered purely on `requires_grad` (`model.named_parameters()`), which silently drops every BUFFER. `RandomFourierFeatures.B` (the fixed random coordinate-encoding projection) is a buffer, not a parameter — on reload, `build_model()` reconstructs it with a *different* random value, so a reloaded model's positional encoding never matched what it was actually trained with. Same root cause would have silently reset `VectorQuantizer`'s entire EMA-updated codebook (`embed`/`ema_cluster_size`/`ema_embed_sum`) to random init on every VQ-VAE+AR checkpoint reload. This didn't affect any PCC number reported above (train+eval happen in the same in-memory process, same buffers) — but it would have silently corrupted any future `--skip-training` re-evaluation, exactly the safety net this week's `checkpoint_every_n_steps` work was built around. Fixed via `_is_frozen_backbone_module` (`train.py`): saves every non-frozen buffer alongside trainable params, while still correctly excluding genuine frozen pretrained backbones (Gigapath's `tile_encoder`, STPath's `self.model` when `pretrained=True`) by checking "owns >=1 parameter AND all are `requires_grad=False`" — this correctly distinguishes those from buffer-only modules (`RandomFourierFeatures`) and EMA-only modules (`VectorQuantizer`), neither of which have any `nn.Parameter` at all. 3 new regression tests added (`tests/test_model_save_load.py`).

**Still unresolved / next steps (today's batch, weekend daytime run)**:
1. Re-run `mome_both_bigger` (single-sample) with the gradient-clip fix — does it actually resolve the collapse?
2. Re-run the multi-sample flagship (`mome_both_bigger`) with the same fix, at scale.
3. Bigger capacity + the winning decoder (`panel_invariant`/`add`) combined — do the two real wins stack?
4. Winning decoder pushed to 80k epochs — does it keep climbing, unlike plain `mome_both`?
5. Multi-sample `mome_both` with the winning decoder swapped in — does it rescue multi-sample StormLite the way it helped single-sample?
6. Multi-sample STPath `bothresidual` re-run with gradient clipping — control: is STPath's multi-sample degradation (0.12 vs. 0.47) partly an optimization-instability issue too, or purely architectural/data-heterogeneity?
7. Winning config with `gene_encoder_type="novae"` (no `"both"`) — isolates whether the decoder swap is the dominant lever regardless of gene-encoder choice.
8. Winning config re-run with a different seed — is PCC 0.4933 reproducible, or a lucky draw?

## 2026-07-19: day1 batch results — "bigger" StormLite capacity confirmed dead (clipping made it WORSE), decoder swap holds up but needs a seed average

Full results from `scripts/run_parallel_8gpu_day1.sh` (`logs/parallel_run_day1/`):

| job | PCC | RMSE | AUC | ST-FID | note |
|---|---|---|---|---|---|
| `mome_both_bigger` (single, +clip) | nan | 0.4254 | 0.8608 | **377.87** | collapse got WORSE with clipping (was RMSE 0.3492/ST-FID 9.05 overnight, unclipped) |
| `mome_both_bigger_paneldecoder_add` (single, new combo) | nan | 0.5214 | **0.5000** | **689.69** | AUC=exact chance — total collapse to one universal constant, worse than the "predict-the-mean-per-gene" pattern seen before |
| `mome_both_paneldecoder_add` @ 80k (single) | 0.4681 | 0.3040 | 0.9211 | 1.06 | landed just below STPath's 0.4717 |
| `mome_novae_paneldecoder_add` (single, no "both") | 0.4557 | 0.3015 | 0.9183 | 1.73 | decoder swap still wins big even without the "both" gene encoder |
| `mome_both_paneldecoder_add_seed1` (single) | 0.4501 | 0.3030 | 0.9190 | 2.26 | below STPath's 0.4717 |
| `multisample mome_both_bigger` (flagship, +clip) | nan | **0.5247** | — | — | still collapsed; RMSE got worse than overnight's 0.2745 too |
| `multisample mome_both_paneldecoder_add` | **0.1361** | 0.2024 | — | — | **no longer nan** — decoder swap rescues the multi-sample collapse (was nan overnight with the dense decoder) |
| `multisample stpath_bothresidual` (+clip, control) | 0.1887 | 0.2005 | — | — | up from 0.1244 overnight — clipping DID help here |

**"Bigger" StormLite capacity (4L/8H/512d) is now a confirmed dead end, not investigated further**: `gradient_clip_val=1.0` didn't just fail to fix it — it made the single-sample collapse measurably worse (ST-FID 9.05 → 377.87), and stacking it with the winning decoder produced the most degenerate result seen in this project (AUC exactly 0.5000 — every prediction has converged to essentially one value, losing even the loose across-gene signal earlier collapsed runs still had). Multi-sample "bigger" also stayed collapsed with worse RMSE than before. Dropped entirely from `scripts/run_parallel_8gpu_day2.sh` — not worth more GPU-time chasing without a real root-cause diagnosis (LR warmup, different init, etc. — none attempted yet).

**Decoder swap (`panel_invariant`/`add`) confirmed as the one real, reliable lever** — but four data points (0.4933 seed0/40k, 0.4681 @80k, 0.4557 novae-only, 0.4501 seed1) all land in PCC 0.45-0.49, a massive win over StormLite's own dense-decoder baseline (~0.28-0.35), but **only the original seed0/40k run clearly beats STPath's 0.4717** — the others land at-or-below it. Not yet a statistically confident "StormLite beats STPath" claim on a single seed; needs averaging over more seeds.

**Gradient clipping partially validated as useful, independent of the "bigger"-capacity failure**: multi-sample STPath's own control run improved with clipping (0.1244 → 0.1887), and the decoder swap separately rescued multi-sample StormLite's dense-decoder collapse (nan → 0.1361) — two genuinely different fixes for two genuinely different problems, both real.

**day2 batch** (`scripts/run_parallel_8gpu_day2.sh`, launched — results pending): built entirely around the decoder swap, since it's the only proven lever. 3 more StormLite seeds (2/3/4) + 1 more STPath-with-decoder seed (toward a real seed-averaged StormLite-vs-STPath comparison, not single-run noise vs. single-run noise), the decoder swap extended to multi-sample STPath and multi-sample novae-only StormLite (does it rescue/improve those the way it did `mome_both`?), and the working multi-sample `mome_both_paneldecoder_add` pushed to 80k epochs (does more training help now that it's no longer collapsed?). Also queued but not yet run: `exp_hest1k_fm_ot_stpath_bothresidual_paneldecoder_add.yaml` @ 40k, the direct decoder-held-constant STPath comparison.

## 2026-07-19: day2/day3 results + deep code audit and three literature-grounded improvements

**day2/day3 seed-averaged picture** (all real runs). StormLite + `panel_invariant`/`add` decoder, per seed: 0.4933 (s0), 0.4501 (s1), 0.3657 (s2), 0.3896 (s3), 0.4782 (s4) — mean ~0.435 WITHOUT EMA. **With EMA (day3): seed5 = 0.5391** (new project best, RMSE 0.2874, ST-FID 0.79), seeds 2/3 for STPath+decoder = 0.5228/0.4968. STPath (pretrained) + decoder over 4 seeds (0.4874, 0.5061, 0.5228, 0.4968) averages ~0.503. **Honest read**: once decoder is held constant and seeds are averaged, StormLite+decoder (~0.435 raw, up to 0.539 with EMA on a good seed) is competitive with but not yet cleanly ahead of STPath's *pretrained* result — though it clearly beats STPath's *unfrozen* (from-scratch, no pretraining advantage) arm (0.3857 dense). Multi-sample stayed hard: everything landed PCC ~0.11-0.19; the `panel_invariant` decoder *hurt* STPath's multi-sample number (0.1136 vs 0.1449 with STPath's own dense decoder) even though it rescued StormLite's multi-sample collapse — a StormLite-specific fix, not universal. EMA barely moved the multi-sample numbers.

**Deep code audit (2026-07-19) — one real latent bug found:** `basic_qc_and_normalize` (`loaders.py`) applies `sc.pp.normalize_total` + `sc.pp.log1p` to `adata.X`, and then every gene encoder (`StormLiteContextEncoder._encode_gene`, `STPathContextEncoder`'s residual path) applies `torch.log1p` AGAIN — a genuine **double-log1p** that squashes the already-log-normalized expression's dynamic range a second time (log1p of [0, ~9.2] → [0, ~2.3]). It's a *consistent* confound (every model gets it, so past comparisons stay internally fair), and the prediction TARGET is correctly single-log1p, so it never broke anything outright — but it wastes representational range in the context signal. Made opt-out via `input_already_log1p` (StormLite), default False (preserves all history), and queued as an A/B tonight. The rest of the audit (MoME attention masking, mask-token gradient flow, EMA buffer handling, in-place assignments) came back clean.

**Three literature-grounded improvements implemented (all opt-in, prior-behavior-preserving defaults, 18/18 test files pass):**

1. **QK-normalization** (`storm_lite_qk_norm`, `_QKNormAttention` in `storm_lite_encoder.py`) — LayerNorm each attention head's queries and keys before the dot product (Henry et al. 2020, EMNLP, "Query-Key Normalization for Transformers"; the LayerNorm-on-head-dim variant from ViT-22B, Dehghani et al. 2023, and SD3's DiT, Esser et al. 2024). This is the real STRUCTURAL fix for the "bigger" StormLite collapse: a stress test (`tests/test_qknorm_timesampling_log1p.py`) confirmed that under blown-up Q/K weights (simulating the weight growth that destabilizes a bigger transformer), QK-normed attention logits stay bounded at ~2.8 while the un-normed path explodes to ~4200 — softmax saturating to near-one-hot (attention-entropy collapse) IS the documented root cause of the mode-collapse, and gradient clipping (which we already tried, twice, and which made it *worse*) caps gradient norm without touching that root cause. Only the MoME path threads it (that's the flagship and where the collapse happened). Reimplements attention manually only because `nn.MultiheadAttention` exposes no hook between the Q/K projection and the dot product; preserves the exact same additive-bias / batch-first contract.

2. **Logit-normal flow-matching timestep sampling** (`fm_time_sampling="logit_normal"`, FM-OT) — draws `t = sigmoid(m + s·eps)` instead of `U[0,1]`, concentrating supervision on the informative middle of the noise→data trajectory (SD3 / Esser et al. 2024, arXiv 2403.03206, which showed it outperforms plain uniform rectified-flow sampling and EDM/LDM-linear). Verified to stay in [0,1], center at 0.5, and put ~0.51 of its mass in the middle third vs uniform's 0.33.

3. **`input_already_log1p`** — the double-log1p opt-out above.

**day4 OVERNIGHT batch** (`scripts/run_parallel_8gpu_day4_overnight.sh`, launched — results pending): honors the requested structure (multisample StormLite / STPath-unfrozen / STPath-frozen, bigger StormLite, single-sample variants) while A/B-testing each new improvement against the 0.5391 bar. Multi-sample (0-2): StormLite+decoder+QK-norm, STPath-unfrozen+decoder, STPath-frozen+decoder, all +EMA, decoder held constant = clean encoder comparison. Single-sample (3-7, all +EMA): bigger StormLite +QK-norm+warmup+lower-lr @80k (THE test of whether QK-norm finally makes bigger capacity work), flagship +QK-norm (does it help the small model too?), flagship +logit-normal, flagship +input_already_log1p (double-log1p A/B), and flagship +ALL-three-stacked (the "best model" candidate). All override keys validated as real model params; not smoke-tested (built from proven override patterns), so watch the first `[step ...]` prints per job.

## 2026-07-19 (continued): three more literature-grounded improvements, a checkpoint-path bug found, and a genuine pretrain→finetune mechanism

**Superseded by the 16-config batch** (`scripts/run_parallel_16configs_2per_gpu_overnight.sh`, launched, 2 jobs/GPU sequential — results pending): doubles day4's 8 slots to 16, seed-averaging the best arms (EMA-baseline, QK-norm-small-model, all-3-stacked) toward real 3-seed means instead of single lucky draws, plus the full requested multi-sample structure (StormLite/unfrozen/frozen/bigger) and both A/Bs.

**Three more improvements implemented while that batch runs (all opt-in, defaults unchanged, 20/20 test files pass):**

4. **Minibatch OT coupling** (`fm_coupling="minibatch_ot"`, FM-OT) — closes a real naming gap: this class is called `FlowMatchingOT` for its straight-line OT-style path formulation, but its `z_0`↔`z_1` pairing was never actually OT-coupled — just an independent i.i.d. draw, no different from plain (non-OT) conditional flow matching. Real minibatch OT (Tong et al. 2023, "Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport"; Pooladian et al. 2023, "Multisample Flow Matching: Straightening Flows with Minibatch Couplings", ICML) solves the assignment between the noise pool and the real targets in one training step via the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`, exact, already a dependency) on squared-Euclidean cost — this project's own natural "minibatch" (every query point in one masking draw) makes this directly applicable with no new machinery. Verified: produces a genuine permutation of the noise pool (not new samples) with total cost ≤ the best of 300 random permutations of the same pool.

5. **Heun's 2nd-order ODE sampler** (`ode_solver="heun"`, FM-OT, `path_type="ot"` only) — the EDM paper's (Karras et al. 2022) own recommended sampler, which this project's `path_type="edm"` already cites but `path_type="ot"` never used (a documented simplification until now). Inference-time only, zero training-time effect. Verified mathematically correct: on a constant velocity field the predictor-corrector formula reduces EXACTLY to the analytic solution `z(1) = z(0) + v`, matching Euler bit-for-bit in that trivial case (proving the formula itself is right, not merely "different").

6. **`load_pretrained_weights_into` + `training.init_checkpoint_dir`** (`train.py`) — a genuine two-stage pretrain→finetune mechanism this codebase never had. Motivated directly by the pretraining-gap finding: STPath's own pretrained-vs-unfrozen ablation is worth ~0.086 PCC, architecture held constant, and StormLite has always trained directly on the target task with no pretraining stage at all. Unlike `load_trained_model` (reconstructs a FRESH model from a checkpoint's own saved config, for eval, and asserts strict trainable-param completeness), this loads INTO an already-built model, matching by `(name, shape)`, tolerant of architectural drift between pretrain and finetune configs (e.g. a different sample's gene-panel size changing a decoder's width) — mismatched-shape params are skipped and reported, not crashed on. Verified: exact transfer when architectures match; graceful partial warm-start (panel-agnostic layers only) when `n_genes` differs; params absent from the checkpoint (e.g. a decoder_type the pretrain run didn't use) correctly stay at fresh init.

**Real bug found while designing the pretrain config**: checked whether tonight's own checkpoints were safe to warm-start from, and found that `scripts/run_parallel_16configs_2per_gpu_overnight.sh` (and `day2`/`day3` before it) reuse the SAME base config file across multiple CONCURRENT jobs (different seeds/overrides) without ever overriding `training.checkpoint_dir` — so jobs sharing a config file clobber each other's periodic checkpoint saves in real time. An earlier script (`run_parallel_5gpu_night6.sh`) correctly overrode `checkpoint_dir` per seed variant; that pattern was dropped in this week's later scripts. **Does NOT affect any reported PCC/RMSE number** — every batch script evaluates each job's own in-memory model right after `trainer.fit()`, never reloading from disk for the comparison table. **Does** mean the on-disk checkpoint at a shared path is unreliable for `--skip-training` reuse or warm-starting — avoid pointing `init_checkpoint_dir` at any config that ran concurrently with a sibling reusing the same file this week. Fix for future scripts: always add a distinct `training.checkpoint_dir=...` override per job whenever a base config is reused concurrently (`run_parallel_5gpu_night6.sh`'s own pattern).

**Queued for tomorrow, not yet run**: a dedicated pretrain (multi-sample INT1-INT8, long schedule, own unique `checkpoint_dir`) → finetune (single-sample INT1, `init_checkpoint_dir` pointing at the pretrain run) config pair — see `configs/exp_pretrain_multisample_stormlite_mome_both_paneldecoder_add.yaml` / `exp_finetune_hest1k_stormlite_mome_both_paneldecoder_add.yaml`. This is the actual test of whether real pretraining closes StormLite's remaining gap to pretrained STPath, the way it does for STPath itself.

## 2026-07-19 (continued): 16-config batch — full results, QK-norm confirmed to fix "bigger" capacity for real, and a real negative result on stacking

Complete results from `scripts/run_parallel_16configs_2per_gpu_overnight.sh` (`logs/parallel_run_16/`), all 16 jobs.

**Multi-sample (6 jobs, INT1-INT8, all +EMA, decoder held constant = clean encoder comparison):**

| job | mean PCC | RMSE |
|---|---|---|
| `ms_stormlite_baseline_ema` | 0.1371 | 0.2008 |
| `ms_stormlite_qknorm_ema` | 0.1280 | 0.2014 |
| **`ms_stormlite_bigger_qknorm_ema`** | **0.1886** | 0.1976 |
| `ms_stormlite_all3_ema` | 0.0967 | 0.2034 |
| `ms_stpath_frozen_decoder_ema` | 0.1151 | 0.2013 |
| `ms_stpath_unfrozen_decoder_ema` | -0.0203 | 0.2269 |

**Headline win: multi-sample StormLite now clearly beats multi-sample STPath, on both arms** (0.1886 vs 0.1151 frozen / -0.0203 unfrozen) — and it's the "bigger" capacity model doing it, which was pure `nan` in every earlier multi-sample attempt (overnight batch, day1). This is the first genuine confirmation that QK-norm fixes the bigger-capacity collapse in the multi-sample setting, not just single-sample.

**Single-sample (10 jobs, flagship unless noted, all +EMA):**

| job | PCC | RMSE | AUC | ST-FID | ST-MMD | plausible |
|---|---|---|---|---|---|---|
| `flagship_baseline_seed10` | 0.3331 | 0.3250 | 0.8995 | 1.0387 | 0.0679 | 0.3810 |
| `flagship_baseline_seed11` | 0.5466 | 0.2846 | 0.9393 | 0.7150 | 0.0477 | 0.3333 |
| `flagship_qknorm_seed10` | 0.3971 | 0.3094 | 0.9096 | 0.6319 | 0.0341 | 0.2857 |
| `flagship_qknorm_seed11` | 0.4838 | 0.2878 | 0.9241 | 0.9477 | 0.0653 | 0.3333 |
| `flagship_logitnormal_seed10` | 0.4732 | 0.2999 | 0.9247 | 0.8995 | 0.0543 | 0.2857 |
| `flagship_nolog1p_seed10` | 0.5205 | 0.2907 | 0.9324 | 0.8638 | 0.0529 | 0.3810 |
| `flagship_all3_seed10` | **0.0582** | 0.3568 | 0.8464 | **6.4390** | **0.4548** | 0.1905 |
| `flagship_all3_seed11` | 0.5074 | 0.2848 | 0.9286 | 0.5401 | 0.0449 | 0.3333 |
| `bigger_all3_warmup_80k` | **0.5028** | 0.2831 | 0.9257 | 0.5483 | 0.0362 | 0.2381 |
| `bigger_qknorm_warmup_40k` | 0.3886 | 0.3058 | 0.9067 | 0.5134 | 0.0474 | 0.2381 |

**"Bigger" StormLite capacity is now confirmed genuinely fixed, not just patched over**: both bigger-capacity configs give real, competitive numbers (0.3886, and 0.5028 — matching the best flagship-scale results) instead of the `nan`/AUC-0.5000 total collapse seen in every prior attempt (overnight batch, day1). `bigger_all3_warmup_80k` in particular is now a legitimate flagship-tier result at 4x the capacity — directly answers the supervisor's "StormLite is even smaller than STPath" concern: bigger now actually works, given QK-norm + warmup.

**Seed-averaged picture for the small flagship, 2 seeds each + EMA:**
- baseline+EMA: (0.3331, 0.5466) → mean 0.4399; combined with day3's seed5+EMA (0.5391), 3-seed mean = **0.4729**
- qknorm+EMA: (0.3971, 0.4838) → mean **0.4405**
- all3+EMA: (0.0582, 0.5074) → mean 0.2828 — dragged down entirely by the seed10 collapse below

**Real negative result: stacking QK-norm + logit-normal + no-double-log1p together is NOT safe, despite each individually looking fine.** `flagship_all3_seed10` collapsed hard — PCC 0.0582, ST-FID 6.44 and ST-MMD 0.45 (both roughly 10x every other run's range, not ordinary seed noise), AUC dropped to 0.8464 (every other flagship run is 0.90-0.94). `flagship_all3_seed11` is fine (0.5074), and the two individual "all3" ingredients each look solid alone (qknorm mean 0.4405, logitnormal 0.4732, nolog1p 0.5205) — so this reads as a genuine seed-dependent interaction between the stacked changes, not any one of them being bad on its own. Not yet root-caused which pairwise combination is responsible. **Practical conclusion: don't default to stacking all literature-motivated improvements together** — QK-norm looks safe and worth keeping on by default (it's the actual fix for the bigger-capacity collapse, no observed downside at small scale either), but logit-normal timestep sampling and/or the log1p change should be re-tested paired with QK-norm alone (not all three at once) before adopting either as a new default.

**Where this leaves the StormLite-vs-STPath priority**: STPath pretrained+decoder's own 4-seed mean is ~0.503 (0.4874/0.5061/0.5228/0.4968, tight spread). StormLite's best individual results this batch (0.5466 baseline-seed11, 0.5205 nolog1p, 0.5074 all3-seed11, 0.5028 bigger-all3-80k) all land at-or-above that STPath mean — but StormLite's own seed-to-seed spread is still much wider (0.058-0.547) than STPath's, so no StormLite variant yet has a mean that clearly and reliably beats STPath's mean; it wins on best-case draws, not yet on typical-case reliability. Multi-sample is the one setting where the win is now clean and unambiguous (0.1886 vs 0.1151/-0.0203).

**Next steps**: (1) the already-prepared pretrain→finetune pair (`scripts/run_pretrain_then_finetune_stormlite.sh`) is the next real lever to try — untested by tonight's batch; (2) a smaller, targeted follow-up isolating QK-norm+logit-normal and QK-norm+nolog1p (2-way, not 3-way) across 2-3 seeds each would identify which pairing is actually safe to keep; (3) `bigger_all3_warmup_80k`'s result (0.5028, matching flagship scale at 4x capacity) makes "bigger + QK-norm + warmup" as the new default single-sample architecture worth strongly considering going forward, independent of the logit-normal/nolog1p question.

## 2026-07-19 (continued): st-a100 setup — real bugs found reusing a shared HEST-1k copy, plus a second architecture audit finding an undersized generative core

Moved onto a second machine (`st-a100`, 8x A100-80GB, shared with labmates — only GPU 0/6 free, confirmed via `ps` on every process on the other 6, not just an `nvidia-smi` snapshot). Reused a labmate's already-downloaded HEST-1k copy (`/data/hest`, 1.1TB, INT1-INT28) via symlinks rather than re-downloading, which surfaced three real, previously-latent bugs:

1. **`find`/`pathlib.rglob` don't descend into symlinked directories** — symlinking `data/raw/hest1k` -> `/data/hest` (or its subfolders) as directories meant `load_hest_sample`'s `rglob("*INT1*.h5ad")` silently found nothing. Fixed operationally (not a code bug) by making `hest1k`'s subfolders real directories containing per-FILE symlinks instead — leaf-level file symlinks are traversed by any tool with no ambiguity.
2. **Gigapath/Novae feature caches hardcoded to write inside `hest_data_dir`** — broke with `PermissionError` once that directory was a read-only shared dataset owned by someone else. Real code fix: `data.hest_cache_dir` (new optional config field, `_cache_root()` in `train.py`), defaulting to the old behavior (cache next to the data) when unset. This is a generally useful fix, not a one-off — any future shared-read-only-dataset reuse would hit the same wall.
3. **Single-sample `n_genes` was hardcoded per-config, unlike every other data-derived param** — the shared HEST-1k copy produced a genuinely different post-QC gene count (19179) than whoever originally hardcoded `16570`, crashing with a matmul shape error deep in `MLPGeneEncoder`. Fixed via `inject_single_sample_n_genes` (`train.py`), always deriving `n_genes` from the real loaded `adata` — mirrors what multi-sample training already did, just never applied to single-sample. Unconditional override (not `setdefault`): there's no legitimate reason for `n_genes` to differ from the real data.

All three confirmed fixed via a full 14-job `SMOKETEST=1` run of `scripts/run_2gpu_confirm_batches_st_a100.sh` completing end-to-end (2-epoch results are meaningless, but every job produced real output with no errors) — that 14-job full-day batch (bigger+QK-norm 4-seed mean, STPath-unfrozen "full retrain" 3-seed mean, the QK-norm/warmup 2×2 diagnostic, QK-norm+logit-normal and QK-norm+no-log1p 2-way disambiguation, and the pretrain→finetune run) is now running for real.

**Second architecture audit, while that batch runs** (first audit was the double-log1p finding, 2026-07-19 earlier): read through `FlowMatchingOT`'s actual generative core rather than re-examining the context encoder again. Real finding: `velocity_net` — the network that has to learn the entire conditional noise→data flow — is a plain 2-hidden-layer feedforward MLP with **no residual connections**, receiving the context vector `c` via **one-time input concatenation only** (not re-injected at deeper layers), and is **>100x smaller** than the context encoder feeding it (590K-459K params vs 63.6M-101M, per the smoke test's own param table). Every comparable published flow-matching/diffusion architecture (DiT, SiT, SD3's MM-DiT) uses residual blocks with per-layer AdaLN conditioning specifically because a plain deep feedforward net without either is hard to optimize and dilutes conditioning with depth — and every "make it bigger" experiment so far scaled the context encoder, never the velocity net.

**Implemented**: `_AdaLNResidualBlock`/`_AdaLNVelocityNet` (`registry.py`), opt-in via `velocity_net_type="adaln_residual"` (default `"mlp"` = old architecture, byte-for-byte unchanged). AdaLN-Zero initialized (Peebles & Xie 2023) — the modulation projection's weight and bias are zero-initialized, so every block is an exact residual identity at construction, the standard trick for making a deep conditioned stack trainable from scratch without a separate warmup schedule. Accepts the identical concatenated-tensor calling convention as the plain MLP, so `_velocity`/`_edm_denoise` call sites needed zero changes. 6 new regression tests (`tests/test_adaln_velocity_net.py`), full existing suite reverified passing.

**A real bug caught by the existing test suite while making this change**: `@register_model("fm_ot")` must sit directly above `class FlowMatchingOT` — inserting the two new helper classes between the decorator and the class silently registered `_AdaLNResidualBlock` as `"fm_ot"` instead. `test_pretrain_finetune_warmstart.py` (which goes through `build_model()`'s registry dispatch, unlike most other FM-OT tests which construct `FlowMatchingOT` directly) caught this immediately with a clear `TypeError`. Fixed by moving the decorator back to directly precede the real class.

**Investigated but not implemented**: organ/technology conditioning (`OrganTechEmbedding`). Its own docstring already documents the real constraint — every sample actually available (INT1-INT28) is the same organ (ccRCC) and platform (Visium), so it currently contributes a constant offset any model trivially folds into its bias terms. Improving its architecture right now would be unverifiable (nothing to distinguish on single-organ/single-platform data) — a data availability gap, not an architecture one. Left as-is, ready to revisit once genuinely multi-organ/multi-platform HEST-1k samples are downloaded.

## 2026-07-19 (continued): st-a100 abandoned mid-run (shared node, Markus + a second labmate both landed jobs on our GPUs), pivoted to tkdgx1 — real results that correct two earlier narratives

**st-a100 salvage, before killing the run**: 2 of 14 jobs completed for real before the node got too contended to keep waiting on — `bigger_qknorm_warmup_ema_seed11` (real, not collapsed: PCC 0.1877, RMSE 0.2258, AUC 0.8947, ST-FID 0.7956, plausible 0.6190 — weaker than the earlier confounded estimates suggested, exactly why a real seed-averaged mean matters) and `pretrain_finetune_stormlite` (a genuine, real negative result: PCC **nan**, AUC exactly **0.5000**, ST-FID 835.5, plausible **0.0000** — the classic "collapsed to a constant" signature, same as the pre-QK-norm bigger-capacity failures. This is the first actual end-to-end run of the pretrain→finetune mechanism — the weight-transfer *mechanics* were unit-tested and correct, but the resulting model didn't train well. Most likely explanation, not yet confirmed: multi-sample pretraining has been the weakest/least stable setting in this whole project (mean PCC ~0.10-0.19 even before this), so the pretrain checkpoint itself may already have been in a bad region; a second, structurally real risk is that only panel-agnostic layers transfer (gene-panel-dependent layers stay fresh-init, since the multi-sample shared panel differs in width from INT1's own panel) — the transferred layers were co-adapted with pretrain's own gene encoder, which suddenly gets replaced with a fresh one at finetune time. Not chased further given the pivot away from this mechanism for now; flagged as an open question, not silently dropped.

**8-config batch on tkdgx1** (`scripts/run_parallel_8gpu_tkdgx1_confirm.sh`, all 8 jobs parallel, real full-epoch runs) — full results:

| job | PCC | RMSE | AUC | ST-FID | plausible |
|---|---|---|---|---|---|
| `bigger_qknorm_warmup_seed10` | 0.5079 | 0.2832 | 0.9270 | 0.3463 | 0.3333 |
| `bigger_qknorm_warmup_seed12` | 0.5019 | 0.2838 | 0.9243 | 0.7094 | 0.3810 |
| `stpath_unfrozen_seed10` | 0.4125 | 0.3142 | 0.9139 | 1.0413 | 0.2381 |
| `stpath_unfrozen_seed11` | 0.5160 | 0.2920 | 0.9307 | 0.8518 | 0.2857 |
| `bigger_qknormonly_flatlr_seed10` | **nan** | 0.5214 | **0.5000** | 689.4 | 0.0476 |
| `bigger_warmuponly_noqknorm_seed10` | 0.3892 | 0.3066 | 0.9070 | 0.8532 | 0.2857 |
| `adalnvelocity_seed10` | 0.2086 | 0.3393 | 0.8767 | 8.4637 | 0.3333 |
| `adalnvelocity_seed11` | 0.3704 | 0.3206 | 0.9051 | 0.8345 | 0.2857 |

**Correction #1 — QK-norm was NOT the fix for the "bigger" capacity collapse; the LR/warmup change was.** The 2×2 factorial is now resolved: QK-norm alone (flat lr=1e-3, no warmup) **still collapses** (PCC nan, AUC exactly 0.5000 — identical signature to every pre-fix "bigger" failure). Warmup+lower-LR alone (no QK-norm) **works fine** (PCC 0.3892, healthy AUC/ST-FID). Both together also work (0.19-0.51 across seeds). This flips the mechanism attribution this project has been documenting since day1 — QK-norm isn't harmful and may still be worth keeping (no observed downside combined with warmup), but it was never the load-bearing fix. The lowered learning rate + linear warmup schedule was the actual fix the whole time.

**Correction #2 — STPath's real "full retrain" number is much stronger than previously measured, once the decoder is held constant properly.** 2-seed mean for `stpath_unfrozen` (decoder swap applied, same as every other current arm): **(0.4125 + 0.5160) / 2 = 0.464**. The old single data point (0.3857) predates the decoder swap entirely — it was never a fair comparison. This closes most of the previously-reported "pretraining advantage" gap: STPath pretrained's 4-seed mean is 0.503, so the real pretrained-vs-unfrozen gap (decoder held constant) is now only ~0.039, not the ~0.117 implied by the old numbers. Most of STPath's edge over from-scratch StormLite is architecture, not pretrained weights — a materially different conclusion than what was documented earlier this week.

**Where this leaves "bigger capacity"**: `bigger_qknorm_warmup`'s 3-seed mean (0.1877, 0.5079, 0.5019) = **0.399** — actually *below* the small flagship StormLite's own 8-seed mean (~0.462), despite the collapse being genuinely fixed. Bigger capacity is not currently showing a real win over the small flagship on average, once seed variance is accounted for — the earlier "bigger + QK-norm reaches flagship-level PCC" framing was true for individual lucky seeds (0.50-0.51) but not for the mean. Not a dead end (0.50+ is achievable), but not yet a confirmed upgrade either.

## 2026-07-19 (continued): matched-seed batch — the STPath-unfrozen "tie" doesn't hold up with more data

`scripts/run_parallel_8gpu_matched_seeds.sh` (3 more STPath-unfrozen seeds, 3 more StormLite-small seeds, 2 more StormLite-bigger seeds — all plain, non-stacked configs, all real full-epoch runs on tkdgx1). Note: this batch's own `SMOKETEST=1` sanity-check output was initially mistaken for real results (uniformly near-chance PCC/AUC across all 8 jobs) — that was correctly just the 2-epoch smoke test behaving as expected, not a bug; the real run below is what actually counts.

**Updated seed-averaged means (n = sample count, range = max-min spread across seeds)**:

| arm | n | mean | range |
|---|---|---|---|
| STPath unfrozen (full retrain, decoder held constant) | 5 | **0.4706** | 0.1041 (0.412–0.516) |
| StormLite small (flagship) | 11 | 0.4546 | 0.2135 (0.333–0.547) |
| StormLite bigger+QK-norm+warmup | 5 | 0.4388 | 0.3202 (0.188–0.508) |
| STPath pretrained | 4 | 0.503 | — |

**The earlier "tie" doesn't hold up with a fairer seed count.** With STPath-unfrozen's sample tripled (2→5 seeds) and StormLite-small's grown too (8→11), STPath-unfrozen now clearly leads StormLite-small (0.4706 vs 0.4546) rather than sitting at parity — a modest but real architectural edge even with zero pretraining advantage. STPath pretrained still leads further (0.503).

**A second, distinct finding worth noting**: STPath-unfrozen's seed-to-seed spread (0.104) is roughly HALF of StormLite-small's (0.214) and a THIRD of StormLite-bigger's (0.320) — STPath's architecture isn't just scoring slightly higher on average, it's also considerably more consistent run-to-run. StormLite-bigger in particular remains highly seed-variable (0.188 to 0.508) even with the collapse fixed and 5 seeds behind it.

**Honest read, now on firmer statistical footing than before**: StormLite does not currently beat STPath, with or without pretraining, on a fair decoder-controlled comparison — and STPath is also more reliable seed-to-seed. Bigger StormLite capacity still isn't earning its cost (mean below the small flagship, worst variance of any arm). This corrects and firms up the earlier "statistically indistinguishable" read from the thinner-sample comparison — worth taking as the current honest state of the project rather than continuing to look for the next tweak that closes the gap.

## 2026-07-19 (continued): overnight capacity batch — StormLite bigger's variance looks like ONE outlier, not general noise; STPath pretrained's ceiling firms up at ~0.51

`scripts/run_parallel_8gpu_overnight_capacity.sh` — 3 more `bigger_qknorm_warmup` seeds, 2 more AdaLN-velocity seeds, 2 seeds of a brand-new arm (raw velocity_net capacity, `hidden_dim` 512→1024, zero new code), and one more STPath-pretrained seed. Updated means:

| arm | n | mean | range |
|---|---|---|---|
| StormLite bigger+QK-norm+warmup | 8 | 0.4637 | 0.330 (0.188–0.518) |
| StormLite bigger+QK-norm+warmup, **excl. seed11** | 7 | **0.5031** | **0.031** (0.487–0.518) |
| AdaLN velocity net | 4 | 0.3875 | 0.302 (0.209–0.510) |
| Bigger velocity_net capacity (hidden_dim=1024) | 2 | 0.4216 | 0.149 (0.347–0.496) |
| STPath pretrained | 5 | 0.5060 | 0.035 (0.487–0.523) |
| StormLite small (flagship) | 11 | 0.4546 | 0.214 |
| STPath unfrozen | 5 | 0.4706 | 0.104 |

**A striking, worth-flagging-carefully pattern**: 7 of the 8 `bigger_qknorm_warmup` seeds now land in a remarkably tight band (0.487–0.518, range 0.031 — as tight as STPath pretrained's own spread) with a mean of **0.5031**, matching STPath pretrained's 0.5060 almost exactly. The 8th seed (seed11, 0.1877) is a clear, isolated outlier — nothing else in this arm's history comes close to that low. Reported both ways deliberately (0.4637 with it, 0.5031 without) rather than picking the more flattering number: excluding an inconvenient data point without a confirmed root cause would be cherry-picking, not analysis. **Not yet claiming this closes the gap to STPath** — it's a real, promising pattern (if seed11 genuinely is anomalous rather than representative variance, StormLite-bigger may already be at parity with STPath's pretrained ceiling), but needs either a root cause for seed11's failure or more seeds to confirm the tight cluster holds, before treating 0.50 as StormLite-bigger's real mean.

**AdaLN velocity net, now 4 seeds**: mean rose from 0.2895 (2 seeds) to 0.3875 (4 seeds) — the 2 new seeds (0.5102, 0.4608) are much healthier than the first 2. Still below StormLite-small's mean but the gap is narrowing; not enough data yet either way.

**Bigger velocity_net capacity (new arm), 2 seeds**: 0.3473, 0.4959 — mean 0.4216, mixed and too early to read anything into.

**STPath pretrained, now 5 seeds**: mean 0.5060, range only 0.035 — the tightest, most reliable arm in the whole project. This is the real target StormLite needs to match: not just beat the mean, but do so with comparable consistency.

**Next step**: more `bigger_qknorm_warmup` seeds specifically, to determine whether the tight ~0.50 cluster is real or seed11 was simply an unlucky draw within genuine wider variance. This is now the single most information-dense thing to run — it directly tests whether the flagship priority (StormLite beating STPath) is achievable with the current architecture.

## 2026-07-20: CRITICAL — Novae features leak masked query information into every Novae-involving config's context

**Independent audit** (external review pass, verified against the real code before accepting — this project's own established practice, same as every prior "corrected" narrative above) surfaced a real, structural leakage bug: **Novae's precomputed embeddings are graph-propagated over the WHOLE sample, computed BEFORE any masking draw exists.**

**Verified directly in code** (2026-07-20):
- `precompute_novae_features` (`src/models/conditioning.py:591-638`) calls `novae.spatial_neighbors(adata)` and `model.compute_representations(adata, zero_shot=True)` on the full, intact AnnData — no masking has happened yet.
- `get_novae_features` (`src/training/train.py:923+`) computes and caches this ONCE per sample, before any masking draw.
- `_build_masked_item` (`src/training/train.py`) only *indexes* `context_novae_features[context_mask]` from that precomputed array afterward — the embeddings themselves were already computed with the full spatial graph (including what later becomes the masked/query region) intact.

**Consequence**: because Novae's representations are graph-propagated (not per-spot-independent), a context spot adjacent to a masked hole legitimately carries graph-diffused information from its masked neighbors' real expression, before that expression is ever supposed to be "hidden." This affects every config using:
- `gene_encoder_type` in `("novae", "both", "tokenizer_novae")` — StormLite's context encoder
- `stpath_new_gene_encoder_type` in `("novae", "both")` — STPath's Route-B residual

That includes several of the strongest results logged above: STPath pretrained hybrid (~0.506), and every StormLite/STPath arm using `gene_encoder_type="both"` (the flagship default for most of this project's batches, including the "small flagship" 0.4546 mean, the "bigger+QK-norm+warmup" 0.5031 cluster, and the gene-tokenizer batch's `tokenizer_novae` arm).

**What this does NOT mean**: it does not mean these models are learning nothing real, or that the reported PCCs are meaningless — `gene_encoder_type="mlp"`-only and plain STPath/interpolation-baseline arms are unaffected and remain clean. It means every Novae-involving number above **cannot currently be interpreted as clean missing-region reconstruction** — the experiment cannot distinguish learned reconstruction from indirect graph-propagated access to hidden expression.

**Status: documented, not yet fixed.** Correct fix (not yet implemented): recompute Novae on a context-only AnnData/graph per masking draw (expensive per-draw; a cached fixed mask-bank is the practical compromise), or drop Novae from same-slide masking experiments entirely until a leak-free precomputation path exists. Until fixed, treat every `novae`/`both`/`tokenizer_novae` result in this log as **potentially contaminated by graph-level target leakage — not suitable for final comparison.** The leak-free reference points going forward are the `mlp`-only, image-only, plain-STPath, and non-Novae interpolation arms.

**Related scoping note, same audit pass**: the masking pipeline hides query gene expression but **never** hides the query location's real H&E image — confirmed via `_build_masked_item`'s `query["images"] = _images_tensor(images, query_mask)` (`src/training/train.py`), which indexes the real, unmasked image at query positions. This is a deliberate, documented design choice, not a bug, but it means every result in this log answers "predict hidden expression where tissue morphology is still visible," not the (also useful, currently untested) "predict expression where imaging is also unavailable." Worth testing under `target_zero`/`all_zero`/shuffled-image conditions before claiming the models are robust to physically missing tissue, not just missing assay measurements.

## 2026-07-20 (continued): held-out-spot generalization test built — real diagnostic for a second, distinct validity concern

**A second, independent review (separate session)** raised a further, mechanistically distinct concern from the Novae leak above: does the model just recall spots it saw as a **supervised training target** many times over a run, rather than genuinely reconstructing from context? Every training step gives a query spot's real coordinates and real H&E image as input while training the network via MSE to output that spot's real expression — masking is redrawn randomly every step across thousands of steps on the SAME slide, so most spots become a training target repeatedly before the model is ever scored on them. The existing "held-out masking draw" eval only guarantees a fresh hole *placement*, not spots the model has never been trained to predict — its own code comment ("Evaluate on a held-out masking draw not seen during training") was, on inspection, not actually testing what it claimed to.

**Verdict on the claim itself (before building anything)**: assessed as real and mechanistically plausible, not overblown — this project's own encoders have tens of millions of parameters (more than enough capacity to memorize a lookup over even 16,000 discrete spots), and 40k-step training runs give ample repeated exposure. This is a sharper, more concrete version of the *first* audit's own point 5 ("training and test are on the same slide... not a generalization experiment").

**Built (not yet run)**: `masking.held_out_mask` (`src/data/masking.py`) — a fixed, seeded subset of spots (`heldout_fraction`, default off). `make_context_query_split`'s new `heldout_mask` param (`src/training/train.py`) guarantees any held-out spot a masking draw would have placed in the query set gets reassigned back to context instead — held-out spots are structurally never a training target for the entire run. `evaluate_heldout_generalization` (new function) evaluates ONLY on those never-seen spots at the end of training, using `get_novae_features_context_only` (the same-day leak fix above) so this diagnostic isn't also confounded by Novae's graph-level leakage on top of the memorization question it exists to isolate. `main()`'s standard eval is also heldout-excluded for a fair "seen-eligible" baseline comparison, printed alongside the held-out number and their gap in every run's own log.

Opt-in (`masking.heldout_fraction` unset by default — zero effect on any existing config). 5 new regression tests (`tests/test_heldout_generalization.py`) verify the core guarantee directly: across 50 random masking draws, held-out spots never once appear in a training query set (and — sanity check — DO appear without the guard, proving it's doing real work, not a vacuous no-op).

**8-job diagnostic batch prepared** (`scripts/run_parallel_8gpu_heldout_generalization_test.sh`, tkdgx1, `heldout_fraction=0.15`, 20k epochs — a diagnostic pass, not a final result): StormLite flagship ×2 seeds, StormLite flagship with `gene_encoder_type="mlp"` (isolates memorization from the separate Novae-leak question), STPath unfrozen ×2 seeds, STPath pretrained hybrid (the project's current highest-scoring arm, most Novae-dependent), StormLite bigger+QK-norm+warmup (does more capacity memorize more?), and STPath pretrained with Novae residual disabled (clean non-Novae data point for the pretrained arm). **Not yet run — results pending.**
