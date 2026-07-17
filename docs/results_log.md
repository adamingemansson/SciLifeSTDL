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
Results not yet available as of this entry.
