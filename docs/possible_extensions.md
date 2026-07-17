# Possible Extensions

Ideas and adjacent methods surfaced during scoping that aren't part of the
current plan (Track A/B, WAE-GAN + diffusion architecture) but are worth
revisiting later — either as stretch goals or as a prerequisite step that
got scoped out for now. Not verified to the same depth as
`docs/literature_review.md` unless noted.

## Text/label-conditioned tissue generation ("Track C")

Supervisor-suggested direction, set aside for now to avoid overcomplicating
the initial scope ("input a text like 'prostate with grade X', get back a
tissue"). Structurally different from Track A/B — generates *de novo* from
a clinical descriptor rather than completing/interpolating real tissue,
closer to text-to-image than to inpainting. Still unresolved: exact meaning
of "MCCI" (possibly a typo/mishearing), which grading system, and expected
output format (full spatial layout vs. expression profile only).

Candidate datasets already scoped if this gets revisited: Berglund et al.
2018 (*Nat Commun*, prostate cancer, Gleason-grade-annotated, KTH/KI/
SciLifeLab) and the Gleason-progression Visium study (*iScience* 2023);
HEST-1k for broader multi-cancer generality.

## Slice alignment / registration (excluded from the generative-model
literature search — not generative architectures, but relevant to Track B)

- **GEASO** (*Nature Communications*) — optimal-transport/graph-based
  registration for 3D spatial-omics reconstruction. Not independently
  verified in depth (surfaced during a broader search, not the focused
  fact-check pass the rest of this repo's citations went through).
- **3D-OT** (*Nature Methods*, DOI 10.1038/s41592-026-03034-9) — same
  caveat, not independently verified in depth.

Relevant if slice alignment turns out to be a needed prerequisite step for
Track B (see also PASTE/STAligner/STitch3D already in
`docs/literature_review.md`).

## Other diffusion-based ST imputation methods (found, not yet verified)

Surfaced alongside the Nature-family generative model search but excluded
there for venue reasons (bioRxiv/arXiv or non-Nature-family journals, e.g.
*Bioinformatics*) rather than for quality — **not independently
fact-checked**, treat as leads to check before citing:
- **SpaDiT** — diffusion-based ST method, not yet verified.
- **SpotDiffusion** — diffusion-based ST method, not yet verified.
- **scDiffusion** — diffusion-based single-cell method, not yet verified;
  also referenced as a baseline inside the Squidiff paper
  (`docs/literature_review.md` would need a new entry to add Squidiff too).

(stDiff and DiffusionST turned up in the same search but are already
verified and documented in `docs/literature_review.md`'s imputation
section — no duplicate entry needed.)

## Cross-platform decoder ("target platform space" conditioning) — IMPLEMENTED 2026-07-17

Surfaced 2026-07-17 from a user-drafted architecture roadmap (informally
compared against STPath/STORM/Novae's real architectures, all
independently verified this session — see conditioning.py/
storm_lite_encoder.py docstrings). The roadmap's proposed model
conditions its decoder on sequencing technology so it can output
expression for a *different* gene panel than the one the context data
came from (e.g. context from Visium, predict for Xenium's panel).

Every generator here (WAE-GAN/FM-OT/VQ-VAE+AR) originally used a dense,
fixed-width `Linear(hidden_dim, n_genes)` decoder tied to one specific
gene panel at construction time — unlike STPath's real tokenized gene
output head (verified via its source, stpath_encoder.py), which is
panel-agnostic by construction. This gap is now closed:
`PanelInvariantGeneDecoder` (src/models/conditioning.py) predicts
expression via a learned per-gene identity embedding looked up by name,
not a fixed output column, so the same trained decoder can be queried
against a different gene subset at inference time than it was trained on
— a continuous-regression simplification of STPath's real per-gene-token
classification head, not a literal port of its tokenizer/binning
machinery. Also takes an optional target-platform `tech` string (separate
from the context encoder's own source-platform tech conditioning),
matching the roadmap's "GEX decoder ... also fed by tech embedding"
arrow.

Opt-in via `decoder_type: "panel_invariant"` (default remains `"dense"`,
zero behavior change for every existing config) on any of the three
generator families; `decoder_gene_names` is auto-derived from the loaded
AnnData's `var_names` the same way `stpath_gene_names` already is (see
`inject_decoder_gene_names`, src/training/train.py) unless set explicitly.
See `tests/test_panel_invariant_decoder.py` for the mechanism-level checks
(shape, gradient flow, panel-subset scoring matches full-panel slicing
exactly, missing-gene assertion, tech conditioning) and
`configs/exp_hest1k_fm_ot_stormlite_mome_both_paneldecoder.yaml` for a
runnable sanity-check config.

**Still not validated end-to-end on genuinely cross-platform data** — no
multi-platform training set is currently available (INT1-24, this
project's only confirmed-available HEST-1k samples, are all-Visium — see
load_multi_sample's own docstring in src/data/loaders.py). On today's
data, `decoder_gene_names` is always auto-injected as the SAME panel used
for context/training, so every run so far exercises the mechanism at
`gene_names=None` (full-vocab default) rather than the actual
smaller/different-panel case the roadmap describes. Revisit the real
cross-platform test once genuinely multi-platform training data is in
hand; the multi-sample training infrastructure (organ_vocab/tech_vocab,
`OrganTechEmbedding`) is already built and generalizes the same way.

## Deprioritized architecture options

Already noted in `docs/architecture_plan.md`'s prioritization, repeated
here for visibility:
- **Vanilla conditional GAN** — considered, WAE-GAN chosen instead for
  lower training-instability risk on sparse/zero-inflated expression data.
  Worth revisiting if WAE-GAN underperforms and a supervisor specifically
  wants a "real" GAN comparison point.
- **Normalizing flows** — mentioned in the original project proposal;
  deprioritized due to invertibility constraints on the network and no
  pull from prior art reviewed so far.
