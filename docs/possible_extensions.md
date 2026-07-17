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

## Cross-platform decoder ("target platform space" conditioning)

Surfaced 2026-07-17 from a user-drafted architecture roadmap (informally
compared against STPath/STORM/Novae's real architectures, all
independently verified this session — see conditioning.py/
storm_lite_encoder.py docstrings). The roadmap's proposed model
conditions its decoder on sequencing technology so it can output
expression for a *different* gene panel than the one the context data
came from (e.g. context from Visium, predict for Xenium's panel).

Not something this project currently has: every generator here
(WAE-GAN/FM-OT/VQ-VAE+AR) uses a dense, fixed-width `Linear(hidden_dim,
n_genes)` decoder tied to one specific gene panel at construction time —
unlike STPath's real tokenized gene output head (verified via its source,
stpath_encoder.py), which is panel-agnostic by construction. Adding this
would be a genuine decoder redesign (token-based/panel-invariant output),
not a small addition — deferred until the current architecture-selection
work (StormLite bias_type/fusion_mode/gene_encoder_type comparisons) is
settled, and until training data spans more than one platform (currently
INT1-24, all-Visium — see load_multi_sample's own docstring in
src/data/loaders.py). Revisit once genuinely multi-platform training data
is in hand; the multi-sample training infrastructure (organ_vocab/
tech_vocab, src/models/conditioning.py OrganTechEmbedding) is already
built and would need extending in the same spirit.

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
