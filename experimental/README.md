# Experimental architecture variants (2026-07-23 round)

Working area for new `hierarchical_gene_transport_regressor` architecture
variants beyond the existing suite (`configs/recovery_suite/165-219`), per
the deep-research pass on published spatial transcriptomics
architectures/pretrained modules (see `docs/hierarchical_missing_tissue.md`
"Round 3/4 diagnostics" and the chat record for full citations).

Ground rule for everything in this folder: every variant must (1) test one
specific, falsifiable hypothesis, (2) share the same train/val/test slide
split, optimizer, schedule, and masking config as its comparison point
wherever the axis being tested allows it, and (3) be directly comparable in
a results table -- no scattershot changes without a stated purpose.

## Hypothesis matrix

| # | Variant | Hypothesis | Compares against | Status |
|---|---|---|---|---|
| 1 | Niche-conditioned global candidate (BANKSY) | A per-niche mean beats a flat whole-slide mean because local neighbors can be unrepresentative of the tissue a hole actually contains (e.g. a hole straddling a spatial domain boundary) | 215-219 (flat global candidate), C05/harmonic baseline | not started |
| 2 | BLEEP-style embedding-retrieval candidate | Content-similarity retrieval (learned joint image/expression embedding) selects better candidates than physical k-NN distance, at least as a competing signal in the gate | C05/harmonic baseline, geometry-only (C07/218) | not started |
| 3 | Alternate gene encoder(s) | `weighted_linear` (STPath-style) may not be the best gene encoder for this regime (real observed expression always available as context, 6-slide cohort) | C05/harmonic baseline | research in progress (background agent) |
| 4 | Alternate image encoder (lower priority) | Given geometry-only already wins twice, unlikely to move the needle, but cheap to test if GPU budget allows | C05/harmonic baseline | not started |

## Layout

- `README.md` -- this file, kept up to date as variants land.
- One subdirectory per variant once implementation starts (e.g.
  `niche_candidate/`, `embedding_retrieval/`, `gene_encoders/`), holding any
  precompute scripts, notes, and pointers to the actual model code changes
  (which live in `src/models/` alongside the existing architecture, gated
  behind new config flags -- this folder is for staging/notes/precompute
  artifacts, not a parallel copy of the model code).
- Final comparable configs for anything that graduates out of here land in
  `configs/recovery_suite/` following the existing naming/collision-safety
  conventions (distinct `experiment_name`/`checkpoint_dir`/
  `training_mask_bank_path`, shared `evaluation.mask_bank_dir` only when the
  masking config genuinely matches).
