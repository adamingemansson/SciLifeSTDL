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
| 1 | Niche-conditioned global candidate (BANKSY) | A per-niche mean beats a flat whole-slide mean because local neighbors can be unrepresentative of the tissue a hole actually contains (e.g. a hole straddling a spatial domain boundary) | 215-219 (flat global candidate), C05/harmonic baseline | implemented + tested, configs 230-235 queued |
| 2 | BLEEP-style embedding-retrieval candidate | Content-similarity retrieval (learned joint image/expression embedding) selects better candidates than physical k-NN distance, at least as a competing signal in the gate | C05/harmonic baseline, geometry-only (C07/218) | not started |
| 3 | Alternate gene encoder(s) | `weighted_linear` (STPath-style) may not be the best gene encoder for this regime (real observed expression always available as context, 6-slide cohort) | C05/harmonic baseline | research done, see below -- implementation not started |
| 4 | Alternate image encoder (lower priority) | Given geometry-only already wins twice, unlikely to move the needle, but cheap to test if GPU budget allows | C05/harmonic baseline | not started |

## Variant 3 (gene encoder) -- research summary (2026-07-23)

Two independent research passes, reconciled:

- **Our own `WeightedGeneExpressionEncoder` is verified (from STPath's real
  source) to be bit-for-bit STPath's actual `gene_embed` mechanism** --
  `nn.Linear(n_genes, hidden_dim, bias=False)`, no nonlinearity. Not a
  simplification of STPath, a faithful copy of it.
- **We already have an internal, architecture-held-constant ablation** in
  `docs/results_log.md`: STPath pretrained vs. STPath unfrozen/from-scratch,
  mean PCC 0.503 vs. 0.4706 -- pretraining's edge was only ~0.03-0.04 PCC,
  with the log's own conclusion that most of STPath's advantage is
  architecture, not pretrained weights.
- **Four independent external sources converge on the same direction**:
  CellBench-LS (bioRxiv 2026), Kedzierska et al. (*Genome Biology* 2025,
  zero-shot FM limitations), Souza & Mehta (bioRxiv 2026, parameter-free
  reps beat FMs OOD), and Ahlmann-Eltze/Huber/Anders (*Nat. Methods* 2025,
  linear baselines beat scGPT/scFoundation/GEARS on perturbation
  prediction) -- pretrained expression FMs tend to lose specifically on
  *quantitative value-prediction* tasks (vs. cell-type/classification
  tasks) and in low-data/OOD settings. Our task is exactly that shape.
- **BLEEP's own expression encoder** (verified from the paper we read in
  full) is just a plain FCN, no pretraining mentioned, and BLEEP still beat
  HisToGene/ST-Net by 39-120% PCC -- a fifth, independently-read data point
  for "simple is enough here."
- **CellCharter's own expression embedding** (also verified from the paper
  we read in full) uses a VAE (scVI) -- a real, different nonlinear
  architecture the first research pass missed entirely because it didn't
  know we'd already read CellCharter for a different reason (niche
  detection). scVI is built for batch-corrected spot/cell-level embeddings,
  not single-cell-token pretraining, so it sidesteps the
  single-cell-vs-pseudobulk distribution-mismatch risk flagged against
  scGPT/Geneformer/UCE below -- but like everything else here, it's
  unproven for *conditioning a predictive transformer* specifically (its
  proven use is clustering).

Candidates, ranked by engineering risk (lowest first):

1. **`weighted_linear` vs `mlp`** -- both already implemented in
   `hierarchical_slide.py`, never benchmarked against each other on this
   model. Zero risk, config-only, run first.
2. **`TokenizedGeneEncoder`** (set-attention over genes, from scratch) --
   already implemented and used by `StormLiteContextEncoder`, never wired
   into `hierarchical_slide.py`'s `gene_encoder_type` option or benchmarked
   there. Low risk, no new dependency; needs an HVG-reduced gene subset
   (attention cost caps below our full ~16k panel).
3. **scVI/VAE-style expression embedding** -- new candidate from the
   CellCharter cross-reference above. Would need a scVI dependency and a
   training step (fit per-cohort or per-slide) before it can feed the
   transport head. Medium risk, no precedent for our exact conditioning use
   case.
4. **Frozen STPath gene-embedding table alone** (not STPath's whole
   architecture) spliced into our own encoder -- isolates "does the
   *pretraining* help" from "does STPath's whole transformer help," which
   our internal ablation above conflates. Medium risk: needs gene-vocab
   alignment via `symbol2ensembl.json` (already in the repo from the STPath
   integration).
5. **BulkFormer** (bulk RNA-seq pretrained, MIT-licensed, public
   checkpoints) -- structurally the closest pretraining distribution to a
   Visium spot's dense pseudobulk profile of any candidate found, but zero
   precedent as a conditioning encoder for a spatial model, new dependency,
   new checkpoint, new vocab alignment. Exploratory, do last, only if 1-4
   show real signal.

**Explicitly not recommended**: scGPT, Geneformer, UCE, scFoundation as
frozen encoders -- real, unverified risk that their single-cell training
distribution (sparse, rank/bin-encoded per-cell) doesn't transfer to a
spot-level pseudobulk vector, and no precedent found for using them this
way for spatial *conditioning* (only as pieces of an ensemble, or as a
distillation teacher like PEKA -- different usage patterns than ours).

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
