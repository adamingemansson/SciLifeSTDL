# 2026-07-13 — Architecture & preliminary workflow session

Consolidated summary of this session's decisions — what's settled, what's
built, what's next. Full technical detail lives in `docs/architecture_plan.md`
(design + workflow schematic), `docs/literature_review.md` (Mimyr/MORPHE),
and `docs/metrics_notes.md` (scoring plan). This note is the short version.

## Scope recap

Two active tracks, one shared architecture (`docs/project_outline.md`):
- **Track A** — fill in broken/missing tissue within a slice (gene
  expression; histology image reconstruction is a possible extension, not
  in current scope — see `docs/architecture_plan.md` "Known gaps").
- **Track B** — generate gene expression data *between* real slices to
  build a 3D tissue reconstruction.

## Models chosen ("favorites")

Prioritized in `docs/architecture_plan.md`:
1. **VAE** — built (`src/models/registry.py` `VAEBaseline`). Proves the
   pipeline plumbing end-to-end.
2. **WAE-GAN** — built (`WAEGAN`). Our GAN entry — adversarial regularizer
   on the encoder's latent code only (Tolstikhin et al. 2017), chosen over
   a vanilla conditional GAN for lower training-instability risk on sparse/
   zero-inflated expression data.
3. **Diffusion / Flow Matching** — chosen, **not yet built**. Next priority
   — every close prior-art paper (Mimyr, isoST, stDiff, LGDiST) is
   diffusion-based, so this is where real comparability lives. Flow
   Matching (Lipman et al. 2022) preferred over classic DDPM for simpler,
   more stable training.
4. Normalizing flows — considered, deprioritized (no prior-art pull,
   restrictive network constraints).

## Key architecture decision: skip explicit cell-type conditioning

Mimyr (closest prior art) generates location → cell type → expression, in
three chained stages. We generate **location → expression directly** — no
discrete cell-type commitment inside the generation path. Full reasoning in
`docs/architecture_plan.md` "Design decision: no explicit cell-type
conditioning":
- Avoids the bias of forcing continuous expression through a fixed external
  taxonomy, and avoids Mimyr's error-propagation risk (misclassified type →
  wrong conditioning downstream).
- **Real trade-off, not a free win**: explicit type conditioning is also an
  efficiency scaffold; removing it is a harder learning problem. The bet is
  that the bias/generalizability gain is worth it — to be checked
  empirically.
- **Why the backbone comparison matters here specifically**: without a
  discrete type variable, a location can be genuinely multimodal (cell-type
  boundary). A plain-regression backbone (MSE, vanilla VAE) risks collapsing
  to a blurry average; diffusion/GAN-style sampling shouldn't have this
  failure mode. This is a concrete, checkable reason to compare backbones,
  not just "try things for its own sake."
- **Validation plan**: an independent cell-type classifier, trained on
  held-out real data, checks whether generated expression still recovers
  sensible types post-hoc. Reuses the ARI/NMI metric already planned.

## Comparison target

**Mimyr is the primary benchmark** — same core task, close enough for a
real head-to-head. Reproduce it on our own data rather than just cite its
numbers. **isoST** is the secondary reference for Track B specifically.
Both have public code (`gkrieg/mimyr`, `deng-ai-lab/isoST`).

Why bother given Mimyr already exists and works: not because the task is
unsolved, but to build and understand a working version ourselves, check
whether removing the cell-type bottleneck actually helps, and do it on our
own data. A legitimate internship-scale goal on its own.

## Scoring plan

- Standard suite (already built, `src/evaluation/metrics.py`): PCC, RMSE,
  AUC (0 vs. non-zero).
- Downstream: ARI/NMI — doubles as the cell-type-plausibility check above.
- Spatial coherence: Moran's I (free via `squidpy.gr.spatial_autocorr`).
- **New distributional metric (ST-FID) — side goal, not primary**, time
  permitting. Plan and validation criteria in `docs/metrics_notes.md`.

## Datasets (from `docs/dataset_notes.md`)

- **Track B primary**: whole mouse brain spatial atlas (Stereo-seq, 123
  sections, documented spacing). Secondaries: MOSTA, STARmap PLUS
  (genuinely volumetric — best for rigorous validation).
- **Track A primary**: HEST-1k (largest, most tissue-diverse). Secondaries:
  MOSTA, 10x Xenium public datasets.
- Still open: actually pulling and loading one of these through
  `src/data/loaders.py` — not yet done (Phase 1 in `docs/project_outline.md`
  is dataset *selection*, not yet data *acquisition*).

## What's built vs. not (see `docs/architecture_plan.md` "Build vs. reuse"
for the full table)

Built: data loading skeleton, masking simulators (both tasks), pointwise +
FID/MMD metric skeleton, `BaseGenerativeModel` interface, VAE, WAE-GAN,
Lightning-based training loop.

**Not built — the real remaining work:**
1. **Conditioning encoder** — the one box in the workflow schematic that
   doesn't exist yet. Every current model is an unconditioned placeholder.
2. Diffusion/Flow Matching backbone.
3. Dataset-specific data-loading adapter for whichever pilot dataset gets
   pulled first.
4. The independent cell-type-plausibility classifier (evaluation-side).
5. ST-FID metric implementation + validation (side goal).

## Workflow schematic

See `docs/architecture_plan.md` "Workflow schematic" for the full Mermaid
diagram: data → masking simulator → context/query split → conditioning
encoder (not yet built) → swappable generator (VAE / WAE-GAN / diffusion) →
generated expression → evaluation (pointwise + downstream/plausibility +
spatial + distributional) → comparison against Mimyr/isoST. Held-out ground
truth only ever touches evaluation and comparison, never the generator.

## Next steps
- [ ] Build the conditioning encoder (spatial context representation) —
      biggest remaining architecture gap.
- [ ] Pull and load one real pilot dataset through `src/data/loaders.py`.
- [ ] Implement the diffusion/Flow Matching backbone.
- [ ] Set up the independent cell-type classifier for the plausibility check.
- [ ] Get Mimyr's/isoST's public code running as an external baseline.
