# Project Outline

Last updated: 2026-07-13. This is the working roadmap — phase-based rather
than calendar-dated, since internship length/pace isn't pinned down yet.
Recalibrate the phase lengths once you know your actual timeline. See
`docs/project_proposal.md` for the motivation/aims write-up this roadmap
executes against.

## The two directions, one architecture

Both are conditional generative modeling over the same underlying object: a
spatially-indexed point cloud (cells or spots) with a high-dimensional
feature vector (gene expression) at each point.

| | **Track A — Intra-slice inpainting** | **Track B — Inter-slice 3D generation** |
|---|---|---|
| Problem | A region within one real 2D slice is torn/folded/missing. Fill it in. | Physical sections only exist every N µm–mm apart. Generate the tissue *between* them to approximate a continuous 3D volume. |
| Closest to | Classic image inpainting, but on an irregular point cloud with gene-expression features instead of RGB. | 3D interpolation/extrapolation with real physical z-spacing; more like implicit-field or SDE-based generation than pixel interpolation. |
| Ground truth for eval | Hide a real region from a complete slice, reconstruct, compare to what was hidden. | Hold out a real intermediate slice from a serial-section series, reconstruct, compare to the held-out slice. |
| Closest prior art | C2-STi (histology-conditioned interpolation), general inpainting literature | Mimyr, isoST, SpatialZ, STitch3D/Spa3D/SPACEL (alignment, not generation) |
| Data need | One damaged slice + surrounding context is enough in principle | Requires a *serial section series* — multiple slices with known/documented z-spacing |

`src/data/masking.py` already implements the shared machinery: `mask_region_2d`
/ `random_dropout_patches` for Track A, `hold_out_slice` for Track B. The
`BaseGenerator` interface in `src/models/registry.py` is written so the same
model class can, in principle, serve either task — a concrete architecture
decision for Phase 4, not before.

**You don't have to pick one track exclusively** — the proposal's Aim 5
suggests a primary target with the other as secondary/stretch. But the
dataset requirements differ enough (Track B strictly needs multi-slice serial
series; Track A just needs one good slice) that dataset choice and track
choice are coupled — which is why dataset discovery comes first.

---

## Phase 1 — Dataset discovery & scoping (start here)

**Goal:** land on 1–2 pilot datasets with confirmed access, before investing
in architecture work that might assume the wrong data shape.

**Why first:** track choice, data representation (spot vs. single-cell),
and even which generative backbone is natural (categorical/count vs.
continuous latent) all flow from what data you actually have. Committing to
architecture before this is scoped risks rework.

Checklist per candidate dataset (already the template in
`docs/dataset_notes.md`):
- [ ] Number of serial sections + real z-spacing (Track B needs this to be
      meaningful — a dataset with 2 sections and no documented spacing is
      unusable for it)
- [ ] Single-cell / segmented resolution vs. spot-level (mixed-cell)
- [ ] Gene panel size (targeted vs. whole transcriptome)
- [ ] License / data use agreement, especially anything human or internal
- [ ] Whether prior work already benchmarks this dataset (gives you numbers
      to match/beat)
- [ ] For Track A specifically: does it contain *real* damage (torn/folded
      QC-excluded regions), or would damage need to be fully synthetic?

Concrete actions:
1. **Public datasets** — `docs/dataset_notes.md` already has a
   fact-checked shortlist (MOSTA, whole mouse brain atlas, STARmap PLUS,
   DLPFC, etc.) with verified section counts/resolution. Next step: pick the
   1–2 strongest candidates and actually download/open a sample to confirm
   the AnnData structure loads cleanly with `src/data/loaders.py`.
2. **Lab-internal datasets** — ask your supervisor directly (see the
   question list below). Internal data with real damage or documented
   serial-section geometry would beat any public dataset for both relevance
   and novelty, and this is the single highest-leverage conversation to have
   early.
3. **Decide primary track** (or confirm "both, with a stated primary") once
   you know what data is realistically available — don't decide this in the
   abstract.

**Deliverable:** `docs/dataset_notes.md` updated with a chosen pilot
dataset (or two — one per track if pursuing both), access confirmed, and a
one-paragraph justification for the choice.

**Blocking questions for your supervisor** (carried over from
`notes/2026-07-13.md` — resolve alongside dataset choice, not after):
1. Primary track: intra-slice repair, inter-slice 3D, or both with one as
   primary framing for the report/thesis?
2. Any internal datasets with real damaged sections, or multi-slice series
   with documented z-spacing?
3. Compute allocation (Berzelius/UPPMAX/Rackham)?
4. Spot-based (Visium) vs. single-cell (Stereo-seq/Xenium) — affects the
   whole architecture.
5. Experiment tracking: wandb (cloud) vs. local MLflow — depends on data
   sensitivity.

---

## Phase 2 — Literature deep dive & baseline shortlist (overlaps Phase 1)

- Read Mimyr and isoST in full (not just abstracts — the fact-check pass
  already caught abstract-level summaries that didn't fully match the
  papers). Take structured notes: what exactly conditions the generator,
  what's the loss, how do they evaluate.
- Pick 2–3 baselines to actually reproduce/benchmark — realistically:
  the parameter-free `interp_baseline` (already implemented, a sanity
  floor), one learned baseline per track, and if time allows, a re-run of a
  published method's reported numbers on your pilot dataset.
- **Deliverable:** `docs/literature_review.md` updated with full-paper notes
  (not abstract-level) for the 2–3 papers closest to your chosen track.

## Phase 3 — Metrics (overlaps Phases 1–2, needed before any benchmarking)

- Standard pointwise suite is already stubbed in `src/evaluation/metrics.py`
  (PCC, RMSE, AUC). Add the downstream-task and spatial-coherence metrics
  listed in `docs/metrics_notes.md` (ARI/NMI, Moran's I) once a pilot
  dataset exists to test them on.
- Prototype the "ST-FID" distributional metric (skeleton already in
  `st_fid`/`st_mmd`). Run the validation plan already written in
  `docs/metrics_notes.md` §2 (monotonicity sanity check, correlation with
  pointwise metrics, sensitivity to spatial-arrangement corruption that
  pointwise metrics miss) — this validation is what makes the metric
  trustworthy enough to report, not just implementing the formula.
- **Deliverable:** a short internal note showing the metric passes its own
  validation plan, or documenting where it doesn't (also a valid, useful
  result).

## Phase 4 — Architecture design

- Finalize the model-agnostic pipeline: data → masking simulator →
  `BaseGenerator` → decode → metrics (interfaces already stubbed in
  `src/models/registry.py`, `src/training/train.py`).
- Key design decisions to make explicitly, once real data is in hand:
  data representation (point cloud/graph vs. voxel grid vs. fixed patches),
  conditioning strategy (how neighboring slices / surrounding tissue
  actually enter the model), and whether cell identity is generated
  explicitly (Mimyr-style decomposition) or implicitly.
- **Deliverable:** a working end-to-end run of `interp_baseline` on the real
  pilot dataset through the full pipeline (proves the plumbing works before
  any real model is trained).

## Phase 5 — Baseline implementation & benchmarking

- Implement the 2–3 chosen baselines (Phase 2) against the real pipeline
  (Phase 4) on the pilot dataset(s).
- Run the full metric suite (Phase 3) on each.
- **Deliverable:** a benchmark table — model × dataset × task × metric
  suite — establishing the numbers everything else is compared against.

## Phase 6 — Model development / iteration (main body of the internship)

- Swap in and compare generative backbones (diffusion / VAE / flow /
  GNN-based) via the registry — this is the point of building it
  model-agnostic in Phase 4.
- Ablate design decisions from Phase 4 against the benchmark from Phase 5.
- **Deliverable:** best-performing model(s) per track, with ablations.

## Phase 7 — Biological validation & write-up

- If scope/time allows, apply the pipeline to a Lundeberg-lab-relevant
  tissue and check the reconstruction makes biological sense, not just a
  good metric score (Aim 5, stretch goal).
- Characterize *where* reconstruction fails (sharp anatomical boundaries,
  rare cell types) — per the proposal's success criteria, this is often
  more valuable than a single leaderboard number.
- **Deliverable:** final report/thesis writeup.

---

## Immediate next action

Everything above is sequenced, but the only thing blocking real progress
right now is **Phase 1**: pick a pilot dataset. Two parallel paths:
1. I can go deeper on the public dataset candidates already shortlisted —
   actually pull sample data and confirm it loads, rather than just reading
   descriptions.
2. You raise the 5 blocking questions with your supervisor — this could
   change the answer entirely if lab-internal data is available, so it's
   worth doing in parallel rather than after.

Tell me which to prioritize, or if you'd like both running at once.
