# Project Proposal (draft — revise with supervisor)

## Working title
Generative reconstruction of missing/damaged tissue in spatial transcriptomics:
a model-agnostic architecture for 3D inter-slice and intra-slice completion.

## Motivation
Spatial transcriptomics (ST) profiles gene expression with spatial context,
but two structural gaps limit downstream biology:

1. **Inter-slice (3D) gap:** tissues are physically sectioned into a handful
   of thin 2D slices along the depth (z) axis. The in-plane (x-y) resolution
   is rich, but the z-axis is extremely sparse (often just a few to tens of
   sections spanning mm-cm), and intermediate sections may be lost to
   sectioning damage or diverted to other assays. Reconstructing a continuous
   3D expression volume from sparse slices is an interpolation/extrapolation
   problem in an irregular, non-Euclidean feature space (genes are not
   pixels).
2. **Intra-slice damage:** individual sections can be torn, folded, or have
   regions excluded by QC, leaving holes in an otherwise 2D dataset — closer
   to a classic inpainting problem, but on a discrete point cloud / graph of
   cells or spots with a very high-dimensional feature vector (gene
   expression) at each node, rather than RGB pixels.

Both are, at their core, **conditional generative modeling problems over a
spatially-indexed, high-dimensional, sparse signal.** That framing is what
should let one underlying architecture serve both problems.

## Concrete aims (fill in / prioritize with supervisor)
- [ ] Aim 1: Build a masking/gap simulator that can produce controlled
      "missing intermediate slice" and "missing region within a slice" test
      cases from complete, held-out real data (ground truth needed for
      quantitative evaluation).
- [ ] Aim 2: Build a model-agnostic pipeline: data interface → conditioning
      (neighboring slices / surrounding tissue) → swappable generator →
      decode to (cell/spot location, cell type, gene expression) → metrics.
- [ ] Aim 3: Implement/benchmark 2-3 existing approaches as baselines
      (see literature_review.md — Mimyr, isoST, STDIFFUSION-style, or a
      simpler VAE/interpolation baseline).
- [ ] Aim 4: Propose and validate a distributional ("FID-style") fidelity
      metric for generated ST data (see metrics_notes.md).
- [ ] Aim 5 (stretch): apply the pipeline to a Lundeberg-lab-relevant tissue
      (e.g. developing human heart, breast cancer, or brain data already in
      the lab) and show a biologically meaningful reconstruction.

## Scope decisions to make early (ask supervisor)
- Which failure mode is primary: inter-slice 3D gap-filling, or intra-slice
  damage repair? (They can share an architecture, but pick a primary target
  for the thesis/report narrative and a first dataset.)
- Spot-based (Visium, resolution ~55 µm, mixed-cell spots) vs. single-cell
  resolution (Stereo-seq, Xenium, MERFISH, STARmap)? Single-cell data is
  cleaner for "cell identity + location + expression" generative modeling
  (see Mimyr's decomposition) but spot-based data is what much of the lab's
  legacy/clinical data uses.
- Targeted gene panel (hundreds of genes, imaging-based) vs. whole
  transcriptome (sequencing-based, Visium/Stereo-seq)? Affects which
  generative backbone is natural (categorical/count models vs.
  high-dim continuous latent).
- Mouse vs. human tissue for the primary pilot (mouse brain has by far the
  best public 3D serial-section resources; the lab has extensive human
  heart/cancer data if you want direct lab relevance).

## Success criteria (draft)
- A working, documented pipeline that can swap in ≥3 different generator
  architectures without touching the data/eval code.
- Quantitative benchmark of those generators on ≥1 held-out real dataset
  using standard metrics + the new custom metric.
- A clear characterization of where reconstruction fails (e.g. near sharp
  anatomical boundaries, rare cell types) — this is often more valuable than
  a single leaderboard number.
