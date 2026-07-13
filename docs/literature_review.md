# Literature Review (running doc)

Format per entry: **Title (venue, year)** — one-line summary — why it matters
to us — link.

Last updated: 2026-07-13 (initial scoping pass — verify all claims by reading
the actual papers before citing; summaries below are from abstracts/searches,
not full reads).

---

## Directly on-topic: generative reconstruction of missing ST tissue

- **Mimyr: Generative modeling of missing tissue in spatial transcriptomics**
  (bioRxiv, Nov 2025). Closest existing work to this project. Decomposes
  reconstruction into three coupled generative steps: (1) a plane-conditioned,
  KDE-guided diffusion model that generates cell *locations* for a missing
  region/slice, (2) an MLP that assigns *cell type* identities from those
  locations, (3) a transformer that generates full *gene expression* profiles
  conditioned on geometry + identity. Evaluated on mouse brain, generalizes
  across gene panels and slicing orientations, fine-tuned on Alzheimer's data.
  → Read in full first. Their location→identity→expression decomposition is
  a strong candidate skeleton for our "general architecture."
  https://www.biorxiv.org/content/10.1101/2025.11.24.690239

- **isoST: Three-dimensional spatial transcriptomics at isotropic resolution
  enabled by generative deep learning** (bioRxiv, Aug 2025). Models gene
  expression as a continuous field along tissue depth using stochastic
  differential equations (SDEs), trained on serial sections, to reconstruct
  a continuous isotropic-resolution 3D volume. Directly tackles the
  inter-slice gap problem with an explicit continuity assumption. Evaluated
  on mouse brain, kidney, spleen.
  https://www.biorxiv.org/content/10.1101/2025.08.15.670472

- **STDIFFUSION** — diffusion model extended to spatial transcriptomics for
  in-between slices, but (per Mimyr's related-work description) relies on
  blending heuristics and doesn't handle missing planes/interior gaps
  robustly. Useful as a baseline / cautionary tale about naive blending.

- **C2-STi** — interpolates intermediate histology sections using ST as
  auxiliary input; not a full generative reconstruction (doesn't generate new
  cells/transcriptomes). Baseline candidate.

- **SpatialZ** — 3D reconstruction from planar ST slices via a lookup-based
  procedure rather than a generative model. Useful non-DL baseline.

- **X-Pression: Deep learning-based 3D spatial transcriptomics** (bioRxiv,
  Mar 2025). Aims to reduce the need for many physical sections. Check
  whether it predicts unmeasured depth directly vs. needs thick-section
  imaging input.
  https://www.biorxiv.org/content/10.1101/2025.03.21.644627

## 3D reconstruction / alignment across slices (not generative per se, but
foundational — needed for building any 3D dataset from serial slices)

- **PASTE** — pairwise/center-slice alignment via optimal transport; assumes
  full 2D overlap between adjacent slices; doesn't use real z-distance.
  Classic baseline for slice registration.
- **STAligner** — landmark/domain-based alignment across slices; also doesn't
  build genuine 3D structure per spot.
- **STitch3D** (Nature Machine Intelligence) — deep-learning 3D reconstruction
  integrating multiple 2D slices with paired scRNA-seq.
  https://github.com/YangLabHKUST/STitch3D
- **Spa3D** — GCN-based 3D reconstruction that explicitly incorporates real
  z-axis distances (addresses a specific PASTE limitation).
- **SPACEL** — deep-learning suite including 3D architecture reconstruction
  from multi-slice ST.
- **SpaBatch** (Advanced Science, 2025) — batch-effect-corrected multi-slice
  integration + 3D spatial domain identification across 8 real datasets;
  code at https://github.com/wenwenmin/SpaBatch — good source of already
  batch-corrected multi-slice datasets to reuse.

## Gene-expression imputation (different problem: missing *genes*, not
missing *tissue/space* — but architecturally related, good source of
generative building blocks)

- **gimVI** (2019) — deep generative (VAE-style) model imputing missing genes
  in ST using unpaired scRNA-seq.
- **stDiff** — conditional diffusion model imputing spatial gene expression
  guided by scRNA-seq.
- **DiffusionST** — diffusion-based ST data quality enhancement / imputation,
  benchmarked on 12 datasets, also used for spatial domain ID.
- **SpaLSTF** — diffusion + BiLSTM + transformer hybrid for ST imputation.
- **LGDiST** — reference-free latent diffusion model for completing missing
  gene *values* (not missing regions) in ST.
- **SpaVGN** — CNN+ViT+GNN hybrid for high-fidelity imputation and spatial
  domain ID; reports Pearson correlation as main metric.

## Metrics / evaluation

- No existing "FID-for-spatial-transcriptomics" found in this initial pass —
  this looks like a genuine gap and supports the idea of developing one.
  Adjacent art:
  - **FID** itself (Heusel et al. 2017): Fréchet distance between Gaussian
    fits to Inception-net features of real vs. generated images. The general
    recipe (embed → fit distribution → compare) is reusable for ST if we
    replace the image embedding network with an ST-appropriate one (e.g. a
    pretrained scRNA-seq/ST foundation model embedding, or a simple PCA/
    scVI latent space).
  - **STEAM** — spatial transcriptomics evaluation metric, but for
    *clustering* performance, not generative fidelity. Relevant only as a
    sibling metric, not a template to copy directly.
  - **Coverage Index (CI)** — gene-panel representation metric; not directly
    relevant but shows the field does invent new bespoke ST metrics when
    existing ones don't fit.
  - Standard benchmark metrics used for *gene expression prediction/
    imputation* (from HEXST, translational-potential benchmark papers):
    Pearson correlation coefficient (PCC) per gene/per spot, mutual
    information, AUC for zero vs. non-zero expression (dropout-aware),
    Silhouette score / Davies-Bouldin for downstream cluster preservation.
    These matter but are pointwise/marginal — none of them capture "does the
    *joint distribution* of generated tissue look realistic," which is
    exactly the gap FID fills for images. That's the opening for a custom
    metric (see docs/metrics_notes.md).
  - **sCCIgen**, **SpatialSimBench/SimAdaptor** — simulators and benchmarking
    frameworks for *simulated* ST data; good source of ideas for what
    properties (cell-cell interaction patterns, gene-gene correlation
    structure) a good fidelity metric should be sensitive to.

## Lab-relevant context (Lundeberg lab, for framing significance / choosing
a human-relevant dataset later)

- Lázár & Lundeberg, "Spatial architecture of development and disease,"
  *Nat Rev Genetics* (Sep 2025) — likely a good general framing reference,
  read this first for how the lab talks about 3D/spatial tissue architecture.
- Lázár, Mauron, Andrusivova et al., "Spatiotemporal gene expression and
  cellular dynamics of the developing human heart" (2025) — combines spatial
  + single-cell across developmental weeks 5.5–14; possible candidate 3D/
  temporal dataset if sectioning geometry is documented.
- Li et al., "Computational pathology annotation enhances the resolution and
  interpretation of breast cancer spatial transcriptomics data," *Nat
  Precision Oncology* (2025).
- "Spatial multimodal analysis of transcriptomes and metabolomes in
  tissues," *Nat Biotechnol* (2024) — Visium-compatible multimodal method
  from the lab; worth knowing about even if not used directly.

**Action:** ask your supervisor directly which internal/lab datasets already
have multi-slice or damaged-tissue data — this is likely far more relevant
than anything found by public search, and may determine the whole project
scope.

---

## Reading queue (priority order)
1. Mimyr (full paper + supplement) — closest match, read first.
2. isoST (full paper) — alternative framing (continuous field via SDE) worth
   contrasting with Mimyr's discrete decomposition.
3. Lázár & Lundeberg Nat Rev Genetics 2025 review — for framing/context.
4. STitch3D + PASTE — for understanding slice alignment (a likely
   prerequisite step regardless of which generative approach is chosen).
5. gimVI / stDiff — for generative-modeling building blocks even though they
   solve a different sub-problem.
