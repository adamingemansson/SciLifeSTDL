# Literature Review (running doc)

Format per entry: **Title (venue, year)** — one-line summary — why it matters
to us — link.

Last updated: 2026-07-13 — every citation below has been checked against a
live web search (title, venue, date, URL, and method description verified or
corrected against the actual paper/preprint/repo). None were outright
fabricated, but several original summaries had errors (wrong tissues, a
reversed method description, an unsupported venue) — those are fixed below
and flagged with **[corrected]**. Full papers still need to be read before
citing in any report; this is a scoping-level pass.

---

## Directly on-topic: generative reconstruction of missing ST tissue

- **Mimyr: Generative modeling of missing tissue in spatial transcriptomics**
  (bioRxiv, Nov 2025; Deshpande, Bei, Ma, Krieger — Jian Ma lab, CMU).
  Closest existing work to this project. Decomposes reconstruction into three
  coupled generative steps: (1) a DDPM-style diffusion model that generates
  cell *locations* for a missing region/slice, with an optional KDE-based
  biological prior, (2) a neural classifier that assigns *cell type*
  identities from those locations, (3) a transformer that generates full
  *gene expression* profiles conditioned on geometry + identity. Evaluated on
  mouse brain, generalizes across gene panels and slicing orientations,
  fine-tuned on Alzheimer's data. → Read in full first. Their
  location→identity→expression decomposition is a strong candidate skeleton
  for our "general architecture."
  https://www.biorxiv.org/content/10.1101/2025.11.24.690239
  **[design critique, 2026-07-13]** The location model is "plane-conditioned"
  and uses backward-guidance from neighboring slices *when available*
  (falls back to a learned density prior + metadata otherwise) — so it's not
  strictly a hard requirement to have real local context, which matters for
  distinguishing genuine interpolation from prior-only generation when
  reading their results. More importantly: the cell-type classifier stage is
  a real generalizability bottleneck — it requires a target dataset with a
  cell-type taxonomy compatible with (or retrained against) their reference
  annotations, and any misclassification propagates as a wrong conditioning
  signal into the expression transformer. This is our chosen point of
  departure (see `docs/architecture_plan.md` "Design decision: no explicit
  cell-type conditioning") — skip the discrete cell-type bottleneck and
  generate location→expression directly, validated by checking whether an
  independent classifier still recovers sensible types from the generated
  expression post-hoc, rather than baking type into the generation path.

- **MORPHE: Bridging Image Generation and Spatial Omics for Tissue
  Synthesis** (bioRxiv, Mar 2026; Feng, Robers, Rasheed, Miao, Wen, Lee,
  Sohigian, Brbić — Duke BME + EPFL). Converts discrete cell identity +
  spatial-relationship graphs into a continuous "RGB-like" latent embedding,
  letting it reuse large pretrained image-generation diffusion models, then
  refines with a cascaded diffusion architecture to single-cell pixel
  precision. Does outpainting (beyond imaged field of view), inpainting
  (damaged/missing tissue — Track A), and cross-tissue imputation connecting
  separated regions in both 2D and 3D (Track B-adjacent). Tested on CODEX
  intestine (proteomics) and MERFISH mouse brain (transcriptomics), millions
  of cells. **Important limitation:** generates cell type/identity +
  spatial architecture only, **not full gene expression profiles** — the
  expression vector is an input feature, not a generative output. Solves a
  related but genuinely different problem than ours (architecture/cell-type
  synthesis vs. expression synthesis) — not a direct competitor on the exact
  task, but the RGB-embedding trick (piggyback spatial-omics generation on
  pretrained image diffusion models) is a potentially reusable idea for our
  diffusion backbone. Code: https://github.com/HickeyLab/MORPHE

- **HoloTea** ("3D-Guided Scalable Flow Matching for Generating Volumetric
  Tissue Spatial Transcriptomics from Serial Histology," arXiv 2511.14613,
  Nov 2025; Sanian, Hemmat, Vahidi, Maaskola, Lee, Makarchuk, Demirci,
  Chipampe, Haniffa, Bayraktar, Paavolainen, Lotfollahi — Sanger Institute /
  Lotfollahi lab; note Jonas Maaskola also co-authored the Bergenstråhle
  super-resolution ST paper, a Lundeberg-lab-adjacent connection). Uses
  **Flow Matching** — the same generative family chosen for our own
  diffusion-backbone entry (`docs/architecture_plan.md`). **Important
  caveat, changes how directly this applies**: HoloTea imputes expression
  **from paired H&E histology**, not from surrounding real ST expression
  context — every location needs a histology image as its primary
  conditioning input. That's a different task shape than Track A/B as
  currently scoped (no paired-histology requirement). The reusable part is
  the *mechanism*, not the task: a lightweight ControlNet retrieves
  morphologically-corresponding spots on neighboring slides in a shared
  feature space and fuses that cross-section context in, so 3D consistency
  comes from retrieval + conditioning rather than an explicit continuity
  assumption (contrast with isoST's SDE-continuity approach). Worth
  revisiting if the histology stretch goal (`docs/architecture_plan.md`
  "Known gaps") gets activated, or as a conditioning-mechanism reference
  even without histology.

- **DRIFT** ("Diffusion-based Representation Integration for Foundation
  Models Improves Spatial Transcriptomics Analysis," bioRxiv, Nov 2025;
  code: https://github.com/rsinghlab/DRIFT). **Not a generator** — a
  representation-learning technique, noted here because it's directly
  relevant to our conditioning encoder, the one piece of the architecture
  not yet built (`docs/architecture_plan.md` "Known gaps"). Builds a
  spatial adjacency graph over cells/spots and applies a heat-kernel
  diffusion process that propagates expression signal across local
  neighborhoods while preserving tissue boundaries, producing a spatially-
  coherent representation that can feed into any pretrained foundation
  model without retraining it. Validated on cell-type annotation,
  cross-section alignment, and clustering. A plausible concrete design for
  the conditioning encoder's spatial-graph step — worth reading before
  designing that layer from scratch.
  (code: search GitHub for `gkrieg/mimyr`)

- **isoST: Three-dimensional spatial transcriptomics at isotropic resolution
  enabled by generative deep learning** (bioRxiv, Aug 2025). Models gene
  expression as a continuous field along tissue depth using stochastic
  differential equations (SDEs) with GNNs predicting spatial/expression
  gradients, trained on serial sections, to reconstruct a continuous
  isotropic-resolution 3D volume. Directly tackles the inter-slice gap
  problem with an explicit continuity assumption. **[corrected]** Evaluated
  on mouse brain, mouse embryo, kidney, and spinal cord — not "spleen" as
  originally noted here.
  https://www.biorxiv.org/content/10.1101/2025.08.15.670472

- **stDiffusion** — **[corrected/downgraded]** No bioRxiv or journal paper
  exists under the name "STDIFFUSION." The real, closest match is
  *"stDiffusion: A Diffusion Based Model for Generative Spatial
  Transcriptomics,"* an **OpenReview workshop submission**, not a
  peer-reviewed paper or preprint — treat any claims from it as much weaker
  evidence than the other entries here. It does use a DDPM to interpolate
  unseen slices, roughly matching the original "blending heuristics" framing,
  but that framing itself is unverified (we could not confirm the
  Mimyr-related-work characterization). Note there are two unrelated real
  papers with confusingly similar names, don't conflate them:
  - **stDiff** (*Briefings in Bioinformatics* 25(3):bbae171, 2024) — a real,
    peer-reviewed conditional diffusion model, but it solves the *gene
    imputation* problem (see below), not slice interpolation.
  - **SpatialDiffusion** (bioRxiv 2024.05.21.595094) — a different, separate
    diffusion-based ST paper, not checked in detail yet.

- **C2-STi: Adaptive Spatial Transcriptomics Interpolation via Cross-modal
  Cross-slice Modeling** (arXiv 2505.10729, MICCAI 2025). **[corrected —
  roles were reversed]** It interpolates **missing ST slices**, using paired
  **H&E histology as the auxiliary/cross-modal conditioning input** (the
  original note here had this backwards). Not a full generative
  reconstruction from scratch — still a useful baseline, especially if
  paired histology is available in lab data.

- **SpatialZ: Bridging the dimensional gap from planar spatial
  transcriptomics to 3D cell atlases** (Nature Methods, 2025; preprint
  bioRxiv 2024.12.06.627127). **[corrected]** Not a "lookup-based procedure"
  — it's a computational/generative framework that synthesizes intermediate,
  single-cell-resolved slices to build dense 3D atlases. Demonstrated at
  scale on a 38M-cell mouse brain atlas and on imaging-mass-cytometry breast
  cancer data. Reclassify as a genuine (non-lookup) baseline, potentially a
  strong one given the demonstrated scale.

- **X-Pression** (bioRxiv, Mar 2025).
  https://www.biorxiv.org/content/10.1101/2025.03.21.644627
  **[corrected — method is different from originally noted]** This is a CNN
  that reconstructs 3D expression signatures from **micro-CT volumes paired
  with a single 2D Visium section** — i.e. it needs an auxiliary imaging
  modality (micro-CT), not multiple ST slices. Demonstrated on a SARS-CoV-2
  vaccine efficacy study. Less directly applicable to our multi-slice
  inter-slice task than originally assumed; useful mainly as an example of
  using an auxiliary 3D imaging modality to constrain reconstruction.

## 3D reconstruction / alignment across slices (not generative per se, but
foundational — needed for building any 3D dataset from serial slices)

- **PASTE** (Zeira, Land, Strzalkowski, Raphael; *Nature Methods* 19:567–575,
  2022; preprint bioRxiv 2021.03.16.435604) — pairwise/center-slice alignment
  via Fused Gromov-Wasserstein optimal transport; assumes full 2D overlap
  between adjacent slices; doesn't use real z-distance. Classic baseline for
  slice registration.
- **STAligner** (Zhou, Dong, Zhang; *Nature Computational Science* 3:894–906,
  Oct 2023). **[corrected]** Uses a graph-attention autoencoder for
  batch-corrected integration and spatial domain identification across
  conditions/technologies/stages — **not** landmark-based, as originally
  noted here. (Don't confuse with **STalign**, *Nature Communications* 2023,
  which genuinely is landmark/diffeomorphic-registration-based — a different
  tool with a very similar name.)
- **STitch3D**: "Construction of a 3D whole organism spatial atlas by joint
  modelling of multiple slices with deep neural networks" (*Nature Machine
  Intelligence*, 2023) — deep-learning 3D reconstruction integrating multiple
  2D slices with paired scRNA-seq.
  https://github.com/YangLabHKUST/STitch3D
- **Spa3D**: "3D reconstruction of spatial transcriptomics with spatial
  pattern enhanced graph convolutional neural network" (*Briefings in
  Bioinformatics*, article bbag060) — GCN-based 3D reconstruction that
  explicitly incorporates real z-axis distances (addresses a specific PASTE
  limitation). Code: `Lin-Xu-lab/Spa3D` on GitHub.
- **SPACEL**: "SPACEL: deep learning-based characterization of spatial
  transcriptome architectures" (*Nature Communications* 14, 2023) —
  three-module deep-learning suite: Spoint (deconvolution), Splane (spatial
  domain ID), Scube (3D architecture reconstruction from multi-slice data).
- **SpaBatch**: "SpaBatch: Deep Learning-Based Cross-Slice Integration and 3D
  Spatial Domain Identification in Spatial Transcriptomics" (*Advanced
  Science* 12(44), 2025; Niu, Fang, Chen, Xiong, Liu, Min) —
  batch-effect-corrected multi-slice integration + 3D spatial domain
  identification, evaluated on 8 real datasets (human cortex, mouse brain
  ×2 platforms, mouse embryo, human embryonic heart, HER2+ breast cancer,
  mouse hypothalamus/MERFISH). Code at
  https://github.com/wenwenmin/SpaBatch — good source of already
  batch-corrected multi-slice datasets to reuse.

## Gene-expression imputation (different problem: missing *genes*, not
missing *tissue/space* — but architecturally related, good source of
generative building blocks)

- **gimVI** — "A joint model of unpaired data from scRNA-seq and spatial
  transcriptomics for imputing missing gene expression measurements" (Lopez,
  Nazaret, Langevin, Samaran, Regier, Jordan, Yosef; arXiv:1905.02269, ICML
  2019 Workshop on Computational Biology — a workshop paper, not a journal).
  VAE-style deep generative model (built on scVI) imputing missing genes in
  ST using unpaired scRNA-seq.
- **stDiff** (*Briefings in Bioinformatics* 25(3):bbae171, 2024) —
  conditional diffusion model imputing spatial gene expression guided by
  scRNA-seq, evaluated across 16 datasets.
- **DiffusionST** (bioRxiv 2025.06.12.659243; published *Briefings in
  Bioinformatics* 26(4):bbaf390, 2025) — GCN + ZINB denoising followed by a
  diffusion model for ST data quality enhancement/imputation, benchmarked on
  12 datasets, also used for spatial domain ID.
- **SpaLSTF** (*PLOS Computational Biology*, ~Feb 2026) — full name involves
  a diffusion model + BiLSTM + **XCA-Transformer** (cross-covariance
  attention transformer, not a generic transformer) hybrid for ST
  imputation.
- **LGDiST**: "Latent Gene Diffusion for Spatial Transcriptomics Completion"
  (arXiv:2509.01864, Sep 2025; Cárdenas, Manrique, Vega, Ruiz, Arbeláez,
  BCV-Uniandes). Reference-free latent diffusion model (no scRNA-seq needed)
  operating over a learned gene-neighborhood autoencoder space, for
  completing missing gene *values* (not missing regions) in ST. Code:
  https://github.com/BCV-Uniandes/LGDiST
- **SpaVGN**: "A hybrid deep learning framework for high-resolution spatial
  transcriptomics data reconstruction and spatial domain identification"
  (*PLOS ONE*, 2025) — CNN+ViT+GNN hybrid for imputation and spatial domain
  ID; reports Pearson correlation as main metric (0.609 melanoma, 0.682
  mouse brain).

## Metrics / evaluation

- No published Fréchet/FID-style distributional fidelity metric specific to
  spatial transcriptomics *generative-model* evaluation was found — this
  looks like a genuine gap and supports developing one. **[added]** The
  nearest existing precedent is **scFID** ("single-cell Fréchet Inception
  Distance," bioRxiv 2025.04.14.648850) — a Fréchet-distance metric built for
  evaluating generative models of **single-cell** (non-spatial)
  transcriptomics, using a single-cell foundation-model embedding in place
  of Inception features. Cite this explicitly as the nearest prior art in
  any writeup to avoid overclaiming total novelty — the contribution of a
  "ST-FID" would be adapting the recipe to spatial data specifically
  (patch/neighborhood-level embeddings that capture spatial arrangement, not
  just per-cell expression).
  - **FID** itself (Heusel et al. 2017): Fréchet distance between Gaussian
    fits to Inception-net features of real vs. generated images. The general
    recipe (embed → fit distribution → compare) is reusable for ST if we
    replace the image embedding network with an ST-appropriate one (e.g. a
    pretrained scRNA-seq/ST foundation model embedding, or a simple PCA/
    scVI latent space).
  - **STEAM**: "Spatial Transcriptomics Evaluation Algorithm and Metric for
    clustering performance" (bioRxiv 2025.02.17.636505; *Briefings in
    Bioinformatics* 26(5):bbaf570, 2025) — evaluates clustering/annotation
    reliability in spatial omics via ML classifiers plus Kappa/F1/ARI/NMI/PAS
    metrics. For *clustering* performance, not generative fidelity — relevant
    only as a sibling metric, not a template to copy directly.
  - **Coverage Index (CI)** (bioRxiv 2025.04.07.647642, Apr 2025) —
    quantifies gene-signature representation/read-coverage across
    pre-designed gene panels (10x Xenium 5k, Nanostring GeoMx WTA; CI=0.5 is
    the random-distribution baseline). Not directly relevant but shows the
    field does invent new bespoke ST metrics when existing ones don't fit.
  - Standard benchmark metrics used for *gene expression prediction/
    imputation*: Pearson correlation coefficient (PCC) per gene/per spot,
    mutual information, AUC for zero vs. non-zero expression (dropout-aware),
    Silhouette score / Davies-Bouldin for downstream cluster preservation.
    These matter but are pointwise/marginal — none of them capture "does the
    *joint distribution* of generated tissue look realistic," which is
    exactly the gap FID fills for images, and which scFID fills for
    non-spatial single-cell data. That's the opening for a custom ST metric
    (see docs/metrics_notes.md).
  - **sCCIgen**: "a high-fidelity spatially resolved transcriptomics data
    simulator for cell–cell interaction studies" (bioRxiv 2025.01.07.631830;
    *Genome Biology*, 2025) — simulates SRT data with known cell
    colocalization/CCI patterns.
  - **SpatialSimBench / SimAdaptor** — project name for "Multi-task
    benchmarking of spatially resolved gene expression simulation models"
    (bioRxiv 2024.05.29.596418; *Genome Biology*, 2025; site:
    sydneybiox.github.io/SpatialSimbench_website). SimAdaptor extends
    single-cell simulators to spatial data; benchmark covers 13 simulators,
    10 datasets, 35 metrics. Good source of ideas for what properties
    (cell-cell interaction patterns, gene-gene correlation structure) a good
    fidelity metric should be sensitive to.

## Lab-relevant context (Lundeberg lab, for framing significance / choosing
a human-relevant dataset later)

- Lázár & Lundeberg, "Spatial architecture of development and disease,"
  *Nature Reviews Genetics*, published online Sep 30, 2025, DOI
  10.1038/s41576-025-00892-5 — likely a good general framing reference, read
  this first for how the lab talks about 3D/spatial tissue architecture.
- Lázár, Mauron, Andrusivová et al., "Spatiotemporal gene expression and
  cellular dynamics of the developing human heart," **[corrected — venue
  added]** *Nature Genetics* 57:2756–2771, published online Oct 29, 2025 —
  combines spatial + single-cell across developmental weeks; possible
  candidate 3D/temporal dataset if sectioning geometry is documented. Code:
  https://github.com/rmauron/HDCA_heart_dev
- Li et al., "Computational pathology annotation enhances the resolution and
  interpretation of breast cancer spatial transcriptomics data," **[corrected
  — journal name]** ***npj Precision Oncology*** (not "Nature Precision
  Oncology" — different journal), 9:310, published Sep 9, 2025, DOI
  10.1038/s41698-025-01104-3. Confirmed Lundeberg co-authorship (joint
  Lundeberg/Hartman labs, KI/KTH).
- "Spatial multimodal analysis of transcriptomes and metabolomes in
  tissues," *Nature Biotechnology* 42(7):1046–1050, July 2024 (preprint
  bioRxiv 2023.01.26.525195); first author Vicari M., senior author
  Lundeberg J. — Visium-glass-slide-compatible multimodal (MS imaging +
  transcriptomics + histology) method from the lab; worth knowing about even
  if not used directly.

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
6. SpatialZ (Nature Methods) — re-read given the corrected, stronger method
   description above; may be a more serious baseline than first assumed.
