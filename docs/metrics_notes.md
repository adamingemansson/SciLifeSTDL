# Evaluation Metrics — Notes & Plan

Last verified: 2026-07-13 — see docs/literature_review.md "Metrics /
evaluation" section for full citations. One correction to the framing below:
a Fréchet-distance metric for *non-spatial* single-cell generative models
already exists (**scFID**, bioRxiv 2025.04.14.648850) — there is no
published spatial-specific analog yet, but "no FID-style metric exists
anywhere in transcriptomics" would be an overclaim. Position "ST-FID" as
extending scFID's recipe to spatial data (patch/neighborhood embeddings that
capture spatial arrangement, not just per-cell expression), and cite scFID
explicitly wherever this metric is written up.

## 1. Standard / pointwise metrics (use as a baseline suite regardless of
   what else you do)

- **Pearson correlation coefficient (PCC)** per gene and/or per spot/cell,
  between generated and held-out ground truth. The dominant metric in the ST
  imputation literature (gimVI, stDiff, SpaVGN, HEXST benchmarks, etc.) —
  report it for comparability with prior work, but be aware it only measures
  linear, marginal (gene-by-gene) agreement.
- **Mutual information (MI)** — captures nonlinear dependency, used in HEXST
  as a complement to PCC.
- **AUC (0 vs. non-zero, and above/below median)** — ST data is sparse/
  dropout-heavy; these check whether the model gets the *presence/absence*
  pattern right, not just magnitude.
- **RMSE / MAE** on expression values (log-normalized) — simple, standard.
- **Downstream-task preservation**: cluster the generated data (Leiden/
  Louvain) and compare to ground-truth clustering (ARI, NMI), or compute
  Silhouette / Davies-Bouldin on generated spatial domains (used in SpaVGN).
  This checks whether generated tissue is *usable*, not just numerically
  close.
- **Spatial coherence metrics**: e.g. Moran's I or Geary's C on generated
  expression fields, to check that generated tissue doesn't look like spatial
  noise (smoothness/structure should match real tissue's spatial
  autocorrelation, not be arbitrarily higher or lower).

**Limitation to state explicitly in the report:** none of the above evaluate
the *joint, distributional* realism of a whole reconstructed region the way
FID evaluates a whole generated image — they're all pointwise/marginal or
downstream-task proxies. That's the motivation for a custom metric.

## 2. Plan for a custom "FID-style" metric for ST

Classic FID recipe: embed real and generated images with a pretrained
Inception network, fit a Gaussian to each embedding set, report the Fréchet
(Wasserstein-2) distance between the two Gaussians. Three ingredients to
port:

1. **An embedding function** for a "unit" of ST data (a cell, a spot, or a
   local neighborhood/patch of tissue) that captures biologically meaningful
   structure, analogous to Inception features for images.
   Candidate choices, roughly in order of effort:
   - PCA / scVI latent space fit on real reference data (fast, simple,
     defensible starting point).
   - A pretrained single-cell/ST foundation model embedding if one is
     available and license-appropriate (e.g. scGPT, Geneformer-style models,
     or an ST-specific self-supervised encoder) — closer in spirit to
     Inception (trained on a large, diverse corpus) but adds a dependency to
     justify/validate.
   - A small encoder you train yourself (e.g. autoencoder or contrastive
     model) on the reference dataset(s) — most controllable, but you own the
     burden of showing the embedding is meaningful (sanity check: does it
     separate known cell types / spatial domains?).
2. **Unit of comparison**: decide if "samples" are individual cells,
   fixed-size spatial patches (like image patches for FID), or whole
   reconstructed regions/slices. Patches probably map best to the original
   FID logic and let you compute a distribution over many patches even from
   a single reconstructed slice.
3. **Distance**: start with the standard Fréchet distance assuming Gaussian
   embeddings (fast, matches literature expectations). Consider also
   reporting **Maximum Mean Discrepancy (MMD)** as a comparison, since MMD
   drops the Gaussian assumption and is often used in the "FID for
   non-image domains" literature — flag this as a design choice to validate
   empirically rather than assume.

**Validation plan before trusting the new metric** (important — a metric
paper reviewer / your supervisor will ask this): show that the metric
(a) decreases monotonically as you interpolate from pure noise → blurred
approximation → ground truth (sanity gradient test), (b) correlates with
existing pointwise metrics on cases where those already agree it's clearly
good/bad, and (c) is *sensitive to failure modes that pointwise metrics
miss* (e.g. correct marginal gene distributions but wrong spatial
arrangement) — this last point is the actual selling point of a new metric,
so design at least one experiment specifically to demonstrate it.

## 3. Suggested reporting table for any experiment
For each model × dataset × task (inter-slice vs. intra-slice):
PCC | MI | AUC(0 vs nz) | ARI/NMI (cluster preservation) | Moran's I gap |
new distributional metric | qualitative figure (side-by-side generated vs.
real spatial plot).

## 4. Summary: all candidate metrics considered, pros/cons

Consolidates the metrics/FID discussion (2026-07-13) plus what the 4
reference papers actually used (`Flow Matching for Generative Modeling`
Lipman et al. 2022; `Super-resolved spatial transcriptomics by deep data
fusion` Bergenstråhle et al., *Nat Biotech*, Lundeberg lab; `Spatial landmark
detection and tissue registration` Ekvall et al., *Nat Methods*, Lundeberg
lab; `Wasserstein Auto-Encoders` Tolstikhin et al. 2017).

### Distributional / generative-fidelity metrics

- **FID** (Fréchet distance between Gaussian-fit embeddings)
  - \+ closed-form, cheap, well-understood, widely reported (used by both
    the Flow Matching paper and WAE for exactly this purpose)
  - \+ sensitive to both sample quality and diversity (catches mode collapse)
  - − assumes embeddings are Gaussian — bad fit for tissue, which is a
    mixture of distinct cell types/domains, not one blob
  - − needs large N to estimate a high-dim covariance reliably; a single
    held-out region rarely has enough cells/spots
  - − no off-the-shelf "Inception" embedding exists for ST; result quality
    is dominated by whatever embedding you choose
- **CMMD** (MMD with RBF kernel on CLIP embeddings)
  - \+ no Gaussian assumption — handles multimodal embeddings correctly
  - \+ unbiased at any sample size — much better for small eval sets
  - − still fully dependent on the embedding; no ST equivalent of CLIP exists
  - − adds a kernel-bandwidth hyperparameter to tune
- **scFID** (Fréchet distance in a single-cell LLM foundation-model
  embedding space — C2S-Scale/Cell2Sentence)
  - \+ rich pretrained embedding from a large biological corpus — more
    context per cell than PCA
  - \+ closest existing precedent to "FID for transcriptomics"
  - − embeds individual cells only — **no spatial notion at all**; a model
    could get per-cell composition right and completely botch tissue
    architecture and scFID wouldn't notice. This is the core gap our
    project's metric needs to close.
  - − inherits FID's Gaussian-assumption and large-N weaknesses
  - − very new (2025), heavy LLM dependency, not independently validated yet
- **MMD** (generic, no fixed embedding — already stubbed as `st_mmd` in
  `src/evaluation/metrics.py`)
  - Same pros/cons as CMMD minus the CLIP-specific embedding choice — this
    is the general pattern CMMD is a specific instance of.
- **Precision & Recall for generative models** (Sajjadi 2018,
  Kynkäänniemi 2019) — noted but not deep-dived
  - \+ separates "are samples realistic" from "is the real diversity
    covered" — more diagnostic than one scalar
  - − another embedding-dependent method; adds complexity for a
    two-number instead of one-number report

### Pointwise / established metrics (already in §1, recapped)
- **PCC** — dominant in ST imputation literature (gimVI, stDiff, SpaVGN);
  linear + marginal only.
- **RMSE/MAE, MI, AUC (0 vs. non-zero)** — standard, cheap, marginal.
- **R² (coefficient of determination)** — used by the Bergenstråhle
  super-resolution paper alongside PCC.
- **ARI/NMI, Silhouette/Davies-Bouldin** — downstream-task/cluster
  preservation, checks usability not just closeness.
- **Moran's I / Geary's C** — spatial autocorrelation sanity check.
- **PSNR / SSIM / Inception Score** — from the Flow Matching paper's
  super-resolution experiments; image-domain metrics, only useful here if
  we ever render expression as an image-like grid.
- **NLL / bits-per-dimension** — likelihood-based, only applicable to
  models that expose a tractable likelihood (not diffusion samplers in
  general, not GANs).
- **ATRE / landmark forward-backward-consistency error** — from the Ekvall
  registration paper; relevant only if slice alignment becomes a
  prerequisite step for Track B, not for generation fidelity itself.
- **Hypergeometric enrichment test** — used by Bergenstråhle et al. to check
  generated marker genes overlap known cell-type markers; a cheap
  biological-plausibility sanity check worth borrowing.
- **Sharpness (Laplace-filter edge variance)** — WAE's blur heuristic;
  image-specific, not directly portable, but the underlying idea
  ("does the model default to a bland average output") is worth a
  ST-native analog.

### Evaluation-protocol conclusions (how to actually run FID/MMD here)
- Compare generated region against the **actual held-out ground truth**,
  not the rest of the visible tissue — comparing to the rest of the tissue
  only checks "looks like plausible tissue in general" and would pass a
  bland, locally-wrong reconstruction.
- A single hole/held-out slice is too few points for a stable distributional
  estimate — aggregate generated vs. ground-truth points across the **whole
  test set** (many holes, many held-out slices) before computing one
  FID/MMD number.
- Bergenstråhle et al.'s occlusion-robustness protocol is directly reusable
  as an eval design template for Track A.

### What a "perfect" custom ST fidelity metric needs
1. **Spatially-aware embedding** — a local neighborhood/patch embedding
   (e.g. GNN or transformer over a k-NN spatial graph, self-supervised on
   real tissue), not a per-cell embedding. This is the one thing FID, CMMD,
   and scFID all lack, and it's the actual point of building our own metric
   rather than reusing scFID.
2. **MMD-style distance**, not Fréchet/Gaussian — cell-type/domain mixtures
   are genuinely multimodal.
3. **Count-aware embedding space** — expression is sparse/zero-inflated;
   an embedding trained with a proper count likelihood (e.g. scVI-style)
   is a better starting geometry than raw PCA on log-normalized data.
4. **Sample-efficient** — must work on small held-out regions (hundreds of
   cells), not require thousands of samples.
5. **Validated to catch spatial-only corruption** — the concrete pass/fail
   test: keep every gene's marginal distribution correct, scramble spatial
   positions, metric must go up. If it doesn't, it's not measuring the
   thing this project cares about.
6. **Evaluated against real held-out ground truth, aggregated across the
   full test set** — per the protocol conclusions above.
