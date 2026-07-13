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
