# Missing-tissue recovery plan

## What the old numbers mean

The earlier high PCC values are not all fabricated, but they do not estimate
the same problem as the current held-out test. Historical runs mixed several
easier or invalid conditions: masks from a slide also seen in training, target
H&E retained inside the missing region, a prefix-based HEST resolver that could
map `INT1` to `INT10`–`INT19`, and PCC eligibility that dropped a gene when the
model predicted it as constant. Those results remain descriptions of those
specific runs; they are not evidence of generalization to a new patient with
both target H&E and target GEX absent.

## Simplest defensible model

1. A harmonic interpolation from observed context GEX is the explicit anchor.
2. Each observed context spot is encoded from its GEX MLP, optional
   context-only Novae embedding, and optional surrounding GigaPath feature.
3. For each missing query coordinate, only its 32 nearest observed spots are
   pooled. Weights use relative distances normalized by the query's 32nd
   neighbour distance. No absolute coordinate, query H&E, query GEX token,
   transformer, MoME block, or flow model is used.
4. A small deterministic head predicts a standardized correction to the
   harmonic anchor.

This follows the useful common denominator in spatial-imputation work: STPath
uses normalized coordinates and pairwise relative geometry; GraphST and Novae
represent space through neighbour graphs; masked-imputation methods such as
stMCDI and SpaMask condition masked locations on observed spatial neighbours.
The exact local-pooling implementation is intentionally simpler than any of
those models so a failure is interpretable.

Primary references:

- [STPath paper](https://www.nature.com/articles/s41746-025-02020-3)
- [STPath coordinate/data code](https://github.com/Graph-and-Geometric-Learning/STPath/blob/main/stpath/data/dataset.py)
- [STPath spatial transformer](https://github.com/Graph-and-Geometric-Learning/STPath/blob/main/stpath/model/encoder/spatial_transformer.py)
- [GraphST](https://pmc.ncbi.nlm.nih.gov/articles/PMC9977836/)
- [Novae](https://www.nature.com/articles/s41592-025-02899-6)
- [stMCDI](https://arxiv.org/abs/2403.10863)
- [SpaMask](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012881)

## Four-run diagnostic

All four runs use exact HEST filenames, no HVG restriction, identical
INT1–INT6 training / INT7 validation / INT8 final-test splits, one contiguous
hole expressed in spot-spacing units, identical mask banks, and nearest-query
context capping.

| Run | Purpose |
|---|---|
| harmonic k=32 | non-learned floor and promotion anchor |
| local GEX + Novae | whether observed expression plus spatial pretraining can improve the anchor |
| local H&E + GEX + Novae | whether surrounding histology adds information |
| local H&E + GEX, MLP only | whether Novae itself contributes |

The learned runs use 5,000 optimization steps but 256 distinct holes rather
than repeating only 64 holes for 10,000–40,000 steps. Validation and test use
8 and 16 fixed holes respectively. The final test is not loaded until model
selection has completed.

## Decision rule

Do not promote a model because one metric is numerically larger. The full
Novae model must:

1. beat its harmonic anchor on unseen INT7 validation RMSE;
2. beat the pure harmonic run on unseen INT8 in both PCC and RMSE;
3. use the same evaluated gene width and report `n_pcc_genes`;
4. show a real correction (`correction_rms`) rather than reproducing the
   anchor; and
5. show a coherent ablation: full versus GEX-only measures context H&E, while
   full-Novae versus full-MLP measures Novae.

ST-FID/ST-MMD, nonzero AUC and spatial-domain plausibility remain secondary
diagnostics. PCC and RMSE are the promotion metrics.

## What follows

- If the simple model clears the gate: repeat it across seeds and rotating
  held-out samples, then test a non-ccRCC HEST cohort and a matched STPath
  benchmark. Only after deterministic reconstruction is stable should flow
  matching be reintroduced to model uncertainty.
- If it matches but does not beat harmonic: the data currently support local
  interpolation more than learned residuals. Increase patients and distinct
  masks before adding capacity.
- If it loses to harmonic: stop architecture expansion. Audit expression
  alignment and sample provenance from the saved manifests, then test the
  deterministic model on a non-ccRCC cohort.

STPath remains an important benchmark, but its released predictor normally
uses histology at the target location. Removing target H&E is therefore an
out-of-distribution stress test, not a perfectly matched missing-tissue
baseline. The STPath paper also reports weak CCRCC transfer, so one INT ccRCC
test sample cannot be the final biological conclusion.
