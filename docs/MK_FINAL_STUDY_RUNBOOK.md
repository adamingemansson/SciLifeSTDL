# MK final-study runbook

This layer freezes the scientific questions around the existing evaluators; it
does not redefine the 242/14 cohort, target space, panels, or primary metrics.
The machine-readable contract is
`configs/benchmarks/mk_final_study_2026.yaml`.

## 1. Locked deterministic result

Use `mk_wb_parallel_gated` as the primary deterministic model and
`mk_wbw_sandwich` as the structural reference.  Both require independent model
seeds 1, 2, and 10.  Keep the existing fixed-mask report, whole-slide report,
per-gene sidecar, seed summary, gene-rank stability, and Track-B leaderboard.

## 2. Stain-domain diagnosis

Run `gen3_multiscale.scripts.analyze_mk_stain_domain_shift` against the primary
whole-slide report.  It samples real H&E patches, fits the stain reference from
training slides only, and measures held-out distance without modifying pixels.

Interpret the result with `analyze_mk_stain_confounders`.  The retraining gate
is directional: normalization/augmentation is triggered only if greater stain
distance is associated with *lower* PCC (rho <= -0.30, n >= 14).  A positive
association does not justify normalization even when its absolute magnitude is
large.  The post-hoc table reports raw, organ-residualized, within-organ-pair,
and optional spatial-template/noise-ceiling associations.  Any transformed
UNI2 inputs need their own transform-aware cache and may never overwrite the
current raw-patch cache.

## 3. WAE uncertainty

Use 32 latent draws per held-out slide.  Report within-model predictive
dispersion, calibration to absolute error, and finite-draw ensemble convergence.
Compare this with deterministic seed stability using
`compare_mk_wae_and_seed_variability`, but retain the distinction:

- WAE predictive SD: variation in output across latent draws for one model.
- deterministic seed SD: variation in performance across independently trained
  models.

They answer different questions and must not be merged into one uncertainty
number.

Use `summarize_mk_wae_best_individual_draws` to test whether any fixed latent
draw beats the deterministic point prediction on the same sampled spot-gene
values.  Its per-slide oracle is deliberately labelled non-deployable because
it selects a draw after observing target GEX.

## 4. Underdispersion and spatial reuse

The required existing diagnostic is
`gen3_multiscale.scripts.analyze_mk_spatial_template_reuse`.  The final report
must include predicted/target amplitude, effective-rank ratio, exact versus
blurred PCC, and same-gene template reuse.  These explain cases where a smooth,
low-amplitude map obtains reasonable PCC without recovering the true local
expression field.

## 5. External validation

Copy `configs/benchmarks/mk_external_cohort_template.yaml`, fill it without using
the external cohort for model selection, and run
`gen3_multiscale.scripts.audit_mk_external_validation`.  A blocked audit is not
an error to work around; it identifies missing provenance or an invalid external
claim.
