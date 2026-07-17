# Results Log

Dated, report-facing notes on real-data findings that are worth citing
directly, as they're confirmed — separate from `docs/architecture_plan.md`
(design rationale/roadmap) and `docs/model_schematics.md` (build status).
Each entry: what was tested, the numbers, and the interpretation.

## 2026-07-17: WAE-GAN's residual-source ordering (Novae alone beats "both")

**Setup**: WAE-GAN + STPath's real pretrained fusion (`context_encoder_type:
stpath`), HEST-1k INT1, 10000 epochs, comparing STPath's Route-B residual
mechanism (`stpath_new_gene_encoder_type`) across its three options —
none (plain baseline), `both` (MLPGeneEncoder + frozen Novae, concat-mode),
and `novae` alone.

| config | PCC | ST-FID |
|---|---|---|
| `exp_hest1k_wae_gan_stpath.yaml` (plain, no residual) | 0.1975 | 12.66 |
| `exp_hest1k_wae_gan_stpath_bothresidual.yaml` (MLP+Novae) | 0.2901 | 3.03 |
| `exp_hest1k_wae_gan_stpath_novaeresidual.yaml` (Novae only) | **0.3453** | 3.44 |

**Finding**: Novae residual alone is WAE-GAN's best configuration found so
far — it beats the combined MLP+Novae ("both") residual, which in turn
beats the plain (no-residual) baseline. Adding MLP's signal into the mix
*hurts* relative to Novae alone for this generator family.

**Why this is likely real, not noise**: it echoes the same pattern already
seen for FM-OT (`docs/architecture_plan.md`'s Route-B ablation, 2026-07-16):
there, `mlpresidual` alone (PCC 0.4291) beat `bothresidual` (0.4154), which
in turn beat the plain baseline (0.3961) and `novaeresidual` alone (0.3904).
So across two structurally different generator families now, the
*combined* MLP+Novae residual underperforms whichever single source is
actually best for that family's training dynamics — FM-OT's best single
source is MLP, WAE-GAN's is Novae, but in both cases concatenating the two
sources dilutes rather than complements the stronger one. Reads as a real,
reportable result: naive concatenation of two residual signals isn't
free — it can cost accuracy relative to just using the better source alone.

**Not yet tested**: `wae_gan_stpath_mlpresidual` (MLP alone) — needed to
confirm Novae is genuinely WAE-GAN's optimum, not just better than MLP by
default. Queued as a follow-up, not yet run as of this entry.

**Related, same investigation**: WAE-GAN + StormLite (our own from-scratch
context encoder, not STPath) shows a reproducible mode collapse (PCC=nan,
constant output across 2 seeds — `exp_hest1k_wae_gan_stormlite_mome_both.yaml`,
seed 0 and seed 1) regardless of `fusion_mode` (`exp_hest1k_wae_gan_stormlite_both.yaml`,
sum mode, scored PCC -0.007 / ST-FID 47.9 — not collapsed to a constant, but
not learning anything useful either). Confirmed via the plain
`exp_hest1k_wae_gan_stpath.yaml` control above (finite, normal PCC=0.1975)
to be StormLite-specific instability, not a general WAE-GAN-family issue —
WAE-GAN is fine with a well-behaved context encoder, it's StormLite's added
complexity that destabilizes its adversarial training specifically.
