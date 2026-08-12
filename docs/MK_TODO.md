# MK project TODO

The experiment backlog contains **four distinct batches of four arms**. The
first three batches are the intended 4 x 3 plan; Batch D is the additional
conditional structured-field comparison retained after the planning
clarification. Do not merge these into one ambiguous factorial experiment.

All arms use the same frozen UNI2 H&E features and coordinates and predict the
same full normalized-log1p gene panel. No real GEX is visible at inference.

## 0. Preserve current evidence

- [ ] Finish and evaluate the active `refine0` control.
- [ ] Record its exact config, best checkpoint and TensorBoard root.
- [ ] Preserve the completed WAE-MMD diagnostics showing that deterministic
  H&E prediction generally beat prior sampling.

## 1. Shared H&E-to-GEX pathway

For every spot:

`H&E patch -> frozen UNI2 morphology vector -> trainable predictor -> full GEX`

UNI2 does not receive genes. It converts an image into a 1536-dimensional
morphology representation. The trainable predictor learns from paired training
H&E/GEX which morphology patterns correspond to each output gene.

## 2. Batch A — original four architecture/inference controls

These are the four arms already present in the earlier TODO and code. They ask
whether spatial H&E context, deterministic prediction and an image-conditioned
latent prior improve over the original local standard-prior WAE.

- [ ] `mk_local_wae_mmd`
- [ ] `mk_spatial_deterministic`
- [ ] `mk_local_conditional_wae_mmd`
- [ ] `mk_spatial_conditional_wae_mmd`

## 3. Structured mechanisms shared by B–D

| Suffix | Structured hypothesis |
|---|---|
| `within` | A training-only, within-slide-centered gene-program basis helps preserve within-spot coexpression. |
| `between` | Spatial H&E/coordinate attention plus predicted-GEX graph refinement helps spots use neighbouring morphology and predicted fields. |
| `gradient` | Signed local k=6 and wider k=18 gradient supervision prevents spatial transitions from being over-smoothed. |
| `combined` | Within-spot programs, between-spot prediction and multiscale gradient supervision are complementary. |

The gene-program basis must be fitted on training slides only, after
within-slide gene centering/scaling and organ/slide balancing. No ontology,
cell labels or held-out expression may enter it.

## 4. Batch B — deterministic structured-field screen

- [ ] `mk_field_within`
- [ ] `mk_field_between`
- [ ] `mk_field_gradient`
- [ ] `mk_field_combined`

These directly predict one GEX vector from H&E context, with no expression
encoder, latent variable or generative sampling. The implementation exists.

## 5. Batch C — standard WAE-MMD structured-field screen

- [ ] `wae_within`
- [ ] `wae_between`
- [ ] `wae_gradient`
- [ ] `wae_combined`

During training, true GEX is encoded into posterior z and MMD matches the
aggregate posterior to N(0,I). At inference, H&E context is combined with
z sampled from N(0,I). These four arms were part of the intended 4 x 3 plan.

## 6. Batch D — conditional WAE-MMD structured-field screen

- [ ] `cwae_within`
- [ ] `cwae_between`
- [ ] `cwae_gradient`
- [ ] `cwae_combined`

Here H&E context predicts p(z|H&E), and inference samples z from that
image-conditioned distribution. This is the additional four-arm batch retained
after the planning clarification.

Batches C and D are implemented with the same within/between/gradient modules
on posterior reconstruction, deterministic point prediction and sampled
inference paths as appropriate. Their contract tests fail if training and
inference silently use different structured modules. They remain unchecked
until real-data smoke and full runs complete.

## 7. Matched controls

- [ ] Same expanded manifest and patient-disjoint split.
- [ ] Same frozen UNI2 cache and pinned revision.
- [ ] Same full output gene vocabulary and normalization.
- [ ] Same training masks, seed, optimizer, shared dimensions and time budget.
- [ ] Same RMSE+PCC value objective; only declared structured/latent terms differ.
- [ ] Same best-checkpoint rule based on deterministic point-validation loss.
- [ ] Preserve existing TensorBoard scalar and whole-slide image tags exactly;
  add new structured/latent diagnostics under new namespaces only.
- [ ] Preserve the same representative validation slides, genes, colour limits,
  coordinate orientation and target/prediction/error layout.
- [ ] Smoke-test every arm before a full launch.

The prepared scheduler runs A+B together and then C+D, with two processes on
each of GPUs 0,2,3,5. Each four-arm batch keeps a separate TensorBoard. Use
bounded convergence screens before extending any arm; preparation does not
authorize blind long runs. See `docs/MK_16_ARM_RUNBOOK.md`.

## 8. Validation and final evaluation

Keep these four scopes distinct:

1. **Training-time masked validation:** currently 224 fixed items
   (14 validation samples x 4 strata x 4 masks). This selects `best/` and
   provides the frequent TensorBoard validation curves.
2. **Training-time whole-slide validation:** an occasional diagnostic on a
   fixed representative slide subset. This produces TensorBoard maps but does
   not define the 448-item final score.
3. **Final fixed-mask evaluation on the validation split:** currently **448
   items** (14 validation samples x 4 strata x 8 masks). This is the shared
   comparison table for the current experiments.
4. **Final whole-slide evaluation:** every spot in all 14 validation slides,
   used for point, coexpression, Moran's-I and gradient metrics.

- [ ] Use 448 for the current **final fixed-mask evaluation**, not as a generic
  synonym for every validation pass. Do not reuse the old cohort's 288 items.
- [ ] Fail comparison if reports resolve different validation sample IDs,
  strata, mask seeds or item counts; record resolved `n_samples` and `n_items`.
- [ ] Use `n_masks_per_stratum_per_sample` in new configs/reports. Retain
  `n_masks_per_sample` only as a documented backward-compatible alias.
- [ ] Preserve existing evaluator JSON keys and prediction-role names so the
  same extraction scripts work across Gen3/4/5/6, earlier MK and STPath.
- [ ] Use the same normalized-log1p target space; label native-STPath metrics
  as non-comparable rather than mixing spaces.
- [ ] Evaluate every spot in every held-out slide once using H&E only.
- [ ] Report all genes, CCRCC-50, HVG-50 and HVG-200.
- [ ] Report within-spot expression-profile PCC and panel-scale gene-gene
  correlation-matrix agreement.
- [ ] Report per-gene Moran's-I preservation.
- [ ] Report signed k=6 and k=18 gradient PCC/error/energy/direction agreement.
- [ ] Report per-slide, patient-macro 95% CIs and organ-stratified results.
- [ ] Optionally report count-split noise-ceiling-adjusted PCC alongside—not
  instead of—ordinary PCC.
- [ ] For latent arms, report deterministic conditional mean, posterior
  reconstruction and sampled inference separately.
- [ ] Never call posterior reconstruction an inference result: it uses real GEX.

### Backward-compatible reporting contract

Every arm must continue to emit:

- `train/total`, `train/reconstruction`, `train/pcc_loss`, `train/grad_norm`;
- `validation/total`, `validation/rmse`, `validation/pcc_loss`, best step/value;
- the same whole-slide target, prediction and absolute-error images for the
  same fixed slides and genes;
- all-gene, CCRCC-50, HVG-50 and HVG-200 patient-aggregated PCC/RMSE/AUC;
- checkpoint step, evaluated item/sample counts and input-visibility contract.

New values are additive and namespaced (`structured/*`, `gradient/*`,
`latent/*`, `calibration/*`). They must not replace or change established
metrics or images.

## 9. Decision rules

- Use Batch A to decide whether local/spatial and standard/conditional inference
  paths work before interpreting biological structured components.
- Compare the four rows within B, C and D to identify useful structure.
- Compare matching suffixes across B/C/D to isolate the value of generation.
- Retain a latent model only if sampled inference improves held-out prediction
  or supplies useful, calibrated variation.
- If standard WAE-MMD again underperforms its deterministic point prediction,
  treat that as replicated evidence against unconditional N(0,I) inference.
- Prefer the simplest model supported by patient-level held-out evidence.
