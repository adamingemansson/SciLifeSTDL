# Gen5 addendum — full-expression generation

Additive to `gen3_multiscale/`, dependent on the audited Gen4 suite
(`gen3_multiscale/gen4/`, `GEN4_CONTRACT.md`). Branched from Gen4's tip
into its own branch (`claude/gen5-full-latent-flow`) so Gen4 stays a
clean, independently-reviewable commit. **Nothing in this document
authorizes any change to a file outside `gen3_multiscale/gen5/`,
`gen3_multiscale/configs/gen5/`, `gen3_multiscale/scripts/gen5_*.py`, or
`gen3_multiscale/tests/test_gen5_*.py` / `_gen5_fixtures.py`.** Every
Gen3 and Gen4 module is reused unmodified, by import.

Written before any Gen5 code, per the task's own instruction and the same
discipline `GEN4_CONTRACT.md` established.

## 1. Objective and the critical distinction from Gen4

Gen4's flow arms predict a **residual** around Architecture 3's
deterministic transport mean, in a fixed rank-64 basis:
`prediction = deterministic_mean + decode_residual(flow_sample)`.

Gen5 mirrors Gen4's four conditioning arms exactly, but replaces the
target and the flow's own state space entirely: the flow generates a
**complete latent representation of every gene**, decoded through a
**shared, frozen, full-expression autoencoder** — never added to, mixed
with, or shortcut through the deterministic conditioner's own predicted
mean:

```
prediction = expression_decoder(generated_latent)          # Gen5 — the ONLY allowed form
prediction = deterministic_mean + decoded_residual          # Gen4 — NEVER Gen5's form
```

This is enforced structurally in `gen5/latent_flow.py::Gen5LatentFlowModel`:
`compute_losses`/`sample_predictive_distribution` read `query_hidden` from
the wrapped conditioner and explicitly discard `conditioner_out["expression"]`
— it is never referenced by either method's return value. A leakage test
(`tests/test_gen5_leakage.py::test_conditioner_deterministic_mean_never_enters_the_prediction`)
verifies this by construction (source inspection), not merely by
convention.

## 2. The four arms — identical conditioning to Gen4

| Arm | id | Image representation | GEX conditioning representation | Reused Gen4 conditioner |
|---|---|---|---|---|
| 1 | `gen5a` | UNI2 | `WeightedGeneExpressionEncoder` | `gen4a` |
| 2 | `gen5b` | GigaPath | frozen scFoundation | `gen4b` |
| 3 | `gen5c` | UNI2 | frozen scFoundation | `gen4c` |
| 4 | `gen5d` | context-only STPath | (STPath occupies the image slot; GEX slot unchanged) | `gen4d` |

Each Gen5 arm's `Gen5LatentFlowModel.conditioner` is constructed with
**exactly the same kwargs** as the matching Gen4 arm's `Gen4Conditioner`
(`gen5/model_factory.py` reuses `gen4.model_factory.ARM_TABLE` directly,
imported not copied) — "keeping the conditioning systems matched" is
satisfied by literal kwarg identity, not by two independently-maintained
config shapes that could silently drift apart.

**Conditioner weights are not retrained for Gen5.** The intended
deployment loads each arm's already-trained, validation-selected Gen4
conditioner checkpoint (from Gen4's own Stage 1 — `training/checkpoint.py`,
reused unmodified) into `Gen5LatentFlowModel.conditioner` and freezes it,
mirroring Gen4's own `freeze_conditioner_initially` discipline
(`configs/gen4/*_flow.yaml`). This is the strongest possible "conditioning
systems matched" guarantee: Gen4 and Gen5's flow arms condition on the
literal same trained weights, differing only in what the flow generates
and how it is decoded.

## 3. Shared full-expression autoencoder

One frozen `gen5/autoencoder.py::ExpressionAutoencoder` (encoder
`[G] -> [256]`, decoder `[256] -> [G]`, both plain row-independent MLPs —
no spatial mixing belongs in a per-spot expression compressor, spatial
reasoning is entirely the conditioner/flow's job) is fit **once**, from
**training-sample expression only**, and reused **unchanged across all
four arms** — GEN5_CONTRACT.md's own requirement that "differences come
from conditioning rather than decoder quality."

- Gene order/panel: taken directly from the dataset manifest's
  `gene_panel` (exact order, `gene_panel_hash` recorded and verified on
  load — same discipline `models/gene_basis.py::GeneResidualBasis`
  already uses for its own gene-order binding).
- No fixed top-HVG restriction: the encoder/decoder operate over the
  manifest's complete gene panel, never a pre-selected subset (the
  `train_log1p_variance_top50/200` panels remain evaluation-only
  diagnostics, per Gen3's own convention — GEN4_CONTRACT.md never touches
  this either).
- `gen5/autoencoder_training.py::train_expression_autoencoder` accepts
  only a caller-supplied training-expression matrix — it has no argument
  through which validation/test expression could reach it, mirroring
  `models/gene_basis.py::fit_gene_residual_basis`'s identical "caller's
  responsibility, no validation/test argument exists" limit.
- Persistence: `gen5/autoencoder.py::save_expression_autoencoder_checkpoint`/
  `load_expression_autoencoder_checkpoint` — atomic write, and records
  (verified on load, fail-closed on mismatch): `gene_names`/`gene_names_hash`,
  `dataset_manifest_fingerprint` (caller-supplied, mirrors
  `evaluation/train_gene_panels.py`'s identical fingerprint contract),
  `preprocessing_spec` (the exact target-expression transform string,
  e.g. `"normalize_log1p"`, taken from the manifest's own
  `build_args.expression_transform`), and `code_identity` (a caller-supplied
  string — typically the git commit the checkpoint was fit under, the
  same "fail closed on drift, opt out explicitly" pattern
  `training/train.py`'s `allow_code_drift` already establishes elsewhere
  in this codebase, not new machinery invented for this module).
- **Capacity gate** (`gen5/autoencoder.py::evaluate_autoencoder_reconstruction`),
  run before any flow training: full-gene PCC/RMSE on (a) held-out
  TRAINING spots (a generalization-within-split sanity check on rows the
  autoencoder's own optimizer never saw as gradient targets, if the
  caller splits its training pool) and (b) real validation spots. This
  is the "reconstruction ceiling" — no flow arm can ever exceed it, since
  every arm's output passes through this exact frozen decoder.

## 4. Full latent spatial flow

`gen5/latent_flow.py::Gen5LatentFlowModel` reuses, **unmodified, by
import**:
- `gen4.conditioner.Gen4Conditioner` — produces `query_hidden` (the same
  fused image+GEX+regional+global context tokens Gen4's own flow
  conditions on).
- `models.flow.VelocityNetwork`, `flow_matching_loss`,
  `sample_residual_coefficients` (`models/flow.py`) — already a fully
  generic joint-attention-over-queries transformer operating on
  `[N_query, state_dim]` (its own `residual_rank` constructor argument is
  simply the state width; Gen5 sets it to the autoencoder's `latent_dim`,
  256 by default, rather than Gen4's rank-64 gene-residual-basis width).
  `QueryQuerySelfAttention` inside every `VelocityBlock` is exactly the
  "jointly attend across query spots" mechanism the task requires — no
  new attention module was written for Gen5.

**Training** (`Gen5LatentFlowModel.compute_losses`):
1. `conditioner_out = self.conditioner(inputs)`; `query_hidden =
   conditioner_out["query_hidden"].detach()` (stop-gradient into the
   frozen conditioner, exactly like Gen4's own flow).
2. `z_target = self.autoencoder.encode(target_expression).detach()` (the
   frozen encoder; target expression is the true query GEX — this is the
   **only** place query GEX is ever used, and only during training).
3. `flow_matching_loss(self.velocity_network, z_target, query_coords,
   query_hidden, generator=...)` — identical call shape to Gen4's,
   different target array.

**Inference** (`Gen5LatentFlowModel.sample_predictive_distribution`):
1. `sample_residual_coefficients(self.velocity_network, n_query,
   query_coords, query_hidden, n_samples, n_steps, generator=...)` —
   starts from `torch.randn(...)` (pure Gaussian noise), **never** seeded
   or offset by `conditioner_out["expression"]` or any other deterministic
   value; the function's own body (`models/flow.py`, reused unmodified)
   has no such input.
2. `self.autoencoder.decode(...)` on every sampled latent, per spot.
3. `predictive_mean = decoded_samples.mean(dim=0)` — the primary
   PCC/RMSE comparison value, matching Gen3/Gen4's own convention for
   what the headline metric is computed against.

`conditioner_out["expression"]` is read nowhere in either method's return
value — see §1.

## 5. Leakage requirements and how each is enforced

1. **Query GEX enters only the target expression encoder, during training
   only.** `compute_losses`'s own signature takes `target_expression`
   only for this one call (`self.autoencoder.encode(...)`); no other
   method on `Gen5LatentFlowModel` accepts a query-expression-shaped
   argument at all.
2. **`z_target` is a training target, never a conditioning input.** It is
   `.detach()`'d immediately and used only as `flow_matching_loss`'s
   target argument (`x1`) — never concatenated into `query_hidden` or any
   other tensor `sample_predictive_distribution` could read.
3. **At inference, neither query GEX nor its latent encoding exists.**
   `sample_predictive_distribution` never calls `self.autoencoder.encode`
   at all — only `self.autoencoder.decode`, on the flow's own sampled
   noise-integrated latents.
4. **Query H&E / overlapping WSI tiles remain absent.** Unchanged from
   Gen4 — `self.conditioner` is a `Gen4Conditioner`, subject to every
   leakage guarantee GEN4_CONTRACT.md section 11 already establishes
   (structurally, not by a new check).
5. **scFoundone/STPath context-only.** Unchanged from Gen4 — same
   providers, same cache builders, same `encode_context_only`/`encode_rows`
   interfaces with no query-shaped parameter.
6. **Autoencoder fitting/normalization/gene statistics are training-only.**
   `train_expression_autoencoder` takes one expression matrix; a caller
   passing validation/test rows in that argument is the same
   caller-responsibility limit `fit_gene_residual_basis` already has —
   no code path inside the function can distinguish or reach any other
   data.
7. **Test data is never used for checkpoint/decoder/latent-dim/flow
   selection.** The autoencoder's capacity gate (§3) reports validation
   metrics for diagnostic purposes only; nothing in this package
   auto-selects a latent dimension or checkpoint based on any metric —
   that remains an explicit, logged human/operator decision, matching
   Gen3/Gen4's own "test split exactly once, after all decisions are
   frozen" discipline (`MANUAL_RELEASE_RUNBOOK.md`'s closing line).

## 6. Fairness

Reused directly from Gen4, not re-specified: dataset manifest, mask bank/
mask strata, patient-disjoint train/validation/test split, encoder
caches (`gen4/uni2_spot_cache.py`, `gen4/scfoundation_cache.py`), the
spatial conditioner (§2), seeds, evaluator metrics (`evaluation/metrics.py`,
`pearson_per_gene`/`rmse`, reused unmodified). One common latent-flow
network size (`VelocityNetwork` hidden_dim/n_heads/n_blocks) is used for
all four arms in every committed config (`configs/gen5/gen5{a,b,c,d}.yaml`)
— documented, not merely asserted, since diffing those four files' `model.params.velocity_*`
blocks shows they are identical by construction (all four configs are
generated from one shared params block in this repo's own authoring, and
verified equal by `tests/test_gen5_configs.py`).

## 7. Required gates — where each lives

| Gate | Implementation |
|---|---|
| 1. Autoencoder reconstruction | `gen5/autoencoder.py::evaluate_autoencoder_reconstruction` + `tests/test_gen5_autoencoder.py` |
| 2. One-sample/full-gene overfit | `tests/test_gen5_autoencoder.py::test_autoencoder_overfits_a_single_batch` |
| 3. One fixed-mask latent-flow overfit | `tests/test_gen5_latent_flow.py::test_latent_flow_overfits_a_fixed_mask` |
| 4. Noise-sensitivity proof | `tests/test_gen5_leakage.py::test_different_seeds_produce_different_predictions` |
| 5. Context-sensitivity proof | `tests/test_gen5_leakage.py::test_visible_context_mutation_changes_generation` |
| 6. Target-leakage proof | `tests/test_gen5_leakage.py::test_hidden_target_mutation_cannot_affect_inference` |
| 7. Spatial-coupling proof | `tests/test_gen5_leakage.py::test_query_predictions_are_not_generated_independently` |
| 8. Decoder/checkpoint/gene-order tamper tests | `tests/test_gen5_autoencoder.py::test_*_fails_closed` |
| 9. CUDA memory smoke test | `scripts/gen5_smoke_launcher.py` — see §9 for the honest CPU-only limit |
| 10. Paired evaluator | `gen5/evaluator.py::compare_gen5_arm` (thin, reuses `evaluation/metrics.py` directly) |

## 8. Deliverables map

| Deliverable | Path |
|---|---|
| This addendum | `gen3_multiscale/GEN5_CONTRACT.md` |
| Autoencoder + capacity gate | `gen3_multiscale/gen5/autoencoder.py` |
| Autoencoder trainer | `gen3_multiscale/gen5/autoencoder_training.py` |
| Latent flow model | `gen3_multiscale/gen5/latent_flow.py` |
| Arm dispatch | `gen3_multiscale/gen5/model_factory.py` |
| Paired evaluator | `gen3_multiscale/gen5/evaluator.py` |
| Static preflight | `gen3_multiscale/gen5/preflight.py` |
| Configs (4 arms + shared autoencoder) | `gen3_multiscale/configs/gen5/*.yaml` |
| Smoke-only launcher | `gen3_multiscale/scripts/gen5_smoke_launcher.py` |
| Runbook | `gen3_multiscale/gen5/RUNBOOK.md` |
| Tests | `gen3_multiscale/tests/_gen5_fixtures.py`, `test_gen5_*.py` |

## 9. Explicitly out of scope / honest gaps this round

- No real UNI2/scFoundation/STPath weights (same as Gen4 — none
  downloaded or run).
- No real Gen4 conditioner checkpoint exists yet (Gen4 has not been
  trained on real data), so Gen5's flow arms have never been trained
  against a genuinely frozen, real, trained conditioner — only against
  freshly-constructed (random-init) `Gen4Conditioner` instances in tests/
  smoke, which is sufficient to prove the *mechanism* but not the
  *quality* of conditioning transfer.
- **CUDA memory smoke test**: this environment has no GPU. §7 gate 9 is
  therefore a documented, un-run gap — `scripts/gen5_smoke_launcher.py`
  runs the identical construction/forward/backward/sample path on CPU
  and reports peak CPU RSS only; a real CUDA memory profile
  (`torch.cuda.max_memory_allocated`) must be captured on real hardware
  before any full run.
- No full manifest-driven training-mask iterator wiring (same documented
  gap as `GEN4_CONTRACT.md` section 13/`RUNBOOK.md` section 4) — this
  round's autoencoder trainer and flow tests operate on caller-supplied
  expression matrices / synthetic examples, not a live HEST-1k pipeline.
- `gen5/evaluator.py::compare_gen5_arm` is a thin reuse of
  `evaluation/metrics.py`'s primitives, not a full CLI matching
  `evaluation/gen3_evaluator.py`'s own — building that full parity is a
  follow-up once a real trained checkpoint exists to evaluate.
