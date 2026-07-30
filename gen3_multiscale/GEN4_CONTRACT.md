# Gen4 — encoder/flow ablation suite: contract

Additive to `gen3_multiscale/`. Branched from `codex/gen3-manual-release` at
commit `98db938` into its own worktree/branch
(`claude/gen4-encoder-flow-suite`) so the audited Gen3 checkout can keep
running its own cache-preparation/training jobs undisturbed. **Nothing in
this document authorizes any change to a file outside `gen3_multiscale/gen4/`,
`gen3_multiscale/configs/gen4/`, `gen3_multiscale/scripts/gen4_*.py`, or
`gen3_multiscale/tests/test_gen4_*.py` / `gen3_multiscale/tests/_gen4_fixtures.py`.**
Every existing Gen3 module (`models/architectures.py`,
`models/model_factory.py`, `training/checkpoint.py`,
`evaluation/gen3_evaluator.py`, `data/*`, …) is reused **unmodified, by
import**, never edited, never subclassed-and-monkeypatched in place.

Written before any Gen4 code, per the task's own implementation order.

## 1. Objective

Four additive alternatives to the audited Gen3 Architecture 4 (GigaPath +
`WeightedGeneExpressionEncoder` + multiscale spatial conditioner + residual
flow, henceforth **the reference arm**), swapping only the per-spot
image/GEX encoding while reusing every other piece of Gen3's infrastructure
verbatim: dataset manifest, patient-disjoint splits, collision-free
training masks and fixed held-out validation/test masks, the Architecture 3
spatial-fusion backbone (token projections, attention blocks, transport
head), the Architecture 4 low-rank residual-flow apparatus (gene basis,
velocity network, ODE sampler), transactional checkpoints, and the
evaluator's metric machinery (full-gene PCC/RMSE plus the train-derived
`train_log1p_variance_top50/200` panels).

## 2. The four arms

| Arm | id | Image representation | GEX conditioning representation | Global slide context |
|---|---|---|---|---|
| A | `gen4a` | UNI2 (frozen, precomputed per spot) | `WeightedGeneExpressionEncoder` (trainable, from raw observed GEX — **unchanged from Gen3**) | `MaskAwareCoordinateAttentionPool` over UNI2 tiles (new, trainable, small) |
| B | `gen4b` | GigaPath (frozen, precomputed per spot — **unchanged from Gen3**) | frozen scFoundation context embeddings (precomputed per spot) | `FrozenGigaPathSlideEncoder` (**unchanged from Gen3**) |
| C | `gen4c` | UNI2 (frozen, precomputed per spot) | frozen scFoundation context embeddings (precomputed per spot) | `MaskAwareCoordinateAttentionPool` over UNI2 tiles |
| D | `gen4d` | — (STPath context representation stands in for the image branch; see §5) | — (STPath context representation stands in for the GEX-conditioning branch too) | disabled (STPath's own context representation already fuses image+GEX+space; layering a second global-slide branch on top is out of scope this round) |

Each arm has two model stages, exactly mirroring Gen3 Architecture 3 → 4:

- **`<arm>` (conditioner)** — a `Gen4Conditioner`, structurally
  `gen3_multiscale.models.architectures.Architecture3` with its per-spot
  image/GEX encoding swapped via the table above. Deterministic; trained
  first, alone.
- **`<arm>-flow` (flow)** — a `Gen4ResidualFlowModel`, structurally
  `gen3_multiscale.models.architectures.Architecture4` with its
  `self.conditioner` built as the matching `Gen4Conditioner` instead of a
  plain `Architecture3`. Trained second, against a frozen, validation-
  selected conditioner checkpoint plus a training-only-fit rank-64 residual
  basis (§7).

The reference Gen3 Architecture 4 (`GigaPath + WeightedGeneExpressionEncoder`)
is **not** one of the four arms — it remains the existing, unmodified
baseline all four are compared against.

## 3. Frozen vs. trainable components, per arm

| Component | Arm A | Arm B | Arm C | Arm D |
|---|---|---|---|---|
| Image tile encoder (UNI2 / GigaPath) | frozen, precomputed | frozen, precomputed | frozen, precomputed | frozen, precomputed (GigaPath features feed STPath's image tokenizer, same convention `STPathContextEncoder` already uses) |
| GEX context encoder (scFoundation) | n/a | frozen, precomputed | frozen, precomputed | n/a |
| `WeightedGeneExpressionEncoder` | trainable | n/a (replaced by scFoundation) | n/a | n/a |
| Global/regional pooling module | trainable (`MaskAwareCoordinateAttentionPool`, `pool_regional_tokens`) | frozen (`FrozenGigaPathSlideEncoder`) + trainable regional projection | trainable | n/a |
| STPath spatial transformer | n/a | n/a | n/a | frozen (official pretrained weights) or, until weights are available, a stub satisfying the same interface (§5, §11) |
| STPath output projection (`proj`/`embedding_norm`) | n/a | n/a | n/a | trainable |
| Spot/query token projections, backbone, transport head | trainable, **identical module classes as Gen3**, one fresh instance per arm | same | same | same |
| Gene residual basis (rank 64) | fixed, fit once per arm from that arm's own frozen conditioner's training-only residuals | same | same | same |
| Velocity network (flow) | trainable | trainable | trainable | trainable |

No arm shares trainable weights with another arm or with the Gen3 reference
run — each is an independent instance, matching Gen3 Architecture 1-4's own
independence (this round does **not** attempt cross-arm weight
synchronization the way `model_factory.synchronize_four_architecture_initialization`
does across Gen3's four architectures; that guarantee doesn't transfer
cleanly across encoder families of different width/identity, and inventing
a new one is out of scope).

## 4. Shared parameters (fairness) and intentional divergences

**Shared across all four arms and identical to Gen3's own configs wherever
technically possible:**
- Dataset manifest, patient-disjoint train/validation/test sample split,
  gene panel and gene panel hash.
- Mask bank construction: same `masking.strata`, same collision-free
  training-schedule construction, same fixed validation/test mask bank.
- Coordinate representation: `observed_coords`/`query_coords` stay Gen3's
  stable centered/spot-spacing-normalized frame — no arm introduces a new
  coordinate convention.
- `hidden_dim`, `n_heads`, `n_blocks`, `dense_threshold`, `sparse_k`,
  `chunk_size`, transport-head width/temperature/gate settings, flow rank
  (64), `n_flow_blocks`, `n_flow_samples`, `n_ode_steps`.
- Target expression transform (`normalize_log1p`) and target sum.
- Optimizer (Adam, same `lr`/`weight_decay`/`betas`/`eps`), gradient clip,
  `total_steps`/`max_wall_clock_hours` per diagnostic run, seed.
- Evaluator: full-gene PCC/RMSE as the primary metric, `train_log1p_variance_top50/200`
  as secondary panels, same baselines (mean / nearest-neighbor / harmonic).

**Documented, intentional divergences:**
1. **`image_feature_dim` differs by encoder family.** GigaPath (arm B) is
   1536-d (Gen3's existing value). UNI2 (arms A/C) and STPath's context
   representation (arm D) use their own native output widths, configured
   per-arm (`configs/gen4/<arm>_conditioner.yaml`, `model.params.image_feature_dim`)
   — never coerced to 1536 by truncation or padding. `SpotTokenProjection`
   already normalizes-then-projects this branch to `image_proj_dim`
   regardless of input width, so this divergence has no structural cost.
2. **Global slide context mechanism differs (arm A/C vs. arm B).** Arm B
   reuses Gen3's real, pretrained `FrozenGigaPathSlideEncoder` (a genuine
   LongNet slide transformer). Arms A/C use a new, small, trainable,
   coordinate-aware attention pool (`MaskAwareCoordinateAttentionPool`,
   §6) over the arm's own visible UNI2 tile features — explicitly **not**
   a second large pretrained slide transformer, per the task's own
   instruction. This is a real capability difference between arms, not a
   bug: Arm B's global token is pretrained-informed, arms A/C's is
   learned from scratch on this suite's own training data. Reported
   honestly in the results, never presented as "the same global context
   mechanism."
3. **Arm D has no separate regional/global slide branch at all.** STPath's
   own spatial transformer already performs joint image+expression+space
   reasoning over the context set in one pass; its output *is* arm D's
   per-spot conditioning representation, handed to `SpotTokenProjection`'s
   image-feature slot (see §5). Layering Gen3's regional-grid/global-FiLM
   machinery on top of an already-fused representation was judged
   confusing to interpret and is out of scope for this first suite.
   Consequently arm D's conditioner sets `use_regional_he=False,
   use_global_slide=False` — its capacity is not directly comparable
   width-for-width to arms A-C's regional/global branches, and the
   runbook must say so.
4. **Arm D's `gex_feature_dim`/gene-conditioning slot.** Arm D still
   builds and calls `WeightedGeneExpressionEncoder` on raw observed
   expression for `SpotTokenProjection`'s GEX slot (exactly as Gen3 and
   arms A/B do) — STPath's context representation replaces only the
   *image* slot. This keeps arm D structurally as close as possible to
   the other three arms (same GEX-conditioning module, same input) rather
   than introducing a second point of divergence; the alternative (an
   arm whose *only* per-spot signal is the joint STPath embedding, with
   no separate raw-GEX conditioning term at all) is a plausible follow-up
   ablation, not this round's arm D.
5. **scFoundation embedding dim and STPath `d_model`** are read from each
   encoder's own frozen checkpoint at cache-build time (never assumed) and
   threaded through as `model.params.gex_feature_dim`
   (arms B/C) / `model.params.image_feature_dim` (arm D) — configs carry a
   placeholder value with a preflight check that the built cache's actual
   width matches (§9); real weights are required to know the true value
   (§11).

## 5. Encoder-provider interfaces

Three narrow additive interfaces, all consumed by `Gen4Conditioner`
(`gen4/conditioner.py`) — no arm-specific branching lives in the
conditioner's `forward()` itself beyond "which provider populated this
field," matching Gen3's own "explicit feature flags, not four copy-pasted
models" discipline (`models/architectures.py`'s own module docstring).

- **Image provider** (`gen4/providers.py::ImageContextProvider`):
  produces, *offline*, a `[n_spots, image_feature_dim]` array of visible
  per-spot tile features aligned to a sample's `adata.obs_names` (same
  contract `data/spot_feature_cache.py::load_gen3_spot_features` already
  has for GigaPath) plus, when the arm uses global/regional slide context,
  the same `wsi_tile_features`/`wsi_tile_*_coords`/`full_slide_coord_bounds`
  contract `SpatialFieldInputs` already defines. **`Gen4Conditioner` never
  calls an image provider at forward-time** — it only ever reads whatever
  array `example_builder.build_spatial_field_example`'s
  `precomputed_spot_features` argument placed into
  `inputs.observed_gigapath_features` (the field name is inherited
  unchanged from Gen3; it is encoder-agnostic in practice, since it is
  just `[n_observed, image_feature_dim]` floats). Cache builders (§6) are
  the only place a real encoder forward pass happens.
- **GEX context provider** (`gen4/providers.py::GexContextProvider`):
  produces, *offline*, a `[n_spots, gex_feature_dim]` array of frozen
  context embeddings aligned to `adata.obs_names`, from **row-independent**
  encoding of each spot's own raw observed expression (never a statistic
  computed across spots, never touching validation/test/query rows). At
  forward time, `Gen4Conditioner` selects only the rows for
  `inputs.observed_barcodes` (the same “observed” set the rest of the
  model already sees) — this selection is enforced structurally, since
  `Gen4SpatialFieldInputs.context_gex_embedding` (§8) is built by
  `build_gen4_spatial_field_example` from exactly the same
  `context_barcodes` argument every other observed-* field is built from,
  never from a full-sample array sliced later.
- **Deterministic conditioner** (`gen4/conditioner.py::Gen4Conditioner`):
  same contract as `Architecture3`/`_SharedFieldArchitecture` —
  `forward(inputs) -> {"expression": Tensor[n_query, n_genes],
  "query_hidden": Tensor[n_query, hidden_dim], ...}`. The existing
  residual-flow apparatus (`gen4/flow.py::Gen4ResidualFlowModel`) consumes
  exactly this output, unchanged from how `Architecture4` consumes
  `Architecture3`'s output.

## 6. UNI2 (arms A, C)

- `gen4/uni2_encoder.py::FrozenUNI2TileEncoder` — same fail-closed-on-
  missing-checkpoint discipline as `FrozenGigaPathSlideEncoder`: requires a
  real local checkpoint path and a pinned, 40-hex immutable revision
  string; records `checkpoint_sha256` (real file bytes), `package_version`,
  and an explicit `preprocessing_spec` string at construction. **Never**
  falls back to a different UNI model on load failure — any failure to
  load the pinned checkpoint/revision raises, full stop.
- `gen4/uni2_spot_cache.py` — structurally mirrors
  `data/spot_feature_cache.py`: one encode pass per manifest sample,
  writing an atomic `.npz` (`uni2_gen3_spot_cache/<sample_id>.npz`,
  deliberately its own directory, never collides with the GigaPath cache)
  carrying `features`, `barcodes`, `image_source_available`,
  `patch_content_sha256`, and UNI2's own provenance fields
  (`uni2_checkpoint_sha256`, `uni2_hf_repo_id`, `uni2_hf_revision`,
  `uni2_package_version`, `uni2_preprocessing_spec`,
  `uni2_schema_version`). Loading re-validates provenance and patch
  content exactly like `load_gen3_spot_features` does; a cache built
  against different patches/barcodes/provenance is rejected, never
  silently trusted. The trainer/example builder is only ever given
  `load_uni2_spot_features(...)["features"]` as
  `precomputed_spot_features` — **UNI2 is never called per training
  example.**
- **Regional context**: reuses `models.slide_encoder.pool_regional_tokens`
  / `regional_grid_cell_centers` completely unmodified — these functions
  are already generic over "whatever per-tile feature array," with no
  GigaPath-specific assumption.
- **Global context**: `gen4/uni2_global_pool.py::MaskAwareCoordinateAttentionPool`
  — a single learned inducing query cross-attends over the item's visible
  UNI2 tile features (Fourier-encoded `wsi_tile_regional_coords` appended
  before projection, so the pool is coordinate-aware), producing one
  `[global_slide_dim]` vector. Structurally the same "one learned query
  attends over an observed/visible set" pattern `InducedGlobalGEXPool`
  already uses (`models/global_context.py`), applied to image tiles
  instead of GEX tokens — **not** a second slide-level transformer, no
  positional/sequence modeling beyond the single attention pool, per the
  task's explicit instruction. Zero-count visible-tile items raise
  (mirrors `_regional_he_tokens`'s existing "no visible tile" failure).

## 7. scFoundation (arms B, C)

- `gen4/scfoundation_encoder.py::FrozenSCFoundationEncoder` — lazy-import,
  fail-closed-on-missing-checkpoint wrapper (same discipline as UNI2/
  GigaPath). Requires an exact gene-vocabulary mapping file; records the
  vocabulary's own hash alongside the checkpoint identity so a later
  mismatch between the manifest's gene panel and the encoder's vocabulary
  is caught at cache-build time, not silently zero-padded.
- `gen4/scfoundation_cache.py` — mirrors §6's cache-builder pattern:
  **row-independent** encoding (no batch norm / dataset statistic that
  would leak across spots or across splits), one `.npz` per sample
  (`scfoundation_gen3_spot_cache/<sample_id>.npz`) carrying `features`,
  `barcodes`, `gene_panel_hash` (must equal the live manifest's, checked
  on load), and scFoundation's provenance
  (`scfoundation_checkpoint_sha256`, `scfoundation_vocab_sha256`,
  `scfoundation_package_version`, `scfoundation_schema_version`).
  Availability is tracked the same way GigaPath/UNI2 track image
  availability: a spot with no computable embedding gets an explicit
  zero row plus `feature_available=False`, never a garbage value.
- **At runtime**, `Gen4Conditioner` selects only observed-barcode rows
  (§5) — query rows are never looked up in this cache at all, since
  `Gen4SpatialFieldInputs.context_gex_embedding` never contains query
  rows to begin with (built exactly like every other `observed_*` field,
  from `context_barcodes` only — see §8).
- **Raw observed expression is untouched and separate.** scFoundone
  embeddings only ever reach `SpotTokenProjection`'s `gex_features`
  conditioning slot; `inputs.observed_full_gene_expression` (the real,
  untouched values the transport head draws candidate gene values from)
  is populated identically to Gen3, from `example_builder`'s existing
  path, never derived from or replaced by a scFoundation embedding.

## 8. STPath (arm D)

- `gen4/stpath_context.py::Gen4STPathContextEncoder` — wraps the real,
  already-verified `src.models.stpath_encoder.STPathContextEncoder`
  (lazy import; requires the external `stpath` package + pretrained
  weights + gene-vocabulary file, same fail-closed discipline as
  everything else in this section) but adds one **new** method,
  `encode_context_only(context_coords, context_expression,
  context_image_features, context_image_available) -> Tensor[n_context, hidden_dim]`,
  that is structurally incapable of receiving any query-spot data — its
  signature has no query parameter at all, unlike the existing class's
  `forward()` (which the base class needs for its own, different, pilot-
  study use case and which concatenates context+query rows — **that
  method is never called by Gen4**). `encode_context_only` builds STPath's
  token inputs (`ge_tokens`, `img_feats`, `organ_ids`, `tech_ids`) with
  `n_total = n_context` only, calls `self.model.prediction_head(...,
  return_all=True)` under `torch.no_grad()` (the encoder is frozen), and
  returns the resulting per-context-spot pre-head hidden state, projected
  and normalized by the same trainable `proj`/`embedding_norm` the base
  class already has.
- **Per-mask, not per-sample-once.** Unlike UNI2/scFoundation, STPath's
  contextual mixing genuinely depends on *which* spots are in the context
  set (its spatial transformer attends across them), so it cannot be
  cached once per sample independent of the mask the way a row-independent
  encoder can. It **is** cheap enough to run at training time without
  violating "never run a large encoder per training example" — its own
  image tokenizer consumes already-cached GigaPath features (never raw
  pixels), and the only real forward pass per training step is STPath's
  own (frozen) spatial transformer over the (typically few-hundred-spot)
  context set, not a whole-slide tile-encoder pass. What **is** cached
  once per sample (`gen4/stpath_context.py`'s own small cache helper) is
  the per-spot GigaPath tile feature STPath's image tokenizer consumes —
  an "independent base token," per the task's own instruction — never the
  contextualized output itself.
- **No query patch, no query GEX, ever.** `encode_context_only`'s
  signature has no query-shaped argument; `Gen4Conditioner` calls it once
  per forward pass with exactly `inputs.observed_*` arrays, the same
  context-only content every other arm's providers see.
- The resulting `[n_context, hidden_dim]` representation is placed into
  `SpotTokenProjection`'s `image_features` slot (`image_feature_dim =
  hidden_dim` for arm D's config) — see §4 divergence 4 for why the GEX
  slot is left as the ordinary `WeightedGeneExpressionEncoder` rather than
  also being replaced.
- **`Gen4STPathStub`** (`tests/_gen4_fixtures.py`) — a tiny deterministic
  CPU-only stand-in exposing the identical `encode_context_only` signature
  (a fixed linear layer over concatenated coordinate/expression/image
  summary features), used by every Gen4 test that needs a working arm-D
  conditioner without the real `stpath` package or weights, mirroring
  `_step6_fixtures.py::stub_gigapath`'s existing monkeypatch pattern.

## 9. Cache/provenance discipline (all three new encoder families)

Every new cache builder (§6, §7) follows the exact discipline
`data/spot_feature_cache.py` already established for GigaPath, because
that discipline — not any encoder-specific detail — is what "fail closed
on cache barcode order / provenance mismatch" actually means in this
codebase:
1. Pinned, immutable revision/checkpoint identity, recorded from real
   bytes at cache-build time, never trusted from a caller-supplied string
   alone.
2. Barcode order and a real patch/expression **content** hash recorded at
   build time; both re-verified byte-for-byte on every load.
3. Atomic writes (`os.replace` from a process-specific temp path).
4. A required, versioned field set — an old-format or hand-edited cache
   fails closed with a specific, actionable error, never a silent
   reinterpretation.
5. Unavailable rows are explicit zero vectors plus an explicit
   availability flag — never a garbage or omitted row.

`gen4/preflight.py` runs a **static** check of all of the above (config
schema, required-fingerprint presence, cache-directory/shape/dtype
sanity where a cache already exists) without touching a GPU or requiring
real weights to be loaded; it is the mandatory gate a smoke run refuses
to start without passing (§12).

## 10. Staged flow training (identical shape for every arm)

For every arm `X` independently:

1. Train `gen4X` (the deterministic conditioner) to convergence/diagnostic
   budget, exactly like Gen3 Architecture 3.
2. Select `gen4X`'s best checkpoint using **validation only** (never test).
3. Freeze that exact checkpoint.
4. Fit a **rank-64** gene-residual basis (`models/gene_basis.py::fit_gene_residual_basis`,
   reused unmodified) from `gen4X`'s frozen-checkpoint residuals on
   **training samples/masks only** — never validation or test residuals,
   enforced the same way Gen3 already enforces it: the fitting script
   only ever receives a training-split mask schedule as input, structurally,
   not by a runtime flag a caller could mis-set.
5. Train `gen4X-flow` (`Gen4ResidualFlowModel`, wrapping a fresh
   `Gen4Conditioner` for arm `X`, then loading the frozen `gen4X`
   checkpoint's weights into `self.conditioner`) against that basis.
6. No arm implements a direct ~17k-gene flow this round — every arm's
   flow operates in the same rank-64 coefficient space as Gen3
   Architecture 4.

`gen4/basis_fit.py::fit_gen4_residual_basis` is a thin, arm-parameterized
wrapper around the existing `models/gene_basis.py` primitives and
`scripts/fit_architecture4_residual_basis.py`'s masking/residual-collection
logic, generalized to build a `Gen4Conditioner` instead of a plain
`Architecture3` — it does not reimplement basis fitting.

## 11. Leakage and evaluation invariants (must hold for every arm)

Restating the task's own hard invariants, plus how each is enforced
*structurally* in this design (not merely "checked at runtime"):

1. **Query H&E patches never enter any model.** No Gen4 provider/module
   anywhere has a parameter or method that accepts a query-indexed patch
   array. `Gen4STPathContextEncoder.encode_context_only` has no query
   argument at all (§8) — the one place in this codebase (`STPathContextEncoder.forward`)
   that *does* accept query images is never called by Gen4.
2. **Tiles overlapping the hole are excluded before aggregation.** Reused
   unchanged from Gen3: `data/slide_context.py::visible_slide_context`
   already filters tiles by hole overlap before any WSI field reaches
   `SpatialFieldInputs`; `gen4/uni2_spot_cache.py` and
   `gen4/scfoundation_cache.py` both build from
   `example_builder.load_sample_for_examples`'s own image-availability
   contract, the same one Gen3's GigaPath cache uses.
3. **Query expression never enters scFoundation, STPath, or any other
   encoder.** scFoundation's cache builder (§7) and STPath's context
   encoder (§8) both only ever receive `context_expression`/observed rows,
   constructed the same way Gen3 already constructs
   `observed_full_gene_expression` — from `context_barcodes`, never from a
   full-sample array a query index could accidentally slice into.
4. **STPath cannot predict directly from the hidden query patch.**
   `encode_context_only`'s output feeds `SpotTokenProjection` (a per-
   observed-spot token), which only ever reaches queries through the
   *existing*, unmodified Architecture 3 backbone's query-to-observed
   cross-attention — the same path GigaPath/UNI2 features already use.
   There is no separate "STPath predicts query directly" code path.
5. **Raw observed GEX remains the sole gene-value candidate pool.** No
   Gen4 provider output (image or GEX) is ever concatenated into
   `pool["local_expression"]`/`pool["boundary_expression"]`/
   `shared_expression_parts` — those are built exclusively from
   `inputs.observed_full_gene_expression` in the inherited
   `_SharedFieldArchitecture._candidate_pool`/`forward`, which
   `Gen4Conditioner` does not override.
6. **Query coordinates use Gen3's stable centered/normalized
   representation.** `build_gen4_spatial_field_example` (§ below) calls
   `example_builder.build_spatial_field_example` unmodified for coordinate
   construction — Gen4 introduces no new coordinate transform anywhere.
7. **Validation/test masks are unseen and patient-disjoint.** Reused
   unchanged: same `dataset_manifest`, same `mask_bank`/`mask_fingerprint`
   held-out schedule construction Gen3 already uses.

## 12. Deliverables map

| Deliverable | Path |
|---|---|
| This contract | `gen3_multiscale/GEN4_CONTRACT.md` |
| Provider interfaces | `gen3_multiscale/gen4/providers.py` |
| Extended inputs + builder | `gen3_multiscale/gen4/inputs.py` |
| UNI2 encoder + global pool + cache | `gen3_multiscale/gen4/uni2_encoder.py`, `uni2_global_pool.py`, `uni2_spot_cache.py` |
| scFoundation encoder + cache | `gen3_multiscale/gen4/scfoundation_encoder.py`, `scfoundation_cache.py` |
| STPath context-only encoder | `gen3_multiscale/gen4/stpath_context.py` |
| Conditioner / flow model | `gen3_multiscale/gen4/conditioner.py`, `gen4/flow.py` |
| Arm dispatch (model factory) | `gen3_multiscale/gen4/model_factory.py` |
| Residual-basis fitting wrapper | `gen3_multiscale/gen4/basis_fit.py` |
| Configs (4 conditioner + 4 flow) | `gen3_multiscale/configs/gen4/*.yaml` |
| Static preflight/audit | `gen3_multiscale/gen4/preflight.py` |
| Smoke-only launcher | `gen3_multiscale/scripts/gen4_smoke_launcher.py` |
| Parameter-count / frozen-trainable report | `gen3_multiscale/gen4/param_report.py` |
| Runbook | `gen3_multiscale/gen4/RUNBOOK.md` |
| Tests | `gen3_multiscale/tests/_gen4_fixtures.py`, `gen3_multiscale/tests/test_gen4_*.py` |

## 13. Explicitly out of scope this round

- Downloading or running any real UNI2/scFoundation/STPath checkpoint.
- A full production trainer/evaluator CLI byte-for-byte matching
  `training/train.py`/`evaluation/gen3_evaluator.py`'s own CLIs — the
  smoke launcher (§12) proves construction + one real optimizer step +
  one real inference step per arm on synthetic CPU data; wiring Gen4 arms
  into those exact CLIs (which would require either editing the audited
  Gen3 files or duplicating them wholesale) is a follow-up once real
  weights make an actual run possible.
- Cross-arm weight synchronization (`synchronize_four_architecture_initialization`'s
  guarantee does not extend to this suite; see §3).
- Novae, a harmonic anchor arm, or a combined STPath+UNI2+scFoundation
  hybrid arm (excluded per the task's own instruction).
- Direct ~17k-gene flow (excluded per the task's own instruction; §10).
