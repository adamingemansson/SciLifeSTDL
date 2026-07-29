# gen3_multiscale — frozen contract and phase log

Implements `CLAUDE_HANDOFF_MULTISCALE_SPATIAL_FIELD_ARCHITECTURES.md`'s
staged phases. Status: **All 8 named phases have been implemented and
tested.** All four architectures run real, tested, end-to-end forward
passes on synthetic data, with a full loss/metrics/diagnostics layer
(Phase 7) and four fairness-matrix configs plus a tested launcher (Phase
8). **This is NOT the same as being ready for a real 24-hour run** — see
§20 below: no real gen3_multiscale training entrypoint or real per-sample
HEST-1k data builder exists anywhere in this package, a gap larger in
scope than any single named phase and not assigned to any of them. This
document is extended, not replaced, as further work lands.

## 1. Base commit

Branched from `codex/mask-aware-slide-context` at commit `fd23737`
("Precompute Architecture 4's STPath context once per sample, not every
step") — the branch's current tip at the time this document was written,
carrying every audit fix referenced by the handoff: patient-level
held-out splitting (`split_by_patient`, default `True`), cache
provenance (`model_signature`/`_cheap_file_identity`), checkpoint-resume
fixes (optimizer/RNG state, gene-identity verification, wall-clock
curriculum continuation), STPath raw-count-log1p preprocessing,
`strict_broken_region`, and the metric-naming/ST-FID stability fixes.

`gen3_multiscale/` is a **new top-level package**, developed on the same
`codex/mask-aware-slide-context` branch (per Adam's direction — no new
branch requested). It does not modify, import from, or depend on the
currently-running gen2 4-architecture training jobs' checkpoint
directories or process state in any way; those keep running unaffected.

## 2. Reused audited infrastructure (verbatim copies, not new code)

Per the handoff's "Reuse audited data hygiene and cache logic" and "Do
not copy the older fusion design blindly, and do not compare old and new
result numbers as if they came from the same protocol" — the following
are copied **verbatim** from `gen2_architectures/` (their own docstrings
now note the copy provenance and a "do not let this drift" warning):

- `data/hest1k_catalog.py` — patient-level `resolve_sample_selection`
  (including this session's cross-organ patient-conflict fix).
- `data/mask_bank.py` — mask bank persistence + `query_overlap_report`.
- `training/checkpoint.py` — save/load with optimizer+RNG state,
  `verify_gene_names` (fail-closed gene-identity check), history/
  rollback.

All 41 of these modules' existing gen2_architectures tests were copied
alongside (import paths rewritten only) and pass unmodified against the
gen3_multiscale copies — see `tests/test_hest1k_catalog.py`,
`tests/test_gene_panel_compatibility.py`, `tests/test_checkpoint.py`,
`tests/test_query_overlap_report.py`.

`src/models/hierarchical_slide.py`'s `FrozenGigaPathSlideEncoder`
(official Prov-GigaPath LongNet, mask-aware, fail-closed checkpoint
validation) and `src/models/registry.py`'s
`HierarchicalGeneTransportRegressor` (the gene-value-preserving
transport head this whole design is built around) are **not yet
copied** — they're Phase 3/4 dependencies, pulled in when those phases
start, per the staged plan agreed with Adam.

## 3. Gene panel, preprocessing, and normalization

Not yet pinned to a specific run — no training sample selection has been
resolved for gen3_multiscale yet (that happens once a real config exists,
Phase 8). The MECHANISM is fixed now: `hest1k_catalog.resolve_sample_selection`
+ `checkpoint.verify_gene_names` (both reused verbatim, §2) are the same
audited mechanism gen2_architectures uses today, so gene panel pinning,
ordering, and resume-time identity verification inherit every fix already
made to them this session with zero new code.

## 4. Mask banks and hole-size/shape stratification

`data/mask_bank.py` (reused, §2) is the persistence layer; hole
generation itself comes from `gen2_architectures/data/masking.py`'s
`random_dropout_patches`, which already supports everything Phase 0
item 7 needs without new code:

- **Size stratification**: `radius_range` per masking config, with
  `radius_unit="spot_spacing"` to keep hole size comparable across
  slides with different physical spot spacing (rather than raw
  coordinate units). Plan: generate 3 explicit tiers (small/medium/large)
  as 3 separate masking configs sharing one mask-bank naming scheme, not
  one blended distribution — so results can be reported per-stratum as
  the handoff requires ("The schedule must cover predeclared hole-size/
  shape strata; report results by the same strata").
- **Shape stratification**: `shape="circle"` (compact) vs
  `shape="mixed"` (circle/ellipse/irregular blob, randomly chosen per
  hole) already exist. Plan: circle = "compact" stratum, mixed =
  "irregular" stratum.
- **"Background" contamination is structurally impossible in this
  masking scheme**: Visium's spot grid only contains positions with real
  captured tissue to begin with (no background/off-tissue rows exist in
  `coords_xy`/`slice_ids` at all), so a hole's nominal circular footprint
  can only ever cover real spots — there is no analogue of "the hole
  extends into empty background" the way there would be for a
  pixel/WSI-space mask. What Phase 0 item 8's "natural-edge and
  tear-like masks" stress test actually needs is a hole whose center
  lands close to the tissue's own boundary (producing an asymmetric,
  bite-shaped hole once cropped to real spots) versus one well inside the
  tissue interior. That requires a "distance from hole center to the
  tissue edge" computation gen3_multiscale doesn't have yet — **deferred
  to Phase 2**, where the full per-slide spot adjacency graph (built for
  boundary-ring BFS) makes that distance a natural byproduct rather than
  a separate computation. Primary-benchmark masks for Phase 0 will use
  `center_mode="random"` (uniform over real tissue spots, the existing
  default) without an edge-distance filter; the edge-distance-stratified
  stress-test tier is a Phase 2 follow-up, not blocking Phase 0/1
  sign-off.

## 5. Query-spot overlap policy across validation/test masks

Per Phase 0 item 9 ("Prefer nonoverlapping query spots among
validation/test masks. If overlap is unavoidable, record it explicitly
and never treat those masks as independent biological replicates."):
`query_overlap_report` (reused verbatim, §2) is the measurement
mechanism, already proven in gen2_architectures this session. Not yet
customized for gen3's own mask-generation defaults — `build_mask_bank`'s
existing seed-per-index scheme does not actively avoid overlap, it only
lets it be measured after the fact. Actively minimizing overlap (e.g.
rejection-sampling a new hole against already-placed test-split holes)
is a real, separate piece of work; deferred rather than silently assumed
solved, consistent with what I told Adam about this exact gap in
gen2_architectures's own equivalent fix.

## 6. Cache signature scheme

Not yet implemented as gen3-specific code. Plan (Phase 3, when LongNet/
transport-head caching is actually built): mirror
`gen2_architectures/data/context_features.py`'s `_adata_feature_signature`
+ `data_prep._cheap_file_identity` pattern — content hash of the input
data plus an explicit `model_signature` (checkpoint path + mtime + size)
plus a versioned preprocessing-code string, exactly the discipline that
fixed two real staleness bugs earlier this session (GigaPath preprocessing
version, scFoundation checkpoint identity). No new cache-signature design
needed; only wiring once there's a real cache to protect.

## 7. Shared example object (Phase 1 — implemented)

`data/example.py` defines `SpatialFieldInputs` and `SpatialFieldTargets`
as two separate frozen dataclasses (not one dataclass with a target
field a caller could pass through by mistake) — satisfies "Model-forward
inputs must be separated from target-only fields by type and API, not
merely by convention." `validate_spatial_field_example` enforces every
structural invariant Phase 2/3's real builder will need to satisfy:
observed/query index disjointness and no duplicates, all secondary index
arrays (`query_local_neighbor_idx`, `boundary_idx`) resolve within
`observed_idx`'s range (never the raw per-slide ordering), `boundary_ring`
∈ {1, 2, 3}, non-negative depth-to-boundary, finite values everywhere,
and matching gene-panel width between observed and target arrays.
12 tests in `tests/test_example.py` cover valid construction and each
violation.

**Design fix caught during Phase 2** (recorded here, not silently
smoothed over): the first version of `SpatialFieldInputs` sized `coords`
to `n_observed + n_query` rows while describing `observed_idx`/
`query_idx` as positions in a potentially much larger full-slide
ordering — inconsistent, since a slide's real observed set is typically a
large fraction of the whole slide. Fixed before any Phase 2 code was
built on top of it: `observed_idx`/`query_idx` were replaced with
`observed_barcodes`/`query_barcodes` (provenance-only, never used to
index anything) plus separate `observed_coords`/`query_coords` arrays,
position-aligned with every other `observed_*` content array — the same
convention the other fields already used. This is exactly why phases are
being staged and tested individually rather than built all at once.

Still not yet implemented: the real BUILDER that constructs a
`SpatialFieldExample` from an actual HEST-1k sample + mask bank record,
wiring `boundary_graph.py`'s output together with real loaded GEX/
GigaPath features — that's Phase 3 (WSI tile attachment) territory, once
LongNet/transport-head reuse is wired in.

## 8. Boundary-ring extraction and local context (Phase 2 — implemented)

`data/boundary_graph.py`:

- `build_knn_adjacency(coords, k_neighbors=6)` — geometry-only k-NN graph
  over a combined observed+query coordinate array (default `k=6` matches
  Visium's hexagonal spot lattice). No expression/H&E content anywhere in
  this function.
- `extract_boundary_and_local_context(observed_coords, query_coords, ...)`
  returns:
  - `query_local_neighbor_idx` — TRUE `local_k` (default 32) nearest
    observed spots per query, an independent k-NN search over the full
    observed set (not graph hops) — "the high-resolution local context",
    per the handoff, distinct from the boundary below.
  - `boundary_idx`/`boundary_ring` — Rings 1–3, BFS over observed-observed
    graph edges seeded by Ring 1 (observed spots directly graph-adjacent
    to any query spot). Each observed spot appears in at most one ring
    (its nearest). No random sampling, no hard cap — `max_boundary_size`
    raises `ValueError` rather than truncating if set and exceeded, per
    the handoff's explicit "fail closed... never silently truncate"
    requirement.
  - `query_depth_to_boundary` — BFS over query-query graph edges, seeded
    by query spots that directly touch an observed spot (depth 0);
    verified by test to increase toward a synthetic hole's interior.
    Unreachable query spots (disconnected from any boundary-touching
    query — a pathological case for a genuinely contiguous hole) get a
    `max_rings + 1` sentinel, recorded in the diagnostic manifest, never
    a crash or NaN.
  - `diagnostic` — Phase 2 item 6's required manifest fields: observed/
    query counts, per-ring boundary counts, local-k requested vs.
    effective (padding is reported, not silent), boundary-touching query
    count, unreachable-query count.

11 tests in `tests/test_boundary_graph.py` directly exercise several of
the handoff's own named gates: "Opposite sides of a synthetic hole are
both visible", "Local neighbours differ across suitably separated
queries", and "Fail closed if a configured safety maximum is exceeded;
never silently truncate it" — plus an integration test proving
`boundary_graph.py`'s output actually satisfies `example.py`'s
`validate_spatial_field_example`.

## 9. Mask-aware WSI path (Phase 3 — implemented)

`data/slide_context.py` — copied VERBATIM from `src/data/slide_context.py`
(§2's copy-provenance discipline): dense-WSI-tile-cache loading,
conservative axis-aligned tile/hole footprint overlap removal
(`_overlaps_query_hole`, `visible_slide_context`), and the local-patch
equivalent (`nonoverlapping_context_patch_mask`) already used by
`src/training/train.py`. This is the exact audited mechanism the handoff
asks Phase 3 to reuse, not reimplement. 5 tests (adapted from
`tests/test_hierarchical_missing_tissue.py`'s existing coverage of the
same logic, kept self-contained rather than imported so gen3's copy is
independently verified) cover tile-overlap removal, `all_zero`/`full`
image modes, and the fail-loud "every tile removed" case.

`models/slide_encoder.py`:

- `FrozenGigaPathSlideEncoder` — copied VERBATIM (narrow extraction, not
  the whole file) from `src/models/hierarchical_slide.py`'s identically-
  named class: the official Prov-GigaPath LongNet wrapper, fail-closed on
  a missing checkpoint (never silently falls back to randomly-initialized
  LongNet weights), FlashAttention capability checks, FP16 CUDA inference
  with a bounded output cache. The REST of `hierarchical_slide.py` (the
  older `HierarchicalMissingTissueEncoder` fusion design) is deliberately
  NOT copied — the handoff is explicit: "Do not copy the older fusion
  design blindly." 1 test covers the fail-closed missing-checkpoint path
  (the only part testable without a real ~350MB Prov-GigaPath checkpoint
  and a CUDA+FlashAttention environment); full LongNet forward-pass
  testing is deferred to when real GPU/checkpoint access is available.
- `pool_regional_tokens` — NEW. Prov-GigaPath's stable public interface
  returns only one global CLS vector per slide-encoder call (confirmed by
  reading the class above, not assumed) — it does not expose per-tile
  contextualized states, so regional H&E tokens come from mean-pooling
  the VISIBLE pre-LongNet tile embeddings into a fixed `grid_size x
  grid_size` grid instead, exactly the handoff's specified fallback
  ("Do not depend on private unstable internals merely to obtain
  regional tokens"). Grid cell boundaries are computed from the
  slide's COMPLETE tile extent (`full_slide_coord_bounds`), not the
  hole-shrunk visible subset — verified by test that the same physical
  tile lands in the identical grid cell regardless of which other tiles
  a particular hole happens to remove, so a model could learn to
  interpret "region (i, j)" consistently across different training items
  on the same slide. Empty cells (e.g. fully hole-covered) get an
  explicit `available=False` flag alongside their zero vector, never a
  silently fabricated value indistinguishable from real all-zero content.
  6 tests cover cell assignment, multi-tile averaging, empty-cell
  marking, zero-tile input, bounds-stability across different visible
  subsets, and invalid-bounds rejection.

`evaluation/debug_plot.py` — Phase 3 item 5's visual/debug artifact:
`plot_mask_wsi_boundary_debug` renders one PNG showing observed/query ST
spots, boundary rings 1–3 (colored by ring), and retained/rejected WSI
tile footprints in one coordinate frame — a human-auditable sanity check
that mask-aware WSI filtering and boundary-ring extraction agree with
each other, before any of it is trusted inside a model. matplotlib is
imported lazily (only inside the function) so nothing that transitively
imports this module needs it in a headless environment that never calls
it. 2 tests prove it runs end-to-end against real `boundary_graph.py`
output (not a pixel/visual-regression test — proof of a clean execution
path against the real data shapes it will be fed).

## 10. Gene-encoder selection (Phase 4 item 1 — a real finding, not a formality)

The handoff says: "Read the current gene-encoder ablation results and
choose one conditioning encoder by validation evidence, not test
performance... Choose that encoder from the already implemented full-
panel alternatives... MLP or the existing weighted-linear encoder."

**Finding**: no such ablation has actually been run. `docs/
hierarchical_missing_tissue.md` (the authoritative doc for this exact
model family) states outright, in its own words: "The weighted-linear
GEX encoder is likewise a STPath-faithful baseline, not a claim that it
is the best final representation. A follow-up should compare it... — this
was flagged as FUTURE work, never completed. Every real experiment run
against `HierarchicalMissingTissueEncoder`/`HierarchicalGeneTransportRegressor`
(Suite 1, Suite 2, Round 3, Round 4, the C01-C16/O01-O04 recovery suite)
used `gene_encoder_type="weighted_linear"` throughout; `MLPGeneEncoder`
(`src/models/conditioning.py`, confirmed to exist and be usable) has
never actually been run in this task/model combination. Checked directly
against real docs and code, not assumed from the handoff's phrasing.

**Decision**: freeze `weighted_linear` for all four architectures. This
is the ONLY option with real held-out validation evidence in this exact
model family — by the handoff's own stated criterion ("choose... by
validation evidence"), an option with zero evidence cannot be the
evidence-based choice, however plausible it might be a priori. This is
flagged here explicitly rather than silently treated as "the ablation
was read and this was the answer" — it wasn't; this is the most
defensible call available given what evidence actually exists, not a
claim that a real comparison was performed. If Adam wants an actual
MLP-vs-weighted_linear comparison before committing four 24-hour jobs to
this choice, that is a small, cheap, separate diagnostic — not implied
by anything already run.

## 11. Gene-value-preserving transport head (Phase 4 items 2, 4-7 — implemented)

`models/transport_head.py::GeneValueTransportHead` — the transport MATH
and initialization are extracted from
`src/models/registry.py::HierarchicalGeneTransportRegressor.sample()`
(same tensor shapes, same `* 0.02` score-parameter init scale, same
`1/sqrt(rank)` residual-embedding init, same exact-zero residual-
projection init, same softmax/entropy formulas) — but NOT the whole
class, which hard-constructs `HierarchicalMissingTissueEncoder` (the
older fusion conditioner) inside its own `__init__`. Per the handoff's
"Do not copy the older fusion design blindly": this module is
deliberately ENCODER-AGNOSTIC, taking query/candidate hidden states as
plain tensors so gen3's own spatial-field backbone (Phase 5/6, not yet
built) can feed it, rather than only ever receiving hidden states from
the old encoder's `forward_with_neighbors()`.

Deliberate NEW divergence, required by the handoff's fairness matrix:
the old class ALWAYS blends against an IDW anchor. The handoff instead
requires Architectures 1/3/4 to have "no computational path from any
interpolation output to their predictions or losses" — stricter than
"blend weight near zero." `use_anchor_blend` (default `False`) governs
this structurally: when `False`, `blend_logit` is never constructed at
all (`head.blend_logit is None`, verified by test) and `forward()`
rejects an `anchor_expression` argument outright; when `True`
(Architecture 2 only), `anchor_expression` is REQUIRED and must be
computed OUTSIDE this module by a harmonic solver with no learned
parameters (a Phase 5/6 dependency, not yet built).

14 tests directly exercise the handoff's own named Phase 4 gates:
noncollapse/gradient flow through the scorer, query/candidate hidden
states, and gene gates (item 4); untouched real observed values are
genuinely what gets mixed, verified by swapping in different real
matrices and checking the output changes, plus a uniform-expression
exactness check (item 5); Architecture 2's anchor/candidate/blend/
residual/final outputs are all independently present and distinct in
the returned dict (item 6); the anchor-free structural guarantee for
Architectures 1/3/4 (item 7); transport weights and gene gates are
finite, convex (sum to 1, non-negative), and vary across queries; the
low-rank residual begins at exactly zero and only diverges once its
weight is perturbed (never a free per-query-per-gene bias, since its
gene-side factor has a fixed low rank).

Not yet addressed: Phase 4 item 3 (save normalization, ordered gene
vocabulary, conditioning panel, training-only per-gene scales) is
mechanically INHERITED for free from the reused `checkpoint.py`
(`verify_gene_names`, gene-panel pinning) — but `target_gene_scale`
itself must be computed from TRAINING data only once a real data builder
exists; that's a Phase 5/6 concern, not yet implemented. Item 8 (all
four configs share encoder type/transport heads/gate semantics/residual
rank/init seed) is a config-level requirement, not meaningful until
Phase 8 configs exist — tracked, not yet actionable.

## 12. What is still NOT covered

- No token modules, no model code for the spatial-field backbone itself — Phase 5/6.
- No losses, no diagnostics, no configs, no launcher — Phase 7/8.
- No real per-sample builder wiring `boundary_graph.py` + `slide_context.py`
  + `transport_head.py` + real loaded HEST-1k data into a
  `SpatialFieldExample` and a forward pass yet — that glue code is a
  natural Phase 5/6 dependency once the token modules and architecture
  wrappers exist to consume it.
- No harmonic-solver implementation yet (needed for Architecture 2's
  `anchor_expression` input) — Phase 6.
- `FrozenGigaPathSlideEncoder`'s actual LongNet forward pass is untested
  here (requires a real checkpoint + CUDA + FlashAttention) — only its
  fail-closed missing-checkpoint behavior is verified.
- No real per-sample builder wiring everything together (`example.py` +
  `boundary_graph.py` + `slide_context.py` + real HEST-1k data) into a
  full forward pass through `tokens.py`/`attention.py`/`transport_head.py`
  yet — that assembly IS the four architecture wrappers, Phase 6.
- No full leakage/geometry/learning/numerical gate suite — those tests
  only become fully meaningful once there's a real model and a real data
  builder; Phase 1-5's tests cover the schema/geometry/WSI-overlap/
  transport/token/attention contract layers those later tests will build
  on top of (many individual gates are already directly covered, see
  §8/§9/§11/§13).

## 13. Shared token/attention modules (Phase 5 — implemented)

`models/tokens.py`:

- `FourierCoordinateEncoding` — relative/normalized `[*, 2]` coordinates
  -> log-spaced Fourier features -> a small MLP -> `output_dim` (default
  64, matching the handoff's "~64 dimensions before final projection").
  Used by both spot and query tokens below, and nowhere else invents its
  own coordinate encoding.
- `SpotTokenProjection` — builds one 512-d (default) context-spot token
  from the handoff's exact 5-part schema (§4): local GigaPath H&E (256d),
  compact GEX conditioning (256d), coordinate encoding (~64d), boundary-
  ring identity (4 categories: 0=non-boundary observed, 1/2/3=Rings 1-3
  from `boundary_graph.py`, 16d embedding), and modality-availability
  flags (16d, projected up from a small flag vector, not raw-
  concatenated). Every branch is normalized/projected independently, THEN
  concatenated and projected once to the shared hidden width — never
  summed — per the handoff's explicit "Do not add unrelated modalities
  together before normalization" instruction.
- `QueryTokenProjection` — builds one query token from its coordinate
  encoding, a bucketed depth-to-boundary embedding (from
  `boundary_graph.py`'s BFS hop counts), a single learned "this is a
  query" identity parameter (broadcast to every query token in the item),
  and optional hole-level geometry (area, normalized distance to
  centroid). Its `forward()` signature structurally cannot carry target
  GEX or target H&E — verified by test via `inspect.signature`, not just
  by convention.

10 tests cover shape correctness, the "boundary ring / depth actually
changes the token" (proves each branch is genuinely used, not silently
dropped by the projection), full gradient flow through every modality
branch, and the query-token target-absence guarantee.

`models/attention.py`:

- `RelativeGeometryBias` — the same MLP-on-`(dx, dy, distance)` pattern
  `transport_head.py::GeneValueTransportHead`'s own scorer already uses
  (Phase 4), factored out so every attention site in this backbone
  treats relative geometry identically rather than inventing its own
  encoding per module.
- `ChunkedCrossAttention` — boundary cross-attention processed in
  fixed-size chunks via an online (running) softmax, the same numerical
  recurrence FlashAttention uses — mathematically EXACT regardless of
  chunk size, not an approximation. Verified directly by test: the same
  module, same weights, produces numerically identical output (`atol=
  1e-4`) across chunk sizes 1/3/10/37/100/1000 on the same 37-item
  context (deliberately not evenly divisible by any of them). Also
  verified permutation-invariant to context ordering (handoff's "arbitrary
  barcode or file order cannot become a positional cue" gate) and
  fail-closed on `max_context_size` (raises, never truncates — Phase 2's
  identical policy, reused here).
- `QueryQuerySelfAttention` — dense O(n²) self-attention at or below
  `dense_threshold` (default **256**, the handoff's own stated number),
  sparse k-nearest-query attention above it (reuses
  `boundary_graph.py::build_knn_adjacency`, default `sparse_k=10`, inside
  the handoff's stated 8-12 range). `forward()` returns `(output, mode)`
  so a caller/test can confirm which path actually ran rather than only
  inferring it from the query count.

11 tests cover both attention modules' shapes, the critical chunked-vs-
unchunked numerical correctness gate, gradient flow, permutation
invariance, and dense/sparse mode selection at and above the exact
threshold.

`models/global_context.py` (Architecture 3/4-specific per the fairness
matrix, but a Phase 5 shared-module deliverable):

- `InducedGlobalGEXPool` — 16 (default) learned inducing queries cross-
  attend to all observed GEX tokens; multi-head attention builds each
  inducing token's hidden (molecular-context) output, while a SINGLE
  per-inducing-token convex distribution (the mean of the per-head
  weights — a mean of convex combinations is itself convex) mixes the
  UNTOUCHED real `observed_expression` into 16 value-preserving
  candidates. Query exclusion is enforced structurally: `forward()`'s
  signature (`observed_hidden`, `observed_expression`) has no field a
  query spot could ever occupy — verified by test via
  `inspect.signature`, matching `QueryTokenProjection`'s identical
  discipline above.
- `GlobalConditioningFiLM` — zero-initialized scale/shift FiLM
  modulation for injecting the LongNet global token (Phase 3) into the
  backbone. Zero-init means a freshly-constructed model is
  mathematically IDENTICAL with or without the global token — verified
  by test — so Architecture 3/4's later "slide-token zero/swap must
  measurably change the prediction" diagnostic (Phase 7, not yet built)
  will have a clean, testable pre-training baseline: any dependence a
  trained model shows is something training actually learned, never an
  artifact of initialization.

10 tests cover shape, convexity, untouched-value mixing (identical
pattern to the transport head's own gate: swapping in a different real
expression matrix must change the output; a uniform matrix must be
reproduced exactly), the FiLM identity-at-init property, and structural
query-exclusion.

## 14. Real design bug caught while assembling Phase 6

`ChunkedCrossAttention` (Phase 5) was built assuming ONE shared context
set every query attends to (correct for the boundary, regional H&E, and
global-GEX inducing tokens — all genuinely the same set for every query
in an item). Wiring it up for the LOCAL candidates (Phase 6) exposed a
real mismatch: each query has its OWN local_k=32 nearest neighbors, not
a shared set. Fixed by adding `GatheredCrossAttention` (Phase 5's
`attention.py`, same module the tests below cover): per-query gathered
candidate-set attention, no chunking needed since per-query candidate
counts are small by construction. `MultiscaleBlock`'s local branch now
uses `GatheredCrossAttention`; boundary/regional/global-GEX branches
correctly keep `ChunkedCrossAttention`. Documented rather than silently
fixed — this is exactly the kind of integration bug staged, tested
implementation is meant to surface before it reaches real training.

## 15. Architecture 1/2/3 wrappers (Phase 6 items 1-4 — implemented)

`models/harmonic.py::harmonic_interpolation` — Architecture 2's ONLY
anchor input, and the exact-mask harmonic baseline every arm is compared
against. Deliberately never imports torch (verified by test, reading its
own source) — "The harmonic solver must be outside the trainable neural
input path" is structurally true, not a convention to remember. Solved
by vectorized Jacobi relaxation over the same k-NN graph
`boundary_graph.py` builds. Verified against a REAL mathematical
property, not a smoke test: on a regular grid, the discrete-Laplace
solution for a linear boundary field is exactly linear (a linear
function equals the mean of any symmetric neighbor set) — reconstructed
to within 0.5 absolute error on a 21×21 grid. Also verified: the maximum
principle (no interior overshoot beyond the observed value range) and
determinism. 7 tests.

`models/geometry_utils.py` — `compute_relative_geometry` (handles both
per-query-gathered and shared-context candidate coordinates),
`compute_hole_geometry` (log-compressed query count as an area proxy +
per-query normalized distance to the hole centroid, the handoff's query-
token "hole-level geometry" field), `scatter_boundary_ring` (expands
`boundary_graph.py`'s sparse boundary-only ring labels into a full
per-observed-spot ring array `SpotTokenProjection` needs). 7 tests.

`models/backbone.py` — `MultiscaleBlock` assembles Phase 5's modules
into Architecture 1's exact 5-step block (query-query self-attention,
gated local+boundary cross-attention, feed-forward) with
`use_regional_he`/`use_global_gex`/`use_global_slide` flags extending it
to Architecture 3/4's richer version — the SAME class, never a
subclassed or duplicated block. `SpatialFieldBackbone` stacks
`n_blocks` (default 4, within the handoff's 4-6 range) of them. 9 tests,
including a direct test of Phase 6 item 4 ("Confirm common state-dict
modules initialize identically across arms for the same seed") at the
backbone level: two backbones built with identical config and the same
seed have byte-identical parameters.

`models/architectures.py` — `_SharedFieldArchitecture` (the common
assembly: token projections, `SpatialFieldBackbone`, `GeneValueTransportHead`,
optional `InducedGlobalGEXPool`), with `Architecture1`/`Architecture2`/
`Architecture3` as thin flag-setting subclasses — never four (well,
three so far) copy-pasted models, per Phase 6's explicit instruction.
`Architecture2`'s only difference from `Architecture1` is
`use_anchor_blend=True`; `Architecture3`'s is `use_global_gex=True`
(`use_regional_he`/`use_global_slide` raise `NotImplementedError` with a
clear message when set — Phase 3's WSI regional/global-slide tokens
exist but aren't wired into `forward()` yet, tracked below, not silently
ignored).

**Two things documented rather than silently glossed over** (both in
`architectures.py`'s own module docstring too):
- The transport candidate pool CONCATENATES local + boundary candidates
  rather than the handoff's literal "deduplicated union" — a query's
  true-nearest local neighbor is often also a Ring-1 boundary spot, so a
  handful of candidates can appear twice, receiving correlated extra
  weight in the transport gate's softmax. Not a leakage or correctness
  bug (weights still sum to 1), but a real, flagged simplification;
  proper deduplication needs per-query masked attention, a real follow-up.
- Modality-availability flags are hardcoded to "available" — no real
  per-spot H&E-missing signal is wired in from `SpatialFieldInputs` yet
  (that data-builder plumbing doesn't exist).

7 integration tests build a full synthetic `SpatialFieldInputs` via
`boundary_graph.py` (the same square-grid-with-a-hole pattern Phase 2's
own tests use) and run REAL forward passes through all three
architectures for the first time — covering output shapes, Architecture
1/3's structural anchor-freedom, Architecture 2's real (not synthetic)
harmonic anchor and its near-anchor behavior at initialization,
Architecture 3's global-GEX pool and its `NotImplementedError` guards,
end-to-end gradient flow across dozens of parameters, and Phase 6 item 4
at the FULL-MODEL level: Architecture 1 vs 2 (identical parameter
structure apart from one deterministically-filled, non-RNG-consuming
`blend_logit`) are proven byte-identical everywhere they share a
parameter name for the same seed; Architecture 1 vs 3 (which
legitimately diverges in RNG-stream order once its extra randomly-
initialized modules are constructed) are proven identical specifically
in the token-projection modules constructed before that divergence — a
more modest, honestly-scoped claim than "the whole model," which isn't
actually a meaningful property once architectures have different
parameter counts.

## 16. Architecture 4 (Phase 6 item 5 — implemented)

`models/gene_basis.py::GeneResidualBasis`/`fit_gene_residual_basis` — the
fixed (never an `nn.Parameter`, never gradient-trained) low-rank basis
mapping full-gene residuals to/from a compact coefficient field, fit via
truncated SVD on a caller-supplied TRAINING-only residuals matrix (this
module has no way to enforce the training-only part from inside, the
same structural limitation `harmonic.py` and `target_gene_scale` already
have — documented, not silently assumed safe). Its gene ordering is
hashed and `verify_gene_residual_basis` fails closed (mirrors
`checkpoint.verify_gene_names`) if a caller's current panel doesn't
match. Verified against real linear-algebra properties: basis rows are
provably orthonormal, reconstruction error strictly decreases as rank
increases, and a full-rank basis reconstructs its own fitting data almost
exactly. 9 tests.

`models/flow.py` — `VelocityNetwork` built from the SAME
`QueryQuerySelfAttention`/`ChunkedCrossAttention` modules every other
architecture uses (never a from-memory reimplementation, per the
handoff's explicit warning), conditioned on a single shared
`sinusoidal_time_embedding` broadcast identically to every query token
("one shared continuous-time value for the whole hole").
`flow_matching_loss` implements linear/rectified conditional flow
matching (`x_t = (1-t)x0 + t x1`, target velocity `x1 - x0`, one shared
`t` and noise draw per hole). `sample_residual_coefficients`
Euler-integrates the learned ODE for multiple independent draws.

**Real bug caught by testing, not by inspection**: sampling initially
left the velocity network in whatever training-mode it was already in,
so an active dropout mask (drawing from the GLOBAL torch RNG on every
forward call) silently broke reproducibility even under a fixed
`generator` — caught by `test_sampling_is_reproducible_with_a_fixed_generator`
failing with small, not-obviously-wrong-looking deltas. Fixed: sampling
now forces `eval()` for its duration and restores the caller's original
training mode afterward (`try`/`finally`), verified by a dedicated
regression test. 11 tests total, including a direct test of the
handoff's own gate ("Architecture 4's velocity output must depend on
both time and conditioning, and multiple samples must not be identical").

`models/architectures.py::Architecture4` — wraps a full, unmodified
`Architecture3` instance as `self.conditioner` (satisfies "Use exactly
the Architecture 3 conditioner and deterministic transport mean... do not
change its width, number of blocks, boundary selection, slide inputs,
gene encoder, or transport head" by literally reusing the same class, not
re-deriving an equivalent one). Three methods, not a single overloaded
`forward()`:
- `forward(inputs)` — runs ONLY the deterministic conditioner, identical
  contract to `Architecture3.forward()` (a caller computing the shared
  deterministic losses never needs to know which architecture it holds).
- `compute_flow_matching_loss(inputs, target_expression)` — computes the
  target residual against the DETACHED deterministic mean, projects it
  through the gene basis, and calls `flow_matching_loss`. Both
  `query_hidden` and `deterministic_mean` are unconditionally `.detach()`'d
  before use — "Initially stop gradients from the flow loss into the
  deterministic conditioner" with no flag to disable it in this first
  implementation, matching the handoff's "initially" framing.
- `sample_predictive_distribution(inputs, n_samples, n_steps)` — draws
  multiple residual-field samples, adds them to the deterministic mean,
  and returns `predictive_mean`/`predictive_std`/`predictive_samples`
  (plus `expression` aliased to `predictive_mean`, per "Primary PCC/RMSE
  comparison should use the predictive mean across samples").

4 integration tests build on the same synthetic `SpatialFieldInputs`
fixture as Architectures 1-3: `forward()` matches the conditioner
contract and stays anchor-free; backpropagating the flow loss ALONE
leaves every conditioner parameter's `.grad` as `None` while the velocity
network's parameters receive finite gradients (the stop-gradient contract
verified directly, not just documented); the predictive distribution has
correct shapes and its samples are provably not identical; and a gene
basis fit on a different gene panel is rejected at construction.

**This completes Phase 6** — all four architectures now run real,
tested, end-to-end forward passes.

## 17. What is still NOT covered

- Regional H&E tokens and the real LongNet global-slide token are not
  wired into any architecture's `forward()` yet (`use_regional_he=True`/
  `use_global_slide=True` raise `NotImplementedError`) — Phase 3's
  `slide_context.py`/`slide_encoder.py` exist but a data-builder that
  produces regional-grid tokens and a real LongNet forward pass per
  training item doesn't yet.
- Transport candidate pool is concatenated, not deduplicated (§15).
- Modality-availability flags are not yet real per-spot signals (§15).
- No losses beyond a plain MSE used in the gradient-flow test — the
  handoff's spatial-gradient loss, full metric suite, and diagnostic
  interventions are Phase 7.
- No configs, no launcher — Phase 8.
- No real per-sample DATA BUILDER wiring actual HEST-1k samples (not
  synthetic square grids) into `SpatialFieldExample`/forward passes —
  everything through Phase 6 has been verified on synthetic geometry;
  real-data wiring is a Phase 7/8 concern.
- No full leakage/geometry/learning/numerical gate suite from the
  handoff's own "Mandatory pre-run gates" section — many individual
  gates are already directly covered across Phases 1-6 (see the running
  list in each phase's section above), but a systematic pass against
  that full checklist hasn't been done.

## 18. Losses, metrics, and diagnostic interventions (Phase 7 — implemented)

`models/losses.py` — the shared deterministic training objective, called
identically by all four architectures (never a per-architecture
reimplementation): `primary_reconstruction_loss` (plain MSE, matching the
only loss actually exercised through Phase 6's gradient-flow tests) plus
`spatial_gradient_loss`, the handoff's "one weak spatial-gradient
objective on query graph edges that compares predicted differences with
true differences ... This loss must match gradients rather than force
neighbouring spots to be identical." Built on the SAME k-NN graph
`boundary_graph.build_knn_adjacency` uses everywhere else (never a
second, different notion of "query graph edges"). `combined_reconstruction_loss`
assembles both with the handoff's stated default weight (0.05).

**Verified against the exact design claim, not a smoke test**: a
prediction offset from the truth by the same constant vector at every
query has identical edge-to-edge differences to the truth, so it incurs
near-zero gradient loss despite large primary-reconstruction loss —
`test_spatial_gradient_loss_matches_gradients_not_absolute_levels`
exercises this directly. 11 tests total.

Documented simplification: `per_gene_scale` ("after scale normalization")
falls back to the per-gene std of the target batch WITHIN the call when
not supplied — the same structural limitation `harmonic.py` and
`transport_head.py`'s `target_gene_scale` already have (a training-only
fit must happen outside this function; there is no data builder yet to
fit one from). Not silently assumed to already be a training-set scale.

`evaluation/metrics.py` — `pearson_per_gene`/`rmse`/`nonzero_auc`/
`frechet_distance`/`st_fid`/`st_mmd`/`embed_pca`/`pool_knn_neighborhood`
are copied VERBATIM from `gen2_architectures/evaluation/metrics.py` (same
§2 provenance discipline; these are pure numpy/scipy/sklearn functions
with no gen2-specific dependency, so no adaptation was needed).
`resolve_gene_panels` is adapted (same logic, public name) from
`gen2_architectures/evaluation/audit_evaluation.py`'s private
`_resolve_gene_panels`.

Confirmed absent anywhere else in the repo before building (checked
directly by an Explore agent, not assumed from the handoff's phrasing —
see the research summary that motivated this phase): patient-level
aggregation, hole-size/boundary-to-interior binning, a spatial-gradient
loss, and any variogram/graph-Laplacian agreement metric. Built new:

- `aggregate_patient_metrics` — "aggregate primary results first within
  held-out patients and then macro-average across patients... report the
  pooled descriptive value [too]... include confidence intervals at the
  patient level when the number of held-out patients permits them;
  otherwise state that uncertainty is not estimable from one patient."
  Returns `patient_mean` (primary number), `pooled_mean` (explicitly
  separate, flat descriptive value), and a 95% CI with `ci_estimable=0.0`
  when `n_patients < 2` rather than a misleadingly-computed interval.
  Verified with a deliberately imbalanced fixture (1 item from patient A,
  3 items from patient B) proving the patient-mean (0.5) differs from the
  pooled mean (0.75) — the exact failure mode patient-level aggregation
  exists to prevent.
- `boundary_interior_bins` / `hole_size_bins` — bin a per-query or
  per-item metric by `query_depth_to_boundary` or hole size respectively;
  shared `_bin_by_value` helper, `np.digitize`-based, fail-visible bin
  labels (not silently dropped out-of-range values).
- `edge_gradient_agreement` — query-edge gradient agreement decomposed
  into components normal/tangential to the hole boundary, per Phase 7's
  explicit request. "Normal" is defined as radial alignment with the
  query set's own centroid — the SAME hole-geometry convention
  `geometry_utils.compute_hole_geometry` already uses elsewhere in this
  project (Phase 6), not a new one invented for this metric — classified
  by whichever of radial/tangential the edge direction's absolute dot
  product is larger against.
- `spatial_variogram_agreement` / `graph_laplacian_agreement` — the
  handoff asks for "spatial variogram OR graph-Laplacian agreement"; both
  are provided since each is cheap given the k-NN graph already built
  elsewhere. Verified with a real property test:
  `test_spatial_variogram_agreement_detects_a_spatially_scrambled_field`
  proves the variogram (unlike PCC/RMSE) is sensitive to a
  same-values-wrong-locations corruption — the identical detectability
  gap `pool_knn_neighborhood`'s own docstring already documents for
  ST-FID/ST-MMD's plain per-point embeddings.

21 tests total (`tests/test_metrics.py`), including smoke tests of the
verbatim-copied functions (proving the copy itself is correct, not
re-deriving gen2's own much larger test battery for functions that did
not change).

`evaluation/diagnostics.py` — evaluation-time modality-ablation
interventions, matching the handoff's Phase 7 list exactly:
`zero_observed_gex`, `shuffle_observed_gex`, `zero_he`,
`shuffle_boundary_gex`, `permute_boundary_order`,
`zero_global_slide_vector`/`swap_global_slide_vector`. Each of the first
five returns a NEW `SpatialFieldInputs` via `dataclasses.replace` (never
in-place mutation); the caller runs a real architecture's `forward()` on
the original and perturbed inputs and compares outputs.

**Real subtlety handled explicitly, not glossed over**:
`shuffle_boundary_gex` cannot naively permute every `boundary_idx`
position, because `_candidate_pool` (Phase 6) concatenates local and
boundary candidates from the SAME `observed_full_gene_expression` array
WITHOUT deduplication (§15's documented simplification) — a spot can be
both a query's true-nearest local neighbor AND a Ring-1 boundary spot.
Shuffling that spot's boundary role would silently corrupt its local role
too, violating "local neighbours remain intact." `shuffle_boundary_gex`
computes `boundary_idx - {all local_neighbor positions}` and only
shuffles within that set; a hand-built minimal example with a
deliberately overlapping local/boundary index (`test_shuffle_boundary_gex_leaves_locally_referenced_and_non_boundary_spots_untouched`)
verifies the overlapping position is provably untouched while the
boundary-only positions are exactly a permutation of their original rows.

**Scope limitation, documented rather than silently worked around**: the
global-slide-token zero/swap diagnostic requires a real LongNet global
token wired into an architecture's `forward()`, which §15/§17 already
record does not exist yet (`use_global_slide=True` raises
`NotImplementedError` in every architecture wrapper). `zero_global_slide_vector`/
`swap_global_slide_vector` are trivial tensor functions, exercised
instead directly against `SpatialFieldBackbone`/`MultiscaleBlock`'s own
`use_global_slide` path (which DOES already structurally accept a
`global_slide_vector`, per `backbone.py`) — proving the mechanism itself
is correct and ready, not proving any current architecture uses it yet.
`test_global_slide_intervention_has_no_effect_at_initialization_but_measurable_effect_once_trained`
is a genuine two-part learning-test analogue of the handoff's own gate
("Architecture 3's prediction must measurably change under slide-token
zero/swap on a smoke example; otherwise the slide branch is functionally
ignored"): at fresh construction `GlobalConditioningFiLM`'s zero-init
(Phase 5) means the intervention correctly has NO effect (confirms the
test doesn't manufacture a false positive); after manually perturbing the
FiLM weights away from zero to simulate a trained state, the same
intervention MUST and does measurably change the output. A real bug was
caught building this test: the backbone defaults to `dropout=0.1` and
`nn.Module.training=True`, so per-call dropout masks alone changed the
output between calls regardless of the global vector, confounding the
comparison — fixed by calling `backbone.eval()` before the comparison
(the same dropout-during-inference class of bug already caught once in
Phase 6's `sample_residual_coefficients`, §16 — now caught a second time
in a different module, reinforcing that this is a recurring hazard to
check for, not a one-off).

8 tests total (`tests/test_diagnostics.py`).

## 19. What is still NOT covered

- Everything §17 already listed (regional H&E/global-slide tokens not
  wired into any architecture's `forward()`; non-deduplicated transport
  candidate pool; hardcoded modality flags; no real per-sample data
  builder; full leakage/geometry/learning/numerical gate-suite pass not
  yet systematic) remains true — Phase 7 did not touch any of it.
- The global-slide-token zero/swap diagnostic is verified at the
  `MultiscaleBlock`/`SpatialFieldBackbone` level only, not against a full
  architecture wrapper — blocked on the same `use_global_slide` wiring
  gap as everything else involving the real LongNet token.
- No training loop calls `combined_reconstruction_loss` or any Phase 7
  metric/diagnostic yet — this phase built and tested the functions
  themselves, not an integration into a training/eval harness (that's a
  natural Phase 8 dependency, once configs/a launcher/a real data builder
  exist to run them against).
- No 90%-interval-coverage / predictive-diversity-vs-uncertainty-stratified-by-depth
  reporting for Architecture 4 specifically — `sample_predictive_distribution`
  (Phase 6) already returns `predictive_std`/`predictive_samples`, and
  `boundary_interior_bins`/`hole_size_bins` (this phase) already support
  binning any per-query/per-item scalar by depth or hole size, so the
  building blocks exist, but nothing yet composes them into that specific
  named report — flagged, not silently assumed done by proximity.
- Diagnostic evaluations only build PERTURBED INPUTS; no orchestrator
  runs "the full checkpoint through every intervention and tabulates the
  deltas" end to end yet (a natural Phase 8 launcher/report concern, not
  a Phase 7 one per the handoff's own phase split).

## 20. Four configs and a launcher (Phase 8 — implemented)

`configs/architecture{1,2,3,4}.yaml` — one resolved config per
architecture, matching the fairness matrix exactly: architecture1/2/3
share `_SharedFieldArchitecture`'s constructor shape (`model.params`
mirrors `architectures.py::Architecture1/2/3` kwargs field-for-field);
architecture2 differs only in `use_anchor_blend: true`; architecture3
differs only in `use_global_gex: true` (its `use_regional_he`/
`use_global_slide` stay `false` with an explicit header comment pointing
at the real reason — §15/§17/§19's NotImplementedError gap — never
silently downgrading the fairness matrix's "Yes" cells without saying
so). architecture4.yaml has a genuinely different `model.params` shape
(no `use_anchor_blend`/`use_regional_he`/`use_global_slide` keys at all,
since `Architecture4.__init__` doesn't accept them — they're fixed by
its internal `Architecture3` construction) plus flow-only fields
(`n_flow_blocks`, `n_flow_samples`, `n_ode_steps`, `gene_basis_rank`).
Every config declares `documented_divergences`, an explicit dotted-key
allow-list for the fields it's intentionally different on, which
`launch_four_gpu_suite.py::static_config_audit` reads back and enforces
— "All fields not shown in [the fairness matrix] table must remain
identical unless a difference is structurally required and documented"
is machine-checked, not just asserted in prose. Verified directly: the
real four YAML files, loaded through the same `OmegaConf.load` +
`OmegaConf.to_container(resolve=True)` path `gen2_architectures`'s own
training entrypoints use, pass `static_config_audit` with zero
violations (`test_static_config_audit_passes_for_the_real_four_configs`).

`training/launch_four_gpu_suite.py`:

- `static_config_audit(named_configs)` — flattens each config to dotted
  keys, compares every key present in ALL configs being audited, and
  flags any that differ without appearing in the fixed
  `_ALWAYS_ALLOWED_TO_DIFFER` set or a config's own
  `documented_divergences`. Keys present in only SOME configs (e.g.
  Architecture 4's flow-only params) are never compared — a different
  parameter SHAPE is not itself a violation, only an undocumented
  disagreement on a key every config claims to share.
- `check_required_fingerprints(config)` — "refuse to start if any
  required checkpoint, vocabulary, cache, split, or mask-bank fingerprint
  is absent." Reads a config's own `required_fingerprints: {name: path}`
  block and fails closed on both a `null` path and a path that doesn't
  exist on disk.
- `launch_suite(...)` — one job per GPU via `CUDA_VISIBLE_DEVICES`, CPU
  threads capped via `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/
  `OPENBLAS_NUM_THREADS`/`NUMEXPR_NUM_THREADS` (same env-var discipline
  `scripts/run_gene_aware_job.py` already uses elsewhere in this repo,
  not a new convention). Runs `static_config_audit` and (unless
  explicitly skipped) `check_required_fingerprints` on every config
  BEFORE spawning any subprocess — either failing means nothing is ever
  started, verified by asserting the log directory stays empty. All jobs
  are started with `Popen` before any is waited on, so "one job per GPU"
  genuinely means concurrent, not sequential. Writes one log file per
  job plus one machine-readable `suite_summary.json`; `SuiteResult.ok` is
  `False` if any job's exit code is nonzero — "preserve nonzero exit
  status and stop promotion if any arm fails."
- `run_suite_with_smoke_gate(...)` — runs the smoke variant of every
  config first; the full run is only started if every smoke job
  succeeded, returning `(smoke_result, None)` rather than starting
  anything when the gate fails — "run fail-closed smoke tests before
  full training."
- `main()` — the only code path that can start a real subprocess against
  a real GPU, guarded by `if __name__ == "__main__":`. Nothing in this
  module runs on import; every test drives the library functions
  directly with a stub `command_builder` (a `python -c` one-liner that
  echoes its env/argv and exits 0 or 1 on command) — never a real
  training job. This is how "do not automatically launch the 24-hour
  jobs as part of implementation" is satisfied structurally rather than
  by only not calling `main()` in a test file.

17 tests (`tests/test_launch_four_gpu_suite.py`), including the audit
running against the real config files (not just synthetic fixtures), a
proof that a failed audit/fingerprint-check leaves the log directory
completely empty (nothing spawned), GPU-pinning/thread-capping verified
by reading back what the stub subprocess actually saw in its own
environment, and the smoke-gate's "full run never starts" behavior
verified by asserting the `full/` log subdirectory doesn't even get
created.

## 21. The gap Phase 8 alone cannot close — no real training entrypoint

**This is the single most important thing to flag before any Codex
audit or real run, and is deliberately NOT buried in a bullet list.**

The handoff's Phase 8 says "create exactly four primary configs
corresponding to the fairness matrix" and "one launcher" — both are done
and tested (§20). But a launcher only launches something. No phase of
this project (0 through 7) built a real gen3_multiscale TRAINING
ENTRYPOINT: nothing wires an optimizer loop, checkpoint save/load, and a
real per-sample HEST-1k data loader together into a runnable
`gen3_multiscale/training/train.py` the way `gen2_architectures` has for
its own four (different) architectures. Concretely, still missing:

- **A real per-sample data builder.** Every one of Phases 1-6's tests
  (and Phase 7's) runs against a SYNTHETIC square coordinate grid
  (`_synthetic_inputs`, repeated per test file). Nothing yet loads a real
  HEST-1k sample, applies a mask-bank record, and calls
  `boundary_graph.extract_boundary_and_local_context` +
  `slide_context.visible_slide_context` + real loaded GEX/GigaPath
  features to construct an actual `SpatialFieldExample`. §7/§17/§19
  already flagged this gap after Phase 1/6; it is UNCHANGED by Phase 7
  or 8 — nothing in either phase touched it, because neither phase was
  scoped to.
- **A real optimizer/training loop.** No code anywhere in
  `gen3_multiscale/` calls `.backward()` outside a test, builds an
  optimizer, or iterates over epochs/steps. `models/losses.py`'s
  functions are ready to be called by one; nothing calls them from a
  loop yet.
- **Checkpoint wiring.** `training/checkpoint.py` (copied verbatim from
  `gen2_architectures`, §2) can save/load state dicts, but nothing
  constructs an `Architecture1/2/3/4` instance from a resolved config and
  hands it to `checkpoint.save`/`checkpoint.load` — the glue is missing,
  not the mechanism.
- **`default_command_builder`'s target module,
  `gen3_multiscale.training.train`, does not exist.** `launch_four_gpu_suite.py`
  names it explicitly as the intended entrypoint and its own module
  docstring flags this; every launcher test exercises the launcher's
  OWN mechanics through a stub command instead, precisely because the
  real target isn't buildable from what exists today.
- **Consequently, none of the handoff's own "Deliverables for the Codex
  audit" that require a RUN are producible yet**: item 9 ("tiny-overfit
  and held-out smoke results for all four arms"), item 10 ("parameter
  counts... peak GPU memory, and measured steps per hour" — parameter
  counts ARE producible right now by just constructing each architecture
  from its resolved config, but peak memory/steps-per-hour need a real
  step to execute), and the "Mandatory pre-run gates"' Learning/
  Numerical/Operational sections (one-slide overfit, resume-reproduces-
  next-loss, mixed-precision NaN check, etc.) all need a real training
  loop to run against, which does not exist.

None of Phases 0-8 as literally named in the handoff was "build the
training loop" — Phase 8 presupposes one already exists to be launched.
Building it (a real data builder plus a real train.py) is a substantial,
separate body of work, comparable in scope to `gen2_architectures`'s own
`training/data_prep.py` + `train_local_neighborhood.py` (tasks #43/#48/
#49 in this session's own history) — not a small addendum to Phase 8.
Flagged here explicitly, exactly as every other simplification and gap
in this document has been, rather than letting "all 8 phases done" read
as "ready to run."

## 22. Response to the external Codex audit of commit 386bcf4

Adam forwarded an independent Codex audit of the exact remote commit
`386bcf4` (the Phase 8 tip). Its verdict: do not start the four 24-hour
runs yet. Every numbered finding below was checked against the actual
code before any fix was made or declined -- consistent with this
project's standing discipline of never accepting an audit claim on
faith. Status: **CONFIRMED AND FIXED**, **CONFIRMED, ALREADY DISCLOSED**
(no new action -- §21 already said this), or **CONFIRMED, DELIBERATELY
NOT FIXED** (a real judgment call, not a bug, surfaced rather than
silently resolved either way).

1. **"No real training system."** CONFIRMED, ALREADY DISCLOSED -- this
   is exactly what §21 already says in this document, written before the
   audit arrived. No new action; still true.
2. **"Architectures 3/4 do not contain the intended whole-slide
   architecture."** CONFIRMED, ALREADY DISCLOSED -- §15/§17/§19/§20
   already record that `use_regional_he`/`use_global_slide` raise
   `NotImplementedError` and that Architecture 3 currently only adds
   global observed-GEX tokens. No new action; still true.
3. **"Models are CPU-bound internally and would fail on CUDA."**
   CONFIRMED AND FIXED. Checked directly: `_observed_tokens`,
   `_candidate_pool`, `_harmonic_anchor`, `compute_flow_matching_loss`,
   and `sample_predictive_distribution` all built fresh tensors
   (`torch.as_tensor`/`torch.zeros`/`torch.ones`) with no `device=`
   argument -- a model moved to CUDA would still receive CPU-resident
   coordinates, indices, modality flags, and a CPU-resident harmonic
   anchor at every forward call. `geometry_utils.py`'s
   `compute_hole_geometry`/`scatter_boundary_ring` had the identical bug
   one level down. Architecture 4's `gene_basis.basis` was a plain
   dataclass tensor, never registered with the module, so `.to(cuda)`
   would not move it. Fixed: every one of the above now threads
   `next(self.parameters()).device` (or an already-device-correct
   tensor's own `.device`) through explicitly; `Architecture4` now
   registers `_gene_basis_matrix` as a real buffer and computes
   `to_coefficients`/`from_coefficients` against it directly instead of
   the dataclass's own tensor. Verified by
   `test_forward_output_tensors_live_on_the_models_own_device` and
   `test_architecture_4_flow_and_sampling_output_tensors_live_on_the_models_own_device`
   (this sandbox has no CUDA to move the model to, but the same code
   paths are exercised regardless of which device they resolve to) and
   `test_architecture_4_registers_the_gene_basis_as_a_buffer_not_a_plain_tensor`.
4. **"Configs cannot directly construct the models."** CONFIRMED AND
   FIXED. Checked directly: `configs/architectureN.yaml`'s `model.params`
   includes `gene_encoder_type`/`init_seed` (and Architecture 4 also
   `gene_basis_rank`), none of which any architecture constructor
   accepts -- `Architecture1(**config["model"]["params"])` raised
   `TypeError`. New `models/model_factory.py::build_architecture`/
   `resolve_model_kwargs` filter a resolved config down to real
   constructor kwargs, fail closed (raise) on any UNRECOGNIZED field
   rather than silently dropping it, and construct Architecture 4's
   extra `gene_basis`/`gene_names` requirement explicitly. Verified
   directly against all four REAL config files
   (`test_build_architecture_constructs_a_real_model_from_each_real_config`),
   not just synthetic fixtures.
5. **"Full-gene transport will probably run out of memory."** CONFIRMED
   AND FIXED -- and confirmed to be a real, serious bug: `_candidate_pool`
   broadcast the shared boundary candidate pool's full-gene expression to
   `[Nq, n_boundary, G]` before concatenating with the local pool; for
   500 queries x 500 boundary candidates x ~17,000 genes x 4 bytes that
   is ~17 GB for one tensor, in one forward pass. Root cause: the audited
   transport math (`GeneValueTransportHead`, extracted from
   `HierarchicalGeneTransportRegressor`) only ever accepted one dense
   per-query candidate pool, with no notion of "this subset of candidates
   is the same for every query." Fixed properly, not patched around:
   `GeneValueTransportHead.forward()` gained an optional
   `shared_candidate_hidden`/`shared_candidate_relative_geometry`/
   `shared_candidate_expression` path (`shared_candidate_expression` is
   always `[S, G]`, NEVER `[Nq, S, G]`) that computes ONE joint softmax
   over the combined local+shared logits (which stay small -- hidden_dim,
   not gene-dim) and then splits the final gene-value einsum into a
   per-query term (small, local-only) plus an ordinary matmul against the
   shared pool (no `[Nq, S, G]` tensor ever materialized). This is a pure
   memory-scaling refactor of the SAME audited math, not a change to it
   -- proven by
   `test_shared_candidate_path_matches_the_dense_per_query_equivalent`,
   which asserts the split path and the old dense-broadcast path produce
   numerically identical output on a small, tractable case. All 14
   pre-existing `transport_head.py` tests (which never use the new
   shared path) still pass unmodified, confirming full backward
   compatibility.
6. **"Global-GEX value candidates are computed and then discarded."**
   CONFIRMED AND FIXED -- and it fixed naturally alongside #5, since both
   shared the same underlying broadcast-and-concatenate pattern.
   Checked directly: `InducedGlobalGEXPool.forward()` genuinely returns a
   value-preserving `expression` field per inducing token, but an earlier
   version of `_SharedFieldArchitecture.forward()` only ever read
   `gex_out["hidden"]` (for the backbone's attention) and never passed
   `gex_out["expression"]` to the transport head at all -- real GEX
   candidates were computed and silently thrown away. Fixed: when
   `use_global_gex=True`, `gex_out["expression"]` is now concatenated
   into the SAME shared-candidate pool as boundary spots (§5's fix) and
   reaches the transport head's actual prediction. Verified by
   `test_architecture_3_uses_global_gex_pool_expression_in_the_transport_candidate_pool`,
   which monkeypatches the pool to emit two different expression values
   for the same hidden state and confirms the final prediction changes.
7. **"The selected GEX encoder is not implemented."** CONFIRMED AND
   PARTIALLY FIXED. Checked directly: `weighted_linear` was recorded in
   CONTRACT.md section 10 as the frozen encoder CHOICE, and every config
   names it, but no module anywhere in `gen3_multiscale/` actually
   computed `observed_gex_conditioning` from raw expression --
   `SpatialFieldInputs` only ever documented that field as already-encoded
   upstream input. Fixed the part that was genuinely just missing:
   `models/gene_encoder.py::WeightedGeneExpressionEncoder`, copied
   VERBATIM (same provenance discipline as every other reused module)
   from `src/models/hierarchical_slide.py`'s identically-named class --
   a small, self-contained, already-audited `nn.Linear(n_genes,
   output_dim, bias=False)` module. **NOT fully closed**: nothing yet
   CALLS this encoder inside a real per-sample data builder to actually
   populate `observed_gex_conditioning` from raw counts, and
   training-only normalization/scale-fitting for its input doesn't exist
   either -- that wiring is part of the still-missing real training
   system (§21), not something addable in isolation. What was
   unambiguously true before this fix -- "the encoder implementation
   doesn't exist in this codebase at all" -- is no longer true; "it
   isn't wired into a real pipeline yet" remains true and is already
   covered by §21.
8. **"The shared residual specified in the design is disabled in all
   four configs."** CONFIRMED, DELIBERATELY NOT CHANGED -- a real
   modeling decision, not a bug, and not mine to make unilaterally. The
   handoff's own wording is permissive: "the output head transports the
   untouched observed full-gene vectors and MAY add the same
   zero-initialized low-rank residual in every arm" -- not "must." All
   four configs agreeing on `use_residual: false` also satisfies the
   handoff's actual hard requirement ("the SAME... in every arm").
   Codex's underlying concern is legitimate on its merits, independent of
   the wording question: with `use_residual: false`, Architectures 1-3
   are strictly convex transports of observed values (can only ever
   predict within the convex hull of what's observed, never extrapolate
   beyond it), and Architecture 4's flow field is the only source of
   any non-convex correction. Since the residual is zero-initialized
   either way, flipping this flag currently has NO effect at all (no
   training loop exists yet to move it away from zero) -- it only
   matters once real training exists. Left as an explicit open decision
   for Adam rather than silently resolved in either direction here.

Also addressed, from the audit's "required fix order" list rather than
its numbered findings:

- **#9, "normalize effective configs before comparing them... guarantee
  identical shared initialization, not merely identical seeds."**
  `model_factory.py` (finding #4's fix) makes this checkable at the
  level that actually matters:
  `test_build_architecture_from_real_configs_gives_architecture_1_and_2_identical_shared_initialization`
  constructs REAL Architecture1/2 instances from the REAL config files
  (via the factory, the same path a real training entrypoint would use)
  and diffs their actual parameters -- stronger evidence than diffing
  either raw YAML text or hand-crafted kwargs dictionaries, which is what
  Phase 6's original identical-initialization tests did. A separate
  "resolved-kwargs" audit function in the launcher was considered and
  deliberately not built -- constructing the real models and diffing
  parameters directly is strictly stronger evidence of the same claim,
  so a second, weaker mechanism checking the same thing would be
  redundant, not additive.
- **#10 (real exact-mask harmonic/IDW/nearest-neighbour baselines plus
  the full evaluator) and #11 (the mandatory CUDA/leakage/learning/
  numerical gate suite)** remain blocked on the same missing real
  training system and real per-sample data builder §21 already
  describes -- no new code in this pass, not because they're
  unimportant, but because they are not fixable in isolation from that
  larger gap.

**What this pass did NOT do**: build the real training entrypoint, the
real per-sample HEST-1k data builder, real exact-mask baselines, or the
full evaluator. §21's core verdict is UNCHANGED by this pass: completing
the 8 named phases plus these fixes is still not the same as being ready
for a real 24-hour run. What changed is that the code that DOES exist
had a genuine, confirmed OOM bug, a genuine discarded-computation bug, a
genuine CUDA-readiness bug, and a genuine config/constructor mismatch --
all four are now fixed and regression-tested, so whenever the real
training system is eventually built, it will be built on top of a
transport head, device handling, and config/model bridge that are
already correct, rather than inheriting these four bugs silently.

## 23. Response to the second external Codex re-audit (of commit 547f51e)

Adam forwarded a second, independent Codex re-audit, this time of
`547f51e` (§22's fixes). Verdict: the four bug fixes from §22 are
legitimate, but the system still isn't ready for the four 24-hour runs
-- "the model components are safer; the real experiment pipeline still
does not exist." Same discipline as every prior audit response: every
claim checked against the actual code before any fix or non-fix.

**Small factory bug -- CONFIRMED AND FIXED.** `build_architecture`
indexed `_ARCHITECTURE_CLASSES[architecture_id]` BEFORE calling
`resolve_model_kwargs` (which does the real validation), so an unknown
architecture id raised a raw `KeyError` instead of the intended
actionable `ValueError`. Fixed by reordering the two calls; verified by
`test_build_architecture_raises_value_error_not_key_error_for_an_unknown_id`.

**"Gene encoder not functionally integrated" -- CONFIRMED AND FIXED,
properly this time.** Checked directly: `WeightedGeneExpressionEncoder`
existed (§22) but nothing called it, `SpatialFieldInputs.observed_gex_conditioning`
was a plain numpy array populated before `forward()` with no gradient
path back into any encoder, and `gene_encoder_type` was silently stripped
by the factory as pure metadata -- so the configs claimed an encoder the
model never used, exactly as the audit said. Fixed by implementing the
audit's own recommended fix ("owned and called inside the model from raw
observed GEX tensors"): `_SharedFieldArchitecture` now constructs and
owns a `WeightedGeneExpressionEncoder`, calling it on
`observed_full_gene_expression` (an existing, always-populated field --
no schema change needed) inside `_observed_tokens`, rather than reading
the separate `observed_gex_conditioning` field at all. `model_factory.py`
now VALIDATES `gene_encoder_type` (raises if it names anything other
than `"weighted_linear"`, since that's the only encoder ever
constructed) instead of silently discarding it. `observed_gex_conditioning`
stays in the schema for backward compatibility/a future precomputed-
conditioning path, with its docstring updated to say plainly that no
current architecture reads it. Verified two ways:
`test_gene_encoder_is_genuinely_wired_into_the_model` (gradients reach
`gene_encoder.projection.weight`; changing `observed_full_gene_expression`
changes the output) and `test_observed_gex_conditioning_field_no_longer_affects_the_model`
(changing ONLY the now-unused field changes nothing). A real dropout-vs-
`eval()` test bug (the same class of bug already caught twice before in
this project -- Phase 6's `sample_residual_coefficients`, §18's global-
slide diagnostic test) was caught building the second test and fixed by
calling `model.eval()` before comparing two forward passes.

**"Shared initialization only proven for Architectures 1 and 2" --
CONFIRMED AND FIXED, not just re-documented.** The audit's own diagnosis
was correct and matched what §15 already honestly admitted: Architecture
3 constructs extra randomly-initialized global-GEX modules before some
shared modules, so RNG-stream order legitimately diverges and "same
seed" alone doesn't guarantee identical shared parameters once that
happens. Rather than continuing to document this as a scoped limitation,
`model_factory.py::synchronize_shared_initialization` now removes it: for
every non-reference model, every parameter is copied FROM a reference
model wherever a same-name (after stripping a per-model name prefix,
e.g. Architecture 4 wraps a whole Architecture 3 as `self.conditioner`)
same-shape parameter exists in the reference -- a guarantee that holds
regardless of construction order, not merely "started from the same
seed." `test_synchronize_shared_initialization_gives_all_four_real_architectures_identical_shared_parameters`
builds all FOUR real architectures from the real config files,
synchronizes them, and proves every genuinely shared parameter (correctly
excluding parameters that are structurally different shapes across arms,
like `branch_gate`'s output width, which legitimately differs between
2-branch and 3-branch backbones) is byte-identical across every pair --
including Architecture 3 and Architecture 4, the previously-unaddressed
case the audit specifically called out.

**"The configured mask schedule cannot be generated" -- CONFIRMED AND
FIXED.** Checked directly and reproduced exactly as predicted: the
reused `mask_bank.py::make_split` (verbatim copy) expects
`masking_cfg.strategy`/`masking_cfg.params`; the real configs'
`masking.strata` list had no consuming code anywhere, so building a real
schedule from them would indeed raise `ValueError("unknown masking
strategy None")`. New `data/mask_schedule.py::build_stratified_mask_bank`
converts each stratum entry into a real `masking_cfg`
(`strategy="random_dropout_patches"`, matching `gen2_architectures/data/masking.py`'s
existing hole-size/shape stratification support per CONTRACT.md section
4's original plan), calls `mask_bank.build_mask_bank` once per stratum
with per-stratum seed offsets (so seeds never collide across strata),
and merges the results into one stratum-tagged schedule. This is
genuinely different in kind from the still-missing real per-sample
HEST-1k data builder: mask scheduling operates purely on
coordinate/slice-id arrays, so it's fully testable on synthetic data
today (the same reasoning that let Phase 2's boundary extraction be
built and tested before any real data pipeline existed). 12 tests,
including one that loads the masking.strata block from all FOUR real
config files and builds a real stratified bank from each, and one that
calls the actual reused `mask_bank.make_split` directly with a converted
stratum config to prove the exact failure mode the audit predicted no
longer occurs.

**Findings re-confirmed as ALREADY DISCLOSED, no new action:**
- "There is no real training pipeline" -- unchanged; this is exactly
  §21's own subject.
- "Architectures 3 and 4 do not implement the intended slide
  architecture" -- unchanged; §15/§17/§19/§20/§21 already say this.
- "Leakage safety is structurally promising but not demonstrated
  end-to-end" -- unchanged; blocked on the same missing real data
  builder §21 already describes.
- "The GigaPath slide cache key hashes coordinates and a namespace, but
  not tile-feature contents, preprocessing fingerprint, or checkpoint."
  Checked directly against `models/slide_encoder.py`'s
  `_tensor_digest(coords, namespace)` -- confirmed accurate, and this is
  EXACTLY what §6 already flagged before this audit arrived: "Cache
  signature scheme... Not yet implemented as gen3-specific code... No new
  cache-signature design needed; only wiring once there's a real cache to
  protect." Re-confirmed, not new; still deferred to when the real WSI/
  cache system is actually built.

**Findings that are real but deferred, not fixed in this pass:**
- **Harmonic anchor recomputed on CPU every forward() call, not cached.**
  A genuine performance (not correctness) concern for a real 24-hour run
  -- Architecture 2's `_harmonic_anchor` does run a full NumPy Jacobi
  solve per call today. Caching it properly requires the same
  sample+mask-fingerprint cache-signature system §6 already defers to
  "when there's a real cache to protect" -- building it in isolation,
  before the real data builder exists to define what a "mask fingerprint"
  even is in production, would be premature. Tracked here explicitly
  rather than silently dropped.
- **A representative CUDA forward/backward + AMP + peak-memory gate is
  mandatory before long runs.** Cannot be done in this environment at
  all -- confirmed no CUDA device is available in this sandbox (checked
  directly: `torch.cuda.is_available()` returns `False`). §22's device
  fixes make the CODE PATH correct for whichever device the model
  actually lives on; they cannot substitute for actually running that
  code on a GPU. This remains an explicit, unclosable-here gate for
  whoever has GPU access before any real run.

**What this pass did NOT do**: the core §21 verdict is UNCHANGED --
completing every fix in §22 and §23 is still not the same as having a
real training system. What changed: two concrete, previously-real
correctness gaps this audit specifically named (the gene encoder being
present-but-unused, and the mask schedule being non-functional) are now
genuinely fixed and tested, not just documented as gaps; the shared-
initialization guarantee that was honestly scoped down to "Architecture
1 vs 2 only" now covers all four architectures for real; and a small but
real error-handling bug is fixed. Nothing here builds the missing
training entrypoint, real data builder, real baselines, or GPU-dependent
gates -- those remain exactly where §21 left them.

## 24. Response to the third external Codex re-audit (of commit ca7cf53)

Adam forwarded a third re-audit, of `ca7cf53` (§23's fixes). Verdict:
"another meaningful improvement... but Claude's statement that shared
initialization is now guaranteed 'across all four real architectures' is
premature, and the mask schedule is only a tested generator -- not yet
the schedule used by an experiment." Same discipline as every prior
round: every claim checked against the actual code before any fix.

**Accepted without dispute** (build_architecture validation order, gene
encoder wiring/gradients, `observed_gex_conditioning` no longer
influencing predictions, mask-strata-to-`random_dropout_patches`
conversion, CUDA/transport-memory/global-GEX fixes remaining intact) --
no new action needed for these; re-confirmed accurate.

**"Shared initialization is not guaranteed in the real four-GPU
experiment" -- CONFIRMED, two real sub-issues, both addressed.**

1. *"The launcher starts four separate subprocesses... and never calls
   the synchronizer."* Correct: `synchronize_shared_initialization`
   only ever proved something within one Python process. Added
   `model_factory.py::persist_synchronized_initializations`, which
   writes each architecture's (already-synchronized) weights to its own
   checkpoint directory via the existing, audited
   `training/checkpoint.py::save_trainable_state`, plus one manifest
   recording a SHA256 hash of every parameter AND a `shared_hash_groups`
   section listing every set of architecture.param_name locations that
   turn out to be byte-identical -- "a manifest containing SHA256 hashes
   and the exact shared-parameter comparison," per the audit's own
   suggested fix. `load_synchronized_initialization` is the loading
   counterpart. **Honestly scoped, not overclaimed**: this produces the
   durable, cross-process ARTIFACT a training entrypoint could load; it
   cannot make that entrypoint actually load it, because that entrypoint
   still doesn't exist (§21). "Require each training subprocess to load
   its assigned initialization checkpoint before constructing the
   optimizer" remains unenforced until a real trainer exists to enforce
   it -- flagged explicitly, not glossed over.
2. *"The 'all four' test... does not verify Architecture 3-specific
   parameters against Architecture 4's matching conditioner parameters,
   because those parameters do not exist in Architecture 1."* Checked
   directly and confirmed: a single `reference="architecture1"` call can
   never even ATTEMPT to copy `gex_pool.*` (Architecture 3/4-only,
   e.g. the global-GEX inducing pool) because no name in Architecture
   1's parameter dict can ever match it -- a structural limit, not a
   bug in the sync function itself. Fixed with
   `synchronize_four_architecture_initialization`, a two-hop
   synchronization (architecture1 -> architecture2/architecture3, then
   architecture3 -> architecture4's conditioner) that transitively
   reaches every genuinely shared parameter, including `gex_pool.*`.
   Verified directly: `test_synchronize_four_architecture_initialization_gives_all_four_real_architectures_identical_shared_parameters`
   confirms `gex_pool.inducing_queries` matches between Architecture 3
   and Architecture 4's conditioner (impossible to check with the old
   single-hop call), and a companion test
   (`test_synchronize_shared_initialization_alone_cannot_reach_architecture_3_only_modules`)
   locks in exactly why the single-hop version could never reach it, so
   this can't silently regress back to the narrower guarantee.

**"The mask schedule is generatable, but not yet operational" --
CONFIRMED, addressed as far as buildable without a real trainer.**
Every specific technical point checked:
- *"nothing outside its tests calls it"* -- still true; unchanged,
  since nothing calls it until a real training entrypoint exists (§21).
- *"no ensure/load/save path"* -- CONFIRMED AND FIXED. Added
  `save_stratified_mask_bank`/`load_stratified_mask_bank`/
  `ensure_stratified_mask_bank`, mirroring `mask_bank.py`'s own
  `save_mask_bank`/`load_mask_bank`/`ensure_mask_bank` exactly (same
  atomic-temp-file-then-`os.replace` write, same fail-closed staleness
  validation on reload).
- *"its combined bank is version 1 and incompatible with the existing
  version-2 `load_mask_bank()` staleness validation"* -- correct
  diagnosis, and a deliberately DIFFERENT dedicated trio rather than a
  forced fit: a stratified bank has no single `masking_fingerprint`
  (it has one per stratum), so reusing `mask_bank.load_mask_bank`
  directly was never the right target; `load_stratified_mask_bank`
  validates against `strata_fingerprint` instead (see next point).
- *"it has no combined fingerprint covering the ordered strata
  definition"* -- CONFIRMED AND FIXED. Added `strata_fingerprint()`,
  hashing the ORDERED list of converted `masking_cfg`s (plus split
  counts/seeds) -- verified sensitive to both content changes AND
  stratum reordering.
- *"its default schedule contains validation and test masks only, not
  training masks"* -- the underlying observation is correct, but the
  fix is NOT a third `split_counts` entry on the explicit-record path:
  `mask_bank.py`'s own existing design already splits this way on
  purpose (`build_training_seed_bank`'s docstring: storing millions of
  explicit training records "would create multi-gigabyte JSON," so
  training masks use a lossless SEED SCHEDULE instead of per-record
  storage). Added `build_stratified_training_seed_bank`/
  `ensure_stratified_training_seed_bank`, the stratified counterpart of
  that existing pattern: round-robins training draws across strata
  (never starving one by chance for small `n_items`) with the same
  per-stratum seed-offset discipline `build_stratified_mask_bank` uses.
- *"the config's `train_mask_bank`, `validation_mask_bank`, and
  `test_mask_bank` remain `null`"* -- unchanged; these stay null until a
  real training run resolves real output paths, which is downstream of
  the still-missing training system, not something to fake here.

**"The unused conditioning field should be removed or optional" --
CONFIRMED AND FIXED, choosing removal.** `observed_gex_conditioning`
is gone from `SpatialFieldInputs` entirely (not kept-but-optional): the
codebase's own standing principle is to delete what's genuinely unused
rather than leave a vestigial field a future change could silently start
reading again ("dead inputs are dangerous," per the audit itself). Every
reference across `data/example.py`, `evaluation/diagnostics.py`
(`zero_observed_gex`/`shuffle_observed_gex`/`shuffle_boundary_gex`
now touch only `observed_full_gene_expression`), and every affected test
fixture (`test_architectures.py`, `test_boundary_graph.py`,
`test_diagnostics.py`, `test_example.py`) was updated; a test whose
entire premise was "the field exists but is unread" was removed as moot
rather than kept as dead test code.

**"The main blockers are unchanged" -- re-confirmed accurate, no new
action.** Every item in that list (no real HEST builder, no trainer/
evaluator, no executable initialization loading, no global/regional WSI
inputs in Architectures 3/4, no complete leakage proof, no harmonic
caching, no realistic CUDA/AMP/memory gate) is exactly what §21/§23
already say. "No executable initialization loading" specifically is now
narrower than before this pass -- the checkpoint format and manifest
exist and are tested; only the LOADING CALL inside a real training loop
is still missing, because that loop is still missing.

## 25. Response to the fourth external Codex re-audit (of commit 0fd46e5)

Adam forwarded a fourth re-audit, of `0fd46e5` (§24's fixes). Verdict:
"the two-hop synchronization logic and mask persistence are structurally
correct... I would accept this commit as progress," with two further
correction sets before building the trainer. Same discipline as every
prior round: every claim checked against the actual code before any fix.

**"Hashed initialization checkpoints are not actually verified" --
CONFIRMED, multiple real sub-issues, all addressed.**

1. *"`load_synchronized_initialization()` ignores the manifest
   completely."* CONFIRMED -- checked directly, it only ever called
   `checkpoint.load_trainable_state`. Fixed: `load_synchronized_initialization`
   now accepts an optional `manifest`/`architecture_name` and, when given,
   fails closed in two stages -- (1) the checkpoint FILE's own SHA256 must
   match the manifest before anything is loaded, catching a modified,
   corrupted, or wrong-architecture-directory checkpoint; (2) after
   loading, every persisted tensor's hash is recomputed and compared
   against the manifest, catching a `load_state_dict` that silently
   dropped or mismatched a key. Verified directly by two new regression
   tests: a byte-corrupted checkpoint file, and a checkpoint file
   deliberately copied into the WRONG architecture's directory (a
   real, plausible operational mistake) -- both now raise instead of
   silently loading.
2. *"`shared_hash_groups` groups tensors by raw byte hash. This can group
   unrelated zero-initialized parameters together and is not a semantic
   comparison."* CONFIRMED -- a real, valid critique of §24's own design.
   Fixed by removing hash-collision inference entirely:
   `synchronize_shared_initialization`/`synchronize_four_architecture_initialization`
   now return an explicit `(target_name -> "source_model.source_name")`
   provenance mapping recording the EXACT copies actually performed;
   `persist_synchronized_initializations` accepts this mapping and
   writes it into the manifest as `shared_parameter_mapping` -- ground
   truth, never inferred after the fact. Verified by a regression test
   using two DELIBERATELY-coincidentally-zero biases that were NEVER
   synchronized with each other -- proving they no longer appear grouped
   together the way raw hash-collision grouping would have shown them.
3. *"The synchronizer also copies parameters only -- not shared buffers.
   `target_gene_scale`... is not synchronized."* CONFIRMED --
   `synchronize_shared_initialization` only ever iterated
   `named_parameters()`. Fixed at the source: it now iterates parameters
   AND buffers together, using the identical copy/verification logic for
   both. Verified by a unit test with a plain buffer-only module, and an
   integration test that deliberately perturbs Architecture 3's
   `transport_head.target_gene_scale` before synchronizing and confirms
   the two-hop sync reaches it (this buffer is currently all-ones by
   default for every architecture, so a before/after check needed a
   deliberate perturbation to be meaningful -- otherwise-identical
   defaults would make the test tautological).
4. *Manifest should store per-tensor shape/dtype, checkpoint file hash,
   and Architecture 4's gene-basis identity.* All added: every manifest
   tensor entry now carries `sha256`/`shape`/`dtype`; every architecture
   entry carries `weights_file_sha256`; Architecture 4's entry carries
   `gene_basis_hash` (from `gene_basis.gene_names_hash`) whenever the
   model has a `gene_basis` attribute. **Deliberately NOT added**:
   resolved-model-config hash and ordered-gene-name hash -- no real
   trainer exists yet to supply a resolved config or a real gene panel to
   hash, and a placeholder hash for data that doesn't exist yet would
   misrepresent verification strength rather than add it.

**"Training mask schedule semantics have edge cases" -- CONFIRMED, all
three sub-issues fixed.**

1. *`unique_mask_count` naming was misleading -- the audit's own worked
   example (4 strata, `unique_mask_count=1` still producing 4 distinct
   combinations) is correct and was reproduced exactly as a regression
   test before fixing.* Renamed to `unique_masks_per_stratum` throughout
   (parameter, docstring, stored field); behavior unchanged, only the
   name now states what the value actually controls.
2. *"The 'small n_items never starves a stratum' claim is also only true
   when `n_items >= number_of_strata`."* CONFIRMED -- the previous
   docstring's phrasing implied protection against chance that round-
   robin assignment can't actually provide once `n_items < n_strata`.
   Fixed: `build_stratified_training_seed_bank` now raises `ValueError`
   in that case instead of silently producing partial coverage a caller
   didn't ask for.
3. *"The training seed bank also fingerprints observation names but not
   coordinates or slice IDs."* CONFIRMED -- `coords3d`/`slice_ids` are
   now required parameters, fingerprinted via the same
   `mask_bank.spatial_fingerprint` `build_stratified_mask_bank` already
   uses, and validated on every `ensure_stratified_training_seed_bank`
   reload. Verified by a regression test: identical barcodes with
   coordinates moved keep the same `dataset_fingerprint` but get a
   DIFFERENT `spatial_fingerprint`, and reloading against moved
   coordinates now raises.

*"Record the mask-generation implementation/version in the eventual run
manifest"* -- deliberately deferred, not built now: this is explicitly
about a RUN manifest for a real training system that doesn't exist yet
(§21); adding a version tag with nothing yet to consume or compare it
against would be speculative, not a real fix.

**"Still honestly blocked" list -- re-confirmed accurate, no new
action.** Every item (no real trainer, no real HEST builder,
initialization artifacts/mask schedules not consumed by any training
process, no regional/global WSI context in Architectures 3/4, no
end-to-end leakage audit, no CUDA/AMP/memory gate, no harmonic caching)
is exactly what §21/§23/§24 already say. The audit's own closing
instruction -- "after that, stop expanding infrastructure helpers and
build the real data builder and trainer" -- is noted for whenever Adam
next directs work to continue; this pass fixed the specific findings
raised, it did not start that larger, separate body of work.

## 26. Response to the fifth external Codex re-audit (of commit c02a5d1)

Adam forwarded a fifth re-audit, dramatically larger than the first four:
13 numbered launch-blocking findings, a "genuinely fixed" list, a
16-step implementation order, and 10 required gates before any 24-hour
run. Verdict: "the system is still not ready for a long run... the
real builder, trainer, WSI wiring, and held-out evaluator still do not
exist." Same discipline as every prior round: every claim checked
against the actual code before any fix; nothing accepted on faith.

**"Genuinely fixed" list -- spot-checked, all ten items hold up.**
Rather than re-deriving each from scratch, the two most checkable
claims were verified directly against code: "Query GEX and query H&E
are not present in the model-forward schema" -- confirmed,
`SpatialFieldInputs`'s own docstring states this and
`validate_spatial_field_example` enforces it structurally, not by
convention. "Patient-disjoint splitting and cross-organ patient
conflict handling are sensible" -- confirmed,
`hest1k_catalog.py::_resolve_cross_organ_patient_conflicts` exists and
does exactly what its name says. The other eight items in the list
describe fixes from §22-§25 that this document already verified in
their own rounds; no new action needed on any of them.

**Finding #1, "the training mask diversity fix is still wrong" --
CONFIRMED AND FIXED.** The exact bug the audit describes: `seed =
unique_seeds[i % unique_masks_per_stratum] + stratum_offset` only
visits seed indices congruent to `i`'s own value modulo
`unique_masks_per_stratum`'s relationship to `n_strata` -- with 4
strata and `unique_masks_per_stratum=64`, each stratum actually only
ever saw 16 unique seeds, not 64. Reproduced computationally before
fixing (`n_strata=4, unique_masks_per_stratum=64, n_items=256` ->
16 unique seeds/stratum, confirmed by direct calculation). Fixed with
the audit's own corrected formula: `occurrence_in_stratum = i //
n_strata` (a genuine per-stratum visit counter, independent of
`n_strata`) instead of `i % unique_masks_per_stratum`. Also added, as
specified: `unique_masks_per_stratum >= _STRATUM_SEED_STRIDE` now
raises (was previously unchecked), and the exact regression test the
audit specified --
`test_build_stratified_training_seed_bank_realizes_the_full_promised_unique_seed_pool_per_stratum`
-- asserting 4 strata x 64 unique masks x 256 items yields exactly 64
unique seeds per stratum and exactly 256 unique `(stratum, seed)`
combinations. The `unique_masks_per_stratum > n_items` check (no
longer a meaningful constraint under the corrected formula) was
removed.

**Finding #2, "initialization loading is not genuinely fail-closed" --
CONFIRMED, all three sub-issues fixed.**

1. *`load_synchronized_initialization()`'s `manifest=None`,
   `architecture_name=None` defaults let verification be silently
   skipped by omission.* CONFIRMED -- this is a real, fair correction
   of §25's own overclaim: §25 called this function "genuinely
   fail-closed," but fail-closed behavior that's opt-in by keyword
   argument is not actually fail-closed. Fixed: both parameters are now
   required (no defaults); the old unverified behavior moved to a
   separately-named `load_ad_hoc_checkpoint_unverified()`, so skipping
   verification now requires calling a function whose name says so,
   not merely omitting two keyword arguments. Verified by
   `test_load_synchronized_initialization_requires_a_manifest_and_architecture_name`,
   which inspects the function's signature directly (via
   `inspect.signature`) to confirm neither parameter has a default, and
   `test_load_ad_hoc_checkpoint_unverified_is_the_only_way_to_skip_verification`.
2. *`gene_basis_hash` is misleadingly named -- it only ever held
   `gene_names_hash` (panel identity), not a hash describing how the
   basis was FITTED (rank, residual source, seed).* CONFIRMED --
   accurate; `GeneResidualBasis` doesn't currently store any
   fitting-provenance metadata at all, so no field could have described
   it. Fixed by renaming the manifest field to
   `gene_basis_gene_names_hash` and documenting precisely what it does
   and does not cover (the basis matrix's own numeric content is
   separately covered by `tensor_hashes`, since `_gene_basis_matrix` is
   a registered buffer and gets hashed like any other persisted
   tensor). Fitting-provenance metadata itself (rank, residual source,
   fitting sample IDs) is not added here -- `GeneResidualBasis` isn't
   fit by any code in this repository yet, so there is nothing real to
   record; adding placeholder provenance fields would misrepresent
   verification strength rather than add it, the same reasoning §25
   already applied to the resolved-config-hash question.
3. *`persist_synchronized_initializations()` accepts unsynchronized
   models and an absent synchronization mapping despite what its name
   implies.* CONFIRMED. Rather than tightening the general low-level
   function (still needed in its permissive form by its own toy-module
   unit tests), added a new strict wrapper,
   `persist_four_architecture_initializations()`, that: requires
   exactly the four `architectureN` keys; always calls
   `synchronize_four_architecture_initialization` itself (never trusts
   a caller-supplied mapping); verifies every mapped tensor pair is
   `torch.equal` before persisting anything, raising `ValueError`
   otherwise. Verified by
   `test_persist_four_architecture_initializations_refuses_to_persist_an_inconsistent_state`,
   which monkeypatches in a "lying" synchronizer that claims a sharing
   relationship without performing any copy, and confirms persistence
   is refused.

The additional manifest fields the audit lists (resolved effective
model configuration hash, training cohort/split hash,
normalization/QC specification hash, GigaPath checkpoint SHA256, code
commit, mask-generation version, training/validation/test mask-bank
hashes) are NOT added in this pass -- every one of them describes state
belonging to a real trainer/data builder that does not exist yet (a
resolved config, a real cohort split, a real normalization fit, a real
mask bank). Adding placeholder fields for data nothing yet produces
would be exactly the kind of "looks more verified than it is" gap this
whole audit round is about avoiding. This is deferred to whenever the
real training system (§21) is built, not silently dropped.

**Finding #3, "there is still no real data builder or trainer" --
CONFIRMED, ALREADY DISCLOSED.** Unchanged from §21/§23/§24/§25's own
repeated statement of the same fact. The list of things this leaves
unproven (query leakage, physical overlap removal, patient
disjointness in an actual run, training-only normalization, four
subprocesses actually loading synchronized initialization, checkpoint
resume correctness, AMP/OOM behavior) is accurate and is exactly what
those sections already say is blocked on the same missing system.

**Finding #4, "Architectures 3 and 4 still do not implement their
defining WSI context" -- CONFIRMED, ALREADY DISCLOSED.** Checked
directly: both `use_regional_he` and `use_global_slide` still raise
`NotImplementedError` when set `True` (§17/§20 already documented this
exact gap, including the observation that Architecture 3 currently
means "Architecture 1 plus global observed-GEX inducing tokens," not
the full hierarchical regional-H&E-plus-WSI-LongNet design the handoff
specifies).

**Finding #5, "physical image masking is incomplete" -- CONFIRMED,
DEFERRED.** Checked directly: `SpatialFieldInputs` has no
`image_available` field and nothing currently zeroes an observed
patch's image features or excludes WSI tiles intersecting a hole --
because nothing yet BUILDS a `SpatialFieldInputs` from real masked
data at all (§19 already lists "physical hole-overlap removal for
observed image patches" as not covered). This is real, but it is a
per-example DATA BUILDER concern -- there is no builder to add this
logic to yet. Implementing it in isolation, with no real hole geometry
or WSI tile grid to test it against, would produce untested code with
no way to verify it's actually correct. Deferred to when the real
builder (implementation-order step 8, which the audit itself groups
this under) is built.

**Finding #6, "'only completely new masks and no same spots' is not
enforced yet" -- CONFIRMED, ALREADY DISCLOSED.** Checked directly:
`mask_bank.py::query_overlap_report`'s own docstring states plainly
that "query spots are NOT required or verified to be disjoint across
masks" -- it reports overlap, it does not enforce disjointness, exactly
as the audit describes. The stricter composite `(sample_id, barcode)`
identity policy, and failing the launch on any cross-split identity
collision, both require an actual train/validation/test split and
actual per-split mask banks to check identities against -- i.e. the
same missing real data builder/trainer as findings #3-#5. Tracked here,
not silently ignored, but not buildable as an isolated unit today.

**Finding #7, "the geometry graph can create biologically false edges"
-- CONFIRMED, DEFERRED (new finding, substantial redesign).** Checked
directly: `build_knn_adjacency()` always returns a fixed-k, directed
nearest-neighbor graph over raw coordinates -- no lattice-adjacency
detection, no symmetry enforcement, no distance-based edge rejection,
no diagnostic reporting of edge-length quantiles or connected
components. This is a real, previously-unraised gap: near tissue gaps
or disconnected fragments, a spot's 6 "nearest" neighbors can include
ones across empty space simply because they're the closest points
available. Not fixed in this pass -- this is a genuine boundary-graph
REDESIGN (symmetric radius graph, lattice-adjacency detection,
distance-bounded edge rejection, relative-to-hole coordinate units,
new diagnostic fields), not a contained bug fix, and it changes what
`extract_boundary_and_local_context` returns for every architecture
that consumes it. Given how much this single round has already
changed, redesigning the graph construction underlying every
architecture's local context deserves explicit sign-off before
starting, not a unilateral change buried in an audit-response pass.

**Finding #8, "the WSI cache key remains unsafe" -- CONFIRMED, ALREADY
DISCLOSED, re-confirmed not new.** Checked directly against
`models/slide_encoder.py`'s `_tensor_digest(coords, namespace)` --
confirmed accurate, it hashes only tile coordinates and a namespace
string, never tile-feature values, checkpoint identity, or
preprocessing version. This is the exact same finding §23 already
recorded in response to the second audit ("Cache signature scheme...
not yet implemented as gen3-specific code... only wiring once there's
a real cache to protect") -- re-confirmed true, still deferred to when
a real cache actually needs protecting, i.e. when the real trainer
exists to define what "preprocessing version" and "exact visible mask"
even mean in production.

**Finding #9, "the 'global GEX' branch is actually multimodal" --
CONFIRMED, DEFERRED (new finding, design decision).** Checked directly:
`_SharedFieldArchitecture.forward()` calls `self.gex_pool(observed_tokens,
...)`, and `observed_tokens` (built by `_observed_tokens()`) already
fuses GEX, H&E, coordinates, and ring-embedding information before the
pool ever sees it -- so `InducedGlobalGEXPool`'s attention weights are
provably influenced by non-GEX signal, exactly as the audit says. This
is accurate and is a genuine, previously-unraised architectural point,
not a bug in the sense of violating a stated contract (the handoff
never specifies the global pool must be GEX-ONLY input, only that it
produces a "global GEX" summary). Giving it a dedicated GEX-only token
projection (optionally with relative coordinates, no H&E) is a real
design change affecting Architecture 3/4's actual learned behavior --
left as an explicit open decision for Adam, the same treatment §22
already gave the "shared residual disabled in all four configs"
finding, rather than resolved unilaterally in either direction.

**Finding #10, "candidate duplication biases transport" -- CONFIRMED,
ALREADY DISCLOSED.** This is exactly what §15/§17 already call out as
"Transport candidate pool is concatenated, not deduplicated... a
query's local neighbors are NOT excluded from the boundary set, so a
close spot can appear twice (once as a local candidate, once as a
boundary candidate)... proper deduplication needs per-query masked
attention, a real follow-up" -- re-confirmed true by re-reading
`_candidate_pool()`/`forward()` directly, not newly discovered. The
audit's suggested fix (exclude boundary members from local-only
candidates, pad, add a candidate mask to the transport softmax) is a
reasonable design for that follow-up but is not implemented here,
consistent with §15's own framing of this as already-scoped future
work rather than a silent gap.

**Finding #11, "the spatial-gradient loss uses target-dependent
scaling" -- CONFIRMED, ALREADY DISCLOSED.** Checked directly:
`spatial_gradient_loss`'s own docstring already states this exactly --
"When None (the common case until a real data builder supplies a
training-fit scale), this function falls back to the per-gene std of
target_expression WITHIN THIS CALL -- a documented simplification...
not a claim that this is the training-set scale." The function already
accepts an optional `per_gene_scale` parameter specifically so a real
trainer can pass a training-fit scale once one exists; no caller does
so today because no trainer exists to fit one. No code change needed;
this is the same "training-only fitting must happen outside this
function" limitation §18 already documents for `harmonic.py` and
`target_gene_scale`.

**Finding #12, "Architecture 4 needs a better training contract" --
CONFIRMED, PARTIALLY FIXED.**

1. *"The conditioner is evaluated once for reconstruction and again
   inside `compute_flow_matching_loss()`. With dropout enabled, the two
   passes are not identical."* CONFIRMED -- verified by reading
   `forward()` and `compute_flow_matching_loss()`, which independently
   call `self.conditioner(inputs)`; every config in this repository
   sets `dropout` > 0, so two separate calls in one training step would
   draw different dropout masks. Fixed: added `Architecture4.compute_losses()`,
   which runs `self.conditioner(inputs)` exactly ONCE and derives both
   the deterministic output and the flow-matching loss from that single
   pass. `forward()` and `compute_flow_matching_loss()` are kept
   unchanged (not merged or deprecated) for callers that genuinely only
   need one or the other, and for the existing tests exercising them
   independently -- `compute_losses()` is the one a real trainer
   computing both losses per step should call. Verified by
   `test_architecture_4_compute_losses_runs_the_conditioner_exactly_once_and_stays_consistent`,
   which puts the model in `eval()` (so dropout becomes a no-op and the
   three call patterns are directly comparable) and confirms
   `compute_losses()`'s output matches calling `forward()` and
   `compute_flow_matching_loss()` separately.
2. *"`predictive_std` uses the default unbiased estimator and becomes
   NaN with one sample."* CONFIRMED -- reproduced directly
   (`torch.randn(1, 5).std(dim=0)` returns all-NaN with a
   degrees-of-freedom UserWarning). Fixed: `predictive_std` now uses
   `std(dim=0, unbiased=False)`, which reports the well-defined
   population std of exactly 0 for a single sample instead of NaN.
   Verified by
   `test_architecture_4_predictive_std_is_finite_zero_not_nan_for_a_single_sample`.
3. *"Flow coefficient scales are not standardized," "a residual basis
   fitted on Architecture 3 residuals implies Architecture 3 must
   already be trained, preventing four clean parallel starts," "add an
   explicit `flow_weight` to the config."* CONFIRMED, NOT FIXED, blocked
   on the same missing real trainer/data builder as findings #3-#6:
   standardizing coefficients requires a training-only coefficient
   mean/std fit; fitting the basis on training-only centered expression
   variation (rather than Architecture-3 residuals) is a real,
   reasonable redesign of how `GeneResidualBasis` gets fit, but there is
   currently no code anywhere that fits one at all outside tests, so
   there's no real fitting pipeline to change yet; `flow_weight` belongs
   in a training config/loop that doesn't exist. Noted for whoever
   builds the real basis-fitting step and trainer, not silently dropped.

**Finding #13, "evaluation aggregation is not correct for PCC yet" --
CONFIRMED, DEFERRED (needs the real evaluator).** Checked directly:
`aggregate_patient_metrics()`'s signature is
`(per_item_metrics: list[dict[str, float]], patient_ids: list[str])`
-- it receives ALREADY-COMPUTED per-item metric values and macro-averages
them within and across patients. It has no access to raw
predictions/targets at all, so it structurally cannot do the
statistically correct thing the audit describes (concatenate raw
predictions within a patient, then compute PCC once over the pooled
set -- PCC is nonlinear, so averaging per-item PCCs is not equivalent).
Fixing this correctly means a different function, one that collects
raw `(patient, sample, barcode, mask, stratum)`-keyed predictions,
deduplicates repeated spots, and computes per-gene PCC/RMSE from
concatenated raw values per patient -- which requires an actual
evaluator collecting real predictions across real held-out items, i.e.
the same missing real trainer/evaluator as finding #3. Not built here;
`aggregate_patient_metrics` is left as-is (still useful for its
current, narrower contract) rather than partially reworked toward a
signature nothing yet calls correctly.

**Step-by-step implementation order and required gates -- read and
acknowledged, not executed.** The audit's own framing is explicit:
"Implement in this exact order and stop expanding unrelated helper
infrastructure... The next correct step is finishing and auditing the
data/training protocol -- not another architecture sweep." This is the
first audit round to explicitly instruct a change in the KIND of work
that should happen next, not just more fixes within the current scope.
This pass fixed everything in the 13 findings that was genuinely
fixable in isolation (findings #1, #2, and parts of #12) and gave every
remaining finding an explicit, checked verdict rather than silence. It
did NOT start building `gen3_multiscale.training.train`, a real
`Gen3DataBundle`/HEST data builder, the WSI/regional-H&E wiring, the
geometry-graph redesign, the GEX-pool purity redesign, candidate
deduplication, basis standardization, or the raw-prediction evaluator
-- each of those is a substantial, independent body of work (the
audit's own 16-step order spans them across 6 more steps after this
one), and starting any of them unilaterally, in the middle of an
audit-response pass, would be exactly the kind of scope creep the
audit's own closing line warns against. This is surfaced back to Adam
as the explicit decision point it is, not resolved unilaterally in
either direction.

**Unrelated, pre-existing test failure noticed while running the full
suite for this pass: `tests/test_multi_sample.py::test_inject_multi_sample_n_genes`
fails on this exact commit (`c02a5d1`), before any change in this
section.** Confirmed via `git stash` -- the failure reproduces
identically with none of this round's changes applied. This lives in
the top-level `tests/` directory (not `gen2_architectures/` or
`gen3_multiscale/`) and concerns `inject_multi_sample_n_genes`, code
this session has never touched. Out of scope for a `gen3_multiscale/`
audit-response pass; noted here rather than silently passed over, for
whoever next works in that area.

**No 24-hour run has been started or will be auto-started. This
document only records fixes to already-written code and tests.**

## 27. Response to the sixth external Codex re-audit (of commit 06f5cce)

Adam forwarded a sixth re-audit, of `06f5cce` (§26's fixes), this time
verified against the real commit directly (`git show`/`git diff`
against `c02a5d1`, extracted into a local checkout) rather than only
against a pasted summary. Verdict: "the fixes are real... but the code
is still not ready for the four 24-hour runs," with 4 "remaining
contained bugs" to fix before starting the real trainer, plus
re-confirmation of the larger launch blockers. Same discipline as every
prior round: every claim checked against the actual code before any
fix.

**"Verified fixes" list -- all eight items independently re-confirmed
true** by re-reading the relevant code paths directly (occurrence-based
seed indexing, mandatory manifest/architecture_name,
`load_ad_hoc_checkpoint_unverified`, `gene_basis_gene_names_hash`,
`persist_four_architecture_initializations`'s pre-save verification,
`Architecture4.compute_losses()`, `predictive_std`'s `unbiased=False`,
launcher CLI validation). No corrections needed to this list.

**Remaining contained bug #1, "seed-pool guarantee is still overstated"
-- CONFIRMED, FIXED.** The audit's specific complaint is real:
`build_stratified_training_seed_bank` never enforced
`n_items >= n_strata * unique_masks_per_stratum`, so a caller could ask
for e.g. 99 unique masks per stratum with only 4 total items and
silently get 1. It also correctly flagged that the previous default
(`unique_masks_per_stratum = n_items`) can never be exhausted once
`n_strata > 1`, since each stratum is only visited `n_items // n_strata`
times.

This finding is in real tension with a DELIBERATE decision from §25 (in
response to the 5th audit's OWN earlier recommendation): the 5th audit
asked for the opposite of what the 6th now asks for -- it said the
4th round's rejection of `unique_masks_per_stratum > n_items` was based
on "an incorrect mental model (a global item budget)" and should be
REMOVED, which §25 did, with a regression test
(`test_build_stratified_training_seed_bank_allows_unique_masks_per_stratum_larger_than_n_items`)
locking in the permissive behavior. Silently flipping that back now
would contradict a previous deliberate decision the moment a new audit
happens to prefer the opposite default, which is not a defensible way
to resolve genuinely competing designs.

Resolution: BOTH audits' legitimate concerns are real, so both are
served without re-reversing either one. `build_stratified_training_seed_bank`
now always reports `realized_unique_seeds_per_stratum` (a dict of
stratum -> actual count of distinct seeds realized) in its output, so
any caller can verify their coverage intent was met without redoing the
`n_items`/`n_strata` arithmetic themselves. A new optional
`require_full_seed_pool: bool = False` parameter (threaded through
`ensure_stratified_training_seed_bank` too) turns the 6th audit's exact
recommended check (`n_items >= n_strata * unique_masks_per_stratum`,
verified against the ACTUAL realized counts, which correctly accounts
for the off-by-one stratum split when `n_items` isn't evenly divisible
by `n_strata`) into an opt-in, fail-closed precondition. The permissive
DEFAULT (`False`) is unchanged, preserving §25's regression test and the
5th audit's own reasoning; a caller that genuinely needs the guarantee
(e.g. "validation must show exactly N distinct masks per stratum") now
has an explicit way to demand it. Verified by three new regression
tests: `realized_unique_seeds_per_stratum`'s value in the
under-provisioned case, `require_full_seed_pool=True` rejecting that
same under-provisioned case, and `require_full_seed_pool=True`
succeeding on the audit's own 4-strata/64-unique/256-item example.

**Remaining contained bug #2, "checkpoint loading still uses Python
`assert`" -- CONFIRMED, FIXED.** `checkpoint.py::load_trainable_state`
used two bare `assert` statements for its fail-closed config-mismatch/
incomplete-checkpoint checks -- `assert` is compiled out entirely under
`python -O`/`PYTHONOPTIMIZE`, silently turning fail-closed validation
into a no-op in that mode. Both replaced with explicit `RuntimeError`.
Per this file's own copy-provenance discipline ("If a bug is found
here, check whether gen2_architectures' copy has the same issue"),
checked `gen2_architectures/training/checkpoint.py` directly -- it has
the identical `assert` statements -- and applied the identical fix
there too, keeping the two copies in sync. While fixing this, also
found and fixed an independent, pre-existing bug in BOTH copies' own
test (`test_load_trainable_state_raises_on_genuine_mismatch`): it used
`try: ... assert False, "expected an AssertionError" \n except
AssertionError: pass`, which could never actually fail the test even if
`load_trainable_state` stopped raising anything at all, since the
test's OWN `assert False` fallback raises the same `AssertionError`
type the `except` block swallows. Replaced with `pytest.raises(RuntimeError,
match=...)` in both copies.

Additionally, the audit's "verify manifest tensor shape/dtype
explicitly, not only its byte hash" sub-point is CONFIRMED AND FIXED in
`model_factory.py::load_synchronized_initialization`:
`_tensor_hash` hashes raw tensor bytes only
(`tensor.numpy().tobytes()`), which does not encode shape -- two
tensors with the same total byte content but different shapes (e.g. a
`[2,3]` and a `[3,2]` all-zeros buffer, a realistic case for
zero-initialized residuals/scales) hash identically. Shape/dtype were
already recorded in the manifest by `persist_synchronized_initializations`
but never checked on load. Both are now verified explicitly before the
hash check. (In practice, the on-disk `load_trainable_state` path this
feeds is itself protected by `load_state_dict`'s own shape enforcement
on matched keys, so this specific check is defense-in-depth against a
hand-edited/corrupted manifest or a future direct hash-comparison
caller bypassing `load_state_dict` -- worth having since the data was
already computed and sitting unused in the manifest, not because a
concrete exploit path through the current loader exists today.)
Verified by two new regression tests that mutate a real manifest's
`shape`/`dtype` fields and confirm both are rejected.

**Remaining contained bug #3, "launcher validation is only in `main()`"
-- CONFIRMED, FIXED.** Duplicate-GPU and thread-count validation
previously lived only in the CLI entrypoint; every test in this file,
and any real Python caller, invokes `launch_suite()` directly and
bypassed both checks entirely. Both are now enforced inside
`launch_suite()` itself (the actual invariant-enforcing location every
caller goes through); `main()` keeps its own early check too, since
failing before even loading config YAML files is still a real, harmless
optimization for the CLI path specifically. Also fixed the audit's
second half of this finding: a `Popen` call failing partway through the
spawn loop (e.g. a `command_builder` producing a command naming a
nonexistent executable) used to leave every earlier-started subprocess
running unmonitored, with every already-opened log file handle leaked.
The spawn loop is now wrapped in `try/except`: on any exception, every
already-started process is `terminate()`d and `wait()`ed, and every
already-opened log handle is closed, before the original exception is
re-raised. Verified by three new regression tests: duplicate GPU ids
rejected by `launch_suite()` directly, non-positive `threads_per_job`
rejected the same way, and (using a monkeypatched fake `subprocess.Popen`
to avoid real-process-table timing races) a mid-loop spawn failure
confirmed to `terminate()`+`wait()` every already-started fake process
and leave every opened log file closed.

**Remaining contained bug #4, "Architecture 4's safe training path
remains optional" -- PARTIALLY CONFIRMED, PARTIALLY FIXED, one part
DISPUTED.**

- *"The old two-pass combination remains callable... the future
  trainer must exclusively use `compute_losses()`."* CONFIRMED as a
  factual description, but NOT changed -- this is exactly §26's own
  explicit, deliberate design choice, not an oversight: `forward()`/
  `compute_flow_matching_loss()` were kept unchanged specifically so
  existing callers/tests that only need one loss aren't forced through
  a two-loss interface. "The future trainer must exclusively use
  `compute_losses()`" is a constraint on a training LOOP that doesn't
  exist yet (§21); there is nothing in `gen3_multiscale/` today that
  calls `forward()` and `compute_flow_matching_loss()` together in one
  step for it to be wrong on. "Add an integration test at trainer level
  that counts conditioner calls during a real optimizer step" is
  correctly identified by the audit itself as belonging to the real
  trainer, which doesn't exist -- noted for whoever builds it, not
  fixable in isolation today.
- *"Both flow-loss methods should also move and validate
  target_expression against the model's actual device/dtype and check
  shape/finiteness."* CONFIRMED AND FIXED. Added
  `Architecture4._prepare_target_expression()`, called from both
  `compute_flow_matching_loss()` and `compute_losses()`: moves the
  target onto the conditioner's own device/dtype, then raises
  `ValueError` on a shape mismatch or any non-finite value, instead of
  silently broadcasting a wrong shape or letting NaN/Inf poison the flow
  loss. Verified by three new regression tests: a shape-mismatched
  target rejected, a non-finite target rejected, and a float64 target
  (a realistic case from raw numpy/anndata conversion) accepted and
  correctly moved onto the model's own dtype rather than rejected.

**Real launch blockers #1-4 (no data builder/trainer, WSI context, image
masking, mask-novelty enforcement) -- re-confirmed accurate, no new
action; unchanged from §21/§23/§24/§26.**

**Real launch blocker #5, "the static fairness audit is too weak" --
CONFIRMED, PARTIALLY FIXED.** Checked directly: `static_config_audit`'s
previous `shared_keys = set.intersection(*(...))` computed the
intersection across ALL configs being compared. Since
`architecture4.yaml` genuinely and deliberately lacks
`model.params.use_regional_he`/`use_global_slide` entirely (Architecture
4 wraps a full Architecture 3 conditioner rather than accepting those
kwargs directly -- documented in the config's own header), those keys
dropped out of the FOUR-WAY intersection entirely -- which meant they
were never checked even AMONG architecture1/2/3.yaml, which genuinely do
all three share them. A real, undocumented divergence between just
those three would have gone completely unflagged, purely because a
fourth, structurally different config didn't have the key at all --
exactly the concrete failure mode the audit describes.

Fixed the specific, contained part of this: `static_config_audit` now
compares each key among whichever configs actually declare it (at least
two), rather than requiring presence in literally every config passed
in. A key genuinely unique to one config (Architecture 4's flow-only
params) is still never compared -- that structural difference is not
itself a violation -- but a key shared by a SUBSET of the configs is no
longer silently exempted from comparison just because some OTHER config
in the batch happens to lack it. Verified by a new regression test:
4 configs, 3 of which share a field with an undocumented divergence, the
4th genuinely lacking the field entirely -- the divergence is now
caught, with the non-participating 4th config correctly excluded from
the reported violation's `values`.

NOT built (the larger part of this finding, genuinely out of scope for
a contained fix): a canonical "effective experiment schema" that
explicitly maps Architecture 4's nested conditioner fields onto
Architecture 3's flat ones, so the "these fields should agree" set is
asserted positively rather than inferred from which keys happen to
co-occur across the configs actually passed in. This is a real,
reasonable design (it would additionally catch e.g. Architecture 4
silently having a DIFFERENT effective `use_regional_he` than the
Architecture 3 conditioner it wraps, which the current per-key
comparison still cannot see since Architecture 4's config has no such
key at all), but requires deciding what the canonical schema even
contains -- a real design question, not a mechanical fix, and one this
already-large round did not take on unilaterally.

**Real launch blocker #6, "'fingerprints' only check path existence" --
CONFIRMED, ALREADY DISCLOSED.** Checked directly:
`check_required_fingerprints` only calls `Path(path).exists()`; it does
not hash file contents. This is the same underlying gap §26 already
covered for the initialization manifest specifically (resolved-config
hash, cohort/split hash, mask-bank hashes, etc. -- "every one of them
describes state belonging to a real trainer/data builder that does not
exist yet... adding placeholder fields for data nothing yet produces
would misrepresent verification strength"). The launcher's
`required_fingerprints` block is the same class of not-yet-real
artifact for the same reason -- there is no real gene vocabulary, split,
or mask bank on disk yet for a content hash to protect, since the real
data builder that would produce them doesn't exist. Re-confirmed true,
not newly discovered, still blocked on the same missing system.

**Architectural issues (geometry graph, global-GEX purity, candidate
duplication, flow training, PCC aggregation, Novae) -- re-confirmed
accurate, no new action; unchanged from §26's findings #7/#9/#10/#12/#13.**
The audit's new "Novae" section is a restatement, not a new claim:
`gen3_multiscale` genuinely has no Novae implementation, uses
`weighted_linear` instead (§10's recorded, deliberate choice), and the
"initially establish one clean baseline, then run weighted-linear versus
Novae as a controlled encoder ablation" recommendation matches §10's own
framing already. Left as the explicit open decision it already was, not
resolved unilaterally here.

**"Exact implementation order for Claude" -- read and acknowledged, not
executed**, for the identical reason §26 gave for the 5th audit's
16-step order: starting `Gen3DataBundle`, the real trainer, WSI wiring,
the geometry-graph redesign, or the evaluator unilaterally mid-audit-
response would be scope creep the audit's own framing warns against.
This round did complete every step of the order that was genuinely a
contained fix (steps 1-4 of the 18-step list); step 5 onward remains the
explicit decision point already surfaced to Adam in §26, unchanged by
this round.

**No 24-hour run has been started or will be auto-started. This
document only records fixes to already-written code and tests.**

## 28. Response to the seventh external Codex re-audit (of commit 2782ff0)

Adam forwarded a seventh re-audit, of `2782ff0` (§27's fixes), again
verified directly against the real commit (diffed against `06f5cce` in a
local checkout). Verdict: "the four contained fixes are real
improvements," with 4 additional contained issues found before the real
trainer should be built. Same discipline as every prior round: every
claim checked against the actual code before any fix.

**"Verified fixes" list -- all six items independently re-confirmed
true.** No corrections needed.

**Issue #1, "seed-pool enforcement is not yet experiment-safe" --
MULTIPLE SUB-CLAIMS, SPLIT VERDICT.**

1. *"It verifies unique random seeds, not unique realized query masks.
   Different seeds can still generate the same masked spots."*
   CONFIRMED, DEFERRED (structurally out of scope for this module).
   `build_stratified_training_seed_bank`'s own docstring is explicit
   about this design: it produces a "lossless, deterministic (stratum,
   seed) schedule" specifically so it never has to materialize actual
   per-item masks -- its docstring cites `build_training_seed_bank`'s
   own reasoning for why NOT doing so avoids "the exact multi-gigabyte-
   JSON problem." Checking that two different seeds produce genuinely
   DIFFERENT realized query-spot sets requires actually running the
   masking algorithm against real coordinates for every seed, which
   this module deliberately does not do (and, per that avoided-JSON-
   size reasoning, should not do). This can only be checked once the
   real per-sample data builder (§21) exists to run the masking
   algorithm and inspect its output; verifying it here would mean either
   materializing every mask (defeating the module's whole design) or
   adding a check with no real masking function to check against yet.
2. *"The mask-bank schema changed, but `_MASK_GENERATION_VERSION`
   remains '2'... an older bank without
   `realized_unique_seeds_per_stratum` can therefore be silently
   reused."* CONFIRMED, FIXED, but NOT via a version bump.
   `_MASK_GENERATION_VERSION` is reserved for changes to the actual
   (stratum, seed) GENERATION ALGORITHM (per its own docstring) -- this
   round's change added a new DERIVED REPORTING field without changing
   what seeds get generated for the same inputs, so bumping it would
   conflate two genuinely different kinds of change and cause an
   on-disk bank to be needlessly regenerated for a schema-only reason.
   The actual practical problem the audit describes is real, though:
   `ensure_stratified_training_seed_bank` returned the raw on-disk JSON
   on a validated reuse, which could predate this round's field
   addition. Fixed at the correct layer: once every identifying field
   and the exact `items` sequence are confirmed identical between the
   on-disk bank and a fresh recomputation, the two are semantically
   equivalent by definition, so the function now returns the freshly-
   recomputed (always schema-current) dict instead of the possibly-
   stale on-disk one. Verified by a regression test that writes a bank,
   deletes `realized_unique_seeds_per_stratum` from the on-disk file to
   simulate an older artifact, and confirms a subsequent
   `ensure_stratified_training_seed_bank` call backfills the field while
   returning the exact same `items` schedule.
3. *"Strict mode combined with the default `unique_masks_per_stratum=
   n_items` is mathematically impossible when there is more than one
   stratum."* CONFIRMED AND FIXED. Verified directly: with the previous
   default, `require_full_seed_pool=True` would raise UNCONDITIONALLY
   for any `n_strata > 1` if a caller left `unique_masks_per_stratum`
   unset -- a guaranteed-to-fail footgun, not a real precondition, since
   a stratum can only ever be visited `n_items // n_strata` times, which
   is always less than `n_items` once there's more than one stratum.
   Fixed: in strict mode, an unset `unique_masks_per_stratum` now
   defaults to the naturally achievable ceiling (`n_items // n_strata`)
   instead of `n_items`, so the strict default trivially satisfies its
   own guarantee by construction. The permissive (non-strict) default of
   `n_items` is unchanged. Verified by a new regression test.
4. *"Before training, the builder must: require an explicit feasible
   mask count per stratum; enable strict seed-pool checking; fingerprint
   the actual realized query-spot sets; reject duplicate masks and
   train/validation/test query overlap; increment the mask-bank version
   and reject stale banks."* CONFIRMED, ALREADY DISCLOSED/DEFERRED. This
   is a checklist for the real data builder/trainer (§21), not a gap in
   this module -- items 1-3 above already give that future builder the
   TOOLS it would need (`realized_unique_seeds_per_stratum`,
   `require_full_seed_pool`, a version field that means what it says);
   items about duplicate-mask/split-overlap rejection are the same
   still-missing composite-identity enforcement §26 finding #6 and §27
   already cover.

**Issue #2, "Architecture 4 gene identity is persisted but not
verified" -- CONFIRMED AND FIXED.** Checked directly:
`load_synchronized_initialization` recorded and could read
`gene_basis_gene_names_hash` from the manifest but never compared it
against anything. The reason this specific check matters, verified by
reading `checkpoint.load_trainable_state`: `_gene_basis_matrix` (the
basis MATRIX's numeric content) is a registered buffer, so
`load_state_dict` OVERWRITES it with the checkpoint's own values
regardless of what the freshly-constructed model computed -- the tensor-
hash check the loader already does will therefore always pass for the
matrix itself. The separate, plain-dataclass `gene_basis` ATTRIBUTE
(holding `gene_names_hash`, i.e. which genes/order the matrix's numbers
apply to) is not part of any tensor state_dict and is never touched by
loading -- it silently keeps reflecting whatever gene_names the fresh
model happened to be CONSTRUCTED with. A model built against a permuted
gene order (same rank, same n_genes, so construction itself succeeds)
would previously pass every existing check despite its basis matrix now
meaning something different than its own metadata claims. Fixed: added
an explicit comparison between the freshly-constructed model's
`gene_basis.gene_names_hash` and the manifest's
`gene_basis_gene_names_hash` whenever both are present. Verified by the
audit's own suggested regression test: persist Architecture 4 normally,
then attempt to load onto a fresh Architecture 4 built with a genuinely
PERMUTED gene order (same rank/n_genes) -- now rejected with an explicit
"gene panel/order mismatch" error instead of silently loading.

**Issue #3, "Architecture 4's target API is inconsistent" -- CONFIRMED
AND FIXED.** Checked directly: `_prepare_target_expression` (added in
§27 for a different, real gap) called `target_expression.to(...)`, which
assumes a `torch.Tensor`. `SpatialFieldTargets.query_expression` --
documented and typed as `np.ndarray` throughout `data/example.py`, the
NATURAL source of this argument for any real caller -- has no `.to()`
method; a trainer passing `targets.query_expression` directly, exactly
as the schema describes, would have hit an `AttributeError` instead of
the validation this method exists to provide. Fixed: switched to
`torch.as_tensor(target_expression, device=device,
dtype=deterministic_mean.dtype)`, which accepts both a raw numpy array
and an existing tensor uniformly; both call sites' type hints updated to
`torch.Tensor | np.ndarray`. Verified by a new regression test that
passes `targets.query_expression` (confirmed via `isinstance` to
genuinely be a raw `np.ndarray`, not a tensor) directly into
`compute_flow_matching_loss` and confirms it now succeeds.

**Issue #4, "launcher cleanup is not interruption-safe" -- CONFIRMED AND
FIXED, all four sub-points.** Checked directly against `launch_suite`'s
§27 cleanup code:

1. *"Not `KeyboardInterrupt` or `SystemExit`."* CONFIRMED -- the
   handler was `except Exception`, and both inherit from `BaseException`
   directly, not `Exception`. Changed to `except BaseException`.
2. *"Not interruption during the waiting phase."* CONFIRMED -- the
   `try/except` only wrapped the SPAWN loop; the WAIT loop (where a
   real, hours-long run spends nearly all its time) had no cleanup
   coverage of any kind. Both loops are now inside the same
   `try/except BaseException/finally` block.
3. *"Children that ignore `SIGTERM`."* CONFIRMED -- cleanup only ever
   called `proc.terminate()` once with no escalation. A new
   `_terminate_process_group` helper now does SIGTERM -> wait(timeout)
   -> SIGKILL, verified by a REAL (not monkeypatched) subprocess test:
   a child that installs `signal.signal(signal.SIGTERM, signal.SIG_IGN)`
   is still reaped within the timeout.
4. *"Subprocess-owned dataloader workers."* CONFIRMED -- `proc.terminate()`
   only ever signals the immediate child PID, not any processes IT
   spawns (e.g. PyTorch `DataLoader` worker processes, which a real
   training job would have). Every job is now started with
   `subprocess.Popen(..., start_new_session=True)`, putting it (and
   anything it spawns) in its own process group; `_terminate_process_group`
   signals that whole group via `os.killpg`, with a fallback to
   `proc.terminate()`/`proc.kill()` if group-based signaling is
   unavailable (e.g. a `ProcessLookupError`/`PermissionError` platform
   difference).

Verified additionally by a monkeypatched-`Popen` regression test proving
a `KeyboardInterrupt` raised from one job's `wait()` call during the
main wait loop still triggers `_terminate_process_group` for every
already-spawned job, not just the ones spawned before the interrupt.

**"Still unresolved... actual launch blockers" list -- re-confirmed
accurate, no new action; unchanged from §21/§26/§27.** Every item (no
real data builder/trainer, no physical H&E masking enforcement, no
composite split-disjointness gate, no held-out-mask novelty gate, no
regional/global WSI conditioning in Architectures 3/4, no cache-content
provenance, no pooled-prediction patient-level evaluator, no CUDA/AMP/
memory smoke test, no functional Novae path) restates gaps already
tracked in prior sections. Nothing in this list is new; nothing
regressed.

**No 24-hour run has been started or will be auto-started. This
document only records fixes to already-written code and tests.**

## 29. Response to the eighth external Codex re-audit (of commit 7b5c267)

Adam forwarded an eighth re-audit, of `7b5c267` (§28's fixes), again
verified directly against the real commit. Verdict: "the fixes are
mostly verified," with 3 more "final contained corrections" requested
before the audit-helper cycle should stop and the real trainer/data
builder should be built. Same discipline as every prior round: every
claim checked against the actual code before any fix.

**Verdict on the four §28 claims -- three re-confirmed correct, one
(strict seed-pool default) correct in substance with a flagged wording
slip -- both addressed.**

**Issue #1, "process-group cleanup can still orphan workers" --
CONFIRMED AND FIXED.** Checked directly: `_terminate_process_group`'s
success condition was `proc.wait(timeout=timeout)` returning -- i.e.
only the PARENT process being reaped. If the parent exits promptly on
SIGTERM (the common, default case -- no custom handler means the
default disposition terminates it) while a DESCENDANT in the same
process group (e.g. a DataLoader worker) ignores SIGTERM and keeps
running, that `wait()` call succeeds and the function returned WITHOUT
ever checking whether the group had a live member, so SIGKILL was never
sent -- exactly the audit's described gap. §27's own regression test
only covered the parent-ignores-SIGTERM case (a single process, its own
process group, no descendants), which is why this specific scenario
went unverified. Fixed: added `_process_group_alive(pgid)` (probes via
`os.killpg(pgid, 0)`, which sends no signal, just checks whether any
process in the group still exists) and a `_group_gone()` predicate
requiring BOTH the parent to have exited AND the group to be confirmed
empty; `_wait_until` polls that predicate, escalating to `SIGKILL` on
the whole group if it isn't satisfied within `timeout`. Verified by the
audit's own suggested test: a parent with no custom SIGTERM handler
(dies immediately) spawns a plain child (inherits the parent's process
group, since it's not started with its own new session) that installs
`SIG_IGN` for SIGTERM and keeps running -- confirmed the WHOLE group,
including the surviving descendant, is gone after cleanup (`os.killpg(pgid,
0)` now raises `ProcessLookupError`), not just the parent.

**Issue #2, "gene identity still fails open when metadata disappears"
-- CONFIRMED AND FIXED.** Checked directly: §27's check was `if
model_gene_basis is not None and expected_gene_basis_hash is not
None:` -- an AND, not the mandatory/symmetric check the audit correctly
says this needs. Two real one-sided gaps: a manifest missing
`gene_basis_gene_names_hash` (older manifest, or corrupted/hand-edited)
would silently skip verification for a model that DOES use a gene
basis, and a model missing a `gene_basis` attribute entirely (wrong
architecture class) would silently skip verification against a manifest
that DOES record one. Fixed: verification now fires whenever EITHER
side has gene-basis metadata; if only one side does, that's an explicit
`ValueError` naming which side is missing, not a skip. Verified by two
new regression tests, one per side of the asymmetry: a manifest with
`gene_basis_gene_names_hash` deleted (model still has `gene_basis`) and
a model with `gene_basis` deleted via `del fresh.gene_basis` (manifest
still has the hash) -- both now rejected instead of silently passing.

**Issue #3, "stale mask files are not actually repaired" -- CONFIRMED
AND FIXED.** Checked directly: §27's fix made `ensure_stratified_training_seed_bank`
return the freshly-recomputed `expected` dict on a validated reuse, but
the ON-DISK JSON itself was left untouched -- a different reader (a
concurrent process, or any future code reading the file directly instead
of through this function) would still see the schema-stale artifact.
The audit's own framing is correct that a generation-algorithm version
and an artifact-schema version are distinct concepts; rather than
introduce a new version field for what's fundamentally a "the disk copy
lagged behind an in-memory recomputation that's already proven
equivalent" situation, the simpler and more directly corrective fix was
chosen: once every identifying field and the exact `items` sequence are
confirmed identical (the existing checks), the on-disk file is now
atomically rewritten with the current, schema-complete representation
via the same `save_stratified_mask_bank` atomic writer already used
elsewhere in this module. Verified by extending the existing
schema-backfill regression test to also read the file BACK OFF DISK
after the reuse and confirm it now matches a fresh build exactly, not
just checking the in-memory return value.

**Wording nitpick, "floor division... incorrectly call[ed] a
'ceiling'" -- CONFIRMED AND FIXED.** `n_items // n_strata` is a floor
(rounds down), and §28's comment called it "the naturally achievable
ceiling" -- a genuine wording error (a ceiling rounds up). Corrected the
comment to describe it accurately: the largest value every stratum can
UNIFORMLY guarantee, via floor division, not an upper rounding bound.
No behavior changed, only the comment's wording. (Per this document's
append-only discipline, §27/§28's own prose is left as originally
written rather than retroactively edited.)

**Issue #4, "unique seeds still do not mean unique masks" -- CONFIRMED,
ALREADY DISCLOSED/DEFERRED, unchanged from §27 finding #1's first
sub-point.** This is the same structural point §27 already made in
depth: `build_stratified_training_seed_bank` deliberately produces a
lossless (stratum, seed) SCHEDULE, specifically so it never has to
materialize actual per-item masks (avoiding "the exact multi-gigabyte-
JSON problem" its own docstring cites). Proving two different seeds
produce genuinely different REALIZED query-spot sets requires actually
running the masking algorithm against real coordinates, which this
module deliberately does not do. The additional items the audit lists
here (training-mask uniqueness, validation/test-never-used-in-training,
cross-split query-spot disjointness, cross-sample barcode-collision
safety) are exactly the composite-identity enforcement §26 finding #6
and §27 already name as blocked on the still-missing real data builder
-- restated, not new.

**Closing instruction -- "stop the audit-helper cycle and build the
real data builder/trainer" -- surfaced as the same open decision point
already raised in §26/§27/§28, not resolved unilaterally here.** This
is now the third consecutive audit round ending with this same
instruction. Every fix in this round is complete, tested, and
documented; nothing about starting the real trainer/data builder has
been begun, since that's a substantially larger body of work requiring
explicit direction on scope, not something to start unilaterally in the
middle of an audit-response pass.

**No 24-hour run has been started or will be auto-started. This
document only records fixes to already-written code and tests.**

## 30. Ninth external Codex re-audit (of commit a29be53) — launcher edge case, then the real trainer

A ninth re-audit verified `a29be53` and confirmed the three §29 fixes
correct for their tested scenarios (`git diff --check` clean). It found
one more small launcher edge case, explicitly said it "does not justify
another audit-only round," and gave the productive instruction: fix
this one item alongside starting the real trainer, then move on.

**"`_terminate_process_group()` immediately returns when the parent has
already exited before cleanup begins" -- CONFIRMED AND FIXED.** Checked
directly: the function's first line was `if proc.poll() is not None:
return` -- if the parent had already exited by the time cleanup ran
(plausible whenever cleanup runs some time after whatever triggered it,
not necessarily right after this function's own SIGTERM), it returned
immediately without ever checking whether the process GROUP still had a
live member, even though a surviving descendant (a DataLoader worker,
in the real scenario this whole mechanism exists for) could still be
running. Separately, the function looked up the pgid LAZILY via
`os.getpgid(proc.pid)`, which raises `ProcessLookupError` once the
leader's own pid no longer exists as a process -- even though the
process group itself (the same numeric id) can still have live members
and still be validly signaled via `os.killpg`. Fixed both together:
`launch_suite` now captures each job's pgid via `os.getpgid(proc.pid)`
**immediately after `Popen`**, while the process is definitely still
alive, and stores it alongside the job; `_terminate_process_group` now
accepts that pre-captured `pgid` and always attempts group cleanup
(checking `_group_gone()`, which requires the group to be CONFIRMED
empty via `os.killpg(pgid, 0)`, not just the parent's `poll()`)
regardless of whether the parent has already exited. Verified by a new
regression test matching the audit's exact scenario: a parent spawns a
SIGTERM-ignoring child and exits immediately on its own (no signal from
this launcher at all); cleanup is only invoked once the parent is
CONFIRMED already dead (`proc.wait()` already returned); the orphaned
descendant is still killed.

**Then: proceed with the real Gen3 data builder and trainer, per the
explicit instruction and the 9-step order forwarded** (immutable
dataset manifest and splits; example builder with physical query
removal; realized-mask fingerprinting and leakage rejection;
context-only Novae graphs; real regional/global WSI wiring into
Architectures 3/4; the trainer itself; a pooled-prediction evaluator;
mandatory preflight gates; staged testing culminating in the four long
runs). This is a substantially larger body of work than any single
audit-response round in this document -- work on it is tracked
separately from this section (see the CONTRACT.md sections that follow,
added as each stage lands) rather than folded into this launcher-fix
entry.

**No 24-hour run has been started or will be auto-started.**

## 31. Real Gen3 data builder -- Step 1: immutable dataset manifest

First stage of the real Gen3 data builder/trainer (§30's 9-step order,
item 1: "Build an immutable dataset manifest with sample IDs, composite
(sample_id, spot_id) identities, gene order/hash, coordinates, WSI/cache
provenance and patient-level splits").

**`data/dataset_manifest.py`** (new module):

- `composite_spot_id(sample_id, barcode) -> str` -- `f"{sample_id}::{barcode}"`,
  the TRUE globally-unique spot identity every later leakage/novelty
  check (Step 3) is built on. Plain Visium barcodes are NOT globally
  unique and are reused across different samples/slides -- a gap this
  document has flagged repeatedly (§26 finding #6, §27, §28) but never
  closed until now. Rejects sample_ids/barcodes containing the "::"
  delimiter itself (both are real, plausible strings that must never be
  ambiguous once combined). Verified by a regression test that gives
  TWO different synthetic samples the exact same literal barcode
  strings (the real scenario) and confirms their composite identities
  are still disjoint.
- `gene_panel_hash(gene_names) -> str` -- ordered-panel hash, same
  discipline as `gene_basis.fit_gene_residual_basis`'s
  `gene_names_hash` and `checkpoint.verify_gene_names`: hashes the
  exact ORDER, not just set membership.
- `build_dataset_manifest(...)` -- calls `hest1k_catalog.resolve_sample_selection`
  (unchanged, reused as-is) for the patient-disjoint splits, then
  derives the gene panel by calling `loaders.load_multi_sample` on
  TRAINING samples ONLY (discarding the loaded expression matrices
  immediately afterward -- only the resulting panel is kept), reads
  every kept sample's (train AND held-out) real barcodes/coordinates
  via a cheap `backed="r"` anndata read (mirroring
  `hest1k_catalog._real_var_names`'s identical pattern -- never loads a
  held-out sample's expression matrix), and records lightweight WSI
  cache provenance (path/size/mtime -- the same cheap "identity"
  convention `slide_context.py`'s own runtime cache key already uses,
  deliberately NOT a full-file SHA256 hash, which would be
  prohibitively slow for many large per-sample GigaPath tile caches at
  manifest-build time). Returns `None` for a sample's `wsi_cache` field
  when the cache file doesn't exist yet -- not an error here; Step 8's
  preflight gates are where "required but missing" becomes fail-closed,
  not this bookkeeping step.
- `save_dataset_manifest`/`load_dataset_manifest`/`ensure_dataset_manifest` --
  the same atomic-write (process-specific temp file + `os.replace`) and
  fail-closed-reuse pattern every other persisted artifact in this
  package already follows (`mask_bank.save_mask_bank`,
  `mask_schedule.save_stratified_mask_bank`). `ensure_dataset_manifest`
  compares a fresh in-memory rebuild against the on-disk file
  byte-for-byte (not just a few fingerprint fields) and raises if they
  differ -- real HEST-1k data or build arguments can change between
  runs, and a manifest that silently kept describing stale data would
  defeat the entire point of building one.
- `all_composite_spot_ids(manifest, sample_ids=None)` -- the base set
  Step 3's leakage/novelty checks will be built on.

**`data/loaders.py`** (new module, copied VERBATIM from
`gen2_architectures/data/loaders.py` at commit 621c610, same
copy-provenance discipline as every other reused module in this
package): `load_hest_sample`, `load_multi_sample` (with its
`reference_genes` strict-held-out-vocabulary path, already exactly what
"derive the gene panel from training samples only" needs),
`basic_qc_and_normalize`, `load_hest_patches`/`align_patches_to_adata`
for H&E patches, `get_coords_3d`. This is the first module in
`gen3_multiscale/` that actually loads real per-sample expression data
-- Phases 0-8 built the schema/model/architecture/mask-scheduling
layers around real data without ever loading any.

**`hest1k_catalog.resolve_sample_selection`** (both `gen3_multiscale`
and `gen2_architectures` copies, kept in sync): now also returns
`patient_by_sample` in its result dict. The function already computed
this internally (used by `_resolve_cross_organ_patient_conflicts`) but
never exposed it; the dataset manifest needs each kept sample's real
patient identity, and re-deriving it independently from the metadata
CSV a second time would risk silently disagreeing with the split
function's own internal computation. A small, additive, backward-
compatible change (new dict key, existing callers unaffected); applied
identically to both copies, with an identical new regression test in
both test files.

**Verified with real (not `.touch()`-placeholder) synthetic AnnData**:
unlike `test_hest1k_catalog.py`'s existing fixtures (which only need
files to exist, never reads their content unless
`check_gene_panel_compatibility=True`), this module reads real
barcodes, coordinates, and gene panels, so `test_dataset_manifest.py`
writes genuine small `.h5ad` files (`anndata.AnnData` with
`obsm['spatial']`, real `var_names`/`obs_names`) via `anndata`/`scanpy`,
both already available in this environment. 10 new tests, including the
barcode-collision regression above and a test that gives held-out
samples EXTRA genes train samples don't have, confirming `gene_panel`
reflects only what training samples actually have (the split's actual
reported `validation_sample_ids` is determined via a dry run first,
rather than guessing `resolve_sample_selection`'s internal shuffle, so
the assertion is deterministic and legible rather than a hardcoded
guess).

**Deliberately NOT built in this step** (later steps in the 9-step
order): loading/aligning H&E patches or WSI tiles, mask realization,
composite-identity leakage ENFORCEMENT (this step only provides the
identity vocabulary; Step 3 builds the actual reject-on-collision
checks), Novae graphs, or anything touching the trainer/evaluator. This
module's job is exactly what its name says -- one authoritative,
persisted description of what data exists and how it's split -- nothing
that consumes it yet exists.

**No 24-hour run has been started or will be auto-started.**

## 32. Real Gen3 data builder -- Step 1 fix + Step 2: real example builder

**Step 1 fix, found while designing Step 2, before any Step-2 code
shipped:** `_read_sample_barcodes_and_coords` used a cheap `backed='r'`
read with NO per-spot QC filtering, so the manifest's declared spot set
for a sample could silently disagree with what `loaders.load_multi_sample`
(used for gene-panel derivation, and now reused by the example builder)
actually keeps after `min_genes` filtering. Fixed: applies the
IDENTICAL filter, for every sample (train and held-out), via a real
(non-backed) read -- a one-time manifest-build cost, not a
per-training-step one. Verified by a new regression test with one
deliberately all-zero-expression spot.

**Step 2: `data/example_builder.py`** (new module), Step 2 of the
9-step order: "Build training examples from realized masks. Query
spots must be physically absent from both input GEX and target-region
H&E. Context may contain surrounding H&E and observed GEX only."

- `load_sample_for_examples(manifest, sample_id, hest_data_dir=None)` --
  real, QC'd, gene-panel-aligned expression AND H&E patches for one
  manifest sample. Reads every QC/normalization parameter FROM the
  manifest's own `build_args`, never re-specified independently, so the
  spot/gene universe this returns is guaranteed identical to what the
  manifest already declared -- one authoritative source, not two that
  could drift apart. Documented, deliberate scope boundary: HEST-1k's
  own H&E patch extraction can drop a further, normal ~4.5% of spots
  that have no matching patch (`loaders.align_patches_to_adata`'s
  existing, audited behavior); this function's returned barcode set can
  therefore be a SUBSET of the manifest's declared one, checked (never
  silently GROWS beyond the manifest) but not itself an error.
- `build_spatial_field_example(adata, patches, context_barcodes,
  query_barcodes, image_feature_fn, ...)` -- the real per-example
  builder. Query spots are already structurally absent from every
  `observed_*` array by construction (this function never reads a query
  spot's expression or patch for anything beyond the geometry check and
  `SpatialFieldTargets` itself); the NEW, real safety logic is
  PHYSICAL, not just barcode-level: a context spot whose H&E patch
  FOOTPRINT overlaps the query hole (a real square of pixels around its
  coordinate, not a point) is excluded from the observed set entirely,
  via `slide_context.nonoverlapping_context_patch_mask` (already-
  audited overlap geometry from Phase 3, reused here for the first
  time, not reimplemented). Verified directly: a context spot placed
  immediately adjacent to a query spot (same barcode-level disjointness,
  genuinely different barcode) is excluded when the patch size is large
  enough to physically overlap, and correctly retained when it isn't --
  both directions of the same test.
- Coordinates are normalized before being stored, not raw physical
  pixels/microns -- `SpatialFieldInputs`' own documented contract
  ("Coordinate memorization" risk) and a repeated audit recommendation
  (6th Codex re-audit of commit 06f5cce: "Use coordinates relative to
  the hole or slide centre in units of median spot spacing. Do not feed
  raw pixel coordinates."). Centered on this example's own
  observed+query centroid; scaled by `_median_nearest_neighbor_spacing`,
  computed from the sample's WHOLE real spot lattice (a caller-supplied
  `full_sample_coords`) so the unit doesn't fluctuate mask-to-mask.
  Verified by a regression test using a real, non-trivial physical
  spacing (37.5) and confirming the stored coordinates end up centered
  near zero and scaled to roughly spot-spacing units (~1.0), not the
  raw value.
- `image_feature_fn` (H&E tile -> feature vector) is INJECTED, not
  called directly -- this module has no hard dependency on a real
  GigaPath checkpoint, matching every other pluggable-feature-function
  pattern already established in this codebase
  (`models/slide_encoder.py`, `gen2_architectures/data/context_features.py`'s
  `ContextOnlyNovaeProvider`/`PrecomputedSpotFeatureProvider`). Fully
  testable with a cheap deterministic stub.

**Deliberately NOT built in this step, and why:**

- **Real GigaPath tile-encoding + disk caching wired in as the default
  `image_feature_fn`.** `gen2_architectures/training/data_prep.py`'s
  `get_gigapath_features` (cfg-coupled, patch-content + preprocessing-
  version fingerprinted, disk-cached) is the audited real implementation
  to eventually plug in here -- not reimplemented in this pass because
  it needs a real GigaPath tile-encoder checkpoint to run or meaningfully
  test, and this module's `image_feature_fn` injection point already
  makes wiring it in later a small, additive change, not a redesign.
- **A per-spot `image_available` schema flag + model-layer masking.**
  Several audit rounds (most explicitly the 5th, commit c02a5d1 finding
  #5) asked for observed spots to be KEPT with a zeroed feature and an
  explicit `image_available=false` flag, rather than excluded outright.
  This pass instead EXCLUDES overlapping-footprint context spots from
  the observed set entirely -- satisfying the literal 9-step Step 2
  instruction ("physically absent... H&E") without a schema change.
  Adding a soft "kept but flagged unavailable" field would require
  extending `SpatialFieldInputs` (a schema every architecture already
  consumes) AND wiring every architecture wrapper's attention/token
  logic to actually respect the flag -- a substantial, separate change
  to already-heavily-audited model code, not a data-builder concern.
  Left as an explicit design note for Step 5 (wiring real WSI context
  into Architectures 3/4), which already touches that model-layer
  surface for a related reason.
- **Dense WSI tile loading/filtering** (`slide_context.load_slide_context`/
  `visible_slide_context`, the regional/global GigaPath path) -- Step
  5's job, not Step 2's; this module only builds the local
  observed/query/boundary example, matching Architecture 1/2's scope,
  not yet the full regional-H&E/global-slide context Architectures 3/4
  are meant to eventually use.

**No 24-hour run has been started or will be auto-started.**

## 33. Response to the tenth external Codex re-audit (of commit 9592d9e)

Codex reviewed the real Step 1/2 code (§31-32) itself, not just the
helper infrastructure. Verdict: "The foundation is substantially
better, but Steps 1-2 are not fully closed yet" -- 2 experiment-critical
findings and 3 smaller requirements, with an explicit instruction: fix
points 1-3 before starting Step 3. All 5 verified against the real code
before any fix; verdicts and fixes below.

**Finding #1 (CRITICAL, CONFIRMED): held-out samples still influenced
training-cohort selection.** `resolve_sample_selection`'s
`check_gene_panel_compatibility` block called `resolve_compatible_sample_ids`
on `sorted(set(train_ids) | set(validation_ids) | set(test_ids))` --
POOLED train+validation+test together. A held-out sample's real gene
panel could change which genes counted as "core" (the `min_gene_coverage`
threshold denominator includes every candidate, held-out or not),
which could change which TRAINING samples cleared `min_sample_coverage`
and what the final frozen panel was -- held-out data indirectly
influencing the training cohort/vocabulary, exactly what "derive the
gene panel from training samples only" (5th Codex re-audit, carried
into `dataset_manifest.build_dataset_manifest`) is meant to prevent.
Confirmed by direct re-read of the pooling call site and reproduced in
a new regression test before fixing.

Fixed in both `gen3_multiscale/data/hest1k_catalog.py` and
`gen2_architectures/data/hest1k_catalog.py` (kept in sync, same
discipline as every other reused-infra fix in this document):
`resolve_compatible_sample_ids` is now called on `train_ids` ONLY,
producing a frozen `(train_ids, shared_genes)`. A new
`_filter_held_out_ids_against_reference_panel(hest_data_dir, held_out_ids,
reference_genes, min_sample_coverage)` independently checks each
validation/test sample's real panel against that already-frozen
reference and drops the ones that don't cover it -- it only ever READS
held-out panels and can never feed back into `train_ids` or
`shared_genes`. Verified by
`test_resolve_sample_selection_derives_panel_from_train_only_not_pooled_with_held_out`
(both copies): 11 samples share an identical, fully-compatible 30-gene
panel; whichever 2 land in validation (determined by a dry run, not a
hardcoded guess about the split RNG) are then degraded to a real
25-gene panel (83% coverage of the training panel -- below the 90%
threshold against the true training reference, but enough to have
silently redefined "core" down to 25 genes and passed under the old
pooled logic). After the fix: train_ids are completely unaffected and
the frozen panel derived from train_ids alone is still all 30 genes;
the degraded validation samples are correctly excluded from
`validation_sample_ids`.

This fix changes real threshold-boundary behavior in one pre-existing
test, `test_resolve_sample_selection_end_to_end_drops_incompatible_organ_member`
(both copies): with only 9 compatible samples total, whether the
seeded split happens to place the incompatible `OUTLIER` sample inside
`train_ids` itself now matters (counted as one of only 9 train
candidates, genes it doesn't share fall to 8/9=0.889 coverage, just
under the 0.9 threshold, which can legitimately exclude the whole
cohort) -- a real, separate small-N boundary effect, not a bug. Fixed
by having the test pick (via a cheap dry run over `split_seed`, never a
hardcoded guess) a seed where `OUTLIER` lands outside `train_ids`,
preserving the test's actual intent ("one held-out-quality outlier gets
silently dropped, training untouched").

**Finding #2 (CRITICAL, CONFIRMED): the "immutable" manifest didn't
identify its own actual data content.** `_wsi_cache_provenance` recorded
path/size/mtime only (a deliberate, documented choice for a *runtime
cache key*, where detecting *most* changes cheaply is enough), and
nothing at all identified the metadata CSV, h5ad files, or patch h5
files. Confirmed: a manifest claiming immutability needs to actually
identify the bytes it describes, not just a cheap proxy that can miss a
same-size, same-mtime content change (rare but real -- e.g. a corrected
re-download that preserves both).

Fixed in `gen3_multiscale/data/dataset_manifest.py`: real SHA256
content hashing (`_sha256_file`, streamed, not loaded fully into
memory) for the metadata CSV (`metadata_csv_provenance`) and every kept
sample's h5ad + patch h5 files (`samples[sid]["content_provenance"]`),
plus the WSI cache (`wsi_cache["sha256"]`, added alongside the existing
path/size/mtime fields, not replacing them). Hashing large files at
every manifest build is real, non-trivial cost, so it is memoized in an
on-disk digest database (`_cached_file_content_hash` /
`load_digest_cache` / `save_digest_cache`, default
`hest_cache_dir/content_digest_cache.json`) keyed by each file's own
path+size+mtime -- exactly the mitigation the audit itself proposed
("hashing can be cached via a separately-verified digest database for
the one-time cost"). A cache hit still re-verifies size/mtime against
the real file before ever trusting a memoized digest. `build_dataset_manifest`
now has one deliberate side effect (reading/updating this digest cache)
beyond the pure-function contract its own docstring previously claimed;
the docstring is updated to say so explicitly. Verified by 4 new tests:
every recorded sha256 matches an independent hash of the real file
bytes; a real content mutation (with a real mtime bump) changes the
recorded hash; the digest cache is actually CONSULTED on a repeat build
(proven by tampering a cached digest for an unchanged file and
observing the tampered value come back out, which could only happen if
the cache were read rather than the real file re-hashed).

**Finding #3 (important, CONFIRMED): patient identity wasn't
study-namespaced.** Raw HEST-1k `patient` metadata values are not
proven globally unique ACROSS different contributing studies -- two
unrelated real patients from two different studies could share a
literal patient label and get incorrectly merged into one "patient" for
the train/val/test disjointness guarantee `_resolve_cross_organ_patient_conflicts`
depends on.

Fixed in both `hest1k_catalog.py` copies: when the real metadata has
both `patient` and `study_link` columns (confirmed present in HEST-1k's
real 28-column metadata CSV, `docs/dataset_notes.md`) and both are
non-blank for a given row, the resolved patient identity is namespaced
`f"{study_link}::{patient}"`. When `study_link` is absent as a column
entirely, or blank for a specific row, behavior falls back to the
pre-existing, already-tested bare-patient-value clustering -- namespacing
is never fabricated from data that isn't there. Verified by 2 new
tests (both copies):
`test_patient_identity_is_namespaced_by_study_link` (two samples from
study S1 both labeled patient "P0" cluster together; two OTHER samples
from study S2 also labeled "P0" get a DIFFERENT resolved identity than
S1's, and neither equals the bare "P0") and
`test_patient_identity_falls_back_to_bare_value_without_a_study_link_column`
(no `study_link` column at all -- behavior is unchanged from before this
fix, so none of this file's other patient-column fixtures broke).

**Finding #4 (important, RECORDED, deferred to Step 5 as instructed):
per-modality GEX/H&E availability.** `example_builder.build_spatial_field_example`
currently excludes a context spot's GEX *and* H&E entirely when its
patch footprint overlaps the query hole, even though its real GEX
reading is still valid and uncontaminated -- only the H&E contribution
is actually compromised. The audit's own framing: "The deferred
`image_available` mechanism must be completed during Step 5. It cannot
remain optional for the real trainer." Recorded here as a firm
requirement (not a soft "consider later" note, per the audit's explicit
correction to §32's original wording): Step 5 (wiring real regional/
global GigaPath WSI context into Architectures 3/4) MUST introduce a
per-modality availability signal -- GEX+H&E observed / GEX observed
with H&E unavailable / neither (query) -- so a physically-overlapping
context spot can retain its valid GEX contribution while only its image
contribution is masked, instead of losing both. This is a schema change
(`SpatialFieldInputs` plus every architecture wrapper's attention/token
logic) too large to fold into a Step 1/2 patch, but Step 5 cannot be
considered complete without it.

**Finding #5 (important, CONFIRMED): example-builder boundary checks
were too loose.** `build_spatial_field_example` didn't verify
`patches.shape[0] == adata.n_obs`, didn't check `image_feature_fn`'s
output was 2D/finite/the expected width, didn't check
`adata.obsm['spatial']` was `[N, 2]`/finite/non-duplicated, and silently
fell back to a mask-dependent coordinate subset for spot-spacing
normalization whenever `full_sample_coords` wasn't supplied -- a
real, mask-to-mask-fluctuating spacing unit for no biological reason,
acceptable only for small/synthetic tests, not the real trainer.

Fixed in `gen3_multiscale/data/example_builder.py`:
- `full_sample_coords` is now REQUIRED by default (raises if omitted);
  the mask-dependent fallback is gated behind an explicit, clearly-named
  `require_full_sample_coords=False` escape hatch reserved for
  small/synthetic tests -- the real trainer must never set it.
- `patches.shape[0] != adata.n_obs` now raises immediately.
- `adata.obsm['spatial']` is checked for shape `[N, 2]`, all-finite, and
  no exact-duplicate rows (a real Visium sample should never have two
  spots at the identical physical coordinate; a duplicate indicates
  corrupted or misaligned real data).
- `image_feature_fn`'s output is checked for 2D-ness, row-count match
  (pre-existing), all-finite values, and (via a new optional
  `expected_feature_width` parameter) the caller's expected width, e.g.
  GigaPath's real 1536.

Verified by 7 new tests: the default-requires-full_sample_coords gate
(and its explicit opt-out); non-2D features rejected; non-finite
features rejected; a wrong feature width rejected; a patches/adata
row-count mismatch rejected; duplicate spot coordinates rejected. The
existing end-to-end test (`test_end_to_end_manifest_to_example`) was
updated to pass the sample's real `adata.obsm["spatial"]` as
`full_sample_coords` -- exercising the actual recommended production
path instead of the test-only escape hatch.

**What the audit confirmed as correct, no fix needed:** patient-disjoint
splitting logic and global cross-organ reconciliation; the gene panel
ultimately computed from training AnnData objects (the SOURCING was the
bug, not the computation); query expression/patches never entering the
feature path; physical patch-footprint overlap removal being
geometrically conservative; coordinates translated/scaled, not raw;
observed/query barcode separation and tensor alignment validated; the
launcher's recorded-PGID fix (§30) "now satisfactory."

Per the user's explicit instruction ("continue building, but do not
start Step 3 from the current manifest without fixing points 1-3"):
findings #1, #2, and #3 (the three hard blockers) are fixed above;
finding #5 (good practice, not explicitly gating) is fixed too; finding
#4 is recorded as a firm Step 5 requirement, not deferred silently.
Step 3 ("implement Step 3 using realized composite query identities
rather than seeds", per the same instruction) is unblocked as of this
section.

**No 24-hour run has been started or will be auto-started.**

## 34. Real Gen3 data builder -- Step 3: realized-mask fingerprinting and leakage rejection

`data/mask_fingerprint.py` (new module), Step 3 of the 9-step order:
"Fingerprint sorted realized query identities", sharpened by §33's
finding #1 discussion and the 10th Codex re-audit's own closing
instruction: "implement Step 3 using realized composite query
identities rather than seeds."

The real gap this closes: `mask_bank.py`/`mask_schedule.py`'s training
seed banks deliberately store only a (stratum, seed) SCHEDULE, not
every realized context/query barcode list (`build_training_seed_bank`'s
own docstring: storing every draw explicitly would create multi-
gigabyte JSON). Their `unique_mask_count`/`unique_masks_per_stratum`
fields count DISTINCT SEED VALUES and implicitly assume distinct seeds
always realize distinct masks. `mask_bank.make_split` is a
deterministic function of (data, masking_cfg, seed), but it is NOT
proven injective in seed -- nothing before this module ever checked
whether two different seeds could realize the identical actual
context/query split.

- `sorted_composite_query_fingerprint(sample_id, query_obs_names)` --
  SHA256 of the SORTED set of composite (sample_id, spot) identities
  (`dataset_manifest.composite_spot_id`, Step 1's true globally-unique
  spot identity) for a realized query set. Order-independent and
  sample-namespaced, so two different samples sharing a raw barcode
  string can never collide.
- `realize_seed_and_fingerprint(coords3d, slice_ids, obs_names,
  sample_id, masking_cfg, seed)` -- realizes ONE (masking_cfg, seed)
  draw exactly as `mask_bank.build_mask_bank` does internally (same
  `make_split` call, same context capping via `cap_context_mask`), then
  fingerprints the REALIZED query set. Verified to match a direct,
  independent `mask_bank.make_split` + `cap_context_mask` call.
- `verify_realized_seed_uniqueness(coords3d, slice_ids, obs_names,
  sample_id, masking_cfg, seeds)` -- the direct, positive check: realizes
  every seed in a candidate pool (e.g. one stratum's
  `unique_masks_per_stratum` seed range from
  `mask_schedule.build_stratified_training_seed_bank`) and raises,
  fail-closed, if any two seeds realize an IDENTICAL query composite
  fingerprint -- naming exactly which seeds collided. Verified two ways:
  (a) 10 genuinely distinct seeds on a real `random_dropout_patches`
  config realize 10 distinct masks; (b) a real, engineered, NON-flaky
  collision using `hold_out_slice` with two candidate slices -- `numpy`'s
  `rng.choice` over only two options means many different seeds
  coincidentally pick the identical held-out slice, producing
  byte-identical masks; a dry run (never a hardcoded guess about the RNG)
  finds two such seeds first, then confirms the function raises exactly
  as designed.
- `realized_query_composite_ids(sample_id, records)` /
  `verify_no_cross_split_query_leakage(realized_composite_ids_by_split)`
  -- a real, positive cross-split leakage check: for one sample, a
  training draw's realized query spots must never coincide with a
  held-out (validation/test) mask's realized query spots for that same
  sample, or the model would be evaluated on exactly what it was
  trained to reconstruct. This is a leak the sample-level
  train/validation/test split (`dataset_manifest.py`, Step 1) cannot
  see by construction, since it operates ACROSS samples, not within one
  sample's own in-sample validation/test masks. Verified: a genuine
  overlapping query barcode between two splits for the same sample_id
  raises with the specific overlapping composite identities named; a
  disjoint case passes; two different samples sharing a raw barcode
  string never register as an overlap (composite identity, not raw
  barcode, is what's compared).

**Deliberately NOT built in this step:** wiring `verify_realized_seed_uniqueness`/
`verify_no_cross_split_query_leakage` into `ensure_stratified_training_seed_bank`
or a preflight gate as an unconditional, always-run check -- realizing
every seed in a large production seed pool for every training launch
would be real, non-trivial per-launch cost (unlike the seed-schedule
generation itself, which is O(n_items) integer arithmetic, not O(n_items)
mask realizations). This module provides the verification primitives;
wiring them into a mandatory, but appropriately-scoped (e.g. run once
at preflight, not every epoch) gate is Step 8's job ("mandatory
preflight gates"), not Step 3's.

**No 24-hour run has been started or will be auto-started.**

## 35. Response to the eleventh external Codex re-audit (of commits 9dab8fe/d1fa094)

Codex verified the 10th-round fixes as real (train-only compatibility,
patient namespacing, example validation, partial-overlap leakage
detection all confirmed), then found 3 further issues before Step 4:
digest-cache trust, a held-out gene-panel coverage inconsistency, and
Step 3 not yet being a complete enforcement gate. All 3 verified against
the real code before fixing.

**Finding #1 (CONFIRMED): the digest cache was TRUSTED, not verified.**
`_cached_file_content_hash` returned a memoized digest whenever a
file's path+size+mtime matched a `digest_cache` entry -- the 10th
round's own regression test (`test_content_digest_cache_is_consulted_on_repeat_builds`)
proved this by tampering the cached SHA256 and asserting the FAKE value
came back out of a rebuild. Correct per the audit: "For an immutable
manifest/preflight, rehash the files. A stat-based cache can be used
for convenience, but it cannot be the authority for cryptographic
verification." Fixed: `_cached_file_content_hash` now ALWAYS computes a
fresh hash by reading the real file bytes; `digest_cache` is still
updated and persisted to disk, but purely as a write-only historical
ledger (path -> {size, mtime, sha256}) for auditing/debugging, never
read to skip a real hash computation. The old test was replaced with
its logical opposite,
`test_content_hash_is_always_freshly_computed_never_trusted_from_the_digest_cache`:
tampering the on-disk digest cache for an UNCHANGED file must NOT
affect the next build's recorded hash.

**Finding #2 (CONFIRMED): held-out gene-panel compatibility was
inconsistent with the actual frozen panel.** `hest1k_catalog`'s
held-out pre-filter (`_filter_held_out_ids_against_reference_panel`,
added in §33) only requires `min_sample_coverage` (typically 90%) of
`shared_genes` -- a RAW var_names intersection computed independently
of, and not necessarily identical to, `build_dataset_manifest`'s actual
frozen `gene_panel` (which additionally applies `gene_min_cells`'s
pooled QC filter and is typically a STRICT SUBSET of `shared_genes`).
Verified by tracing both code paths: `resolve_compatible_sample_ids`
computes `shared_genes` from RAW, unfiltered `_real_var_names`;
`load_multi_sample` computes the real `gene_panel` from the SAME raw
intersection further reduced by a pooled `min_cells` filter -- two
genuinely different computations that can disagree. A held-out sample
passing the coarse 90% pre-filter can still be missing gene(s) that end
up in the real frozen panel, which `example_builder.load_sample_for_examples`
requires 100% of -- crashing at evaluation time instead of at
manifest-build time. Fixed per the audit's own "ideally" recommendation:
a new, definitive check in `dataset_manifest.build_dataset_manifest`
(`_filter_ids_missing_final_panel_genes`), run AFTER `gene_panel` is
frozen, requiring exact 100% coverage of the ACTUAL final panel; the
coarse pre-filter is kept as a cheap, non-authoritative early filter,
not removed. Non-conforming held-out samples are dropped with a printed
notice, matching the codebase's established pattern, rather than
crashing the whole run. Verified by
`test_held_out_samples_missing_genes_from_the_final_panel_are_excluded`:
5 train samples share an identical 20-gene panel; the held-out sample
is missing exactly 1 of those genes (95% coverage -- passes the coarse
pre-filter) but is correctly excluded by the new exact check.

**Finding #3 (CONFIRMED): Step 3 was a useful utility but not yet a
complete enforcement gate.** Five concrete gaps, all closed in
`data/mask_fingerprint.py`:
- **Duplicate identities inside a query list.** `sorted_composite_query_fingerprint`
  now raises if `query_obs_names` contains the same barcode more than
  once (a real query set must be a set of distinct spots); `realized_query_composite_ids`
  inherits the same check per record.
- **Validate every realized identity against the manifest and expected
  sample.** New `validate_realized_barcodes_against_manifest(manifest,
  sample_id, obs_names)`; `realize_seed_and_fingerprint` gained an
  optional `manifest` parameter that validates both context and query
  barcodes before fingerprinting.
- **Detect identical masks across different strata/configurations, not
  only seeds passed to one masking configuration.** New
  `verify_realized_pool_uniqueness(coords3d, slice_ids, obs_names,
  sample_id, items, manifest=None)` checks uniqueness across a POOL of
  `{"label", "masking_cfg", "seed"}` entries spanning ANY number of
  different configs, not just one config's seed range;
  `verify_realized_seed_uniqueness` is now a thin single-config wrapper
  over it. Verified with an engineered, deterministic cross-config
  collision: `mask_bank.make_split`'s `hold_out_slice` branch never
  reads `masking_cfg['params']` at all, so two configs differing only
  in an ignored params field realize byte-identical masks for the same
  seed.
- **Process the complete production schedule and persist a leakage
  report.** New `build_mask_fingerprint_report(manifest, sample_id,
  coords3d, slice_ids, obs_names, strata, stratified_training_seed_bank,
  stratified_mask_bank)`: deduplicates the FULL training schedule's
  (stratum, seed) pairs (a schedule's `n_items` can legitimately repeat
  pairs via round-robin cycling), realizes each exactly once, pool-
  checks uniqueness across strata, validates every identity against the
  manifest, and runs the cross-split leakage check against the real
  validation/test records -- plus `save_mask_fingerprint_report`/
  `load_mask_fingerprint_report` (atomic write, mirroring every other
  artifact in this package) so Step 8's preflight gate has a concrete
  artifact to require before training starts.
- **Validate `full_sample_coords` itself.** `example_builder.build_spatial_field_example`
  now checks a caller-supplied `full_sample_coords` for shape `[M, 2]`,
  finiteness, no duplicate rows, at least as many rows as the aligned
  sample, and that every one of the aligned sample's own coordinates is
  actually present in it (real agreement, not just a same-shaped
  unrelated array) -- previously only `adata.obsm['spatial']` itself
  received these checks.

**A real, separate finding surfaced while testing the production-
schedule report, not something to silently work around:** on a small-
to-moderate synthetic grid, `random_dropout_patches` scattering
independently-seeded train/validation/test query patches across ONE
shared per-sample coordinate space produces cross-split query overlap
by chance alone with real regularity -- confirmed at both a 144-spot
and a 1600-spot grid with this project's existing stratum radii. This
is not a bug in `mask_fingerprint.py` (its job is to DETECT this, which
it correctly does) -- it is a real characteristic of the existing
per-sample mask-drawing scheme (`mask_bank.py`/`mask_schedule.py`,
unmodified here) that has never guaranteed cross-split disjointness
within one sample. Recorded here as a real, open question for whoever
wires per-sample validation/test masks into the real trainer (Step 6)
or preflight gates (Step 8): either the mask-drawing scheme needs an
explicit disjointness mechanism (e.g. excluding already-claimed
validation/test query spots from the training draw pool), or per-sample
in-sample validation/test masking needs to be reconsidered in favor of
relying solely on the sample-level train/validation/test split. Not
fixed in this pass -- out of scope for a fingerprinting/detection module
whose job is to surface exactly this kind of problem, not to redesign
the mask-generation scheme itself.

**No 24-hour run has been started or will be auto-started.**

## 36. Real Gen3 data builder -- Step 4: context-only Novae graphs

`data/novae_graph.py` (new module), Step 4 of the 9-step order: "Build
context-only Novae graphs", sharpened by the 11th Codex re-audit's Step
4 decision: "the Novae graph must use all GEX-available context spots
-- including spots whose H&E patch is unavailable due to overlap --
while query nodes are physically absent. The cache key and preflight
report must prove that no query composite identity appears in: graph
nodes; graph edges; Novae input expression; global/induced pools;
cached embeddings."

Researched before writing any code (no code changes from this
research, findings only): `gen3_multiscale` has no prior Novae code at
all. `gen2_architectures/data/context_features.py`'s
`ContextOnlyNovaeProvider` -- the existing, already-audited leakage-safe
pattern for Novae (subsets `adata` to `context_mask`, calls an injected
`feature_fn`, disk-caches by a mask digest) -- is a caching/injection
WRAPPER only; it never builds or exposes a graph structure as its own
artifact, delegating graph construction entirely to the injected
function. No `novae` or `torch_geometric` package is installed anywhere
in this repo. This module is the gen3_multiscale analog, adapted to
this package's barcode-list convention (matching `example_builder.py`)
and composite-identity leakage discipline (matching
`mask_fingerprint.py`), and additionally builds the GRAPH STRUCTURE
itself (nodes, edges) as a first-class, independently-verifiable
object -- required by the explicit five-way non-leakage proof above,
which an opaque feature-vector wrapper alone cannot provide.

- `build_context_only_novae_graph(adata, context_barcodes, query_barcodes,
  *, sample_id, k_neighbors=6) -> NovaeGraphInputs` -- builds a k-NN
  graph (`boundary_graph.build_knn_adjacency`, reused, not
  reimplemented) over EVERY barcode in `context_barcodes`, with NO
  H&E-overlap-based reduction: unlike `example_builder.build_spatial_field_example`
  (which excludes context spots whose H&E patch physically overlaps the
  query hole from `observed_*` entirely, both modalities -- CONTRACT.md
  section 33 finding #4, still deferred to Step 5), Novae is GEX-only
  and has no H&E constraint, so this function reads directly from the
  realized mask's `context_barcodes` (physically disjoint from
  `query_barcodes` -- the only real constraint) and uses every
  GEX-available spot. `node_barcodes` is sorted (deterministic,
  independent of the caller's list order); `node_coords`/`node_expression`/
  `edge_index` are all aligned 1:1 to it by position. `cache_key` is a
  SHA256 over SORTED context-node composite identities ONLY -- verified
  directly (`test_cache_key_depends_on_context_but_never_on_query_barcodes`)
  to be identical across two graphs built from the same context set with
  DIFFERENT query sets, and to change when the context set itself
  changes: the cache key structurally cannot leak or vary with which
  spots were held out. `verify_novae_graph_excludes_query_identities` is
  called automatically before returning.
- `verify_novae_graph_excludes_query_identities(graph, query_barcodes) -> dict`
  -- proves (1) no query composite identity is among `node_composite_ids`,
  (2) every `edge_index` entry references a real node position in
  `[0, n_nodes)` (structurally verified, not merely assumed to be safe
  by construction), and (3) `node_expression`/`node_coords` are row-
  aligned 1:1 with `node_barcodes` -- which together prove the "Novae
  input expression" is leakage-free too, since its rows are the same
  nodes. Verified with an engineered corruption test for each: directly
  mutating `node_composite_ids[0]` to a query's composite id (frozen
  dataclass, mutable numpy array contents) is caught; directly
  corrupting `edge_index` to an out-of-bounds position is caught.
- `pool_novae_embeddings(node_embeddings) -> np.ndarray` -- the
  "global/induced pool": mean-pools node-level embeddings into one
  per-graph summary vector, leakage-safe BY CONSTRUCTION since it only
  ever operates on an array whose rows are already the context-only
  node embeddings.
- `compute_novae_embeddings(graph, novae_feature_fn)` -- calls an
  INJECTED `novae_feature_fn(node_expression, edge_index) -> [n_nodes, dim]`,
  matching every other pluggable-feature-function pattern already
  established in this codebase (`example_builder.image_feature_fn`,
  `ContextOnlyNovaeProvider`'s own `feature_fn`) -- no real Novae
  checkpoint dependency. Validates shape and finiteness.
- `ensure_cached_novae_embeddings(cache_dir, graph, novae_feature_fn,
  query_barcodes) -> (embeddings, path)` -- atomic disk cache keyed by
  `graph.cache_key`. Before ever returning a CACHED result, re-verifies
  the cache file's own recorded `node_composite_ids` against the
  CURRENT query set and raises if any overlap is found -- proven by a
  test that hand-tampers a valid cache file's `node_composite_ids` to
  include a query composite identity and confirms the next load refuses
  to reuse it, rather than assuming a matching filename/cache_key
  implies safety.
- `build_novae_preflight_report(adata, context_barcodes, query_barcodes,
  novae_feature_fn, *, sample_id, cache_dir) -> dict` /
  `save_novae_preflight_report` / `load_novae_preflight_report` -- the
  complete preflight artifact: builds the graph, computes/caches
  embeddings, pools them, and records an explicit `"checks"` dict with
  one boolean per one of the five things Step 4's requirement names
  (graph nodes, graph edges, Novae input expression, the global/induced
  pool, cached embeddings) -- the artifact Step 8's mandatory preflight
  gate is meant to require before any real training run touches Novae
  features. Atomic write, mirroring every other artifact in this
  package.

**Deliberately NOT built in this step:** the real Novae model/checkpoint
call itself (no such dependency exists anywhere in this repo; would be
injected via `novae_feature_fn` exactly like GigaPath's real tile
encoder is meant to be injected into `example_builder.image_feature_fn`
-- a separate, later integration task, not this step's); wiring
`build_novae_preflight_report` into `models/architectures.py`'s
Architecture 3/4 forward pass (that is Step 5's job, "wire real
regional/global GigaPath WSI context into Architectures 3/4" -- Novae
wiring was never part of the original 9-step Step 5 description, so
whether/how Novae features flow into the architectures is left as an
open design question for whoever picks that up, not assumed here);
multi-sample/whole-cohort orchestration of `build_novae_preflight_report`
across every training sample in a manifest (this module operates on one
sample at a time, matching `mask_bank.py`'s own established per-sample
scope; a cohort-level wrapper belongs with Step 8's preflight gates,
which already need to loop over the whole manifest for other checks
too).

**No 24-hour run has been started or will be auto-started.**

## 37. Response to the twelfth external Codex re-audit (of commit 1bb66d6)

Codex confirmed the 11th round's fixes (train-only compatibility,
patient namespacing, example validation, partial-overlap leakage
detection) as real improvements, then found the Step 4 "Novae graph"
was not actually usable with real Novae, plus three deeper gaps in
Step 3's mask scheduling/reporting and one in Step 1's held-out
dropping -- 8 launch-blocking findings total, verified individually
against the real code (and, for finding #1, against Novae's real
published API, fetched live from github.com/MICS-Lab/novae) before any
fix.

**Finding #1 (CONFIRMED against Novae's real quickstart): the Novae
adapter signature could not accept a real Novae call.** Fetched
`github.com/MICS-Lab/novae`'s README directly: real Novae's interface
is AnnData-in, AnnData-mutated-in-place-out --
`novae.spatial_neighbors(adata)` then `Novae.from_pretrained(...)` then
`model.compute_representations(adata, zero_shot=True)` -- `spatial_neighbors`
builds Novae's OWN graph internally from `adata.obsm['spatial']`; it
does not accept a caller-supplied graph at all. §36's
`build_context_only_novae_graph` built its own separate k-NN adjacency
(`boundary_graph.build_knn_adjacency`) and fed `(node_expression,
edge_index)` to the injected function -- a signature real Novae could
never actually be plugged into, and a graph never proven to be the one
Novae would build. Also confirmed: `gen2_architectures.models.conditioning.precompute_novae_features`
ALREADY implements the real, correct adapter (read in full) -- it takes
a whole `adata`, calls `novae.spatial_neighbors(adata)` itself, loads a
cached `Novae.from_pretrained(checkpoint)`, calls
`model.compute_representations(adata, zero_shot=True)`, and extracts
the new `obsm` key defensively. Fixed: `novae_graph.py` rewritten so
`build_context_only_novae_input` builds the PHYSICALLY context-only
AnnData real Novae needs (real `var_names`, context-only `obs_names`,
context-only `obsm['spatial']`, context-only `X` -- a genuine
`adata[...].copy()`, never a view or closure over the full `adata`),
and the injected `novae_feature_fn` now has the SAME
`Callable[[adata], np.ndarray]` contract `ContextOnlyNovaeProvider`
already uses -- `precompute_novae_features` (or an equivalent) can be
plugged in directly, unmodified. Since the graph Novae builds is
constructed FROM this exact context-only AnnData (inside the injected
function), proving the AnnData itself excludes every query row is what
proves the downstream graph does too; this module no longer fabricates
a second, unverified graph structure to police instead. `NovaeGraphInputs`
is renamed `NovaeContextInputs`; `edge_index`/`node_coords`/
`node_expression` fields are replaced by `context_adata`. Verified:
`test_build_context_only_novae_input_is_a_real_copy_not_a_view_of_the_full_adata`
mutates the returned `context_adata.X` and confirms the original
`adata` is unaffected.

**Finding #2 (CONFIRMED): the cache key hashed only context-node
identities.** Fixed by reusing
`gen2_architectures.data.context_features._adata_feature_signature` --
the ALREADY-AUDITED comprehensive fingerprint (real expression matrix
bytes, obs/var names, spatial coordinates, preprocessing state, and the
feature function's module-qualified name) this project has already
relied on for the exact same problem, rather than inventing a weaker
one. `ensure_cached_novae_embeddings` gained a `checkpoint_signature`
parameter folded into `_adata_feature_signature`'s `extra_signature`
(its own documented mechanism for a caller-supplied real checkpoint
identity, e.g. `f"{repo}:{revision}"`) plus this module's own
`_CACHE_SCHEMA_VERSION`. Verified:
`test_cache_key_changes_when_checkpoint_signature_changes` confirms two
calls differing only in `checkpoint_signature` write to different
cache files.

**Finding #3 (CONFIRMED): cached embeddings were not validated on
load.** `ensure_cached_novae_embeddings` now checks, before ever
returning a cached result: every required `.npz` key is present; the
recorded `cache_key` matches exactly; the recorded node identities/
order match the current context exactly; the embeddings array is 2D
with the correct row count; `float32`; all-finite; and (if an
`expected_dim` is supplied) has the expected width -- any failure is a
hard refusal, not a silent cache miss. Verified by 4 new tests, each
tampering one specific aspect of a valid cache file (missing keys,
wrong dtype, wrong width via `expected_dim`) and confirming the
specific corresponding rejection.

**Finding #4 (CONFIRMED): the preflight report's "checks" dict
overclaimed.** §36's report asserted `True` for claims like "an
arbitrary injected function didn't leak external data," which no
amount of input/output shape checking can actually establish. Fixed:
`build_novae_preflight_report` now returns a `"verified"` list (what
was actually, structurally checked) and a
`"not_provable_from_this_module_alone"` list (named limits -- an
injected function's internal behavior, and checkpoint identity beyond
whatever string the caller supplies), replacing the old boolean
`"checks"` dict.

**Finding #5 (CONFIRMED): Step 3 could detect a mask collision but not
prevent one.** The 11th round's own tests proved
`verify_realized_pool_uniqueness` DETECTS a collision, but nothing in
this package could PRODUCE a collision-free schedule other than a test
searching thousands of `base_seed` values for a lucky non-colliding
one. Fixed: new `mask_fingerprint.build_collision_free_training_schedule`
deterministically builds a training schedule GUARANTEED, by
construction, to never query a reserved (validation/test) composite
identity and never repeat an already-accepted mask -- round-robinning
across strata, trying candidate seeds from a deterministic per-stratum
counter (reusing `mask_schedule._STRATUM_SEED_STRIDE`'s offset
convention so seeds never collide with that module's own ranges),
advancing by 1 on every rejection, up to a bounded `max_attempts_per_item`,
raising (fail-closed) if no valid seed is found for any item. Verified:
`test_build_collision_free_training_schedule_avoids_reserved_and_duplicate_masks`
reuses the EXACT small-grid, large-radius-strata configuration that
previously forced a thousands-of-seeds search, and now succeeds
deterministically on the first call with zero retries from the caller.

**Finding #6 (CONFIRMED): the mask report was not bound to its
inputs.** §34's report recorded only summary counts. Fixed: every
report now includes `input_fingerprints` (SHA256 of the manifest, the
`mask_bank.spatial_fingerprint` of the coordinate lattice, and the
`mask_schedule.strata_fingerprint`) -- a consumer (Step 8's preflight
gate, or the trainer) is expected to recompute these from its own live
data and refuse a report whose fingerprints don't match; this module
records them but does not itself re-verify a loaded report against live
data (documented as the consuming caller's job, since only it has the
live data to compare against). Verified:
`test_report_input_fingerprints_change_when_the_manifest_changes`.

**Finding #7 (CONFIRMED): duplicate masks within one split went
undetected.** The prior cross-split leakage check only ever compared
DIFFERENT splits against each other -- never whether two records
WITHIN the same split (e.g. validation mask #2 and validation mask #5)
realized the identical mask, silently halving that split's real
effective sample size while still counting as two independent draws.
Fixed: new `verify_no_duplicate_masks_within_split`, wired into both
new report builders (below) for every split they process.

**Finding #8 (CONFIRMED): sample splitting and mask splitting were
conceptually mixed.** §34's single `build_mask_fingerprint_report`
processed train/validation/test masks for one sample as if all three
normally coexist. Under this project's patient-disjoint SAMPLE-level
split (`dataset_manifest.py`), a TRAINING sample's real evaluation
happens on ENTIRELY DIFFERENT held-out samples -- `mask_bank.py`/
`mask_schedule.py`'s per-sample validation/test split_counts concept,
realized on a training sample at all, is at most a SECONDARY
same-sample capacity/early-stopping diagnostic. `build_mask_fingerprint_report`
is REMOVED (not deprecated in place -- no external caller depends on
it yet, and keeping two semantically-conflicting APIs would be worse
than a clean replacement) and split into two sample-role-aware
functions:
- `build_training_sample_mask_report(manifest, sample_id, coords3d,
  slice_ids, obs_names, strata, training_schedule, *,
  same_sample_diagnostic_mask_bank=None)` -- primary = the
  collision-free training schedule; if same-sample validation/test
  masks are supplied, they appear under an explicitly separate,
  clearly-labeled `"same_sample_capacity_diagnostic"` section (with an
  inline warning never to substitute it for cross-sample evaluation),
  cross-checked for leakage against the primary schedule.
- `build_held_out_sample_mask_report(manifest, sample_id, coords3d,
  slice_ids, obs_names, strata, split, stratified_mask_bank)` --
  primary = the ONE split (`"validation"` or `"test"`) this held-out
  sample actually belongs to; FAILS CLOSED if the mask bank contains
  any record for a different split (a held-out sample must never carry
  training-labeled masks, and a validation sample must never carry
  test-labeled masks or vice versa).

**Finding #9 (CONFIRMED): held-out samples were silently dropped after
splitting, without re-checking quotas.** `dataset_manifest.py`'s exact
100%-panel-coverage check (§35 finding #2's fix) drops non-conforming
held-out samples, but `resolve_sample_selection` always selects EXACTLY
`n_validation_per_organ`/`n_test_per_organ` distinct PATIENTS for every
organ it keeps at all -- so dropping even one held-out sample can
silently leave an organ's quota unmet or its balance distorted, with
nobody noticing. Fixed: `build_dataset_manifest` now re-validates, per
organ, that the requested validation/test PATIENT counts (not just raw
sample counts -- patient-disjoint splitting keeps every sample from one
patient in the same split) are still met whenever any held-out sample
was dropped for a panel mismatch; raises (fail-closed) otherwise.
Verified by two companion tests: the original "excluded" scenario (only
1 validation sample requested, that sample gets dropped) now correctly
raises; a new scenario where the held-out PATIENT has a SECOND, fully
compatible sample demonstrates the quota surviving and the manifest
building successfully, excluding only the specific bad sample.

**What the audit confirmed as correct, no fix needed:** file hashes
freshly recomputed (not cache-trusted); the gene panel derived from
training samples only; held-out samples checked against the final
frozen panel; query fingerprints using composite (sample, spot)
identities; duplicate query identities and cross-configuration mask
collisions detected; query spots physically absent from the Novae
input; Novae using all GEX-available context spots independent of H&E
availability.

**No 24-hour run has been started or will be auto-started.**

## 38. Response to the thirteenth external Codex re-audit (of commit 65611c7)

Codex confirmed the 12th round's fixes (real Novae adapter interface,
deterministic collision-free scheduling, sample-role-aware reports) as
genuine improvements, then found 8 further launch-blocking gaps across
Step 4's context-only AnnData sanitization and Step 3's schedule/report
binding and validation depth -- all 8 individually re-confirmed against
the real code before any fix, exactly as every prior round.

**Finding #1 (CONFIRMED): the context-only AnnData was not actually
sanitized, only row-subset.** `build_context_only_novae_input` built
`context_adata = adata[node_pos].copy()` -- AnnData's own slicing
correctly row-subsets every array-shaped obs-indexed field (`X`,
`obsm`, `obsp`, `layers`), but it also blanket-COPIES `.uns` and `.raw`
verbatim, and `.uns` entries have no obs-indexed subsetting semantics
at all -- a full-slide summary statistic stashed in `.uns` would carry
forward into the "context-only" object completely unchanged. Fixed:
new `_build_sanitized_context_adata` constructs the object EXPLICITLY
field-by-field -- `X` (row-subset), `var` (gene identities, needed to
match Novae's own vocabulary by name), `obsm['spatial']` (row-subset),
and `.uns` populated ONLY from `_ALLOWED_CONTEXT_UNS_KEYS`
(`_scilifestdl_expression_state`, `expression_preprocessing` -- exactly
the two keys `_adata_feature_signature` itself reads, so the cache
fingerprint still reflects real preprocessing state). `.obsp`,
`.layers`, and `.raw` are never copied. The module docstring's prior
claim that `novae_feature_fn` could never reach the full adata "even
via a closure" was also corrected -- no callee can police what a
caller's own Python closure captures; that specific claim was already,
correctly, listed under `not_provable_from_this_module_alone` and now
the docstring says so instead of overclaiming. Verified:
`test_build_context_only_novae_input_excludes_unwhitelisted_uns_and_obsp_layers_raw`
stashes a full-slide statistic under a non-whitelisted `.uns` key plus
extra `.layers`/`.obsp` entries and confirms none of them appear on
`context_adata`, while the whitelisted key does.

**Finding #2 (CONFIRMED): reports were not bound to the schedule/bank
content or the ordered observation sequence, only to manifest/spatial/
strata summary fingerprints.** `_mask_bank_records_fingerprint` existed
but was DEAD CODE -- defined, never called. Fixed: `_input_fingerprints`
gained `observation_order_fingerprint` (SHA256 of the obs_names
sequence IN ORDER -- `realize_seed_and_fingerprint` indexes by
position, so a report is only meaningful for the exact ordered sequence
it was computed against). `build_training_sample_mask_report` now
records `training_schedule_content_fingerprint` (over the schedule's
own `items`) and `realized_query_composite_fingerprints_fingerprint`
(over the accepted fingerprints set), plus a `content_fingerprint_by_split`
entry inside the same-sample diagnostic section when present.
`build_held_out_sample_mask_report` now records
`mask_bank_content_fingerprint` (over the actual filtered `records`
used) via the now-wired-in `_mask_bank_records_fingerprint`. Verified:
`test_report_input_fingerprints_include_observation_order`.

**Finding #3 (CONFIRMED): a persisted collision-free training schedule
could never be re-verified against live data, and its reserved-set
binding was a COUNT, not a fingerprint.** No `validate_*` function
existed for `build_collision_free_training_schedule`'s output, and its
`n_reserved_composite_ids` field is a size, not an identity -- a
DIFFERENT reserved set of the same size would look identical. Fixed:
the schedule now also records `reserved_composite_ids_fingerprint`
(SHA256 of the sorted reserved-set contents). New
`validate_collision_free_training_schedule(schedule, coords3d,
slice_ids, obs_names, sample_id, strata, reserved_query_composite_ids,
*, manifest=None)` checks schema version/kind, sample_id, strata
identity and fingerprint, the reserved-set fingerprint, internal
item-count consistency, and -- the strongest check -- RE-REALIZES
every stored `(stratum, seed)` item from scratch and confirms its fresh
query composite fingerprint matches the one on record, with no reserved
collisions and no internal duplicates. `_SCHEDULE_VERSION` bumped
2 (schema changed: new required field). Verified: a happy-path pass, a
tampered-fingerprint detection test, and a same-SIZE-different-CONTENT
reserved-set detection test (proving the fingerprint, not the count, is
what's actually checked).

**Finding #4 (CONFIRMED): neither report builder enforced the
manifest's own sample-role assignment, and an empty held-out mask bank
silently "passed."** `build_training_sample_mask_report` accepted any
`sample_id` the `training_schedule` itself claimed, never checking
`manifest["samples"][sample_id]["split"] == "train"`;
`build_held_out_sample_mask_report` never checked
`manifest["samples"][sample_id]["split"] == split`; and if the
requested split's `records` list came back empty,
`verify_no_duplicate_masks_within_split` happily returned
`{"n_records": 0, ...}` with no rejection. Fixed: both functions now
require `sample_id` to exist in the manifest and require its recorded
`split` to match the role being reported (train / the requested
validation-or-test split) -- raising immediately otherwise.
`build_held_out_sample_mask_report` now also rejects an empty `records`
list outright. Verified by 4 new tests covering each direction (unknown
sample, wrong role for training, wrong role for held-out, empty bank).

**Finding #5 (CONFIRMED): `build_held_out_sample_mask_report` trusted
the supplied `stratified_mask_bank`'s records without ever validating
the BANK ITSELF against live data, or checking expected record
counts.** Fixed: the function now validates the bank's own
`dataset_fingerprint`/`spatial_fingerprint`/`strata_fingerprint`/
`mask_generation_version` against live `obs_names`/`coords3d`/
`slice_ids`/`strata` (mirroring `mask_schedule.load_stratified_mask_bank`'s
own staleness checks, applied here to an in-memory bank rather than one
re-read from disk), confirms the expected per-split record count from
the bank's own `split_counts` (`split_counts[split] * len(strata)`),
confirms every declared stratum has at least one record for this split,
and confirms every record's context/query sets are individually
non-empty and disjoint. Verified by 2 new tests (a stale
`dataset_fingerprint`, and a bank missing one expected record).

**Finding #6 (CONFIRMED): the per-organ held-out quota re-validation in
`dataset_manifest.build_dataset_manifest` only ran when THIS function's
own final-exact-panel filter happened to drop something.**
`resolve_sample_selection`'s OWN internal filtering (its coarse
compatibility pre-filter, and its cross-organ patient conflict
resolution) can also leave an organ short of its requested
validation/test patient quota, without this function's own
`dropped_for_final_panel` ever becoming non-empty -- silently skipping
the quota check entirely. Fixed: the per-organ quota loop now runs
UNCONDITIONALLY after sample-selection/held-out-panel filtering, not
gated behind `if dropped_for_final_panel:` (the informational "excluded
N held-out sample(s)" print stays conditional; only the quota
enforcement itself is now unconditional). Verified:
`test_quota_check_fires_even_when_this_functions_own_final_panel_filter_drops_nothing`
monkeypatches `resolve_sample_selection` to return a split where an
organ already has zero validation patients but every returned sample's
panel is fully compatible (so this function's own filter drops
nothing) -- confirms the manifest build now still fails closed.

**Finding #7 (CONFIRMED): `checkpoint_signature` was an OPTIONAL,
default-empty-string parameter.** A caller could silently omit real
Novae checkpoint identity entirely and still get a cache hit or a
"passed" preflight report. Fixed: `ensure_cached_novae_embeddings` and
`build_novae_preflight_report` now take a REQUIRED `checkpoint_provenance:
dict`, validated by new `_validate_checkpoint_provenance` against 5
required, non-empty keys -- `checkpoint_repo`, `checkpoint_revision`
(a real repo revision OR a local checkpoint file's own SHA256),
`novae_package_version`, `adapter_version`, `spatial_neighbors_settings`
(since Novae's own `spatial_neighbors(adata)` graph-construction
settings also affect the embeddings, not just the model checkpoint).
The preflight report now echoes `checkpoint_provenance` and lists the
provenance check under `"verified"`. Verified:
`test_validate_checkpoint_provenance_rejects_missing_and_empty_fields`,
`test_ensure_cached_novae_embeddings_requires_checkpoint_provenance`,
`test_build_novae_preflight_report_requires_checkpoint_provenance`.

**Finding #8 (CONFIRMED): the schedule-builder's seed-retry loop caught
bare `ValueError`, swallowing structural errors as if they were
retriable.** `build_collision_free_training_schedule`'s `except
ValueError: continue` treated a genuinely empty context/query
realization THE SAME as a manifest-validation failure or a malformed
`masking_cfg` -- retrying up to `max_attempts_per_item` times on an
error no different seed could ever fix, before eventually surfacing a
misleading "could not find a collision-free training mask" error
instead of the real cause. Fixed: new `EmptyMaskRealizationError(ValueError)`
is raised ONLY by `realize_seed_and_fingerprint`'s genuine
empty-context/query case; the schedule builder's retry loop now catches
only that narrow type, so any other `ValueError` propagates immediately
with its real, specific message. Verified:
`test_build_collision_free_training_schedule_does_not_retry_a_structural_error`
uses a manifest declaring only 2 of many real barcodes for the sample,
so every realized mask trips the manifest-validation check -- confirms
the real "never declared" error surfaces immediately, not a
retry-budget-exhausted message.

`_REPORT_VERSION` bumped 2 -> 3 (schema changed: new fingerprint
fields on every report). `_SCHEDULE_VERSION` bumped 1 -> 2 (schema
changed: `reserved_composite_ids_fingerprint` is now a required field).

**What the audit confirmed as correct, no fix needed:** the real
AnnData-based Novae interface itself (finding #1 from the 12th round);
comprehensive Novae cache fingerprinting via `_adata_feature_signature`;
cached-embedding validation on load; honest preflight
verified-vs-not-provable claims; the collision-free scheduling
algorithm's core guarantee (never queries a reserved identity, never
repeats an accepted mask); duplicate-mask-within-split detection.

**No 24-hour run has been started or will be auto-started.**

## 39. Response to the fourteenth external Codex re-audit (of commit 8d4e276)

The 13th round's audit-response was judged "mostly solid" but genuinely
incomplete -- Codex found 3 further, narrower gaps in Step 3/4 before
agreeing Step 5 was safe to start. All 3 re-confirmed against the real
code before any fix.

**Finding #1 (CONFIRMED): `_build_sanitized_context_adata` still copied
`adata.var` wholesale.** The 13th round's fix explicitly built `X`/
`obsm['spatial']`/whitelisted `.uns` keys field-by-field, but left
`var=adata.var.copy()` -- copying every column, not just gene identity.
Scanpy/AnnData conventionally stores full-slide-derived per-gene
statistics in `.var` (detection counts, means, dispersion/variability),
computed over EVERY spot including query spots -- the exact same class
of leak as the `.uns` gap the 13th round fixed, just on the gene axis.
Fixed: `var` is now an INDEX-ONLY `pd.DataFrame` built straight from
`adata.var_names` -- gene identity strings, the only thing real Novae
needs to match its own vocabulary by name. Verified:
`test_build_context_only_novae_input_var_is_index_only_not_full_slide_gene_stats`
stashes `mean_counts_full_slide`/`n_cells_by_counts` columns on
`adata.var` and confirms `context_adata.var` has zero columns while
`var_names` is preserved exactly.

**Finding #2 (CONFIRMED): `build_training_sample_mask_report` never
called `validate_collision_free_training_schedule`.** The 13th round
built that validator but never wired it into the report builder -- the
report's own re-realization loop only proved POOL uniqueness among the
schedule's own items; it had no reserved set to check against at all,
so a report could be built successfully from a schedule that (a) no
longer avoids the CURRENT reserved set, or (b) has a tampered/stale
`realized_query_composite_fingerprints` entry that happens to still be
internally self-consistent. Fixed: `build_training_sample_mask_report`
now REQUIRES a `reserved_query_composite_ids` parameter and calls
`validate_collision_free_training_schedule` before doing anything else
-- re-realizing every stored item, confirming it matches BOTH the
recorded fingerprint AND the live reserved set's fingerprint, and
confirming no duplicates -- fail-closed on any mismatch. The report's
own subsequent re-realization loop (still needed to build
`train_query_ids` for the same-sample diagnostic leakage check) no
longer needs its own separate pool-uniqueness check, since a passing
`validate_collision_free_training_schedule` call already guarantees it;
`report["n_unique_masks"]` is now simply `len(training_schedule["items"])`,
correct by construction. The report also now records
`reserved_composite_ids_fingerprint` and
`schedule_validated_against_live_data_and_reserved_set: True`. Verified
by 2 new tests: a reserved set that's the same SIZE but different
CONTENT than what the schedule was built against (proving the
fingerprint, not a stale count, is checked), and a schedule with one
tampered `realized_query_composite_fingerprints` entry.

**Finding #3 (CONFIRMED): `build_held_out_sample_mask_report`'s
provenance/count checking was self-referential and coarse.** Three
distinct real gaps, all in the same function:
- The strata-fingerprint check compared the bank's OWN recorded
  `strata_fingerprint` against a fingerprint computed FROM the bank's
  own recorded `split_counts`/`split_seeds` -- self-referential, and
  could never catch a bank built with the WRONG counts/seeds for the
  experiment actually being run, only an INTERNALLY inconsistent bank.
- The per-stratum check only required AT LEAST ONE record per stratum,
  and the total-count check alone could not detect a MISDISTRIBUTED
  bank (e.g. 3 records in stratum A + 1 in stratum B totalling the same
  "4" as a correctly-distributed 2+2).
- Records were trusted as truth: the function fingerprinted the STORED
  `context_obs_names`/`query_obs_names` directly, never re-deriving
  them from live coordinates -- a tampered record with internally
  well-formed (non-empty, disjoint) but WRONG barcodes would pass.

Fixed: `build_held_out_sample_mask_report` now REQUIRES
`expected_split_counts`/`expected_split_seeds` from the RESOLVED
EXPERIMENT CONFIGURATION (never read off the supplied bank) and:
explicitly compares the bank's own `split_counts`/`split_seeds` against
these expected values before any fingerprint check; requires EXACTLY
`expected_split_counts[split]` records per stratum (checked
independently per stratum); requires each stratum's record `index`
values to be exactly `{0, ..., count-1}`; requires each record's `seed`
to equal the exact deterministic value
`expected_split_seeds[split] + stratum_index * _STRATUM_SEED_STRIDE +
record_index`; and RE-REALIZES every record via
`realize_seed_and_fingerprint` at its expected seed, comparing fresh
`context_obs_names`/`query_obs_names` against the stored ones for exact
list equality. Verified by 6 new tests: the audit's own "3+1 vs 2+2"
misdistribution example (relabels one record's `stratum` field,
confirms the per-stratum check catches it while the aggregate total
stays unchanged); a tampered record's barcodes caught by re-realization;
a tampered `seed` field; a bank whose `split_counts` don't match the
resolved config; and a bank whose `split_seeds` don't match.

`_REPORT_VERSION` bumped 3 -> 4 (schema changed: training reports gain
`reserved_composite_ids_fingerprint`/
`schedule_validated_against_live_data_and_reserved_set`; held-out
reports gain `expected_split_counts`/`expected_split_seeds`).

**What the audit confirmed as correct, no fix needed:** the sanitized
`X`/`obsm['spatial']`/whitelisted-`.uns` construction itself; the
checkpoint-provenance requirement; the manifest sample-role enforcement;
the empty-held-out-bank rejection; the schedule's reserved-set
fingerprint mechanism itself (only its wiring into the report builder
was missing); `EmptyMaskRealizationError`'s narrow retry scoping.

**No 24-hour run has been started or will be auto-started.**

## 40. Real Gen3 data builder -- Step 5, part 1: per-spot H&E availability and WSI-context schema

Adam's Step 5 instruction ("wire real regional/global GigaPath WSI
context into Architectures 3/4") comes with a long, specific acceptance
list. This section covers only the DATA-LAYER foundation -- the actual
regional/global wiring into `_SharedFieldArchitecture.forward()` (still
`NotImplementedError` for `use_regional_he`/`use_global_slide`), config
changes, and the adversarial hole-vs-visible-tile independence tests are
NOT done yet and are explicitly NOT claimed here. This section is
honest about scope: what follows is real, tested, and pushed, but it is
Step 5's foundation, not its completion.

**Per-spot H&E availability (real gap, confirmed against the actual
code before fixing).** `example_builder.build_spatial_field_example`
previously EXCLUDED a context spot from `observed_*` entirely whenever
its H&E patch footprint physically overlapped the query hole, even
though that spot's GEX is real, measured, and available -- GEX and
imaging are independent real-world failure modes (imaging can fail
locally while transcriptomics stays readable), and dropping the GEX too
throws away real signal for no reason. `_SharedFieldArchitecture`'s own
`modality_flags` input (consumed by `SpotTokenProjection`, already
built) was correspondingly hardcoded to `torch.ones` -- a real flag
input with nothing real behind it yet.

Fixed: every GEX-available context spot is now retained.
`SpatialFieldInputs` gained a new required field
`observed_image_available: np.ndarray` ([n_observed] bool);
`validate_spatial_field_example` enforces its shape/dtype AND
structurally enforces that any spot flagged unavailable has an
EXPLICITLY zeroed `observed_gigapath_features` row -- never a real or
garbage feature value with no flag to distinguish it (this is checked
both directions: a non-zero feature for a flagged-unavailable spot
raises, and a correctly-zeroed one passes).
`build_spatial_field_example` now computes this flag via the
already-audited `slide_context.nonoverlapping_context_patch_mask`,
feeds `image_feature_fn` ONLY the available patches (an unavailable
spot's real pixels are never even touched, modeling the actual
deployment scenario where that image would not exist), and zero-fills
the rest. When EVERY context spot is H&E-unavailable, `image_feature_fn`
has nothing to call at all -- `expected_feature_width` is then required
(raises a clear, actionable error if omitted) so a correctly-shaped
all-zero feature array can still be built. Verified by 4 new tests in
`test_example_builder.py` (retained-but-flagged, zero-feature
enforcement in both directions via `test_example.py`, and the
all-unavailable edge case) plus updates to the 2 pre-existing tests that
asserted the OLD drop-the-spot behavior (`test_novae_graph.py`'s
GEX-availability contrast test updated to reflect that
`build_spatial_field_example` now ALSO retains every GEX-available
spot, differing from Novae's context-only input only in that it tracks
per-spot H&E availability as an explicit flag -- a concept Novae's
input has no field for and does not need).

**WSI-context schema (groundwork for the regional/global wiring still
to come).** `SpatialFieldInputs` gained `full_slide_coord_bounds:
tuple[float,float,float,float] | None` (the complete-slide bounds
`models.slide_encoder.pool_regional_tokens` requires so a regional grid
cell refers to the same physical region across every example on a
slide, regardless of which hole was cut) and `slide_cache_namespace:
str | None` (real cache-key material -- content hash + visible-tile-set
identity -- a caller combines with the loaded GigaPath checkpoint's own
SHA256 to form the complete LongNet cache namespace; this dataclass
never assumes a specific checkpoint). `validate_spatial_field_example`
enforces `wsi_tile_features`/`wsi_tile_coords`/`full_slide_coord_bounds`
are set together (all three or none), that bounds are non-degenerate,
and that every visible tile coordinate actually falls within them (a
visible tile outside the complete-slide bounds would mean the bounds
were computed from stale or mismatched data). Verified by 5 new tests
in `test_example.py`.

**GigaPath cache-key strengthening (`slide_context.py`, one of Step 5's
explicit acceptance items, addressed independently of the regional/
global wiring since it's fully self-contained).** `load_slide_context`'s
`context_id` previously identified a dense WSI tile cache by file
path+size+mtime -- a proxy for content, not content itself (a file
replaced in-place with different bytes at the same size, at a moment
that rounds to the same mtime granularity, would silently collide).
Fixed: `context_id` is now a real SHA256 over the actual tile feature/
coordinate/tile-size bytes. `visible_slide_context`'s own `context_id`
previously appended only the literal `image_mode` string -- two
DIFFERENT query holes on the SAME cached slide produce two different
VISIBLE tile sets (the whole point of `target_zero` filtering) but
would collide on an identical `context_id`, risking a cached LongNet
global vector computed for the WRONG visible-tile set being silently
reused. Fixed: the visible tile coordinates themselves (post-filtering)
are now hashed into `context_id`. Verified by 3 new tests in
`test_slide_context.py` (content-hash binding for the base cache,
visible-set binding for two different holes, and stability for the
same hole).

**Explicitly still open (Step 5 is not closed by this section):**
wiring `pool_regional_tokens`/`FrozenGigaPathSlideEncoder` into
`_SharedFieldArchitecture.forward()` (both branches still raise
`NotImplementedError`); `configs/architecture3.yaml`/`architecture4.yaml`
actually enabling `use_regional_he`/`use_global_slide`; the adversarial
test proving a corrupted query-hole-overlapping tile cannot affect any
model input or prediction while a corrupted VISIBLE tile can; the
structural/test proof that regional/global H&E hidden state never
enters the untouched GEX value-candidate pool (true by construction
today, per `_SharedFieldArchitecture.forward()`'s existing
`shared_expression_parts` composition, but not yet exercised by a
dedicated regression test against the real regional/global wiring since
that wiring doesn't exist yet); Architecture 4 constructing its
conditioner with the checkpoint-SHA256/slide-encoder apparatus threaded
through; and the real A100 smoke-test script (frozen/eval-mode LongNet,
FP16/FlashAttention, bounded memory) -- undeliverable as anything other
than a script for the user to run themselves, per this session's
standing "cannot execute on the remote GPU server directly" constraint.

**No 24-hour run has been started or will be auto-started.**

## 41. Real Gen3 data builder -- Step 5, part 2: real regional/global GigaPath wiring, plus a response to the sixteenth external Codex re-audit (of Step 5 part 1)

Adam forwarded 3 further findings against part 1's data-layer
foundation, then authorized continuing directly into Part 2's real
architecture wiring. All 3 findings re-confirmed against the actual code
before any fix; Part 2 is now genuinely wired end to end, not a
foundation-only checkpoint.

**Finding #1 (CONFIRMED): `slide_cache_namespace` was absent from the
WSI all-or-none validation.** `example.py`'s `wsi_fields_set` tuple only
checked `wsi_tile_features`/`wsi_tile_coords`/`full_slide_coord_bounds`
-- a caller could set every field except `slide_cache_namespace` (left
`None`) and `validate_spatial_field_example` would accept it, silently
losing the real cache-key material a downstream `use_global_slide=True`
forward pass needs. Fixed: `slide_cache_namespace` is now the fourth (of
what became five, see finding #2) required WSI field, plus an explicit
non-empty-string check.

**Finding #2 (CONFIRMED): one `wsi_tile_coords` field conflated two
genuinely different coordinate systems.** Real GigaPath/LongNet needs
its own UNNORMALIZED, native level-0 tile-pixel coordinates for its
pretrained positional encoding; regional spatial attention needs the
SAME centered/spot-spacing-normalized frame as `observed_coords`/
`query_coords`. A single field could never correctly serve both
consumers. Fixed: split into `wsi_tile_native_coords` (raw level-0
pixels, fed to `FrozenGigaPathSlideEncoder`) and
`wsi_tile_regional_coords` (the same `(raw - reference) / scale`
transform as every other coordinate in the example, fed to
`compute_relative_geometry` for regional cross-attention) -- both
derived from the identical transform in `example_builder.py`, never two
independently-computed values that could drift apart.

**Finding #3 (CONFIRMED): the dense-cache content digest covered only
half the fields that actually affect masking.** `load_slide_context`'s
`context_id` hashed `features`/`coords`/`tile_size` only --
`visible_slide_context`'s hole-overlap test runs entirely in the
`mask_coords`/`mask_tile_size`/`coords_are_centers` frame, which for a
`dense_wsi_cache` with a separate `level0_coords`/`level0_tile_size` is
a genuinely DIFFERENT coordinate system than `coords`/`tile_size`. A
cache file changed ONLY in those masking-relevant fields would silently
keep its old `context_id` despite producing different visible tiles for
every hole. Fixed: all six real fields are now hashed together. Also
fixed in the same pass (no duplicate-tile check existed at all): a
`dense_wsi_cache` with two tiles at the identical coordinate now raises
explicitly rather than silently double-counting that region's
contribution to both regional pooling and the LongNet global vector.

Verified by 2 new tests in `test_slide_context.py` (level0-mask-field
digest sensitivity, duplicate-coordinate rejection), 3 new tests in
`test_example.py` (missing/blank `slide_cache_namespace`,
native-vs-regional independent scaling), and every existing WSI test
call site updated to the new five-field schema.

**Part 2: the real regional/global architecture wiring (Adam's Part 2
design, all six bullets satisfied).**

`_SharedFieldArchitecture` (`models/architectures.py`) gained
`_regional_he_tokens`/`_global_slide_vector`, and `forward()`'s two
`NotImplementedError` blocks are now real code:

- *Regional tokens use normalized model coordinates and stable pre-mask
  bounds.* `_regional_he_tokens` pools `wsi_tile_features` via the
  already-audited `pool_regional_tokens` (mean-pooling into a
  `regional_grid_size x regional_grid_size` grid, using
  `full_slide_coord_bounds` -- computed from the COMPLETE tile set
  before hole-filtering, so grid cell (i, j) is slide-stable across
  different holes) against `wsi_tile_regional_coords`. Grid cells with
  zero visible tiles are INDEX-SELECTED out of the returned tensors
  entirely (`available.nonzero(...)`), never zero-valued-but-still-
  attended -- the same discipline `boundary_idx`/`local_idx` already use
  for excluded content elsewhere in this codebase. A new
  `regional_grid_cell_centers` helper (`models/slide_encoder.py`)
  computes each surviving cell's real center in the same frame, fed to
  `compute_relative_geometry` for real regional cross-attention geometry
  (`SpatialFieldBackbone` already had the cross-attention consumer side
  built and unchanged).
- *LongNet uses native GigaPath coordinates.* `_global_slide_vector`
  calls the injected `FrozenGigaPathSlideEncoder` with
  `wsi_tile_native_coords` -- never the regional frame, which would
  silently corrupt LongNet's real pretrained positional encoding.
  Verified directly by a dedicated test that captures the actual
  coordinates the stub encoder receives and confirms they are the
  large-magnitude native values, never the small-magnitude regional
  ones.
- *Regional/global H&E enters hidden conditioning only.* Both branches
  write ONLY into `block_kwargs` (consumed by
  `SpatialFieldBackbone`'s cross-attention/FiLM path), never into
  `shared_hidden_parts`/`shared_expression_parts` (the real GEX
  value-candidate pool `GeneValueTransportHead` predicts from) -- true
  by construction, not a runtime check, mirroring how
  `gex_inducing_expression` IS deliberately added to that pool as a
  contrasting example already in this file. Verified by a dedicated
  structural test: `shared_candidate_expression`/`shared_candidate_hidden`
  have the EXACT same shape whether or not `use_regional_he`/
  `use_global_slide` are enabled (holding `use_global_gex` fixed),
  proving neither branch ever appends a row to the real prediction
  candidate pool.
- *Architecture 3 enables both branches.* `configs/architecture3.yaml`
  now sets `use_regional_he: true`/`use_global_slide: true` (previously
  `false` with "NOT YET WIRED" comments) plus a new `regional_grid_size:
  4` and `data.slide_context_source: dense_wsi_cache` (the real,
  tissue-wide tile grid -- `spot_aligned` stays an explicit diagnostic
  fallback, never used by a production config, per Adam's explicit
  requirement). The Python class-level kwarg defaults on
  `Architecture3.__init__` deliberately stay `False` (an interpretation
  choice, not literally what was asked at the class-default level) --
  flipping the class default would have broken roughly 20 unrelated
  existing tests across this codebase that construct `Architecture3`/
  `Architecture4` without synthetic WSI data; the config is what a real
  training entrypoint actually reads, so "Architecture 3 enables both
  branches" is satisfied at the level that matters operationally.
- *Architecture 4 reuses that exact Architecture 3 conditioner.* A real,
  confirmed gap found while implementing this (not a numbered finding,
  but directly required by this bullet): `Architecture4.__init__`
  previously silently OMITTED `use_regional_he`/`use_global_slide`/
  `global_slide_dim`/`regional_grid_size`/`slide_encoder`/
  `gigapath_checkpoint_sha256` entirely when constructing its internal
  `self.conditioner = Architecture3(...)` -- so `Architecture3`'s own
  `kwargs.setdefault(False)` (and its `global_slide_dim` default) always
  won regardless of what a caller asked Architecture4 for, meaning this
  bullet was never actually true for these fields. Fixed: all six are
  now explicit `Architecture4.__init__` parameters, forwarded verbatim.
  `configs/architecture4.yaml` mirrors architecture3.yaml's flags/
  `regional_grid_size`/`slide_context_source` for the same reason.
- *Architectures 1/2 remain unaffected.* Unchanged at the Python level;
  `configs/architecture1.yaml`/`architecture2.yaml` keep
  `use_regional_he`/`use_global_slide: false` by design (their stale
  "NotImplementedError if set true" comments, now factually wrong since
  the wiring is real, were corrected to say "kept false by design").

**`use_global_slide=True` fails closed at construction, not at forward
time.** `_SharedFieldArchitecture.__init__` now requires (mirroring Step
4's `checkpoint_provenance` required-dict pattern) both a real
`slide_encoder` (`FrozenGigaPathSlideEncoder`) and a non-empty
`gigapath_checkpoint_sha256` whenever `use_global_slide=True`, raising
`ValueError` immediately otherwise -- never a silent default. The
complete LongNet cache namespace the handoff specifies (tile-cache
content hash + visible-tile identity + GigaPath checkpoint SHA256 +
model architecture/version) is assembled at `forward()` time by
combining the data layer's `inputs.slide_cache_namespace` (already real,
see Finding #2 above) with the model-construction-time checkpoint SHA256
and a new `model_architecture_version` string -- properties of WHICH
MODEL is running, correctly supplied by the model layer, never assumed
by the data layer.

**`data/example_builder.py` now actually populates the WSI fields from
real data** -- the missing link connecting the Part 1 schema and Part 2
wiring to a real HEST-1k dense WSI tile cache. `build_spatial_field_example`
gained `slide_context`/`image_mode` parameters; when `slide_context` is
given, it calls the already-audited `slide_context.visible_slide_context`
(removing every tile whose footprint physically overlaps the query
hole, matching the same physical-damage model context H&E patches
already use) and derives `wsi_tile_native_coords`/
`wsi_tile_regional_coords`/`wsi_tile_features`/`slide_cache_namespace`
from the result, plus `full_slide_coord_bounds` from the COMPLETE
(pre-hole) tile set in the same normalized frame. Verified by 3 new
tests in `test_example_builder.py`: real native-vs-regional frame
wiring end to end, and Adam's explicit acceptance criterion --
corrupting a WSI tile that overlaps the hole changes NOTHING in the
built example, while corrupting a visible tile changes
`wsi_tile_features` directly.

**`models/model_factory.py` gained `slide_encoder`/
`gigapath_checkpoint_sha256` passthrough parameters** on
`resolve_model_kwargs`/`build_architecture` -- neither is representable
in static YAML (a live frozen encoder instance and a real checkpoint
hash), so they can never come from `model.params`; a caller (the real
trainer, Step 6, or a test) supplies them directly and they are
forwarded into the constructor unmodified, exactly like `gene_basis`/
`gene_names` already are for Architecture 4. Without this, flipping
`configs/architecture3.yaml`/`architecture4.yaml`'s flags to `true`
would have made `test_model_factory.py`'s real from-YAML construction
tests unable to build a model at all (a real regression this session
caught and fixed by adding this passthrough, not by reverting the
config flags) -- `test_model_factory.py` now builds Architecture 3/4
with a duck-typed stub `slide_encoder` (no real checkpoint needed,
matching every other pluggable-component test stub in this codebase)
whenever it constructs from the real config files.

**Not done in this pass (unchanged from Part 1's honest scope
boundary):** the real trainer (Step 6) that would construct a genuine
`FrozenGigaPathSlideEncoder` from a real checkpoint path and compute its
SHA256; the real A100 smoke-test script (frozen/eval-mode LongNet,
FP16/FlashAttention, bounded memory) -- still undeliverable as anything
other than a script for Adam to run himself, per this session's standing
"cannot execute on the remote GPU server directly" constraint.

**No 24-hour run has been started or will be auto-started.**

## Test status as of this document

```
gen3_multiscale/tests/: 472 passed (45 reused-infra + 23 example-schema +
  11 boundary-graph + 10 slide-context + 7 slide-encoder + 2 debug-plot +
  18 transport-head + 10 tokens + 16 attention + 10 global-context +
  7 harmonic + 7 geometry-utils + 9 backbone + 28 architectures +
  9 gene-basis + 11 flow + 11 losses + 21 metrics + 8 diagnostics +
  27 launch-four-gpu-suite + 36 model-factory + 4 gene-encoder +
  37 mask-schedule + 17 dataset-manifest + 20 example-builder +
  46 mask-fingerprint + 22 novae-graph)
gen2_architectures + gen3_multiscale: 645 passed, 1 skipped
```

The block immediately below (pre-Step-5-part-2 test counts) is kept for
historical continuity rather than deleted, per this document's
append-only discipline:

```
gen3_multiscale/tests/: 459 passed (45 reused-infra + 20 example-schema +
  11 boundary-graph + 8 slide-context + 7 slide-encoder + 2 debug-plot +
  18 transport-head + 10 tokens + 16 attention + 10 global-context +
  7 harmonic + 7 geometry-utils + 9 backbone + 23 architectures +
  9 gene-basis + 11 flow + 11 losses + 21 metrics + 8 diagnostics +
  27 launch-four-gpu-suite + 36 model-factory + 4 gene-encoder +
  37 mask-schedule + 17 dataset-manifest + 17 example-builder +
  46 mask-fingerprint + 22 novae-graph)
gen2_architectures + gen3_multiscale: 632 passed, 1 skipped
```

The block immediately below (pre-Step-5-part-1 test counts) is kept for
historical continuity rather than deleted, per this document's
append-only discipline:

```
gen3_multiscale/tests/: 447 passed (45 reused-infra + 12 example-schema +
  11 boundary-graph + 5 slide-context + 7 slide-encoder + 2 debug-plot +
  18 transport-head + 10 tokens + 16 attention + 10 global-context +
  7 harmonic + 7 geometry-utils + 9 backbone + 23 architectures +
  9 gene-basis + 11 flow + 11 losses + 21 metrics + 8 diagnostics +
  27 launch-four-gpu-suite + 36 model-factory + 4 gene-encoder +
  37 mask-schedule + 17 dataset-manifest + 16 example-builder +
  46 mask-fingerprint + 22 novae-graph)
gen2_architectures + gen3_multiscale: 620 passed, 1 skipped
```

The block immediately below (pre-14th-audit-response test counts) is
kept for historical continuity rather than deleted, per this document's
append-only discipline:

```
gen3_multiscale/tests/: 440 passed (45 reused-infra + 12 example-schema +
  11 boundary-graph + 5 slide-context + 7 slide-encoder + 2 debug-plot +
  18 transport-head + 10 tokens + 16 attention + 10 global-context +
  7 harmonic + 7 geometry-utils + 9 backbone + 23 architectures +
  9 gene-basis + 11 flow + 11 losses + 21 metrics + 8 diagnostics +
  27 launch-four-gpu-suite + 36 model-factory + 4 gene-encoder +
  37 mask-schedule + 17 dataset-manifest + 16 example-builder +
  40 mask-fingerprint + 21 novae-graph)
gen2_architectures + gen3_multiscale: 613 passed, 1 skipped
```

Note: a full monorepo run (`pytest -q` from the repo root, everything
including the top-level `tests/` directory) still shows the same one
pre-existing failure noted since §26,
`tests/test_multi_sample.py::test_inject_multi_sample_n_genes`,
unrelated to `gen2_architectures/` or `gen3_multiscale/`; unchanged and
still out of scope for this pass.

The block immediately below (pre-13th-audit-response test counts) is
kept for historical continuity rather than deleted, per this document's
append-only discipline:

```
gen3_multiscale/tests/: 422 passed (45 reused-infra + 12 example-schema +
  11 boundary-graph + 5 slide-context + 7 slide-encoder + 2 debug-plot +
  18 transport-head + 10 tokens + 16 attention + 10 global-context +
  7 harmonic + 7 geometry-utils + 9 backbone + 23 architectures +
  9 gene-basis + 11 flow + 11 losses + 21 metrics + 8 diagnostics +
  27 launch-four-gpu-suite + 36 model-factory + 4 gene-encoder +
  37 mask-schedule + 16 dataset-manifest + 16 example-builder +
  27 mask-fingerprint + 17 novae-graph)
gen2_architectures + gen3_multiscale: 595 passed, 1 skipped
```

Note: a full monorepo run (`pytest -q` from the repo root, everything
including the top-level `tests/` directory) still shows the same one
pre-existing failure noted since §26,
`tests/test_multi_sample.py::test_inject_multi_sample_n_genes`,
unrelated to `gen2_architectures/` or `gen3_multiscale/`; unchanged and
still out of scope for this pass.

The block immediately below (pre-12th-audit-response test counts) is
kept for historical continuity rather than deleted, per this document's
append-only discipline:

```
gen3_multiscale/tests/: 409 passed (45 reused-infra + 12 example-schema +
  11 boundary-graph + 5 slide-context + 7 slide-encoder + 2 debug-plot +
  18 transport-head + 10 tokens + 16 attention + 10 global-context +
  7 harmonic + 7 geometry-utils + 9 backbone + 23 architectures +
  9 gene-basis + 11 flow + 11 losses + 21 metrics + 8 diagnostics +
  27 launch-four-gpu-suite + 36 model-factory + 4 gene-encoder +
  37 mask-schedule + 15 dataset-manifest + 16 example-builder +
  18 mask-fingerprint + 14 novae-graph)
gen2_architectures + gen3_multiscale: 582 passed, 1 skipped
```

Note: a full monorepo run (`pytest -q` from the repo root, everything
including the top-level `tests/` directory) still shows the same one
pre-existing failure noted since §26,
`tests/test_multi_sample.py::test_inject_multi_sample_n_genes`,
unrelated to `gen2_architectures/` or `gen3_multiscale/`; unchanged and
still out of scope for this pass.

The block immediately below (pre-Step-4 test counts) is kept for
historical continuity rather than deleted, per this document's
append-only discipline:

```
gen3_multiscale/tests/: 395 passed (45 reused-infra + 12 example-schema +
  11 boundary-graph + 5 slide-context + 7 slide-encoder + 2 debug-plot +
  18 transport-head + 10 tokens + 16 attention + 10 global-context +
  7 harmonic + 7 geometry-utils + 9 backbone + 23 architectures +
  9 gene-basis + 11 flow + 11 losses + 21 metrics + 8 diagnostics +
  27 launch-four-gpu-suite + 36 model-factory + 4 gene-encoder +
  37 mask-schedule + 15 dataset-manifest + 16 example-builder +
  18 mask-fingerprint)
gen2_architectures + gen3_multiscale: 568 passed, 1 skipped
```

Note: a full monorepo run (`pytest -q` from the repo root, everything
including the top-level `tests/` directory) still shows the same one
pre-existing failure noted since §26,
`tests/test_multi_sample.py::test_inject_multi_sample_n_genes`,
unrelated to `gen2_architectures/` or `gen3_multiscale/`; unchanged and
still out of scope for this pass.

The block immediately below (pre-11th-audit-response test counts) is
kept for historical continuity rather than deleted, per this document's
append-only discipline:

```
gen3_multiscale/tests/: 381 passed (45 reused-infra + 12 example-schema +
  11 boundary-graph + 5 slide-context + 7 slide-encoder + 2 debug-plot +
  18 transport-head + 10 tokens + 16 attention + 10 global-context +
  7 harmonic + 7 geometry-utils + 9 backbone + 23 architectures +
  9 gene-basis + 11 flow + 11 losses + 21 metrics + 8 diagnostics +
  27 launch-four-gpu-suite + 36 model-factory + 4 gene-encoder +
  37 mask-schedule + 14 dataset-manifest + 14 example-builder +
  7 mask-fingerprint)
gen2_architectures + gen3_multiscale: 554 passed, 1 skipped
```

Note: a full monorepo run (`pytest -q` from the repo root, everything
including the top-level `tests/` directory) still shows the same one
pre-existing failure noted since §26,
`tests/test_multi_sample.py::test_inject_multi_sample_n_genes`,
unrelated to `gen2_architectures/` or `gen3_multiscale/`; unchanged and
still out of scope for this pass.

The block immediately below (pre-Step-3 test counts) is kept for
historical continuity rather than deleted, per this document's
append-only discipline:

```
gen3_multiscale/tests/: 374 passed (45 reused-infra + 12 example-schema +
  11 boundary-graph + 5 slide-context + 7 slide-encoder + 2 debug-plot +
  18 transport-head + 10 tokens + 16 attention + 10 global-context +
  7 harmonic + 7 geometry-utils + 9 backbone + 23 architectures +
  9 gene-basis + 11 flow + 11 losses + 21 metrics + 8 diagnostics +
  27 launch-four-gpu-suite + 36 model-factory + 4 gene-encoder +
  37 mask-schedule + 14 dataset-manifest + 14 example-builder)
gen2_architectures + gen3_multiscale: 547 passed, 1 skipped
```

Note: `reused-infra` (13 checkpoint + 23 hest1k-catalog + 4
query-overlap-report + 5 gene-panel-compatibility) grew by 3 tests this
round: 2 study-namespacing tests plus 1 train-only-compatibility test,
added identically to both `gen3_multiscale` and `gen2_architectures`
copies of `test_hest1k_catalog.py`/`test_gene_panel_compatibility.py`.
`dataset-manifest` grew by 3 (content-provenance regression tests) and
`example-builder` grew by 6 (boundary-hardening regression tests),
both `gen3_multiscale`-only (these two modules have no `gen2_architectures`
copy).

Note: a full monorepo run (`pytest -q` from the repo root, everything
including the top-level `tests/` directory) still shows the same one
pre-existing failure noted since §26,
`tests/test_multi_sample.py::test_inject_multi_sample_n_genes`,
unrelated to `gen2_architectures/` or `gen3_multiscale/`; unchanged and
still out of scope for this pass.

The block immediately below (pre-10th-audit-response test counts) is
kept for historical continuity rather than deleted, per this document's
append-only discipline:

```
gen3_multiscale/tests/: 362 passed (42 reused-infra + 12 example-schema +
  11 boundary-graph + 5 slide-context + 7 slide-encoder + 2 debug-plot +
  18 transport-head + 10 tokens + 16 attention + 10 global-context +
  7 harmonic + 7 geometry-utils + 9 backbone + 23 architectures +
  9 gene-basis + 11 flow + 11 losses + 21 metrics + 8 diagnostics +
  27 launch-four-gpu-suite + 36 model-factory + 4 gene-encoder +
  37 mask-schedule + 11 dataset-manifest + 8 example-builder)
gen2_architectures + gen3_multiscale: 532 passed, 1 skipped
```

Note: `reused-infra` (13 checkpoint + 21 hest1k-catalog + 4
query-overlap-report + 4 gene-panel-compatibility) grew by one test this
round -- `test_resolve_sample_selection_returns_patient_by_sample`,
added identically to both `gen3_multiscale` and `gen2_architectures`'
copies of `test_hest1k_catalog.py`.

Note: a full monorepo run (`pytest -q` from the repo root, everything
including the top-level `tests/` directory) still shows the same one
pre-existing failure noted since §26,
`tests/test_multi_sample.py::test_inject_multi_sample_n_genes`,
unrelated to `gen2_architectures/` or `gen3_multiscale/`; unchanged and
still out of scope for this pass.

The block immediately below (pre-8th-audit-response test counts) is
kept for historical continuity rather than deleted, per this document's
append-only discipline:

```
gen3_multiscale/tests/: 338 passed (41 reused-infra + 12 example-schema +
  11 boundary-graph + 5 slide-context + 7 slide-encoder + 2 debug-plot +
  18 transport-head + 10 tokens + 16 attention + 10 global-context +
  7 harmonic + 7 geometry-utils + 9 backbone + 23 architectures +
  9 gene-basis + 11 flow + 11 losses + 21 metrics + 8 diagnostics +
  25 launch-four-gpu-suite + 34 model-factory + 4 gene-encoder +
  37 mask-schedule)
gen2_architectures + gen3_multiscale: 507 passed, 1 skipped
```

Note: a full monorepo run (`pytest -q` from the repo root, everything
including the top-level `tests/` directory) still shows one additional
pre-existing failure, `tests/test_multi_sample.py::test_inject_multi_sample_n_genes`,
first noted in §26 as confirmed pre-existing on `c02a5d1` (reproduces
under `git stash`) and unrelated to `gen2_architectures/` or
`gen3_multiscale/`; unchanged and still out of scope for this pass.

The block immediately below (pre-7th-audit-response test counts) is
kept for historical continuity rather than deleted, per this document's
append-only discipline:

```
gen3_multiscale/tests/: 332 passed (41 reused-infra + 12 example-schema +
  11 boundary-graph + 5 slide-context + 7 slide-encoder + 2 debug-plot +
  18 transport-head + 10 tokens + 16 attention + 10 global-context +
  7 harmonic + 7 geometry-utils + 9 backbone + 22 architectures +
  9 gene-basis + 11 flow + 11 losses + 21 metrics + 8 diagnostics +
  23 launch-four-gpu-suite + 33 model-factory + 4 gene-encoder +
  35 mask-schedule)
gen2_architectures + gen3_multiscale: 501 passed, 1 skipped
```

Note: this pass also fixed the identical `assert`-based bug (see §27,
remaining contained bug #2) in `gen2_architectures/training/checkpoint.py`,
so `gen2_architectures/tests/` is included in the 501 figure with its
own `test_load_trainable_state_raises_on_genuine_mismatch` updated in
lockstep.

The block immediately below (pre-6th-audit-response test counts) is
kept for historical continuity rather than deleted, per this document's
append-only discipline:

```
gen3_multiscale/tests/: 320 passed (41 reused-infra + 12 example-schema +
  11 boundary-graph + 5 slide-context + 7 slide-encoder + 2 debug-plot +
  18 transport-head + 10 tokens + 16 attention + 10 global-context +
  7 harmonic + 7 geometry-utils + 9 backbone + 19 architectures +
  9 gene-basis + 11 flow + 11 losses + 21 metrics + 8 diagnostics +
  19 launch-four-gpu-suite + 31 model-factory + 4 gene-encoder +
  32 mask-schedule)
gen2_architectures + gen3_multiscale: 489 passed, 1 skipped
```

The block immediately below (pre-5th-audit-response test counts) is
kept for historical continuity rather than deleted, per this document's
append-only discipline:

```
gen3_multiscale/tests/: 310 passed (41 reused-infra + 12 example-schema +
  11 boundary-graph + 5 slide-context + 7 slide-encoder + 2 debug-plot +
  18 transport-head + 10 tokens + 16 attention + 10 global-context +
  7 harmonic + 7 geometry-utils + 9 backbone + 17 architectures +
  9 gene-basis + 11 flow + 11 losses + 21 metrics + 8 diagnostics +
  17 launch-four-gpu-suite + 27 model-factory + 4 gene-encoder +
  30 mask-schedule)
full repo (gen2_architectures + gen3_multiscale): 479 passed, 1 skipped
```

The block immediately below (pre-4th-audit-response test counts) is kept
for historical continuity rather than deleted, per this document's
append-only discipline:

## Test status as of this document

```
gen3_multiscale/tests/: 300 passed (41 reused-infra + 12 example-schema +
  11 boundary-graph + 5 slide-context + 7 slide-encoder + 2 debug-plot +
  18 transport-head + 10 tokens + 16 attention + 10 global-context +
  7 harmonic + 7 geometry-utils + 9 backbone + 17 architectures +
  9 gene-basis + 11 flow + 11 losses + 21 metrics + 8 diagnostics +
  17 launch-four-gpu-suite + 21 model-factory + 4 gene-encoder +
  26 mask-schedule)
full repo (gen2_architectures + gen3_multiscale): 469 passed, 1 skipped
```

The block immediately below (pre-3rd-audit-response test counts) is kept
for historical continuity rather than deleted, per this document's
append-only discipline:

```
gen3_multiscale/tests/: 281 passed (41 reused-infra + 12 example-schema +
  11 boundary-graph + 5 slide-context + 7 slide-encoder + 2 debug-plot +
  18 transport-head + 10 tokens + 16 attention + 10 global-context +
  7 harmonic + 7 geometry-utils + 9 backbone + 18 architectures +
  9 gene-basis + 11 flow + 11 losses + 21 metrics + 8 diagnostics +
  17 launch-four-gpu-suite + 15 model-factory + 4 gene-encoder +
  12 mask-schedule)
full repo (gen2_architectures + gen3_multiscale): 450 passed, 1 skipped
```

The block immediately below (pre-2nd-audit-response test counts) is kept
for historical continuity rather than deleted, per this document's
append-only discipline:

```
gen3_multiscale/tests/: 261 passed (41 reused-infra + 12 example-schema +
  11 boundary-graph + 5 slide-context + 7 slide-encoder + 2 debug-plot +
  18 transport-head + 10 tokens + 16 attention + 10 global-context +
  7 harmonic + 7 geometry-utils + 9 backbone + 16 architectures +
  9 gene-basis + 11 flow + 11 losses + 21 metrics + 8 diagnostics +
  17 launch-four-gpu-suite + 9 model-factory + 4 gene-encoder)
full repo (gen2_architectures + gen3_multiscale): 430 passed, 1 skipped
```

The block immediately below (pre-audit-response test counts) is kept for
historical continuity rather than deleted, per this document's
append-only discipline:

```
gen3_multiscale/tests/: 239 passed (41 reused-infra + 12 example-schema +
  11 boundary-graph + 5 slide-context + 7 slide-encoder + 2 debug-plot +
  14 transport-head + 10 tokens + 16 attention + 10 global-context +
  7 harmonic + 7 geometry-utils + 9 backbone + 11 architectures +
  9 gene-basis + 11 flow + 11 losses + 21 metrics + 8 diagnostics +
  17 launch-four-gpu-suite)
full repo (gen2_architectures + gen3_multiscale): 408 passed, 1 skipped
```
