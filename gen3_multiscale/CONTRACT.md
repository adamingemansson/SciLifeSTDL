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

## Test status as of this document

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
