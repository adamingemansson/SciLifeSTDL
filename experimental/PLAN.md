# Round-4 architecture matrix -- full plan (2026-07-23)

Scope: as many architecturally-distinct, purposeful variants of
`hierarchical_gene_transport_regressor` as can be built and matched into
comparable configs, run consecutively across 4 GPUs at 10k steps each
(shorter than the 20k-step suite, since this round trades per-run depth
for breadth -- with this many comparable runs, the cross-run pattern
matters more than any single run's final decimal). Every config shares
train/val/test slide split (INT1-6/INT7/INT8), optimizer, and schedule
with its comparison point; only the axis(es) under test differ.

Ground rule, restated: every row below answers a specific question. No
config exists "because we can."

## Axes

**A. Gene encoder** (`model.params.gene_encoder_type`, plus a new option
being added this round):
- `weighted_linear` (current default -- verified bit-for-bit STPath's own
  `gene_embed` mechanism, see `experimental/README.md`)
- `mlp` (already implemented, never benchmarked on this model)
- `tokenized` (NEW this round -- wires the existing-but-unused
  `TokenizedGeneEncoder` set-attention-over-genes module into
  `HierarchicalMissingTissueEncoder`; needs an HVG-reduced gene subset,
  same `sc.pp.highly_variable_genes` pattern already used for
  `GeneAttentionDecoder`'s panel derivation)
- `scvi` / `stpath_frozen_table` / `bulkformer` -- deferred, need new
  external dependencies (scvi-tools, or checkpoint downloads) that must be
  installed on st-a100 before they can run at all; code paths will be
  built defensively (fail loudly with an install instruction, not
  silently) but these configs are held out of THIS round's run queue
  until the dependency question is resolved.

**B. Candidate mechanism** (`model.params.use_*_candidate`):
- baseline (k-nearest only, no extra candidate) -- 194/206
- `use_global_candidate` (flat whole-slide mean) -- already built, 215-219
- `use_niche_candidate` (NEW this round -- per-niche mean via BANKSY,
  run per-slide, no cross-slide training needed) -- needs `banksy_py` as
  a new dependency on st-a100; code will be built now, run queued
  provisionally pending that install
- `use_retrieval_candidate` (NEW this round -- BLEEP-style: a small,
  from-scratch, no-new-dependency contrastive image/expression embedding
  trained jointly with the model, used to add embedding-similarity
  candidates alongside the k physically-nearest ones) -- directly tests
  the BLEEP-vs-us tension surfaced in the research pass (BLEEP argues
  physical distance can overfit in data-scarce settings; our own results
  say geometry wins). No external dependency, can run this round.

**C. Already covered by the running suite** (not re-explored
combinatorially this round, held fixed at each config's own baseline
value): `conditioning_mode`, `local_k`, hole size, `fusion_mode`,
`transport_heads`, `use_residual`, `gene_gate_mode`, `use_query_gate`.
Where a NEW axis (A or B) is combined with one of these, it's because the
combination itself answers a question already raised in Round 3 (e.g.
"does a niche candidate still matter once geometry-only scoring is
already isolating position" or "does a better gene encoder change whether
the global candidate helps").

## Config matrix (new configs this round)

Base = C05-equivalent settings (all modalities, concat fusion, 8 heads, no
residual, per-gene gate, query gate on), original hole size, `local_k=128`,
unless the row itself is testing hole size or k.

| # | Gene encoder | Candidate | Other axis | Question answered | Depends on new install? |
|---|---|---|---|---|---|
| 220 | mlp | none | -- | Does a nonlinear-but-from-scratch gene encoder beat linear at all? | no |
| 221 | tokenized | none | -- | Does set-attention over genes beat both linear and MLP? | no |
| 222 | weighted_linear | retrieval | -- | Does content-embedding retrieval add anything on top of k-NN? | no |
| 223 | weighted_linear | retrieval | conditioning_mode=geometry | Does retrieval still help once geometry already dominates scoring? | no |
| 224 | weighted_linear | retrieval | smallhole | Does retrieval matter more when local context is sparse? | no |
| 225 | weighted_linear | retrieval | local_k=256 | Is retrieval redundant with a wider physical neighborhood? | no |
| 226 | mlp | retrieval | -- | Best gene encoder (from 220/221) x retrieval candidate | no |
| 227 | tokenized | retrieval | -- | Best gene encoder x retrieval candidate | no |
| 228 | mlp | global | -- | Best gene encoder x flat global candidate (215's own axis) | no |
| 229 | tokenized | global | -- | Best gene encoder x flat global candidate | no |
| 230 | weighted_linear | niche | -- | Does per-niche pooling beat flat global pooling (215)? | banksy_py |
| 231 | weighted_linear | niche | conditioning_mode=geometry | Niche candidate under geometry-only scoring | banksy_py |
| 232 | weighted_linear | niche | smallhole | Niche candidate where local context is sparsest | banksy_py |
| 233 | weighted_linear | niche | local_k=256 | Niche candidate redundancy with wider k | banksy_py |
| 234 | mlp | niche | -- | Best gene encoder x niche candidate | banksy_py |
| 235 | tokenized | niche | -- | Best gene encoder x niche candidate | banksy_py |

16 new configs (220-235), all matched to the existing suite's eval masks
per the same sharing rules as 216-219 (reuse a sibling's
`evaluation.mask_bank_dir` only when the masking config genuinely
matches; always distinct `experiment_name`/`checkpoint_dir`/
`training_mask_bank_path`). 220-229 (10 configs) need no new
dependencies and can run as soon as the code lands. 230-235 (6 configs)
need `banksy_py` installed on st-a100 first -- queued but held back from
launch until that's confirmed.

Combined with everything already running/queued this session
(165-185 v1, 186-206 v2, 207-214 smallhole/k-sweep, 215-219 global
candidate + combos), that's 60+ configs in the full lineage by the time
this round lands -- real coverage, not padding.

## Build order (this session)

1. **`TokenizedGeneEncoder` wiring** -- DONE. `hierarchical_slide.py`
   gene_encoder_type='tokenized' branch, `inject_tokenized_gene_names` in
   `train.py`, 2 new tests, all 19 prior tests still pass unchanged.
2. **Retrieval candidate (BLEEP-style)** -- DONE.
   `use_retrieval_candidate`/`retrieval_k`/`retrieval_dim`/
   `retrieval_temperature`/`retrieval_loss_weight` on
   `HierarchicalGeneTransportRegressor`; two new linear projections
   (`retrieval_query_projection`, `retrieval_expression_projection`); at
   inference, ranks ALL visible context spots by embedding similarity to
   `query_hidden` and adds the top `retrieval_k` as extra transport-gate
   candidates with REAL relative geometry (not a sentinel, since retrieved
   candidates are genuine positioned spots) via the encoder's own shared
   `relative_coord` module; trained via an in-batch InfoNCE loss (query's
   projection pulled toward its own real target, pushed away from every
   other query's target AND every visible context spot in the same draw).
   `hierarchical_slide.py`'s `forward_with_neighbors` now also exposes
   `context_hidden` (the full per-context-spot fused token, not just the
   k-nearest-gathered or mean-pooled versions) so retrieval candidates get
   the same fused representation every k-nearest neighbor gets. 5 new
   tests (off-by-default, runs-finite, anchor-unaffected, far-context
   reaches prediction, loss zero-when-off/finite-when-on-with-real-
   gradient); all 24 prior tests still pass unchanged (28 total across
   both hierarchical test files).
3. **Niche candidate (BANKSY-based)** -- needs `banksy_py`. Build the
   code path (precompute script + model wiring) defensively now; flag the
   dependency install requirement clearly; configs 230-235 wait on
   confirmation that install succeeded on st-a100 before launch.
4. **10k-step 4-GPU runner** for 220-229 (and 230-235 once unblocked),
   same smoke-test-first pattern as `run_transport_extra_diagnostics_4gpu.sh`.

## GPU/time budget

10k steps is roughly half of the 20k-step suite's per-run cost. With 10
runnable-now configs (220-229) across 4 GPUs (2-3 sequential jobs per
GPU) plus whatever's still finishing from 207-219, this is a genuine
multi-hour-to-overnight batch, consistent with "let it run for a longer
time." 230-235 add a further 6 once BANKSY is confirmed installed.
