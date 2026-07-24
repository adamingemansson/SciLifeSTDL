# Architecture Backlog

Running list of architecture/method ideas that come up in discussion but
aren't being built right now — captured here so they don't get lost,
checked off once actually implemented and run. Unlike
`docs/possible_extensions.md` (broader scoping ideas from early
planning), this is specifically for concrete model/encoder variants
proposed mid-round that we deliberately deferred.

## Open

- [ ] **Gene-attention encoder (self-attention over genes, not spots)**
  — proposed 2026-07-24, lung round.

  Every gene encoder built so far (`MLPGeneEncoder`,
  `UniversalMLPGeneEncoder`) is a single dense projection over the whole
  expression vector at once — it has the raw *capacity* to learn
  gene-gene co-expression structure (each hidden unit is a weighted sum
  across all input genes), but nothing pushes it to do so explicitly.
  STPath's own real gene encoder (`nn.Linear(n_genes, d_model,
  bias=False)`, verified against its source) has the exact same
  limitation — this isn't a gap relative to STPath, it's an open
  question for both.

  The explicit version: treat each gene as its own token (not each
  spot) and let self-attention learn which genes actually relate to
  which — closer to how Geneformer/scGPT-style single-cell foundation
  models handle gene relationships, and structurally different from
  every attention mechanism built this round so far (all of which
  attend over *spots*, not *genes* within one spot's own expression
  vector).

  Real tradeoff flagged before deferring: with only 6 training samples
  in the lung round, a gene-attention layer adds real parameters
  (pairwise attention over ~12,791 or ~39k genes) with very little data
  to constrain it — meaningful overfitting risk on top of the
  sample-size problems already showing up in this round's other
  results (see the HEST-50/HVG-200 panel-metric collapse noted for the
  first lung round results). Worth building once the current 2x2
  (gene-encoder x architecture, `306`/`307`/`308`/`309`) has real
  results to compare against, not before.

  If/when built: would naturally cross with both existing "richness"
  architectures (SpatialTransformer backbone, cross-attention), same
  pattern as the current 2x2 — 2 more configs, not a wholesale redesign.
