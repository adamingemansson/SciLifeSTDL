# Hierarchical missing-tissue model

## Scientific task

Predict expression at a contiguous physically missing region. The model may
use the missing spots' coordinates, but receives neither their GEX nor any H&E
pixels intersecting the region. It observes the remaining WSI and measured GEX
spots from the same section. Validation and test hold out complete samples.

## Data path

1. Normalize library size to 10,000 counts per spot and apply `log1p` once.
2. Draw one contiguous query hole. Query rows are absent from every input GEX
   tensor and from the graph passed to Novae.
3. Remove query-centred H&E patches and every boundary patch that overlaps the
   physical hole.
4. Tile the complete WSI at 0.5 um/px, encode tissue tiles with the frozen
   GigaPath tile encoder, and remove every tile intersecting the hole before
   the frozen GigaPath LongNet slide encoder.

The WSI cache records both GigaPath's normalized positional coordinates and
the raw level-0 WSI coordinates used for physical masking. Loading fails when
ST spot coordinates do not align with the cached WSI.

## Model

- **Raw GEX:** the full shared gene panel is projected using an
  expression-weighted gene-vocabulary embedding (`x @ W`). This is STPath's
  published gene representation and preserves exact observed gene values.
- **Novae:** a frozen context-only representation is computed on the observed
  spatial graph. It is additive and never replaces raw GEX; its role is
  spatial-domain/niche context.
- **Local H&E:** frozen GigaPath spot-tile features are aligned one-to-one with
  observed GEX spots.
- **Global H&E:** frozen GigaPath LongNet provides a mask-specific whole-slide
  morphology vector from visible tiles only.
- **Prediction:** missing-coordinate queries cross-attend only to nearby
  observed multimodal tokens. Query tokens are never used as keys or values.
  A small query Transformer enforces coherence within the missing region, then
  a decoder predicts the complete shared gene panel.

Flow matching is deliberately excluded from this first gate. The conditioning
architecture must beat held-out controls before adding a stochastic generator.

The weighted-linear GEX encoder is likewise a STPath-faithful baseline, not a
claim that it is the best final representation. A follow-up should compare it
under identical masks against (1) a residual nonlinear full-panel encoder and
(2) a gene-token/set encoder that retains gene identity and expression
magnitude without restricting evaluation to HVGs.

## First matched runs

| Run | Visible inputs | Question |
|---|---|---|
| full | slide H&E + local H&E + raw GEX + Novae | primary model |
| no slide | local H&E + raw GEX + Novae | does LongNet add real value? |
| no Novae | slide/local H&E + raw GEX | does Novae add value beyond raw GEX? |
| H&E only | slide/local H&E | how much is predictable without observed GEX? |
| global H&E only | slide H&E | does LongNet carry useful morphology by itself? |
| local H&E only | local spot H&E | does local morphology work without global context? |
| raw GEX only | raw GEX | learned spatial GEX control without Novae or H&E |
| raw GEX + Novae | raw GEX + Novae | does Novae improve expression-only context? |
| harmonic k128 | observed GEX only | non-learned exact-mask baseline |

All runs share samples, mask seeds, target holes, optimizer, decoder, and
20,000 training draws.

## Gene evaluation contract

Every learned model is trained against and predicts the complete shared gene
panel. HVGs do not replace its inputs, targets, loss, or decoder. Evaluation
reports the full-panel PCC/RMSE plus four prespecified views of those same full
predictions:

- the fixed HEST-Bench CCRCC top-50 panel used by STPath, for a like-for-like
  benchmark comparison;
- top 50, 100 and 250 genes ranked by variance across the six training samples
  after the configured normalization/log1p transform.

INT7 and INT8 expression never participates in the train-variance ranking.
The fixed STPath panel is explicitly labelled as an external benchmark panel,
not a leakage-safe model-selection panel. Full-panel metrics remain the main
task result; subset metrics diagnose whether useful variable-gene structure is
being learned but diluted by thousands of sparse or locally constant genes.

## Parallel exact-mask controls

A separate four-GPU suite fills interaction and negative-control gaps without
duplicating the primary eight DGX runs:

| Control | Visible inputs | Purpose |
|---|---|---|
| coordinate only | query/context coordinates | detect spatial-grid shortcuts |
| local H&E + raw GEX | local morphology and observed expression | smallest learned multimodal model |
| global H&E + raw GEX | LongNet slide context and observed expression | isolate global morphology/GEX interaction |
| official STPath | released STPath head, local H&E and context GEX | exact-mask external benchmark stress test |

The official STPath control is evaluated only on genes supported by its
released vocabulary; the report records both that actual count and the full
shared training-panel size. Target-zero H&E remains out of distribution for
released STPath, so this is a task-matched stress test rather than a recreation
of the paper's standard image-to-expression benchmark.

## WSI runtime

The HEST images used here are generic pyramidal tiled TIFFs. Dense WSI
precomputation uses OpenSlide's dedicated generic-TIFF backend; TiffSlide is
retained only as a fallback for other supported slide formats. Install the
official Python binding and bundled library in the training environment:

```bash
python3 -m pip install openslide-bin openslide-python
MAX_JOBS=4 python3 -m pip install flash-attn==2.5.8 --no-build-isolation
```

The launcher first opens INT1, reads its explicit TIFF resolution and one
pixel region, and only then starts GPU workers. MPP is never guessed. Cache
files are written to a temporary path and atomically renamed after all tile
features have been encoded. The training launcher also verifies that
GigaPath's compiled FlashAttention callable is available before starting any
smoke or full jobs. FlashAttention 2.5.8 is the version pinned by the official
Prov-GigaPath environment; build concurrency is capped here to avoid excessive
CPU and RAM use on a shared server.

## Gene-value-preserving transport (hierarchical_gene_transport_regressor)

Held-out results above showed the dense `256 -> 512 -> G` decoder loses to
exact IDW/harmonic interpolation: routing thousands of exact observed gene
values through one 256D bottleneck before decoding back out discards
gene-specific spatial structure a six-slide cohort cannot re-learn from
scratch. `hierarchical_gene_transport_regressor`
(`src/models/registry.py`) keeps `HierarchicalMissingTissueEncoder` as the
conditioner but predicts a weighted combination of untouched observed
full-gene vectors instead: an IDW anchor blended with a learned multi-head,
per-gene-gated transport (`sigmoid(blend_logit)` starts near 0.05, so
training begins close to plain IDW), plus an optional zero-initialized
low-rank residual. See `HierarchicalMissingTissueEncoder.forward_with_neighbors()`
for the exact query/neighbor contract this model consumes, and
`configs/recovery_suite/165-185_transport_*.yaml` +
`scripts/run_transport_suite_4gpu.sh` for the matched 20-run capacity-gate
and held-out ablation suite (O01-O04, C01-C16, harmonic k128 control).

**Suite 1 result (2026-07-22)**: only the simplest configs beat harmonic --
raw-GEX-only (C01, full-panel PCC 0.0399, top-50 HVG PCC 0.3135) and GEX+Novae
(C02) cleared harmonic (0.0386 / 0.2882) on every full-panel and
variance-selected metric; every richness axis tested (Novae vs. not, H&E vs.
not, multimodal neighbor scoring vs. geometry-only, richer gene gates, more
transport heads) made things *worse*, not better, and the primary
"everything on" config C05 lost to harmonic (0.0330 PCC). Root cause,
diagnosed from the training logs themselves: `transport_head_entropy` sat at
~4.81-4.85 (`ln(128)=4.852`, the true maximum for k=128 neighbors) for the
*entire* 20k-step run on every single config -- the multi-head
neighbor-weighting mechanism never learned to specialize at all, which would
explain why no amount of extra conditioning signal could ever get expressed
in the output. The likely cause was `transport_reg_weight`'s own entropy
term (our invented interpretation of the handoff's unspecified "small
transport regularization" -- see `HierarchicalGeneTransportRegressor.
training_step`'s docstring), which explicitly rewards staying uniform.
Fixed by defaulting `transport_reg_weight` to `0.0` (was `1e-3`); the
identical ablation grid was rerun as suite 2 --
`configs/recovery_suite/186-206_transport_*_v2.yaml` +
`scripts/run_transport_suite_v2_4gpu.sh` (`summarize_transport_suite_v2.py`
also prints a direct per-run v1-vs-v2 PCC delta).

**Suite 2 result (2026-07-22)**: the entropy fix was only partially
effective -- `transport_head_entropy` dropped modestly (from ~4.81-4.85 to
~4.61-4.77) but stayed close to the theoretical max, so the
neighbor-weighting mechanism still barely specialized. PCC deltas vs. suite
1 were small and inconsistent across configs; C01 (raw-GEX-only) actually
got *worse*, and no config beat the harmonic control. C07 (geometry-only
scoring) was consistently the strongest learned config in both suites,
suggesting geometric signal dominates and that the six-slide training cohort
may be the more fundamental limiting factor rather than any single richness
axis tested so far.

**Round 3 diagnostics (2026-07-23), still pending results**: three
independent axes, run together via
`scripts/run_transport_extra_diagnostics_4gpu.sh` +
`scripts/summarize_transport_extra_diagnostics.py`:
- **Hole size** (`configs/recovery_suite/207-210_*_smallhole.yaml`):
  `radius_range` shrunk from `[3.0, 6.0]` to `[0.5, 1.0]` spot-spacings
  (~24-36x smaller hole area) for C01/C07/C05 plus a matched harmonic
  control, testing whether hole size itself is capping every config's
  performance.
- **Local neighborhood size** (`configs/recovery_suite/211-214_*.yaml`):
  `local_k`/harmonic's `k` swept 128 -> 256 -> 512 at the original hole
  size, testing whether the model is neighbor-starved rather than
  architecture-limited (motivated by C07's consistent strength in both
  prior suites).
- **Global candidate** (`configs/recovery_suite/215_transport_c05_global_candidate_v2.yaml`,
  `model.params.use_global_candidate: true`): adds exactly one extra
  candidate to the transport gate's softmax competition per query -- a
  whole-slide mean-pooled fallback
  (`HierarchicalMissingTissueEncoder.forward_with_neighbors()`'s new
  `global_hidden` token, paired with the literal mean of every visible
  context spot's real expression) alongside the k local neighbors. The IDW
  anchor is untouched; this only widens the *learned* candidate's options.
  Motivation: pure k-nearest-neighbor conditioning has no way to recover if
  a hole's local neighborhood happens to be unrepresentative of the tissue
  it actually contains -- e.g. a hole straddling a tumor invasive front,
  where expression can shift sharply over a short distance even though the
  missing tissue is still drawn from the same overall section. See
  `HierarchicalGeneTransportRegressor`'s docstring in `src/models/registry.py`
  for the exact mechanism (sentinel relative-geometry entry, not a
  fabricated position). Launch config 215 independently of the
  4-GPU runner above once a GPU is free -- it shares its
  `evaluation.mask_bank_dir` with the original suite 2 C05/harmonic-k128
  runs (194/206) for a direct, same-test-mask comparison.

## Primary references

- Prov-GigaPath: <https://www.nature.com/articles/s41586-024-07441-w>
- Official GigaPath implementation: <https://github.com/prov-gigapath/prov-gigapath>
- STPath: <https://www.nature.com/articles/s41746-025-02020-3>
- Novae: <https://www.nature.com/articles/s41592-025-02899-6>
- STAGATE: <https://www.nature.com/articles/s41467-022-29439-6>
