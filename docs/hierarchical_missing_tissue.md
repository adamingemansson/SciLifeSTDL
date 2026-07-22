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

## Primary references

- Prov-GigaPath: <https://www.nature.com/articles/s41586-024-07441-w>
- Official GigaPath implementation: <https://github.com/prov-gigapath/prov-gigapath>
- STPath: <https://www.nature.com/articles/s41746-025-02020-3>
- Novae: <https://www.nature.com/articles/s41592-025-02899-6>
- STAGATE: <https://www.nature.com/articles/s41467-022-29439-6>
