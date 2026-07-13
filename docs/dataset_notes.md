# Candidate Datasets

Last updated: 2026-07-13. Shortlisted and ranked by training utility (not
just presence of serial sections) — see `docs/project_outline.md` for the
Track A / Track B definitions these map to. All entries below verified
against live sources.

## Track B (inter-slice 3D) — top 3

Ranked by real depth per specimen (sections/specimen), since that's what
actually yields usable "hold out the middle slice" training examples —
a handful of adjacent-pair sections isn't enough (see DLPFC note below).

### ★ Primary pick: Whole mouse brain spatial atlas
Stereo-seq + snRNA-seq. "Single-cell spatial transcriptomic atlas of the
whole mouse brain," *Neuron* 113(13):2141–2160, 2025.
https://mouse.digital-brain.cn/spatial-omics
- >4M cells, 29,655 genes (whole transcriptome), 308 clusters
- 123 coronal sections after QC, 100 µm intervals, bregma coordinates —
  deepest documented z-series of any candidate, largest volume of
  hold-one-out training pairs
- **Open question:** number of individual animals underlying the atlas not
  confirmed from public search — if it's built from one/few reference
  brains, training on it alone only validates *within-brain* interpolation.
  Pair with MOSTA (below) for a genuine held-out-specimen generalization
  test rather than relying on this dataset alone for both train and eval.

### 2. MOSTA (mouse organogenesis)
Stereo-seq. Chen et al., *Cell* 2022. STOmics ID STDS0000058.
https://db.cngb.org/stomics/mosta/
- 53 sagittal sections spanning E9.5–E16.5, single-cell/bin resolution
- Multiple independent embryos/stages — best source of cross-specimen
  diversity to pair with the whole-brain atlas above
- Most widely benchmarked dataset in the field — easiest to compare against
  prior published numbers

### 3. STARmap PLUS mouse CNS atlas
In situ sequencing. Zeng/Wang lab, *Nature* 2023, "Spatial atlas of the
mouse central nervous system at molecular resolution."
- 1,022 genes (targeted panel, not whole transcriptome), 1.09M cells, 230
  molecular cell types, 106 tissue regions, covers brain + spinal cord
- **Genuinely volumetric** (194×194×345 nm voxels) — not stitched from
  discrete slices. Synthetically carve out training/eval pairs at any
  density with exact continuous ground truth, effectively unlimited
  supervision from one dataset. Best option for rigorous validation, not
  just training.
- Includes disease/injury conditions — worth checking for Track A relevance
  too.

**Demoted, not in top 3:** DLPFC (Visium) — only 2 adjacent-pair sections
per donor (10 µm apart) + one 300 µm jump; no case has 3+ consecutive
sections, so it can't produce a real "predict the missing middle slice"
example. Useful later as a small, fast, well-annotated pipeline sanity
check, not as a training set.

## Track A (intra-slice inpainting) — top 3

Ranked by training volume + tissue diversity, since any complete slice
works — no serial-section requirement.

### ★ Primary pick: HEST-1k
NeurIPS 2024 Datasets & Benchmarks. https://github.com/mahmoodlab/hest
- 1,229 ST samples paired with H&E whole-slide images, 26 organs, 2 species
  (human + mouse), 367 cancer samples across 25 cancer types
- 2.1M expression-morphology pairs, 76M+ nuclei
- Largest and most tissue-diverse candidate — best fit for training one
  general model across tissue types (matches the "general architecture"
  framing of the project). Paired histology available for free if useful
  as auxiliary conditioning (C2-STi-style).

### 2. MOSTA
Same dataset as Track B #2 above — whole-transcriptome, single-cell
resolution, many distinct developmental tissue states. Large volume,
reusable across both tracks without adding a new dependency.

### 3. 10x Genomics Xenium public datasets
E.g. Human Breast Cancer panel, Human Multi-Tissue and Cancer panel.
https://www.10xgenomics.com/datasets
- Single-cell resolution, full cell segmentation, clean whole tissue
  sections
- Good for scaling up once the pipeline works on HEST-1k/MOSTA — higher
  per-cell resolution than either

**Eval-only, not for training volume:** DLPFC — small but has manually
annotated cortical layers, useful as a ground-truth-labeled benchmark for
domain-preservation metrics (ARI/NMI) once a model exists, not for training
volume.

## Likely available through the lab (ask supervisor — probably better than
public data for relevance)

- Developing human heart spatial + single-cell dataset (Lázár, Mauron,
  Andrusivová et al., *Nature Genetics* 57:2756–2771, Oct 2025; code:
  github.com/rmauron/HDCA_heart_dev) — check if sectioning geometry/spacing
  is suitable for Track B.
- Breast cancer spatial transcriptomics + pathology annotations (Li et al.,
  *npj Precision Oncology* 9:310, Sep 2025).
- Any Visium/Visium HD, Xenium, or Stereo-seq runs with known damaged/torn
  sections — would define Track A on *real* (not synthetic) damage, a
  strong differentiator vs. papers that only evaluate on synthetic masks.

## Other verified candidates (not top 3, kept for reference)

| Dataset | Platform | Notes |
|---|---|---|
| P7 mouse brain sagittal atlas | Stereo-seq | 99,365 cells, 41 cell types. STDS0000139. Single section, good for fast prototyping only. |
| ABC Atlas (Allen Institute) | scRNA-seq + MERFISH | ~4.0M cells scRNA-seq + ~4.3M MERFISH. 34→338→1,201→5,322 hierarchical taxonomy. Primarily a reference/annotation resource. |
| STOmics DataBase (general) | various | db.cngb.org/stomics/datasets. Includes CBMSTA cerebellum 3D atlas (*Science* 2024) and ARTISTA axolotl regeneration atlas (*Science* 2022, STDS0000056) — real biological damage/regeneration, interesting for Track A framing but atypical (regrowth, not artifact damage). |
| Open-ST human lymph node | Open-ST (subcellular) | *Cell* 2024. Real serial sections, 350 µm span, human tissue — but only 21 sections total, too shallow for primary training. Candidate if a human Track B pilot is specifically wanted. |

## What to check before committing to a dataset
- [ ] License / data use agreement, especially anything human or
      internal-to-lab.
- [ ] Confirm number of independent specimens (not just total sections) —
      needed to assess generalization, not just training volume.
- [ ] Whether an existing paper already reports numbers on this dataset for
      a comparable task (gives a benchmark to match/beat).
