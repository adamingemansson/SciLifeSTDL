# Candidate Datasets (initial scoping, verify licensing/access before use)

Last verified: 2026-07-13 — all entries below checked against live sources;
all 7 confirmed real with accurate core facts (one soft spot noted at #7).

## Public, multi-slice / 3D, good for methods development

| Dataset | Platform | Resolution | Notes |
|---|---|---|---|
| Mouse organogenesis (E9.5, E14.5) — MOSTA | Stereo-seq | single-cell/bin | Chen et al., *Cell* 2022. 53 sagittal sections spanning E9.5–E16.5. STOmics ID STDS0000058. Classic benchmark, used across many methods (MuST, etc.). Good first pilot: dense serial sections through development. https://db.cngb.org/stomics/mosta/ |
| Whole mouse brain spatial atlas | Stereo-seq + snRNA-seq | single-cell | "Single-cell spatial transcriptomic atlas of the whole mouse brain," *Neuron* 113(13):2141–2160, 2025. Confirmed: >4M cells, 29,655 genes, 308 clusters, coronal sections (10 µm, 100 µm intervals, 123 sections after QC) with bregma coordinates — natural fit for inter-slice 3D gap-filling experiments. https://mouse.digital-brain.cn/spatial-omics |
| P7 mouse brain sagittal atlas | Stereo-seq | single-cell | "A cellular resolution spatial transcriptomic landscape of the postnatal mouse brain" — P7 sagittal section near the midline, 99,365 cells, 41 cell types confirmed. STDS0000139 on STOmics DB. |
| ABC Atlas (Allen Institute) | scRNA-seq + spatial | single-cell | Yao et al., *Nature* 2023, "A high-resolution transcriptomic and spatial atlas of cell types in the whole mouse brain." ~4.0M cells (scRNA-seq QC-passed) + ~4.3M cells via MERFISH spatial data. Hierarchical taxonomy confirmed: 34 classes → 338 subclasses → 1,201 supertypes → 5,322 clusters. Primarily reference/annotation resource — check whether raw serial-section spatial data is included or only cell-type reference. |
| STARmap PLUS mouse CNS atlas | in situ sequencing | true 3D voxel resolution (194×194×345 nm) | Zeng/Wang lab, *Nature* 2023, "Spatial atlas of the mouse central nervous system at molecular resolution." Confirmed: 1,022 genes, 1.09M cells, 230 molecular cell types, 106 tissue regions. Genuinely volumetric, not just serial 2D slices — useful as a "gold standard" 3D dataset to validate reconstruction against, or to synthetically down-sample into sparse slices for training. |
| DLPFC (LIBD human dorsolateral prefrontal cortex) | 10x Visium | spot | Maynard et al., spatialLIBD/HumanPilot project. Confirmed: 12 sections from 3 donors (4 sections each, 2 pairs of adjacent replicates), widely used spatial-domain clustering benchmark; smaller/easier for pipeline debugging before scaling up. |
| STOmics DataBase (general) | various | various | https://db.cngb.org/stomics/datasets is a real, live listing page. Confirmed sub-resources: cerebellum 3D atlas (mouse/marmoset/macaque) = **CBMSTA**, *Science* 2024 (Hao et al.), at db.cngb.org/stomics/cbmsta/; **ARTISTA** axolotl regeneration atlas ("Axolotl Regenerative Telencephalon Interpretation via Spatiotemporal Transcriptomic Atlas," *Science* 2022, Wei et al.), STOmics ID STDS0000056, at db.cngb.org/stomics/artista/ — interesting for "damaged tissue" framing since it's literally regenerating tissue. The "aging atlas" (Stereo-seq spatiotemporal aging atlas across mouse organs, Ma et al.) exists in the literature but its exact STOmics-hosted dataset ID/URL was **not independently confirmed** — verify directly on the STOmics site before citing a specific ID. |

## Likely available through the lab (ask supervisor — probably better than public data for relevance)

- Developing human heart spatial + single-cell dataset (Lázár, Mauron,
  Andrusivová et al., *Nature Genetics* 57:2756–2771, Oct 2025; code:
  github.com/rmauron/HDCA_heart_dev) — check if sectioning geometry/spacing
  is suitable for 3D reconstruction.
- Breast cancer spatial transcriptomics + pathology annotations (Li et al.,
  *npj Precision Oncology* 9:310, Sep 2025 — note: journal is *npj Precision
  Oncology*, not "Nature Precision Oncology").
- Any Visium/Visium HD, Xenium, or Stereo-seq runs with known damaged/torn
  sections — could directly define the "intra-slice damage repair" task with
  real (not synthetic) damage patterns, which would be a strong differentiator
  vs. papers that only evaluate on synthetically masked data.

## What to check for each candidate dataset
- [ ] Number of serial sections and z-spacing (needed for realistic
      "held-out slice" experiments).
- [ ] Whether cell segmentation / single-cell resolution is available, or
      only spot-level (mixed-cell) resolution.
- [ ] Gene panel size (targeted vs. whole transcriptome) — affects which
      generative backbone (categorical/count vs. continuous latent) is
      natural.
- [ ] License / data use agreement, especially for anything human or
      internal-to-lab.
- [ ] Whether an existing paper already reports numbers on this dataset for
      a comparable task (gives you a benchmark to match/beat).
