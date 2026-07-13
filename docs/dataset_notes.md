# Candidate Datasets (initial scoping, verify licensing/access before use)

## Public, multi-slice / 3D, good for methods development

| Dataset | Platform | Resolution | Notes |
|---|---|---|---|
| Mouse organogenesis (E9.5, E14.5) — MOSTA | Stereo-seq | single-cell/bin | Classic benchmark, used across many methods (MuST, etc.). Good first pilot: dense serial sections through development. https://db.cngb.org/stomics/mosta/ |
| Whole mouse brain spatial atlas | Stereo-seq + snRNA-seq | single-cell | >4M cells, 308 clusters, serial coronal sections w/ bregma coordinates — natural fit for inter-slice 3D gap-filling experiments. https://mouse.digital-brain.cn/spatial-omics |
| P7 mouse brain sagittal atlas | Stereo-seq | single-cell | Smaller, well-annotated (41 cell types), good for fast prototyping. STDS0000139 on STOmics DB. |
| ABC Atlas (Allen Institute) | scRNA-seq + spatial | single-cell | ~4M cells, hierarchical taxonomy (34 classes → 5,322 types); primarily reference/annotation resource, check whether raw serial-section spatial data is included or only cell-type reference. |
| STARmap PLUS mouse CNS atlas | in situ sequencing | true 3D voxel resolution (194×194×345 nm) | Genuinely volumetric, not just serial 2D slices — useful as a "gold standard" 3D dataset to validate reconstruction against, or to synthetically down-sample into sparse slices for training. |
| DLPFC (LIBD human dorsolateral prefrontal cortex) | 10x Visium | spot | 12 slices, widely used spatial-domain benchmark; smaller/easier for pipeline debugging before scaling up. |
| STOmics DataBase (general) | various | various | Browse for additional serial-section / disease datasets: https://db.cngb.org/stomics/datasets — includes cerebellum 3D atlas (mouse/marmoset/macaque), aging atlas, axolotl regeneration atlas (ARTISTA — interesting for "damaged tissue" framing since it's literally regenerating tissue). |

## Likely available through the lab (ask supervisor — probably better than public data for relevance)

- Developing human heart spatial + single-cell dataset (Lázár et al. 2025) —
  check if sectioning geometry/spacing is suitable for 3D reconstruction.
- Breast cancer spatial transcriptomics + pathology annotations (Li et al.
  2025).
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
