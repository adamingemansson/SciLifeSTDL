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

### How to actually download a HEST-1k sample (2026-07-14, verified)

Full dataset is 825 GB — never download it all. Download one or a few
specific samples by ID or metadata filter instead.

```bash
pip install huggingface-hub
huggingface-cli login   # free account at huggingface.co, generate a token first
```

```python
from huggingface_hub import snapshot_download

# NOTE: earlier drafts of this doc said `from hest.download import download_hest` -
# WRONG, confirmed by inspecting the actual hest package source: no such function
# is exported. Use huggingface_hub's own snapshot_download directly instead - this
# is what that (non-existent, or at least non-public) helper would have wrapped
# anyway. Verified working 2026-07-14: produces data/raw/hest1k/st/INT1.h5ad.
local_dir = "data/raw/hest1k"
snapshot_download(
    repo_id="MahmoodLab/hest",
    repo_type="dataset",
    local_dir=local_dir,
    allow_patterns=["*INT1[_.]**"],   # confirmed Visium (one of 24 ccRCC samples,
                                        # INT1-INT24, fresh-frozen, all Visium)
)
```

Only install/clone the `mahmoodlab/HEST` repo (`pip install -e .`) if you
need the `hest` package's other convenience helpers later (e.g.
`iter_hest`) — not required for `load_hest_sample()` in
`src/data/loaders.py`, which reads the `.h5ad` directly via `anndata`.

Or filter by metadata instead of a fixed ID — `technology` column selects the
platform (`Visium`, `Xenium`, `Visium HD`, legacy `ST`):
```python
import pandas as pd
meta_df = pd.read_csv("hf://datasets/MahmoodLab/hest/HEST_v1_3_0.csv")
meta_df = meta_df[(meta_df["oncotree_code"] == "IDC") & (meta_df["organ"] == "Breast")
                   & (meta_df["technology"] == "Visium")]
ids_to_query = meta_df["id"].values
allow_patterns = [f"*{id}[_.]**" for id in ids_to_query]
```

After download, each sample's expression data lands as a standard scanpy
`.h5ad` under a `st/` subfolder (exact nesting not independently confirmed
byte-for-byte — `src/data/loaders.py`'s `load_hest_sample` searches for the
file by pattern rather than assuming one). Coordinates are already in the
standard `adata.obsm['spatial']` key, so no remapping is needed against the
rest of this repo's pipeline. Full tutorial:
https://github.com/mahmoodlab/HEST/blob/main/tutorials/1-Downloading-HEST-1k.ipynb

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
