# ST3D — Generative Reconstruction of Spatial Transcriptomics Volumes

Internship project @ SciLifeLab, Lundeberg Lab.

**Core question:** Can a general, swappable generative architecture reconstruct
missing/damaged spatial transcriptomics (ST) data — either (a) *between*
serial 2D tissue slices to build a continuous 3D volume, or (b) *within* a
slice where tissue is torn/folded/missing?

This repo is the working environment: code, experiment configs, literature
notes, and a running lab notebook, all in one place so the project is easy to
pick up, hand off, or write up later.

## How this workspace is organized

```
st3d-project/
├── README.md                 <- you are here
├── environment.yml            <- conda env (primary)
├── requirements.txt           <- pip fallback / for pip-only installs
├── docs/
│   ├── project_proposal.md    <- 1-pager: aim, scope, success criteria
│   ├── project_outline.md     <- phased execution roadmap, start here
│   ├── literature_review.md   <- running annotated bibliography
│   ├── dataset_notes.md       <- candidate datasets, access, pros/cons
│   ├── metrics_notes.md       <- evaluation metrics + FID-style metric plan
│   └── weekly_log_template.md <- copy into notes/ each week
├── notes/                     <- dated lab notebook, one .md per day/session
├── configs/                   <- YAML configs per experiment (model + data + training)
├── src/
│   ├── data/                  <- loading, preprocessing, masking/gap simulation
│   ├── models/                <- model registry + individual architectures
│   ├── training/               <- train/eval loops, Lightning or plain PyTorch
│   ├── evaluation/             <- metrics incl. custom FID-style score
│   └── utils/                  <- misc (viz, logging, seeding)
├── notebooks/                  <- exploratory analysis (keep heavy compute in src/)
├── data/                        <- raw/ (read-only, gitignored) + processed/ (cache)
├── results/                     <- generated outputs, checkpoints, figures
└── logs/                        <- training logs, tensorboard/wandb local mirrors
```

## Suggested tool stack

| Purpose | Tool | Why |
|---|---|---|
| Version control | Git + GitHub/GitLab (SciLifeLab often has internal GitLab) | mandatory for a multi-month project |
| Environment | conda/mamba (`environment.yml`) | ST tooling (scanpy, squidpy) is easier via conda-forge/bioconda |
| Experiment tracking | Weights & Biases (free academic) or MLflow (self-hosted, no external data egress) | check with lab on data-sharing policy before using a cloud tracker with real patient data |
| Notes | Markdown files in `notes/` (this repo) + Obsidian or plain editor for cross-linking | keeps notes versioned with code; no separate app *required* |
| Reference manager | Zotero (free, tag-based, exports BibTeX) | feed `docs/literature_review.md` from it |
| Compute | SciLifeLab likely has access to Berzelius / UPPMAX / Rackham (SNIC/NAISS clusters) | ask your supervisor for allocation; design code to run on SLURM early |
| Core ST libraries | `scanpy`, `squidpy`, `anndata`, `spatialdata` | standard ST data model (AnnData) — build everything around this |
| DL framework | PyTorch + PyTorch Lightning (or plain PyTorch if you prefer full control) | most spatial/generative baselines below are PyTorch |

## Getting started

```bash
# 1. clone / init git
git init
git add .
git commit -m "init workspace"

# 2. create environment
mamba env create -f environment.yml
conda activate st3d

# 3. sanity check
python -c "import scanpy, torch; print(torch.cuda.is_available())"
```


## Validity-audit experiment suite

The clean benchmark path and the staged four-device runner are documented in
[`docs/audit_fixes_and_experiment_suite.md`](docs/audit_fixes_and_experiment_suite.md).
After installing the environment and making the HEST-1k data/caches available:

```bash
# Print every wave without launching training.
DRY_RUN=1 DEVICES="0 1 2 3" bash scripts/run_audit_suite_4gpu.sh

# Run consecutive waves, with at most one job on each of four GPUs.
DEVICES="0 1 2 3" bash scripts/run_audit_suite_4gpu.sh
```

The runner first validates harmonic-residual autoencoders, then executes
nonparametric baselines, clean controls, deterministic residual models,
residual flow matching, matched seeds and held-out-slide experiments.

## Project phases (see docs/project_proposal.md for detail)

1. **Scoping** — pick the concrete sub-problem (3D inter-slice gap filling vs.
   intra-slice damage repair vs. both), pick 1-2 pilot datasets.
2. **Literature + baselines** — survey existing methods, pick 2-3 to
   reproduce/benchmark as reference points.
3. **Metrics** — assemble a metric suite; prototype a custom distributional
   ("FID-style") metric for ST.
4. **Architecture design** — build a model-agnostic pipeline (data → masking
   simulator → swappable generator → evaluation) so you can drop in different
   generative backbones (diffusion / GAN / VAE / flow / GNN-based) without
   rewriting the pipeline.
5. **Experiments & iteration** — train, evaluate, write up.

See `docs/literature_review.md` and `docs/dataset_notes.md` for a first pass
already filled in from initial literature scoping (July 2026) — treat it as a
starting point, not the final word.
