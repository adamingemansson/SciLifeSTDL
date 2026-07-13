# Environment Setup

What to set up before the next working session, and how the
write-code / run-code loop actually works given where this project lives.

## Quick start (run this first, every new session)

```bash
mamba env create -f environment.yml   # preferred — scanpy/squidpy are easier via conda-forge/bioconda
conda activate st3d
# or, pip-only fallback:
pip install -r requirements.txt

python -c "import torch, pytorch_lightning, scanpy; print('CUDA available:', torch.cuda.is_available())"
```

## Current state, honestly

This remote sandbox (where this repo has been developed so far) has bare
Python 3.11 and **no ML packages installed** — no torch, no
pytorch-lightning, nothing in `requirements.txt`/`environment.yml` beyond
what ships with the container. Every model written so far
(`src/models/registry.py`) has been written but **never executed** here.
That's the main reason to fix first.

## GPU vs. CPU — set expectations correctly

Check with `torch.cuda.is_available()`. This remote environment is very
likely CPU-only. That's fine for one thing and not fine for another:

- **Fine for**: smoke-testing correctness — does the conditioning encoder's
  output shape match what the model expects, does the ODE sampler run
  without crashing, does the VQ codebook lookup work — on tiny synthetic
  data (a handful of fake cells/genes, a few training steps).
- **Not fine for**: real training on an actual pilot dataset (the whole
  mouse brain atlas is >4M cells). That needs real GPU compute — Berzelius/
  UPPMAX/Rackham, already flagged in `README.md`'s tool-stack table. Ask
  your supervisor about allocation before assuming this sandbox can do it.

## The actual workflow this implies

1. **Install packages, every session** (Quick start above) — confirms the
   environment is ready, whether this is a continued session or a fresh
   one (see persistence caveat below).
2. **I write one component** (e.g. the conditioning encoder).
3. **We smoke-test it together, here, on tiny synthetic data** — a few
   fake cells, a few genes, a handful of training steps. Catches shape
   mismatches, API misuse, and obvious logic bugs fast, without needing
   real data or GPU.
4. **Once a component is correctness-verified**, real training on the
   actual pilot dataset happens on proper compute, separately — your
   machine or an HPC allocation, using this same repo and
   `environment.yml` for a consistent environment.
5. **If something breaks during real training elsewhere**, paste the
   error/log back — I can still help debug from that even without direct
   execution access to wherever it's actually running.

## Persistence caveat

This remote environment can be reclaimed after a period of inactivity.
Anything not committed to git — installed packages, scratch files, this
session's shell state — may not survive between sessions. Everything
tracked in git (all of `docs/`, `src/`, `configs/`) is safe regardless.
