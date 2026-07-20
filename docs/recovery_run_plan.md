# Collapse recovery and component-ablation plan (8×A100)

This plan repairs the failed harmonic-residual ladder without replacing the
working project. It keeps the full post-QC gene panel, INT1 screening cohort,
mask geometry, expression preprocessing, FM-OT, StormLite, context-only Novae
and the existing STPath integration.

## What failed

The new `harmonic_residual` head started with an exactly zero output layer and
optimized absolute full-panel MSE. The zero layer initially blocked gradients
to the upstream encoder; the strong harmonic anchor, approximately 16k outputs
and early validation made staying near zero correction an easy solution. The
eight deterministic variants consequently behaved like the same harmonic
model and stopped together at step 4500. This does not demonstrate that the
older FM-OT/StormLite/STPath implementations are broken.

The repair is deliberately narrow:

- small nonzero residual-output initialization;
- direct standardized residual target;
- per-gene scale computed from training expression only;
- correction and upstream-gradient logging;
- delayed/disabled early stopping for the first diagnostic;
- fail-closed requirement to beat the harmonic validation anchor;
- safe formatting of absent image-mode metrics.

## Step 1 — prepare the remote machine

From the repository root after pulling the pushed branch:

```bash
conda activate st3d
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export STPATH_GENE_VOC_PATH=/absolute/path/to/STPath/utils_data/symbol2ensembl.json
export STPATH_MODEL_WEIGHT_PATH=/absolute/path/to/the/pretrained/stpath/checkpoint.pth
export GPU_IDS=0,1,2,3,4,5,6,7
export PYTHON_BIN=python3
```

Set `data.hest_data_dir`/`data.hest_cache_dir` in the recovery configs if the
remote paths differ. Do not change the gene panel or masking parameters.

## Step 2 — remote verification only

Codex did not execute tests or training locally. Run these on the A100 machine:

```bash
git status --short --branch
git rev-parse HEAD
git diff --check
python3 -m pytest -q \
  tests/test_residual_models_audit.py \
  tests/test_validation_sampling_audit.py \
  tests/test_mask_bank_audit.py \
  tests/test_run_comparison.py
```

Optional infrastructure-only smoke (results are not scientific and cannot
promote the ladder):

```bash
SMOKETEST=1 STAGE=repair bash scripts/run_recovery_suite_8gpu.sh
```

## Step 3 — Wave 0: mandatory repair gate

```bash
STAGE=repair bash scripts/run_recovery_suite_8gpu.sh
```

Eight single-GPU jobs run concurrently:

| GPU slot | Run | Isolated question |
|---:|---|---|
| 0 | harmonic anchor | exact deterministic reference |
| 1–3 | repaired builtin residual, seeds 0/1/2 | can the repaired head learn beyond harmonic? |
| 4–6 | repaired StormLite concat, seeds 0/1/2 | does the context/image encoder now reach the output? |
| 7 | fixed-Novae flagship, seed 10 | did the audited branch preserve the older FM-OT path? |

Before training, the context-only Novae cache is sharded across all eight
A100s. The script then calls `check_recovery_gate.py`. It blocks promotion unless all six
residual runs beat their own validation anchor with nontrivial corrections and
the current-audit flagship produces finite, nonconstant, image-sensitive
results. Preserve failed logs; do not launch later stages after a failure.

## Step 4 — Wave 1: FM-OT and STPath controls

Only after Wave 0 prints `RECOVERY GATE: PASS`:

```bash
STAGE=controls bash scripts/run_recovery_suite_8gpu.sh
```

This runs clean builtin/StormLite FM-OT controls, two more current-code
fixed-Novae flagship seeds, and two seeds each of pretrained and unfrozen
STPath with the same custom downstream FM head/decoder. These STPath rows are
the repository's established STPath-conditioned benchmarks; they are not
mislabelled as official zero-shot STPath.

## Step 5 — repaired deterministic ablations

Only after reviewing Wave 1:

```bash
STAGE=deterministic bash scripts/run_complexity_ladder_8gpu.sh
```

The existing full configs `01`–`08` change one encoder/fusion component at a
time: builtin, GigaPath, StormLite sum, concat, MoME, no bias, relative bias and
GNN. Every learned residual config now fails closed unless it beats harmonic.

## Step 6 — later waves, one decision at a time

```bash
STAGE=robustness bash scripts/run_complexity_ladder_8gpu.sh
STAGE=generative bash scripts/run_complexity_ladder_8gpu.sh
STAGE=novae      bash scripts/run_complexity_ladder_8gpu.sh
```

- Robustness (`09`–`10`): whether modality dropout helps missing H&E.
- Generative (`11`–`18`): plain FM-OT, then the dependent residual
  autoencoder and residual FM-OT models.
- Clean Novae (`19`–`23`): MLP/Novae/both on identical context-only masks,
  followed by small/bigger flagship capacity.

Do not run `screen` until the repaired staged path has passed. It intentionally
chains every stage and is unsuitable for diagnosing a failure.

## Step 7 — confirmation and held-out tissue

Promote only components that improve the same masks/seed, using the existing
promotion scripts. Confirm seeds 0/1/2 at 40k, then generate held-out configs:

```bash
python3 scripts/promote_complexity_winners.py <winner-configs...>
STAGE=confirm bash scripts/run_complexity_ladder_8gpu.sh

python3 scripts/promote_heldout_winners.py <confirmed-finalists...>
STAGE=heldout bash scripts/run_complexity_ladder_8gpu.sh
```

Held-out data remain train INT1–INT8, validation INT9–INT10 and final test
INT11–INT12. Test slides do not select the gene vocabulary or checkpoints.

## Interpretation rules

- Keep a component only when it wins on matched masks and seeds.
- Do not discard failed seeds.
- H&E contributes only if `full` meaningfully differs from and outperforms
  `shuffled`; report `target_zero` and `all_zero` too.
- Novae contributes only if context-only Novae beats matched MLP.
- Flow matching contributes only if it beats the deterministic alternative or
  provides useful calibrated uncertainty.
- STPath pretrained versus unfrozen separates pretraining from architecture.
- Never compare the new audited mask-bank scores numerically as if they were
  identical to older single-draw historical metrics.
