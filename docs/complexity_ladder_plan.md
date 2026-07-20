# Post-fixed-Novae complexity ladder

Run this only after the current PDF-launched fixed-Novae batch has finished and
its logs/checkpoints have been preserved. The goal is not another exhaustive
Cartesian product. Each stage asks one question and keeps complexity only when
it improves fixed-mask validation/test results.

## Order

1. **Archive the current run.** Record Git commit, resolved configs, command,
   logs and final metrics. Treat it as a separate historical comparison.
2. **Smoke test.** Verify loading, forward/backward, validation, evaluation and
   checkpoint writing with 300-step overrides.
3. **Baselines.** Global/local mean, nearest, IDW, harmonic, learned mean/sum.
4. **Deterministic ladder.** Harmonic residual with builtin expression,
   GigaPath, StormLite sum, simple concat, MoME, bias variants and sparse GNN.
5. **Missing-image training.** Compare concat/MoME with modality dropout.
6. **Generative value.** Plain FM-OT versus deterministic residual, then
   residual FM-OT using a validated residual-expression autoencoder.
7. **Clean Novae.** Matched MLP/Novae/both on the same 256 unique training
   masks, followed by clean small and larger flagship screens.
8. **Confirmation.** Promote only the best two or three screens to 40k steps
   and seeds 0/1/2.
9. **Held-out samples.** Run finalists with INT1-INT8 train, INT9-INT10
   validation and INT11-INT12 test.

## Commands on eight A100 devices

```bash
# One quick infrastructure check; clean Novae is intentionally excluded.
STAGE=smoke bash scripts/run_complexity_ladder_8gpu.sh

STAGE=baselines     bash scripts/run_complexity_ladder_8gpu.sh
STAGE=deterministic bash scripts/run_complexity_ladder_8gpu.sh
STAGE=robustness    bash scripts/run_complexity_ladder_8gpu.sh
STAGE=generative    bash scripts/run_complexity_ladder_8gpu.sh
STAGE=novae         bash scripts/run_complexity_ladder_8gpu.sh
```

The clean-Novae stage first runs `precompute_context_novae_cache.py`. Only
that expensive matched study cycles a finite, immutable seed bank. For a 10k
screen with `unique_mask_count: 256`, Novae is computed only for 256
context-only graphs, never for the intact slide, and cached for all matched
models. Non-Novae screens keep one fresh masking draw per step.

Use `DRY_RUN=1` to print commands, `FRESH=1` to rerun completed checkpoints and
`CONTINUE_ON_ERROR=1` only when independent jobs should continue after a
failure.

## Selection rules

Retain a component only when it wins on the same masks and seed.

- Every learned model must beat harmonic interpolation.
- Prefer the simpler model unless the added component improves PCC by at least
  0.01 or RMSE by at least 1% without harming `target_zero` materially.
- Keep H&E only as a claimed benefit when `full` beats `shuffled`.
- Keep modality dropout when it improves `target_zero` or `all_zero` while
  preserving most of the `full` result.
- Keep flow matching only when predictive-mean metrics improve or uncertainty
  coverage is useful; training loss alone is not a reason to keep it.
- Keep Novae only when clean context-only Novae or MLP+Novae beats the matched
  MLP run. Historical full-graph scores do not count.
- Do not discard a poor confirmation seed.

Create a wide result table after each stage:

```bash
python scripts/summarize_complexity_ladder.py
column -s, -t reports/complexity_ladder/latest_wide_summary.csv | less -S
```

## Promote winners

Example for two selected screening configs:

```bash
python scripts/promote_complexity_winners.py \
  configs/complexity_ladder/04_hr_stormlite_concat_mlp.yaml \
  configs/complexity_ladder/17_residual_fm_stormlite_mome_mlp.yaml

STAGE=confirm bash scripts/run_complexity_ladder_8gpu.sh
```

Promotion creates matched 40k configs for seeds 0, 1 and 2 under
`configs/complexity_ladder/confirm/`. For a clean-Novae winner, precompute its
new confirmation mask bank before launching confirmation jobs:

```bash
python scripts/precompute_context_novae_cache.py \
  --config configs/complexity_ladder/confirm/<clean-novae-config>.yaml
```

## Promote held-out finalists

After confirmation, generate held-out configs from the winning deterministic
and residual-flow YAML files:

```bash
python scripts/promote_heldout_winners.py \
  configs/complexity_ladder/confirm/<deterministic-winner-seed0>.yaml \
  configs/complexity_ladder/confirm/<residual-flow-winner-seed0>.yaml

STAGE=heldout bash scripts/run_complexity_ladder_8gpu.sh
```

The runner pretrains the training-panel residual autoencoder first when a
residual-flow finalist is present.

## What each stage decides

| Stage | Decision |
|---|---|
| Baselines | Minimum useful performance and harmonic anchor strength |
| Deterministic | Whether images, StormLite, fusion, bias or GNN earn their cost |
| Robustness | Whether training tolerates absent target/all H&E |
| Generative | Whether FM-OT adds value beyond deterministic residual prediction |
| Clean Novae | Whether graph-pretrained gene features still help without leakage |
| Confirmation | Stability across seeds at full budget |
| Held-out | Generalization beyond repeatedly used INT1 masks |
