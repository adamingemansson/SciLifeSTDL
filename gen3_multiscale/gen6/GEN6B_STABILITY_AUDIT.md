# Gen6-B stability, overfitting and seed-reproducibility audit

Scope: Adam's masked-hole project only (`query H&E hidden + query GEX
hidden -> reconstruct query GEX`). This document, and every script it
references, is deliberately silent about the separate supervisor (MK)
project's numbers -- MK keeps query H&E visible and is a different task;
comparing the two would not be a like-for-like comparison.

Reference architecture under audit: **Gen6-B** (weighted-linear GEX
encoder + frozen UNI2 + simple fusion + Fourier coordinates +
spatial-field transformer + gene transport), from
`gen3_multiscale/results/gen6_screen_20260803T211802Z/configs/gen6b.yaml`.
Reference result at the previously-reported best checkpoint (step
52,000): all-gene PCC 0.045384, RMSE 0.315108; HVG-50 PCC 0.165430;
HVG-200 PCC 0.149055.

**Environment note**: this branch was developed in a sandbox with no
access to `/data/adam.ingemansson/...` or the live GPU server. Every
script below was built and tested against small synthetic fixtures, not
the real `gen6b.yaml`/checkpoints/logs -- Phase 1's actual numeric
findings and Phase 3's actual run results are **pending real execution**
by Adam. Nothing in this document states a real Gen6-B stability verdict
or real seed-variance number; the tools that produce those honestly are
what got built and verified here.

## What changed in code

1. **`gen3_multiscale/training/early_stopping.py`** (new) -- resumable
   early stopping for the shared trainer (`training/train.py`, which
   Gen6-B and every other Gen3/4/5/6 deterministic arm already run
   through). `resolve_early_stopping_config(training_cfg)` reads an
   opt-in `training.early_stopping` block (absent/falsy means disabled,
   byte-for-byte unchanged behavior from before this feature existed).
   `early_stopping_status(validation_history, config)` is a **pure**
   function of the full `validation_history.json` list -- recomputed
   fresh after every validation, never a separately incremented counter,
   so the decision is automatically correct across any number of
   resumes without new checkpoint plumbing. Config keys: `monitor`
   (only `"validation_total"` supported), `mode` (`"min"`/`"max"`),
   `patience_validations`, `min_delta`.
2. **`gen3_multiscale/training/train.py`** -- wired in: after each
   validation, `early_stopping_status` recomputes on the just-updated
   history; when `should_stop`, `completion_reason` is set to
   `"early_stopping"` and the loop breaks (mirroring the existing
   `wall_clock_limit_reached` pattern exactly). The hard wall-clock
   limit (`max_wall_clock_hours`) is untouched and still checked first
   on every iteration. Early-stopping state is recorded into every
   checkpoint's `training_state.json` (`extra_metadata["early_stopping"]`)
   and into `run_training`'s returned summary. `best/` checkpoint
   selection is unaffected -- it already tracked `min(validation total)`
   before this change and still does.
3. **`gen3_multiscale/training/core_tensorboard.py`** (new) -- opt-in,
   scalar-only TensorBoard logger for the shared trainer (which had
   zero TensorBoard support before this change; only the separate MK
   `conditional_wae`/`conditional_flow` trainers had one). Gated behind
   `training.tensorboard.log_dir`; reuses the same `purge_step`-on-resume
   discipline already validated for the MK trainers, so a resumed run
   does not leave stale post-crash points in the same TensorBoard run.
   Writes: `train/total`, `train/primary`, `train/rmse_loss`,
   `train/pcc_loss`, `train/gradient`, `train/grad_norm`,
   `train/learning_rate`, `validation/*` (every `validation_history`
   entry field), `early_stopping/best_value`, `early_stopping/best_step`,
   `early_stopping/patience_remaining`, `early_stopping/n_since_improvement`.
4. **`gen3_multiscale/scripts/prepare_gen6b_stability_suite.py`** (new)
   -- given a real, resolved `gen6b.yaml`, writes three immutable configs
   differing ONLY in `training.seed`, `training.checkpoint_dir`,
   `training.max_wall_clock_hours` (forced to the requested cap),
   `training.early_stopping`, `training.tensorboard.log_dir`. Asserts
   this invariant itself (`_assert_matched_except`, checked pairwise
   across all three written configs, not just each against the
   reference) before writing anything -- never trusts the loop body to
   have gotten it right by construction alone. Writes only configs and a
   `run_plan.json`; never starts training.
5. **`gen3_multiscale/scripts/gen6b_stability_audit.py`** (new) -- Phase
   1 tool. Reads a real `validation_history.json` (required) and an
   optional captured stdout training log. Computes: best/final step and
   relative degradation, recent-window validation slope, retained
   checkpoint coverage near the minimum (via `checkpoint.
   list_checkpoint_history`, which only ever reports steps genuinely
   still on disk -- never a fabricated full step range), a retrospective
   application of the SAME early-stopping rule from item 1 to the full
   history (so the audit and the mechanism agree), and -- when a log is
   given -- gradient-norm distribution/spikes, non-finite-crash
   detection (regex over the exact `train.py` raise messages), and the
   train/validation loss gap (trailing train moving average aligned to
   each validation step). Emits an explicit `stable`/`plateaued`/
   `mildly_overfit`/`unstable` verdict with a documented rationale for
   whichever one it picked. Raises rather than fabricating a result when
   `validation_history.json` is missing or empty.
6. **`gen3_multiscale/scripts/summarize_gen6b_seed_stability.py`** (new)
   -- Phase 3 final-evaluation aggregator. Reads three real
   `gen3_evaluator.py` JSON reports (already has per-item patient-level
   CIs and paired mean/nearest-neighbour/harmonic baseline deltas built
   in from prior work -- this script does not reimplement that). Refuses
   to compare seeds whose `n_items` or `dataset_manifest_fingerprint`
   differ. Reports mean +/- SD across seeds for all-gene, HVG-50,
   HVG-200 and CCRCC-50 PCC/RMSE/AUC and for every paired baseline
   delta, best-step variability (from each seed's
   `validation_history.json`), and, when Gen6 B-J reference PCC values
   are supplied via `--reference-arm-pcc` (never hardcoded), whether the
   seed-induced PCC spread exceeds the previously observed
   cross-architecture spread.

Every new module above has a matching test file
(`test_early_stopping.py`, `test_core_tensorboard.py`,
`test_prepare_gen6b_stability_suite.py`, `test_gen6b_stability_audit.py`,
`test_summarize_gen6b_seed_stability.py`, plus new cases in
`test_train.py`), all against small synthetic/fake fixtures -- see each
file for what it actually proves.

## Commands (to run on the real server, in `SciLifeSTDL-gen45`)

Phase 3 setup (writes configs only, never launches):

```bash
python -m gen3_multiscale.scripts.prepare_gen6b_stability_suite \
  --reference-config gen3_multiscale/results/gen6_screen_20260803T211802Z/configs/gen6b.yaml \
  --seeds 0,1,2 \
  --output-root gen3_multiscale/results/gen6b_stability_$(date -u +%Y%m%dT%H%M%SZ) \
  --hours 8 --patience-validations 8 --min-delta 0.0001
```

Launch (pick 3 genuinely free GPUs from `{0,2,3,5}` via `nvidia-smi`
first; the printed `run_plan.json["launch_commands"]` has the exact
per-seed command):

```bash
SCILIFESTDL_CPU_THREADS=8 CUDA_VISIBLE_DEVICES=<gpu> nohup python -m gen3_multiscale.training.train \
  --config <suite_root>/configs/gen6b_seed<N>.yaml \
  > <suite_root>/logs/gen6b_seed<N>.log 2>&1 &
echo $! > <suite_root>/logs/gen6b_seed<N>.pid
```

TensorBoard (one shared root, one run per seed automatically, since each
seed's `tensorboard.log_dir` is a distinct subdirectory of
`<suite_root>/tensorboard/`):

```bash
python -m tensorboard.main --logdir <suite_root>/tensorboard \
  --host 127.0.0.1 --port <dynamically-selected-free-port>
```

Phase 1 audit, once a run has produced real validation history (works on
an in-progress run too):

```bash
python -m gen3_multiscale.scripts.gen6b_stability_audit \
  --checkpoint-dir <suite_root>/checkpoints/seed0 \
  --train-log <suite_root>/logs/gen6b_seed0.log \
  --output <suite_root>/logs/gen6b_seed0_stability_report.json
```

Phase 3 final evaluation, once each seed has finished (never the test
split):

```bash
for seed in 0 1 2; do
  python -m gen3_multiscale.evaluation.gen3_evaluator \
    --config <suite_root>/configs/gen6b_seed${seed}.yaml \
    --checkpoint-dir <suite_root>/checkpoints/seed${seed} \
    --output <suite_root>/logs/gen6b_seed${seed}_evaluation.json \
    --split validation --use-best --device cuda
done

python -m gen3_multiscale.scripts.summarize_gen6b_seed_stability \
  --evaluation-report 0=<suite_root>/logs/gen6b_seed0_evaluation.json \
  --evaluation-report 1=<suite_root>/logs/gen6b_seed1_evaluation.json \
  --evaluation-report 2=<suite_root>/logs/gen6b_seed2_evaluation.json \
  --checkpoint-dir 0=<suite_root>/checkpoints/seed0 \
  --checkpoint-dir 1=<suite_root>/checkpoints/seed1 \
  --checkpoint-dir 2=<suite_root>/checkpoints/seed2 \
  --reference-arm-pcc gen6b=0.045384 \
  --output <suite_root>/logs/gen6b_seed_summary.json
```

(Add one `--reference-arm-pcc gen6X=<pcc>` per other retained Gen6 B-J
result to get the seed-spread-vs-architecture-spread comparison; only
gen6b's own previously-reported number is known to this document.)

## Known limitation carried over from Adam's Phase 1 note

"Repeatedly selecting among many architectures using eight validation
patients creates project-level validation-selection overfitting" --
nothing in this audit's tooling corrects for that; it is a property of
comparing many architectures against the same 8-patient validation set
over time, independent of any single run's own stability. The
seed-spread-vs-architecture-spread comparison in
`summarize_gen6b_seed_stability.py` is the closest this audit gets to
speaking to it directly (if seed noise alone is comparable to or larger
than the spread previously observed across Gen6 B-J, that spread cannot
be read as a real architecture ranking).
