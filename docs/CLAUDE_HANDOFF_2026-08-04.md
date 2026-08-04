# Claude handoff — SciLifeSTDL Gen3–6 and MK projects

**Snapshot:** 2026-08-04, approximately 14:00 CEST. Live process state will
change; verify it before acting. This document describes what is implemented,
what has actually been run, the trustworthy results so far, and the next audit
work. It is intentionally concise.

## 1. Non-negotiable working rules

- Preserve all user files and active run directories. Several worktrees are
  detached and contain live checkpoints/results.
- Do not start, stop, resume, evaluate, or modify a long run without Adam's
  explicit instruction.
- Never use validation/test target GEX as an inference input. Test remains
  locked until model selection is frozen.
- Do not compare numbers unless manifest, split, ordered gene panel,
  preprocessing, masks, encoder weights, and cache fingerprints match.
- Train the full 17,189-gene panel. HVG-50/HVG-200 are evaluation views only.
- Make bounded commits on a dedicated branch and give Codex the commit for an
  independent audit before any new long launch.
- On the server, `watch` has repeatedly segfaulted. Use a shell `while` loop.
  Logs are append-only, so old tracebacks may precede a healthy resumed run.
  Resolve the active log with `readlink /proc/PID/fd/1`.
- `queue_status.json` can be stale after manual retries. `/proc/PID`, the full
  command line, actual stdout, and advancing steps are authoritative.

## 2. Repositories, branches, and current code

Primary development branch:

```text
origin/codex/conditional-wae-gex
current tip: f74b0c7
```

Important recent commits:

```text
f74b0c7  purge stale TensorBoard steps on resume
3c57899  deterministic undersized-mask resampling + CPU caps
a0be111  matched MK conditional latent-flow suite
2922866  MK TensorBoard diagnostics
c9ce676  conditional WAE GEX suite
910f50f  require Gen6 HVG evaluation panels
d61c4f1  TensorBoard history bridge
b95825f  batch-independent flow evaluation samples
1c56e39  bounded Gen4/5 evaluation memory
d3a4fc8  Gen6 audit/hardening
2121c98  runnable Gen6 component screen
4aa35ae  Gen3-release-compatible no-image evaluation
725d591  zero-shot pretrained STPath benchmark
9bb8790  faster Gen3 evaluation/retryable basis fitting
a67dc43  safe batched scFoundation caching
6cbe727  mandatory Gen4/5 preflight and real train→checkpoint→evaluate proof
```

Server worktrees:

```text
/data/adam.ingemansson/SciLifeSTDL          Gen3 release/results
/data/adam.ingemansson/SciLifeSTDL-gen45    Gen4/5/6 runs
/data/adam.ingemansson/SciLifeSTDL-MK       supervisor WAE runs
/data/adam.ingemansson/SciLifeSTDL-MK-flow  supervisor conditional-flow runs
```

Do not use ordinary `git pull` in a detached worktree. Fetch the intended
branch and switch/cherry-pick only after confirming no tracked local changes.

## 3. Two different scientific tasks — do not conflate them

### A. Gen3–6 physical-hole completion (Adam's main scientific interest)

Input: surrounding H&E, surrounding measured GEX, query coordinates, local
boundary and optional regional/global context. Both query-region H&E and query
GEX are unavailable. H&E patches and dense WSI tiles overlapping the physical
hole are excluded. Output: full-panel GEX jointly across the hole.

### B. MK supervisor task

Task I is **full visible H&E → query ST**. Task II is **full visible H&E plus
legal surrounding ST → masked query ST**. Query GEX is always hidden, but query
H&E deliberately remains visible. This is not the physical-hole contract.

## 4. Data and evaluation contract

Manifest:

```text
data/cache/hest1k/gen3_multiorgan_80_qc2_manifest.json
```

- 56 retained Kidney/Lung/Liver/Bowel samples: 37 train, 9 validation, 10
  locked test; patient-disjoint splits.
- 17,189 training-derived common genes; 90,928 training spots used to derive
  evaluation panels.
- Excluded for WSI/ST misalignment: NCBI827/828/830/831/832/833 and TENX72.
- Missing H&E never removes a GEX-valid spot: it remains with
  `image_source_available=False` and a zero image placeholder.
- Training uses 500 deterministic, fingerprinted masks per retained training
  slide. Standard reported validation used 8 masks per stratum across four
  strata: 32 items/sample, 288 items over 9 validation samples (8 patients).
- Primary metrics: patient-aggregated full-panel PCC and RMSE with 95% CI.
  Also report train-derived HVG-50/HVG-200, nonzero AUC, mask strata, and
  matched mean/nearest/harmonic baselines. Never select using test.
- UNI2 and scFoundation caches were verified at 56/56 retained samples.

## 5. Architecture map

### Gen3

Shared: 512 hidden, 8 heads, 4 spatial-transformer blocks, relative geometry,
joint query tokens, local context plus complete boundary rings, and
gene-value-preserving transport over all 17,189 genes.

1. **Arch1:** weighted GEX encoder + spot GigaPath + anchor-free boundary-field
   transformer + transport head.
2. **Arch2:** Arch1 plus an explicit harmonic anchor/blend.
3. **Arch3:** Arch1 plus all-observed-GEX inducing tokens, regional dense-WSI
   GigaPath tokens, and mask-aware frozen LongNet global slide context.
4. **Arch4:** frozen selected Arch3 deterministic mean plus conditional flow
   in a rank-64 training-residual gene basis. It is residual uncertainty, not
   a full-expression decoder.

### Gen4 conditioner + residual-flow arms

```text
gen4c  UNI2 + scFoundation
gen4b  GigaPath + scFoundation
gen4d  STPath joint conditioner
gen4e  STPath + UNI2 + scFoundation
gen4a  optional UNI2 + weighted-linear baseline
```

Each deterministic conditioner is trained first. A rank-64 training-only
residual basis is then fit against its exact selected checkpoint, followed by
the matched residual flow.

### Gen5 full-latent flows

The same c/b/d/e conditioners feed one shared training-only expression
autoencoder (17,189→1024→z=256→1024→17,189). Conditioner and autoencoder are
frozen; a velocity network generates the complete expression latent. This
design performed poorly (results below).

### Gen6 component screen

```text
gen6a  pretrained STPath, fully unfrozen, native full-panel head
gen6b  weighted-linear GEX + UNI2 + simple fusion
gen6c  scFoundation + UNI2 + simple fusion
gen6d  weighted-linear GEX + GigaPath/LongNet + simple fusion
gen6e  scFoundation + GigaPath/LongNet + simple fusion
gen6f  scFoundation + UNI2 + MoME + frame averaging
gen6g  same encoders/MoME + learned relative-geometry bias
gen6h  same encoders/MoME + Fourier relative attention
gen6i  same encoders + bidirectional image/GEX cross-attention
gen6j  full local/boundary/regional/global spatial-field transformer
gen6k  frozen Gen6C + shared autoencoder + minibatch-OT latent flow (staged)
gen6l  frozen Gen6C + conditional WAE-GAN (staged)
```

Gen6 B:E is the encoder 2×2. F:J fixes scFoundation+UNI2 to isolate
fusion/geometry. Deterministic arms use the existing full-panel transport
objective and trainer/evaluator.

### MK WAE arms

Shared Architecture-1-derived image/spatial conditioner, z=256 target encoder,
1024-wide expression autoencoder/decoder, explicit conditional-mean head,
8 inference samples, full-panel RMSE+PCC reconstruction, and prior weight 0.1.

```text
wae_he_mmd     H&E → ST; IMQ-MMD aggregate-posterior matching
wae_he_gan     H&E → ST; adversarial latent prior matching
wae_he_st_mmd  H&E + surrounding ST → masked ST; MMD
wae_he_st_gan  H&E + surrounding ST → masked ST; GAN
```

True query GEX enters the expression encoder only during training. Inference
uses z~N(0,I) plus legal context. Target GEX cannot affect checkpoint selection
through the diagnostic TensorBoard posterior view.

### MK matched conditional-flow arms

Same data, conditioner, target encoder, decoder, masks, metrics and diagnostics
as WAE. The flow target is stop-gradient; inference integrates Gaussian noise
through a 2-block conditional velocity network for 20 ODE steps.

```text
flow_he        H&E → ST; independent rectified-flow coupling
flow_he_ot     H&E → ST; Sinkhorn-OT coupling
flow_he_st     H&E + surrounding ST → masked ST; independent coupling
flow_he_st_ot  H&E + surrounding ST → masked ST; Sinkhorn-OT coupling
```

## 6. Trustworthy completed validation results

All tables below are validation-only, patient aggregated (`N=8` patients).
PCC columns are full panel / HVG-50 / HVG-200; RMSE is full-panel.

### Gen3 and matched pretrained STPath benchmark

| Model | PCC all | PCC 50 | PCC 200 | RMSE all | AUC |
|---|---:|---:|---:|---:|---:|
| Gen3 Arch1 | 0.042623 | 0.164614 | 0.146833 | 0.315068 | 0.870151 |
| Gen3 Arch2 | 0.037780 | 0.178200 | 0.147424 | 0.315939 | 0.868437 |
| Gen3 Arch3 | 0.038518 | 0.162667 | 0.139878 | 0.314597 | 0.873932 |
| Gen3 Arch4 | 0.035365 | 0.108053 | 0.096176 | 0.315597 | 0.872600 |
| pretrained STPath | 0.010544 | 0.107851 | 0.091163 | 0.410139 | 0.831053 |

Arch4 uncertainty is badly under-dispersed: nominal 68/90/95% coverage was
0.317/0.399/0.432 and standardized residual `z_std=20.24`.

### Gen3 no-image ablation

Full-panel PCC without image was Arch1 0.041677, Arch2 0.036388, Arch3
0.038025, Arch4 0.034509. These are almost unchanged from normal evaluation.
Thus these trained checkpoints use little measurable image signal; this does
not establish that morphology is intrinsically uninformative.

### Gen4 deterministic conditioners

| Arm | PCC all | PCC 50 | PCC 200 | RMSE all | AUC |
|---|---:|---:|---:|---:|---:|
| gen4c | 0.041034 | 0.125673 | 0.113454 | 0.314759 | 0.873232 |
| gen4b | 0.038385 | 0.119084 | 0.107284 | 0.315094 | 0.872472 |
| gen4d | 0.042657 | 0.128466 | 0.115519 | 0.314611 | 0.871802 |
| gen4e | 0.043455 | 0.134729 | 0.122118 | 0.314398 | 0.873535 |

Gen4 flow uncertainty was also severely under-dispersed (`z_std≈19–21`).

### Gen5 full-latent flows

| Arm | PCC all | RMSE all | AUC |
|---|---:|---:|---:|
| gen5c | 0.001295 | 0.362805 | 0.816785 |
| gen5b | 0.000989 | 0.365371 | 0.821843 |
| gen5d | 0.000959 | 0.372368 | 0.817577 |
| gen5e | 0.000902 | 0.371078 | 0.808275 |

Conclusion: the current shared-autoencoder full-latent flow destroys the
predictive signal and should not be promoted.

### Gen6 A–D after 8-hour component screen

| Arm | PCC all | PCC 50 | PCC 200 | RMSE all | AUC |
|---|---:|---:|---:|---:|---:|
| gen6a | **0.058736** | 0.137920 | 0.133473 | 0.322381 | 0.864305 |
| gen6b | 0.045384 | 0.165430 | 0.149055 | 0.315108 | 0.870503 |
| gen6c | 0.047113 | 0.168406 | **0.152351** | 0.315171 | 0.870690 |
| gen6d | 0.044333 | **0.168842** | 0.150719 | 0.315314 | 0.870281 |

Gen6A has the highest full-panel PCC but worse RMSE; B–D are close and their
CIs overlap. These are component-screen signals, not a final winner.

## 7. Live state at this snapshot

### Gen6

Root:

```text
/data/adam.ingemansson/SciLifeSTDL-gen45/gen3_multiscale/results/gen6_screen_20260803T211802Z
```

- gen6a–d finished and were evaluated above.
- Actual live manual retries: gen6e GPU0 PID 4053088 (~step 41,400), gen6f
  GPU2 PID 4150490 (~31,800), gen6g GPU3 PID 4150408 (~33,150), gen6h
  GPU5 PID 301656 (~26,000).
- gen6h writes to `logs/gen6h_gpu5_resume_20260804T102904Z.log`, not the old
  `gen6h.log`; it validated at step 24,000 (`total=0.483038`) and checkpointed.
- Original queue controller PID 3505756 waits for gen6e and should then launch
  gen6i. Its 0% CPU is expected. The original GPU2 queue will not launch gen6j
  after the manual gen6f retry; explicitly queue gen6j after gen6f unless a
  separate wrapper is verified.
- `queue_status.json` retains old failures for f/g/h and is not authoritative
  for their manual retries.

### MK WAE and conditional flow

Pointers:

```text
/data/adam.ingemansson/SciLifeSTDL-MK/gen3_multiscale/results/LATEST_MK_ROOT.txt
/data/adam.ingemansson/SciLifeSTDL-MK-flow/gen3_multiscale/results/LATEST_CONDITIONAL_FLOW_SUITE_ROOT.txt
```

The initial H&E-only WAE arms crashed around step 3,200 because a legitimate
sparse-edge mask contained one query spot and the latent distribution loss
requires at least two. Commit 3c57899 deterministically advances to the next
eligible mask, fails only if the entire dataset is invalid, and applies CPU
caps. Commit f74b0c7 purges stale post-checkpoint TensorBoard steps on resume.

All WAE/flow arms were restarted from their latest checkpoint with the same
TensorBoard directories and explicit code-drift acknowledgement:

```text
WAE:  12 CPU threads/arm
Flow:  6 CPU threads/arm
GPUs:  0,2,3,5 respectively
```

Verify actual current arms/PIDs before acting. Synthetic package smokes and
real-data one-step WAE smokes passed for all four arms. Conditional-flow
synthetic smokes passed for all four arms; real runs loaded all required
samples, passed validation-boundary preflight, and entered training.

TensorBoard is intentionally separate for historical Gen3–6, MK WAE, and MK
flow. Generated configs log scalars, bounded Projector cohorts, H&E thumbnails,
spatial PCA/error/gene maps, and validation metrics. Keep those directories.

## 8. Testing and important fixes already exercised

- At Gen4/5 runnable integration commit 6cbe727, the reported full suite was
  940 passed with one pre-existing unrelated deselection.
- Gen4/5 synthetic conditioner/flow smokes passed all primary arms; a genuine
  non-smoke train→checkpoint→evaluator round trip passed.
- scFoundation caching originally exceeded 70 GB RAM and was extremely slow.
  Batched official-pipeline inference and bounded row workspaces now cover all
  56 samples with approximately 8 GB observed RAM in the corrected run.
- Architecture-4/Gen4 residual basis fitting initially took many CPU hours and
  hit MKL/SVD problems. Finite bounded iteration, reusable verified residual
  matrices, and GPU randomized SVD were added. Gen3 rank-64 basis passed shape,
  finiteness, SHA, and orthogonality error `1.79e-5` over 90,447 residual rows.
- Gen3 evaluation originally took >13 hours and emitted thousands of constant
  input warnings. Evaluation was batched/bounded; constant genes are handled
  without changing the scientific metric.
- Gen4/5 evaluation was later bounded for memory and made batch-independent.
- Gen6 synthetic smokes B–J passed. A–D full evaluation was rerun after making
  HVG-50/HVG-200 a mandatory panel contract.
- The newest CPU-cap/mask-resampling patch was syntax/diff checked locally.
  Runtime verification is the current resumed server runs; inspect their new
  log sections rather than old appended tracebacks.

## 9. What Claude should do next, after Adam confirms runs are finished

1. **Do not start with implementation.** Inventory every actual process,
   actual stdout file, final checkpoint, completion reason, and validation
   history for Gen6 E–J, MK WAE, and MK flow.
2. **Audit completion integrity:** code/manifest/cache fingerprints, resumed
   optimizer/RNG state, checkpoint step monotonicity, TensorBoard purge/resume,
   no stale status/log confusion, and no silent undersized-mask bias. Quantify
   how often masks were skipped and which strata/samples produced them.
3. **Evaluate Gen6 E–J** with the same 288 fixed validation items and mandatory
   full/HVG50/HVG200 panels. Report patient CIs and matched baselines. Do not
   touch test. Evaluate gen6j only after its sequential run completes.
4. **Evaluate MK WAE and flow** with their dedicated evaluators. For each task,
   compare sampled prediction, deterministic conditional mean, MMD vs GAN,
   independent vs OT flow, full/HVG panels, calibration/diversity, and legal
   conditioning ablations. H&E-only and H&E+surrounding-ST are different tasks;
   do not rank them as though inputs were identical.
5. **Scientific leakage audit:** prove query GEX is absent at MK inference;
   Task II observed-expression indices exclude every query; target posterior
   is TensorBoard diagnostic only; full H&E visibility is intentional for MK;
   Gen3–6 still remove query-region H&E.
6. **Convergence audit:** use validation history/TensorBoard, not noisy training
   batches. Determine whether 8 hours was sufficient and select the immutable
   best validation bundle, not the final step.
7. **Decision memo:** identify component effects with uncertainty. Do not claim
   improvements from point estimates with overlapping patient CIs. Explicitly
   address the near-null Gen3 image ablation and the Gen5 collapse.
8. Commit audit scripts/tests/results extraction separately and give Codex the
   commit for adversarial review. Do not launch Gen6 K/L or any new architecture
   until Adam reviews the deterministic and MK results.

## 10. Key files

```text
HANDOFF_20260730.md                              Gen3 release details
gen3_multiscale/MANUAL_RELEASE_RUNBOOK.md        Gen3 staged procedure
gen3_multiscale/gen4/RUNBOOK.md                  Gen4 contract
gen3_multiscale/gen5/RUNBOOK.md                  Gen5 contract
gen3_multiscale/gen6/contract.py                 canonical Gen6 arm table
gen3_multiscale/gen6/RUNBOOK.md                  Gen6 execution/evaluation
gen3_multiscale/conditional_wae/RUNBOOK.md       MK WAE scientific contract
gen3_multiscale/conditional_flow/RUNBOOK.md      MK flow contract
gen3_multiscale/training/train_conditional_wae.py
gen3_multiscale/training/train_conditional_flow.py
gen3_multiscale/evaluation/conditional_wae_evaluator.py
gen3_multiscale/evaluation/conditional_flow_evaluator.py
```

The present scientific picture is: Gen3/4 detect modest held-out spatial
signal but currently exploit little image information; Gen5 full-latent flow
failed; early Gen6 components modestly improve PCC without a decisive winner;
and the MK WAE/flow comparison is still running. Finish and audit those matched
experiments before adding complexity.
