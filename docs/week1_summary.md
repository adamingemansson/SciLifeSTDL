# Project Summary: Week 1–2 (2026-07-13 → 2026-07-20)

Concise chronological record for supervisor review. One-line rationale per decision/component. Full detail in `docs/results_log.md`, `docs/architecture_plan.md`.

**Goal:** generate spatial gene expression from H&E histology images (HEST-1k benchmark), comparing a home-grown architecture ("StormLite") against a published pretrained baseline ("STPath").

---

## Day 1 — 2026-07-13: Scaffold & model selection
- Repo scaffold, configs, literature notes — starting point.
- `BaseGenerativeModel` interface + WAE-GAN backbone — model-agnostic base so multiple generative approaches share one training/eval harness.
- Locked model roster: WAE-GAN, Flow-Matching OT (FM-OT), VQ-VAE+autoregressive — three architecturally distinct generative paradigms to compare, not just one.

## Day 2 — 2026-07-14: Data pipeline + remaining models
- HEST-1k data loader — the benchmark dataset the whole project targets.
- FM-OT model built — became the eventual flagship (continuous, ODE-based generation).
- VQ-VAE stage 1 (codebook) + stage 2 (autoregressive prior) built — third paradigm for comparison.
- Cell-type classifier — auxiliary task to sanity-check generated expression is biologically meaningful.
- FID/MMD validation metrics — standard generative-model quality metrics adapted to expression space (ST-FID, ST-MMD later).

## Day 3 — 2026-07-15: Image branch + baseline + hardening
- H&E image branch added to the models — spatial transcriptomics generation must condition on histology, not just coordinates.
- STPath integration — the published, pretrained baseline architecture the project benchmarks against.
- Multiple real-hardware bug fixes — first real-GPU runs surfaced issues invisible in unit tests.
- Multi-sample training scaffolding started — HEST-1k has many tissue samples; single-sample-only was a known limitation.

## Day 4 — 2026-07-16: Speed, robustness, StormLite born
- Caching: Gigapath image features cached (no network needed at train time), AnnData/QC cached across configs — repeated re-computation was the dominant runtime cost.
- `image_patch_size` actually controls CNN input resolution — real bug: the setting was silently ignored.
- ST-MMD metric + `--skip-training` fast re-eval mode — faster iteration on already-trained checkpoints.
- Cross-machine checkpoint bug fixed — STPath paths were baked in at save time, broke when resuming on a different machine.
- MLP and Novae gene-expression encoder options added — first alternatives to a single fixed gene encoder.
- Atomic checkpoint/cache writes — protects against corruption from concurrent DDP writers.
- TF32 matmul enabled — free Tensor Core speedup on CUDA, no accuracy cost.
- "Route B": trainable gene encoder injected as a residual into STPath's frozen pretrained transformer — tests whether STPath's weakness is its gene encoder specifically, without discarding its pretraining.
- **StormLite born**: STPath-unfrozen and STORM-lite fusion-transformer arms added for comparison — the home-grown architecture that becomes the project's flagship.
- 8-GPU parallel launch script — enables running many configs simultaneously instead of serially.
- Swin-V2-style relative position bias added to StormLite's context encoder — gives the transformer spatial awareness of context-spot geometry.
- Organ/technology conditioning + multi-sample training wired end-to-end — needed to train across HEST-1k's heterogeneous samples.
- Varied mask geometries (ellipse, irregular blob, sparse dropout) + coordinate rotation/reflection augmentation — richer, more realistic masking/augmentation than one fixed shape.

## Day 5 — 2026-07-17: Architecture deepening, decoder, results log started
- Fixed relative-position bias and Fourier features exploding on real (large) pixel-scale coordinates — worked in small synthetic tests, broke on real data scale.
- `FrameAveragingBias` added — STPath's actual, verified relative-position attention mechanism, ported for a fairer architectural comparison.
- MoME-FFN (mixture-of-modality-experts feed-forward) + per-branch LayerNorm added to StormLite — verified directly from the STORM paper; shared attention with per-modality (image/gene) expert FFNs.
- Fixed a real reporting bug: parallel-run summaries were printing the shuffle-diagnostic row instead of the real result — could have silently misled every prior comparison.
- Panel-invariant gene-expression decoder implemented — StormLite's decoder was previously fixed-panel-size; this generalizes it, needed for cross-panel/cross-technology comparisons.
- `docs/results_log.md` started — establishes a single source of truth for experiment outcomes going forward.
- Literature-grounded decoder improvement: switched gene-identity+expression combination from concatenation to **element-wise addition** — matches scGPT's (Cui et al. 2024, *Nature Methods*) real published design and roughly halves decoder memory footprint.
- `GeneAttentionDecoder` + `LLOKIStyleDecoder` implemented — two alternative decoder designs benchmarked against the panel-invariant one.
- Several real bugs fixed under load: missing gene-name injection, tech/organ vocab never populated, checkpoint cadence off by 2x for WAE-GAN, `np.savez` silently corrupting atomic writes — all found via actual overnight-scale runs, not unit tests.
- Multi-sample config + launch script added — first real multi-sample experiments queued.

## Day 6 — 2026-07-18: Mode collapse, LR/EMA fixes, first StormLite win
- Fixed StormLite "bigger" capacity config's mode collapse via gradient clipping + a buffer-save bug — bigger capacity was previously unusable, this made it trainable.
- Confirmed via day1/day2 batches: bigger StormLite capacity was still dead even after the collapse fix (later diagnosed further, see Day 9); decoder swap (add-combine) holds up as a real win.
- LR warmup + EMA (exponential moving average) weight averaging added — standard stabilizers targeting real instability seen in day2 results.
- QK-normalization, logit-normal flow-matching time sampling, and log1p-vs-raw input A/B all added — a literature-driven audit of standard stabilization tricks, tested individually.
- 16-config overnight batch launched — first large, systematic sweep combining these levers.

## Day 7 — 2026-07-19: Correcting the record, StormLite-vs-STPath, new components
- 16-config batch result: multi-sample StormLite beats STPath; bigger-capacity collapse looked fixed by QK-norm; **stacking all 3 improvements together is fragile** — motivates isolating levers one at a time going forward.
- 4-seed confirmation batch: bigger+QK-norm+warmup+EMA in isolation — separates this one lever from the untested stack.
- STPath-unfrozen (full retrain) + winning decoder + EMA, seed-averaged — closes a gap where this exact fair comparison had been planned but never actually run.
- Pivoted from shared node `st-a100` to dedicated `tkdgx1` — labmates' concurrent jobs on `st-a100` made batch timing unreliable; a dedicated 40GB-card machine removes that contention.
- Real bugs fixed on `st-a100` before the pivot: cache-directory needed write access to a read-only shared HEST-1k copy (added `hest_cache_dir` override); `n_genes` was hardcoded per-config instead of auto-derived, crashed on a shared copy with a different post-QC gene count (added `inject_single_sample_n_genes`).
- **AdaLN-residual velocity_net** added (opt-in via `velocity_net_type="adaln_residual"`) — residual blocks + adaptive layer-norm conditioning, a stronger conditioning mechanism than the plain 2-layer MLP, tested in isolation.
- Fixed a real ST-FID/ST-MMD bug: the PCA embedding's rank was capped by context-set size only, not query-set size, causing rank-deficient covariance whenever a masking draw was small — `_fid_n_components` now caps by both.
- 8-config `tkdgx1` batch run — **corrects two prior narratives**: (1) QK-norm alone was never the actual fix for bigger-capacity collapse, the LR/warmup change was; (2) STPath's real full-retrain number is much stronger than previously measured once the decoder is held constant — StormLite does not yet reliably beat STPath on a fair comparison.
- 8-config matched-seed batch (3 STPath-unfrozen, 3 StormLite-small, 2 StormLite-bigger) — firms up seed counts behind the StormLite-vs-STPath claim rather than adding new architecture.
- Matched-seed result: STPath-unfrozen leads on both mean (0.4706 vs 0.4546) and consistency (~half StormLite's seed-to-seed spread) — the earlier "tie" narrative doesn't hold at a fairer seed count.
- 8-config overnight batch: 3 more StormLite-bigger seeds (variance check), 2 more AdaLN seeds, 2 new `hidden_dim=1024` capacity-only seeds (isolates raw capacity from the AdaLN architecture question, zero new code), 1 more STPath-pretrained seed.

## Day 8 — 2026-07-20: Overnight results, gene-tokenizer architecture
- Overnight batch result: StormLite-bigger's seed variance looks like **one outlier** (seed11) against a tight 7-seed cluster at ~0.50, nearly matching STPath-pretrained's 0.5060 ceiling — reported both ways (with/without the outlier), not yet claiming the gap is closed pending root cause or more seeds.
- STPath-pretrained ceiling firms up as the tightest-spread arm of any (0.035 range across 5 seeds).
- Architecture research pass (code-reading, not more batches, per explicit instruction to "do research" instead of repeating runs): identified STPath's **per-gene tokenization** as the one significant, never-tested structural difference from StormLite's existing gene encoders — all of StormLite's prior encoders (mlp/novae/both) collapse the whole expression vector into one dense token before the transformer ever sees it.
- **`TokenizedGeneEncoder`** implemented — per-gene identity embedding added (scGPT's real combination rule) to a linear projection of each gene's own value, one token per gene, pooled via a small self-attention layer. Ports STPath's real design principle without its literal binning machinery.
- Wired in as `gene_encoder_type="tokenizer"` / `"tokenizer_novae"` (combined with Novae — an orthogonal axis, since Novae's spatial awareness comes from its own external pretrained neighbor graph, independent of per-gene identity) — opt-in, zero effect on existing configs.
- Auto-derives a 512-gene HVG-selected vocabulary (`inject_storm_lite_tokenizer_gene_names`) — mirrors the existing `GeneAttentionDecoder` pattern rather than inventing a new one.
- 10 regression tests added, full existing suite reverified passing — no regressions.
- 8-config gene-tokenizer confirmation batch designed: 2×2×2 = {tokenizer, tokenizer_novae} × {small flagship, bigger+QK-norm+warmup} × {seed10, seed11} — tests whether gene-identity tokenization and capacity are additive or redundant levers.
- Epoch counts matched at 40k for all 8 jobs (not the bigger arm's usual 80k) — every prior big-vs-small comparison in this project was confounded by training length; this isolates capacity as the only difference.
- Real bug found via the user's own smoke-test run: `gene_encoder_type="tokenizer_novae"` crashed ("requires novae_dim") because 6 separate Novae-detection checks across the codebase only recognized `("novae", "both")`, not the new combined value — Novae features were silently never computed. Fixed all 6 sites; added 2 regression tests specifically exercising the auto-detection path that the original tests had bypassed.
- **Status as of today**: fix pushed, batch ready to re-run (smoke test, then the real 8-job run); results not yet in `docs/results_log.md`.

---

## Where things stand
- **Best established numbers** (seed-averaged, same decoder): STPath-pretrained ≈ 0.506 (ceiling), STPath-unfrozen ≈ 0.471, StormLite-small ≈ 0.455, StormLite-bigger+QK-norm+warmup ≈ 0.464–0.503 (outlier-dependent, under investigation).
- **Open question**: does StormLite's bigger capacity genuinely close the gap to STPath, or is the 7/8-seed cluster near STPath-pretrained partly noise? More seeds / root-causing the outlier is the direct next step.
- **In flight**: gene-tokenizer batch (per-gene identity tokens, StormLite's one remaining untested structural gap vs. STPath) — bug just fixed, not yet run to completion.
- **Engineering discipline established along the way**: isolate one architecture lever per experiment (stacking multiple untested changes collapsed results once already); match epoch budgets in big-vs-small comparisons; report seed outliers both ways, never cherry-picked.
