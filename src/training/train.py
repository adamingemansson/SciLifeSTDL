"""
Minimal training/eval loop skeleton showing how the pieces connect:
data -> masking -> model (from registry) -> Lightning Trainer -> metrics.

MaskedContextQueryDataset draws a FRESH random masking split per training
step (a different seed per index), so an epoch sees many different
damaged/held-out regions instead of one repeated split — the real point of
masking.random_dropout_patches already being seed-parameterized. Replaces
the earlier _SingleBatchDataset placeholder now that a real pilot dataset
(HEST-1k, docs/dataset_notes.md) can be loaded.

Run with: python -m src.training.train --config configs/base_config.yaml
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

# Must be set before `import torch` / any MPS usage, not just before the
# specific op that needs it — PyTorch reads this once when the MPS
# fallback mechanism initializes, not per-op. Confirmed 2026-07-15: setting
# it later in src/models/stpath_encoder.py (imported lazily, well after
# this process had already touched MPS via the Gigapath precompute step)
# was too late and the crash still happened. STPath's own SpatialTransformer
# uses torch.linalg.eigh (fa.py create_frame), not implemented on MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from omegaconf import OmegaConf

from src.data import loaders, masking
from src.data.augmentation import augment_coords_xy
from src.data.context_features import (
    ContextOnlyFeatureProvider, ContextOnlyNovaeProvider, model_uses_novae, model_uses_scfoundation,
    novae_input_mode,
)
from src.data.niche_features import (
    compute_banksy_augmented_niche_labels, model_uses_niche_candidate, niche_input_mode,
)
from src.data.slide_context import (
    load_slide_context,
    nonoverlapping_context_patch_mask,
    visible_slide_context,
)
from src.data.mask_bank import (
    cap_context_mask, ensure_mask_bank, ensure_training_seed_bank, record_masks, split_records,
)
from src.training.validation import FixedMaskValidationCallback, predictive_samples
from src.models.registry import build_model
from src.evaluation import metrics as ev


def _primary_image_mode(cfg) -> str:
    """Return the image condition that defines the task's headline metrics."""
    evaluation = cfg.get("evaluation", {})
    modes = [str(mode) for mode in evaluation.get("image_modes", ["full"])]
    primary = str(evaluation.get("primary_image_mode", "full"))
    if primary not in modes:
        raise ValueError(
            f"evaluation.primary_image_mode={primary!r} is not present in "
            f"evaluation.image_modes={modes!r}"
        )
    return primary


def _validated_sample_groups(cfg) -> tuple[list[str], list[str], list[str]]:
    """Return disjoint train/validation/test sample IDs or fail closed.

    Different mask seeds on one slide do not create an independent test set:
    the same spot coordinates and expression targets recur across training.
    The held-out-sample contract therefore partitions whole samples before
    any model/data construction and rejects every overlap or undeclared ID.
    """
    data = cfg.get("data", {})
    configured = [str(x) for x in data.get("sample_ids", [])]
    train = [str(x) for x in data.get("train_sample_ids", configured)]
    validation = [str(x) for x in data.get("validation_sample_ids", [])]
    test = [str(x) for x in data.get("test_sample_ids", [])]

    for label, values in (
        ("data.sample_ids", configured),
        ("data.train_sample_ids", train),
        ("data.validation_sample_ids", validation),
        ("data.test_sample_ids", test),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"{label} contains duplicate sample IDs")

    groups = {"train": set(train), "validation": set(validation), "test": set(test)}
    overlaps = {
        "train/validation": groups["train"] & groups["validation"],
        "train/test": groups["train"] & groups["test"],
        "validation/test": groups["validation"] & groups["test"],
    }
    bad = {name: sorted(values) for name, values in overlaps.items() if values}
    if bad:
        raise ValueError(f"sample holdout groups overlap: {bad}")

    declared = groups["train"] | groups["validation"] | groups["test"]
    unknown = declared - set(configured)
    if configured and unknown:
        raise ValueError(f"sample split references IDs absent from data.sample_ids: {sorted(unknown)}")

    if str(data.get("holdout_unit", "")) == "sample":
        if not configured:
            raise ValueError("data.holdout_unit='sample' requires data.sample_ids")
        if not train or not validation or not test:
            raise ValueError(
                "data.holdout_unit='sample' requires non-empty train, validation, and test groups"
            )
        unused = set(configured) - declared
        if unused:
            raise ValueError(f"data.sample_ids contains unassigned samples: {sorted(unused)}")
    return train, validation, test


def _validate_task_contract(cfg) -> None:
    """Fail before loading data when a named task contradicts its masks.

    ``missing_tissue`` always means query spots inside the held-out region
    contain neither H&E nor GEX. Context spots normally retain both; explicit
    modality ablations may remove one or both context modalities while query
    coordinates remain available.
    """
    contract = str(cfg.get("data", {}).get("task_contract", "") or "")
    if not contract:
        return
    if contract != "missing_tissue":
        raise ValueError(f"unknown data.task_contract {contract!r}")

    if cfg.get("data", {}).get("sample_ids") is not None:
        _validated_sample_groups(cfg)
        if bool(cfg.get("training", {}).get("exclude_evaluation_query_spots", False)):
            raise ValueError(
                "training.exclude_evaluation_query_spots is a within-slide safeguard; "
                "sample-held-out training already excludes complete validation/test samples"
            )

    training = cfg.get("training", {})
    evaluation = cfg.get("evaluation", {})
    validation = cfg.get("validation", {})
    validation_mask_source = str(validation.get("mask_source", "evaluation"))
    if validation_mask_source not in {"evaluation", "training_seed"}:
        raise ValueError(
            "validation.mask_source must be 'evaluation' or 'training_seed'"
        )
    if validation_mask_source == "training_seed":
        if cfg.get("data", {}).get("sample_ids") is not None:
            raise ValueError(
                "validation.mask_source='training_seed' is currently restricted to "
                "single-sample overfit diagnostics"
            )
        if bool(training.get("augment_coords", False)):
            raise ValueError(
                "training-seed validation requires training.augment_coords=false"
            )
    ablation = str(cfg.get("data", {}).get("modality_ablation", "both"))
    expected = {
        # Every variant still withholds query H&E.  The names describe only
        # which modalities remain visible in the observed context.
        "both": ("target_zero", "full"),
        "gex_only": ("all_zero", "full"),
        "he_only": ("target_zero", "zero"),
        "neither": ("all_zero", "zero"),
        # Training-time modality dropout starts from the complete, valid
        # missing-tissue input and independently drops context H&E/GEX.
        "dropout": ("target_zero", "full"),
    }
    if ablation not in expected:
        raise ValueError(
            f"unknown data.modality_ablation {ablation!r}; expected one of {sorted(expected)}"
        )
    expected_image_mode, expected_gex_mode = expected[ablation]

    training_mode = str(training.get("image_mode", "full"))
    validation_mode = str(
        evaluation.get("validation_image_mode", "full")
    )
    primary_mode = _primary_image_mode(cfg)
    training_gex_mode = str(training.get("context_gex_mode", "full"))
    evaluation_gex_mode = str(evaluation.get("context_gex_mode", "full"))
    violations = []
    if training_mode != expected_image_mode:
        violations.append(f"training.image_mode={training_mode!r}")
    if validation_mode != expected_image_mode:
        violations.append(f"evaluation.validation_image_mode={validation_mode!r}")
    if primary_mode != expected_image_mode:
        violations.append(f"evaluation.primary_image_mode={primary_mode!r}")
    if training_gex_mode != expected_gex_mode:
        violations.append(f"training.context_gex_mode={training_gex_mode!r}")
    if evaluation_gex_mode != expected_gex_mode:
        violations.append(f"evaluation.context_gex_mode={evaluation_gex_mode!r}")
    gex_dropout = float(training.get("context_gex_dropout_p", 0.0))
    image_dropout = float(training.get("all_image_dropout_p", 0.0))
    if ablation == "dropout":
        if not (0.0 < gex_dropout < 1.0):
            violations.append(f"training.context_gex_dropout_p={gex_dropout!r}")
        if not (0.0 < image_dropout < 1.0):
            violations.append(f"training.all_image_dropout_p={image_dropout!r}")
    elif gex_dropout != 0.0:
        violations.append(f"training.context_gex_dropout_p={gex_dropout!r}")
    if violations:
        raise ValueError(
            "data.task_contract='missing_tissue' with "
            f"data.modality_ablation={ablation!r} requires context modalities "
            f"image_mode={expected_image_mode!r}, context_gex_mode={expected_gex_mode!r}; got "
            + ", ".join(violations)
        )


def _is_frozen_backbone_module(module: torch.nn.Module) -> bool:
    """True iff every parameter this module owns (recursively) is frozen
    (requires_grad=False) AND it owns at least one parameter at all — the
    real structural signature of this codebase's two genuine frozen-
    pretrained-backbone cases (GigapathPatchEncoder's tile_encoder,
    STPathContextEncoder's self.model when pretrained=True), which are
    deliberately reconstructed fresh from their own external pretrained
    source every time build_model() runs, rather than saved here (see
    save_trainable_state_dict's own docstring on the ~4.7GB redundancy
    this avoids).

    Requiring >=1 owned parameter is deliberate: it excludes parameter-
    free-but-buffer-only modules — RandomFourierFeatures (a FIXED RANDOM
    buffer, not reconstructible from model_cfg, not reloadable from any
    external source) and VectorQuantizer (an EMA-updated codebook buffer,
    genuinely LEARNED training state, updated by training but not via
    gradient descent so it's never in named_parameters() at all) — from
    ever being misclassified as "frozen" just because they happen to have
    zero nn.Parameter objects. Both would otherwise be silently dropped by
    a plain requires_grad-based filter — see save_trainable_state_dict's
    own docstring for the real, confirmed bug this fixes (2026-07-19,
    found via test_model_save_load.py failing: a reloaded model produced
    different output than the original because RandomFourierFeatures.B
    was never saved, so build_model() gave the reloaded model a
    DIFFERENT random coordinate-encoding basis than the one it was
    actually trained with)."""
    params = list(module.parameters(recurse=True))
    return bool(params) and all(not p.requires_grad for p in params)


def save_trainable_state_dict(model, checkpoint_dir: str, filename: str = "trainable_weights.pt") -> Path | None:
    """Save trainable parameters AND non-frozen buffers — NOT frozen
    backbones (Gigapath/STPath). Frozen backbones are identical to the
    public pretrained weights and get reloaded fresh from HuggingFace/the
    local weight file every time a model is rebuilt (see conditioning.py/
    stpath_encoder.py __init__) — saving them again here would be pure
    redundancy: a STPath-conditioned model's full state_dict() is ~4.7GB
    (1.2B frozen params) vs a few tens of MB for just the trainable ones.
    No-op (returns None) for parameter-free models like interp_baseline.

    Added 2026-07-15 after a real question: training runs weren't saving
    ANYTHING before this, meaning a completed 10000-step run's weights
    were gone the moment run_comparison.py's _free() deleted the model
    object — any later use (e.g. testing on a newly downloaded sample)
    would have required retraining from scratch.

    Real bug fixed 2026-07-19 (see _is_frozen_backbone_module's own
    docstring for the full mechanism): originally filtered purely by
    requires_grad on model.named_parameters(), which misses every BUFFER
    entirely (RandomFourierFeatures.B, VectorQuantizer's EMA codebook) —
    these are genuinely part of a trained model's state (either fixed-
    random-at-construction or EMA-updated during training) and are NOT
    reconstructible identically by simply rebuilding the architecture from
    model_cfg. Every non-frozen buffer is now saved alongside the
    trainable parameters; only buffers belonging to a genuine frozen
    pretrained backbone are still excluded (correctly redundant to save,
    since those get reloaded from their own real external source)."""
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    frozen_module_names = {name for name, m in model.named_modules() if _is_frozen_backbone_module(m)}

    def _under_frozen_module(buf_name: str) -> bool:
        parts = buf_name.split(".")
        return any(".".join(parts[:i]) in frozen_module_names for i in range(1, len(parts)))

    save_names = set(trainable_names)
    for buf_name, _ in model.named_buffers():
        if not _under_frozen_module(buf_name):
            save_names.add(buf_name)
    if not save_names:
        return None
    state = {k: v for k, v in model.state_dict().items() if k in save_names}
    path = Path(checkpoint_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    # atomic write (same reasoning/bug as _atomic_savez above) — under
    # PyTorch Lightning's default multi-GPU DDP launcher, THIS function
    # runs once per GPU subprocess (each re-executes the whole script),
    # so multiple processes call this concurrently for the same path.
    # torch.save's default format is also a zip file — concurrent writers
    # to the same path can produce the same torn-zip crash _atomic_savez
    # was added to fix. Write to a sibling temp file, then os.replace()
    # (atomic on POSIX) into the real path.
    tmp_path = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)
    return path


def save_trained_model(model, model_cfg: dict, gene_names: list, checkpoint_dir: str) -> Path | None:
    """save_trainable_state_dict's weights alone aren't enough to actually
    reuse a trained model later — reconstructing it needs (1) the exact
    architecture (model_cfg, so build_model() rebuilds something
    load_state_dict-compatible) and (2) which gene each of the n_genes
    fixed output positions corresponds to.

    (2) matters because these models (WAE-GAN/FM-OT/VQ-VAE+AR) use a
    dense fixed-width decoder — position i always means "the i-th gene in
    whatever adata.var_names order was used at construction time," with
    no notion of gene IDENTITY built into the architecture (unlike
    STPath, which tokenizes genes by a fixed vocabulary and is naturally
    panel-agnostic — see stpath_encoder.py). A DIFFERENT HEST-1k sample
    will not have an identical gene panel after its own independent QC,
    so testing a saved model against new data needs to align genes by
    NAME, not position — impossible without recording gene_names here.
    Discussed 2026-07-15 when planning cross-sample testing; this is the
    save half, load_trained_model below is the load half."""
    # Real bug fixed 2026-07-24 (config 305 stpath_pretrained_eval, an
    # entirely-frozen-weights config with zero trainable parameters by
    # design): this used to return None here whenever
    # save_trainable_state_dict found nothing to save, which ALSO skipped
    # model_cfg.json/gene_names.json below -- throwing away the only
    # record of the exact architecture/gene vocabulary needed to
    # reconstruct even a fully deterministic, weights-free model later. A
    # frozen model still needs those two files (there just won't be a
    # trainable_weights.pt alongside them).
    weights_path = save_trainable_state_dict(model, checkpoint_dir)
    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # same concurrent-DDP-writer reasoning as save_trainable_state_dict's
    # atomic torch.save above — plain-text JSON corrupts less
    # catastrophically than a torn zip (a parse error, not a segfault),
    # but it's still a real correctness bug under concurrent writers, so
    # fixed the same way.
    for name, payload in [("model_cfg.json", model_cfg), ("gene_names.json", list(gene_names))]:
        final_path = out_dir / name
        tmp_path = out_dir / f"{name}.tmp{os.getpid()}"
        with open(tmp_path, "w") as f:
            json.dump(payload, f)
        os.replace(tmp_path, final_path)
    return weights_path if weights_path is not None else out_dir / "model_cfg.json"


class PeriodicCheckpointCallback(pl.Callback):
    """Saves trainable weights + config + gene names every
    save_every_n_steps training steps, OVERWRITING the same checkpoint_dir
    each time (not versioned/accumulated — save_trained_model/
    save_trainable_state_dict already write to a fixed filename via an
    atomic os.replace, see that function's own docstring) — so a
    long-running job that gets killed, disconnected, or crashes partway
    through still leaves a recent, loadable checkpoint behind (via
    load_trained_model / run_comparison.py's --skip-training), rather
    than only ever saving once at the very end of training.

    2026-07-17, user request ("save and overwrite each 10K steps, so
    worst case I can just stop things and compare as is") — motivated by
    tonight's overnight batch including multiple long (40k-80k epoch)
    runs with no interactive supervision.

    Reuses save_trained_model exactly, not a separate/cheaper mechanism —
    deliberately NOT Lightning's own built-in ModelCheckpoint callback
    (why every trainer in this codebase already sets
    enable_checkpointing=False): that would serialize the FULL
    state_dict, including frozen backbones (Gigapath/STPath) — see
    save_trainable_state_dict's own docstring on why a STPath-conditioned
    model's full state_dict is ~4.7GB vs. a few tens of MB for just the
    trainable params. Doing that every 10k steps for an 80k-epoch run
    would be real, avoidable disk/time cost.

    Opt-in via training.checkpoint_every_n_steps in a config (unset/None
    default — every existing config's behavior is completely unchanged,
    only saving once at the end of training as before).

    Keys off batch_idx, NOT trainer.global_step — real discrepancy found
    2026-07-17 by an actual empirical check (see docs/results_log.md):
    trainer.global_step increments once per optimizer.step() call, not
    once per training batch, so it advances TWICE per batch for WAE-GAN
    specifically (its manual-optimization training_step calls .step() on
    two separate optimizers — opt_disc then opt_ae) but only once per
    batch for FM-OT/VQ-VAE+AR's single-optimizer automatic optimization.
    Using global_step would have silently checkpointed WAE-GAN twice as
    often as every other family for the same save_every_n_steps value.
    batch_idx is uniform across every family here (every trainer in this
    codebase sets max_epochs=1, with n_items=cfg.training.epochs items,
    so batch_idx directly IS the true 0-indexed count of training items
    for the whole run, matching this codebase's own "epochs means masking
    draws, not literal epochs" convention — see MaskedContextQueryDataset's
    own docstring)."""

    def __init__(self, model_cfg: dict, gene_names: list, checkpoint_dir: str,
                 save_every_n_steps: int):
        self.model_cfg = model_cfg
        self.gene_names = gene_names
        self.checkpoint_dir = checkpoint_dir
        self.save_every_n_steps = save_every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = batch_idx + 1  # batch_idx is 0-indexed; report/key off the 1-indexed item count
        if step % self.save_every_n_steps == 0:
            saved_path = save_trained_model(pl_module, self.model_cfg, self.gene_names, self.checkpoint_dir)
            if saved_path is not None:
                print(f"[PeriodicCheckpointCallback] step {step}: saved checkpoint to {saved_path.parent}")


class PeriodicPrintCallback(pl.Callback):
    """Prints trainer.callback_metrics (populated by every model's own
    self.log_dict() call regardless of logger=False — Lightning tracks
    logged metrics internally for the progress bar even when no external
    Logger is attached) via plain print() every print_every_n_steps steps.

    Real gap found 2026-07-19 while investigating a real collapsed run
    (exp_hest1k_fm_ot_stormlite_mome_both_bigger, overnight batch —
    PCC=nan, ConstantInputWarning from pearsonr: the model had collapsed
    to predicting a constant output regardless of context, see
    docs/results_log.md): every trainer in this codebase sets logger=False
    (see PeriodicCheckpointCallback's own docstring for why — avoiding
    Lightning's default ModelCheckpoint's full-state-dict cost), and
    relies on Lightning's own TQDMProgressBar for loss visibility instead
    — but tqdm auto-disables its progress bar when stdout isn't a real
    terminal, which is exactly the case for every one of this project's
    parallel launch scripts (`> logfile 2>&1`). Checked directly: the
    collapsed run's full log contained ZERO train/loss values anywhere,
    only the final eval result — making it impossible to tell WHEN during
    an 40000-step run training actually degenerated. This callback is
    independent of both the disabled Logger and the disabled progress bar
    — a plain print() always reaches a redirected log file — so future
    collapses/instabilities leave an actual loss trajectory behind.

    Opt-in via training.log_print_every_n_steps (unset/None default —
    every existing config's behavior, and its log file's size/content, is
    completely unchanged unless a config explicitly turns this on) — same
    convention as training.checkpoint_every_n_steps. Keys off batch_idx,
    not trainer.global_step (same reasoning as PeriodicCheckpointCallback's
    own docstring — uniform cadence across manual- and automatic-
    optimization model families)."""

    def __init__(self, print_every_n_steps: int):
        self.print_every_n_steps = print_every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = batch_idx + 1
        if step % self.print_every_n_steps == 0:
            metrics = {k: (round(v.item(), 6) if hasattr(v, "item") else v)
                       for k, v in trainer.callback_metrics.items()}
            print(f"[step {step}] {metrics}", flush=True)


class EMACallback(pl.Callback):
    """Exponential moving average of trainable parameters (Polyak
    averaging) — after every training step, each tracked parameter's
    shadow value is updated as shadow = decay*shadow + (1-decay)*param.
    apply_to_model() (called once, right after trainer.fit() returns —
    see main()/_main_multi_sample()/_train_model()'s own call sites)
    copies the shadow values into the live model IN PLACE, so the final
    eval/save_trained_model() call sees the EMA-smoothed weights, not
    whatever noisy state the raw optimizer trajectory happened to land on
    at the very last training step.

    2026-07-19, motivated by a real, measured problem (see
    docs/results_log.md's day2 entry): 5 identically-configured StormLite+
    decoder runs, differing only in random seed, landed anywhere from PCC
    0.366 to 0.493 — a huge spread for "the same config." EMA is the
    standard, well-established fix for exactly this kind of run-to-run
    noise (used throughout modern generative-model training: diffusion
    models, GANs, and increasingly standard transformer training) — it
    is NOT a fix for the SEPARATE "bigger StormLite mode-collapses"
    problem (see warmup_steps/_linear_warmup_lr_lambda in registry.py for
    that one); the two address different failure modes found the same day
    and are independent, composable opt-ins.

    Only tracks PARAMETERS (model.named_parameters(), same set
    save_trainable_state_dict already saves), not buffers — buffers in
    this codebase are either fixed-random-at-construction
    (RandomFourierFeatures.B — never updated by training at all, nothing
    to average) or already EMA-updated by their own internal mechanism
    (VectorQuantizer's codebook — averaging an average would just distort
    its own, already-correct EMA dynamics). Excludes frozen backbone
    parameters (requires_grad=False, e.g. Gigapath/STPath when pretrained)
    the same way save_trainable_state_dict does — nothing to average
    there either, they never change.

    Opt-in via training.ema_decay (unset/None default — every existing
    config's behavior is completely unchanged unless a config explicitly
    turns this on). decay=0.999 (a config's typical choice, not hardcoded
    here) gives an effective averaging window of roughly 1/(1-decay) =
    1000 steps — sized for this project's typical 10k-80k step runs, not
    the whole run and not just the last few steps."""

    def __init__(self, decay: float):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        with torch.no_grad():
            for name, param in pl_module.named_parameters():
                if not param.requires_grad:
                    continue
                if name not in self.shadow:
                    self.shadow[name] = param.data.clone()
                else:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_to_model(self, model) -> None:
        """Copies the EMA shadow weights into the live model's parameters,
        IN PLACE and PERMANENTLY (no raw-weights backup kept — this
        codebase's established pattern is "the trained model" means
        exactly one thing per run, not two variants to choose between at
        eval time, matching save_trained_model's own single-checkpoint
        design). Call ONCE, after trainer.fit() returns, before eval/save."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    param.data.copy_(self.shadow[name])


def load_trained_model(checkpoint_dir: str):
    """Reconstruct a model saved by save_trained_model: rebuild an
    architecturally-identical, freshly-initialized model from the saved
    model_cfg (so frozen backbones like Gigapath/STPath get re-loaded
    from their own real pretrained weights, exactly as they were during
    training — see conditioning.py/stpath_encoder.py __init__, neither of
    which was ever saved here in the first place), then load the saved
    trainable-only weights on top. strict=False since frozen/buffer
    entries in the fresh model's state_dict were never in the saved file
    by design — but every TRAINABLE parameter the fresh architecture
    expects must be present, checked explicitly rather than silently
    trusting load_state_dict's missing-keys list (which conflates
    "expected to be missing" with "genuinely lost").

    Returns (model, gene_names) — gene_names is the exact ordered list
    output position i corresponds to; callers must align any new sample's
    genes to this list BY NAME before calling model.sample().

    Real cross-machine bug found 2026-07-15 (user question: can I scp
    checkpoints from the A100 server to my Mac and just run
    --skip-training there): the caller-supplied model_cfg going INTO
    save_trained_model must be the UNRESOLVED config (OmegaConf
    to_container(..., resolve=False)), not the resolved one used to
    actually build the live model. STPath configs' stpath_gene_voc_path/
    stpath_model_weight_path are ${oc.env:...} interpolations
    (2026-07-15's earlier portability fix) — resolving them before saving
    would bake in whichever machine happened to run training (e.g. the
    A100's /nfs/scratch1/... path), making the checkpoint unusable on any
    other machine even though the actual weights transfer fine. Re-
    resolving here, at LOAD time, means the SAME checkpoint correctly
    picks up THIS machine's own STPATH_GENE_VOC_PATH/
    STPATH_MODEL_WEIGHT_PATH env vars — set them before calling this on a
    new machine, same as running a real training config there."""
    from src.models.registry import build_model
    in_dir = Path(checkpoint_dir)
    with open(in_dir / "model_cfg.json") as f:
        raw_model_cfg = json.load(f)
    model_cfg = OmegaConf.to_container(OmegaConf.create(raw_model_cfg), resolve=True)
    with open(in_dir / "gene_names.json") as f:
        gene_names = json.load(f)
    model = build_model(model_cfg)
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    weights_path = in_dir / "trainable_weights.pt"
    if weights_path.is_file():
        state = torch.load(weights_path, map_location="cpu")
        missing_trainable = trainable_names - set(state.keys())
        assert not missing_trainable, (
            f"saved weights at {in_dir} are missing trainable parameters this "
            f"model architecture expects: {missing_trainable} (model_cfg mismatch?)"
        )
        model.load_state_dict(state, strict=False)
    else:
        # save_trained_model only skips writing this file when the model
        # genuinely had zero trainable parameters/non-frozen buffers (see
        # its own 2026-07-24 docstring note) -- a freshly-built model IS
        # that checkpoint exactly, nothing to load on top. If this
        # architecture unexpectedly DOES have trainable params, that's a
        # real save/load mismatch, not something to silently paper over.
        assert not trainable_names, (
            f"{weights_path} is missing but this model architecture has "
            f"trainable parameters {trainable_names}; the checkpoint at "
            f"{in_dir} looks incomplete, not weights-free by design"
        )
    model.eval()
    return model, gene_names


def load_pretrained_weights_into(model, checkpoint_dir: str) -> dict:
    """Warm-start `model` (already built by build_model — NOT necessarily
    architecturally identical to the checkpoint's own saved config) from a
    prior run's saved trainable weights. Unlike load_trained_model (which
    reconstructs a FRESH model from the checkpoint's own config, for eval,
    and asserts every trainable param must be present), this loads INTO
    an already-constructed model, matching by (name, shape) pair —
    tolerant of architectural drift between the two configs (e.g. a
    different sample's post-QC gene panel changing a decoder's width),
    which load_trained_model's strict check would reject outright.

    Real motivation (2026-07-19, see docs/results_log.md): STPath's own
    pretrained-vs-unfrozen ablation showed pretraining alone is worth
    ~0.086 PCC, architecture held constant — StormLite never had an
    actual pretraining stage before this; it always trained directly on
    the target task. This is the mechanism for a genuine two-stage
    recipe: pretrain (e.g. multi-sample across many INT samples, longer
    schedule) -> save via save_trained_model -> finetune (build a fresh
    model for the target task/sample, call this function, THEN
    trainer.fit() as normal — see training.init_checkpoint_dir in
    main()/_main_multi_sample()/run_comparison.py's _train_model()).

    CAVEAT worth understanding, not just accepting: a (name, shape) match
    does NOT by itself guarantee SEMANTIC correctness for name-keyed
    lookup tables — e.g. decoder_type="panel_invariant"'s gene_embed
    table is indexed by decoder_gene_names' ORDER, not just its count.
    This is safe specifically because decoder_gene_names is always
    derived from a SORTED gene set (load_multi_sample's shared_genes /
    inject_decoder_gene_names' adata.var_names — both alphabetically
    sorted): identical gene SETS always produce identical embedding
    ORDER regardless of which run computed it. Mismatched-SHAPE rows are
    correctly skipped below, but same-shape rows built from a genuinely
    DIFFERENT gene set would silently load semantically wrong
    embeddings — verify gene-panel consistency between pretrain/finetune
    configs yourself before relying on this across genuinely different
    sample sets.

    Prints (and returns) exactly what loaded / was skipped and why —
    every finetune run's log shows this explicitly rather than silently
    guessing whether warm-starting actually did anything."""
    in_dir = Path(checkpoint_dir)
    state = torch.load(in_dir / "trainable_weights.pt", map_location="cpu")
    model_state = dict(model.named_parameters())
    loaded, skipped_shape, skipped_missing = [], [], []
    with torch.no_grad():
        for name, param in model_state.items():
            if name not in state:
                skipped_missing.append(name)
                continue
            src = state[name]
            if src.shape != param.shape:
                skipped_shape.append(f"{name} (checkpoint {tuple(src.shape)} vs model {tuple(param.shape)})")
                continue
            param.copy_(src)
            loaded.append(name)
    print(f"load_pretrained_weights_into({checkpoint_dir}): "
          f"loaded {len(loaded)}/{len(model_state)} trainable params, "
          f"{len(skipped_shape)} skipped (shape mismatch), "
          f"{len(skipped_missing)} skipped (not in checkpoint)")
    if skipped_shape:
        preview = skipped_shape[:5]
        print(f"  shape-mismatched (kept fresh init): {preview}{'...' if len(skipped_shape) > 5 else ''}")
    if skipped_missing:
        preview = skipped_missing[:5]
        print(f"  missing from checkpoint (kept fresh init): {preview}{'...' if len(skipped_missing) > 5 else ''}")
    return {"loaded": loaded, "skipped_shape_mismatch": skipped_shape,
            "skipped_missing_from_checkpoint": skipped_missing}


def make_context_query_split(coords3d: np.ndarray, slice_ids: np.ndarray, masking_cfg, seed: int):
    """One random context/query mask draw. Factored out of
    MaskedContextQueryDataset so anything that needs the raw boolean masks
    directly (e.g. src/evaluation/run_comparison.py, task #15, which needs
    to index the source AnnData the same way) doesn't have to duplicate
    the strategy branching logic."""
    strategy = masking_cfg.strategy
    if strategy == "hold_out_slice":
        rng = np.random.default_rng(seed)
        held_out = rng.choice(np.unique(slice_ids))
        context_mask, query_mask = masking.hold_out_slice(coords3d[:, 2], held_out, slice_ids)
    elif strategy == "random_dropout_patches":
        # 2026-07-16: masking_cfg.params can now include shape=
        # "circle"/"ellipse"/"irregular"/"mixed" (default "circle", exact
        # previous behavior) — see masking.random_dropout_patches's own
        # docstring. No new strategy name needed for this axis since it's
        # a pure param, unlike sparse_spot_dropout/mixed_dropout below
        # (genuinely different mask STRUCTURE, not just hole shape).
        context_mask, query_mask = masking.random_dropout_patches(
            coords3d[:, :2], slice_ids, seed=seed, **masking_cfg.params
        )
    elif strategy == "sparse_spot_dropout":
        context_mask, query_mask = masking.sparse_spot_dropout(
            coords3d[:, :2], slice_ids, seed=seed, **masking_cfg.params
        )
    elif strategy == "mixed_dropout":
        # combines contiguous varied-shape holes + sparse dropout in one
        # draw (2026-07-16, "better masks" follow-up) — see
        # masking.mixed_dropout's own docstring
        context_mask, query_mask = masking.mixed_dropout(
            coords3d[:, :2], slice_ids, seed=seed, **masking_cfg.params
        )
    else:
        raise ValueError(f"Unknown masking strategy {strategy}")
    return context_mask, query_mask


def _cap_context_mask(
    context_mask: np.ndarray,
    max_context_points,
    seed: int,
    *,
    coords3d: np.ndarray | None = None,
    query_mask: np.ndarray | None = None,
    selection: str = "random",
) -> np.ndarray:
    """Subsample an oversized context mask down to max_context_points
    (2026-07-20, real-hardware OOM-mitigation follow-up). Every masking
    strategy leaves ALL non-query spots in the drawn slice(s) as context,
    unbounded -- and StormLite's context encoder does full O(n^2)
    self-attention over context+query. Confirmed on tkdgx1: two jobs
    running the IDENTICAL config differed only by seed, and one drew a
    slice with a much larger context set, pushing it to ~39.5GB of 40GB
    while its twin sat at ~20GB. max_context_points=None (default)
    preserves exact prior behavior for every existing config. Uses seed+3
    (distinct from augment's seed+2 and multi-sample's seed+1) so this
    subsampling draw never correlates with those."""
    return cap_context_mask(
        context_mask,
        max_context_points,
        seed,
        coords3d=coords3d,
        query_mask=query_mask,
        selection=selection,
    )


def make_training_context_query_split(
    coords3d: np.ndarray,
    slice_ids: np.ndarray,
    masking_cfg,
    seed: int,
    excluded_training_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw one training split, optionally on evaluation-safe observations.

    With exclusions, candidates are drawn on the original coordinate system
    and rejected if their query overlaps a reserved spot. This preserves the
    original spot-spacing estimate and contiguous patch geometry; drawing on
    the punctured eligible subset would silently change both. Reserved rows
    are then removed from context before its deterministic cap is applied.
    """
    if excluded_training_mask is None:
        context_mask, query_mask = make_context_query_split(
            coords3d, slice_ids, masking_cfg, seed
        )
    else:
        excluded = np.asarray(excluded_training_mask, dtype=bool)
        if excluded.shape != (coords3d.shape[0],):
            raise ValueError(
                f"excluded_training_mask shape {excluded.shape} does not match "
                f"{coords3d.shape[0]} observations"
            )
        eligible = ~excluded
        if int(eligible.sum()) < 2:
            raise ValueError("evaluation-spot exclusion leaves fewer than two training spots")
        context_mask = query_mask = None
        candidate_seed = int(seed)
        for attempt in range(10_000):
            candidate_seed = int(seed) + attempt * 1_000_003
            candidate_context, candidate_query = make_context_query_split(
                coords3d, slice_ids, masking_cfg, candidate_seed
            )
            if np.any(candidate_query & excluded):
                continue
            candidate_context = np.asarray(candidate_context, dtype=bool) & eligible
            if candidate_context.any() and candidate_query.any():
                context_mask = candidate_context
                query_mask = np.asarray(candidate_query, dtype=bool)
                break
        if context_mask is None or query_mask is None:
            raise ValueError(
                "could not draw a training patch disjoint from immutable evaluation "
                "queries after 10000 deterministic attempts; reduce the reserved mask bank"
            )
    context_mask = _cap_context_mask(
        context_mask,
        getattr(masking_cfg, "max_context_points", None),
        candidate_seed if excluded_training_mask is not None else seed,
        coords3d=coords3d,
        query_mask=query_mask,
        selection=str(getattr(masking_cfg, "context_selection", "random")),
    )
    if not context_mask.any() or not query_mask.any():
        raise ValueError(f"training mask seed {seed} produced an empty context or query")
    return context_mask, query_mask


def _images_tensor(images: np.ndarray, mask: np.ndarray) -> torch.Tensor:
    """Slice + tensor-ify per-spot image data for one masking draw.
    Standalone (not a Dataset method) so src/evaluation/run_comparison.py's
    shared held-out eval draw can build the same tensors without going
    through MaskedContextQueryDataset.

    `images` is either raw uint8 patches [N, H, W, 3] (image_encoder_type
    "cnn" — the CNN is trainable, so caching its output would be wrong)
    or precomputed Gigapath features [N, gigapath_dim] float32
    (image_encoder_type "gigapath"/"stpath" — see
    src/models/conditioning.py precompute_gigapath_features, computed
    once by _load_images below rather than per training step). Dispatches
    on ndim so callers don't need to know which case they're in."""
    selected = images[mask]
    if selected.ndim == 4:  # raw uint8 patches [n, H, W, 3]
        return torch.tensor(selected, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    return torch.tensor(selected, dtype=torch.float32)  # already-precomputed features [n, feat_dim]


def _prepare_images_for_split(
    images: np.ndarray | None,
    context_mask: np.ndarray,
    query_mask: np.ndarray,
    mode: str = "full",
    seed: int = 0,
    query_dropout_p: float = 0.0,
    all_dropout_p: float = 0.0,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Build image tensors plus explicit availability masks for one split.

    ``mode`` defines the scientific task rather than silently assuming query
    histology is always present: ``full`` keeps all images, ``target_zero``
    removes query images, ``all_zero`` removes both context and query images,
    and ``shuffled`` permutes query images as a diagnostic. Training-time
    modality dropout is applied after the mode using deterministic RNG draws.
    Missing rows are zeroed *and* marked unavailable so encoders can substitute
    a learned missing-image token instead of treating an all-zero image as a
    real tissue patch.
    """
    if images is None:
        return None, None, None, None
    if mode not in {"full", "target_zero", "all_zero", "shuffled"}:
        raise ValueError(f"unknown image mode {mode!r}")

    context_images = _images_tensor(images, context_mask)
    query_images = _images_tensor(images, query_mask)
    context_available = torch.ones(context_images.shape[0], dtype=torch.bool)
    query_available = torch.ones(query_images.shape[0], dtype=torch.bool)
    rng = np.random.default_rng(seed + 17)

    if mode == "target_zero":
        query_available[:] = False
    elif mode == "all_zero":
        context_available[:] = False
        query_available[:] = False
    elif mode == "shuffled" and query_images.shape[0] > 1:
        perm = torch.as_tensor(rng.permutation(query_images.shape[0]), dtype=torch.long)
        query_images = query_images[perm]

    if all_dropout_p > 0 and rng.random() < all_dropout_p:
        context_available[:] = False
        query_available[:] = False
    elif query_dropout_p > 0:
        drop = torch.as_tensor(rng.random(query_images.shape[0]) < query_dropout_p)
        query_available &= ~drop

    def _zero_unavailable(x: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        if bool(available.all()):
            return x
        x = x.clone()
        x[~available] = 0
        return x

    return (
        _zero_unavailable(context_images, context_available),
        _zero_unavailable(query_images, query_available),
        context_available,
        query_available,
    )


def _build_masked_item(coords3d: np.ndarray, expr: np.ndarray, slice_ids: np.ndarray,
                        masking_cfg, images: np.ndarray | None, seed: int,
                        context_gene_features: np.ndarray | None = None,
                        context_novae_features: np.ndarray | None = None,
                        context_gene_feature_provider=None,
                        context_novae_feature_provider=None,
                        context_niche_feature_provider=None,
                        organ: str | None = None, tech: str | None = None,
                        augment: bool = False,
                        image_mode: str = "full",
                        query_image_dropout_p: float = 0.0,
                        all_image_dropout_p: float = 0.0,
                        context_gex_mode: str = "full",
                        context_gex_dropout_p: float = 0.0,
                        fixed_context_mask: np.ndarray | None = None,
                        fixed_query_mask: np.ndarray | None = None,
                        excluded_training_mask: np.ndarray | None = None,
                        slide_context: dict | None = None,
                        strict_broken_region: bool = False,
                        query_patch_size: float = 224.0) -> dict:
    """One {context, query, target_expression} training item for a SINGLE
    sample's data. Factored out of MaskedContextQueryDataset.__getitem__
    (2026-07-15) so MultiSampleMaskedContextQueryDataset below can reuse
    the exact same masking-draw logic per-sample, rather than duplicating
    it — the only new thing multi-sample training needs is WHICH sample
    to draw from each item, not a different way of drawing from one.

    context_gene_features (2026-07-15, builtin SpatialContextEncoder's
    gene_encoder_type="novae"): optional [N, novae_dim] precomputed array,
    ROW-ALIGNED with expr/coords3d/images, used INSTEAD of raw expr for
    context["expression"] only — target_expression (the query prediction
    target) always comes from the real raw expr regardless of this
    argument, since the actual task is predicting real gene expression,
    not Novae's embedding of it. None (default, "raw"/"mlp"
    gene_encoder_type) preserves the original behavior exactly.

    context_novae_features (2026-07-16, STPathContextEncoder's Route-B
    residual, new_gene_encoder_type="novae"): a SEPARATE, ADDITIVE channel
    — unlike context_gene_features above, this NEVER replaces
    context["expression"]. STPath's own gene_embed pathway always needs
    real raw expression regardless of new_gene_encoder_type, so both must
    be available simultaneously when this path is active — stashed under
    context["novae_features"], read by STPathContextEncoder.forward's own
    context_novae_features param (see registry.py's _encode_context).
    Mutually exclusive with context_gene_features in practice (a given
    config is either "builtin" with gene_encoder_type="novae", or "stpath"
    with new_gene_encoder_type="novae", never both), but nothing here
    enforces that — the caller (train.py main()/run_comparison.py
    _train_model) decides which one to populate based on which config
    option is actually set.

    context_niche_feature_provider (2026-07-23, HierarchicalGeneTransport-
    Regressor's use_niche_candidate): a context-only Callable[[mask], array]
    (see src/data/niche_features.py), invoked here with THIS item's own
    context_mask and stashed under context["niche_labels"] -- same
    context-only-recomputation discipline as context_novae_feature_provider,
    for the same reason (the niche clustering itself is a neighbor-averaging
    operation that would leak hidden query expression if ever computed on
    the full slide).

    organ/tech (2026-07-16, multi-sample training + OrganTechEmbedding
    follow-up): whole-sample metadata, not per-point — stashed identically
    into BOTH context and query dicts (unlike context_gene_features/
    context_novae_features, which are context-only since they'd leak the
    prediction target at query positions; organ/tech describes the whole
    section, so there's no such leak — see SpatialContextEncoder.forward's
    own comment on this same asymmetry). None (default) preserves the
    original behavior exactly — every existing single-sample config keeps
    working unchanged.

    augment (2026-07-16, cfg.training.augment_coords follow-up — see
    src/data/augmentation.py augment_coords_xy's own docstring for the
    real motivation): applies ONE random rotation+reflection to the
    WHOLE coords3d array before the masking split, so context and query
    share exactly the same rigid transform (preserving every pairwise
    relationship the model reasons over) and the masking split's own
    randomly-chosen hole centers/radii are drawn AFTER augmentation (still
    valid — an isometry doesn't change what a circular/elliptical/blob
    hole around some point looks like, only which absolute frame it's
    expressed in). Uses seed+2 (distinct from the seed passed to
    make_context_query_split, and distinct from the seed+1 used for
    "which sample" in MultiSampleMaskedContextQueryDataset) so the three
    random choices this pipeline can make per item never correlate through
    a shared seed. False (default) leaves coords3d byte-identical to the
    original, unaugmented behavior.

    context_gex_mode is the explicit modality intervention used by the
    missing-tissue ablations. ``zero`` removes BOTH raw context expression
    and the context-only Novae channel. Zeroing only one would leave the same
    biological signal available through the other and make an alleged
    H&E-only result invalid. ``shuffled`` applies one shared row permutation
    to both GEX-derived channels, preserving their pairing while breaking
    their association with tissue coordinates. context_gex_dropout_p drops
    the complete context-GEX modality for a training item and is independent
    of image dropout."""
    if slide_context is not None and augment:
        raise ValueError(
            "mask-specific WSI slide context cannot be combined with coordinate augmentation; "
            "the WSI tile coordinates would no longer align with the augmented spot coordinates"
        )
    if fixed_context_mask is not None or fixed_query_mask is not None:
        if fixed_context_mask is None or fixed_query_mask is None:
            raise ValueError("fixed_context_mask and fixed_query_mask must be provided together")
        if augment:
            raise ValueError("fixed mask-bank items must not use coordinate augmentation")
        if excluded_training_mask is not None:
            raise ValueError("excluded_training_mask is only valid for random training draws")
        context_mask = np.asarray(fixed_context_mask, dtype=bool)
        query_mask = np.asarray(fixed_query_mask, dtype=bool)
    else:
        if augment:
            coords3d = augment_coords_xy(coords3d, seed=seed + 2)
        context_mask, query_mask = make_training_context_query_split(
            coords3d, slice_ids, masking_cfg, seed,
            excluded_training_mask=excluded_training_mask,
        )
    if context_gene_feature_provider is not None and context_gene_features is not None:
        raise ValueError("provide either context_gene_features or context_gene_feature_provider, not both")
    if context_novae_feature_provider is not None and context_novae_features is not None:
        raise ValueError("provide either context_novae_features or context_novae_feature_provider, not both")

    if context_gene_feature_provider is not None:
        context_expr = np.asarray(context_gene_feature_provider(context_mask), dtype=np.float32)
    else:
        context_expr_source = expr if context_gene_features is None else context_gene_features
        context_expr = np.asarray(context_expr_source[context_mask], dtype=np.float32)
    context = {
        "coords": torch.tensor(coords3d[context_mask], dtype=torch.float32),
        "expression": torch.tensor(context_expr, dtype=torch.float32),
    }
    if context_novae_feature_provider is not None:
        novae_context = np.asarray(context_novae_feature_provider(context_mask), dtype=np.float32)
        context["novae_features"] = torch.tensor(novae_context, dtype=torch.float32)
    elif context_novae_features is not None:
        context["novae_features"] = torch.tensor(
            context_novae_features[context_mask], dtype=torch.float32
        )
    if context_niche_feature_provider is not None:
        niche_context = np.asarray(context_niche_feature_provider(context_mask), dtype=np.float32)
        context["niche_labels"] = torch.tensor(niche_context, dtype=torch.float32)

    context_gex_mode = str(context_gex_mode).lower()
    if context_gex_mode not in {"full", "zero", "shuffled"}:
        raise ValueError(
            "context_gex_mode must be 'full', 'zero', or 'shuffled', "
            f"got {context_gex_mode!r}"
        )
    context_gex_dropout_p = float(context_gex_dropout_p)
    if not 0.0 <= context_gex_dropout_p <= 1.0:
        raise ValueError(
            "context_gex_dropout_p must be in [0, 1], "
            f"got {context_gex_dropout_p}"
        )
    drop_context_gex = (
        context_gex_mode == "zero"
        or (
            context_gex_dropout_p > 0.0
            and np.random.default_rng(seed + 3).random() < context_gex_dropout_p
        )
    )
    # niche_labels is expression-DERIVED (BANKSY-style clustering over real
    # observed expression), so context_gex_mode's ablation/shuffle must
    # cover it too -- otherwise a "zero"/"shuffled" GEX ablation config
    # would still leak real expression-derived structure through the niche
    # candidate, defeating the point of the ablation.
    gex_keys = [key for key in ("expression", "novae_features", "niche_labels") if key in context]
    if drop_context_gex:
        for key in gex_keys:
            context[key] = torch.zeros_like(context[key])
    elif context_gex_mode == "shuffled" and context["expression"].shape[0] > 1:
        permutation = torch.as_tensor(
            np.random.default_rng(seed + 4).permutation(context["expression"].shape[0]),
            dtype=torch.long,
        )
        for key in gex_keys:
            if context[key].shape[0] != permutation.numel():
                raise ValueError(
                    f"context {key} row count does not match expression for shared GEX shuffle"
                )
            context[key] = context[key][permutation]
    query = {"coords": torch.tensor(coords3d[query_mask], dtype=torch.float32)}
    if images is not None:
        c_img, q_img, c_available, q_available = _prepare_images_for_split(
            images, context_mask, query_mask, mode=image_mode, seed=seed,
            query_dropout_p=query_image_dropout_p, all_dropout_p=all_image_dropout_p,
        )
        context["images"], query["images"] = c_img, q_img
        context["image_available"], query["image_available"] = c_available, q_available
        if strict_broken_region and image_mode == "target_zero":
            # A broken region removes pixels, not merely the query-centred
            # patches.  Exclude any observed patch whose footprint overlaps
            # a missing query footprint, otherwise boundary patches leak part
            # of the supposedly absent H&E into the model.
            safe = torch.as_tensor(
                nonoverlapping_context_patch_mask(
                    coords3d[context_mask], coords3d[query_mask], query_patch_size
                ),
                dtype=torch.bool,
            )
            context["image_available"] &= safe
            context["images"] = context["images"].clone()
            context["images"][~context["image_available"]] = 0

    slide = visible_slide_context(
        slide_context, coords3d[query_mask], image_mode, query_patch_size
    )
    context["slide_available"] = bool(slide["available"])
    if slide["available"]:
        context["slide_images"] = torch.tensor(slide["features"], dtype=torch.float32)
        context["slide_coords"] = torch.tensor(slide["coords"], dtype=torch.float32)
        context["slide_context_id"] = str(slide["context_id"])
        context["slide_n_total"] = int(slide["n_total"])
        context["slide_n_visible"] = int(slide["n_visible"])
    if organ is not None:
        context["organ"] = organ
        query["organ"] = organ
    if tech is not None:
        context["tech"] = tech
        query["tech"] = tech
    target_expression = torch.tensor(expr[query_mask], dtype=torch.float32)
    return {"context": context, "query": query, "target_expression": target_expression}


class MaskedContextQueryDataset(Dataset):
    """Each item = one fresh random context/query split over the same
    underlying AnnData. batch_size stays 1 at the DataLoader level since
    each split has a different N_context/N_query — collating variable-sized
    point clouds isn't handled here yet."""

    def __init__(self, coords3d: np.ndarray, expr: np.ndarray, slice_ids: np.ndarray,
                 masking_cfg, n_items: int, base_seed: int = 0,
                 images: np.ndarray | None = None,
                 context_gene_features: np.ndarray | None = None,
                 context_novae_features: np.ndarray | None = None,
                 context_gene_feature_provider=None,
                 context_novae_feature_provider=None,
                 context_niche_feature_provider=None,
                 organ: str | None = None, tech: str | None = None,
                 augment: bool = False,
                 image_mode: str = "full",
                 query_image_dropout_p: float = 0.0,
                 all_image_dropout_p: float = 0.0,
                 context_gex_mode: str = "full",
                 context_gex_dropout_p: float = 0.0,
                 excluded_training_mask: np.ndarray | None = None,
                 seed_schedule: list[int] | None = None,
                 slide_context: dict | None = None,
                 strict_broken_region: bool = False,
                 query_patch_size: float = 224.0):
        self.coords3d = coords3d
        self.expr = expr
        self.slice_ids = slice_ids
        self.masking_cfg = masking_cfg
        self.seed_schedule = [int(x) for x in seed_schedule] if seed_schedule is not None else None
        self.n_items = len(self.seed_schedule) if self.seed_schedule is not None else int(n_items)
        self.base_seed = int(base_seed)
        # optional per-spot image data (task #17/#18/#20), already aligned
        # to coords3d/expr's row order by the caller — either raw H&E
        # patches [N, H, W, 3] uint8 (image_encoder_type "cnn") or
        # precomputed Gigapath features [N, gigapath_dim] float32
        # ("gigapath"/"stpath" — see _load_images in this file). See
        # _images_tensor above for how the two cases are distinguished.
        self.images = images
        # optional precomputed Novae features [N, novae_dim] float32,
        # row-aligned with everything above (gene_encoder_type="novae" —
        # see get_novae_features/_build_masked_item's context_gene_features
        # docstring for why this is separate from expr rather than
        # replacing it outright).
        self.context_gene_features = context_gene_features
        # SEPARATE, additive channel for STPathContextEncoder's Route-B
        # residual (2026-07-16) — see _build_masked_item's
        # context_novae_features docstring for why this can't reuse
        # context_gene_features above.
        self.context_novae_features = context_novae_features
        self.context_gene_feature_provider = context_gene_feature_provider
        self.context_novae_feature_provider = context_novae_feature_provider
        self.context_niche_feature_provider = context_niche_feature_provider
        self.image_mode = image_mode
        self.query_image_dropout_p = float(query_image_dropout_p)
        self.all_image_dropout_p = float(all_image_dropout_p)
        self.context_gex_mode = str(context_gex_mode)
        self.context_gex_dropout_p = float(context_gex_dropout_p)
        self.slide_context = slide_context
        self.strict_broken_region = bool(strict_broken_region)
        self.query_patch_size = float(query_patch_size)
        self.excluded_training_mask = (
            None if excluded_training_mask is None
            else np.asarray(excluded_training_mask, dtype=bool).copy()
        )
        # whole-sample metadata (2026-07-16, multi-sample follow-up) — see
        # _build_masked_item's organ/tech docstring
        self.organ = organ
        self.tech = tech
        # opt-in rotation/reflection augmentation (2026-07-16) — see
        # _build_masked_item's own augment docstring / src/data/
        # augmentation.py augment_coords_xy
        self.augment = augment

    def __len__(self):
        return self.n_items

    def __getitem__(self, idx):
        seed = self.seed_schedule[idx] if self.seed_schedule is not None else self.base_seed + idx
        return _build_masked_item(
            self.coords3d, self.expr, self.slice_ids, self.masking_cfg, self.images, seed,
            context_gene_features=self.context_gene_features,
            context_novae_features=self.context_novae_features,
            context_gene_feature_provider=self.context_gene_feature_provider,
            context_novae_feature_provider=self.context_novae_feature_provider,
            context_niche_feature_provider=self.context_niche_feature_provider,
            organ=self.organ, tech=self.tech, augment=self.augment,
            image_mode=self.image_mode,
            query_image_dropout_p=self.query_image_dropout_p,
            all_image_dropout_p=self.all_image_dropout_p,
            context_gex_mode=self.context_gex_mode,
            context_gex_dropout_p=self.context_gex_dropout_p,
            excluded_training_mask=self.excluded_training_mask,
            slide_context=self.slide_context,
            strict_broken_region=self.strict_broken_region,
            query_patch_size=self.query_patch_size,
        )


class MultiSampleMaskedContextQueryDataset(Dataset):
    """Multi-sample generalization of MaskedContextQueryDataset (task
    #19 follow-up, 2026-07-15 scaffolding; wired into a real train.py
    entry point, _main_multi_sample, 2026-07-16 — see that function and
    load_multi_sample_data below. run_comparison.py's _train_model does
    NOT support this path yet — see its own docstring note).

    Each item picks ONE sample (uniformly at random, reseeded per item)
    and draws its masking split from JUST that sample via
    _build_masked_item — the same logic MaskedContextQueryDataset uses,
    not a reimplementation. Deliberately does NOT pool samples into one
    shared coordinate space: independent HEST-1k samples (different
    patients/sections) have no real spatial relationship to each other,
    so letting a k-NN/attention context encoder draw "neighbors" across
    samples would be meaningless — see load_multi_sample's docstring in
    src/data/loaders.py for the full reasoning (this mirrors that
    function's design: keep samples separate, never concatenate their
    coordinate spaces).

    samples: list of (coords3d, expr, slice_ids, images, organ, tech,
    context_gene_features, context_novae_features,
    context_gene_feature_provider, context_novae_feature_provider,
    slide_context, context_niche_feature_provider) tuples (the last four
    optional/defaulting to None -- see __getitem__'s own backwards-
    compatible tuple-length migration for 8/10/11/12-entry forms), one per
    already-loaded/QC'd/gene-aligned sample (see src/data/loaders.py
    load_multi_sample for the loading half — it returns a list of
    gene-aligned AnnData; callers derive these tuples from that list the
    same way _load_data already does for one sample, see
    load_multi_sample_data below). organ/tech (2026-07-16) are per-SAMPLE
    strings (or None), fed straight into _build_masked_item so
    OrganTechEmbedding can condition on which sample the current item's
    split was drawn from — see that function's own organ/tech docstring.

    images/context_gene_features/context_novae_features (2026-07-17, real
    gap closed — previously ALWAYS None/None/None here, meaning
    StormLite/STPath/Novae literally could not be exercised through this
    multi-sample path at all, only "builtin"/"storm_lite" with
    gene_encoder_type="raw"/"mlp". Each is now the SAME per-sample
    precomputed array _load_data/main()'s single-sample flow already
    produces (see load_multi_sample_data below for how they're computed
    per sample_ids entry) — meaning, exactly like everywhere else in this
    codebase, images is EITHER raw uint8 patches (image_encoder_type=
    "cnn") OR precomputed frozen Gigapath features
    (image_encoder_type="gigapath"/context_encoder_type in
    ("stpath", "storm_lite")), and context_gene_features/
    context_novae_features follow _build_masked_item's own
    replace-vs-additive distinction (see that function's own docstring) —
    this class doesn't re-decide any of that, just carries whichever
    arrays load_multi_sample_data already computed straight through to
    _build_masked_item, per sample, per item."""

    def __init__(self, samples: list[tuple], masking_cfg, n_items: int, base_seed: int = 0,
                 augment: bool = False, image_mode: str = "full",
                 query_image_dropout_p: float = 0.0, all_image_dropout_p: float = 0.0,
                 context_gex_mode: str = "full", context_gex_dropout_p: float = 0.0,
                 seed_schedule: list[int] | None = None,
                 strict_broken_region: bool = False,
                 query_patch_size: float = 224.0):
        assert samples, "samples must be non-empty"
        self.samples = samples
        self.masking_cfg = masking_cfg
        self.seed_schedule = [int(x) for x in seed_schedule] if seed_schedule is not None else None
        self.n_items = len(self.seed_schedule) if self.seed_schedule is not None else int(n_items)
        self.base_seed = int(base_seed)
        self.augment = augment
        self.image_mode = image_mode
        self.query_image_dropout_p = float(query_image_dropout_p)
        self.all_image_dropout_p = float(all_image_dropout_p)
        self.context_gex_mode = str(context_gex_mode)
        self.context_gex_dropout_p = float(context_gex_dropout_p)
        self.strict_broken_region = bool(strict_broken_region)
        self.query_patch_size = float(query_patch_size)

    def __len__(self):
        return self.n_items

    def __getitem__(self, idx):
        seed = self.seed_schedule[idx] if self.seed_schedule is not None else self.base_seed + idx
        # separate RNG draw for "which sample" vs. the masking split
        # itself (seed + 1, passed to _build_masked_item) so the two
        # choices aren't spuriously correlated through a shared seed
        sample_idx = int(np.random.default_rng(seed).integers(len(self.samples)))
        sample = tuple(self.samples[sample_idx])
        if len(sample) == 8:
            # Backwards-compatible tuple form used by older configs/tests:
            # providers were added later and default to absent.
            sample = (*sample, None, None)
        if len(sample) == 10:
            sample = (*sample, None)
        if len(sample) == 11:
            # niche providers were added later and default to absent.
            sample = (*sample, None)
        if len(sample) != 12:
            raise ValueError(f"multi-sample tuple must have 8, 10, 11, or 12 entries, got {len(sample)}")
        (coords3d, expr, slice_ids, images, organ, tech,
         context_gene_features, context_novae_features,
         context_gene_feature_provider, context_novae_feature_provider,
         slide_context, context_niche_feature_provider) = sample
        return _build_masked_item(
            coords3d, expr, slice_ids, self.masking_cfg, images, seed + 1,
            context_gene_features=context_gene_features,
            context_novae_features=context_novae_features,
            context_gene_feature_provider=context_gene_feature_provider,
            context_novae_feature_provider=context_novae_feature_provider,
            context_niche_feature_provider=context_niche_feature_provider,
            organ=organ, tech=tech, augment=self.augment,
            image_mode=self.image_mode,
            query_image_dropout_p=self.query_image_dropout_p,
            all_image_dropout_p=self.all_image_dropout_p,
            context_gex_mode=self.context_gex_mode,
            context_gex_dropout_p=self.context_gex_dropout_p,
            slide_context=slide_context,
            strict_broken_region=self.strict_broken_region,
            query_patch_size=self.query_patch_size,
        )


def _collate_identity(batch_list):
    return batch_list[0]


def make_dataloader(dataset, cfg) -> DataLoader:
    """DataLoader construction shared by train.py/run_comparison.py, with
    num_workers/pin_memory added 2026-07-15 after a real observation on
    an A100 server: GPU utilization sat at ~22% during training, meaning
    the GPU was idle most of the time waiting for the CPU-side
    __getitem__ work (masking draw, image tensor conversion) to finish —
    a classic CPU-bound-data-loading pattern that num_workers>0 exists to
    fix, by overlapping the NEXT item's CPU prep with the CURRENT item's
    GPU compute. This is the OPPOSITE situation from a small/slow
    accelerator (a Mac's MPS backend, where the model's own forward/
    backward pass is plausibly the bottleneck instead) — there,
    num_workers wouldn't be expected to help much and was deliberately
    not recommended (see this session's discussion). Defaults to 0
    (current/previous behavior, unaffected unless a config opts in via
    training.num_workers) since num_workers>0 has a real cost this
    project has hit before: each worker gets its OWN COPY of the
    Dataset's image array, multiplying RAM by num_workers — a serious
    concern on a RAM-constrained machine (the Mac crashes earlier this
    session), much less so on a data-center server with far more system
    RAM. pin_memory is only actually useful with a CUDA accelerator
    (speeds up host->device transfer), so it's tied to
    torch.cuda.is_available() rather than always on."""
    num_workers = int(cfg.training.get("num_workers", 0))
    has_context_provider = bool(
        getattr(dataset, "context_gene_feature_provider", None)
        or getattr(dataset, "context_novae_feature_provider", None)
        or getattr(dataset, "context_niche_feature_provider", None)
    )
    if isinstance(dataset, MultiSampleMaskedContextQueryDataset):
        has_context_provider = any(
            (len(sample) >= 10 and (sample[8] is not None or sample[9] is not None))
            or (len(sample) >= 12 and sample[11] is not None)
            for sample in dataset.samples
        )
    if has_context_provider and num_workers > 0:
        raise ValueError(
            "context-only Novae/niche providers hold AnnData/model state and must use "
            "training.num_workers=0; multiprocessing copies can corrupt caches or "
            "multiply memory unexpectedly"
        )
    return DataLoader(
        dataset, batch_size=1, collate_fn=_collate_identity,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


def load_adata(cfg):
    """QC'd AnnData for this config's data section — factored out so
    src/evaluation/run_comparison.py (task #15) can get the AnnData object
    itself (needed for cell-type clustering), not just the derived arrays
    _load_data returns."""
    if cfg.data.get("source") == "hest1k":
        adata = loaders.load_hest_sample(cfg.data.hest_data_dir, cfg.data.sample_id)
    else:
        adata = loaders.load_multi_slice(cfg.data.paths, cfg.data.z_positions)
    return loaders.basic_qc_and_normalize(
        adata, min_genes=cfg.data.min_genes, min_cells=cfg.data.min_cells,
        transform=cfg.data.get("expression_transform", "normalize_log1p"),
        target_sum=float(cfg.data.get("expression_target_sum", 1e4)),
    )


def _atomic_savez(cache_path: Path, **arrays) -> None:
    """np.savez, but crash-safe under concurrent writers.

    Real bug found 2026-07-16: running run_comparison.py with multiple
    GPUs visible and no CUDA_VISIBLE_DEVICES pin makes PyTorch Lightning
    auto-launch DDP, which works by RE-RUNNING THE ENTIRE SCRIPT once per
    GPU as a separate subprocess — so get_novae_features/
    get_gigapath_features's data-loading code (everything in _train_model
    before trainer.fit()) runs independently in every subprocess, not
    just once. When a cache file doesn't exist yet, every subprocess
    raced to np.savez() the SAME path simultaneously — np.savez writes a
    zip file, and two processes writing to the same path at once produces
    a torn/corrupt zip that a subsequent np.load() can segfault on
    reading (confirmed: a real 8-GPU run crashed with `Child process
    terminated with code -11` reading a just-written novae_cache/*.npz).
    Plain np.savez(cache_path, ...) was never safe against this — writing
    to a sibling temp file first, then os.replace() (atomic on POSIX)
    into the real path, means every concurrent writer either fully wins
    or is fully overwritten by whichever finishes last; no reader can
    ever observe a partially-written file. Doesn't eliminate the
    redundant computation across subprocesses (a separate, real but
    lower-severity waste — see get_novae_features/get_gigapath_features'
    own docstrings), only the corruption risk.

    Real bug found 2026-07-17 (smoke test): np.savez SILENTLY APPENDS
    ".npz" to any string/Path target that doesn't already end in ".npz"
    (a well-known numpy gotcha). The original tmp_path here was
    "INT2.npz.tmp<pid>" — doesn't end in ".npz", so numpy actually wrote
    "INT2.npz.tmp<pid>.npz", and the os.replace() below then raised
    FileNotFoundError looking for the path numpy never created. Fixed by
    keeping ".npz" as the tmp path's actual suffix."""
    tmp_path = cache_path.with_name(f"{cache_path.stem}.tmp{os.getpid()}.npz")
    np.savez(tmp_path, **arrays)
    os.replace(tmp_path, cache_path)  # atomic on POSIX — no reader ever sees a partial file


def _cache_root(cfg) -> Path:
    """Where cache subfolders (gigapath_cache/, novae_cache/) get created.
    Defaults to cfg.data.hest_data_dir (original, unchanged behavior) —
    but that breaks when hest_data_dir is a READ-ONLY shared dataset (a
    real setup found 2026-07-19, st-a100: a labmate's already-downloaded
    HEST-1k copy, reused via symlink specifically to avoid re-downloading
    it, but not writable by anyone else — caching next to it then fails
    with PermissionError). cfg.data.hest_cache_dir, if set, overrides this
    to any writable path instead, independent of where the read-only data
    itself lives."""
    cache_dir = cfg.data.get("hest_cache_dir") if hasattr(cfg.data, "get") else None
    return Path(cache_dir) if cache_dir else Path(cfg.data.hest_data_dir)


def _gigapath_cache_path(cfg, sample_id: str | None = None) -> Path:
    """Where precomputed Gigapath features for this sample get cached
    across runs (see _load_images) — next to the HEST-1k data itself by
    default (see _cache_root) so it's obvious it belongs to that sample,
    not somewhere in /tmp that would silently vanish between sessions.

    sample_id (2026-07-17, multi-sample image/Novae support — see
    load_multi_sample_data's own docstring for the real gap this closes):
    explicit override for per-sample cache paths when iterating over
    cfg.data.sample_ids. Defaults to cfg.data.sample_id, so every
    single-sample call site's behavior is completely unchanged."""
    sid = sample_id if sample_id is not None else cfg.data.sample_id
    return _cache_root(cfg) / "gigapath_cache" / f"{sid}.npz"


def get_gigapath_features(cfg, patches: np.ndarray, barcodes: np.ndarray,
                           sample_id: str | None = None) -> np.ndarray:
    """Load cached Gigapath features for these patches (see
    _gigapath_cache_path) if available, else compute + cache them.
    Factored out of _load_images (2026-07-15) so
    src/evaluation/run_comparison.py's shared eval-image construction can
    reuse the exact same cache, instead of only ever having whichever
    image format the FIRST config in a comparison run happened to
    produce — see run_comparison.py's _build_shared_eval for why that
    was a real bug (raw patches fed into a Gigapath/STPath model at eval
    time forced an unbatched full-ViT forward pass over the whole eval
    set at once, a real ~25GB RAM crash).

    sample_id: see _gigapath_cache_path's own docstring."""
    from src.models.conditioning import _GIGAPATH_PREPROCESS_VERSION

    cache_path = _gigapath_cache_path(cfg, sample_id=sample_id)
    # HEST Visium samples may reuse the same capture-array barcode strings.
    # Barcode equality alone therefore cannot prove that a cache belongs to
    # the selected H&E file. Hash the actual patch tensor so caches made by
    # the historical INT1/INT10 prefix-resolution bug are rejected even when
    # their barcode arrays happen to be identical. Also fold in the
    # preprocessing version (2026-07-24, real bug found: a fix to
    # _gigapath_preprocess_and_encode was silently masked because this
    # fingerprint used to depend only on the raw patch bytes, which never
    # change even when the CODE that turns them into features does) so a
    # future preprocessing change can never be silently served stale.
    patch_array = np.ascontiguousarray(patches)
    digest = hashlib.sha256()
    digest.update(str(patch_array.shape).encode("ascii"))
    digest.update(str(patch_array.dtype).encode("ascii"))
    digest.update(memoryview(patch_array).cast("B"))
    digest.update(_GIGAPATH_PREPROCESS_VERSION.encode("ascii"))
    patch_fingerprint = digest.hexdigest()
    if cache_path.exists():
        cached = np.load(cache_path)
        cached_fingerprint = (
            str(cached["patch_fingerprint"].item())
            if "patch_fingerprint" in cached.files else None
        )
        if (np.array_equal(cached["barcodes"], barcodes)
                and cached_fingerprint == patch_fingerprint):
            print(f"get_gigapath_features: loaded cached features for "
                  f"{cached['features'].shape[0]} spots from {cache_path} "
                  f"(delete this file to force a recompute).")
            return cached["features"]
        print(f"get_gigapath_features: cache at {cache_path} does not match "
              f"the current patch tensor, barcode set, or preprocessing version — recomputing.")
    from src.models.conditioning import precompute_gigapath_features, _default_device
    print(f"Precomputing Gigapath features for {patches.shape[0]} spots on "
          f"{_default_device()} (one-time cost, cached to {cache_path} "
          f"so future runs skip this step)...")
    features = precompute_gigapath_features(patches)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savez(
        cache_path,
        features=features,
        barcodes=barcodes,
        patch_fingerprint=np.asarray(patch_fingerprint),
    )
    return features


def _dinov2_cache_path(cfg, sample_id: str | None = None) -> Path:
    """Where precomputed DINOv2 features for this sample get cached across
    runs -- mirrors _gigapath_cache_path exactly, separate cache directory
    so a config using local_image_encoder_type="dinov2" never accidentally
    reads/writes the same cache file a "gigapath" config uses (different
    feature dim, would otherwise corrupt or falsely-hit the wrong cache)."""
    sid = sample_id if sample_id is not None else cfg.data.sample_id
    return _cache_root(cfg) / "dinov2_cache" / f"{sid}.npz"


def get_dinov2_features(cfg, patches: np.ndarray, barcodes: np.ndarray,
                         sample_id: str | None = None) -> np.ndarray:
    """Load cached DINOv2 features for these patches (see
    _dinov2_cache_path) if available, else compute + cache them. Mirrors
    get_gigapath_features exactly -- see that function's own docstring for
    the caching/fingerprinting reasoning, identical here."""
    from src.models.conditioning import _DINOV2_PREPROCESS_VERSION

    cache_path = _dinov2_cache_path(cfg, sample_id=sample_id)
    patch_array = np.ascontiguousarray(patches)
    digest = hashlib.sha256()
    digest.update(str(patch_array.shape).encode("ascii"))
    digest.update(str(patch_array.dtype).encode("ascii"))
    digest.update(memoryview(patch_array).cast("B"))
    digest.update(_DINOV2_PREPROCESS_VERSION.encode("ascii"))
    patch_fingerprint = digest.hexdigest()
    if cache_path.exists():
        cached = np.load(cache_path)
        cached_fingerprint = (
            str(cached["patch_fingerprint"].item())
            if "patch_fingerprint" in cached.files else None
        )
        if (np.array_equal(cached["barcodes"], barcodes)
                and cached_fingerprint == patch_fingerprint):
            print(f"get_dinov2_features: loaded cached features for "
                  f"{cached['features'].shape[0]} spots from {cache_path} "
                  f"(delete this file to force a recompute).")
            return cached["features"]
        print(f"get_dinov2_features: cache at {cache_path} does not match "
              f"the current patch tensor, barcode set, or preprocessing version — recomputing.")
    from src.models.conditioning import precompute_dinov2_features, _default_device
    print(f"Precomputing DINOv2 features for {patches.shape[0]} spots on "
          f"{_default_device()} (one-time cost, cached to {cache_path} "
          f"so future runs skip this step)...")
    features = precompute_dinov2_features(patches)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savez(
        cache_path,
        features=features,
        barcodes=barcodes,
        patch_fingerprint=np.asarray(patch_fingerprint),
    )
    return features


def _novae_cache_path(cfg, sample_id: str | None = None) -> Path:
    """Where precomputed Novae features for this sample get cached across
    runs — same reasoning as _gigapath_cache_path (Novae's forward pass
    over a whole sample is a real one-time cost worth not repeating on
    every `python -m src.training.train` invocation).

    sample_id: see _gigapath_cache_path's own docstring (same 2026-07-17
    multi-sample follow-up, same "defaults to cfg.data.sample_id" contract).
    Root directory: see _cache_root (cfg.data.hest_cache_dir override)."""
    sid = sample_id if sample_id is not None else cfg.data.sample_id
    return _cache_root(cfg) / "novae_cache" / f"{sid}.npz"


def get_novae_features(cfg, adata, sample_id: str | None = None) -> np.ndarray:
    """Load cached Novae features for this adata (see _novae_cache_path)
    if available, else compute + cache them. Cache keyed on obs_names
    (not barcodes like Gigapath's cache, since this is called with the
    already-QC'd/filtered adata directly, not a raw barcode array) so a
    cache from before a QC-threshold change is correctly invalidated
    rather than silently reused for a different spot set.

    Row order of the returned array matches adata.obs_names exactly —
    callers must not reorder/subset adata after this without recomputing.

    sample_id: see _gigapath_cache_path's own docstring."""
    cache_path = _novae_cache_path(cfg, sample_id=sample_id)
    obs_names = adata.obs_names.to_numpy()
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        if np.array_equal(cached["obs_names"], obs_names):
            print(f"get_novae_features: loaded cached features for "
                  f"{cached['features'].shape[0]} spots from {cache_path} "
                  f"(delete this file to force a recompute).")
            return cached["features"]
        print(f"get_novae_features: cache at {cache_path} covers a different "
              f"spot set than the current adata — recomputing.")
    from src.models.conditioning import precompute_novae_features
    print(f"Precomputing Novae features for {adata.n_obs} spots "
          f"(one-time cost, cached to {cache_path} so future runs skip this step)...")
    features = precompute_novae_features(adata)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savez(cache_path, features=features, obs_names=obs_names)
    return features



def prepare_novae_inputs(cfg, adata, model_params: dict, coords3d: np.ndarray,
                         slice_ids: np.ndarray, sample_id: str | None = None) -> dict:
    """Resolve Novae inputs without silently exposing hidden query expression."""
    result = {
        "mode": "disabled", "context_gene_features": None,
        "context_novae_features": None, "context_gene_feature_provider": None,
        "context_novae_feature_provider": None, "feature_dim": None,
    }
    if not model_uses_novae(dict(model_params)):
        return result
    mode = novae_input_mode(cfg, dict(model_params))
    result["mode"] = mode
    encoder = model_params.get("context_encoder_type", "builtin")
    # The hierarchical model consumes raw expression AND Novae.  Only the
    # legacy builtin encoder's explicit gene_encoder_type=novae mode replaces
    # raw expression with Novae features.
    is_builtin_replacement = (
        encoder == "builtin" and "use_novae" not in model_params
    )
    if mode == "unsafe_full_graph":
        features = get_novae_features(cfg, adata, sample_id=sample_id)
        key = "context_gene_features" if is_builtin_replacement else "context_novae_features"
        result[key] = features
        result["feature_dim"] = int(features.shape[1])
        print("WARNING: using unsafe full-graph Novae features for historical reproduction; "
              "reported metrics are contaminated by query-expression leakage.")
        return result

    sid = sample_id or str(cfg.data.get("sample_id", "sample"))
    provider = ContextOnlyNovaeProvider(
        adata,
        cache_dir=_cache_root(cfg) / "novae_context_cache",
        sample_id=sid,
    )
    # Probe one deterministic context mask before model construction so the
    # real installed Novae feature width can be injected into the architecture.
    probe_context, probe_query = make_context_query_split(
        coords3d, slice_ids, cfg.masking, seed=606_060
    )
    probe_context = _cap_context_mask(
        probe_context,
        getattr(cfg.masking, "max_context_points", None),
        606_060,
        coords3d=coords3d,
        query_mask=probe_query,
        selection=str(getattr(cfg.masking, "context_selection", "random")),
    )
    probe = provider(probe_context)
    result["feature_dim"] = int(probe.shape[1])
    key = "context_gene_feature_provider" if is_builtin_replacement else "context_novae_feature_provider"
    result[key] = provider
    return result


def prepare_scfoundation_inputs(cfg, adata, model_params: dict, sample_id: str | None = None) -> dict:
    """Resolve gene_encoder_type='scfoundation' inputs. Returns the SAME
    dict shape as prepare_novae_inputs (mode/context_gene_features/
    context_novae_features/context_gene_feature_provider/
    context_novae_feature_provider/feature_dim) so every downstream call
    site that already merges novae_inputs's keys works completely
    unchanged -- scFoundation is always the "builtin replacement" case
    (populates context_gene_features/context_gene_feature_provider,
    which _build_masked_item already substitutes for raw context["expression"]
    whenever either is non-None -- see that function's own docstring),
    never the additive context_novae_features case, since
    gene_encoder_type is a single mutually-exclusive choice in
    _build_gene_encoder (src/models/simple_fusion_encoder.py), not a
    combinable one the way STPathContextEncoder's new_gene_encoder_type
    ("both") is.

    Unlike Novae, scFoundation features depend only on a spot's own
    measured expression (no spatial-neighbor-graph propagation), so there
    is no query-expression-leakage risk and therefore no unsafe_full_graph
    historical mode to support -- always context-only, same as
    prepare_niche_inputs. Reuses ContextOnlyFeatureProvider's existing
    generic engine (mask-digest + data-signature disk caching,
    context-subgraph-only recomputation) with
    precompute_scfoundation_features bound as feature_fn, exactly the way
    prepare_niche_inputs already reuses it for the BANKSY niche
    candidate -- no new caching/plumbing invented here."""
    result = {
        "mode": "disabled", "context_gene_features": None,
        "context_novae_features": None, "context_gene_feature_provider": None,
        "context_novae_feature_provider": None, "feature_dim": None,
    }
    if not model_uses_scfoundation(dict(model_params)):
        return result
    repo_path = cfg.data.get("scfoundation_repo_path")
    model_path = cfg.data.get("scfoundation_model_path")
    if not repo_path or not model_path:
        raise ValueError(
            "This model requests gene_encoder_type='scfoundation', but "
            "data.scfoundation_repo_path and/or data.scfoundation_model_path "
            "are not set."
        )
    result["mode"] = "context_only"

    def _feature_fn(spot_adata):
        from src.models.conditioning import precompute_scfoundation_features
        expr = spot_adata.X if isinstance(spot_adata.X, np.ndarray) else spot_adata.X.toarray()
        return precompute_scfoundation_features(
            expr, list(spot_adata.var_names), str(repo_path), str(model_path),
            # This project's own loaders (src/data/loaders.py's
            # basic_qc_and_normalize) already apply the exact
            # library-size-to-1e4 + log1p transform scFoundation's own
            # real preprocessing expects (pre_normalized='F' branch of
            # its get_embedding.py) BEFORE adata ever reaches this
            # function -- applying it a second time here would silently
            # double-normalize every value. See precompute_scfoundation_
            # features's own docstring for the exact formula this skips.
            already_normalized_log1p=True,
        )

    sid = sample_id or str(cfg.data.get("sample_id", "sample"))
    provider = ContextOnlyFeatureProvider(
        adata,
        cache_dir=_cache_root(cfg) / "scfoundation_context_cache",
        sample_id=sid,
        feature_fn=_feature_fn,
    )
    # Probe one deterministic context mask before model construction so the
    # real scFoundation output width (4 * the loaded checkpoint's encoder
    # hidden dim) can be injected into the architecture -- same reasoning
    # as prepare_novae_inputs's own probe below.
    probe_context, probe_query = make_context_query_split(
        loaders.get_coords_3d(adata), adata.obs["slice_id"].to_numpy(), cfg.masking, seed=606_061,
    )
    probe_context = _cap_context_mask(
        probe_context,
        getattr(cfg.masking, "max_context_points", None),
        606_061,
        coords3d=loaders.get_coords_3d(adata),
        query_mask=probe_query,
        selection=str(getattr(cfg.masking, "context_selection", "random")),
    )
    probe = provider(probe_context)
    result["feature_dim"] = int(probe.shape[1])
    result["context_gene_feature_provider"] = provider
    return result


def prepare_niche_inputs(cfg, adata, model_params: dict, sample_id: str | None = None) -> dict:
    """Resolve use_niche_candidate inputs without silently exposing hidden
    query expression -- mirrors prepare_novae_inputs, but simpler: unlike
    Novae, the niche candidate has no learned architecture-time dimension
    to probe (context['niche_labels'] is always [Nc, 1], read directly by
    HierarchicalGeneTransportRegressor.sample()) and no historical
    unsafe_full_graph reproduction mode to support."""
    result = {"mode": "disabled", "context_niche_feature_provider": None}
    if not model_uses_niche_candidate(dict(model_params)):
        return result
    mode = niche_input_mode(cfg, dict(model_params))
    result["mode"] = mode
    sid = sample_id or str(cfg.data.get("sample_id", "sample"))
    result["context_niche_feature_provider"] = ContextOnlyFeatureProvider(
        adata,
        cache_dir=_cache_root(cfg) / "niche_context_cache",
        sample_id=sid,
        feature_fn=compute_banksy_augmented_niche_labels,
    )
    return result


def _mask_bank_for_config(cfg, adata, coords3d: np.ndarray, slice_ids: np.ndarray) -> tuple[dict, Path]:
    evaluation = cfg.get("evaluation", {})
    default_path = f"results/mask_banks/{cfg.data.get('sample_id', cfg.experiment_name)}.json"
    path = Path(evaluation.get("mask_bank_path", default_path))
    counts = {
        "validation": int(evaluation.get("n_validation_masks", 4)),
        "test": int(evaluation.get("n_test_masks", 8)),
    }
    seeds = {
        "validation": int(evaluation.get("validation_seed", 700_000)),
        "test": int(evaluation.get("test_seed", 900_000)),
    }
    bank = ensure_mask_bank(path, coords3d, slice_ids, adata.obs_names, cfg.masking, counts, seeds)
    return bank, path


def evaluation_query_exclusion_mask(bank: dict, obs_names) -> np.ndarray:
    """Union of every immutable validation/test query spot.

    A within-slide experiment is only a legitimate held-out-spot diagnostic
    when final evaluation spots never appear anywhere in training.  Returning
    one union mask lets the training dataset remove those observations before
    drawing both its context and query sets, which also prevents their GEX from
    entering context-only Novae graphs.
    """
    names = list(obs_names)
    excluded = np.zeros(len(names), dtype=bool)
    records = [
        record for split in ("validation", "test")
        for record in split_records(bank, split)
    ]
    if not records:
        raise ValueError("evaluation mask bank has no validation/test records to reserve")
    for record in records:
        _context, query = record_masks(record, names)
        excluded |= query
    if not excluded.any():
        raise ValueError("evaluation query union is empty")
    if excluded.all():
        raise ValueError("evaluation query union reserves every observation")
    return excluded


def write_training_exclusion_manifest(
    checkpoint_dir: str | Path, obs_names, excluded_mask: np.ndarray
) -> Path:
    """Persist the exact spots forbidden from within-slide training."""
    names = np.asarray([str(name) for name in obs_names])
    excluded = np.asarray(excluded_mask, dtype=bool)
    if excluded.shape != (len(names),):
        raise ValueError("training exclusion mask does not match observation names")
    reserved = names[excluded].tolist()
    payload = {
        "version": 1,
        "policy": "exclude_union_of_validation_and_test_queries_from_training_context_and_targets",
        "n_observations": int(len(names)),
        "n_excluded": int(excluded.sum()),
        "excluded_obs_names_sha256": hashlib.sha256(
            "\n".join(reserved).encode("utf-8")
        ).hexdigest(),
        "excluded_obs_names": reserved,
    }
    path = Path(checkpoint_dir) / "training_exclusion.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)
    return path


def _training_seed_bank_for_config(cfg, obs_names) -> tuple[dict, Path]:
    evaluation = cfg.get("evaluation", {})
    default_path = f"results/mask_banks/training/{cfg.experiment_name}.json"
    path = Path(evaluation.get("training_mask_bank_path", default_path))
    masking_fingerprint_cfg = cfg.masking
    if bool(cfg.training.get("exclude_evaluation_query_spots", False)):
        masking_fingerprint_cfg = {
            "masking": OmegaConf.to_container(cfg.masking, resolve=True),
            "evaluation_query_exclusion": {
                "enabled": True,
                "mask_bank_path": str(evaluation.get("mask_bank_path", "")),
                "n_validation_masks": int(evaluation.get("n_validation_masks", 4)),
                "n_test_masks": int(evaluation.get("n_test_masks", 8)),
                "validation_seed": int(evaluation.get("validation_seed", 700_000)),
                "test_seed": int(evaluation.get("test_seed", 900_000)),
            },
        }
    return ensure_training_seed_bank(
        path,
        obs_names,
        n_items=int(cfg.training.epochs),
        # Keep model initialization seeds and masking schedules independent.
        # Recovery/confirmation runs can then compare seeds on identical
        # tissue holes instead of changing both factors at once.
        base_seed=int(cfg.training.get("mask_seed", cfg.training.seed)),
        masking_cfg=masking_fingerprint_cfg,
        unique_mask_count=int(cfg.training.get("unique_mask_count", cfg.training.epochs)),
    )


def _fixed_items_from_bank(cfg, adata, coords3d, expr, slice_ids, images, bank, split,
                           novae_inputs: dict, organ=None, tech=None,
                           slide_context=None) -> list[dict]:
    evaluation = cfg.get("evaluation", {})
    image_mode = str(evaluation.get("validation_image_mode", "full"))
    context_gex_mode = str(evaluation.get("context_gex_mode", "full"))
    items = []
    for record in split_records(bank, split):
        context_mask, query_mask = record_masks(record, adata.obs_names)
        items.append(_build_masked_item(
            coords3d, expr, slice_ids, cfg.masking, images, int(record["seed"]),
            context_gene_features=novae_inputs.get("context_gene_features"),
            context_novae_features=novae_inputs.get("context_novae_features"),
            context_gene_feature_provider=novae_inputs.get("context_gene_feature_provider"),
            context_novae_feature_provider=novae_inputs.get("context_novae_feature_provider"),
            context_niche_feature_provider=novae_inputs.get("context_niche_feature_provider"),
            organ=organ, tech=tech, augment=False, image_mode=image_mode,
            context_gex_mode=context_gex_mode,
            fixed_context_mask=context_mask, fixed_query_mask=query_mask,
            slide_context=slide_context,
            strict_broken_region=bool(cfg.data.get("strict_broken_region", False)),
            query_patch_size=float(cfg.data.get("query_patch_size_fullres", 224.0)),
        ))
    return items


def _fixed_training_seed_item(
    cfg, adata, coords3d, expr, slice_ids, images, training_bank,
    novae_inputs: dict, excluded_training_mask: np.ndarray | None,
    checkpoint_dir: str | Path, organ=None, tech=None, slide_context=None,
) -> dict:
    """Build and audit the exact repeated mask used by an overfit gate.

    This item is generated through the same exclusion-aware training splitter
    and first seed consumed by ``MaskedContextQueryDataset``. It is therefore a
    real training example, never a validation/test mask disguised as one.
    """
    seeds = [int(seed) for seed in training_bank.get("seeds", [])]
    if not seeds:
        raise ValueError("training seed bank is empty")
    seed = seeds[0]
    context_mask, query_mask = make_training_context_query_split(
        coords3d, slice_ids, cfg.masking, seed,
        excluded_training_mask=excluded_training_mask,
    )
    names = np.asarray([str(name) for name in adata.obs_names])
    manifest = {
        "version": 1,
        "purpose": "training_mask_overfit_gate",
        "seed": seed,
        "n_context": int(context_mask.sum()),
        "n_query": int(query_mask.sum()),
        "context_obs_names": names[context_mask].tolist(),
        "query_obs_names": names[query_mask].tolist(),
    }
    manifest_path = Path(checkpoint_dir) / "overfit_training_mask.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_name(f"{manifest_path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp, manifest_path)
    print(
        f"overfit gate uses immutable training seed {seed}: "
        f"{int(context_mask.sum())} context / {int(query_mask.sum())} query spots; "
        f"audit: {manifest_path}"
    )
    return _build_masked_item(
        coords3d, expr, slice_ids, cfg.masking, images, seed,
        context_gene_features=novae_inputs.get("context_gene_features"),
        context_novae_features=novae_inputs.get("context_novae_features"),
        context_gene_feature_provider=novae_inputs.get("context_gene_feature_provider"),
        context_novae_feature_provider=novae_inputs.get("context_novae_feature_provider"),
        context_niche_feature_provider=novae_inputs.get("context_niche_feature_provider"),
        organ=organ, tech=tech, augment=False,
        image_mode=str(cfg.training.get("image_mode", "full")),
        context_gex_mode=str(cfg.training.get("context_gex_mode", "full")),
        fixed_context_mask=context_mask, fixed_query_mask=query_mask,
        slide_context=slide_context,
        strict_broken_region=bool(cfg.data.get("strict_broken_region", False)),
        query_patch_size=float(cfg.data.get("query_patch_size_fullres", 224.0)),
    )


def build_experiment_logger(cfg, checkpoint_dir: str):
    logging_cfg = cfg.get("logging", {})
    backend = str(logging_cfg.get("backend", "csv")).lower()
    if backend in {"none", "false", "off"}:
        return False
    if backend == "csv":
        from pytorch_lightning.loggers import CSVLogger
        return CSVLogger(save_dir=str(Path(checkpoint_dir) / "logs"), name="lightning")
    if backend == "wandb":
        try:
            from pytorch_lightning.loggers import WandbLogger
        except Exception as exc:
            raise RuntimeError("logging.backend=wandb requires the wandb package") from exc
        return WandbLogger(
            project=str(logging_cfg.get("project", "scilifestdl")),
            name=str(cfg.experiment_name),
            save_dir=str(Path(checkpoint_dir) / "logs"),
            log_model=False,
        )
    raise ValueError(f"unknown logging backend {backend!r}")


def write_run_manifest(cfg, checkpoint_dir: str, mask_bank_path: str | Path | None = None,
                       training_mask_bank_path: str | Path | None = None) -> None:
    """Persist enough state to reproduce and audit one run."""
    import datetime as _datetime
    import importlib.metadata
    import os
    import platform
    import subprocess
    import sys
    from hashlib import sha256

    out = Path(checkpoint_dir)
    out.mkdir(parents=True, exist_ok=True)
    resolved = OmegaConf.to_yaml(cfg, resolve=True)
    (out / "resolved_config.yaml").write_text(resolved)
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        git_dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip())
    except Exception:
        git_commit, git_dirty = "unavailable", None
    try:
        packages = {
            dist.metadata.get("Name", "unknown"): dist.version
            for dist in importlib.metadata.distributions()
            if dist.metadata.get("Name")
        }
    except Exception:
        packages = {}
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            cuda_devices.append({
                "index": index,
                "name": props.name,
                "total_memory_bytes": int(props.total_memory),
                "compute_capability": [int(props.major), int(props.minor)],
            })
    manifest = {
        "experiment_name": str(cfg.experiment_name),
        "created_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "resolved_config_sha256": sha256(resolved.encode()).hexdigest(),
        "git_commit": git_commit,
        "git_worktree_dirty": git_dirty,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "pytorch_lightning": pl.__version__,
        "packages": dict(sorted(packages.items(), key=lambda kv: kv[0].lower())),
        "mask_bank_path": str(mask_bank_path) if mask_bank_path else None,
        "training_mask_bank_path": str(training_mask_bank_path) if training_mask_bank_path else None,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_devices": cuda_devices,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if str(cfg.get("data", {}).get("source", "")) == "hest1k":
        sample_ids = list(cfg.data.get("sample_ids", [cfg.data.get("sample_id")]))
        sample_ids = [str(sid) for sid in sample_ids if sid is not None]
        source_files = []
        for sample_id in sample_ids:
            for kind, suffix, required_part in (
                ("expression", ".h5ad", None),
                ("patches", ".h5", "patches"),
            ):
                if kind == "patches" and not (
                    bool(cfg.data.get("use_images", False))
                    or bool(cfg.data.get("require_image_coverage", False))
                ):
                    continue
                try:
                    source = loaders._resolve_hest_sample_file(
                        cfg.data.hest_data_dir,
                        sample_id,
                        suffix,
                        required_path_part=required_part,
                    ).resolve()
                    stat = source.stat()
                    source_files.append({
                        "sample_id": sample_id,
                        "kind": kind,
                        "path": str(source),
                        "size_bytes": int(stat.st_size),
                        "mtime_ns": int(stat.st_mtime_ns),
                    })
                except Exception as exc:
                    source_files.append({
                        "sample_id": sample_id,
                        "kind": kind,
                        "resolution_error": f"{type(exc).__name__}: {exc}",
                    })
        manifest["hest_source_files"] = source_files
    tmp = out / "run_manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp.replace(out / "run_manifest.json")


def inject_novae_dim(model_cfg: dict, novae_dim: int) -> None:
    """If a config sets gene_encoder_type: "novae" or "both" (2026-07-16;
    "both" added for StormLiteContextEncoder's CombinedGeneEncoder mode —
    see storm_lite_encoder.py), auto-derive novae_dim from the real
    precomputed features' shape rather than requiring it hardcoded into a
    YAML file (same reasoning as inject_stpath_gene_names — the real
    value is only known once Novae has actually been run, and hardcoding
    a guessed number risks silently drifting from whatever the installed
    novae package version actually outputs). Shared by "builtin"
    (SpatialContextEncoder) and "storm_lite" (StormLiteContextEncoder) —
    both use the same gene_encoder_type/novae_dim param names, mutually
    exclusive per model so there's no risk of conflating them. Mutates
    model_cfg["params"] in place; no-op for every other config."""
    params = model_cfg.get("params", {})
    if (
        params.get("gene_encoder_type") in ("novae", "both", "tokenizer_novae")
        or bool(params.get("use_novae", False))
    ) and "novae_dim" not in params:
        params["novae_dim"] = novae_dim


def inject_scfoundation_dim(model_cfg: dict, scfoundation_dim: int) -> None:
    """Same reasoning as inject_novae_dim above, for _build_gene_encoder's
    (src/models/simple_fusion_encoder.py) gene_encoder_type='scfoundation'
    option: the real output width (4 * the loaded checkpoint's encoder
    hidden dim) is only known once scFoundation has actually run, so it's
    probed (prepare_scfoundation_inputs) and injected here rather than
    hardcoded into a YAML file. Mutates model_cfg["params"] in place;
    no-op for every other config."""
    params = model_cfg.get("params", {})
    if params.get("gene_encoder_type") == "scfoundation" and "scfoundation_dim" not in params:
        params["scfoundation_dim"] = scfoundation_dim


def inject_expression_preprocessing(model_cfg: dict, adata) -> None:
    """Tell encoders whether the loader already applied the single log1p.

    The loader records its contract in ``adata.uns``. Encoder flags are
    injected only when a config did not explicitly override them, preserving
    the ability to reproduce a historical double-log run while making clean
    configs correct by default.
    """
    params = model_cfg.get("params", {})
    already_log1p = loaders.expression_is_log1p(adata)
    context_encoder_type = params.get("context_encoder_type")
    if context_encoder_type == "storm_lite":
        params.setdefault("storm_lite_input_already_log1p", already_log1p)
    elif context_encoder_type == "stpath":
        params.setdefault("stpath_input_already_log1p", already_log1p)


def _pooled_gene_std(expressions: list[np.ndarray]) -> np.ndarray:
    """Per-gene standard deviation using training samples only."""
    if not expressions:
        raise ValueError("at least one training expression matrix is required")
    total_n = 0
    total_sum = None
    total_sumsq = None
    for expression in expressions:
        values = np.asarray(expression, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f"expression matrix must be 2-D, got {values.shape}")
        if total_sum is not None and values.shape[1] != total_sum.shape[0]:
            raise ValueError("training expression matrices do not share one gene panel")
        sample_sum = values.sum(axis=0)
        sample_sumsq = np.square(values).sum(axis=0)
        total_sum = sample_sum if total_sum is None else total_sum + sample_sum
        total_sumsq = sample_sumsq if total_sumsq is None else total_sumsq + sample_sumsq
        total_n += values.shape[0]
    if total_n < 2:
        raise ValueError("at least two training observations are required for gene scaling")
    variance = np.maximum(total_sumsq / total_n - np.square(total_sum / total_n), 0.0)
    return np.sqrt(variance).astype(np.float32)


def _pooled_gene_mean_std(expressions: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Training-only pooled mean and standard deviation for direct regression."""
    if not expressions:
        raise ValueError("at least one training expression matrix is required")
    total_n = 0
    total_sum = None
    total_sumsq = None
    for expression in expressions:
        values = np.asarray(expression, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f"expression matrix must be 2-D, got {values.shape}")
        if total_sum is not None and values.shape[1] != total_sum.shape[0]:
            raise ValueError("training expression matrices do not share one gene panel")
        sample_sum = values.sum(axis=0)
        sample_sumsq = np.square(values).sum(axis=0)
        total_sum = sample_sum if total_sum is None else total_sum + sample_sum
        total_sumsq = sample_sumsq if total_sumsq is None else total_sumsq + sample_sumsq
        total_n += values.shape[0]
    if total_n < 2:
        raise ValueError("at least two training observations are required for gene statistics")
    mean = total_sum / total_n
    variance = np.maximum(total_sumsq / total_n - np.square(mean), 0.0)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def inject_residual_gene_scale(model_cfg: dict, expressions: list[np.ndarray]) -> None:
    """Inject training-only gene scales for ``harmonic_residual`` models."""
    if model_cfg.get("name") != "harmonic_residual":
        return
    params = model_cfg.get("params", {})
    if "residual_gene_scale" not in params:
        params["residual_gene_scale"] = _pooled_gene_std(expressions).tolist()


def inject_direct_regression_stats(model_cfg: dict, expressions: list[np.ndarray]) -> None:
    """Inject training-only target normalization for direct context regression."""
    if model_cfg.get("name") not in {
        "direct_context_regressor", "hierarchical_missing_tissue_regressor",
    }:
        return
    params = model_cfg.get("params", {})
    if "target_gene_mean" not in params or "target_gene_scale" not in params:
        mean, scale = _pooled_gene_mean_std(expressions)
        params.setdefault("target_gene_mean", mean.tolist())
        params.setdefault("target_gene_scale", scale.tolist())


def inject_transport_gene_scale(model_cfg: dict, expressions: list[np.ndarray]) -> None:
    """Inject train-only scaling for gene-preserving transport loss."""
    if model_cfg.get("name") not in {
        "context_transport_regressor", "gene_aware_transport_regressor",
        "hierarchical_gene_transport_regressor",
    }:
        return
    params = model_cfg.get("params", {})
    if "target_gene_scale" not in params:
        params["target_gene_scale"] = _pooled_gene_std(expressions).tolist()


def inject_coord_scale(model_cfg: dict, coord_scale: float) -> None:
    """RandomFourierFeatures real-scale bug fix (2026-07-17 — see that
    class's own docstring in conditioning.py): auto-derive coord_scale
    from this sample's real coordinate spread rather than hardcoding it,
    since different HEST-1k samples can have different physical pixel
    resolutions — same "must be derived from real data, not guessed"
    reasoning as inject_novae_dim/inject_stpath_gene_names. Applies to
    context_encoder_type in ("builtin", "storm_lite") only — both
    construct a RandomFourierFeatures coord_encoder (see registry.py's
    _build_context_encoder); "stpath" has its own real, verified
    geometry-aware attention bias, unaffected by this mechanism. Mutates
    model_cfg["params"] in place; no-op for every other config."""
    params = model_cfg.get("params", {})
    context_model_names = {
        "wae_gan", "fm_ot", "vqvae_ar", "harmonic_residual", "residual_fm_ot",
        "direct_context_regressor", "context_transport_regressor",
        "gene_aware_transport_regressor",
    }
    if (model_cfg.get("name") in context_model_names
            and params.get("context_encoder_type", "builtin") in ("builtin", "storm_lite")
            and "coord_scale" not in params):
        params["coord_scale"] = coord_scale


def inject_stpath_novae_dim(model_cfg: dict, novae_dim: int) -> None:
    """Same reasoning as inject_novae_dim above, for STPathContextEncoder's
    Route-B residual (context_encoder_type: "stpath" +
    stpath_new_gene_encoder_type: "novae", 2026-07-16) — separate function/
    param name (stpath_novae_dim, not novae_dim) since the two mechanisms
    are genuinely different (see registry.py's _build_context_encoder
    docstring)."""
    params = model_cfg.get("params", {})
    if (params.get("context_encoder_type") == "stpath"
            and params.get("stpath_new_gene_encoder_type") in ("novae", "both")
            and "stpath_novae_dim" not in params):
        params["stpath_novae_dim"] = novae_dim


def _load_images(cfg, adata, sample_id: str | None = None):
    """Optional H&E patches (task #17) — only loaded when
    cfg.data.use_images is set, since every existing pilot config stays
    expression-only by default. Returns (adata, images): adata may come
    back as a SUBSET of the input — align_patches_to_adata() drops spots
    with no matching patch (a normal partial gap in HEST-1k's own patch
    extraction, not an error) — so callers must use the returned adata,
    not their original one, for everything downstream.

    sample_id (2026-07-17, multi-sample image support — see
    load_multi_sample_data's own docstring): explicit override so
    load_multi_sample_data can call this once per sample_ids entry.
    Defaults to cfg.data.sample_id, so every single-sample call site's
    behavior (main()'s own _load_data) is completely unchanged.

    For image_encoder_type "gigapath" or context_encoder_type "stpath",
    images comes back as PRECOMPUTED Gigapath features [N, gigapath_dim]
    float32, not raw patches — computed once via precompute_gigapath_features()
    rather than recomputed from raw pixels on every training step (task
    #18/#20's real bug, 2026-07-15: the naive per-step version made a real
    STPath training run essentially hang).

    That in-memory cache only helped WITHIN one run though — every fresh
    `python -m src.training.train` invocation still re-ran the ~1.1B-param
    frozen ViT over every patch from scratch (2026-07-15, user question:
    "why is gigapath recomputing every run? cant it be saved?"). Since
    Gigapath is frozen, its output for a given raw patch is fixed forever,
    so it's cached to disk at _gigapath_cache_path(cfg) (features computed
    for the FULL raw barcode set from the .h5 file, before any adata
    filtering — align_patches_to_adata reorders/subsets an array purely by
    barcode lookup, so it works identically whether that array is raw
    patches or a cached feature matrix, meaning this cache stays valid
    even if QC settings change which spots survive downstream). Delete the
    cache file to force a recompute (e.g. after changing which patches
    file is on disk).

    For "cnn" (or no image use), images stays raw uint8 patches
    [N, patch_size, patch_size, 3] — the CNN is trainable so its output
    can't be cached, but the array IS downsampled once here (see
    _downsample_patches) rather than kept at the native 224x224, since
    that resolution reduction is what actually made image_patch_size do
    something (see this function's real fix below).

    local_image_encoder_type="dinov2" (2026-07-23, only meaningful on
    hierarchical_missing_tissue_regressor/hierarchical_gene_transport_
    regressor -- the only model families with this param) routes to
    get_dinov2_features instead of get_gigapath_features -- same
    precompute-once-cache-to-disk reasoning, separate cache directory
    (_dinov2_cache_path) so it can never collide with a "gigapath" config's
    cache."""
    sid = sample_id if sample_id is not None else cfg.data.sample_id
    if not cfg.data.get("use_images", False):
        if cfg.data.get("require_image_coverage", False):
            barcodes = loaders.load_hest_patch_barcodes(cfg.data.hest_data_dir, sid)
            adata = loaders.align_adata_to_patch_barcodes(adata, barcodes)
        return adata, None
    patches, barcodes = loaders.load_hest_patches(cfg.data.hest_data_dir, sid)

    model_params = cfg.model.get("params", {})
    is_hierarchical_transport_family = cfg.model.get("name") in (
        "hierarchical_missing_tissue_regressor", "hierarchical_gene_transport_regressor",
    )
    uses_frozen_dinov2 = (
        is_hierarchical_transport_family
        and model_params.get("local_image_encoder_type") == "dinov2"
    )
    uses_frozen_gigapath = (
        not uses_frozen_dinov2
        and (
            model_params.get("image_encoder_type") == "gigapath"
            or model_params.get("context_encoder_type") in ("stpath", "storm_lite")
            or is_hierarchical_transport_family
            # stpath_backbone_simple_gene / simple_cross_attn_dense_decoder
            # (2026-07-24): both reuse GigapathPatchEncoder internally exactly
            # like simple_fusion/simple_cross_attn/simple_stpath_transformer do on
            # context_transport_regressor -- same real bug those hit (recomputing
            # GigaPath from raw patches every step) if this isn't flagged here too.
            or cfg.model.get("name") in ("stpath_backbone_simple_gene", "simple_cross_attn_dense_decoder")
        )
    )
    if uses_frozen_dinov2:
        features = get_dinov2_features(cfg, patches, barcodes, sample_id=sid)
        adata, images = loaders.align_patches_to_adata(adata, features, barcodes)
    elif uses_frozen_gigapath:
        features = get_gigapath_features(cfg, patches, barcodes, sample_id=sid)
        adata, images = loaders.align_patches_to_adata(adata, features, barcodes)
    else:
        adata, images = loaders.align_patches_to_adata(adata, patches, barcodes)
        if model_params.get("image_encoder_type") == "cnn":
            target_size = model_params.get("image_patch_size", 224)
            if target_size != images.shape[1]:
                print(f"_load_images: downsampling CNN input patches from "
                      f"{images.shape[1]}x{images.shape[1]} to {target_size}x{target_size} "
                      f"(image_patch_size) — real speed fix, 2026-07-15, see "
                      f"_downsample_patches docstring.")
                images = _downsample_patches(images, target_size)
    return adata, images


def _downsample_patches(patches: np.ndarray, target_size: int) -> np.ndarray:
    """Cheap nearest-neighbor downsample of raw uint8 H&E patches
    [N, H, W, 3] -> [N, target_size, target_size, 3], via numpy index
    striding (no new dependency, no torch/float conversion needed) — run
    ONCE at data-loading time, on the raw uint8 array, before any per-step
    cost, since every later masking draw just slices whatever array is
    stored here.

    Real fix (2026-07-15, user question: "why are CNN configs so slow"):
    image_patch_size was already threaded through 4 layers of config/model
    code (registry.py -> SpatialContextEncoder -> ImagePatchEncoder) but
    ImagePatchEncoder never actually used it — accepted, then silently
    ignored, since AdaptiveAvgPool2d(1) makes its conv stack agnostic to
    input spatial size. Every CNN-branch training step was therefore
    converting (uint8->float32) + transferring (host->device) +
    convolving the FULL native 224x224 HEST-1k patches (up to ~700-900
    per masking draw) regardless of this config value — real, substantial
    cost that scales with pixel count, and the actual bottleneck (not
    conv FLOPs alone, which are comparatively small for this tiny
    network — the CPU-side conversion and the ~500MB+ per-step host-to-
    device transfer at full resolution dominate). Downsampling ONCE here,
    at load time, cuts all three simultaneously.

    Nearest-neighbor (via np.linspace index selection, not e.g. bilinear/
    area averaging) is a deliberate choice: ImagePatchEncoder was always
    meant as a cheap "does ANY image signal help at all" ablation baseline
    (task #17), not competing with Gigapath/STPath on image fidelity —
    plain index-based downsampling is dependency-free and fast; a
    higher-quality resize isn't worth the extra complexity for this arm."""
    n, h, w, c = patches.shape
    if h == target_size and w == target_size:
        return patches
    row_idx = np.linspace(0, h - 1, target_size).astype(np.int64)
    col_idx = np.linspace(0, w - 1, target_size).astype(np.int64)
    return patches[:, row_idx][:, :, col_idx]


def _load_data(cfg) -> tuple:
    adata = load_adata(cfg)
    adata, images = _load_images(cfg, adata)
    coords3d = loaders.get_coords_3d(adata)
    expr = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    slice_ids = adata.obs["slice_id"].to_numpy()
    return adata, coords3d, expr, slice_ids, images


def load_multi_sample_data(
    cfg, sample_ids: list[str] | None = None,
    reference_gene_names: list[str] | None = None,
) -> tuple[list[tuple], list]:
    """Multi-sample counterpart to _load_data. Wired into main() below
    (2026-07-16) via cfg.data.sample_ids — a list, in place of the
    single-sample configs' cfg.data.sample_id. Reads optional
    cfg.data.organs/cfg.data.techs (parallel lists, same meaning as
    loaders.load_multi_sample's organs/techs param — see that function's
    docstring for why these are caller-supplied, not auto-parsed).

    UPDATED 2026-07-17 (real gap closed — this function used to hardcode
    images=None and never compute Novae features, meaning StormLite/
    STPath/Novae literally could not be exercised through the multi-sample
    path at all, only "builtin"/"storm_lite" with gene_encoder_type=
    "raw"/"mlp" worked): now loads images (_load_images, per sample_id —
    same real Gigapath/CNN-patch handling as the single-sample path, just
    called once per sample with its own cache path) and precomputes Novae
    features per sample (get_novae_features, same per-sample cache path)
    whenever the model config actually needs them — mirrors main()'s own
    single-sample dispatch logic EXACTLY (see main()'s
    context_gene_features/context_novae_features block): "builtin" +
    gene_encoder_type="novae" REPLACES context["expression"]
    (context_gene_features); "storm_lite" with gene_encoder_type in
    ("novae","both") and "stpath" with stpath_new_gene_encoder_type in
    ("novae","both") both use the ADDITIVE channel (context_novae_features)
    instead — see _build_masked_item's own docstring for why these are
    genuinely different mechanisms, not interchangeable.

    STPath itself is NOT fully multi-sample-aware even with this fix:
    STPathContextEncoder still uses ONE FIXED stpath_organ_type/
    stpath_tech_type string across every sample (its real IDTokenizer
    vocabulary, loaded once at construction — not a per-sample-dynamic
    mechanism the way OrganTechEmbedding is). Fine for single-organ/
    single-platform data like the currently-available INT1-INT24 (all
    ccRCC, all Visium, per docs/dataset_notes.md) — a real limitation
    only if genuinely mixed-organ/platform samples are used with STPath
    specifically; StormLite has no such limitation (OrganTechEmbedding is
    real per-sample conditioning throughout).

    Per-sample image loading can drop spots with no matching H&E patch
    (a normal partial gap, not an error — see _load_images's own
    docstring) — the returned adata for that sample reflects that
    subset, and every array derived below (coords3d/expr/slice_ids/
    organ/tech/novae features) is derived from that SAME
    possibly-subsetted adata, never the pre-image-loading one, so nothing
    can end up row-misaligned.

    Uses loaders.load_multi_sample for the actual loading/QC/shared-gene-
    panel alignment (not reimplemented here) — this function's job is
    converting that list of AnnData into the (coords3d, expr, slice_ids,
    images, organ, tech, context_gene_features, context_novae_features)
    tuples MultiSampleMaskedContextQueryDataset expects.

    Returns (samples, adatas) — the (possibly per-sample image-QC-
    adjusted) adatas are also returned since inject_organ_tech_vocab/
    inject_decoder_gene_names/inject_novae_dim (below) need each sample's
    real organ/tech value and gene panel, and re-deriving it from the
    tuples would just mean unpacking the same thing twice. Image QC only
    ever drops ROWS (spots), never gene columns, so the shared gene panel
    load_multi_sample already aligned stays valid regardless."""
    sample_ids = list(sample_ids if sample_ids is not None else cfg.data.sample_ids)
    configured_ids = list(cfg.data.get("sample_ids", sample_ids))

    def _metadata_for(ids, list_key, mapping_key):
        mapping = cfg.data.get(mapping_key)
        if mapping is not None:
            return [mapping.get(str(sid)) for sid in ids]
        values = cfg.data.get(list_key)
        if values is None:
            return None
        by_id = {str(sid): value for sid, value in zip(configured_ids, list(values))}
        return [by_id.get(str(sid)) for sid in ids]

    adatas = loaders.load_multi_sample(
        cfg.data.hest_data_dir, sample_ids,
        min_genes=cfg.data.min_genes, min_cells=cfg.data.min_cells,
        organs=_metadata_for(sample_ids, "organs", "organ_by_sample"),
        techs=_metadata_for(sample_ids, "techs", "tech_by_sample"),
        expression_transform=cfg.data.get("expression_transform", "normalize_log1p"),
        expression_target_sum=float(cfg.data.get("expression_target_sum", 1e4)),
        reference_genes=reference_gene_names,
    )
    model_params = cfg.model.get("params", {})
    context_encoder_type = model_params.get("context_encoder_type", "builtin")
    use_images = cfg.data.get("use_images", False)

    samples = []
    updated_adatas = []
    for sample_id, adata in zip(sample_ids, adatas):
        images = None
        if use_images or cfg.data.get("require_image_coverage", False):
            adata, images = _load_images(cfg, adata, sample_id=sample_id)

        coords3d = loaders.get_coords_3d(adata)
        expr = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
        slice_ids = adata.obs["slice_id"].to_numpy()
        # organ/tech are constant across a whole sample's obs (see
        # loaders.load_hest_sample) — any row's value is the sample's value
        organ = str(adata.obs["organ"].iloc[0]) if "organ" in adata.obs else None
        tech = str(adata.obs["tech"].iloc[0]) if "tech" in adata.obs else None

        novae_inputs = prepare_novae_inputs(
            cfg, adata, model_params, coords3d, slice_ids, sample_id=str(sample_id)
        )
        scfoundation_inputs = prepare_scfoundation_inputs(cfg, adata, model_params, sample_id=str(sample_id))
        if scfoundation_inputs["mode"] != "disabled":
            # Mutually exclusive with Novae by construction (gene_encoder_type
            # is a single value) -- overlay scFoundation's own
            # context_gene_features/context_gene_feature_provider onto the
            # SAME slots novae_inputs would otherwise populate, so every
            # downstream reader of novae_inputs (this function's own tuple
            # below, _inject's novae_dim probe) works unchanged regardless
            # of which one actually ran.
            novae_inputs["context_gene_features"] = scfoundation_inputs["context_gene_features"]
            novae_inputs["context_gene_feature_provider"] = scfoundation_inputs["context_gene_feature_provider"]
        niche_inputs = prepare_niche_inputs(cfg, adata, model_params, sample_id=str(sample_id))
        slide_context = load_slide_context(
            cfg, str(sample_id), images, coords3d
        )
        samples.append((
            coords3d, expr, slice_ids, images, organ, tech,
            novae_inputs["context_gene_features"], novae_inputs["context_novae_features"],
            novae_inputs["context_gene_feature_provider"], novae_inputs["context_novae_feature_provider"],
            slide_context, niche_inputs["context_niche_feature_provider"],
        ))
        updated_adatas.append(adata)
    return samples, updated_adatas


def inject_organ_tech_vocab(model_cfg: dict, adatas: list) -> None:
    """If a config sets use_organ_tech_conditioning: true (multi-sample
    training only — see load_multi_sample_data/MultiSampleMaskedContext-
    QueryDataset above), auto-derive organ_vocab/tech_vocab from the real
    per-sample organ/tech values rather than requiring them hardcoded into
    a YAML file — same "vocabulary size must be fixed at construction
    time, and must exactly match what training data will supply" reasoning
    as inject_stpath_gene_names/inject_novae_dim. use_organ_tech_conditioning
    is consumed HERE ONLY (not a real param on any context encoder) — it
    exists purely as an opt-in switch, since organ_vocab/tech_vocab=None
    (the default) already means "no conditioning" for both SpatialContext-
    Encoder and StormLiteContextEncoder (see OrganTechEmbedding's own
    docstring on why this is a no-op on single-organ/single-platform data
    like the currently-available INT1-INT24 samples). Mutates
    model_cfg["params"] in place; no-op unless the switch is set."""
    params = model_cfg.get("params", {})
    if not params.pop("use_organ_tech_conditioning", False):
        return
    organs = [str(a.obs["organ"].iloc[0]) for a in adatas if "organ" in a.obs]
    techs = [str(a.obs["tech"].iloc[0]) for a in adatas if "tech" in a.obs]
    from src.models.conditioning import build_organ_tech_vocab
    organ_vocab, tech_vocab = build_organ_tech_vocab(organs, techs)
    params.setdefault("organ_vocab", organ_vocab)
    params.setdefault("tech_vocab", tech_vocab)


def inject_pretrained_autoencoder_gene_names(model_cfg: dict, adata) -> None:
    """Inject the exact fit-derived gene order expected by residual AE checkpoints."""
    params = model_cfg.get("params", {})
    if model_cfg.get("name") == "residual_fm_ot" and params.get("pretrained_autoencoder_path"):
        params.setdefault("pretrained_autoencoder_gene_names", adata.var_names.tolist())


def inject_stpath_gene_names(model_cfg: dict, adata) -> None:
    """If a config sets context_encoder_type: "stpath" (task #18),
    auto-derive stpath_gene_names from the loaded AnnData's var_names
    rather than requiring ~16570 gene symbols hardcoded into a YAML file.
    Mutates model_cfg["params"] in place; no-op for every other config."""
    params = model_cfg.get("params", {})
    if params.get("context_encoder_type") == "stpath" and "stpath_gene_names" not in params:
        params["stpath_gene_names"] = adata.var_names.tolist()


def inject_universal_gene_names(model_cfg: dict, adata) -> None:
    """If a config sets gene_encoder_type: "universal_mlp" or
    "universal_linear" (UniversalMLPGeneEncoder / UniversalLinearGeneEncoder,
    2026-07-24 -- scatter this dataset's local gene panel into STPath's
    real fixed gene-ID vocabulary before encoding it), auto-derive
    gene_names from the loaded AnnData's var_names, same reasoning as
    inject_stpath_gene_names above (a vocabulary/mapping fixed at
    construction time from real data, not hardcoded into a YAML file)."""
    params = model_cfg.get("params", {})
    if params.get("gene_encoder_type") in ("universal_mlp", "universal_linear") and "gene_names" not in params:
        params["gene_names"] = adata.var_names.tolist()


def inject_decoder_gene_names(model_cfg: dict, adata) -> None:
    """If a config sets decoder_type: "panel_invariant" (2026-07-17,
    diagram-5 gap analysis follow-up — see PanelInvariantGeneDecoder's own
    docstring in conditioning.py), auto-derive decoder_gene_names from the
    loaded AnnData's var_names, same "vocabulary fixed at construction
    time, derived from real data rather than hardcoded into a YAML file"
    reasoning as inject_stpath_gene_names above.

    decoder_type: "gene_attention" (2026-07-17, GeneAttentionDecoder —
    Geneformer-inspired, see its own docstring) gets a DIFFERENT
    treatment: it CANNOT safely use the full ~16570-gene training
    vocabulary (its self-attention is O(n_panel^2) per query location,
    guarded by MAX_SAFE_PANEL_SIZE) — the full-vocabulary default that's
    correct for "panel_invariant" would just immediately fail its
    construction-time size check. Instead auto-derives a realistic
    SMALLER target panel via scanpy's standard highly-variable-genes
    selection (Seurat/scanpy's default `sc.pp.highly_variable_genes`
    method — the same real technique actual targeted panels, e.g.
    Xenium's ~300-500 genes, are designed around, not an arbitrary or
    random truncation). n_top_genes=512 is a deliberately realistic
    target-panel-like size (well under the 4096 safety guard), not tuned
    for accuracy.

    Mutates model_cfg["params"] in place; no-op for every other config
    (the default decoder_type="dense", and "lloki", don't use this param
    at all — "lloki" is fixed-width, not panel-based, see
    LLOKIStyleDecoder's own docstring)."""
    params = model_cfg.get("params", {})
    decoder_type = params.get("decoder_type")
    if decoder_type in ("panel_invariant", "gene_conditioned_vocabulary") and "decoder_gene_names" not in params:
        params["decoder_gene_names"] = adata.var_names.tolist()
    elif decoder_type == "gene_attention" and "decoder_gene_names" not in params:
        import scanpy as sc
        n_target = min(512, adata.n_vars)
        hvg_adata = adata.copy()
        sc.pp.highly_variable_genes(hvg_adata, n_top_genes=n_target)
        params["decoder_gene_names"] = (
            hvg_adata.var_names[hvg_adata.var["highly_variable"]].tolist()
        )
        # full_gene_names (2026-07-17): the FULL training-panel gene order,
        # needed so the model can align this decoder's restricted output
        # columns with target_expression's full-width columns for loss
        # computation — see BaseGenerativeModel._slice_target_for_decoder.
        params.setdefault("full_gene_names", adata.var_names.tolist())


def inject_storm_lite_tokenizer_gene_names(model_cfg: dict, adata) -> None:
    """If gene_encoder_type is "tokenizer"/"tokenizer_novae" (2026-07-19
    research — see TokenizedGeneEncoder's own docstring in conditioning.py),
    auto-derive storm_lite_tokenizer_gene_names via the same HVG-selection
    technique inject_decoder_gene_names already uses for
    decoder_type="gene_attention" (same MAX_SAFE_PANEL_SIZE-guarded
    "self-attention over gene tokens is O(n_panel^2)" reasoning —
    TokenizedGeneEncoder's own attention-based pooling has the identical
    cost profile). storm_lite_tokenizer_full_gene_names is always the full
    training panel (adata.var_names), needed to align the selected genes'
    columns in raw_expr every forward pass.

    No-op for every config that doesn't set gene_encoder_type to one of
    the two tokenizer variants — zero effect on any existing config."""
    params = model_cfg.get("params", {})
    gene_encoder_type = params.get("gene_encoder_type")
    if gene_encoder_type not in ("tokenizer", "tokenizer_novae"):
        return
    params.setdefault("storm_lite_tokenizer_full_gene_names", adata.var_names.tolist())
    if "storm_lite_tokenizer_gene_names" not in params:
        import scanpy as sc
        n_target = min(512, adata.n_vars)
        hvg_adata = adata.copy()
        sc.pp.highly_variable_genes(hvg_adata, n_top_genes=n_target)
        params["storm_lite_tokenizer_gene_names"] = (
            hvg_adata.var_names[hvg_adata.var["highly_variable"]].tolist()
        )


def inject_tokenized_gene_names(model_cfg: dict, adata) -> None:
    """If gene_encoder_type is "tokenized" (2026-07-23 round-4 architecture
    matrix — wires TokenizedGeneEncoder into HierarchicalMissingTissueEncoder/
    hierarchical_gene_transport_regressor, previously only used by
    StormLiteContextEncoder), auto-derive tokenized_gene_names via the same
    HVG-selection technique inject_storm_lite_tokenizer_gene_names already
    uses for that model's own "tokenizer"/"tokenizer_novae" options — same
    MAX_SAFE_PANEL_SIZE-guarded self-attention-over-gene-tokens cost profile.
    tokenized_full_gene_names is always the full training panel
    (adata.var_names), needed to align the selected genes' columns in
    context["expression"] every forward pass.

    Kept as its own function (not folded into
    inject_storm_lite_tokenizer_gene_names) because the two encoders use
    different param names for the same underlying purpose — same
    one-function-per-injected-param convention as
    inject_stpath_gene_names/inject_decoder_gene_names. No-op for every
    config that doesn't set gene_encoder_type="tokenized"."""
    params = model_cfg.get("params", {})
    if params.get("gene_encoder_type") != "tokenized":
        return
    params.setdefault("tokenized_full_gene_names", adata.var_names.tolist())
    if "tokenized_gene_names" not in params:
        import scanpy as sc
        n_target = min(512, adata.n_vars)
        hvg_adata = adata.copy()
        sc.pp.highly_variable_genes(hvg_adata, n_top_genes=n_target)
        params["tokenized_gene_names"] = (
            hvg_adata.var_names[hvg_adata.var["highly_variable"]].tolist()
        )


def _stpath_frozen_table_cache_path(cfg, model_cfg: dict) -> Path:
    """Where the extracted (small, [d_model, n_genes]) frozen STPath gene
    table gets cached across runs -- avoids reloading the whole STPath
    checkpoint (multi-GB) every launch just to re-derive a per-gene-panel
    lookup that never changes for a fixed (gene panel, checkpoint) pair.
    Keyed by experiment_name since the gene panel is config-specific."""
    experiment_name = str(cfg.get("experiment_name", "experiment"))
    return _cache_root(cfg) / "stpath_frozen_gene_table_cache" / f"{experiment_name}.npz"


def inject_stpath_frozen_gene_table(model_cfg: dict, adata, cfg) -> None:
    """If gene_encoder_type is "stpath_frozen_table", auto-derive
    stpath_frozen_gene_table by extracting STPath's real pretrained
    gene_embed columns for this config's actual gene panel (see
    src/models/stpath_gene_table.py -- isolates STPath's PRETRAINING from
    its whole architecture, a distinction the existing STPath-vs-unfrozen
    ablation in docs/results_log.md cannot make on its own). Reads
    stpath_gene_voc_path/stpath_model_weight_path/stpath_d_model from
    model_cfg params -- same config convention context_encoder_type="stpath"
    configs already use for the first two
    (${oc.env:STPATH_GENE_VOC_PATH}/${oc.env:STPATH_MODEL_WEIGHT_PATH}), not
    a new one. These three keys are config-only INPUTS to this extraction
    step, not accepted by HierarchicalGeneTransportRegressor's own
    constructor (build_model does a plain **params unpack, no filtering) --
    always popped from params before returning, on every path, so they
    never leak into build_model and crash with an unexpected-keyword error.
    No-op for every config that doesn't set
    gene_encoder_type="stpath_frozen_table"."""
    params = model_cfg.get("params", {})
    if params.get("gene_encoder_type") != "stpath_frozen_table":
        return
    if "stpath_frozen_gene_table" in params:
        params.pop("stpath_gene_voc_path", None)
        params.pop("stpath_model_weight_path", None)
        params.pop("stpath_d_model", None)
        return
    voc_path = params.pop("stpath_gene_voc_path", None)
    weight_path = params.pop("stpath_model_weight_path", None)
    d_model = int(params.pop("stpath_d_model", 512))
    if not voc_path or not weight_path:
        raise ValueError(
            "gene_encoder_type='stpath_frozen_table' requires stpath_gene_voc_path and "
            "stpath_model_weight_path (same params context_encoder_type='stpath' configs "
            "already set, typically ${oc.env:STPATH_GENE_VOC_PATH}/"
            "${oc.env:STPATH_MODEL_WEIGHT_PATH})"
        )
    gene_names = adata.var_names.tolist()

    cache_path = _stpath_frozen_table_cache_path(cfg, model_cfg)
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        if cached["gene_names"].tolist() == gene_names:
            print(f"inject_stpath_frozen_gene_table: loaded cached table for "
                  f"{len(gene_names)} genes from {cache_path} "
                  f"(delete this file to force a recompute).")
            params["stpath_frozen_gene_table"] = cached["table"].tolist()
            return
        print(f"inject_stpath_frozen_gene_table: cache at {cache_path} does not match "
              f"the current gene panel — recomputing.")

    from src.models.stpath_gene_table import extract_stpath_gene_embedding_table
    print(f"Extracting STPath's frozen gene_embed table for {len(gene_names)} genes "
          f"(one-time cost, cached to {cache_path} so future runs skip this step)...")
    table, report = extract_stpath_gene_embedding_table(
        gene_names, voc_path, weight_path, d_model=d_model,
    )
    print(f"inject_stpath_frozen_gene_table: {report['n_found']}/{report['n_genes']} genes "
          f"matched STPath's vocabulary ({report['n_missing']} missing genes get a zero "
          f"frozen column, contributing nothing through this pathway).")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savez(
        cache_path,
        table=table,
        gene_names=np.asarray(gene_names, dtype=object),
    )
    params["stpath_frozen_gene_table"] = table.tolist()


def inject_single_sample_n_genes(model_cfg: dict, adata) -> None:
    """Single-sample n_genes has always been a MANUALLY hardcoded config
    value (e.g. "n_genes: 16570  # INT1 after QC"), unlike every other
    injected param — a real gap found 2026-07-19 when a shared, reused
    HEST-1k download (a labmate's copy, different from whoever originally
    computed 16570) produced a genuinely different post-QC gene count
    (19179) for the same sample_id, crashing with a matmul shape error
    deep in MLPGeneEncoder's first Linear layer. There is no legitimate
    reason for n_genes to differ from adata.n_vars post-QC — nothing
    downstream subsets genes to hit a target count, so any stored value
    is only ever right by coincidence with whatever data happened to
    produce it. UNCONDITIONAL overwrite (not setdefault, unlike every
    other inject_* here) — deliberately corrects a stale/wrong hardcoded
    value rather than trusting it. Mutates model_cfg["params"] in place."""
    if model_cfg.get("name") in {"interp_baseline", "spatial_baseline"}:
        return
    params = model_cfg.get("params", {})
    params["n_genes"] = adata.n_vars


def inject_multi_sample_n_genes(model_cfg: dict, adatas: list) -> None:
    """Multi-sample training's n_genes has NO reliable manual default —
    load_multi_sample's shared-gene-panel intersection across every
    sample_ids entry almost always differs from any single sample's own
    gene count (see exp_hest1k_fm_ot_multisample.yaml's own header, which
    used to require checking real console output and hand-correcting a
    placeholder value — a real, easy-to-get-wrong step, same class of
    problem inject_stpath_gene_names/inject_decoder_gene_names/
    inject_novae_dim already solve for their own params). Auto-derives it
    from the real, already-intersected adatas[0].n_vars instead — every
    sample in adatas shares the same panel size after load_multi_sample's
    intersection (its own documented guarantee), so any one sample's
    count is correct for all of them. Mutates model_cfg["params"] in
    place; no-op if n_genes is already explicitly set."""
    if model_cfg.get("name") in {"interp_baseline", "spatial_baseline"}:
        return
    params = model_cfg.get("params", {})
    params["n_genes"] = adatas[0].n_vars


def _resolved_evaluation_gene_panels(
    cfg,
    train_samples: list[tuple],
    gene_names: list[str],
) -> dict[str, list[str]]:
    """Build evaluation-only gene panels without touching held-out targets.

    Fixed comparison panels are read from versioned JSON files. Diagnostic
    variance panels are ranked using normalized/log1p expression from training
    samples only. The returned names merely slice predictions during metric
    calculation; the model still trains and predicts the complete shared gene
    vocabulary.
    """
    evaluation = cfg.get("evaluation", {})
    panels: dict[str, list[str]] = {}

    fixed_paths = evaluation.get("fixed_gene_panel_paths", {})
    for panel_name, raw_path in fixed_paths.items():
        path = Path(str(raw_path))
        if not path.is_file():
            raise FileNotFoundError(f"fixed evaluation gene panel is missing: {path}")
        payload = json.loads(path.read_text())
        genes = payload.get("genes") if isinstance(payload, dict) else payload
        if not isinstance(genes, list) or not genes:
            raise ValueError(f"fixed evaluation gene panel must contain a non-empty gene list: {path}")
        panels[str(panel_name)] = list(dict.fromkeys(str(gene) for gene in genes))

    requested_sizes = sorted({
        int(size) for size in evaluation.get("train_variance_gene_panel_sizes", [])
    })
    if requested_sizes:
        if requested_sizes[0] <= 0:
            raise ValueError("evaluation.train_variance_gene_panel_sizes must be positive")
        if requested_sizes[-1] > len(gene_names):
            raise ValueError(
                "evaluation train-variance panel exceeds the full shared gene vocabulary: "
                f"{requested_sizes[-1]} > {len(gene_names)}"
            )
        # Accumulate moments in float64 to avoid concatenating every training
        # spot and to make the ranking deterministic across machines.
        count = 0
        total = np.zeros(len(gene_names), dtype=np.float64)
        total_sq = np.zeros(len(gene_names), dtype=np.float64)
        for sample in train_samples:
            expression = np.asarray(sample[1], dtype=np.float64)
            if expression.ndim != 2 or expression.shape[1] != len(gene_names):
                raise ValueError("training expression does not match the shared gene vocabulary")
            count += expression.shape[0]
            total += expression.sum(axis=0)
            total_sq += np.square(expression).sum(axis=0)
        if count < 2:
            raise ValueError("at least two training spots are required to rank variable genes")
        variance = np.maximum(total_sq / count - np.square(total / count), 0.0)
        # Stable sort makes ties deterministic in the already-fixed shared
        # vocabulary order.
        ranked = np.argsort(-variance, kind="stable")
        for size in requested_sizes:
            panels[f"train_variance_top{size}"] = [gene_names[idx] for idx in ranked[:size]]

    return panels


def _main_multi_sample(cfg) -> None:
    """Train on explicit samples and validate/test on held-out samples.

    Clean multi-sample configs should declare ``data.train_sample_ids``,
    ``data.validation_sample_ids`` and ``data.test_sample_ids``. The shared
    gene panel is established from training slides only; validation/test
    slides are aligned to it and are never drawn by the training Dataset.
    """
    train_ids, validation_ids, test_ids = _validated_sample_groups(cfg)
    if not train_ids:
        raise ValueError("multi-sample training requires data.train_sample_ids or data.sample_ids")
    # Establish the model vocabulary from training slides only. Validation
    # and test slides are loaded afterwards against that immutable panel;
    # neither split can shrink or reorder the output vocabulary.
    train_ids = list(dict.fromkeys(train_ids))
    train_loaded_samples, train_adatas = load_multi_sample_data(cfg, sample_ids=train_ids)
    gene_names = train_adatas[0].var_names.tolist()
    validation_samples, validation_adatas = [], []
    if validation_ids:
        validation_samples, validation_adatas = load_multi_sample_data(
            cfg, sample_ids=validation_ids, reference_gene_names=gene_names
        )
    # Do not even load final-test expression before optimization/model
    # selection completes. It is unnecessary for construction: the gene
    # vocabulary and scaling come only from training slides, while validation
    # slides supply checkpoint selection. Test slides are loaded below only
    # after the final trained state has been selected and saved.
    all_ids = [*train_ids, *validation_ids]
    samples = [*train_loaded_samples, *validation_samples]
    adatas = [*train_adatas, *validation_adatas]
    by_id = {str(sid): (sample, adata) for sid, sample, adata in zip(all_ids, samples, adatas)}
    train_samples = [by_id[str(sid)][0] for sid in train_ids]
    evaluation_gene_panels = _resolved_evaluation_gene_panels(
        cfg, train_samples, gene_names
    )
    fit_adatas = train_adatas
    augment = bool(cfg.training.get("augment_coords", False))
    coord_scale = float(np.mean([sample[0][:, :2].std() for sample in train_samples]))
    context_encoder_type = cfg.model.get("params", {}).get("context_encoder_type", "builtin")

    novae_dim = None
    for sample in samples:
        for source in (sample[6], sample[7]):
            if source is not None:
                novae_dim = int(source.shape[1])
        for provider in (sample[8], sample[9]):
            if provider is not None and provider.output_dim is not None:
                novae_dim = int(provider.output_dim)

    def _inject(model_cfg: dict) -> None:
        inject_multi_sample_n_genes(model_cfg, fit_adatas)
        inject_organ_tech_vocab(model_cfg, fit_adatas)
        inject_coord_scale(model_cfg, coord_scale)
        inject_decoder_gene_names(model_cfg, fit_adatas[0])
        inject_pretrained_autoencoder_gene_names(model_cfg, fit_adatas[0])
        inject_stpath_gene_names(model_cfg, fit_adatas[0])
        inject_universal_gene_names(model_cfg, fit_adatas[0])
        inject_storm_lite_tokenizer_gene_names(model_cfg, fit_adatas[0])
        inject_tokenized_gene_names(model_cfg, fit_adatas[0])
        inject_expression_preprocessing(model_cfg, fit_adatas[0])
        inject_residual_gene_scale(model_cfg, [sample[1] for sample in train_samples])
        inject_direct_regression_stats(model_cfg, [sample[1] for sample in train_samples])
        inject_transport_gene_scale(model_cfg, [sample[1] for sample in train_samples])
        if novae_dim is not None:
            if context_encoder_type == "stpath":
                inject_stpath_novae_dim(model_cfg, novae_dim)
            else:
                inject_novae_dim(model_cfg, novae_dim)
                inject_scfoundation_dim(model_cfg, novae_dim)

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    _inject(model_cfg)
    # stpath_frozen_gene_table is NOT in _inject: it needs to read real
    # stpath_gene_voc_path/stpath_model_weight_path FILES, which only hold
    # real paths on the RESOLVED copy -- the unresolved copy below
    # deliberately keeps ${oc.env:...} as literal placeholder strings (see
    # its own comment), so calling this on that copy would try to open a
    # literal "${oc.env:...}" path and crash. Computed once here, copied
    # (not re-derived) into the unresolved copy afterward -- same "derived
    # data is safe to bake into both copies, only PATHS must stay
    # unresolved" reasoning as inject_stpath_gene_names/novae_dim above.
    inject_stpath_frozen_gene_table(model_cfg, fit_adatas[0], cfg)
    model = build_model(model_cfg)
    init_checkpoint_dir = cfg.training.get("init_checkpoint_dir")
    if init_checkpoint_dir:
        load_pretrained_weights_into(model, init_checkpoint_dir)
    unresolved_model_cfg = OmegaConf.to_container(cfg.model, resolve=False)
    _inject(unresolved_model_cfg)
    if "stpath_frozen_gene_table" in model_cfg.get("params", {}):
        unresolved_model_cfg.setdefault("params", {})["stpath_frozen_gene_table"] = (
            model_cfg["params"]["stpath_frozen_gene_table"]
        )

    checkpoint_dir = cfg.training.get("checkpoint_dir", f"results/checkpoints/{cfg.experiment_name}")
    composite_train_obs_names = [
        f"{sid}:{name}"
        for sid in train_ids
        for name in by_id[str(sid)][1].obs_names
    ]
    training_bank, training_bank_path = _training_seed_bank_for_config(
        cfg, composite_train_obs_names
    )
    write_run_manifest(cfg, checkpoint_dir, training_mask_bank_path=training_bank_path)
    split_manifest = {
        "train_sample_ids": train_ids,
        "validation_sample_ids": validation_ids,
        "test_sample_ids": test_ids,
        "shared_gene_count": len(gene_names),
    }
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    (Path(checkpoint_dir) / "sample_split.json").write_text(json.dumps(split_manifest, indent=2))
    (Path(checkpoint_dir) / "evaluation_gene_panels.json").write_text(
        json.dumps({
            "selection_scope": "training_samples_only_or_fixed_external_panel",
            "training_and_output_vocabulary": "full_shared_gene_panel",
            "panels": evaluation_gene_panels,
        }, indent=2)
    )

    evaluation_cfg = cfg.get("evaluation", {})
    mask_bank_dir = Path(evaluation_cfg.get("mask_bank_dir", "results/mask_banks"))

    def _sample_bank(sample_id: str, sample, adata):
        coords3d, _expr, slice_ids = sample[0], sample[1], sample[2]
        counts = {
            "validation": int(evaluation_cfg.get("n_validation_masks", 4)),
            "test": int(evaluation_cfg.get("n_test_masks", 8)),
        }
        seeds = {
            "validation": int(evaluation_cfg.get("validation_seed", 700_000)),
            "test": int(evaluation_cfg.get("test_seed", 900_000)),
        }
        path = mask_bank_dir / f"{sample_id}.json"
        return ensure_mask_bank(path, coords3d, slice_ids, adata.obs_names, cfg.masking, counts, seeds), path

    def _novae_dict(sample):
        return {
            "context_gene_features": sample[6],
            "context_novae_features": sample[7],
            "context_gene_feature_provider": sample[8],
            "context_novae_feature_provider": sample[9],
            "context_niche_feature_provider": sample[11],
        }

    callbacks = []
    checkpoint_every_n_steps = cfg.training.get("checkpoint_every_n_steps")
    if checkpoint_every_n_steps:
        callbacks.append(PeriodicCheckpointCallback(
            unresolved_model_cfg, gene_names, checkpoint_dir,
            save_every_n_steps=checkpoint_every_n_steps,
        ))
    log_print_every_n_steps = cfg.training.get("log_print_every_n_steps")
    if log_print_every_n_steps:
        callbacks.append(PeriodicPrintCallback(log_print_every_n_steps))
    ema_decay = cfg.training.get("ema_decay")
    ema_callback = EMACallback(ema_decay) if ema_decay else None
    if ema_callback is not None:
        callbacks.append(ema_callback)

    validation_cfg = cfg.get("validation", {})
    if validation_ids and bool(validation_cfg.get("enabled", True)):
        validation_items = []
        for sid in validation_ids:
            sample, adata = by_id[str(sid)]
            bank, _ = _sample_bank(str(sid), sample, adata)
            coords3d, expr, slice_ids, images, organ, tech = sample[:6]
            for record in split_records(bank, "validation"):
                context_mask, query_mask = record_masks(record, adata.obs_names)
                ni = _novae_dict(sample)
                validation_items.append(_build_masked_item(
                    coords3d, expr, slice_ids, cfg.masking, images, int(record["seed"]),
                    **ni, organ=organ, tech=tech, augment=False,
                    image_mode=str(evaluation_cfg.get("validation_image_mode", "full")),
                    context_gex_mode=str(evaluation_cfg.get("context_gex_mode", "full")),
                    fixed_context_mask=context_mask, fixed_query_mask=query_mask,
                    slide_context=sample[10],
                    strict_broken_region=bool(cfg.data.get("strict_broken_region", False)),
                    query_patch_size=float(cfg.data.get("query_patch_size_fullres", 224.0)),
                ))
        validation_callback = FixedMaskValidationCallback(
            validation_items,
            every_n_steps=int(validation_cfg.get("every_n_steps", 1000)),
            patience_checks=int(validation_cfg.get("patience_checks", 5)),
            min_delta=float(validation_cfg.get("min_delta", 1e-4)),
            metric=str(validation_cfg.get("metric", "rmse")),
            n_samples=int(validation_cfg.get("n_samples", 4)),
            history_path=Path(checkpoint_dir) / "validation_history.json",
            seed=int(validation_cfg.get("sampling_seed", 800_000)),
            early_stopping_min_steps=int(validation_cfg.get("early_stopping_min_steps", 0)),
            require_anchor_improvement=bool(validation_cfg.get("require_anchor_improvement", False)),
            anchor_min_delta=float(validation_cfg.get("anchor_min_delta", 0.0)),
            min_correction_rms=float(validation_cfg.get("min_correction_rms", 0.0)),
            quality_gate_path=Path(checkpoint_dir) / "quality_gate.json",
        )
        callbacks.append(validation_callback)
    else:
        validation_callback = None

    if any(parameter.requires_grad for parameter in model.parameters()):
        dataset = MultiSampleMaskedContextQueryDataset(
            train_samples, cfg.masking, n_items=int(cfg.training.epochs),
            base_seed=int(cfg.training.seed), augment=augment,
            image_mode=str(cfg.training.get("image_mode", "full")),
            query_image_dropout_p=float(cfg.training.get("query_image_dropout_p", 0.0)),
            all_image_dropout_p=float(cfg.training.get("all_image_dropout_p", 0.0)),
            context_gex_mode=str(cfg.training.get("context_gex_mode", "full")),
            context_gex_dropout_p=float(cfg.training.get("context_gex_dropout_p", 0.0)),
            seed_schedule=training_bank["seeds"],
            strict_broken_region=bool(cfg.data.get("strict_broken_region", False)),
            query_patch_size=float(cfg.data.get("query_patch_size_fullres", 224.0)),
        )
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="auto",
            # devices=1 (2026-07-23, real incident): every launcher script in
            # this project scopes a single GPU via CUDA_VISIBLE_DEVICES before
            # invoking python, so accelerator="auto" always resolved to one
            # device in practice -- but a bare, unscoped invocation (all GPUs
            # visible) let Lightning's default device count silently launch
            # DDP across every visible GPU, stepping on whatever else was
            # running there. Explicit devices=1 makes this safe regardless of
            # how many GPUs happen to be visible to the process.
            devices=1,
            log_every_n_steps=int(cfg.training.log_every_n_steps),
            enable_checkpointing=False,
            logger=build_experiment_logger(cfg, checkpoint_dir),
            callbacks=callbacks,
            gradient_clip_val=(
                float(cfg.training.get("gradient_clip_val", 1.0))
                if model.automatic_optimization else None
            ),
        )
        trainer.fit(model, make_dataloader(dataset, cfg))
        if ema_callback is not None and validation_callback is None:
            ema_callback.apply_to_model(model)
        elif ema_callback is not None:
            print("best held-out-sample validation state selected; EMA final-state application skipped")
        save_trained_model(model, unresolved_model_cfg, gene_names, checkpoint_dir)
        if validation_callback is not None:
            validation_callback.raise_if_quality_gate_failed()

    if test_ids:
        from src.evaluation.audit_evaluation import evaluate_model_on_mask_bank
        test_samples, test_adatas = load_multi_sample_data(
            cfg, sample_ids=test_ids, reference_gene_names=gene_names
        )
        test_by_id = {
            str(sid): (sample, adata)
            for sid, sample, adata in zip(test_ids, test_samples, test_adatas)
        }
        test_results = {}
        for sid in test_ids:
            sample, adata = test_by_id[str(sid)]
            bank, bank_path = _sample_bank(str(sid), sample, adata)
            coords3d, expr, slice_ids, images, organ, tech = sample[:6]
            test_results[str(sid)] = evaluate_model_on_mask_bank(
                model, cfg, adata, coords3d, expr, slice_ids, images, bank,
                _novae_dict(sample),
                output_path=Path(checkpoint_dir) / f"audit_test_metrics_{sid}.json",
                organ=organ, tech=tech,
                slide_context=sample[10],
                gene_panels=evaluation_gene_panels,
                raw_counts=(
                    adata.layers["raw_counts"] if "raw_counts" in adata.layers else None
                ),
                raw_library_size=(
                    adata.obs["_scilifestdl_raw_library_size"].to_numpy()
                    if "_scilifestdl_raw_library_size" in adata.obs else None
                ),
                expression_target_sum=float(cfg.data.get("expression_target_sum", 1e4)),
            )
        aggregate = {}
        aggregate_metrics = (
            "pcc", "n_pcc_genes", "rmse", "nonzero_auc", "st_fid", "st_mmd",
            "spatial_domain_plausibility", "predictive_std", "interval90_coverage",
            "pcc_raw_log1p", "n_pcc_raw_log1p_genes", "rmse_raw_log1p",
            *(
                metric
                for panel_name in evaluation_gene_panels
                for metric in (
                    f"pcc_{panel_name}",
                    f"n_pcc_genes_{panel_name}",
                    f"rmse_{panel_name}",
                )
            ),
        )
        for mode in list(evaluation_cfg.get("image_modes", ["full"])):
            mode_rows = []
            for sid, result in test_results.items():
                summary = result["image_modes"][str(mode)]["summary"]
                row = {"sample_id": sid}
                for metric in aggregate_metrics:
                    value = summary.get(metric, {})
                    row[metric] = value.get("mean") if isinstance(value, dict) else None
                mode_rows.append(row)
            mode_summary = {"per_sample": mode_rows}
            for metric in aggregate_metrics:
                values = [
                    float(row[metric]) for row in mode_rows
                    if row.get(metric) is not None and np.isfinite(float(row[metric]))
                ]
                mode_summary[f"{metric}_mean"] = (
                    float(np.mean(values)) if values else float("nan")
                )
            aggregate[str(mode)] = mode_summary
        heldout_summary = {
            "version": 3,
            "evaluation_scope": "heldout_samples",
            "primary_image_mode": _primary_image_mode(cfg),
            "context_gex_mode": str(evaluation_cfg.get("context_gex_mode", "full")),
            "modality_ablation": str(cfg.data.get("modality_ablation", "both")),
            # Frozen external baselines such as official STPath may emit only
            # the genes represented by their released vocabulary.
            "n_evaluated_genes": int(
                next(iter(test_results.values()))["n_evaluated_genes"]
            ),
            "n_shared_training_genes": len(gene_names),
            "gene_panels": next(iter(test_results.values())).get("gene_panels", {}),
            "test_sample_ids": test_ids,
            "image_modes": aggregate,
        }
        (Path(checkpoint_dir) / "heldout_sample_summary.json").write_text(
            json.dumps(heldout_summary, indent=2)
        )
        primary_mode = heldout_summary["primary_image_mode"]
        primary = aggregate[primary_mode]
        print(f"held-out-sample primary image mode: {primary_mode}")
        print(f"held-out-sample PCC: {primary['pcc_mean']:.4f}")
        print(f"held-out-sample RMSE: {primary['rmse_mean']:.4f}")
        for panel_name in evaluation_gene_panels:
            evaluated = heldout_summary["gene_panels"][panel_name]["evaluated_count"]
            print(
                f"held-out-sample {panel_name} (n={evaluated}) "
                f"PCC: {primary[f'pcc_{panel_name}_mean']:.4f} "
                f"RMSE: {primary[f'rmse_{panel_name}_mean']:.4f}"
            )
    else:
        print("WARNING: no data.test_sample_ids configured; no held-out-sample test was run.")


def main(cfg_path: str, overrides: list[str] | None = None):
    cfg = OmegaConf.load(cfg_path)
    if overrides:
        # dotlist overrides, e.g. ["training.epochs=2"] — smoke-testing a
        # config without editing the file itself (2026-07-15, checking all
        # 18 task #19 configs actually run before committing to full-length
        # training on each)
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    _validate_task_contract(cfg)
    torch.manual_seed(cfg.training.seed)
    # 2026-07-16: free speedup on Ampere+ GPUs (A100 etc, Tensor Cores) -
    # PyTorch defaults FP32 matmuls to full precision even where TF32
    # would do, leaving Tensor Core throughput on the table. "high"
    # (TF32) is the standard/recommended default for training - real,
    # negligible-in-practice precision cost, meaningful speedup. No-op on
    # MPS/CPU (the setting only affects CUDA matmuls).
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    # Multi-sample training (2026-07-16, cfg.data.sample_ids as a LIST in
    # place of the single-sample configs' cfg.data.sample_id) — a genuinely
    # separate, simpler code path rather than threading a branch through
    # every line below. UPDATED 2026-07-17: load_multi_sample_data/
    # MultiSampleMaskedContextQueryDataset now DO support images/Novae/
    # STPath (see load_multi_sample_data's own updated docstring for the
    # real gap this closed) — kept as a separate function regardless,
    # since sharing one code path would still mean threading extra
    # branching through every line below for no real benefit now that
    # both paths already work correctly independently.
    if cfg.data.get("sample_ids") is not None:
        _main_multi_sample(cfg)
        return

    adata, coords3d, expr, slice_ids, images = _load_data(cfg)
    slide_context = load_slide_context(
        cfg, str(cfg.data.get("sample_id", "sample")), images, coords3d
    )

    # A same-slide diagnostic must reserve evaluation targets before *any*
    # learned-model statistic is derived. Besides excluding these rows from
    # training items below, fit gene-loss scaling and coordinate scaling only
    # on eligible observations. Otherwise the final targets would influence
    # optimization even though they never appeared in a minibatch.
    bank, mask_bank_path = _mask_bank_for_config(cfg, adata, coords3d, slice_ids)
    training_excluded_mask = None
    if bool(cfg.training.get("exclude_evaluation_query_spots", False)):
        training_excluded_mask = evaluation_query_exclusion_mask(bank, adata.obs_names)
    training_stat_mask = (
        np.ones(adata.n_obs, dtype=bool)
        if training_excluded_mask is None else ~training_excluded_mask
    )
    training_expr = expr[training_stat_mask]

    model_params = cfg.model.get("params", {})
    context_encoder_type = model_params.get("context_encoder_type", "builtin")
    novae_inputs = prepare_novae_inputs(
        cfg, adata, model_params, coords3d, slice_ids, sample_id=cfg.data.get("sample_id")
    )
    scfoundation_inputs = prepare_scfoundation_inputs(
        cfg, adata, model_params, sample_id=cfg.data.get("sample_id")
    )
    if scfoundation_inputs["mode"] != "disabled":
        # Same overlay reasoning as load_multi_sample_data's identical merge.
        novae_inputs["context_gene_features"] = scfoundation_inputs["context_gene_features"]
        novae_inputs["context_gene_feature_provider"] = scfoundation_inputs["context_gene_feature_provider"]
        novae_inputs["feature_dim"] = scfoundation_inputs["feature_dim"]
    niche_inputs = prepare_niche_inputs(cfg, adata, model_params, sample_id=cfg.data.get("sample_id"))
    context_gene_features = novae_inputs["context_gene_features"]
    context_novae_features = novae_inputs["context_novae_features"]
    context_gene_feature_provider = novae_inputs["context_gene_feature_provider"]
    context_novae_feature_provider = novae_inputs["context_novae_feature_provider"]
    context_niche_feature_provider = niche_inputs["context_niche_feature_provider"]
    # Threaded into the SAME novae_inputs dict (rather than passed
    # separately) since every downstream helper below already accepts one
    # "context provider bag" dict and reads specific keys via .get() --
    # _fixed_items_from_bank/_fixed_training_seed_item/
    # evaluate_model_on_mask_bank all forward whatever novae_inputs
    # contains, so adding this key here is enough for all three.
    novae_inputs["context_niche_feature_provider"] = context_niche_feature_provider

    # 2026-07-17: RandomFourierFeatures real-scale bug fix — see
    # inject_coord_scale's own docstring
    coord_scale = float(coords3d[training_stat_mask, :2].std())

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    inject_single_sample_n_genes(model_cfg, adata)
    inject_stpath_gene_names(model_cfg, adata)
    inject_universal_gene_names(model_cfg, adata)
    inject_decoder_gene_names(model_cfg, adata)
    inject_storm_lite_tokenizer_gene_names(model_cfg, adata)
    inject_tokenized_gene_names(model_cfg, adata)
    # Only ever called on this RESOLVED copy (see the unresolved copy's own
    # comment below, and inject_stpath_frozen_gene_table's docstring) --
    # reads real stpath_gene_voc_path/stpath_model_weight_path FILES, which
    # only hold real paths here, not on the deliberately-unresolved copy.
    inject_stpath_frozen_gene_table(model_cfg, adata, cfg)
    inject_expression_preprocessing(model_cfg, adata)
    inject_residual_gene_scale(model_cfg, [training_expr])
    inject_direct_regression_stats(model_cfg, [training_expr])
    inject_transport_gene_scale(model_cfg, [training_expr])
    inject_coord_scale(model_cfg, coord_scale)
    if novae_inputs["feature_dim"] is not None:
        if context_encoder_type == "stpath":
            inject_stpath_novae_dim(model_cfg, novae_inputs["feature_dim"])
        else:
            inject_novae_dim(model_cfg, novae_inputs["feature_dim"])
            inject_scfoundation_dim(model_cfg, novae_inputs["feature_dim"])
    model = build_model(model_cfg)
    # init_checkpoint_dir (2026-07-19): opt-in pretrain->finetune warm
    # start — see _main_multi_sample's identical block / load_pretrained_
    # weights_into's own docstring for the full reasoning.
    init_checkpoint_dir = cfg.training.get("init_checkpoint_dir")
    if init_checkpoint_dir:
        load_pretrained_weights_into(model, init_checkpoint_dir)
    # UNRESOLVED copy, saved (not model_cfg above) so a STPath config's
    # ${oc.env:STPATH_GENE_VOC_PATH}/${oc.env:STPATH_MODEL_WEIGHT_PATH}
    # interpolations stay literal in the checkpoint rather than getting
    # baked in as THIS machine's resolved path — see load_trained_model's
    # docstring for the real cross-machine bug this fixes (2026-07-15).
    # novae_dim, unlike the STPath paths, is a plain int with nothing
    # machine-specific to resolve — injecting it here too just keeps the
    # saved config self-consistent with model_cfg above.
    unresolved_model_cfg = OmegaConf.to_container(cfg.model, resolve=False)
    inject_single_sample_n_genes(unresolved_model_cfg, adata)
    inject_stpath_gene_names(unresolved_model_cfg, adata)
    inject_universal_gene_names(unresolved_model_cfg, adata)
    inject_decoder_gene_names(unresolved_model_cfg, adata)
    inject_storm_lite_tokenizer_gene_names(unresolved_model_cfg, adata)
    inject_tokenized_gene_names(unresolved_model_cfg, adata)
    if "stpath_frozen_gene_table" in model_cfg.get("params", {}):
        unresolved_model_cfg.setdefault("params", {})["stpath_frozen_gene_table"] = (
            model_cfg["params"]["stpath_frozen_gene_table"]
        )
    inject_expression_preprocessing(unresolved_model_cfg, adata)
    inject_residual_gene_scale(unresolved_model_cfg, [training_expr])
    inject_direct_regression_stats(unresolved_model_cfg, [training_expr])
    inject_transport_gene_scale(unresolved_model_cfg, [training_expr])
    inject_coord_scale(unresolved_model_cfg, coord_scale)
    if novae_inputs["feature_dim"] is not None:
        if context_encoder_type == "stpath":
            inject_stpath_novae_dim(unresolved_model_cfg, novae_inputs["feature_dim"])
        else:
            inject_novae_dim(unresolved_model_cfg, novae_inputs["feature_dim"])
            inject_scfoundation_dim(unresolved_model_cfg, novae_inputs["feature_dim"])


    # Train (skipped entirely for parameter-free baselines like interp_baseline) --
    augment = cfg.training.get("augment_coords", False)
    # organ/tech (2026-07-17, real bug fix — see
    # src/evaluation/run_comparison.py's _build_shared_eval for the same
    # fix and full reasoning): these were NEVER populated for the
    # single-sample path, silently leaving context["tech"]/query["tech"]
    # None for every training step.
    organ = str(adata.obs["organ"].iloc[0]) if "organ" in adata.obs else None
    tech = str(adata.obs["tech"].iloc[0]) if "tech" in adata.obs else None
    training_bank, training_bank_path = _training_seed_bank_for_config(cfg, adata.obs_names)
    checkpoint_dir = cfg.training.get("checkpoint_dir", f"results/checkpoints/{cfg.experiment_name}")
    write_run_manifest(cfg, checkpoint_dir, mask_bank_path, training_bank_path)
    if training_excluded_mask is not None:
        exclusion_path = write_training_exclusion_manifest(
            checkpoint_dir, adata.obs_names, training_excluded_mask
        )
        print(
            f"within-slide leakage guard: excluded "
            f"{int(training_excluded_mask.sum())}/{adata.n_obs} immutable evaluation "
            f"query spots from all training context and targets; audit: {exclusion_path}"
        )
    if any(parameter.requires_grad for parameter in model.parameters()):
        dataset = MaskedContextQueryDataset(
            coords3d, expr, slice_ids, cfg.masking,
            n_items=cfg.training.epochs, base_seed=cfg.training.seed, images=images,
            context_gene_features=context_gene_features,
            context_novae_features=context_novae_features,
            context_gene_feature_provider=context_gene_feature_provider,
            context_novae_feature_provider=context_novae_feature_provider,
            context_niche_feature_provider=context_niche_feature_provider,
            organ=organ, tech=tech, augment=augment,
            image_mode=str(cfg.training.get("image_mode", "full")),
            query_image_dropout_p=float(cfg.training.get("query_image_dropout_p", 0.0)),
            all_image_dropout_p=float(cfg.training.get("all_image_dropout_p", 0.0)),
            context_gex_mode=str(cfg.training.get("context_gex_mode", "full")),
            context_gex_dropout_p=float(cfg.training.get("context_gex_dropout_p", 0.0)),
            excluded_training_mask=training_excluded_mask,
            seed_schedule=training_bank["seeds"],
            slide_context=slide_context,
            strict_broken_region=bool(cfg.data.get("strict_broken_region", False)),
            query_patch_size=float(cfg.data.get("query_patch_size_fullres", 224.0)),
        )
        dataloader = make_dataloader(dataset, cfg)
        # .get() with the same default every real config's own YAML comment
        # documents, not a bare attribute access — configs that don't
        # declare checkpoint_dir (e.g. tests/test_run_comparison.py's
        # synthetic configs) must still work, not crash on a missing key
        # PeriodicCheckpointCallback (2026-07-17): opt-in via
        # training.checkpoint_every_n_steps, unset by default — see that
        # class's own docstring.
        callbacks = []
        validation_cfg = cfg.get("validation", {})
        if bool(validation_cfg.get("enabled", True)):
            if str(validation_cfg.get("mask_source", "evaluation")) == "training_seed":
                validation_items = [_fixed_training_seed_item(
                    cfg, adata, coords3d, expr, slice_ids, images, training_bank,
                    novae_inputs, training_excluded_mask, checkpoint_dir,
                    organ=organ, tech=tech, slide_context=slide_context,
                )]
            else:
                validation_items = _fixed_items_from_bank(
                    cfg, adata, coords3d, expr, slice_ids, images, bank, "validation",
                    novae_inputs, organ=organ, tech=tech, slide_context=slide_context,
                )
            validation_callback = FixedMaskValidationCallback(
                validation_items,
                every_n_steps=int(validation_cfg.get("every_n_steps", 1000)),
                patience_checks=int(validation_cfg.get("patience_checks", 5)),
                min_delta=float(validation_cfg.get("min_delta", 1e-4)),
                metric=str(validation_cfg.get("metric", "rmse")),
                n_samples=int(validation_cfg.get("n_samples", 4)),
                history_path=Path(checkpoint_dir) / "validation_history.json",
                seed=int(validation_cfg.get("sampling_seed", 800_000)),
                early_stopping_min_steps=int(validation_cfg.get("early_stopping_min_steps", 0)),
                require_anchor_improvement=bool(validation_cfg.get("require_anchor_improvement", False)),
                anchor_min_delta=float(validation_cfg.get("anchor_min_delta", 0.0)),
                min_correction_rms=float(validation_cfg.get("min_correction_rms", 0.0)),
                quality_gate_path=Path(checkpoint_dir) / "quality_gate.json",
            )
            callbacks.append(validation_callback)
        else:
            validation_callback = None

        checkpoint_every_n_steps = cfg.training.get("checkpoint_every_n_steps")
        if checkpoint_every_n_steps:
            callbacks.append(PeriodicCheckpointCallback(
                unresolved_model_cfg, adata.var_names.tolist(), checkpoint_dir,
                save_every_n_steps=checkpoint_every_n_steps,
            ))
        # PeriodicPrintCallback / gradient_clip_val (2026-07-19): see
        # _main_multi_sample's identical block above for the real
        # collapsed-run investigation these came from.
        log_print_every_n_steps = cfg.training.get("log_print_every_n_steps")
        if log_print_every_n_steps:
            callbacks.append(PeriodicPrintCallback(log_print_every_n_steps))
        # EMACallback (2026-07-19): opt-in via training.ema_decay — see
        # _main_multi_sample's identical block above for the full reasoning.
        ema_decay = cfg.training.get("ema_decay")
        ema_callback = EMACallback(ema_decay) if ema_decay else None
        if ema_callback is not None:
            callbacks.append(ema_callback)
        trainer = pl.Trainer(
            max_epochs=1,  # one pass over `n_items` fresh masking draws == old epoch count
            accelerator="auto",
            # devices=1 -- see the identical trainer construction in
            # _main_multi_sample above for the real incident this fixes.
            devices=1,
            log_every_n_steps=cfg.training.log_every_n_steps,
            enable_checkpointing=False,
            logger=build_experiment_logger(cfg, checkpoint_dir),
            callbacks=callbacks,
            gradient_clip_val=(
                float(cfg.training.get("gradient_clip_val", 1.0))
                if model.automatic_optimization else None
            ),
        )
        trainer.fit(model, dataloader)
        if ema_callback is not None and validation_callback is None:
            ema_callback.apply_to_model(model)
        elif ema_callback is not None:
            print("best fixed-mask validation state selected; EMA final-state application skipped")
        saved_path = save_trained_model(model, unresolved_model_cfg, adata.var_names.tolist(), checkpoint_dir)
        if saved_path is not None:
            print(f"Saved trained model (weights + config + gene names) to {saved_path.parent}")
        if validation_callback is not None:
            validation_callback.raise_if_quality_gate_failed()

    if not bool(cfg.get("evaluation", {}).get("enabled", True)):
        print("final audit evaluation disabled for this diagnostic run")
        return

    # Full untouched-test-bank evaluation: predictive mean, uncertainty,
    # fixed-dimensional ST-FID/ST-MMD and explicit image-availability modes.
    from src.evaluation.audit_evaluation import evaluate_model_on_mask_bank
    metrics = evaluate_model_on_mask_bank(
        model, cfg, adata, coords3d, expr, slice_ids, images, bank, novae_inputs,
        output_path=Path(checkpoint_dir) / "audit_test_metrics.json",
        organ=organ, tech=tech,
        slide_context=slide_context,
    )
    primary_mode = str(metrics["primary_image_mode"])
    primary_summary = metrics["image_modes"][primary_mode]["summary"]
    print(f"test-bank primary image mode: {primary_mode}")
    print(f"test-bank mean PCC: {primary_summary['pcc']['mean']:.4f} ± {primary_summary['pcc']['std']:.4f}")
    print(f"test-bank RMSE: {primary_summary['rmse']['mean']:.4f} ± {primary_summary['rmse']['std']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--override", nargs="*", default=[],
                         help="dotlist config overrides, e.g. --override training.epochs=2")
    args = parser.parse_args()
    main(args.config, args.override)
