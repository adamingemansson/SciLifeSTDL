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
from src.models.registry import build_model
from src.evaluation import metrics as ev


def save_trainable_state_dict(model, checkpoint_dir: str, filename: str = "trainable_weights.pt") -> Path | None:
    """Save only trainable parameters, not frozen backbones (Gigapath/
    STPath). Those are identical to the public pretrained weights and get
    reloaded fresh from HuggingFace/the local weight file every time a
    model is rebuilt (see conditioning.py/stpath_encoder.py __init__) —
    saving them again here would be pure redundancy: a STPath-conditioned
    model's full state_dict() is ~4.7GB (1.2B frozen params) vs a few
    tens of MB for just the trainable ones. No-op (returns None) for
    parameter-free models like interp_baseline.

    Added 2026-07-15 after a real question: training runs weren't saving
    ANYTHING before this, meaning a completed 10000-step run's weights
    were gone the moment run_comparison.py's _free() deleted the model
    object — any later use (e.g. testing on a newly downloaded sample)
    would have required retraining from scratch."""
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    if not trainable_names:
        return None
    state = {k: v for k, v in model.state_dict().items() if k in trainable_names}
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
    weights_path = save_trainable_state_dict(model, checkpoint_dir)
    if weights_path is None:
        return None
    out_dir = weights_path.parent
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
    return weights_path


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
    only saving once at the end of training as before)."""

    def __init__(self, model_cfg: dict, gene_names: list, checkpoint_dir: str,
                 save_every_n_steps: int):
        self.model_cfg = model_cfg
        self.gene_names = gene_names
        self.checkpoint_dir = checkpoint_dir
        self.save_every_n_steps = save_every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step > 0 and step % self.save_every_n_steps == 0:
            saved_path = save_trained_model(pl_module, self.model_cfg, self.gene_names, self.checkpoint_dir)
            if saved_path is not None:
                print(f"[PeriodicCheckpointCallback] step {step}: saved checkpoint to {saved_path.parent}")


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
    state = torch.load(in_dir / "trainable_weights.pt", map_location="cpu")
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    missing_trainable = trainable_names - set(state.keys())
    assert not missing_trainable, (
        f"saved weights at {in_dir} are missing trainable parameters this "
        f"model architecture expects: {missing_trainable} (model_cfg mismatch?)"
    )
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, gene_names


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


def _build_masked_item(coords3d: np.ndarray, expr: np.ndarray, slice_ids: np.ndarray,
                        masking_cfg, images: np.ndarray | None, seed: int,
                        context_gene_features: np.ndarray | None = None,
                        context_novae_features: np.ndarray | None = None,
                        organ: str | None = None, tech: str | None = None,
                        augment: bool = False) -> dict:
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
    original, unaugmented behavior."""
    if augment:
        coords3d = augment_coords_xy(coords3d, seed=seed + 2)
    context_mask, query_mask = make_context_query_split(coords3d, slice_ids, masking_cfg, seed)
    context_expr_source = expr if context_gene_features is None else context_gene_features
    context = {
        "coords": torch.tensor(coords3d[context_mask], dtype=torch.float32),
        "expression": torch.tensor(context_expr_source[context_mask], dtype=torch.float32),
    }
    if context_novae_features is not None:
        context["novae_features"] = torch.tensor(
            context_novae_features[context_mask], dtype=torch.float32
        )
    query = {"coords": torch.tensor(coords3d[query_mask], dtype=torch.float32)}
    if images is not None:
        context["images"] = _images_tensor(images, context_mask)
        query["images"] = _images_tensor(images, query_mask)
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
                 organ: str | None = None, tech: str | None = None,
                 augment: bool = False):
        self.coords3d = coords3d
        self.expr = expr
        self.slice_ids = slice_ids
        self.masking_cfg = masking_cfg
        self.n_items = n_items
        self.base_seed = base_seed
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
        seed = self.base_seed + idx
        return _build_masked_item(
            self.coords3d, self.expr, self.slice_ids, self.masking_cfg, self.images, seed,
            context_gene_features=self.context_gene_features,
            context_novae_features=self.context_novae_features,
            organ=self.organ, tech=self.tech, augment=self.augment,
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
    context_gene_features, context_novae_features) tuples, one per
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
                 augment: bool = False):
        assert samples, "samples must be non-empty"
        self.samples = samples
        self.masking_cfg = masking_cfg
        self.n_items = n_items
        self.base_seed = base_seed
        self.augment = augment

    def __len__(self):
        return self.n_items

    def __getitem__(self, idx):
        seed = self.base_seed + idx
        # separate RNG draw for "which sample" vs. the masking split
        # itself (seed + 1, passed to _build_masked_item) so the two
        # choices aren't spuriously correlated through a shared seed
        sample_idx = int(np.random.default_rng(seed).integers(len(self.samples)))
        (coords3d, expr, slice_ids, images, organ, tech,
         context_gene_features, context_novae_features) = self.samples[sample_idx]
        return _build_masked_item(
            coords3d, expr, slice_ids, self.masking_cfg, images, seed + 1,
            context_gene_features=context_gene_features,
            context_novae_features=context_novae_features,
            organ=organ, tech=tech, augment=self.augment,
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
    num_workers = cfg.training.get("num_workers", 0)
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
        adata, min_genes=cfg.data.min_genes, min_cells=cfg.data.min_cells
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
    own docstrings), only the corruption risk."""
    tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp{os.getpid()}")
    np.savez(tmp_path, **arrays)
    os.replace(tmp_path, cache_path)  # atomic on POSIX — no reader ever sees a partial file


def _gigapath_cache_path(cfg, sample_id: str | None = None) -> Path:
    """Where precomputed Gigapath features for this sample get cached
    across runs (see _load_images) — next to the HEST-1k data itself so
    it's obvious it belongs to that sample, not somewhere in /tmp that
    would silently vanish between sessions.

    sample_id (2026-07-17, multi-sample image/Novae support — see
    load_multi_sample_data's own docstring for the real gap this closes):
    explicit override for per-sample cache paths when iterating over
    cfg.data.sample_ids. Defaults to cfg.data.sample_id, so every
    single-sample call site's behavior is completely unchanged."""
    sid = sample_id if sample_id is not None else cfg.data.sample_id
    return Path(cfg.data.hest_data_dir) / "gigapath_cache" / f"{sid}.npz"


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
    cache_path = _gigapath_cache_path(cfg, sample_id=sample_id)
    if cache_path.exists():
        cached = np.load(cache_path)
        if np.array_equal(cached["barcodes"], barcodes):
            print(f"get_gigapath_features: loaded cached features for "
                  f"{cached['features'].shape[0]} spots from {cache_path} "
                  f"(delete this file to force a recompute).")
            return cached["features"]
        print(f"get_gigapath_features: cache at {cache_path} covers a different "
              f"barcode set than the current patches file — recomputing.")
    from src.models.conditioning import precompute_gigapath_features, _default_device
    print(f"Precomputing Gigapath features for {patches.shape[0]} spots on "
          f"{_default_device()} (one-time cost, cached to {cache_path} "
          f"so future runs skip this step)...")
    features = precompute_gigapath_features(patches)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savez(cache_path, features=features, barcodes=barcodes)
    return features


def _novae_cache_path(cfg, sample_id: str | None = None) -> Path:
    """Where precomputed Novae features for this sample get cached across
    runs — same reasoning as _gigapath_cache_path (Novae's forward pass
    over a whole sample is a real one-time cost worth not repeating on
    every `python -m src.training.train` invocation).

    sample_id: see _gigapath_cache_path's own docstring (same 2026-07-17
    multi-sample follow-up, same "defaults to cfg.data.sample_id" contract)."""
    sid = sample_id if sample_id is not None else cfg.data.sample_id
    return Path(cfg.data.hest_data_dir) / "novae_cache" / f"{sid}.npz"


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
    if params.get("gene_encoder_type") in ("novae", "both") and "novae_dim" not in params:
        params["novae_dim"] = novae_dim


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
    if params.get("context_encoder_type", "builtin") in ("builtin", "storm_lite") \
            and "coord_scale" not in params:
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
    something (see this function's real fix below)."""
    if not cfg.data.get("use_images", False):
        return adata, None
    sid = sample_id if sample_id is not None else cfg.data.sample_id
    patches, barcodes = loaders.load_hest_patches(cfg.data.hest_data_dir, sid)

    model_params = cfg.model.get("params", {})
    uses_frozen_gigapath = (
        model_params.get("image_encoder_type") == "gigapath"
        or model_params.get("context_encoder_type") in ("stpath", "storm_lite")
    )
    if uses_frozen_gigapath:
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


def load_multi_sample_data(cfg) -> tuple[list[tuple], list]:
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
    adatas = loaders.load_multi_sample(
        cfg.data.hest_data_dir, list(cfg.data.sample_ids),
        min_genes=cfg.data.min_genes, min_cells=cfg.data.min_cells,
        organs=list(cfg.data.organs) if cfg.data.get("organs") is not None else None,
        techs=list(cfg.data.techs) if cfg.data.get("techs") is not None else None,
    )
    model_params = cfg.model.get("params", {})
    context_encoder_type = model_params.get("context_encoder_type", "builtin")
    use_images = cfg.data.get("use_images", False)

    samples = []
    updated_adatas = []
    for sample_id, adata in zip(cfg.data.sample_ids, adatas):
        images = None
        if use_images:
            adata, images = _load_images(cfg, adata, sample_id=sample_id)

        coords3d = loaders.get_coords_3d(adata)
        expr = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
        slice_ids = adata.obs["slice_id"].to_numpy()
        # organ/tech are constant across a whole sample's obs (see
        # loaders.load_hest_sample) — any row's value is the sample's value
        organ = str(adata.obs["organ"].iloc[0]) if "organ" in adata.obs else None
        tech = str(adata.obs["tech"].iloc[0]) if "tech" in adata.obs else None

        context_gene_features = None
        context_novae_features = None
        if context_encoder_type == "builtin" and model_params.get("gene_encoder_type") == "novae":
            context_gene_features = get_novae_features(cfg, adata, sample_id=sample_id)
        elif (context_encoder_type == "stpath"
              and model_params.get("stpath_new_gene_encoder_type") in ("novae", "both")):
            context_novae_features = get_novae_features(cfg, adata, sample_id=sample_id)
        elif (context_encoder_type == "storm_lite"
              and model_params.get("gene_encoder_type") in ("novae", "both")):
            context_novae_features = get_novae_features(cfg, adata, sample_id=sample_id)

        samples.append((coords3d, expr, slice_ids, images, organ, tech,
                         context_gene_features, context_novae_features))
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


def inject_stpath_gene_names(model_cfg: dict, adata) -> None:
    """If a config sets context_encoder_type: "stpath" (task #18),
    auto-derive stpath_gene_names from the loaded AnnData's var_names
    rather than requiring ~16570 gene symbols hardcoded into a YAML file.
    Mutates model_cfg["params"] in place; no-op for every other config."""
    params = model_cfg.get("params", {})
    if params.get("context_encoder_type") == "stpath" and "stpath_gene_names" not in params:
        params["stpath_gene_names"] = adata.var_names.tolist()


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
    if decoder_type == "panel_invariant" and "decoder_gene_names" not in params:
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
    params = model_cfg.get("params", {})
    params.setdefault("n_genes", adatas[0].n_vars)


def _main_multi_sample(cfg) -> None:
    """Multi-sample training entry point (2026-07-16), called from main()
    when cfg.data.sample_ids is set. Mirrors main()'s single-sample flow
    (build model -> train on fresh masking draws -> save -> eval on one
    held-out draw) but over MultiSampleMaskedContextQueryDataset instead
    of MaskedContextQueryDataset — see that class's own docstring for why
    samples are kept spatially separate rather than pooled.

    UPDATED 2026-07-17: images/Novae/STPath ARE now supported (see
    load_multi_sample_data's own updated docstring for the real gap this
    closes and the STPath organ/tech caveat that still applies)."""
    samples, adatas = load_multi_sample_data(cfg)
    gene_names = adatas[0].var_names.tolist()  # shared panel, same order across samples (load_multi_sample's guarantee)
    augment = cfg.training.get("augment_coords", False)
    # 2026-07-17: RandomFourierFeatures real-scale bug fix — see
    # inject_coord_scale's own docstring. Derived from the FIRST sample
    # only (same convention as gene_names above) — every sample fed
    # through this one model instance should share a comparable physical
    # pixel resolution for this single fixed coord_scale to make sense
    # (same "confirm sample_ids share a platform before pooling"
    # assumption load_multi_sample's own docstring already documents).
    coord_scale = float(samples[0][0][:, :2].std())
    # context_gene_features/context_novae_features (2026-07-17): a
    # per-CONFIG decision (context_encoder_type/gene_encoder_type), not
    # per-sample data — every sample in `samples` has them set (or all
    # None), see load_multi_sample_data's own dispatch loop, so any one
    # sample's shape is representative for injecting *_novae_dim below.
    context_encoder_type = cfg.model.get("params", {}).get("context_encoder_type", "builtin")
    context_gene_features, context_novae_features = samples[0][6], samples[0][7]

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    inject_multi_sample_n_genes(model_cfg, adatas)
    inject_organ_tech_vocab(model_cfg, adatas)
    inject_coord_scale(model_cfg, coord_scale)
    inject_decoder_gene_names(model_cfg, adatas[0])
    inject_stpath_gene_names(model_cfg, adatas[0])
    if context_gene_features is not None:
        inject_novae_dim(model_cfg, context_gene_features.shape[1])
    if context_novae_features is not None:
        # storm_lite reuses novae_dim's param name; STPath's residual uses
        # the distinct stpath_novae_dim — same dispatch as main()'s own
        # single-sample injection block.
        if context_encoder_type == "storm_lite":
            inject_novae_dim(model_cfg, context_novae_features.shape[1])
        else:
            inject_stpath_novae_dim(model_cfg, context_novae_features.shape[1])
    model = build_model(model_cfg)
    # unresolved copy for checkpointing — same reasoning as main()'s own
    # unresolved_model_cfg (keeps ${oc.env:...} interpolations literal)
    unresolved_model_cfg = OmegaConf.to_container(cfg.model, resolve=False)
    inject_multi_sample_n_genes(unresolved_model_cfg, adatas)
    inject_organ_tech_vocab(unresolved_model_cfg, adatas)
    inject_coord_scale(unresolved_model_cfg, coord_scale)
    inject_decoder_gene_names(unresolved_model_cfg, adatas[0])
    inject_stpath_gene_names(unresolved_model_cfg, adatas[0])
    if context_gene_features is not None:
        inject_novae_dim(unresolved_model_cfg, context_gene_features.shape[1])
    if context_novae_features is not None:
        if context_encoder_type == "storm_lite":
            inject_novae_dim(unresolved_model_cfg, context_novae_features.shape[1])
        else:
            inject_stpath_novae_dim(unresolved_model_cfg, context_novae_features.shape[1])

    checkpoint_dir = cfg.training.get("checkpoint_dir", f"results/checkpoints/{cfg.experiment_name}")
    if list(model.parameters()):
        dataset = MultiSampleMaskedContextQueryDataset(
            samples, cfg.masking, n_items=cfg.training.epochs, base_seed=cfg.training.seed,
            augment=augment,
        )
        dataloader = make_dataloader(dataset, cfg)
        # PeriodicCheckpointCallback (2026-07-17): opt-in via
        # training.checkpoint_every_n_steps, unset by default — see that
        # class's own docstring.
        callbacks = []
        checkpoint_every_n_steps = cfg.training.get("checkpoint_every_n_steps")
        if checkpoint_every_n_steps:
            callbacks.append(PeriodicCheckpointCallback(
                unresolved_model_cfg, gene_names, checkpoint_dir,
                save_every_n_steps=checkpoint_every_n_steps,
            ))
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="auto",
            log_every_n_steps=cfg.training.log_every_n_steps,
            enable_checkpointing=False,
            logger=False,
            callbacks=callbacks,
        )
        trainer.fit(model, dataloader)
        saved_path = save_trained_model(model, unresolved_model_cfg, gene_names, checkpoint_dir)
        if saved_path is not None:
            print(f"Saved trained model (weights + config + gene names) to {saved_path.parent}")

    eval_item = MultiSampleMaskedContextQueryDataset(
        samples, cfg.masking, n_items=1, base_seed=cfg.training.seed + cfg.training.epochs + 1,
        augment=augment,
    )[0]
    context, query = eval_item["context"], eval_item["query"]
    target_expression = eval_item["target_expression"].numpy()

    model.eval()
    with torch.no_grad():
        output = model.sample(context, query)
    pred = output["expression"].detach().cpu().numpy()
    pcc = ev.pearson_per_gene(pred, target_expression)
    print(f"mean PCC: {np.nanmean(pcc):.4f}")
    print(f"RMSE: {ev.rmse(pred, target_expression):.4f}")


def main(cfg_path: str, overrides: list[str] | None = None):
    cfg = OmegaConf.load(cfg_path)
    if overrides:
        # dotlist overrides, e.g. ["training.epochs=2"] — smoke-testing a
        # config without editing the file itself (2026-07-15, checking all
        # 18 task #19 configs actually run before committing to full-length
        # training on each)
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
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

    # gene_encoder_type="novae" (2026-07-15) / stpath_new_gene_encoder_type=
    # "novae" (2026-07-16, Route B residual): precompute once, row-aligned
    # to the (possibly image-QC-filtered) adata _load_data already
    # returned — must happen before build_model, since *_novae_dim needs
    # to be injected into model_cfg first, same ordering as
    # inject_stpath_gene_names below. The two are mutually exclusive in
    # practice (a config is "builtin" or "stpath", never both) but use
    # separate variables/injectors regardless — see
    # _build_masked_item's context_novae_features docstring for why they
    # can't share a mechanism.
    model_params = cfg.model.get("params", {})
    context_encoder_type = model_params.get("context_encoder_type", "builtin")
    context_gene_features = None
    context_novae_features = None
    if context_encoder_type == "builtin" and model_params.get("gene_encoder_type") == "novae":
        # REPLACES context["expression"] — SpatialContextEncoder's own switch
        context_gene_features = get_novae_features(cfg, adata)
    elif (context_encoder_type == "stpath"
          and model_params.get("stpath_new_gene_encoder_type") in ("novae", "both")):
        # ADDITIVE context["novae_features"] — STPath's Route-B residual
        context_novae_features = get_novae_features(cfg, adata)
    elif (context_encoder_type == "storm_lite"
          and model_params.get("gene_encoder_type") in ("novae", "both")):
        # ADDITIVE context["novae_features"] too — StormLiteContextEncoder
        # reuses the gene_encoder_type param name but, like STPath, needs
        # this as a separate channel from raw context["expression"], never
        # a replacement (its "both" mode needs BOTH simultaneously) — see
        # storm_lite_encoder.py's own forward()/_encode_gene.
        context_novae_features = get_novae_features(cfg, adata)

    # 2026-07-17: RandomFourierFeatures real-scale bug fix — see
    # inject_coord_scale's own docstring
    coord_scale = float(coords3d[:, :2].std())

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    inject_stpath_gene_names(model_cfg, adata)
    inject_decoder_gene_names(model_cfg, adata)
    inject_coord_scale(model_cfg, coord_scale)
    if context_gene_features is not None:
        inject_novae_dim(model_cfg, context_gene_features.shape[1])
    if context_novae_features is not None:
        # storm_lite reuses novae_dim's param name (inject_novae_dim);
        # STPath's residual uses the distinct stpath_novae_dim
        # (inject_stpath_novae_dim) — see each function's own docstring.
        if context_encoder_type == "storm_lite":
            inject_novae_dim(model_cfg, context_novae_features.shape[1])
        else:
            inject_stpath_novae_dim(model_cfg, context_novae_features.shape[1])
    model = build_model(model_cfg)
    # UNRESOLVED copy, saved (not model_cfg above) so a STPath config's
    # ${oc.env:STPATH_GENE_VOC_PATH}/${oc.env:STPATH_MODEL_WEIGHT_PATH}
    # interpolations stay literal in the checkpoint rather than getting
    # baked in as THIS machine's resolved path — see load_trained_model's
    # docstring for the real cross-machine bug this fixes (2026-07-15).
    # novae_dim, unlike the STPath paths, is a plain int with nothing
    # machine-specific to resolve — injecting it here too just keeps the
    # saved config self-consistent with model_cfg above.
    unresolved_model_cfg = OmegaConf.to_container(cfg.model, resolve=False)
    inject_stpath_gene_names(unresolved_model_cfg, adata)
    inject_decoder_gene_names(unresolved_model_cfg, adata)
    inject_coord_scale(unresolved_model_cfg, coord_scale)
    if context_gene_features is not None:
        inject_novae_dim(unresolved_model_cfg, context_gene_features.shape[1])
    if context_novae_features is not None:
        if context_encoder_type == "storm_lite":
            inject_novae_dim(unresolved_model_cfg, context_novae_features.shape[1])
        else:
            inject_stpath_novae_dim(unresolved_model_cfg, context_novae_features.shape[1])

    # Train (skipped entirely for parameter-free baselines like interp_baseline) --
    augment = cfg.training.get("augment_coords", False)
    # organ/tech (2026-07-17, real bug fix — see
    # src/evaluation/run_comparison.py's _build_shared_eval for the same
    # fix and full reasoning): these were NEVER populated for the
    # single-sample path, silently leaving context["tech"]/query["tech"]
    # None for every training step.
    organ = str(adata.obs["organ"].iloc[0]) if "organ" in adata.obs else None
    tech = str(adata.obs["tech"].iloc[0]) if "tech" in adata.obs else None
    if list(model.parameters()):
        dataset = MaskedContextQueryDataset(
            coords3d, expr, slice_ids, cfg.masking,
            n_items=cfg.training.epochs, base_seed=cfg.training.seed, images=images,
            context_gene_features=context_gene_features,
            context_novae_features=context_novae_features,
            organ=organ, tech=tech,
            augment=augment,
        )
        dataloader = make_dataloader(dataset, cfg)
        # .get() with the same default every real config's own YAML comment
        # documents, not a bare attribute access — configs that don't
        # declare checkpoint_dir (e.g. tests/test_run_comparison.py's
        # synthetic configs) must still work, not crash on a missing key
        checkpoint_dir = cfg.training.get("checkpoint_dir", f"results/checkpoints/{cfg.experiment_name}")
        # PeriodicCheckpointCallback (2026-07-17): opt-in via
        # training.checkpoint_every_n_steps, unset by default — see that
        # class's own docstring.
        callbacks = []
        checkpoint_every_n_steps = cfg.training.get("checkpoint_every_n_steps")
        if checkpoint_every_n_steps:
            callbacks.append(PeriodicCheckpointCallback(
                unresolved_model_cfg, adata.var_names.tolist(), checkpoint_dir,
                save_every_n_steps=checkpoint_every_n_steps,
            ))
        trainer = pl.Trainer(
            max_epochs=1,  # one pass over `n_items` fresh masking draws == old epoch count
            accelerator="auto",
            log_every_n_steps=cfg.training.log_every_n_steps,
            enable_checkpointing=False,
            logger=False,
            callbacks=callbacks,
        )
        trainer.fit(model, dataloader)
        saved_path = save_trained_model(model, unresolved_model_cfg, adata.var_names.tolist(), checkpoint_dir)
        if saved_path is not None:
            print(f"Saved trained model (weights + config + gene names) to {saved_path.parent}")

    # Evaluate on a held-out masking draw not seen during training -----------
    eval_item = MaskedContextQueryDataset(
        coords3d, expr, slice_ids, cfg.masking,
        n_items=1, base_seed=cfg.training.seed + cfg.training.epochs + 1, images=images,
        context_gene_features=context_gene_features,
        context_novae_features=context_novae_features,
        organ=organ, tech=tech,
        augment=augment,
    )[0]
    context, query = eval_item["context"], eval_item["query"]
    target_expression = eval_item["target_expression"].numpy()

    model.eval()
    with torch.no_grad():
        output = model.sample(context, query)
    pred = output["expression"].detach().cpu().numpy()
    pcc = ev.pearson_per_gene(pred, target_expression)
    print(f"mean PCC: {np.nanmean(pcc):.4f}")
    print(f"RMSE: {ev.rmse(pred, target_expression):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--override", nargs="*", default=[],
                         help="dotlist config overrides, e.g. --override training.epochs=2")
    args = parser.parse_args()
    main(args.config, args.override)
