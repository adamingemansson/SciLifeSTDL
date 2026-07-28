"""Save/load trainable model state, config, and gene names.

Mirrors the discipline already established (and, this session, real-bug-
fixed) in src/training/train.py::save_trained_model/load_trained_model:
only trainable parameters and non-frozen buffers are saved (frozen
backbones like GigaPath/STPath/scFoundation are reloaded fresh from their
own real pretrained source every time the architecture is rebuilt, not
re-saved here — a STPath-conditioned model's full state_dict is gigabytes
of frozen weights vs a few tens of MB of trainable ones). Config and gene
names are saved even when there are literally zero trainable weights (the
real bug fixed 2026-07-24 in the original codebase: skipping metadata
entirely whenever there was nothing trainable made a fully-frozen
checkpoint impossible to reconstruct later).

Checkpoint HISTORY (added 2026-07-25): save_checkpoint() always used to
overwrite the same fixed filenames in checkpoint_dir's root — only the
single most recent checkpoint ever existed, so a diverged/corrupted run
had no earlier state to roll back to. Every root save now additionally
hard-links (not copies — os.link, falling back to a real copy only if the
filesystem can't hard-link, e.g. across devices) its files into
checkpoint_dir/history/step_XXXXXXXX/, then prunes old history entries
beyond training.checkpoint_keep_last. Hard-linking costs ~zero extra disk
at save time (it's a second directory entry pointing at the same data
blocks) — the old data only becomes exclusively "owned" by the history
copy once the root path is later replaced by a newer save, at which point
it is disk you were always going to spend on that many kept snapshots
regardless of hardlink-vs-copy, hardlinking just avoids doing a second
physical write to get there. With training servers running tight on disk
(50 GB total was flagged as the real budget for this project's training
server), keep training.checkpoint_keep_last small (default 2) and watch
the printed sizes below — trainable-only weights are the cheap part
(tens of MB typically), but Architecture 3 Stage A's autoencoder can be
much larger on a wide gene panel (see README's disk budget note).

Optimizer + RNG state (added 2026-07-27, GPT-audit-flagged fix #6):
save_checkpoint's optional optimizer=/rng= arguments additionally save
AdamW's momentum/variance buffers and every RNG stream (python/numpy/
torch/CUDA + the training loop's own sample-draw random.Random instance)
to optimizer_rng_state.pt, restorable via load_optimizer_and_rng_state().
Without this, a "resumed" run kept the trained weights but reset
optimizer momentum to zero and re-seeded every RNG from scratch — a warm
restart with different optimization dynamics, not a real continuation.
"""
from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_CHECKPOINT_FILENAMES = (
    "trainable_weights.pt", "model_config.json", "gene_names.json", "training_state.json",
    "optimizer_rng_state.pt",
)


def _is_frozen_backbone_module(module: nn.Module) -> bool:
    """A module counts as a frozen backbone if it has parameters and NONE
    of them require grad — covers GigapathPatchEncoder's lazily-loaded
    tile_encoder, STPathContextEncoder's self.model (when pretrained=True),
    scFoundation (never has trainable params by construction), and Stage
    A's autoencoder when Stage B runs with finetune_autoencoder=False."""
    params = list(module.parameters(recurse=False))
    return bool(params) and all(not p.requires_grad for p in params)


def save_trainable_state(model: nn.Module, checkpoint_dir: str | Path) -> Path | None:
    """Save trainable parameters and non-frozen buffers only. Returns the
    weights file path, or None if there was nothing trainable to save
    (e.g. Architecture 4 with its STPath backbone fully frozen and no
    scFoundation residual enabled — an edge case, but handled the same
    way the original codebase's zero-trainable-parameter bug taught us
    to: absence of a weights file is a valid, real state, not an error)."""
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
    path = Path(checkpoint_dir) / "trainable_weights.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)
    return path


def _human_size(n_bytes: int) -> str:
    size = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}GB"


def _history_dir(checkpoint_dir: str | Path) -> Path:
    return Path(checkpoint_dir) / "history"


def _snapshot_into_history(checkpoint_dir: str | Path, step: int) -> Path:
    """Hard-link (falling back to a real copy across filesystems) the just-
    written root checkpoint files into a step-numbered history snapshot.
    Root files must already exist by the time this is called."""
    out_dir = Path(checkpoint_dir)
    snapshot_dir = _history_dir(checkpoint_dir) / f"step_{int(step):08d}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for name in _CHECKPOINT_FILENAMES:
        src = out_dir / name
        if not src.is_file():
            continue
        dst = snapshot_dir / name
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    return snapshot_dir


def _prune_history(checkpoint_dir: str | Path, keep_last: int) -> None:
    history_dir = _history_dir(checkpoint_dir)
    if not history_dir.is_dir() or keep_last <= 0:
        return
    steps = sorted(list_checkpoint_history(checkpoint_dir))
    for stale_step in steps[:-keep_last] if keep_last > 0 else []:
        shutil.rmtree(history_dir / f"step_{stale_step:08d}", ignore_errors=True)


def list_checkpoint_history(checkpoint_dir: str | Path) -> list[int]:
    """Steps with a preserved history snapshot, ascending."""
    history_dir = _history_dir(checkpoint_dir)
    if not history_dir.is_dir():
        return []
    steps = []
    for entry in history_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("step_"):
            try:
                steps.append(int(entry.name[len("step_"):]))
            except ValueError:
                continue
    return sorted(steps)


def rollback_checkpoint(checkpoint_dir: str | Path, step: int) -> None:
    """Overwrite the root ("latest", what main()'s resume logic reads)
    checkpoint with a preserved history snapshot — use this when a run has
    diverged or corrupted state after a later save and you want the next
    resume to pick up from a known-good earlier step instead. Raises with
    the actual available steps if the requested one isn't present (a typo'd
    step should fail loudly, not silently no-op)."""
    available = list_checkpoint_history(checkpoint_dir)
    if step not in available:
        raise ValueError(
            f"no history snapshot for step {step} in {checkpoint_dir}/history — "
            f"available steps: {available}"
        )
    snapshot_dir = _history_dir(checkpoint_dir) / f"step_{int(step):08d}"
    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Root previously may have had a trainable_weights.pt that this snapshot
    # doesn't (a zero-trainable-parameter checkpoint) — remove it first so a
    # stale weights file from a *later* step never lingers after rollback.
    for name in _CHECKPOINT_FILENAMES:
        stale = out_dir / name
        if stale.is_file() and not (snapshot_dir / name).is_file():
            stale.unlink()
    for name in _CHECKPOINT_FILENAMES:
        src = snapshot_dir / name
        if not src.is_file():
            continue
        tmp_path = out_dir / f"{name}.tmp{os.getpid()}"
        shutil.copy2(src, tmp_path)
        os.replace(tmp_path, out_dir / name)
    print(f"rolled back {out_dir} to step {step} (from history snapshot {snapshot_dir})")


def _rng_state_blob(rng: random.Random | None) -> dict:
    """Every source of randomness that affects a training run's future
    trajectory: the module-level python `random` (seeded once by
    seed_everything but mutated by any incidental use), numpy's global
    RNG, torch's CPU RNG, torch's per-GPU CUDA RNGs (if any), and the
    training loop's OWN independent `random.Random(seed)` instance (each
    training script keeps its sample-draw RNG separate from the global
    one specifically so it isn't perturbed by unrelated random calls
    elsewhere — see e.g. train_local_neighborhood.py's `rng =
    random.Random(...)`)."""
    blob = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.get_rng_state(),
    }
    if rng is not None:
        blob["loop_random"] = rng.getstate()
    if torch.cuda.is_available():
        blob["torch_cuda_random"] = torch.cuda.get_rng_state_all()
    return blob


def _restore_rng_state(blob: dict, rng: random.Random | None) -> None:
    if "python_random" in blob:
        random.setstate(blob["python_random"])
    if "numpy_random" in blob:
        np.random.set_state(blob["numpy_random"])
    if "torch_random" in blob:
        torch.set_rng_state(blob["torch_random"])
    if rng is not None and "loop_random" in blob:
        rng.setstate(blob["loop_random"])
    if torch.cuda.is_available() and "torch_cuda_random" in blob:
        torch.cuda.set_rng_state_all(blob["torch_cuda_random"])


def save_checkpoint(
    model: nn.Module, model_config: dict, gene_names: list[str], checkpoint_dir: str | Path,
    step: int, extra_metadata: dict | None = None, keep_last: int = 2,
    optimizer: torch.optim.Optimizer | None = None, rng: random.Random | None = None,
) -> None:
    weights_path = save_trainable_state(model, checkpoint_dir)
    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in [
        ("model_config.json", model_config), ("gene_names.json", list(gene_names)),
        ("training_state.json", {"step": int(step), **(extra_metadata or {})}),
    ]:
        final_path = out_dir / name
        tmp_path = out_dir / f"{name}.tmp{os.getpid()}"
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, final_path)

    # 2026-07-27 (GPT-audit-flagged, fix #6): checkpoints used to save
    # ONLY model weights -- no AdamW optimizer state (momentum/variance
    # buffers reset to zero on resume) and no RNG state (python/numpy/
    # torch/CUDA, plus each training loop's own sample-draw RNG) -- a
    # "resumed" run was actually a warm restart with different
    # optimization dynamics, not a deterministic continuation. Optional:
    # a caller that truly doesn't care (e.g. a one-off inference/eval
    # rebuild) can simply not pass optimizer, and no file is written.
    if optimizer is not None:
        opt_path = out_dir / "optimizer_rng_state.pt"
        opt_tmp_path = out_dir / f"optimizer_rng_state.pt.tmp{os.getpid()}"
        torch.save({"optimizer": optimizer.state_dict(), **_rng_state_blob(rng)}, opt_tmp_path)
        os.replace(opt_tmp_path, opt_path)

    weights_size = weights_path.stat().st_size if weights_path is not None else 0
    if keep_last > 0:
        _snapshot_into_history(checkpoint_dir, step)
        _prune_history(checkpoint_dir, keep_last)
    history_steps = list_checkpoint_history(checkpoint_dir)
    history_size = sum(
        f.stat().st_size for f in _history_dir(checkpoint_dir).rglob("*") if f.is_file()
    ) if history_steps else 0
    print(
        f"checkpoint saved to {out_dir} (step {step}"
        f"{', weights + config + gene names' if weights_path is not None else ', config + gene names only (no trainable weights)'}"
        f", weights={_human_size(weights_size)}"
        f", history kept={history_steps} total_history_size={_human_size(history_size)})"
    )


def load_trainable_state(model: nn.Module, checkpoint_dir: str | Path) -> None:
    """Load a previously-saved trainable state onto a freshly-constructed,
    architecturally-identical model. No-op if the checkpoint genuinely has
    no trainable_weights.pt (a fully-frozen model — verified by checking
    the fresh model ALSO has zero trainable parameters, not silently
    assumed).

    Uses explicit RuntimeError, not `assert` (2026-07-28, real bug fixed
    -- 6th Codex re-audit of gen3_multiscale commit 06f5cce, applied here
    too to keep the two copies in sync): `assert` statements are compiled
    out entirely under `python -O`/`PYTHONOPTIMIZE`, which would silently
    turn this fail-closed config-mismatch/incomplete-checkpoint check
    into a no-op in that mode."""
    in_dir = Path(checkpoint_dir)
    weights_path = in_dir / "trainable_weights.pt"
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    if weights_path.is_file():
        state = torch.load(weights_path, map_location="cpu")
        missing_trainable = trainable_names - set(state.keys())
        if missing_trainable:
            raise RuntimeError(
                f"saved weights at {in_dir} are missing trainable parameters this "
                f"model architecture expects: {missing_trainable} (config mismatch?)"
            )
        model.load_state_dict(state, strict=False)
    else:
        if trainable_names:
            raise RuntimeError(
                f"{weights_path} is missing but this model architecture has trainable "
                f"parameters {trainable_names}; the checkpoint at {in_dir} looks incomplete"
            )


def load_training_state(checkpoint_dir: str | Path) -> dict:
    path = Path(checkpoint_dir) / "training_state.json"
    if not path.is_file():
        return {"step": 0}
    return json.loads(path.read_text())


def verify_gene_names(checkpoint_dir: str | Path, gene_names: list[str]) -> None:
    """GPT-audit-flagged bug (2026-07-27, second-pass re-audit): resuming a
    training run rebuilt the model from the CURRENT run's freshly-derived
    gene panel and loaded weights onto it with no check that this panel
    matches the one the checkpoint was actually trained on. trainable_
    weights.pt's tensors are positional, not gene-name-keyed -- a same-
    shaped but reordered/different gene panel (e.g. a changed
    sample_selection or QC setting between launches) would load without
    error and silently produce a model whose weights and gene identities
    no longer correspond, corrupting the run without any visible symptom
    until metrics look wrong. Call this once at resume time, right after
    confirming a checkpoint exists, before load_trainable_state -- mirrors
    the exact same ordered-identity check already used for Stage A/Stage B
    (train_arch3_stage_b.py::_load_stage_a)."""
    path = Path(checkpoint_dir) / "gene_names.json"
    if not path.is_file():
        raise ValueError(
            f"checkpoint at {checkpoint_dir} has no gene_names.json -- cannot verify its gene "
            "panel matches this run's before loading weights onto it. This looks like an "
            "incomplete checkpoint."
        )
    saved_gene_names = json.loads(path.read_text())
    if list(saved_gene_names) != list(gene_names):
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(saved_gene_names, gene_names)) if a != b),
            min(len(saved_gene_names), len(gene_names)),
        )
        raise ValueError(
            f"checkpoint at {checkpoint_dir} was trained on a different gene panel than this "
            f"run just resolved (first mismatch at index {first_diff}: "
            f"{saved_gene_names[first_diff] if first_diff < len(saved_gene_names) else '<end>'!r} vs "
            f"{gene_names[first_diff] if first_diff < len(gene_names) else '<end>'!r}) -- refusing "
            "to load these weights onto a mismatched gene vocabulary. If this is a deliberate "
            "re-derivation (e.g. changed sample_selection), resume from a fresh checkpoint_dir."
        )


def load_optimizer_and_rng_state(
    optimizer: torch.optim.Optimizer, checkpoint_dir: str | Path, rng: random.Random | None = None,
) -> bool:
    """Restore optimizer momentum/variance buffers and every RNG stream
    saved by save_checkpoint's optimizer=/rng= arguments. Returns False
    (a no-op) for a checkpoint saved before this fix existed, or any
    checkpoint saved without an optimizer (e.g. a fully-frozen model) —
    callers should treat that as "fresh optimizer state, resume anyway"
    rather than an error, since older/lighter checkpoints are still
    otherwise valid to resume from."""
    path = Path(checkpoint_dir) / "optimizer_rng_state.pt"
    if not path.is_file():
        return False
    # weights_only=False: this file isn't a bare state_dict, it also
    # carries numpy's RNG state (a plain ndarray inside a tuple) which
    # torch's default weights_only unpickler (PyTorch >=2.6) refuses to
    # load. Always our own locally-written file, never untrusted input.
    blob = torch.load(path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(blob["optimizer"])
    _restore_rng_state(blob, rng)
    return True
