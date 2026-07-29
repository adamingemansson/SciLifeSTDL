"""2026-07-28: copied VERBATIM from gen2_architectures/training/checkpoint.py
at commit fd23737 into gen3_multiscale/ -- see data/hest1k_catalog.py's
identical copy-provenance note for why. Includes verify_gene_names
(fail-closed gene-identity check on resume), optimizer/RNG state save-
restore, and history/rollback -- all fixed/added during this session's
audit passes. Do not let this drift from gen2_architectures' copy
without a deliberate reason.

Save/load trainable model state, config, and gene names.

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
had no earlier state to roll back to.

Optimizer + RNG state (added 2026-07-27, GPT-audit-flagged fix #6):
save_checkpoint's optional optimizer=/rng= arguments additionally save
AdamW's momentum/variance buffers and every RNG stream (python/numpy/
torch/CUDA + the training loop's own sample-draw random.Random instance)
to optimizer_rng_state.pt, restorable via load_optimizer_and_rng_state().
Without this, a "resumed" run kept the trained weights but reset
optimizer momentum to zero and re-seeded every RNG from scratch — a warm
restart with different optimization dynamics, not a real continuation.

TRANSACTIONAL, UNIQUELY-NAMED BUNDLES (rewritten in response to Codex's
re-audit of commit 90f853e, launch blocker #1): the immediately prior
version named history bundles by STEP ALONE
(`history/step_XXXXXXXX/`) and (a) deleted an existing same-named bundle
with `shutil.rmtree` before replacing it, and (b) pruned old bundles
BEFORE the `latest_step.json` pointer was updated to the new one. Both
are real crash-safety holes: (a) meant re-saving the same step (which
`train.py`'s own loop can genuinely do -- see the "duplicate final
save" fix below) could leave `latest_step.json` pointing at a step whose
bundle a crash had just deleted and not yet replaced; (b) meant a crash
between pruning and the pointer update could leave the pointer
referencing a bundle that pruning had just deleted. Every bundle
directory name now embeds a UNIQUE id
(`step_XXXXXXXX__<time_ns>_<pid>_<seq>`) generated fresh per call, so
`save_checkpoint` NEVER deletes or overwrites an existing directory --
two saves at the same step simply produce two distinct bundles, and the
pointer ends up referencing whichever was written (and pointed-to) most
recently, with no destructive step in between. The pointer
(`latest_bundle.json`) itself now records `step`, `bundle_dir`, AND
`manifest_sha256` (the bundle's own manifest.json content hash) --
Codex's requested `CheckpointIdentity` (`resolve_checkpoint_identity`
below) is resolved from THIS pointer, never from `checkpoint_dir`'s root
convenience-mirror files, closing the "conditioner identity hashes the
wrong file" gap the same re-audit flagged (train.py::
maybe_load_pretrained_conditioner_for_architecture4 and
fit_architecture4_residual_basis.py both used to `file_sha256` the root
mirror directly). Pruning now runs strictly AFTER the pointer update,
and is passed the just-written bundle's name to protect explicitly even
if history ordering is ever ambiguous. Every file written into a bundle
(and the bundle's own directory entry) is now best-effort fsync'd before
the bundle is considered complete, per Codex's "write and fsync the
complete bundle" instruction.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_CHECKPOINT_FILENAMES = (
    "trainable_weights.pt", "model_config.json", "gene_names.json", "training_state.json",
    "optimizer_rng_state.pt",
)

_bundle_id_sequence = itertools.count()


def _new_bundle_id() -> str:
    """A unique, lexicographically-increasing-over-time id -- the
    `time_ns` prefix means bundles for the SAME step naturally sort
    chronologically (most recent last), and the process id + monotonic
    in-process sequence number make collisions impossible even across
    calls that land in the same nanosecond."""
    return f"{time.time_ns():020d}_{os.getpid()}_{next(_bundle_id_sequence):06d}"


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


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fsync_path(path: Path) -> None:
    """Best-effort durability -- fsync a file OR a directory's own inode
    (POSIX permits `os.fsync` on a directory fd opened read-only; this is
    how a directory entry's creation/rename is made durable, not just the
    file content). Never raises: some filesystems/platforms (e.g.
    Windows, some network filesystems, or overlay filesystems as used in
    a container) don't support this, and a checkpoint save must not fail
    outright just because a best-effort durability step wasn't
    available -- the atomic-rename discipline elsewhere in this file is
    what actually prevents corruption; fsync only shortens the window
    where a page-cache-only write could still be lost to a real power
    loss."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _bundle_dir_name_prefix(step: int) -> str:
    return f"step_{int(step):08d}__"


def list_checkpoint_bundles(checkpoint_dir: str | Path) -> list[tuple[int, str]]:
    """`(step, bundle_dir_name)` for every preserved history bundle,
    ascending -- multiple bundles can legitimately exist for the SAME
    step (a repeated/duplicate save at that step is no longer
    destructive; see this module's docstring), so this returns every one
    of them, not a deduplicated step list. Sorted by directory name,
    which sorts chronologically within a step (see `_new_bundle_id`) and
    by step numerically across steps (fixed-width zero-padded prefix)."""
    history_dir = _history_dir(checkpoint_dir)
    if not history_dir.is_dir():
        return []
    bundles = []
    for entry in history_dir.iterdir():
        if not entry.is_dir() or not entry.name.startswith("step_") or entry.name.startswith(".step_"):
            continue
        prefix, _, _bundle_id = entry.name.partition("__")
        try:
            step = int(prefix[len("step_"):])
        except ValueError:
            continue
        bundles.append((step, entry.name))
    bundles.sort(key=lambda pair: (pair[0], pair[1]))
    return bundles


def list_checkpoint_history(checkpoint_dir: str | Path) -> list[int]:
    """Distinct steps with at least one preserved history bundle,
    ascending -- a convenience view over `list_checkpoint_bundles` for
    callers that only care which steps exist, not how many bundles each
    has."""
    return sorted({step for step, _ in list_checkpoint_bundles(checkpoint_dir)})


def _prune_history(checkpoint_dir: str | Path, keep_last: int, *, protect_bundle_dir: str | None = None) -> None:
    """Prune all but the `keep_last` most-recently-written bundles.
    Called strictly AFTER `latest_bundle.json` has been updated to
    reference the newest bundle (never before -- see this module's
    docstring for why the old before-the-pointer ordering was a crash-
    safety hole), and `protect_bundle_dir` (the bundle the pointer now
    references) is skipped defensively even if it were somehow not
    already within the retained tail."""
    history_dir = _history_dir(checkpoint_dir)
    if not history_dir.is_dir() or keep_last <= 0:
        return
    bundles = list_checkpoint_bundles(checkpoint_dir)
    stale = bundles[:-keep_last] if keep_last > 0 else []
    for _step, bundle_name in stale:
        if protect_bundle_dir is not None and bundle_name == protect_bundle_dir:
            continue
        shutil.rmtree(history_dir / bundle_name, ignore_errors=True)


def _latest_bundle_for_step(checkpoint_dir: str | Path, step: int) -> str:
    matches = [name for s, name in list_checkpoint_bundles(checkpoint_dir) if s == step]
    if not matches:
        available = list_checkpoint_history(checkpoint_dir)
        raise ValueError(f"no history bundle for step {step} in {checkpoint_dir}/history — available steps: {available}")
    return matches[-1]  # lexicographically last == chronologically most recent, see _new_bundle_id


def rollback_checkpoint(checkpoint_dir: str | Path, step: int) -> None:
    """Overwrite the root ("latest", what resume logic reads) checkpoint
    with a preserved history bundle -- use this when a run has diverged
    or corrupted state after a later save and you want the next resume
    to pick up from a known-good earlier step instead. Raises with the
    actual available steps if the requested one isn't present (a typo'd
    step should fail loudly, not silently no-op). If more than one
    bundle exists for `step` (a repeated save at that step), rolls back
    to the most recently written one."""
    bundle_name = _latest_bundle_for_step(checkpoint_dir, step)
    bundle_dir = _history_dir(checkpoint_dir) / bundle_name
    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Root previously may have had a trainable_weights.pt that this bundle
    # doesn't (a zero-trainable-parameter checkpoint) — remove it first so a
    # stale weights file from a *later* step never lingers after rollback.
    for name in _CHECKPOINT_FILENAMES:
        stale = out_dir / name
        if stale.is_file() and not (bundle_dir / name).is_file():
            stale.unlink()
    for name in _CHECKPOINT_FILENAMES:
        src = bundle_dir / name
        if not src.is_file():
            continue
        tmp_path = out_dir / f"{name}.tmp{os.getpid()}"
        shutil.copy2(src, tmp_path)
        os.replace(tmp_path, out_dir / name)
    # Every real loader resolves through latest_bundle.json, not the root
    # files directly -- rolling back must move that pointer too, or the
    # next resume would silently ignore the rollback and keep resolving
    # to the newer (rolled-back-FROM) bundle.
    manifest_path = bundle_dir / "manifest.json"
    pointer = {
        "step": int(step), "bundle_dir": bundle_name,
        "manifest_sha256": _file_sha256(manifest_path) if manifest_path.is_file() else None,
    }
    pointer_tmp = out_dir / f".latest_bundle.json.tmp{os.getpid()}"
    pointer_tmp.write_text(json.dumps(pointer, indent=2))
    os.replace(pointer_tmp, out_dir / "latest_bundle.json")
    print(f"rolled back {out_dir} to step {step} (from history bundle {bundle_dir})")


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
    """Transactional checkpoint save into a uniquely-named, immutable
    bundle -- see this module's docstring for the full crash-safety
    rationale (Codex re-audit of commit 90f853e, launch blocker #1).
    Never deletes or overwrites an existing bundle: `step` alone does not
    name the bundle directory, a fresh unique id does, so two saves at
    the same step simply coexist as two bundles until pruning (which
    only ever removes bundles OLDER than the retained tail, and never
    the one the pointer currently references)."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_dir = _history_dir(checkpoint_dir)
    history_dir.mkdir(parents=True, exist_ok=True)

    bundle_name = f"{_bundle_dir_name_prefix(step)}{_new_bundle_id()}"
    bundle_dir = history_dir / bundle_name
    staging_dir = history_dir / f".{bundle_name}.staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    weights_path = save_trainable_state(model, staging_dir)
    if weights_path is not None:
        _fsync_path(weights_path)
    for name, payload in [
        ("model_config.json", model_config), ("gene_names.json", list(gene_names)),
        ("training_state.json", {"step": int(step), **(extra_metadata or {})}),
    ]:
        target = staging_dir / name
        target.write_text(json.dumps(payload, indent=2))
        _fsync_path(target)

    # 2026-07-27 (GPT-audit-flagged, fix #6): checkpoints used to save
    # ONLY model weights -- no AdamW optimizer state (momentum/variance
    # buffers reset to zero on resume) and no RNG state (python/numpy/
    # torch/CUDA, plus each training loop's own sample-draw RNG) -- a
    # "resumed" run was actually a warm restart with different
    # optimization dynamics, not a deterministic continuation. Optional:
    # a caller that truly doesn't care (e.g. a one-off inference/eval
    # rebuild) can simply not pass optimizer, and no file is written.
    if optimizer is not None:
        opt_path = staging_dir / "optimizer_rng_state.pt"
        torch.save({"optimizer": optimizer.state_dict(), **_rng_state_blob(rng)}, opt_path)
        _fsync_path(opt_path)

    weights_size = weights_path.stat().st_size if weights_path is not None else 0

    file_hashes = {
        name: _file_sha256(staging_dir / name) for name in _CHECKPOINT_FILENAMES if (staging_dir / name).is_file()
    }
    step_manifest = {
        "version": 2, "kind": "gen3_checkpoint_step_manifest", "step": int(step),
        "bundle_id": bundle_name, "files": file_hashes,
    }
    # Written LAST inside the staging directory -- a loader can treat its
    # presence (and its own hashes matching) as proof this bundle
    # finished writing completely.
    manifest_path = staging_dir / "manifest.json"
    manifest_path.write_text(json.dumps(step_manifest, indent=2, sort_keys=True))
    _fsync_path(manifest_path)
    manifest_sha256 = _file_sha256(manifest_path)

    # bundle_dir is GUARANTEED not to already exist (unique bundle_name) --
    # this is never a delete-then-replace, only ever a fresh create.
    os.replace(staging_dir, bundle_dir)
    _fsync_path(history_dir)  # durability of the new directory entry itself

    # Atomic pointer update, containing the bundle id, step, AND the
    # bundle's own manifest sha256 (Codex's requested CheckpointIdentity
    # payload) -- written BEFORE pruning, so pruning can never remove the
    # bundle the pointer now references.
    pointer = {"step": int(step), "bundle_dir": bundle_name, "manifest_sha256": manifest_sha256}
    pointer_tmp = checkpoint_dir / f".latest_bundle.json.tmp{os.getpid()}"
    pointer_tmp.write_text(json.dumps(pointer, indent=2))
    _fsync_path(pointer_tmp)
    os.replace(pointer_tmp, checkpoint_dir / "latest_bundle.json")
    _fsync_path(checkpoint_dir)

    # Refresh checkpoint_dir's ROOT files as a convenience mirror, always
    # sourced from the just-completed IMMUTABLE bundle -- real loaders
    # below never trust these root files directly; they resolve through
    # latest_bundle.json instead, so a crash during this refresh can no
    # longer corrupt what gets loaded.
    #
    # Codex re-audit of commit 90f853e, adversarial-test-confirmed real
    # bug: this used to try `os.link` (a HARDLINK) first, falling back to
    # `shutil.copy2` only on OSError. On any filesystem where hardlinking
    # actually succeeds (the common same-filesystem POSIX case), the
    # root mirror and the canonical bundle file are then the SAME inode
    # -- an in-place rewrite of the "mirror" (e.g. `Path.write_bytes`,
    # which opens 'wb' and truncates) silently corrupts the CANONICAL
    # bundle too, defeating the entire "root is only ever a disposable
    # convenience mirror" invariant this module's docstring promises.
    # Always a REAL, independent copy now -- never a hardlink.
    for name in _CHECKPOINT_FILENAMES:
        src = bundle_dir / name
        if not src.is_file():
            continue
        tmp_root = checkpoint_dir / f"{name}.tmp{os.getpid()}"
        shutil.copy2(src, tmp_root)
        os.replace(tmp_root, checkpoint_dir / name)

    # Pruning happens LAST of all, strictly after the pointer is durable,
    # and is told explicitly which bundle to never remove.
    if keep_last > 0:
        _prune_history(checkpoint_dir, keep_last, protect_bundle_dir=bundle_name)

    history_steps = list_checkpoint_history(checkpoint_dir)
    history_size = sum(
        f.stat().st_size for f in _history_dir(checkpoint_dir).rglob("*") if f.is_file()
    ) if history_steps else 0
    print(
        f"checkpoint saved to {checkpoint_dir} (step {step}, bundle {bundle_name}"
        f"{', weights + config + gene names' if weights_path is not None else ', config + gene names only (no trainable weights)'}"
        f", weights={_human_size(weights_size)}"
        f", history kept={history_steps} total_history_size={_human_size(history_size)})"
    )


def _resolve_checkpoint_source(checkpoint_dir: str | Path) -> Path:
    """Fail-closed transactional load resolution. If `checkpoint_dir/
    latest_bundle.json` exists, resolves to the bundle it names and
    verifies (a) the bundle's OWN manifest.json still matches the sha256
    the pointer recorded for it, and (b) every file the bundle's
    manifest itself records still matches its sha256 on disk right now --
    a crash between individual file writes (or the checkpoint being
    altered after being written) can therefore no longer silently
    produce a load built from files belonging to different steps or
    bundles; the whole bundle is rejected instead. Falls back to
    `checkpoint_dir` itself when there is no pointer -- a `best/` bundle
    (verified separately, by `train.py::verify_checkpoint_bundle_identity`)
    or an arbitrary non-transactional checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)
    pointer_path = checkpoint_dir / "latest_bundle.json"
    if not pointer_path.is_file():
        return checkpoint_dir
    pointer = json.loads(pointer_path.read_text())
    step = int(pointer["step"])
    bundle_name = pointer["bundle_dir"]
    expected_manifest_sha256 = pointer.get("manifest_sha256")
    bundle_dir = _history_dir(checkpoint_dir) / bundle_name
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"checkpoint at {checkpoint_dir} points (via latest_bundle.json) at bundle {bundle_name!r} "
            f"(step {step}) but its manifest is missing at {manifest_path} -- refusing to load a "
            "possibly-partial or corrupted checkpoint"
        )
    if expected_manifest_sha256 is not None and _file_sha256(manifest_path) != expected_manifest_sha256:
        raise RuntimeError(
            f"checkpoint at {checkpoint_dir}: bundle {bundle_name!r}'s manifest.json does not match "
            "the sha256 recorded in latest_bundle.json -- corrupted or tampered checkpoint, refusing to load"
        )
    step_manifest = json.loads(manifest_path.read_text())
    manifest_files = step_manifest.get("files") or {}
    # Codex re-audit of commit 90f853e, adversarial coverage for launch
    # blocker #1/#2: a per-file hash loop over ONLY the manifest's own
    # "files" dict cannot catch a file that physically exists in the
    # bundle but was OMITTED from that dict (whether by a bug or by
    # tampering) -- such a file would never be hash-checked at all, so a
    # tampered-but-structurally-valid replacement (e.g. different
    # trainable_weights.pt bytes) would load silently. Cross-check the
    # manifest against the bundle's REAL directory contents first.
    actual_files = {p.name for p in bundle_dir.iterdir() if p.is_file() and p.name != "manifest.json"}
    unlisted = actual_files - set(manifest_files.keys())
    if unlisted:
        raise RuntimeError(
            f"checkpoint bundle {bundle_name!r} (step {step}) contains file(s) {sorted(unlisted)} that "
            "are not listed in its own manifest.json -- refusing to load a bundle whose manifest does "
            "not fully account for its real contents"
        )
    for name, expected_hash in manifest_files.items():
        file_path = bundle_dir / name
        if not file_path.is_file():
            raise RuntimeError(
                f"checkpoint bundle {bundle_name!r} (step {step}) manifest references {name} but it is "
                f"missing from {bundle_dir} -- refusing to load a corrupted checkpoint"
            )
        if _file_sha256(file_path) != expected_hash:
            raise RuntimeError(
                f"checkpoint bundle {bundle_name!r} (step {step}) file {name} does not match the sha256 "
                "recorded in its own bundle manifest -- corrupted or partially-written checkpoint, "
                "refusing to load"
            )
    return bundle_dir


@dataclass(frozen=True)
class CheckpointIdentity:
    """The canonical identity of whatever `checkpoint_dir` currently
    resolves to -- Codex's re-audit of commit 90f853e, launch blocker #2:
    "Make checkpoint resolution return a canonical CheckpointIdentity...
    Use that identity everywhere: resume, Architecture 4 conditioner
    loading, basis fitting and evaluation. Never hash root convenience
    mirrors." `weights_sha256` is always computed from `resolved_dir`
    (the VERIFIED bundle, or `checkpoint_dir` itself for a non-
    transactional dir like `best/`), never from `checkpoint_dir`'s root
    mirror files directly."""
    resolved_dir: Path
    step: int | None
    bundle_dir: str | None
    manifest_sha256: str | None
    weights_sha256: str | None


def resolve_checkpoint_identity(checkpoint_dir: str | Path) -> CheckpointIdentity:
    checkpoint_dir = Path(checkpoint_dir)
    resolved_dir = _resolve_checkpoint_source(checkpoint_dir)
    pointer_path = checkpoint_dir / "latest_bundle.json"
    step = bundle_dir = manifest_sha256 = None
    if pointer_path.is_file():
        pointer = json.loads(pointer_path.read_text())
        step = int(pointer["step"])
        bundle_dir = pointer.get("bundle_dir")
        manifest_sha256 = pointer.get("manifest_sha256")
    weights_path = resolved_dir / "trainable_weights.pt"
    weights_sha256 = _file_sha256(weights_path) if weights_path.is_file() else None
    return CheckpointIdentity(
        resolved_dir=resolved_dir, step=step, bundle_dir=bundle_dir,
        manifest_sha256=manifest_sha256, weights_sha256=weights_sha256,
    )


def load_trainable_state(model: nn.Module, checkpoint_dir: str | Path) -> None:
    """Load a previously-saved trainable state onto a freshly-constructed,
    architecturally-identical model. No-op if the checkpoint genuinely has
    no trainable_weights.pt (a fully-frozen model — verified by checking
    the fresh model ALSO has zero trainable parameters, not silently
    assumed).

    Uses explicit RuntimeError, not `assert` (2026-07-28, real bug fixed
    -- 6th Codex re-audit of commit 06f5cce): `assert` statements are
    compiled out entirely under `python -O`/`PYTHONOPTIMIZE`, which would
    silently turn this fail-closed config-mismatch/incomplete-checkpoint
    check into a no-op in that mode -- a real gap for something whose
    whole job is to fail loudly rather than load a wrong or partial
    state. Same fix applied identically to gen2_architectures/training/
    checkpoint.py's copy of this function, per this file's own
    copy-provenance discipline. Resolves through `_resolve_checkpoint_source`
    first -- for a transactional checkpoint_dir this loads from the
    verified, immutable history bundle, never directly from the (merely
    a convenience mirror) root files."""
    in_dir = _resolve_checkpoint_source(checkpoint_dir)
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
    path = _resolve_checkpoint_source(checkpoint_dir) / "training_state.json"
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
    (train_arch3_stage_b.py::_load_stage_a). Resolves through
    `_resolve_checkpoint_source` first -- the verified, immutable history
    bundle, not the (merely a convenience mirror) root files."""
    resolved_dir = _resolve_checkpoint_source(checkpoint_dir)
    path = resolved_dir / "gene_names.json"
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
    otherwise valid to resume from. Resolves through
    `_resolve_checkpoint_source` first."""
    path = _resolve_checkpoint_source(checkpoint_dir) / "optimizer_rng_state.pt"
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
