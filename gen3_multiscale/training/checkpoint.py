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
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_CHECKPOINT_FILENAMES = (
    "trainable_weights.pt", "model_config.json", "gene_names.json", "training_state.json",
    "optimizer_rng_state.pt", "run_manifest.json",
)
# Codex re-audit of commit f7bb8a1, launch blocker #3/#4: files that MUST
# be present in every bundle's manifest -- "require the exact mandatory
# file set." trainable_weights.pt (absent for a fully-frozen model),
# optimizer_rng_state.pt (absent when no optimizer/rng was passed) and
# run_manifest.json (absent for callers that don't pass one, e.g. best/
# bundles predating this fix) stay conditional.
_MANDATORY_BUNDLE_FILENAMES = frozenset({"model_config.json", "gene_names.json", "training_state.json"})
# Keys `load_optimizer_and_rng_state` will accept inside optimizer_rng_state.pt
# -- "reject unexpected saved state keys" (launch blocker #4).
_OPTIMIZER_RNG_STATE_KEYS = frozenset({
    "optimizer", "python_random", "numpy_random", "torch_random", "loop_random", "torch_cuda_random",
})

# Codex re-audit of commit 2162ff4, finding #3: "constrain bundle_dir to
# the canonical bundle-name regex... validate pointer schema/version and
# exact keys." Matches EXACTLY what `_bundle_dir_name_prefix`/
# `_new_bundle_id` produce (`step_00000042__00000000000000000001_1234_000000`)
# -- a bundle_dir sourced from an on-disk pointer or rollback target is
# never used to build a filesystem path until it has passed this check,
# closing a real path-traversal-shaped gap (a tampered pointer naming
# something like "../../etc" was previously joined onto the history
# directory with no validation at all).
#
# Codex re-audit of commit 66d65f2, minor finding: "the checkpoint bundle
# regex only accepts exactly eight step digits, while the configured
# ceiling can reach 100000000 -- nine digits." Confirmed real:
# `_bundle_dir_name_prefix` formats `step` with Python's `:08d` -- a
# MINIMUM width of 8, zero-padded, but it grows beyond 8 digits for a
# larger step rather than truncating (`f"{100_000_000:08d}"` ==
# "100000000", 9 digits) -- every config's own `training.total_steps:
# 100000000` safety cap is itself a 9-digit value a real long run could
# reach. The prior `\d{8}` (exactly 8) would have REJECTED that
# legitimately-produced bundle name as if it were corrupted. `\d{8,}`
# (8 or more) matches what `:08d` actually produces at any step count
# while still requiring the same minimum width.
_BUNDLE_ID_RE = re.compile(r"^step_\d{8,}__\d{20}_\d+_\d{6}$")
_POINTER_SCHEMA_VERSION = 1
_POINTER_KEYS = frozenset({"version", "step", "bundle_dir", "manifest_sha256"})
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_BUNDLE_MANIFEST_VERSION = 2
_BUNDLE_MANIFEST_KIND = "gen3_checkpoint_step_manifest"

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


def _expected_trainable_state_names(model: nn.Module) -> set[str]:
    """The exact set of state_dict keys `save_trainable_state` saves --
    every trainable parameter plus every buffer NOT owned by a frozen
    backbone module. Shared by `save_trainable_state` (to build the saved
    blob) and `load_trainable_state` (to validate a loaded blob has no
    unexpected keys -- Codex re-audit of commit 2162ff4, finding #3:
    "reject unexpected model-state keys, not only optimizer blob keys")
    so the two can never silently drift apart."""
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    frozen_module_names = {name for name, m in model.named_modules() if _is_frozen_backbone_module(m)}

    def _under_frozen_module(buf_name: str) -> bool:
        parts = buf_name.split(".")
        return any(".".join(parts[:i]) in frozen_module_names for i in range(1, len(parts)))

    names = set(trainable_names)
    for buf_name, _ in model.named_buffers():
        if not _under_frozen_module(buf_name):
            names.add(buf_name)
    return names


def save_trainable_state(model: nn.Module, checkpoint_dir: str | Path) -> Path | None:
    """Save trainable parameters and non-frozen buffers only. Returns the
    weights file path, or None if there was nothing trainable to save
    (e.g. Architecture 4 with its STPath backbone fully frozen and no
    scFoundation residual enabled — an edge case, but handled the same
    way the original codebase's zero-trainable-parameter bug taught us
    to: absence of a weights file is a valid, real state, not an error)."""
    save_names = _expected_trainable_state_names(model)
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
    to the most recently written one.

    Codex re-audit of commit 2162ff4, finding #3: "verify rollback target
    completely before changing the pointer." The prior version picked a
    bundle purely by directory-NAME pattern (`_latest_bundle_for_step`,
    which only parses `list_checkpoint_bundles`' directory listing -- no
    content verification at all) and copied its files to root BEFORE any
    integrity check ran; a corrupted or tampered history bundle would
    only ever be caught LATER, on the next resume/load, by which point
    `latest_bundle.json` had already been pointed at it and root files
    already overwritten. `_verify_bundle` (the same full manifest/file-
    hash/step-consistency chain `_resolve_checkpoint_source` runs for the
    CURRENT pointer's target) now runs FIRST, before any root file is
    touched or the pointer is rewritten -- a bad rollback target raises
    here, leaving the existing checkpoint state completely untouched."""
    bundle_name = _latest_bundle_for_step(checkpoint_dir, step)
    _verify_bundle(checkpoint_dir, bundle_name, expected_step=step, expected_manifest_sha256=None)
    bundle_dir = _history_dir(checkpoint_dir) / bundle_name
    manifest_sha256 = _file_sha256(bundle_dir / "manifest.json")
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
    pointer = {
        "version": _POINTER_SCHEMA_VERSION, "step": int(step), "bundle_dir": bundle_name,
        "manifest_sha256": manifest_sha256,
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
    run_manifest: dict | None = None,
) -> None:
    """Transactional checkpoint save into a uniquely-named, immutable
    bundle -- see this module's docstring for the full crash-safety
    rationale (Codex re-audit of commit 90f853e, launch blocker #1).
    Never deletes or overwrites an existing bundle: `step` alone does not
    name the bundle directory, a fresh unique id does, so two saves at
    the same step simply coexist as two bundles until pruning (which
    only ever removes bundles OLDER than the retained tail, and never
    the one the pointer currently references).

    Codex re-audit of commit 66d65f2, minor finding: "explicitly reject
    negative steps at save time." Confirmed real gap: a negative `step`
    was never checked here -- `_bundle_dir_name_prefix`'s `:08d` format on
    a negative int silently produces a bundle name with a leading minus
    sign (e.g. `step_-0000001__...`), which `_BUNDLE_ID_RE` would then
    reject with an opaque "does not match the canonical bundle-name
    schema" error far downstream, at LOAD time, rather than a clear
    failure right here at the point the bad `step` was actually supplied."""
    if int(step) < 0:
        raise ValueError(f"save_checkpoint: step must be non-negative, got {step!r}")
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
        target.write_text(json.dumps(payload, indent=2, default=str))
        _fsync_path(target)
    # Codex re-audit of commit f7bb8a1, launch blocker #3: binding a
    # verified copy of the run manifest INSIDE the transactional bundle
    # itself (not merely as an unprotected root-level file) means both
    # `best/` and the live checkpoint_dir's identity can be read back and
    # hash-verified through the exact same, already-hardened bundle
    # machinery -- one shared verifier, not two parallel schemes.
    if run_manifest is not None:
        target = staging_dir / "run_manifest.json"
        target.write_text(json.dumps(run_manifest, indent=2, sort_keys=True, default=str))
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
        "version": _BUNDLE_MANIFEST_VERSION, "kind": _BUNDLE_MANIFEST_KIND, "step": int(step),
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
    pointer = {
        "version": _POINTER_SCHEMA_VERSION, "step": int(step), "bundle_dir": bundle_name,
        "manifest_sha256": manifest_sha256,
    }
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


def _load_and_validate_pointer(pointer_path: Path) -> dict:
    """Parse `latest_bundle.json` and validate its FULL schema before any
    field is trusted -- Codex re-audit of commit 2162ff4, finding #3:
    "validate exact pointer keys/types/SHA format." Requires EXACTLY the
    keys this codebase ever writes (no more, no fewer), `version` to
    match the schema this module currently produces, `step` to be a
    non-negative int, `bundle_dir` a non-empty string, and
    `manifest_sha256` to look like a real 64-hex-character sha256 digest
    -- strictly stronger than the prior "truthy" check, which accepted
    any non-empty string (including garbage) as a valid hash."""
    pointer = json.loads(pointer_path.read_text())
    if set(pointer.keys()) != _POINTER_KEYS:
        raise RuntimeError(
            f"{pointer_path}: pointer has keys {sorted(pointer.keys())}, expected exactly "
            f"{sorted(_POINTER_KEYS)} -- refusing to resolve a pointer with an unrecognized schema"
        )
    if pointer.get("version") != _POINTER_SCHEMA_VERSION:
        raise RuntimeError(
            f"{pointer_path}: pointer version {pointer.get('version')!r} != expected "
            f"{_POINTER_SCHEMA_VERSION} -- refusing to resolve a pointer from an unsupported schema version"
        )
    step = pointer.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise RuntimeError(f"{pointer_path}: pointer step {step!r} is not a non-negative integer")
    bundle_dir_name = pointer.get("bundle_dir")
    if not isinstance(bundle_dir_name, str) or not bundle_dir_name:
        raise RuntimeError(f"{pointer_path}: pointer bundle_dir {bundle_dir_name!r} is not a non-empty string")
    manifest_sha256 = pointer.get("manifest_sha256")
    if not isinstance(manifest_sha256, str) or not _SHA256_HEX_RE.match(manifest_sha256):
        raise RuntimeError(
            f"{pointer_path}: pointer manifest_sha256 {manifest_sha256!r} is not a 64-hex-character "
            "sha256 digest -- refusing to resolve an unverifiable or malformed pointer"
        )
    return pointer


def _verify_bundle(
    checkpoint_dir: Path, bundle_name: str, *, expected_step: int, expected_manifest_sha256: str | None,
) -> dict:
    """Full content-integrity verification for ONE history bundle --
    shared by `_resolve_checkpoint_source` (verifying the CURRENT
    pointer's target before ever loading from it) and `rollback_checkpoint`
    (verifying a ROLLBACK target completely BEFORE changing the pointer
    to reference it -- Codex re-audit of commit 2162ff4, finding #3).
    Returns the bundle's own parsed manifest.json. `expected_manifest_
    sha256=None` skips that one specific comparison (rollback has no
    prior pointer recording what this target's hash SHOULD be -- it is
    choosing a fresh target by step number, not verifying an existing
    pointer), every other check still runs.

    Verifies (a) `bundle_name` matches the canonical bundle-name schema
    (`_BUNDLE_ID_RE`) BEFORE it is ever used to build a filesystem path --
    Codex re-audit of commit 2162ff4, finding #3: "constrain bundle_dir
    to the canonical bundle-name regex" -- a tampered pointer/rollback
    target naming e.g. `../../etc` is rejected here, never joined onto
    the history directory; (b) the bundle's OWN manifest.json matches the
    expected sha256 (when given); (c) the manifest's `version`/`kind`/
    `bundle_id` fields (Codex re-audit of commit 2162ff4, finding #3:
    "validate manifest version/kind/bundle_id/step" -- these fields were
    WRITTEN since `save_checkpoint` was first transactionalized but never
    independently READ/validated by any loader); (d) the manifest's `step`
    matches `expected_step`; (e) the manifest's file list is a SUBSET of
    the known checkpoint filenames and a SUPERSET of the mandatory core;
    (f) the real bundle directory contents match the manifest's file list
    exactly (no unlisted extra files); (g) every listed file's real sha256
    matches; (h) `training_state.json`'s own `step` also agrees."""
    if not _BUNDLE_ID_RE.match(bundle_name):
        raise RuntimeError(
            f"checkpoint at {checkpoint_dir}: bundle_dir {bundle_name!r} does not match the canonical "
            f"bundle-name schema ({_BUNDLE_ID_RE.pattern}) -- refusing to resolve a path built from an "
            "unrecognized or potentially unsafe bundle name"
        )
    bundle_dir = _history_dir(checkpoint_dir) / bundle_name
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"checkpoint at {checkpoint_dir} references bundle {bundle_name!r} (step {expected_step}) but "
            f"its manifest is missing at {manifest_path} -- refusing to load a possibly-partial or "
            "corrupted checkpoint"
        )
    actual_manifest_sha256 = _file_sha256(manifest_path)
    if expected_manifest_sha256 is not None and actual_manifest_sha256 != expected_manifest_sha256:
        raise RuntimeError(
            f"checkpoint at {checkpoint_dir}: bundle {bundle_name!r}'s manifest.json does not match "
            "the expected sha256 -- corrupted or tampered checkpoint, refusing to load"
        )
    step_manifest = json.loads(manifest_path.read_text())
    if step_manifest.get("version") != _BUNDLE_MANIFEST_VERSION:
        raise RuntimeError(
            f"checkpoint bundle {bundle_name!r}: manifest version {step_manifest.get('version')!r} != "
            f"expected {_BUNDLE_MANIFEST_VERSION} -- refusing to load a bundle from an unsupported "
            "manifest schema"
        )
    if step_manifest.get("kind") != _BUNDLE_MANIFEST_KIND:
        raise RuntimeError(
            f"checkpoint bundle {bundle_name!r}: manifest kind {step_manifest.get('kind')!r} != "
            f"expected {_BUNDLE_MANIFEST_KIND!r} -- refusing to load a bundle with an unrecognized kind"
        )
    if step_manifest.get("bundle_id") != bundle_name:
        raise RuntimeError(
            f"checkpoint bundle {bundle_name!r}: manifest bundle_id {step_manifest.get('bundle_id')!r} "
            f"does not match its own directory name {bundle_name!r} -- refusing to load a bundle whose "
            "manifest disagrees about its own identity"
        )
    manifest_step = step_manifest.get("step")
    if manifest_step is None or int(manifest_step) != int(expected_step):
        raise RuntimeError(
            f"checkpoint at {checkpoint_dir}: expected step={expected_step} but bundle {bundle_name!r}'s "
            f"own manifest.json records step={manifest_step!r} -- refusing to load a checkpoint whose "
            "pointer/rollback-target and bundle disagree about which step this is"
        )
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
            f"checkpoint bundle {bundle_name!r} (step {expected_step}) contains file(s) {sorted(unlisted)} "
            "that are not listed in its own manifest.json -- refusing to load a bundle whose manifest "
            "does not fully account for its real contents"
        )
    # Codex re-audit of commit f7bb8a1, launch blocker #4: "require the
    # exact mandatory file set." Every name the manifest lists must be
    # one this codebase actually knows how to produce (no unexpected
    # extra state smuggled into a bundle), and every mandatory-core name
    # must be present (removing BOTH a required file and its manifest
    # entry -- the exact adversarial scenario Codex names -- is caught
    # here even though this function itself has nothing left on disk to
    # hash-check against).
    manifest_file_names = set(manifest_files.keys())
    unexpected = manifest_file_names - set(_CHECKPOINT_FILENAMES)
    if unexpected:
        raise RuntimeError(
            f"checkpoint bundle {bundle_name!r} (step {expected_step}) manifest lists unexpected file "
            f"name(s) {sorted(unexpected)} -- refusing to load a bundle with unrecognized saved state"
        )
    missing_mandatory = _MANDATORY_BUNDLE_FILENAMES - manifest_file_names
    if missing_mandatory:
        raise RuntimeError(
            f"checkpoint bundle {bundle_name!r} (step {expected_step}) manifest is missing mandatory "
            f"file(s) {sorted(missing_mandatory)} -- refusing to load an incomplete bundle"
        )
    for name, expected_hash in manifest_files.items():
        file_path = bundle_dir / name
        if not file_path.is_file():
            raise RuntimeError(
                f"checkpoint bundle {bundle_name!r} (step {expected_step}) manifest references {name} but "
                f"it is missing from {bundle_dir} -- refusing to load a corrupted checkpoint"
            )
        if _file_sha256(file_path) != expected_hash:
            raise RuntimeError(
                f"checkpoint bundle {bundle_name!r} (step {expected_step}) file {name} does not match the "
                "sha256 recorded in its own bundle manifest -- corrupted or partially-written checkpoint, "
                "refusing to load"
            )
    training_state_path = bundle_dir / "training_state.json"
    training_state_step = json.loads(training_state_path.read_text()).get("step")
    if training_state_step is None or int(training_state_step) != int(expected_step):
        raise RuntimeError(
            f"checkpoint bundle {bundle_name!r}: expected step={expected_step} but training_state.json "
            f"records step={training_state_step!r} -- refusing to load a checkpoint whose own files "
            "disagree about which step this is"
        )
    return step_manifest


def _resolve_checkpoint_source(checkpoint_dir: str | Path) -> Path:
    """Fail-closed transactional load resolution. If `checkpoint_dir/
    latest_bundle.json` exists, resolves to the bundle it names and runs
    `_verify_bundle`'s full content-integrity chain against it. Falls
    back to `checkpoint_dir` itself only when there is NO pointer, NO
    history bundles, AND no orphaned root-level checkpoint files -- a
    genuinely fresh, never-saved-to directory. A missing pointer with a
    NON-EMPTY history/ is refused (Codex re-audit of commit f7bb8a1,
    launch blocker #4: "reject a missing pointer when canonical
    history/training artifacts exist") -- that combination can only mean
    the pointer was lost/deleted after real bundles were already written,
    not "nothing has been saved here yet."

    Codex re-audit of commit 2162ff4, finding #3: "Root checkpoint files
    can still be accepted when history was removed and the pointer is
    missing." Confirmed real: root-level `_CHECKPOINT_FILENAMES` are
    ONLY EVER written by `save_checkpoint`'s mirror step, which ALWAYS
    also writes `latest_bundle.json` first (see that function) -- there
    is no legitimate code path that produces root files without a
    pointer for a live `checkpoint_dir`. A missing pointer alongside
    EXISTING root files therefore normally means the pointer (and
    possibly the whole history/ directory) was lost or deleted after a
    real save; those orphaned root files are now refused too, not
    silently trusted as "nothing has ever been saved here." The one
    legitimate exception -- callers directly resolving a HISTORY BUNDLE'S
    OWN path (e.g. `checkpoint.py`'s own tests load a specific past
    snapshot by `history/<bundle_name>/`), which genuinely has no pointer
    OF ITS OWN by design -- is distinguished by the presence of that
    bundle's own `manifest.json`, a file `save_checkpoint`'s root-mirror
    step never copies to a live checkpoint_dir's root; only a real bundle
    directory ever has one."""
    checkpoint_dir = Path(checkpoint_dir)
    pointer_path = checkpoint_dir / "latest_bundle.json"
    if not pointer_path.is_file():
        if list_checkpoint_bundles(checkpoint_dir):
            raise RuntimeError(
                f"checkpoint at {checkpoint_dir} has real history bundles but no latest_bundle.json "
                "pointer -- refusing to silently fall back to (possibly stale or hand-edited) root "
                "mirror files. The pointer was lost or deleted after bundles were already written; "
                "this checkpoint_dir cannot be safely resolved without it"
            )
        # `save_checkpoint` ALWAYS writes this exact mandatory triple
        # together, unconditionally, every single call -- their combined
        # presence is a much stronger, more specific signal that a real
        # save_checkpoint call once ran here (and its pointer/history was
        # later lost) than any ONE checkpoint-shaped file existing alone,
        # which could just as easily be a partially/hand-constructed
        # directory (e.g. a test fixture deliberately writing only some
        # metadata files to simulate an incomplete checkpoint) that never
        # went through this module's save path at all.
        existing_root_files = [name for name in _CHECKPOINT_FILENAMES if (checkpoint_dir / name).is_file()]
        has_mandatory_root_triple = _MANDATORY_BUNDLE_FILENAMES.issubset(set(existing_root_files))
        if has_mandatory_root_triple and not (checkpoint_dir / "manifest.json").is_file():
            raise RuntimeError(
                f"checkpoint at {checkpoint_dir} has root-level file(s) {existing_root_files} but no "
                "latest_bundle.json pointer and no history bundles -- root files are only ever written "
                "as a mirror alongside a real transactional save (which always writes the pointer "
                "first), so their presence without a pointer means the pointer (and possibly the whole "
                "history/ directory) was lost or deleted after a real save. Refusing to silently trust "
                "orphaned root files as if nothing had ever been saved here"
            )
        return checkpoint_dir
    pointer = _load_and_validate_pointer(pointer_path)
    _verify_bundle(
        checkpoint_dir, pointer["bundle_dir"], expected_step=pointer["step"],
        expected_manifest_sha256=pointer["manifest_sha256"],
    )
    return _history_dir(checkpoint_dir) / pointer["bundle_dir"]


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
        # `_resolve_checkpoint_source` above already validated this exact
        # pointer via `_load_and_validate_pointer` -- re-validating here
        # (rather than a raw, unvalidated `json.loads`) keeps this
        # function's own reading consistent with everything else that
        # touches a pointer file.
        pointer = _load_and_validate_pointer(pointer_path)
        step = pointer["step"]
        bundle_dir = pointer["bundle_dir"]
        manifest_sha256 = pointer["manifest_sha256"]
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
    a convenience mirror) root files.

    Codex re-audit of commit 2162ff4, finding #3: "reject unexpected
    model-state keys, not only optimizer blob keys." Confirmed real:
    `model.load_state_dict(state, strict=False)` silently DROPS any key
    in `state` that the current model doesn't have a matching parameter/
    buffer for -- a saved blob carrying extra, unrecognized keys (a
    corrupted save, a checkpoint from a different/older architecture
    version with since-removed parameters, or tampering) previously
    loaded without complaint.

    Codex re-audit of commit 66d65f2, finding #2: "load_trainable_state()
    rejects missing trainable parameters and unexpected keys, but not
    missing expected buffers such as target_gene_scale, gene-basis
    buffers, or coordinate frequencies. A damaged checkpoint could
    therefore silently retain freshly initialized buffers." Confirmed
    real: the PRIOR check only required `state`'s keys to be a SUBSET of
    `expected_names` (trainable parameters + non-frozen buffers) and that
    every TRAINABLE parameter specifically was present -- a saved blob
    missing a non-frozen BUFFER (present in `expected_names` but not
    `trainable_names`) passed both checks silently, and
    `model.load_state_dict(state, strict=False)` then left that buffer at
    whatever the freshly-constructed model initialized it to, not the
    checkpoint's own saved value. The check is now a single exact-set
    equality: `state`'s keys must equal `expected_names` exactly, no more
    (still refuses unexpected/tampered/stale keys) and no fewer (now also
    refuses a checkpoint silently missing ANY expected buffer, not merely
    a missing trainable parameter). `strict=False` is kept on the actual
    `load_state_dict` call only because this exact-set check ALREADY
    guarantees the loaded blob's keys are precisely what the model
    expects -- `strict=True` would be equivalent given that guarantee,
    but `strict=False` avoids a second, redundant internal key-set
    comparison inside PyTorch's own implementation."""
    in_dir = _resolve_checkpoint_source(checkpoint_dir)
    weights_path = in_dir / "trainable_weights.pt"
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    expected_names = _expected_trainable_state_names(model)
    if weights_path.is_file():
        state = torch.load(weights_path, map_location="cpu")
        missing = expected_names - set(state.keys())
        if missing:
            raise RuntimeError(
                f"saved weights at {in_dir} are missing expected trainable parameter(s)/buffer(s) this "
                f"model architecture expects: {sorted(missing)} (config mismatch, or a damaged/incomplete "
                "checkpoint that would otherwise silently retain freshly initialized values for these)"
            )
        unexpected = set(state.keys()) - expected_names
        if unexpected:
            raise RuntimeError(
                f"saved weights at {in_dir} contain unexpected key(s) {sorted(unexpected)} that this "
                "model architecture's trainable parameters/non-frozen buffers do not expect -- refusing "
                "to load a checkpoint with unrecognized saved state"
            )
        model.load_state_dict(state, strict=False)
    else:
        # Codex re-audit of commit 7a2d819, finding #1: "if
        # trainable_weights.pt is absent, load_trainable_state() only
        # checks whether trainable parameters exist -- not expected
        # buffers." Confirmed real: this branch (the ENTIRE weights file
        # missing, not merely a key inside it) checked `trainable_names`
        # only -- a fully-frozen model (`trainable_names` empty) that
        # still has real non-frozen registered buffers (`expected_names`
        # non-empty, e.g. `target_gene_scale`) passed silently here even
        # though the checkpoint has NO saved buffer values at all,
        # continuing with whatever the freshly-constructed model
        # happened to initialize those buffers to. `expected_names` is
        # the same complete set (trainable parameters + non-frozen
        # buffers) the `weights_path.is_file()` branch above already
        # requires exact equality against -- checking it here instead of
        # `trainable_names` makes "the whole file is missing" fail
        # exactly whenever "the file exists but is missing some expected
        # keys" would have failed too.
        if expected_names:
            raise RuntimeError(
                f"{weights_path} is missing but this model architecture has expected trainable "
                f"parameter(s)/buffer(s) {sorted(expected_names)}; the checkpoint at {in_dir} looks incomplete"
            )


def load_checkpoint_run_manifest(checkpoint_dir: str | Path) -> dict | None:
    """The `run_manifest.json` bound INSIDE the resolved bundle (via
    `save_checkpoint(..., run_manifest=...)`), hash-verified by
    `_resolve_checkpoint_source` exactly like every other bundle file --
    `None` if this checkpoint (or this specific bundle) was saved without
    one. Codex re-audit of commit f7bb8a1, launch blocker #3: the single
    shared read path `train.py::verify_full_checkpoint_identity` uses for
    BOTH `best/` and a live checkpoint_dir's latest state, so both are
    verified against the exact same complete identity fields."""
    path = _resolve_checkpoint_source(checkpoint_dir) / "run_manifest.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def load_training_state(checkpoint_dir: str | Path) -> dict:
    """`{"step": 0}` ONLY for a genuinely fresh/unstarted `checkpoint_dir`
    (no pointer, no history bundles at all -- see `_resolve_checkpoint_source`,
    which itself now raises rather than falling back here if a pointer is
    missing while real bundles exist). Codex re-audit of commit f7bb8a1,
    launch blocker #4: "never turn a missing training_state into step 0
    for a checkpointed run" -- once `_resolve_checkpoint_source` has
    resolved to a REAL, pointer-referenced bundle, that bundle's manifest
    is already required (via `_MANDATORY_BUNDLE_FILENAMES`) to list
    `training_state.json` and to have verified its content hash, so this
    read can never silently substitute a fabricated step for a bundle
    that genuinely has one missing or corrupted."""
    resolved_dir = _resolve_checkpoint_source(checkpoint_dir)
    path = resolved_dir / "training_state.json"
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
    # Codex re-audit of commit f7bb8a1, launch blocker #4: "reject
    # unexpected saved state keys" -- a hand-edited or corrupted blob
    # carrying an extra top-level key this codebase never wrote must fail
    # loudly rather than silently loading only the keys it recognizes.
    unexpected_keys = set(blob.keys()) - _OPTIMIZER_RNG_STATE_KEYS
    if unexpected_keys:
        raise RuntimeError(
            f"{path}: optimizer_rng_state.pt contains unexpected key(s) {sorted(unexpected_keys)} -- "
            "refusing to load a blob with unrecognized saved state"
        )
    optimizer.load_state_dict(blob["optimizer"])
    _restore_rng_state(blob, rng)
    return True
