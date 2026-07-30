"""Neutral, dependency-free config-loading and -fingerprinting
utilities -- Codex re-audit of commit 57f0e3c: "Prefer moving config
fingerprint/loading utilities into a neutral module to avoid a circular
import between the trainer and resolver." Confirmed real: `scripts/
resolve_experiment_config.py` (the resolver) imports `config_fingerprint`/
`config_identity_fingerprint` FROM `training/train.py` (the trainer) --
a one-way dependency that works today only because nothing in `train.py`
needs anything from the resolver. The moment a future orchestrator
module needs BOTH `train.py`'s own machinery (to actually run training)
AND the resolver's `load_verified_resolved_config` (to consume a
resolved-config bundle) it would trigger training -> orchestrator ->
resolver -> training, a real circular import. These three functions
have no dependency on anything in `train.py` beyond `hashlib`/`json`/
`OmegaConf` -- they are pulled out here, into a module BOTH the trainer
and the resolver (and the orchestrator) can depend on without ever
depending on each other. `training/train.py` re-exports all three
(`from gen3_multiscale.config_identity import ...`) so every existing
`from gen3_multiscale.training.train import config_fingerprint, ...`
caller across this codebase keeps working unchanged."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf


def resolved_config(config_path: str | Path) -> dict:
    return OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)


def config_fingerprint(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode("utf-8")).hexdigest()


# Purely operational/scheduling training fields that a legitimate resume
# workflow ("bump total_steps and keep training", "raise checkpoint
# cadence", ...) must be free to change without verify_resume_consistency
# refusing to continue -- everything else in `training` (seed, lr,
# optimizer hyperparameters, synchronized_init_dir, device, ...) still
# counts toward the run's SCIENTIFIC identity and must stay fixed.
RESUME_EXCLUDED_TRAINING_FIELDS = frozenset({
    "total_steps", "checkpoint_every_n_steps", "log_every_n_steps", "eval_every_n_steps",
    "max_wall_clock_hours", "checkpoint_dir", "checkpoint_keep_last",
})


def config_identity_fingerprint(config: dict) -> str:
    """Same content as `config_fingerprint`, minus
    `RESUME_EXCLUDED_TRAINING_FIELDS` and the entire `evaluation` section
    -- the fingerprint `verify_resume_consistency`/
    `verify_full_checkpoint_identity` actually compare. Adam's Step 6
    audit #5: "Verify the existing run manifest/config/dataset/cache/
    LongNet/init/basis fingerprints before loading anything. Refuse
    changed configs or artifacts" -- but a resumed run legitimately needs
    to be able to ask for MORE steps, a different checkpoint cadence, or
    a raised wall-clock budget without that being treated as "a changed
    config" in the sense this check is meant to catch.

    The `evaluation` section (Codex re-audit of commit f7bb8a1, confirmed
    real gap surfaced while wiring `verify_full_checkpoint_identity` into
    BOTH best/ and latest-checkpoint evaluation): named gene panels,
    `n_masks_per_sample`, `compute_st_fid_mmd`, and similar
    evaluation-only settings never affect what was actually TRAINED --
    only how an already-trained checkpoint is LATER measured. Binding
    them into the checkpoint's scientific identity would mean adding a
    new named evaluation gene panel, or raising an evaluation sample
    count, retroactively "invalidates" every already-trained checkpoint
    for evaluation purposes, which is not a real identity change."""
    identity_config = json.loads(json.dumps(config, default=str))  # deep copy, same serialization the hash itself uses
    training_section = dict(identity_config.get("training") or {})
    for field in RESUME_EXCLUDED_TRAINING_FIELDS:
        training_section.pop(field, None)
    identity_config["training"] = training_section
    identity_config.pop("evaluation", None)
    return hashlib.sha256(json.dumps(identity_config, sort_keys=True, default=str).encode("utf-8")).hexdigest()
