#!/usr/bin/env python3
"""The real staged deployment orchestrator -- Stages A-D, per Adam's own
repeatedly-restated spec (CONTRACT.md sections 53-56's own "What remains
honestly undone" lists) and the Codex re-audit of commit 57f0e3c's
explicit final instruction: "Build Stages A-D with immutable stage
outputs, resumable state, failure gates, exact Architecture 3 bundle
selection, basis fitting from that bundle, and Architecture 4 consuming
that same bundle. Stop before launching GPU training for Codex audit."

Stage A: train Architecture 1/2/3 (`run_stage_train_architecture`, called
    once per architecture with that architecture's own resolved config
    bundle).
Stage B: select Architecture 3's checkpoint (`run_stage_select_
    architecture3`) -- evaluates it on the VALIDATION split ONLY
    (structurally hard-coded; there is no parameter to point this stage
    at the test split) and pins its exact immutable bundle identity.
Stage C: fit Architecture 4's gene-residual basis from THAT exact
    selected bundle (`run_stage_fit_basis`) -- reads the checkpoint path
    from Stage B's own recorded output, never a caller-supplied
    override.
Stage D: train Architecture 4 (`run_stage_train_architecture4`) --
    cross-checks its config bundle's OWN configured `architecture3_
    conditioner_checkpoint`/`gene_residual_basis` against Stages B/C's
    recorded outputs and refuses to proceed on any mismatch, closing
    "never let Architecture 4 use an older/pre-existing Architecture 3
    path merely because it exists" structurally rather than by
    convention.

Every stage:
  - consumes its config through `resolve_experiment_config.py::
    load_verified_resolved_config` -- a resolved-config BUNDLE
    DIRECTORY, never the loose `bundle/config.yaml` path used without
    that verification running first (Codex re-audit of commit 57f0e3c:
    "production commands must consume resolved-config bundle directories
    through load_verified_resolved_config(). They must not bypass
    verification by passing the loose bundle/config.yaml path
    directly").
  - persists its result into one JSON state file, written atomically
    (staging-temp-file-then-`os.replace`, this codebase's standard
    discipline), so an interrupted orchestration run can be resumed by
    simply re-invoking the same stage function again.
  - is IDEMPOTENT and its output IMMUTABLE once completed: re-running an
    already-completed stage with the SAME inputs returns the existing
    recorded result without redoing any work; re-running it with
    DIFFERENT inputs raises rather than silently overwriting what a
    later stage may already trust.
  - is a FAILURE GATE for whatever depends on it: a stage whose
    precondition (an earlier stage's own state) is not `"completed"`
    raises immediately, before doing any work.

What this module deliberately does NOT do, honestly documented rather
than silently assumed: no CLI entrypoint yet (stages are called as
library functions -- a `main()`/argparse wrapper is real, useful,
future work, not built this round); no automatic "run every stage in
order" convenience driver (a caller/operator invokes each stage
explicitly, in order, so a human decision -- or a human-authored
top-level script -- always sits between stages, especially before the
one stage that actually starts the real, hours-long GPU training runs);
no disk/RAM/GPU capacity estimation, no run-plan JSON generation, no
static-config-audit/staged-smoke-gate integration (`launch_four_gpu_
suite.py`'s own preflight machinery is a separate, already-built and
already-tested piece this module does not yet call). This remains a
genuinely large, multi-stage engineering effort; this round builds and
tests the core state machine and the four stage functions against real
(CPU-scale, synthetic) training/evaluation/basis-fitting runs -- never
against real GPU hardware or real HEST-1k data, and no 24-hour run is
started by anything in this module."""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

from gen3_multiscale.evaluation.gen3_evaluator import evaluate_gen3_checkpoint
from gen3_multiscale.scripts.fit_architecture4_residual_basis import fit_and_save_architecture4_basis
from gen3_multiscale.scripts.resolve_experiment_config import load_verified_resolved_config
from gen3_multiscale.training import checkpoint as checkpoint_module
from gen3_multiscale.training import train as train_module

_STATE_SCHEMA_VERSION = 1


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _identity_to_dict(identity: checkpoint_module.CheckpointIdentity) -> dict:
    return {
        "resolved_dir": str(identity.resolved_dir), "step": identity.step, "bundle_dir": identity.bundle_dir,
        "manifest_sha256": identity.manifest_sha256, "weights_sha256": identity.weights_sha256,
    }


def load_orchestrator_state(state_path: str | Path) -> dict:
    state_path = Path(state_path)
    if not state_path.is_file():
        return {"version": _STATE_SCHEMA_VERSION, "kind": "gen3_orchestrator_state", "stages": {}}
    return json.loads(state_path.read_text())


def _save_orchestrator_state_atomic(state: dict, state_path: str | Path) -> None:
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_name(f"{state_path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True, default=str))
    os.replace(tmp, state_path)


def _require_stage_completed(state: dict, stage_name: str) -> dict:
    stage = (state.get("stages") or {}).get(stage_name)
    if not stage or stage.get("status") != "completed":
        raise RuntimeError(
            f"orchestrator stage {stage_name!r} has not completed (status="
            f"{stage.get('status') if stage else None!r}) -- refusing to run a dependent stage before its "
            "precondition is satisfied"
        )
    return stage


def _run_stage(state_path: str | Path, stage_name: str, resume_check: dict, fn) -> dict:
    """Shared bookkeeping for every stage: idempotent resume (an
    already-completed stage with matching `resume_check` inputs is
    returned unchanged, no work redone), immutable output (a completed
    stage invoked again with DIFFERENT inputs raises rather than
    silently overwriting it), and atomic state persistence around
    running/completed/failed transitions."""
    state = load_orchestrator_state(state_path)
    stages = state.setdefault("stages", {})
    existing = stages.get(stage_name)
    if existing and existing.get("status") == "completed":
        if existing.get("resume_check") != resume_check:
            raise ValueError(
                f"orchestrator stage {stage_name!r} already completed with resume_check="
                f"{existing.get('resume_check')!r}, but this call supplied different inputs "
                f"({resume_check!r}) -- refusing to silently overwrite an immutable completed stage; "
                "start a fresh orchestrator state (a new --state-path) for a genuinely different run"
            )
        return existing
    stages[stage_name] = {
        "status": "running", "resume_check": resume_check, "started_at": _now_iso(),
        "completed_at": None, "output": None, "error": None,
    }
    _save_orchestrator_state_atomic(state, state_path)
    try:
        output = fn()
    except BaseException as exc:
        stages[stage_name] = {
            "status": "failed", "resume_check": resume_check, "started_at": stages[stage_name]["started_at"],
            "completed_at": _now_iso(), "output": None, "error": f"{type(exc).__name__}: {exc}",
        }
        _save_orchestrator_state_atomic(state, state_path)
        raise
    stages[stage_name] = {
        "status": "completed", "resume_check": resume_check, "started_at": stages[stage_name]["started_at"],
        "completed_at": _now_iso(), "output": output, "error": None,
    }
    _save_orchestrator_state_atomic(state, state_path)
    return stages[stage_name]


# ---------------------------------------------------------------------------
# Stage A: train Architecture 1/2/3.
# ---------------------------------------------------------------------------

def run_stage_train_architecture(
    state_path: str | Path, stage_name: str, config_bundle_dir: str | Path,
    *, smoke: bool = False, staged_smoke: bool = False, allow_code_drift: bool = False,
) -> dict:
    """Trains ONE architecture from an already-resolved config bundle.
    `stage_name` is caller-chosen (e.g. `"train_architecture1"`,
    `"train_architecture3"`) so this one function serves all of Stage
    A's three training runs without duplicating the training call three
    times."""
    config_bundle_dir = Path(config_bundle_dir)
    resume_check = {
        "config_bundle_dir": str(config_bundle_dir), "smoke": bool(smoke), "staged_smoke": bool(staged_smoke),
    }

    def _do() -> dict:
        verified_config = load_verified_resolved_config(config_bundle_dir)
        config_path = config_bundle_dir / "config.yaml"
        summary = train_module.run_training(
            str(config_path), smoke=smoke, staged_smoke=staged_smoke, allow_code_drift=allow_code_drift,
        )
        if not summary.get("ok"):
            raise RuntimeError(f"training stage {stage_name!r} did not complete successfully: {summary}")
        checkpoint_dir = Path(verified_config["training"]["checkpoint_dir"])
        best_dir = checkpoint_dir / "best"
        identity = checkpoint_module.resolve_checkpoint_identity(best_dir if best_dir.is_dir() else checkpoint_dir)
        return {
            "config_bundle_dir": str(config_bundle_dir),
            "architecture": str(verified_config["model"]["architecture"]),
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_identity": _identity_to_dict(identity),
            "run_summary": summary,
        }

    return _run_stage(state_path, stage_name, resume_check, _do)


# ---------------------------------------------------------------------------
# Stage B: select Architecture 3's exact bundle -- validation split ONLY.
# ---------------------------------------------------------------------------

def run_stage_select_architecture3(state_path: str | Path, *, n_masks_per_sample: int = 8) -> dict:
    """Evaluates the trained Architecture 3 checkpoint on the VALIDATION
    split -- `split` is not a parameter of this function at all, so
    there is no way to point Stage B at the test split, ever. Requires
    `"train_architecture3"` to already be completed; reads that stage's
    OWN recorded `checkpoint_dir`/config bundle, never a caller-supplied
    path -- there is no parameter here through which a caller could
    point selection at a different checkpoint even by mistake."""
    resume_check = {"n_masks_per_sample": int(n_masks_per_sample)}

    def _do() -> dict:
        state = load_orchestrator_state(state_path)
        train_stage = _require_stage_completed(state, "train_architecture3")
        train_output = train_stage["output"]
        checkpoint_dir = Path(train_output["checkpoint_dir"])
        config_bundle_dir = Path(train_output["config_bundle_dir"])
        load_verified_resolved_config(config_bundle_dir)  # verify before use
        config_path = config_bundle_dir / "config.yaml"

        report = evaluate_gen3_checkpoint(
            str(config_path), checkpoint_dir, split="validation", n_masks_per_sample=n_masks_per_sample,
        )
        # A second, independent resolve of the SAME `best/` pointer,
        # immediately after evaluation -- not a repeat of the TOCTOU bug
        # (which was "verify against A, silently load/use B"): this
        # explicitly COMPARES against what evaluation itself already
        # pinned and FAILS CLOSED on any disagreement, rather than
        # silently trusting either value alone.
        selected_identity = checkpoint_module.resolve_checkpoint_identity(checkpoint_dir / "best")
        if selected_identity.weights_sha256 != report["checkpoint_identity"]["weights_sha256"]:
            raise RuntimeError(
                "the Architecture 3 checkpoint changed between evaluation and selection-pinning -- refusing "
                "to select a bundle that could not be re-resolved consistently"
            )
        return {
            "config_bundle_dir": str(config_bundle_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "selected_checkpoint_identity": _identity_to_dict(selected_identity),
            "validation_report_summary": {
                "n_items": report["n_items"],
                "model_pcc": report["per_arm_patient_aggregated_metrics"]["model"].get("pcc"),
            },
        }

    return _run_stage(state_path, "select_architecture3", resume_check, _do)


# ---------------------------------------------------------------------------
# Stage C: fit Architecture 4's basis from the SELECTED Architecture 3 bundle.
# ---------------------------------------------------------------------------

def run_stage_fit_basis(
    state_path: str | Path, architecture3_config_bundle_dir: str | Path, output_basis_path: str | Path,
    *, n_masks_per_sample: int = 20, rank: int = 64,
) -> dict:
    """Fits the residual basis against the EXACT bundle Stage B selected
    -- the checkpoint path is read from `select_architecture3`'s own
    recorded output, never accepted as a parameter of this function, so
    there is no way to fit a basis against a different Architecture 3
    checkpoint than the one this orchestration actually selected.
    `architecture3_config_bundle_dir` supplies only the CONFIG (masking
    strata, dataset manifest path, ...) fitting needs -- never the
    checkpoint path."""
    architecture3_config_bundle_dir = Path(architecture3_config_bundle_dir)
    output_basis_path = Path(output_basis_path)
    resume_check = {
        "architecture3_config_bundle_dir": str(architecture3_config_bundle_dir),
        "output_basis_path": str(output_basis_path), "n_masks_per_sample": int(n_masks_per_sample), "rank": int(rank),
    }

    def _do() -> dict:
        state = load_orchestrator_state(state_path)
        select_stage = _require_stage_completed(state, "select_architecture3")
        selected_checkpoint_dir = select_stage["output"]["selected_checkpoint_identity"]["resolved_dir"]
        load_verified_resolved_config(architecture3_config_bundle_dir)  # verify before use
        config_path = architecture3_config_bundle_dir / "config.yaml"

        provenance = fit_and_save_architecture4_basis(
            str(config_path), selected_checkpoint_dir, str(output_basis_path),
            n_masks_per_sample=n_masks_per_sample, rank=rank,
        )
        return {
            "architecture3_config_bundle_dir": str(architecture3_config_bundle_dir),
            "selected_checkpoint_dir": str(selected_checkpoint_dir),
            "output_basis_path": str(output_basis_path),
            "provenance": provenance,
        }

    return _run_stage(state_path, "fit_basis", resume_check, _do)


# ---------------------------------------------------------------------------
# Stage D: train Architecture 4 -- consuming the SAME selected bundle + basis.
# ---------------------------------------------------------------------------

def run_stage_train_architecture4(
    state_path: str | Path, config_bundle_dir: str | Path,
    *, smoke: bool = False, staged_smoke: bool = False, allow_code_drift: bool = False,
) -> dict:
    """Trains Architecture 4. Requires BOTH `"select_architecture3"` and
    `"fit_basis"` to already be completed, and cross-checks the config
    bundle's OWN configured `required_fingerprints.architecture3_
    conditioner_checkpoint`/`gene_residual_basis` against those stages'
    recorded outputs -- the central "never let Architecture 4 use an
    older/pre-existing Architecture 3 path merely because it exists"
    guarantee, enforced structurally (a mismatch raises before any
    training starts), not merely by convention or operator discipline."""
    config_bundle_dir = Path(config_bundle_dir)
    resume_check = {
        "config_bundle_dir": str(config_bundle_dir), "smoke": bool(smoke), "staged_smoke": bool(staged_smoke),
    }

    def _do() -> dict:
        state = load_orchestrator_state(state_path)
        select_stage = _require_stage_completed(state, "select_architecture3")
        fit_basis_stage = _require_stage_completed(state, "fit_basis")
        expected_conditioner_dir = Path(select_stage["output"]["selected_checkpoint_identity"]["resolved_dir"])
        expected_basis_path = Path(fit_basis_stage["output"]["output_basis_path"]).resolve()

        verified_config = load_verified_resolved_config(config_bundle_dir)
        architecture_id = str(verified_config["model"]["architecture"])
        if architecture_id != "4":
            raise ValueError(
                f"run_stage_train_architecture4 requires an Architecture 4 config bundle, got "
                f"model.architecture={architecture_id!r}"
            )
        required_fingerprints = verified_config.get("required_fingerprints") or {}
        configured_conditioner_dir = required_fingerprints.get("architecture3_conditioner_checkpoint")
        configured_basis_path = required_fingerprints.get("gene_residual_basis")
        if not configured_conditioner_dir:
            raise ValueError(
                f"{config_bundle_dir}: required_fingerprints.architecture3_conditioner_checkpoint is not "
                "set -- cannot verify it matches this orchestration's selected Architecture 3 bundle"
            )
        configured_conditioner_identity = checkpoint_module.resolve_checkpoint_identity(configured_conditioner_dir)
        if configured_conditioner_identity.resolved_dir != expected_conditioner_dir:
            raise ValueError(
                f"{config_bundle_dir}'s required_fingerprints.architecture3_conditioner_checkpoint resolves to "
                f"{configured_conditioner_identity.resolved_dir} but this orchestrator's select_architecture3 "
                f"stage selected {expected_conditioner_dir} -- refusing to train Architecture 4 against a "
                "different, possibly older/pre-existing Architecture 3 bundle than the one this "
                "orchestration actually selected"
            )
        if not configured_basis_path or Path(configured_basis_path).resolve() != expected_basis_path:
            raise ValueError(
                f"{config_bundle_dir}'s required_fingerprints.gene_residual_basis ({configured_basis_path!r}) "
                f"does not match this orchestrator's fit_basis stage output ({expected_basis_path}) -- "
                "refusing to train Architecture 4 against a different basis than the one this "
                "orchestration actually fit"
            )

        config_path = config_bundle_dir / "config.yaml"
        summary = train_module.run_training(
            str(config_path), smoke=smoke, staged_smoke=staged_smoke, allow_code_drift=allow_code_drift,
        )
        if not summary.get("ok"):
            raise RuntimeError(f"architecture4 training did not complete successfully: {summary}")
        checkpoint_dir = Path(verified_config["training"]["checkpoint_dir"])
        return {
            "config_bundle_dir": str(config_bundle_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "architecture3_conditioner_checkpoint": str(expected_conditioner_dir),
            "gene_residual_basis": str(expected_basis_path),
            "run_summary": summary,
        }

    return _run_stage(state_path, "train_architecture4", resume_check, _do)
