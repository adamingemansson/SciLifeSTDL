#!/usr/bin/env python3
"""Step 6 deliverable: tiny single-sample overfit/capacity test.

Takes a real architectureN.yaml (with data.gen3_manifest_path already set
to a real, on-disk dataset manifest) and one manifest sample id, builds a
DERIVED manifest restricted to exactly that one training sample and ZERO
validation/test samples (written to a fresh temp file -- the original,
immutable manifest on disk is never modified), then runs
training/train.py's real `run_training` for a small, explicit number of
real (non-smoke) optimizer steps with no held-out masking at all.

This is the standard "can this architecture memorize one sample" sanity
check before trusting a longer multi-sample run -- reuses run_training's
entire real orchestration (preflight, dataset, model, synchronized init,
checkpointing, finite-loss checks) unmodified; the only thing this script
does is narrow the manifest and steps before calling it.

Deliberately does not default --n-steps high enough to look like a real
training run: the caller must explicitly ask for more steps if they want
a longer capacity check.

    python -m gen3_multiscale.scripts.step6_overfit_test \\
        --config gen3_multiscale/configs/architecture1.yaml \\
        --sample-id INT1 --n-steps 200 --checkpoint-dir /tmp/overfit_arch1
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from gen3_multiscale.data.dataset_manifest import load_dataset_manifest, save_dataset_manifest
from gen3_multiscale.training.train import run_training


def build_single_sample_config(config_path: str | Path, sample_id: str, n_steps: int, checkpoint_dir: str | Path) -> Path:
    """Derive a temp config + temp manifest restricted to one training
    sample and zero validation/test samples. Returns the temp config's
    path (the temp manifest it references lives alongside it)."""
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    manifest_path = (config.get("data") or {}).get("gen3_manifest_path")
    if not manifest_path:
        raise ValueError(f"{config_path}: data.gen3_manifest_path must be set to a real dataset manifest")
    manifest = load_dataset_manifest(manifest_path)
    if sample_id not in manifest.get("samples", {}):
        raise ValueError(f"{sample_id!r} is not a sample the dataset manifest at {manifest_path} declares")
    if int(n_steps) < 1:
        raise ValueError(f"--n-steps must be positive, got {n_steps}")

    overfit_manifest = dict(manifest)
    overfit_manifest["train_sample_ids"] = [sample_id]
    overfit_manifest["validation_sample_ids"] = []
    overfit_manifest["test_sample_ids"] = []

    tmp_dir = Path(tempfile.mkdtemp(prefix="gen3_step6_overfit_"))
    overfit_manifest_path = save_dataset_manifest(overfit_manifest, tmp_dir / "overfit_manifest.json")

    config["data"]["gen3_manifest_path"] = str(overfit_manifest_path)
    config["training"]["total_steps"] = int(n_steps)
    config["training"]["checkpoint_dir"] = str(checkpoint_dir)
    # No held-out samples exist in this derived manifest, so there is
    # nothing for periodic validation to evaluate against -- push it out
    # of reach rather than leaving a val_loader-is-None branch to be
    # silently skipped every step for no visible reason.
    config["training"]["eval_every_n_steps"] = int(n_steps) + 1
    overfit_config_path = tmp_dir / "overfit_config.yaml"
    OmegaConf.save(OmegaConf.create(config), overfit_config_path)
    return overfit_config_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="A real architectureN.yaml with data.gen3_manifest_path set")
    parser.add_argument("--sample-id", required=True, help="Exactly one manifest sample id to overfit on")
    parser.add_argument("--n-steps", type=int, default=200, help="Real optimizer steps (default: 200)")
    parser.add_argument("--checkpoint-dir", required=True, help="Fresh checkpoint dir for this overfit run")
    args = parser.parse_args()

    overfit_config_path = build_single_sample_config(args.config, args.sample_id, args.n_steps, args.checkpoint_dir)
    print(f"single-sample overfit config written to {overfit_config_path}")
    summary = run_training(str(overfit_config_path), smoke=False)
    print(f"overfit run finished: {summary}")


if __name__ == "__main__":
    main()
