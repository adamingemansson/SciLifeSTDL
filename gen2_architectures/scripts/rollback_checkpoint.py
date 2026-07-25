"""Operator CLI for checkpoint history: list what's kept, and roll a
checkpoint_dir's root ("latest", what every training entrypoint's resume
logic reads) back to an earlier known-good step.

Use this if a run's loss/PCC curve or the training diagnostics (see
gen2_architectures/README.md's monitoring section) show a genuine
divergence or corruption after a later checkpoint save -- rolling back
means the NEXT time you launch that training script, it resumes from the
older step instead of the diverged one. This does not touch anything
outside checkpoint_dir/ (does not affect other architectures' runs, does
not delete history it isn't asked to).

Usage:
    python3 -m gen2_architectures.scripts.rollback_checkpoint \
        --checkpoint_dir gen2_architectures/results/arch1_gpt_baseline --list

    python3 -m gen2_architectures.scripts.rollback_checkpoint \
        --checkpoint_dir gen2_architectures/results/arch1_gpt_baseline --step 42000
"""
from __future__ import annotations

import argparse

from gen2_architectures.training.checkpoint import list_checkpoint_history, load_training_state, rollback_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--list", action="store_true", help="print available history steps and exit")
    parser.add_argument("--step", type=int, default=None, help="roll checkpoint_dir back to this step")
    args = parser.parse_args()

    current_step = load_training_state(args.checkpoint_dir).get("step", 0)
    available = list_checkpoint_history(args.checkpoint_dir)
    print(f"{args.checkpoint_dir}: current root step={current_step}, history steps={available}")

    if args.list:
        return
    if args.step is None:
        parser.error("pass --step STEP to roll back, or --list to just see what's available")
    rollback_checkpoint(args.checkpoint_dir, args.step)


if __name__ == "__main__":
    main()
