"""Run Architecture 3's two stages as ONE unsupervised job: Stage A
(denoising transcriptome autoencoder pretraining) followed automatically
by Stage B (spatial latent transformer), splitting a single total
wall-clock budget between them -- no manual "wait for Stage A, then
launch Stage B" step required.

Split rationale (2026-07-25): Stage A is genuinely cheap relative to
Stage B by design (no images, no k-NN neighborhood construction, no
transformer -- pure expression -> expression reconstruction on pooled
spots, see arch3_stage_a_pretrain.yaml's own comment: "expect this to
finish well within the combined 1-2 day budget for the pair"). Default
stage_a_fraction=0.2 gives Stage A a fifth of the total budget and
Stage B the rest; override if a specific run's real steps/sec ratio
looks different once you've watched a run or two.

Usage:
    python3 -m gen2_architectures.training.train_arch3_combined \\
        --stage_a_config gen2_architectures/configs/arch3_stage_a_pretrain.yaml \\
        --stage_b_config gen2_architectures/configs/arch3_stage_b_spatial.yaml \\
        --total_hours 36
"""
from __future__ import annotations

import argparse

from gen2_architectures.training import train_arch3_stage_a, train_arch3_stage_b


def main(
    stage_a_config: str, stage_b_config: str, total_hours: float,
    stage_a_fraction: float = 0.2, smoke_steps: int | None = None,
    skip_final_eval: bool = False,
) -> None:
    if not 0.0 < stage_a_fraction < 1.0:
        raise ValueError(f"stage_a_fraction must be in (0, 1), got {stage_a_fraction}")
    stage_a_hours = total_hours * stage_a_fraction
    stage_b_hours = total_hours * (1.0 - stage_a_fraction)

    print(f"=== Architecture 3 combined run: {total_hours:.1f}h total "
          f"(Stage A: {stage_a_hours:.1f}h, Stage B: {stage_b_hours:.1f}h) ===")
    print("=== Starting Stage A ===")
    train_arch3_stage_a.main(stage_a_config, smoke_steps=smoke_steps, max_wall_clock_hours_override=stage_a_hours)
    print("=== Stage A done -- starting Stage B ===")
    train_arch3_stage_b.main(
        stage_b_config, smoke_steps=smoke_steps, max_wall_clock_hours_override=stage_b_hours,
        skip_final_eval=skip_final_eval,
    )
    print("=== Architecture 3 combined run complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage_a_config", type=str, required=True)
    parser.add_argument("--stage_b_config", type=str, required=True)
    parser.add_argument("--total_hours", type=float, required=True,
                         help="Total wall-clock budget for the combined run, split between "
                              "the two stages via --stage_a_fraction.")
    parser.add_argument("--stage_a_fraction", type=float, default=0.2,
                         help="Fraction of --total_hours given to Stage A (default 0.2 -- "
                              "see this module's own docstring for the reasoning).")
    parser.add_argument("--smoke_steps", type=int, default=None,
                         help="Passed through to BOTH stages -- for a quick end-to-end smoke "
                              "test of the whole combined pipeline before a real run.")
    parser.add_argument("--skip_final_eval", action="store_true",
                         help="Passed through to Stage B -- skip the final held-out test "
                              "evaluation entirely (the checkpoint is still saved).")
    args = parser.parse_args()
    main(
        args.stage_a_config, args.stage_b_config, args.total_hours, args.stage_a_fraction, args.smoke_steps,
        skip_final_eval=args.skip_final_eval,
    )
