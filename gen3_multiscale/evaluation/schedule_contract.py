"""Unambiguous metadata for final fixed-mask model evaluation.

Historically ``n_masks_per_sample`` was used for a value that is actually the
number of masks generated for *each stratum of each sample*.  With four strata,
8 means 32 evaluation items per sample, not 8.  Keep the old report key and CLI
alias for compatibility, but make the canonical semantics explicit.
"""
from __future__ import annotations


def fixed_mask_evaluation_metadata(
    *, split: str, sample_ids: list[str], strata: list[dict],
    masks_per_stratum_per_sample: int, actual_n_items: int,
) -> dict:
    if not sample_ids:
        raise ValueError("fixed-mask evaluation requires at least one sample")
    if not strata:
        raise ValueError("fixed-mask evaluation requires at least one stratum")
    count = int(masks_per_stratum_per_sample)
    if count < 1:
        raise ValueError("masks_per_stratum_per_sample must be positive")
    expected = len(sample_ids) * len(strata) * count
    if int(actual_n_items) != expected:
        raise ValueError(
            "fixed-mask evaluation item count violates the declared schedule: "
            f"actual={actual_n_items}, expected={len(sample_ids)} samples x "
            f"{len(strata)} strata x {count} masks = {expected}"
        )
    return {
        "evaluation_scope": "final_fixed_mask_evaluation",
        "split": str(split),
        "n_samples": int(len(sample_ids)),
        "n_mask_strata": int(len(strata)),
        "n_masks_per_stratum_per_sample": count,
        "n_items": expected,
        # Backward-compatible legacy field. Its historical name is misleading;
        # readers should use n_masks_per_stratum_per_sample above.
        "n_masks_per_sample": count,
        "n_masks_per_sample_semantics": "legacy_alias_for_per_stratum_per_sample",
    }
