import pytest

from gen3_multiscale.evaluation.schedule_contract import fixed_mask_evaluation_metadata


def test_expanded_final_evaluation_resolves_to_448_items():
    report = fixed_mask_evaluation_metadata(
        split="validation",
        sample_ids=[f"s{i}" for i in range(14)],
        strata=[{"name": f"q{i}"} for i in range(4)],
        masks_per_stratum_per_sample=8,
        actual_n_items=448,
    )
    assert report["evaluation_scope"] == "final_fixed_mask_evaluation"
    assert report["n_masks_per_stratum_per_sample"] == 8
    assert report["n_masks_per_sample"] == 8
    assert report["n_items"] == 448


def test_training_validation_example_is_not_mislabeled_as_448():
    report = fixed_mask_evaluation_metadata(
        split="validation",
        sample_ids=[f"s{i}" for i in range(14)],
        strata=[{"name": f"q{i}"} for i in range(4)],
        masks_per_stratum_per_sample=4,
        actual_n_items=224,
    )
    assert report["n_items"] == 224


def test_schedule_count_mismatch_fails_closed():
    with pytest.raises(ValueError, match="actual=224.*expected=.*448"):
        fixed_mask_evaluation_metadata(
            split="validation", sample_ids=[f"s{i}" for i in range(14)],
            strata=[{"name": f"q{i}"} for i in range(4)],
            masks_per_stratum_per_sample=8, actual_n_items=224,
        )
