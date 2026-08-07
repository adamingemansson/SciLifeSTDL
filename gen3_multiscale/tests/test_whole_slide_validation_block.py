"""Tests for prepare_conditional_wae_suite.whole_slide_validation_block --
the shared evaluation.whole_slide_validation config-block builder every
WAE-GAN ablation suite-prep script (uni2/coexpression/histology/...)
reuses so full-slide diagnostic validation stays byte-identical and
points at the SAME shared reference-projection basis across suites."""
from pathlib import Path

import pytest

from gen3_multiscale.scripts.prepare_conditional_wae_suite import whole_slide_validation_block


def test_block_is_enabled_and_covers_every_validation_sample():
    manifest = {"validation_sample_ids": ["V0", "V1", "V2"]}
    block = whole_slide_validation_block(Path("/tmp/suite_root"), manifest)
    assert block["enabled"] is True
    assert block["max_slides"] == 3


def test_block_rejects_a_manifest_with_zero_validation_samples():
    manifest = {"validation_sample_ids": []}
    with pytest.raises(ValueError, match="zero validation_sample_ids"):
        whole_slide_validation_block(Path("/tmp/suite_root"), manifest)


def test_block_points_at_one_shared_path_under_the_suite_roots_parent():
    manifest = {"validation_sample_ids": ["V0"]}
    block_a = whole_slide_validation_block(Path("/tmp/results/suite_a"), manifest)
    block_b = whole_slide_validation_block(Path("/tmp/results/suite_b"), manifest)
    assert block_a["reference_projection_path"] == block_b["reference_projection_path"]
    assert block_a["reference_projection_path"] == str(Path("/tmp/results/mk_wae_shared_reference_gex_projection"))


def test_block_respects_a_custom_every_n_evals():
    manifest = {"validation_sample_ids": ["V0"]}
    block = whole_slide_validation_block(Path("/tmp/suite_root"), manifest, every_n_evals=3)
    assert block["every_n_evals"] == 3


def test_block_caps_max_slides_to_one_per_organ_when_organ_info_is_present():
    """Logging every single validation slide is redundant within an organ
    and needlessly multiplies TensorBoard disk usage -- max_slides should
    cap to the number of distinct organs, not the raw slide count, once
    the manifest carries real organ labels (the trainer's slide-selection
    round-robins across organs so coverage is still complete)."""
    manifest = {
        "validation_sample_ids": ["V0", "V1", "V2", "V3", "V4"],
        "samples": {
            "V0": {"organ": "Kidney"}, "V1": {"organ": "Kidney"},
            "V2": {"organ": "Lung"}, "V3": {"organ": "Bowel"}, "V4": {"organ": "Bowel"},
        },
    }
    block = whole_slide_validation_block(Path("/tmp/suite_root"), manifest)
    assert block["max_slides"] == 3  # Kidney, Lung, Bowel


def test_block_never_caps_below_the_raw_slide_count_when_organs_exceed_slides():
    manifest = {
        "validation_sample_ids": ["V0"],
        "samples": {"V0": {"organ": "Kidney"}},
    }
    block = whole_slide_validation_block(Path("/tmp/suite_root"), manifest)
    assert block["max_slides"] == 1
