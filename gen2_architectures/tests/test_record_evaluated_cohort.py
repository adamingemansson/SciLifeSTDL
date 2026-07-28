"""GPT-audit-flagged bug (2026-07-27, second-pass re-audit, confirmed and
fixed): load_held_out_samples_with_images silently drops any held-out
sample that doesn't cover the exact training gene panel (a deliberate
design, not itself a bug -- see that function's own docstring). But the
pinned split manifest still lists the ORIGINAL, pre-drop cohort, so
nothing durably recorded when the actually-evaluated cohort ends up
smaller than declared -- only a print statement easy to miss."""
import json
import tempfile
from pathlib import Path

from gen2_architectures.training.data_prep import record_evaluated_cohort


def test_records_intended_evaluated_and_dropped_ids():
    with tempfile.TemporaryDirectory() as tmp:
        record_evaluated_cohort(tmp, "test", intended_ids=["a", "b", "c"], kept_ids=["a", "c"])

        record = json.loads((Path(tmp) / "evaluated_cohort.json").read_text())
        assert record["test"] == {
            "intended_sample_ids": ["a", "b", "c"],
            "evaluated_sample_ids": ["a", "c"],
            "dropped_sample_ids": ["b"],
        }


def test_no_drops_gives_an_empty_dropped_list():
    with tempfile.TemporaryDirectory() as tmp:
        record_evaluated_cohort(tmp, "validation", intended_ids=["x", "y"], kept_ids=["x", "y"])

        record = json.loads((Path(tmp) / "evaluated_cohort.json").read_text())
        assert record["validation"]["dropped_sample_ids"] == []


def test_recording_one_split_does_not_clobber_another_splits_prior_record():
    with tempfile.TemporaryDirectory() as tmp:
        record_evaluated_cohort(tmp, "validation", intended_ids=["v1"], kept_ids=["v1"])
        record_evaluated_cohort(tmp, "test", intended_ids=["t1", "t2"], kept_ids=["t1"])

        record = json.loads((Path(tmp) / "evaluated_cohort.json").read_text())
        assert record["validation"]["evaluated_sample_ids"] == ["v1"]
        assert record["test"]["dropped_sample_ids"] == ["t2"]
