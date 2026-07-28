"""GPT-audit-flagged bug (2026-07-27, confirmed and fixed): mask-bank query
spots are not required to be disjoint across masks, but evaluation reported
per-mask summary stats as if they were independent replicates."""
from gen3_multiscale.data.mask_bank import query_overlap_report


def _record(query_names):
    return {"query_obs_names": list(query_names)}


def test_disjoint_masks_report_zero_overlap_and_full_unique_coverage():
    records = [_record(["a", "b"]), _record(["c", "d", "e"])]

    report = query_overlap_report(records)

    assert report["n_masks"] == 2
    assert report["total_query_spot_draws"] == 5
    assert report["unique_query_spots"] == 5
    assert report["max_pairwise_overlap_fraction"] == 0.0


def test_fully_overlapping_masks_report_full_overlap_and_reduced_unique_coverage():
    records = [_record(["a", "b", "c"]), _record(["a", "b", "c"])]

    report = query_overlap_report(records)

    assert report["total_query_spot_draws"] == 6
    assert report["unique_query_spots"] == 3
    assert report["max_pairwise_overlap_fraction"] == 1.0


def test_partial_overlap_reports_the_fraction_relative_to_the_smaller_mask():
    records = [_record(["a", "b"]), _record(["b", "c", "d"])]

    report = query_overlap_report(records)

    # smaller mask has 2 spots, 1 of which ("b") also appears in the other
    assert report["max_pairwise_overlap_fraction"] == 0.5
    assert report["unique_query_spots"] == 4


def test_empty_records_do_not_crash():
    report = query_overlap_report([])

    assert report["n_masks"] == 0
    assert report["unique_query_spots"] == 0
    assert report["max_pairwise_overlap_fraction"] == 0.0
