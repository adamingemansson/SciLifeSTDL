from pathlib import Path

import numpy as np
import pytest

from gen3_multiscale.evaluation.per_gene_diagnostics import (
    PerGeneDiagnosticsAccumulator,
    load_per_gene_diagnostics,
    per_gene_whole_slide_diagnostics,
)
from gen3_multiscale.scripts.analyze_hest_mk_per_gene import analyze


def _arrays():
    coords = np.asarray([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]])
    target = np.asarray([
        [0.0, 1.0, 2.0], [0.2, 1.0, 1.5], [0.4, 1.0, 1.0],
        [0.6, 1.0, 0.5], [0.8, 1.0, 0.2], [1.0, 1.0, 0.0],
    ])
    return coords, target


def test_per_gene_diagnostics_recover_exact_prediction():
    coords, target = _arrays()
    values = per_gene_whole_slide_diagnostics(target, target, coords, local_k=2)
    assert values["pcc"][[0, 2]].tolist() == pytest.approx([1.0, 1.0])
    assert np.isnan(values["pcc"][1])
    assert values["rmse"].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert values["local_gradient_pcc"][[0, 2]].tolist() == pytest.approx([1.0, 1.0])


def _save(path: Path, *, prediction_offset: float = 0.0, target_offset: float = 0.0):
    coords, target = _arrays()
    target = target + target_offset
    accumulator = PerGeneDiagnosticsAccumulator(["g0", "g1", "g2"])
    accumulator.add_slide(
        sample_id="s0", patient_id="p0", organ="kidney",
        predicted=target + prediction_offset, target=target, coords=coords, local_k=2,
    )
    return accumulator.save(path, provenance={"method": path.stem})


def test_per_gene_sidecar_roundtrip_and_analysis_fail_closed(tmp_path: Path):
    first = _save(tmp_path / "first.npz")
    second = _save(tmp_path / "second.npz", prediction_offset=0.1)
    loaded = load_per_gene_diagnostics(first)
    assert loaded["sample_ids"] == ["s0"]
    assert loaded["pcc"].shape == (1, 3)

    outputs = analyze([first, second], output_dir=tmp_path / "analysis", top_n=1)
    assert all(path.is_file() for path in outputs.values())
    assert "fraction_gene_pcc_gt_0_1" in outputs["summary"].read_text().splitlines()[0]

    mismatched_target = _save(tmp_path / "mismatch.npz", target_offset=0.25)
    with pytest.raises(ValueError, match="exact paired comparison"):
        analyze([first, mismatched_target], output_dir=tmp_path / "rejected")
