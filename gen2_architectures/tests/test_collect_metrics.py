import json
from pathlib import Path

from gen2_architectures.evaluation.collect_metrics import collect


def _write_sample(path: Path, pcc: float, rmse: float, extra: dict | None = None) -> None:
    summary = {"pcc": {"mean": pcc, "std": 0.01, "n": 16}, "rmse": {"mean": rmse, "std": 0.01, "n": 16}}
    if extra:
        summary.update(extra)
    payload = {
        "primary_image_mode": "target_zero",
        "image_modes": {"target_zero": {"summary": summary}},
    }
    path.write_text(json.dumps(payload))


def test_collect_averages_per_sample_means_within_one_architecture(tmp_path):
    arch1 = tmp_path / "arch1_gpt_baseline"
    arch1.mkdir()
    _write_sample(arch1 / "audit_test_metrics_sampleA.json", pcc=0.5, rmse=1.0)
    _write_sample(arch1 / "audit_test_metrics_sampleB.json", pcc=0.7, rmse=0.8)

    rows = collect(tmp_path)
    assert len(rows) == 1
    assert rows[0]["architecture"] == "arch1_gpt_baseline"
    assert rows[0]["n_test_samples"] == 2
    assert abs(rows[0]["pcc"] - 0.6) < 1e-9
    assert abs(rows[0]["rmse"] - 0.9) < 1e-9


def test_collect_includes_fixed_gene_panel_metric_only_where_present(tmp_path):
    arch4 = tmp_path / "arch4_stpath_hybrid"
    arch4.mkdir()
    _write_sample(
        arch4 / "audit_test_metrics_sample1.json", pcc=0.4, rmse=1.1,
        extra={"pcc_lung_hest_bench_50": {"mean": 0.6, "std": 0.02, "n": 8}},
    )
    arch1 = tmp_path / "arch1_gpt_baseline"
    arch1.mkdir()
    _write_sample(arch1 / "audit_test_metrics_sample1.json", pcc=0.5, rmse=1.0)

    rows = collect(tmp_path)
    by_arch = {row["architecture"]: row for row in rows}
    assert "pcc_lung_hest_bench_50" in by_arch["arch4_stpath_hybrid"]
    assert by_arch["arch4_stpath_hybrid"]["pcc_lung_hest_bench_50"] == 0.6
    assert "pcc_lung_hest_bench_50" not in by_arch["arch1_gpt_baseline"]


def test_directories_with_no_metrics_files_are_skipped(tmp_path):
    (tmp_path / "empty_dir").mkdir()
    assert collect(tmp_path) == []


def test_unreadable_json_is_skipped_not_fatal(tmp_path):
    arch1 = tmp_path / "arch1_gpt_baseline"
    arch1.mkdir()
    (arch1 / "audit_test_metrics_broken.json").write_text("{not valid json")
    _write_sample(arch1 / "audit_test_metrics_ok.json", pcc=0.5, rmse=1.0)

    rows = collect(tmp_path)
    assert len(rows) == 1
    assert rows[0]["n_test_samples"] == 1
