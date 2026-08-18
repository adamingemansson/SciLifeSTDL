import csv
from pathlib import Path

import pytest

from gen3_multiscale.scripts.build_mk_gene_predictability_atlas import build_atlas


def _write_seed_table(path: Path) -> None:
    rows = []
    values = {
        "left": {"g0": (0.30, 0.28, 0.31), "g1": (0.04, 0.05, 0.03), "g2": (0.10, 0.11, 0.09)},
        "right": {"g0": (0.20, 0.19, 0.21), "g1": (0.08, 0.09, 0.07), "g2": (0.10, 0.10, 0.10)},
    }
    for architecture, by_gene in values.items():
        for index, (gene, seeds) in enumerate(by_gene.items()):
            mean = sum(seeds) / len(seeds)
            rows.append({
                "architecture": architecture,
                "gene": gene,
                "pcc_mean": mean,
                "pcc_seed_sd": 0.01,
                "all_seeds_pcc_positive": True,
                "seed1_pcc": seeds[0],
                "seed2_pcc": seeds[1],
                "seed10_pcc": seeds[2],
                "spearman_mean": mean,
                "r2_mean": mean - 0.1,
                "rmse_mean": 1.0 - mean,
                "mae_mean": 0.8 - mean,
                "local_gradient_pcc_mean": mean / 2,
                "target_mean": 0.1 + index,
                "target_std": 0.2 + index,
                "target_nonzero_fraction": 0.3 + index / 10,
                "target_moran_i": 0.4 + index / 10,
                "target_local_gradient_energy": 0.5 + index,
                "noise_ceiling": (0.5, 0.05, 0.25)[index],
            })
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _rows(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_predictability_atlas_reports_ceiling_and_complementarity(tmp_path: Path):
    source = tmp_path / "seed_genes.tsv"
    _write_seed_table(source)
    outputs = build_atlas(
        source,
        output_dir=tmp_path / "atlas",
        architectures=("left", "right"),
        ceiling_threshold=0.1,
        meaningful_delta=0.02,
        top_n=2,
    )

    assert all(path.is_file() for path in outputs.values())
    atlas = _rows(outputs["atlas"])
    left_g0 = next(row for row in atlas if row["architecture"] == "left" and row["gene"] == "g0")
    left_g1 = next(row for row in atlas if row["architecture"] == "left" and row["gene"] == "g1")
    assert float(left_g0["pcc_over_noise_ceiling"]) == pytest.approx(0.5933333333)
    assert left_g1["ceiling_eligible"] == "False"
    assert left_g1["pcc_over_noise_ceiling"].lower() == "nan"

    comparison = {row["gene"]: row for row in _rows(outputs["architecture_comparison"])}
    assert comparison["g0"]["stable_left_advantage"] == "True"
    assert comparison["g1"]["stable_right_advantage"] == "True"
    assert comparison["g2"]["stable_left_advantage"] == "False"

    summary = _rows(outputs["complementarity"])[0]
    assert int(summary["n_genes"]) == 3
    assert float(summary["oracle_gain_over_best_single"]) > 0


def test_predictability_atlas_rejects_different_gene_sets(tmp_path: Path):
    source = tmp_path / "seed_genes.tsv"
    _write_seed_table(source)
    rows = _rows(source)
    rows = [row for row in rows if not (row["architecture"] == "right" and row["gene"] == "g2")]
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="gene vocabularies differ"):
        build_atlas(source, output_dir=tmp_path / "atlas")
