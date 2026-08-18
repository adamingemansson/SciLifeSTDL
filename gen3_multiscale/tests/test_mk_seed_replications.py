import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from gen3_multiscale.evaluation.per_gene_diagnostics import METRIC_NAMES
from gen3_multiscale.scripts.analyze_mk_seed_gene_robustness import analyze as analyze_genes
from gen3_multiscale.scripts.summarize_mk_seed_replications import (
    ARCHITECTURES,
    PANELS,
    audit_report,
    summarize,
)


def _metric(value):
    return {"patient_mean": value, "patient_ci95_low": value - 0.01,
            "patient_ci95_high": value + 0.01, "n_patients": 14}


def _point(value):
    return {"pcc": value, "rmse": 1.0 - value, "auc": 0.8 + value / 10}


def _structured(value):
    return {
        "spot_profile": {"mean_spot_profile_pcc": value},
        "coexpression": {"correlation_matrix_pcc": value - 0.01},
        "spatial_ssim": {"mean_per_gene_ssim": value - 0.02},
        "moran_local": {"moran_i_pcc": value - 0.03},
        "gradient_local": {"signed_gradient_pcc": value - 0.04},
        "gradient_wide": {"signed_gradient_pcc": value - 0.05},
    }


def _report(
    tmp_path: Path, architecture: str, seed: int, value: float, *,
    legacy_patient_schema: bool = False,
) -> tuple[Path, dict]:
    config = tmp_path / f"{architecture}_seed{seed}.yaml"
    config.write_text(yaml.safe_dump({"training": {"seed": seed}}))
    sidecar = tmp_path / f"{architecture}_seed{seed}.npz"
    sidecar.touch()
    fixed_records = []
    for index in range(448):
        patient = f"p{index % 14}"
        fixed_records.append({
            "sample_id": f"s{index % 14}",
            "patient_id": None if legacy_patient_schema else patient,
            "model": _point(value),
            "gene_panels": {"model": {panel: _point(value) for panel in PANELS[1:]}},
        })
    whole_records = []
    for index in range(14):
        whole_records.append({
            "sample_id": f"s{index}",
            "patient_id": None if legacy_patient_schema else f"p{index}",
            "n_spots": 100 + index,
            "point_metrics": {panel: _point(value) for panel in PANELS},
            "structured_field": {"panels": {panel: _structured(value) for panel in PANELS}},
        })
    aggregate = {"pcc": _metric(value), "rmse": _metric(1.0 - value),
                 "auc": _metric(0.8 + value / 10)}
    structured_aggregate = {panel: _structured(value) for panel in PANELS}
    report = {
        "kind": "conditional_wae_supervisor_evaluation", "split": "validation",
        "config_path": str(config), "checkpoint_step": 1000 + seed,
        "checkpoint_masks_seen": 1000 + seed,
        "n_items": 448, "n_samples": 14, "n_mask_strata": 4,
        "n_masks_per_stratum_per_sample": 8,
        "query_gex_visible": False, "query_he_visible": True, "task": "he_to_st",
        "prediction_roles": {"primary_point_prediction": "model"},
        "per_arm_patient_aggregated_metrics": {"model": aggregate},
        "per_panel_patient_aggregated_metrics": {
            panel: {"model": aggregate} for panel in PANELS[1:]
        },
        "per_item_records": fixed_records,
        "whole_slide_structured_field_evaluation": {
            "scope": "all_held_out_slides_every_spot_exactly_once",
            "primary_prediction": "deterministic_h_and_e_point_prediction",
            "target_gex_visible_to_model": False,
            "per_gene_diagnostics_path": str(sidecar),
            "per_slide_records": whole_records,
            "point_metrics_patient_aggregated": {panel: aggregate for panel in PANELS},
            "structured_metrics_patient_aggregated": structured_aggregate,
        },
    }
    path = tmp_path / f"{architecture}_seed{seed}.json"
    path.write_text(json.dumps(report))
    return path, report


def _records(tmp_path: Path):
    result = {architecture: {} for architecture in ARCHITECTURES}
    for architecture in ARCHITECTURES:
        offset = 0.2 if architecture == ARCHITECTURES[0] else 0.1
        # The historical finalist used seed 10; "seed0 root" means the
        # reference run operationally, not literal training seed zero.
        for seed in (10, 1, 2):
            result[architecture][seed] = _report(
                tmp_path, architecture, seed, offset + seed * 0.01,
                legacy_patient_schema=seed == 10,
            )
    return result


def test_seed_summary_audits_and_separates_seed_and_patient_uncertainty(tmp_path: Path):
    records = _records(tmp_path)
    outputs = summarize(records, output_dir=tmp_path / "summary", n_bootstrap=200)
    assert all(path.is_file() for path in outputs.values())
    text = outputs["seed_delta"].read_text()
    assert "mk_wb_parallel_gated" in text
    # The synthetic architecture delta is +0.1 in every seed.
    assert "0.1" in text
    hierarchical = outputs["hierarchical_delta"].read_text()
    assert "held_out_slide" in hierarchical
    assert "\t14\t" in hierarchical


def test_seed_report_audit_rejects_incomplete_fixed_records(tmp_path: Path):
    path, report = _report(tmp_path, ARCHITECTURES[0], 0, 0.2)
    report["per_item_records"].pop()
    with pytest.raises(ValueError, match="fixed per-item record count"):
        audit_report(report, path, expected_seed=0)


def _write_sidecar(path: Path, architecture: str, seed: int) -> None:
    n_slides, n_genes = 14, 3
    target = np.tile(np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32), (n_slides, 1))
    arrays = {}
    for name in METRIC_NAMES:
        if name.startswith("target_"):
            arrays[name] = target.copy()
        elif name == "pcc":
            base = 0.2 if architecture == ARCHITECTURES[0] else 0.1
            arrays[name] = np.tile(
                np.asarray([[base + 0.01 * seed, base + 0.02, base + 0.03]], dtype=np.float32),
                (n_slides, 1),
            )
        else:
            arrays[name] = np.full((n_slides, n_genes), 0.5 + 0.01 * seed, dtype=np.float32)
    metadata = json.dumps({
        "kind": "hest_mk_per_gene_whole_slide_diagnostics", "version": 1,
        "n_slides": n_slides, "n_genes": n_genes,
        "provenance": {"method": architecture},
    })
    np.savez_compressed(
        path, metadata_json=np.asarray(metadata),
        gene_names=np.asarray(["g0", "g1", "g2"]),
        sample_ids=np.asarray([f"s{i}" for i in range(n_slides)]),
        patient_ids=np.asarray([f"p{i}" for i in range(n_slides)]),
        organs=np.asarray(["organ"] * n_slides), **arrays,
    )


def test_gene_robustness_requires_and_summarizes_three_seeds(tmp_path: Path):
    records = {architecture: {} for architecture in ARCHITECTURES}
    for architecture in ARCHITECTURES:
        for seed in (10, 1, 2):
            sidecar = tmp_path / f"{architecture}_{seed}.npz"
            _write_sidecar(sidecar, architecture, seed)
            records[architecture][seed] = (
                tmp_path / f"{architecture}_{seed}.json",
                {"whole_slide_structured_field_evaluation": {
                    "per_gene_diagnostics_path": str(sidecar),
                }},
            )
    noise_ceiling = tmp_path / "noise_ceiling.json"
    noise_ceiling.write_text(json.dumps({
        "kind": "gene_noise_ceiling_by_count_splitting",
        "per_slide": [
            {
                "sample_id": f"s{index}",
                "ceiling_by_gene": {"g0": 0.2, "g1": 0.5, "g2": 0.8},
            }
            for index in range(14)
        ],
    }))
    outputs = analyze_genes(
        records, output_dir=tmp_path / "genes",
        noise_ceiling_path=noise_ceiling,
    )
    assert all(path.is_file() for path in outputs.values())
    assert "fraction_genes_all_seeds_positive" in outputs["summary"].read_text()
    assert "mean_pcc_delta" in outputs["architecture_delta"].read_text()
    assert "noise_ceiling" in outputs["architecture_delta"].read_text().splitlines()[0]
    assert "target_moran_i" in outputs["per_gene"].read_text().splitlines()[0]
    assert "mean_pcc_delta" in outputs["delta_attribute_associations"].read_text()
    manifest = json.loads(outputs["manifest"].read_text())
    assert manifest["seeds"] == [1, 2, 10]
