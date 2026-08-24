#!/usr/bin/env python3
"""Fail-closed readiness audit for a proposed external MK validation cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


REQUIRED_FIELDS = (
    "dataset_id", "sample_ids", "patient_ids", "organs", "spatial_technology",
    "target_space", "gene_coverage_fraction", "patient_disjoint_from_training",
    "used_for_model_selection", "pretraining_overlap_status", "data_provenance",
)


def audit(candidate: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    missing = [name for name in REQUIRED_FIELDS if name not in candidate]
    failures.extend(f"missing required field: {name}" for name in missing)
    if missing:
        return {"ready": False, "failures": failures}
    sample_ids = list(map(str, candidate["sample_ids"]))
    patient_ids = list(map(str, candidate["patient_ids"]))
    if not sample_ids or len(set(sample_ids)) != len(sample_ids):
        failures.append("sample_ids must be non-empty and unique")
    if not patient_ids:
        failures.append("patient_ids must be non-empty")
    if not bool(candidate["patient_disjoint_from_training"]):
        failures.append("external patients are not explicitly disjoint from training")
    if bool(candidate["used_for_model_selection"]):
        failures.append("external cohort was used for model selection")
    external = contract["external_validation"]
    if str(candidate["target_space"]) not in set(external["accepted_target_spaces"]):
        failures.append(f"unsupported target_space: {candidate['target_space']}")
    if str(candidate["spatial_technology"]) not in set(external["accepted_spatial_technologies"]):
        failures.append(f"unsupported spatial_technology: {candidate['spatial_technology']}")
    if float(candidate["gene_coverage_fraction"]) < float(external["require_gene_coverage_fraction"]):
        failures.append("gene coverage is below the contract threshold")
    if str(candidate["pretraining_overlap_status"]) not in {
        "none", "audited_none", "possible", "known_overlap",
    }:
        failures.append("pretraining_overlap_status is not a recognized audited value")
    if not str(candidate["data_provenance"]).strip():
        failures.append("data_provenance is empty")
    return {
        "ready": not failures,
        "failures": failures,
        "warnings": (
            ["foundation-model pretraining overlap is possible and must be disclosed"]
            if candidate["pretraining_overlap_status"] in {"possible", "known_overlap"}
            else []
        ),
        "n_samples": len(sample_ids),
        "n_patients": len(set(patient_ids)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--study-contract", default="configs/benchmarks/mk_final_study_2026.yaml"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    candidate_path = Path(args.candidate).expanduser().resolve()
    contract_path = Path(args.study_contract).expanduser().resolve()
    candidate = yaml.safe_load(candidate_path.read_text())
    contract = yaml.safe_load(contract_path.read_text())
    result = {
        "kind": "mk_external_validation_readiness_audit",
        "candidate": str(candidate_path),
        "study_contract": str(contract_path),
        **audit(candidate, contract),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ready"] else 2)


if __name__ == "__main__":
    main()
