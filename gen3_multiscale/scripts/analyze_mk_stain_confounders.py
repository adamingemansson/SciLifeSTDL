#!/usr/bin/env python3
"""Interpret the MK stain-distance association without treating correlation as causation.

This post-hoc diagnostic consumes an already completed stain analysis.  It
reports the raw association, an organ-residualized association, within-organ
pair concordance, and associations with existing spatial/template diagnostics.
No image or expression data are loaded.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, pearsonr, rankdata, spearmanr

from gen3_multiscale.config_identity import resolved_config
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"empty table: {path}")
    return rows


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _rho(left: np.ndarray, right: np.ndarray) -> tuple[float, float, int]:
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3 or np.std(left[finite]) <= 1e-12 or np.std(right[finite]) <= 1e-12:
        return float("nan"), float("nan"), int(finite.sum())
    result = spearmanr(left[finite], right[finite])
    return float(result.statistic), float(result.pvalue), int(finite.sum())


def _pearson(left: np.ndarray, right: np.ndarray) -> tuple[float, float, int]:
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3 or np.std(left[finite]) <= 1e-12 or np.std(right[finite]) <= 1e-12:
        return float("nan"), float("nan"), int(finite.sum())
    result = pearsonr(left[finite], right[finite])
    return float(result.statistic), float(result.pvalue), int(finite.sum())


def _residualize_categories(values: np.ndarray, categories: list[str]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(categories),):
        raise ValueError("values and categories differ")
    levels = sorted(set(categories))
    design = np.ones((len(values), 1 + max(len(levels) - 1, 0)), dtype=np.float64)
    for column, level in enumerate(levels[1:], start=1):
        design[:, column] = np.asarray([value == level for value in categories], dtype=np.float64)
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    return values - design @ coefficients


def _partial_rank(left: np.ndarray, right: np.ndarray, covariate: np.ndarray) -> tuple[float, float, int]:
    finite = np.isfinite(left) & np.isfinite(right) & np.isfinite(covariate)
    if finite.sum() < 4:
        return float("nan"), float("nan"), int(finite.sum())
    design = np.column_stack([
        np.ones(int(finite.sum())), rankdata(covariate[finite]),
    ])
    x = rankdata(left[finite]).astype(np.float64)
    y = rankdata(right[finite]).astype(np.float64)
    x -= design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y -= design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return _pearson(x, y)


def _organ_pair_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_organ: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_organ.setdefault(str(row["organ"]), []).append(row)
    comparable = 0
    concordant = 0
    details = []
    for organ, selected in sorted(by_organ.items()):
        if len(selected) != 2:
            continue
        selected = sorted(selected, key=lambda row: float(row["stain_distance"]))
        low, high = selected
        delta_distance = float(high["stain_distance"]) - float(low["stain_distance"])
        delta_pcc = float(high["all_gene_pcc"]) - float(low["all_gene_pcc"])
        if abs(delta_distance) <= 1e-12 or abs(delta_pcc) <= 1e-12:
            call = "tie"
        else:
            comparable += 1
            is_concordant = delta_distance * delta_pcc > 0
            concordant += int(is_concordant)
            call = "concordant" if is_concordant else "discordant"
        details.append({
            "organ": organ,
            "lower_distance_slide": low["sample_id"],
            "higher_distance_slide": high["sample_id"],
            "delta_stain_distance": delta_distance,
            "delta_pcc": delta_pcc,
            "call": call,
        })
    p_value = float(binomtest(concordant, comparable, 0.5).pvalue) if comparable else float("nan")
    return {
        "n_comparable_organ_pairs": comparable,
        "n_concordant_pairs": concordant,
        "fraction_concordant": concordant / comparable if comparable else float("nan"),
        "two_sided_sign_test_p": p_value,
        "details": details,
    }


def analyze(
    *, stain_table: str, output_dir: str, config_path: str | None = None,
    template_diagnostics: str | None = None,
    per_gene_template_diagnostics: str | None = None,
    noise_ceiling_path: str | None = None,
) -> dict[str, Path]:
    rows: list[dict[str, Any]] = _read(Path(stain_table).expanduser().resolve())
    required = {"sample_id", "organ", "stain_distance", "all_gene_pcc", "all_gene_rmse"}
    if not required.issubset(rows[0]):
        raise ValueError(f"stain table lacks required fields: {sorted(required - set(rows[0]))}")
    if len(rows) != 14 or len({row["sample_id"] for row in rows}) != 14:
        raise ValueError("the final MK stain audit requires exactly 14 unique held-out slides")

    if config_path:
        config = resolved_config(config_path)
        manifest = load_dataset_manifest(config["data"]["gen3_manifest_path"])
        if set(manifest["validation_sample_ids"]) != {row["sample_id"] for row in rows}:
            raise ValueError("stain table and configured validation slides differ")
        for row in rows:
            record = manifest["samples"][row["sample_id"]]
            row["technology"] = str(record.get("tech") or record.get("st_technology") or "unknown")

    if template_diagnostics:
        template = {row["sample_id"]: row for row in _read(Path(template_diagnostics).expanduser().resolve())}
        if set(template) != {row["sample_id"] for row in rows}:
            raise ValueError("stain and template diagnostic slide identities differ")
        for row in rows:
            for key, value in template[row["sample_id"]].items():
                if key not in {"sample_id", "organ"}:
                    try:
                        row[f"template_{key}"] = float(value)
                    except ValueError:
                        pass

    if per_gene_template_diagnostics:
        per_gene = _read(Path(per_gene_template_diagnostics).expanduser().resolve())
        by_slide: dict[str, list[float]] = {}
        for row in per_gene:
            try:
                by_slide.setdefault(row["sample_id"], []).append(float(row["target_std"]))
            except (KeyError, ValueError):
                continue
        if set(by_slide) != {row["sample_id"] for row in rows}:
            raise ValueError("per-gene template diagnostics do not cover the stain slides")
        for row in rows:
            values = np.asarray(by_slide[row["sample_id"]], dtype=np.float64)
            values = values[np.isfinite(values)]
            row["target_gene_std_mean"] = float(np.mean(values))
            row["target_gene_std_median"] = float(np.median(values))

    if noise_ceiling_path:
        payload = json.loads(Path(noise_ceiling_path).expanduser().resolve().read_text())
        if payload.get("kind") != "gene_noise_ceiling_by_count_splitting":
            raise ValueError("not a count-split noise-ceiling artifact")
        ceiling = {str(row["sample_id"]): row for row in payload.get("per_slide", [])}
        if set(ceiling) != {row["sample_id"] for row in rows}:
            raise ValueError("noise-ceiling artifact does not cover exactly the stain slides")
        for row in rows:
            source = ceiling[row["sample_id"]]
            row["noise_ceiling_all_genes"] = float(source["mean_ceiling_all_genes"])
            row["noise_ceiling_unmeasurable_fraction"] = float(
                source["fraction_genes_ceiling_below_0.1"]
            )

    distance = np.asarray([float(row["stain_distance"]) for row in rows])
    pcc = np.asarray([float(row["all_gene_pcc"]) for row in rows])
    rmse = np.asarray([float(row["all_gene_rmse"]) for row in rows])
    organ = [str(row["organ"]) for row in rows]
    association_rows = []
    for scope, method, left, right, outcome in (
        ("raw", "spearman", distance, pcc, "all_gene_pcc"),
        ("raw", "spearman", distance, rmse, "all_gene_rmse"),
        (
            "within_organ_residualized", "partial_spearman", _residualize_categories(rankdata(distance), organ),
            _residualize_categories(rankdata(pcc), organ), "all_gene_pcc",
        ),
        (
            "within_organ_residualized", "partial_spearman", _residualize_categories(rankdata(distance), organ),
            _residualize_categories(rankdata(rmse), organ), "all_gene_rmse",
        ),
    ):
        statistic = _rho if method == "spearman" else _pearson
        rho, p_value, n = statistic(np.asarray(left), np.asarray(right))
        association_rows.append({
            "scope": scope, "method": method, "outcome": outcome, "association": rho,
            "p_value": p_value, "n_slides": n,
        })

    if rows[0].get("technology") and len({row["technology"] for row in rows}) > 1:
        technology = [str(row["technology"]) for row in rows]
        rho, p_value, n = _pearson(
            _residualize_categories(rankdata(distance), technology),
            _residualize_categories(rankdata(pcc), technology),
        )
        association_rows.append({
            "scope": "within_technology_residualized", "method": "partial_spearman",
            "outcome": "all_gene_pcc", "association": rho,
            "p_value": p_value, "n_slides": n,
        })

    excluded = {
        "stain_distance", "all_gene_pcc", "all_gene_rmse", "n_spots", "n_genes",
    }
    for key in sorted(rows[0]):
        if not (
            key.startswith("template_") or key.startswith("target_")
            or key.startswith("noise_ceiling_")
        ) or key in excluded:
            continue
        try:
            values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        except (KeyError, ValueError):
            continue
        rho, p_value, n = _rho(distance, values)
        association_rows.append({
            "scope": "raw_potential_confounder", "method": "spearman", "outcome": key,
            "association": rho, "p_value": p_value, "n_slides": n,
        })
        partial, partial_p, partial_n = _partial_rank(distance, pcc, values)
        association_rows.append({
            "scope": f"pcc_adjusted_for_{key}", "method": "partial_spearman",
            "outcome": "all_gene_pcc", "association": partial,
            "p_value": partial_p, "n_slides": partial_n,
        })

    pairs = _organ_pair_summary(rows)
    raw_pcc = next(
        row for row in association_rows
        if row["scope"] == "raw" and row["outcome"] == "all_gene_pcc"
    )
    harmful_gate = len(rows) >= 14 and float(raw_pcc["association"]) <= -0.30
    interpretation = (
        "harmful_stain_shift_gate_passed"
        if harmful_gate else
        "do_not_trigger_stain_normalization_from_this_diagnostic"
    )
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    associations_path = root / "stain_confounder_associations.tsv"
    pairs_path = root / "within_organ_stain_pairs.tsv"
    _write(associations_path, association_rows)
    _write(pairs_path, pairs["details"])
    decision_path = root / "stain_decision.json"
    decision_path.write_text(json.dumps({
        "kind": "mk_stain_confounder_diagnostic",
        "n_slides": len(rows),
        "directional_gate": "trigger only when stain-distance versus PCC rho <= -0.30",
        "raw_stain_distance_vs_pcc": raw_pcc,
        "normalization_gate_passed": harmful_gate,
        "interpretation": interpretation,
        "within_organ_pairs": {key: value for key, value in pairs.items() if key != "details"},
        "warning": "Associations are descriptive; n=14 is too small for a causal stain-normalization claim.",
    }, indent=2, allow_nan=False) + "\n")
    return {"associations": associations_path, "pairs": pairs_path, "decision": decision_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stain-table", required=True)
    parser.add_argument("--config")
    parser.add_argument("--template-diagnostics")
    parser.add_argument("--per-gene-template-diagnostics")
    parser.add_argument("--noise-ceiling")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    outputs = analyze(
        stain_table=args.stain_table, config_path=args.config,
        template_diagnostics=args.template_diagnostics,
        per_gene_template_diagnostics=args.per_gene_template_diagnostics,
        noise_ceiling_path=args.noise_ceiling,
        output_dir=args.output_dir,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
