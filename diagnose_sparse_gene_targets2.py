#!/usr/bin/env python3
"""Per-sample normalized marker-gene levels, with depth and tech, so a
depth artifact can be told apart from an organ-label problem."""
from __future__ import annotations
import argparse, sys
import numpy as np
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.data.loaders import load_hest_sample, basic_qc_and_normalize


def _records(m):
    r = m.get("samples")
    if isinstance(r, dict):
        return list(r.items())
    if isinstance(r, list):
        return [(str(x.get("sample_id") or x.get("id")), x)
                for x in r if isinstance(x, dict) and (x.get("sample_id") or x.get("id"))]
    return []


def _by(m, field):
    return {s: str(r[field]) for s, r in _records(m)
            if isinstance(r, dict) and r.get(field) is not None}


def _vals(adata, gene):
    names = np.asarray(adata.var_names, dtype=str)
    hit = np.flatnonzero(names == gene)
    if hit.size == 0:
        return None
    col = adata.X[:, hit[0]]
    if hasattr(col, "toarray"):
        col = col.toarray()
    return np.asarray(col, dtype=np.float64).ravel()


def _total_counts(adata):
    X = adata.X
    tot = X.sum(axis=1)
    return np.asarray(tot).ravel()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--hest-data-dir", required=True)
    p.add_argument("--genes", default="UMOD,LCN2")
    p.add_argument("--samples", default="", help="comma list; default = kidney train + all-organ stride")
    p.add_argument("--max-per-group", type=int, default=10)
    a = p.parse_args()

    genes = [g.strip() for g in a.genes.split(",") if g.strip()]
    m = load_dataset_manifest(a.manifest)
    cfg = m.get("config", {})
    transform = cfg.get("expression_transform", "normalize_log1p")
    target_sum = float(cfg.get("expression_target_sum", 1e4) or 1e4)
    min_genes = int(cfg.get("gene_min_genes_per_spot", 200) or 200)
    print(f"manifest transform={transform!r} target_sum={target_sum} min_genes={min_genes}\n")

    organ_by, tech_by = _by(m, "organ"), _by(m, "tech")
    train = [str(s) for s in m["train_sample_ids"]]

    if a.samples.strip():
        chosen = [s.strip() for s in a.samples.split(",") if s.strip()]
    else:
        kidney = [s for s in train if organ_by.get(s) == "Kidney"][: a.max_per_group]
        stride = max(1, len(train) // max(1, a.max_per_group))
        chosen = list(dict.fromkeys(kidney + train[::stride][: a.max_per_group] + ["INT14"]))

    header = f"{'sample':<10} {'organ':<10} {'tech':<12} {'spots':>6} {'medUMI':>8}"
    for g in genes:
        header += f" | {g+' mean':>10} {g+' frac>0':>11} {g+' max':>8}"
    print(header)
    print("-" * len(header))

    for sid in chosen:
        try:
            raw = load_hest_sample(a.hest_data_dir, sid,
                                   organ=organ_by.get(sid), tech=tech_by.get(sid))
        except Exception as e:
            print(f"{sid:<10} (skipped: {type(e).__name__}: {e})")
            continue
        med_umi = float(np.median(_total_counts(raw)))
        adata = basic_qc_and_normalize(raw, min_genes=min_genes, min_cells=0,
                                       transform=transform, target_sum=target_sum)
        row = (f"{sid:<10} {str(organ_by.get(sid)):<10} {str(tech_by.get(sid)):<12} "
               f"{adata.n_obs:>6} {med_umi:>8.0f}")
        for g in genes:
            v = _vals(adata, g)
            if v is None:
                row += f" | {'ABSENT':>10} {'-':>11} {'-':>8}"
            else:
                row += (f" | {v.mean():>10.4f} {float(np.mean(v > 1e-8)):>11.4f} "
                        f"{v.max():>8.3f}")
        print(row)

    print("\nRead the medUMI column first: if the Kidney rows are far shallower than the")
    print("others, the earlier raw-count gap is a depth artifact and normalization handles")
    print("it. If depth is comparable but Kidney rows still show near-zero UMOD, the organ")
    print("labels do not match the tissue -- a data problem upstream of every model so far.")


if __name__ == "__main__":
    main()
