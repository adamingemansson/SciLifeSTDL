#!/usr/bin/env python3
"""Is the model emitting the marginal mean for sparse marker genes?"""
from __future__ import annotations
import argparse, sys
from collections import defaultdict
import numpy as np
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.data.loaders import load_hest_sample


def _records(manifest):
    records = manifest.get("samples")
    if isinstance(records, dict):
        return list(records.items())
    if isinstance(records, list):
        out = []
        for r in records:
            if isinstance(r, dict):
                sid = r.get("sample_id") or r.get("id")
                if sid is not None:
                    out.append((str(sid), r))
        return out
    return []


def _field_by_sample(manifest, field):
    return {
        sid: str(r[field])
        for sid, r in _records(manifest)
        if isinstance(r, dict) and r.get(field) is not None
    }


def _gene_values(adata, gene):
    names = np.asarray(adata.var_names, dtype=str)
    hits = np.flatnonzero(names == gene)
    if hits.size == 0:
        return None
    col = adata.X[:, hits[0]]
    if hasattr(col, "toarray"):
        col = col.toarray()
    return np.asarray(col, dtype=np.float64).ravel()


def _describe(v):
    return (f"n={v.size:>7d}  mean={v.mean():.4f}  median={np.median(v):.4f}  "
            f"max={v.max():.4f}  frac>0={float(np.mean(v > 1e-8)):.4f}  "
            f"p90={np.percentile(v,90):.4f}  p99={np.percentile(v,99):.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--hest-data-dir", required=True)
    p.add_argument("--focus-sample", default="INT14")
    p.add_argument("--genes", default="UMOD,LCN2")
    p.add_argument("--max-organ-samples", type=int, default=12)
    p.add_argument("--max-global-samples", type=int, default=12)
    a = p.parse_args()

    genes = [g.strip() for g in a.genes.split(",") if g.strip()]
    m = load_dataset_manifest(a.manifest)
    organ_by = _field_by_sample(m, "organ")
    tech_by = _field_by_sample(m, "tech")
    if not organ_by:
        print("could not read per-sample organ from the manifest", file=sys.stderr)
        sys.exit(1)

    train_ids = [str(s) for s in m["train_sample_ids"]]
    focus, focus_organ = a.focus_sample, organ_by.get(a.focus_sample)
    print(f"focus sample: {focus}  organ={focus_organ}")
    print(f"train slides: {len(train_ids)}   organs: {sorted(set(organ_by.values()))}\n")

    def load(sid):
        return load_hest_sample(a.hest_data_dir, sid,
                                organ=organ_by.get(sid), tech=tech_by.get(sid))

    print(f"=== {focus} (the slide in the whole-slide plot) ===")
    focus_vals = {}
    fa = load(focus)
    for g in genes:
        v = _gene_values(fa, g)
        if v is None:
            print(f"  {g}: NOT PRESENT in this slide's var_names")
        else:
            focus_vals[g] = v
            print(f"  {g}: {_describe(v)}")
    print()

    def pool(ids, label):
        print(f"=== pooled across {len(ids)} TRAIN slides ({label}) ===")
        print(f"    {ids}")
        acc = defaultdict(list)
        for sid in ids:
            try:
                ad = load(sid)
            except Exception as e:
                print(f"    (skipped {sid}: {type(e).__name__}: {e})")
                continue
            for g in genes:
                v = _gene_values(ad, g)
                if v is not None:
                    acc[g].append(v)
        means = {}
        for g in genes:
            if not acc[g]:
                print(f"  {g}: not present in any pooled slide")
                continue
            allv = np.concatenate(acc[g])
            means[g] = float(allv.mean())
            print(f"  {g}: {_describe(allv)}")
        print()
        return means

    same_organ = [s for s in train_ids if organ_by.get(s) == focus_organ][: a.max_organ_samples]
    organ_means = pool(same_organ, f"organ={focus_organ}")

    stride = max(1, len(train_ids) // max(1, a.max_global_samples))
    global_means = pool(train_ids[::stride][: a.max_global_samples], "all organs")

    print("=== compare against the whole-slide PREDICTION panel ===")
    for g in genes:
        fm = float(focus_vals[g].mean()) if g in focus_vals else float("nan")
        print(f"  {g}: this-slide mean={fm:.4f}  "
              f"organ-pooled mean={organ_means.get(g, float('nan')):.4f}  "
              f"global-pooled mean={global_means.get(g, float('nan')):.4f}")
    print("\nIf the near-uniform predicted level matches the organ-pooled (or global-pooled)\n"
          "mean rather than this slide's own spatial pattern, the model is emitting the\n"
          "marginal mean -- the RMSE-optimal answer for a gene it cannot localize.")


if __name__ == "__main__":
    main()
