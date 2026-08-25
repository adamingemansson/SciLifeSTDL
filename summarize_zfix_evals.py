#!/usr/bin/env python3
"""Side-by-side metrics table across conditional-WAE evaluation reports."""
from __future__ import annotations
import argparse, json, glob, os


def short(arm: str) -> str:
    a = arm
    if a.startswith("wae_he_mmd_geneencoder_omiclip_"):
        return "omiclip/" + a[len("wae_he_mmd_geneencoder_omiclip_"):]
    if a.startswith("wae_he_mmd_geneencoder_"):
        return "uni2/" + a[len("wae_he_mmd_geneencoder_"):]
    return a


def num(x, fmt="{:.4f}"):
    return fmt.format(x) if isinstance(x, (int, float)) else "-"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--tag", default="", help="optional suffix filter, e.g. _eval.json")
    a = p.parse_args()

    reports = []
    for root in a.roots:
        for path in sorted(glob.glob(os.path.join(root, "logs", "*_eval.json"))):
            try:
                with open(path) as fh:
                    r = json.load(fh)
            except Exception as e:
                print(f"(skipped {path}: {type(e).__name__}: {e})")
                continue
            arm = os.path.basename(path)[: -len("_eval.json")]
            reports.append((arm, os.path.basename(root.rstrip("/")), r))

    if not reports:
        print("no *_eval.json found under the given roots")
        return

    panels = sorted({
        k for _, _, r in reports
        for k in (r.get("per_panel_patient_aggregated_metrics") or {})
    })

    def full(r, arm_key, metric):
        return (((r.get("per_arm_patient_aggregated_metrics") or {})
                 .get(arm_key) or {}).get(metric) or {}).get("patient_mean")

    def panel(r, pname, arm_key, metric):
        return ((((r.get("per_panel_patient_aggregated_metrics") or {})
                  .get(pname) or {}).get(arm_key) or {}).get(metric) or {}).get("patient_mean")

    # ---------------- accuracy ----------------
    print("=" * 100)
    print("ACCURACY  (patient-mean; 'model' = predictive mean, 'cmean' = image-only conditional-mean head)")
    print("=" * 100)
    hdr = f"{'arm':<34}{'step':>7}{'items':>7}{'PCC':>9}{'RMSE':>9}{'cmeanPCC':>10}"
    for pn in panels:
        hdr += f"{pn+' PCC':>16}{pn+' RMSE':>17}"
    print(hdr)
    print("-" * len(hdr))
    for arm, _root, r in reports:
        row = (f"{short(arm):<34}{str(r.get('checkpoint_step','-')):>7}"
               f"{str(r.get('n_items','-')):>7}"
               f"{num(full(r,'model','pcc')):>9}{num(full(r,'model','rmse')):>9}"
               f"{num(full(r,'conditional_mean','pcc')):>10}")
        for pn in panels:
            row += (f"{num(panel(r,pn,'model','pcc')):>16}"
                    f"{num(panel(r,pn,'model','rmse')):>17}")
        print(row)

    # ---------------- calibration ----------------
    print()
    print("=" * 100)
    print("CALIBRATION  (ideal: z_std=1.00, z_mean=0.00, cov68=0.68, cov90=0.90, cov95=0.95)")
    print("=" * 100)
    hdr2 = (f"{'arm':<34}{'n_values':>13}{'z_mean':>10}{'z_std':>10}"
            f"{'cov68':>9}{'cov90':>9}{'cov95':>9}{'ex-post':>9}")
    print(hdr2)
    print("-" * len(hdr2))
    for arm, _root, r in reports:
        c = r.get("conditional_wae_calibration") or {}
        used = "yes" if r.get("ex_post_prior_path") else "no"
        print(f"{short(arm):<34}{str(c.get('n_values','-')):>13}"
              f"{num(c.get('z_mean')):>10}{num(c.get('z_std')):>10}"
              f"{num(c.get('coverage_68')):>9}{num(c.get('coverage_90')):>9}"
              f"{num(c.get('coverage_95')):>9}{used:>9}")
    print()
    print("z_std >> 1 means predictive_std is far too small (over-confident) -- the")
    print("under-dispersion the z-noise-augmentation retrain was meant to fix.")


if __name__ == "__main__":
    main()
