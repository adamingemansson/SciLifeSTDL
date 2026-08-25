#!/usr/bin/env python3
"""Does the latent z contribute anything to the prediction?"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
from omegaconf import OmegaConf
from gen3_multiscale.conditional_wae import ConditionalWAEMaskedGEXDataset
from gen3_multiscale.conditional_wae.contract import static_audit_conditional_wae_config
from gen3_multiscale.config_identity import resolved_config
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.training import checkpoint as checkpoint_module
from gen3_multiscale.training.gen3_dataset import Gen3SpatialFieldDataset, build_gen3_mask_schedule
from gen3_multiscale.training.gen3_preflight import load_and_preflight_samples
from gen3_multiscale.training.train import expected_tile_encoder_provenance
from gen3_multiscale.training.train_conditional_wae import _build_model, _manifest, _verify_resume


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--n-masks-per-sample", type=int, default=2)
    p.add_argument("--max-items", type=int, default=32)
    p.add_argument("--n-draws", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--allow-code-drift", action="store_true")
    a = p.parse_args()

    config = resolved_config(a.config)
    static_audit_conditional_wae_config(config)
    dm = load_dataset_manifest(config["data"]["gen3_manifest_path"])
    split_ids = list(dm[f"{a.split}_sample_ids"])
    samples, pre = load_and_preflight_samples(
        OmegaConf.create(config), dm, split_ids, expected_tile_encoder_provenance(config))
    sched = build_gen3_mask_schedule(
        dm, samples, config["masking"]["strata"], a.split,
        split_counts={a.split: int(a.n_masks_per_sample)},
        split_seeds={a.split: 700_000 if a.split == "validation" else 900_000})
    base = Gen3SpatialFieldDataset(dm, samples, sched, config["masking"]["strata"], novae_enabled=False)
    ds = ConditionalWAEMaskedGEXDataset(base, include_observed_gex=bool(config["model"]["include_observed_gex"]))

    ckpt = Path(a.checkpoint_dir) / "best"
    old = checkpoint_module.load_checkpoint_run_manifest(ckpt)
    if old is None:
        raise ValueError(f"{ckpt}: no bundle-bound run manifest")
    cur = _manifest(config, dm, pre)
    cur["cache_content_fingerprint"] = old["cache_content_fingerprint"]
    cur["cache_content_by_sample"] = old.get("cache_content_by_sample") or {}
    _verify_resume(old, cur, allow_code_drift=a.allow_code_drift)

    genes = list(dm["gene_panel"])
    checkpoint_module.verify_gene_names(ckpt, genes)
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    model = _build_model(config, len(genes), gene_names=genes).to(dev)
    checkpoint_module.load_trainable_state(model, ckpt)
    model.eval()

    ratios, spreads, corrs, ucorrs = [], [], [], []
    n = min(int(a.max_items), len(ds))
    with torch.no_grad():
        for i in range(n):
            inputs, target, ident = ds[i]
            ctx = model.image_conditioner(inputs)
            true = torch.as_tensor(np.asarray(target, dtype=np.float32), device=ctx.device)
            draws = []
            for d in range(int(a.n_draws)):
                g = torch.Generator(device=ctx.device).manual_seed(1_000_003 * i + d)
                z = torch.randn(ctx.shape[0], model.latent_dim, dtype=ctx.dtype,
                                device=ctx.device, generator=g)
                rec, cm = model.decode(z, ctx)
                draws.append(rec)
            st = torch.stack(draws)
            cmn = float(cm.norm())
            res = st.mean(0) - cm
            ratios.append(float(res.norm()) / max(cmn, 1e-8))
            spreads.append(float(st.std(0, unbiased=False).mean()) /
                           max(float(cm.abs().mean()), 1e-8))
            corrs.append(float(torch.corrcoef(torch.stack(
                [st.mean(0).flatten().double(), cm.flatten().double()]))[0, 1]))
            err = (true - cm).flatten().double()
            rf = res.flatten().double()
            if float(rf.std()) > 1e-12 and float(err.std()) > 1e-12:
                ucorrs.append(float(torch.corrcoef(torch.stack([rf, err]))[0, 1]))
            print(f"item {i+1}/{n} sample={ident['sample_id']} residual/cmean={ratios[-1]:.6f}", flush=True)

    def s(v):
        if not v:
            return {"n": 0}
        x = np.asarray(v, dtype=np.float64)
        return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
                "min": float(x.min()), "max": float(x.max())}

    rep = {"version": 1, "kind": "conditional_wae_latent_contribution",
           "config_path": str(a.config), "checkpoint_dir": str(ckpt),
           "checkpoint_step": (checkpoint_module.load_training_state(ckpt) or {}).get("step"),
           "split": a.split, "n_items": n, "n_draws": int(a.n_draws),
           "residual_norm_over_conditional_mean_norm": s(ratios),
           "across_draw_spread_over_conditional_mean_scale": s(spreads),
           "corr_reconstruction_vs_conditional_mean": s(corrs),
           "corr_residual_vs_conditional_mean_error": s(ucorrs)}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(rep, indent=2, sort_keys=True, default=str))

    print("\n" + "=" * 78)
    print(f"checkpoint step: {rep['checkpoint_step']}   items: {n}   draws: {a.n_draws}")
    print("=" * 78)
    r, sp, c, u = (rep["residual_norm_over_conditional_mean_norm"],
                   rep["across_draw_spread_over_conditional_mean_scale"],
                   rep["corr_reconstruction_vs_conditional_mean"],
                   rep["corr_residual_vs_conditional_mean_error"])
    print(f"  ||residual|| / ||conditional_mean||    mean={r['mean']:.6f}  median={r['median']:.6f}")
    print(f"  across-draw spread / |cmean| scale     mean={sp['mean']:.6f}  median={sp['median']:.6f}")
    print(f"  corr(reconstruction, conditional_mean) mean={c['mean']:.6f}")
    if u.get("n"):
        print(f"  corr(residual, cmean's error)          mean={u['mean']:.6f}   <- useful signal would be > 0")
    print(f"\nsaved: {a.output}")


if __name__ == "__main__":
    main()
