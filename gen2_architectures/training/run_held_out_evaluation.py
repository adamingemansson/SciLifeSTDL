"""Standalone offline evaluation: score an already-trained checkpoint's
held-out TEST samples without re-running (or ever needing to run inside)
the training process.

Built 2026-07-27 specifically to decouple evaluation from training after
the final-eval-inside-the-training-process hang this session spent hours
chasing (see train_local_neighborhood.py/train_arch3_stage_b.py's own
--skip_final_eval flag, added at the same time) -- a training run can now
be killed the moment its wall-clock budget is spent (checkpoint already
saved by then) without ever needing its own process to also survive the
evaluation harness. Run this against the checkpoint_dir afterwards,
whenever convenient, as many times as needed (e.g. after a code fix to
the evaluation harness itself), on a totally separate, disposable process.

Handles Architectures 1/2/4 (train_local_neighborhood.py's checkpoint
layout). Architecture 3 Stage B has its own model-construction shape
(needs a loaded Stage A autoencoder) and is not yet supported here.

Loads gene_names AND the exact trained model_config (architecture +
params + scfoundation_dim) directly from the checkpoint directory itself
-- NOT by reloading and re-deriving them from the full training sample
set the way the training scripts do -- so this never needs to touch the
(often large, multi-organ) training data at all, only the held-out TEST
samples actually being scored.

Usage:
    python3 -m gen2_architectures.training.run_held_out_evaluation \\
        --config gen2_architectures/configs/arch1_gpt_baseline.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from gen2_architectures.training import checkpoint, data_prep, evaluate
from gen2_architectures.training.train_local_neighborhood import build_model


def main(config_path: str) -> None:
    cfg = OmegaConf.load(config_path)
    data_prep.apply_sample_selection(cfg)
    device = torch.device(cfg.training.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    model_config_path = checkpoint_dir / "model_config.json"
    gene_names_path = checkpoint_dir / "gene_names.json"
    if not model_config_path.is_file() or not gene_names_path.is_file():
        raise FileNotFoundError(
            f"{checkpoint_dir} has no saved checkpoint yet (missing model_config.json/"
            "gene_names.json) -- nothing to evaluate"
        )
    saved_model_config = json.loads(model_config_path.read_text())
    gene_names = json.loads(gene_names_path.read_text())
    architecture = str(saved_model_config["architecture"])
    scfoundation_dim = saved_model_config.get("scfoundation_dim")
    model_cfg = OmegaConf.create({"model": {"architecture": architecture, "params": saved_model_config["params"]}})

    test_ids = list(cfg.data.get("test_sample_ids", []))
    if not test_ids:
        raise ValueError("data.test_sample_ids is empty -- nothing to evaluate")

    print(f"loading {len(test_ids)} held-out test sample(s), aligned to the checkpoint's gene panel...")
    kept_ids, adatas, images_list = data_prep.load_held_out_samples_with_images(cfg, test_ids, gene_names)
    held_out = {str(sid): (adata, images) for sid, adata, images in zip(kept_ids, adatas, images_list)}

    scfoundation_providers: dict[str, object] = {}
    if architecture in ("2", "4"):
        print("building scFoundation context feature providers for the held-out test samples...")
        for sid in kept_ids:
            adata = held_out[str(sid)][0]
            scfoundation_providers[str(sid)] = data_prep.build_scfoundation_provider(cfg, adata, str(sid))

    model = build_model(model_cfg, gene_names, scfoundation_dim).to(device)
    checkpoint.load_trainable_state(model, checkpoint_dir)
    step = checkpoint.load_training_state(checkpoint_dir).get("step", 0)
    print(f"loaded checkpoint at step {step}, architecture {architecture}, {len(gene_names)} genes")
    model.eval()

    for sid in kept_ids:
        adata, images = held_out[str(sid)]
        gene_inputs = {
            "context_gene_feature_provider": scfoundation_providers.get(str(sid)) if architecture == "2" else None,
            "context_extra_feature_provider": scfoundation_providers.get(str(sid)) if architecture == "4" else None,
        }
        metrics = evaluate.evaluate_sample(model, cfg, adata, images, gene_inputs, str(sid), "test", checkpoint_dir)
        primary = metrics["image_modes"][metrics["primary_image_mode"]]["summary"]
        print(f"TEST {sid}: PCC={primary['pcc']['mean']:.4f} RMSE={primary['rmse']['mean']:.4f}")
        if "pcc_raw_log1p" in primary:
            print(f"       notebook-comparable PCC (raw log1p space): {primary['pcc_raw_log1p']['mean']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                         help="Same config the training run used -- reads training.checkpoint_dir "
                              "from it, but the model itself is rebuilt from that checkpoint "
                              "directory's own saved model_config.json/gene_names.json, not from "
                              "this file's cfg.model (so it's exact even if the YAML changed since).")
    args = parser.parse_args()
    main(args.config)
