#!/usr/bin/env python3
"""Create held-out-sample configs from selected single-slide finalists."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = {
    "harmonic_residual": ROOT / "configs/audit_suite/heldout_harmonic_residual_stormlite.yaml",
    "residual_fm_ot": ROOT / "configs/audit_suite/heldout_residual_fm_ot_stormlite.yaml",
}


def promote(source: Path, output_dir: Path, standard_unique_masks: int,
            novae_unique_masks: int) -> Path:
    selected = yaml.safe_load(source.read_text())
    model_name = selected.get("model", {}).get("name")
    if model_name not in TEMPLATES:
        raise ValueError(
            f"{source} uses model {model_name!r}; held-out promotion supports "
            "harmonic_residual and residual_fm_ot finalists"
        )
    template = yaml.safe_load(TEMPLATES[model_name].read_text())
    selected_params = deepcopy(selected["model"].get("params", {}))

    # The held-out residual template owns the training-panel autoencoder path
    # and its quality gates. Preserve those while taking the selected context,
    # fusion, latent and solver choices from the screening winner.
    if model_name == "residual_fm_ot":
        for key in (
            "pretrained_autoencoder_path",
            "freeze_pretrained_autoencoder",
            "pretrained_autoencoder_max_rmse",
            "pretrained_autoencoder_min_pcc",
        ):
            selected_params[key] = template["model"]["params"][key]
    selected_params["use_organ_tech_conditioning"] = True
    template["model"]["params"] = selected_params

    uses_novae = selected.get("data", {}).get("novae_mode") == "context_only"
    template["data"]["novae_mode"] = "context_only" if uses_novae else "disabled"
    base = str(selected["experiment_name"]).removeprefix("screen_").removeprefix("confirm_")
    exp = f"heldout_{base}"
    template["experiment_name"] = exp
    template["training"]["checkpoint_dir"] = f"results/checkpoints/complexity_ladder/{exp}"
    unique_masks = novae_unique_masks if uses_novae else (
        standard_unique_masks or int(template["training"]["epochs"])
    )
    template["training"]["unique_mask_count"] = int(unique_masks)
    template["evaluation"]["training_mask_bank_path"] = (
        f"results/mask_banks/training/complexity_ladder/{exp}_u{unique_masks}.json"
    )
    template["evaluation"]["mask_bank_dir"] = "results/mask_banks/complexity_ladder/heldout"
    template["logging"]["backend"] = "csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{exp}.yaml"
    output.write_text(yaml.safe_dump(template, sort_keys=False, width=1000))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("configs/complexity_ladder/heldout"))
    parser.add_argument(
        "--unique-masks", type=int, default=0,
        help="unique masks for non-Novae held-out runs; 0 means one fresh mask per step",
    )
    parser.add_argument("--novae-unique-masks", type=int, default=512)
    args = parser.parse_args()
    for path in args.configs:
        print(promote(path, args.output_dir, args.unique_masks, args.novae_unique_masks))


if __name__ == "__main__":
    main()
