"""Save/load trainable model state, config, and gene names.

Mirrors the discipline already established (and, this session, real-bug-
fixed) in src/training/train.py::save_trained_model/load_trained_model:
only trainable parameters and non-frozen buffers are saved (frozen
backbones like GigaPath/STPath/scFoundation are reloaded fresh from their
own real pretrained source every time the architecture is rebuilt, not
re-saved here — a STPath-conditioned model's full state_dict is gigabytes
of frozen weights vs a few tens of MB of trainable ones). Config and gene
names are saved even when there are literally zero trainable weights (the
real bug fixed 2026-07-24 in the original codebase: skipping metadata
entirely whenever there was nothing trainable made a fully-frozen
checkpoint impossible to reconstruct later).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn as nn


def _is_frozen_backbone_module(module: nn.Module) -> bool:
    """A module counts as a frozen backbone if it has parameters and NONE
    of them require grad — covers GigapathPatchEncoder's lazily-loaded
    tile_encoder, STPathContextEncoder's self.model (when pretrained=True),
    scFoundation (never has trainable params by construction), and Stage
    A's autoencoder when Stage B runs with finetune_autoencoder=False."""
    params = list(module.parameters(recurse=False))
    return bool(params) and all(not p.requires_grad for p in params)


def save_trainable_state(model: nn.Module, checkpoint_dir: str | Path) -> Path | None:
    """Save trainable parameters and non-frozen buffers only. Returns the
    weights file path, or None if there was nothing trainable to save
    (e.g. Architecture 4 with its STPath backbone fully frozen and no
    scFoundation residual enabled — an edge case, but handled the same
    way the original codebase's zero-trainable-parameter bug taught us
    to: absence of a weights file is a valid, real state, not an error)."""
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    frozen_module_names = {name for name, m in model.named_modules() if _is_frozen_backbone_module(m)}

    def _under_frozen_module(buf_name: str) -> bool:
        parts = buf_name.split(".")
        return any(".".join(parts[:i]) in frozen_module_names for i in range(1, len(parts)))

    save_names = set(trainable_names)
    for buf_name, _ in model.named_buffers():
        if not _under_frozen_module(buf_name):
            save_names.add(buf_name)
    if not save_names:
        return None
    state = {k: v for k, v in model.state_dict().items() if k in save_names}
    path = Path(checkpoint_dir) / "trainable_weights.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)
    return path


def save_checkpoint(
    model: nn.Module, model_config: dict, gene_names: list[str], checkpoint_dir: str | Path,
    step: int, extra_metadata: dict | None = None,
) -> None:
    weights_path = save_trainable_state(model, checkpoint_dir)
    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in [
        ("model_config.json", model_config), ("gene_names.json", list(gene_names)),
        ("training_state.json", {"step": int(step), **(extra_metadata or {})}),
    ]:
        final_path = out_dir / name
        tmp_path = out_dir / f"{name}.tmp{os.getpid()}"
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, final_path)
    print(
        f"checkpoint saved to {out_dir} (step {step}"
        f"{', weights + config + gene names' if weights_path is not None else ', config + gene names only (no trainable weights)'})"
    )


def load_trainable_state(model: nn.Module, checkpoint_dir: str | Path) -> None:
    """Load a previously-saved trainable state onto a freshly-constructed,
    architecturally-identical model. No-op if the checkpoint genuinely has
    no trainable_weights.pt (a fully-frozen model — verified by asserting
    the fresh model ALSO has zero trainable parameters, not silently
    assumed)."""
    in_dir = Path(checkpoint_dir)
    weights_path = in_dir / "trainable_weights.pt"
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    if weights_path.is_file():
        state = torch.load(weights_path, map_location="cpu")
        missing_trainable = trainable_names - set(state.keys())
        assert not missing_trainable, (
            f"saved weights at {in_dir} are missing trainable parameters this "
            f"model architecture expects: {missing_trainable} (config mismatch?)"
        )
        model.load_state_dict(state, strict=False)
    else:
        assert not trainable_names, (
            f"{weights_path} is missing but this model architecture has trainable "
            f"parameters {trainable_names}; the checkpoint at {in_dir} looks incomplete"
        )


def load_training_state(checkpoint_dir: str | Path) -> dict:
    path = Path(checkpoint_dir) / "training_state.json"
    if not path.is_file():
        return {"step": 0}
    return json.loads(path.read_text())
