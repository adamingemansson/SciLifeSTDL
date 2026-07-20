"""Fixed-mask validation and early stopping for variable-size point clouds."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import pytorch_lightning as pl


def move_to_device(value: Any, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    return value


def predictive_samples(model, context: dict, query: dict, n_samples: int,
                       seed: int = 0) -> torch.Tensor:
    """Return ``[S, N, G]`` samples with deterministic RNG isolation."""
    outputs = []
    devices = [model.device] if model.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if model.device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        for _ in range(max(1, int(n_samples))):
            outputs.append(model.sample(context, query)["expression"])
    return torch.stack(outputs, dim=0)


def _is_frozen_backbone_module(module: torch.nn.Module) -> bool:
    params = list(module.parameters(recurse=True))
    return bool(params) and all(not p.requires_grad for p in params)


def compact_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    frozen_module_names = {name for name, m in model.named_modules() if _is_frozen_backbone_module(m)}

    def under_frozen(name: str) -> bool:
        parts = name.split(".")
        return any(".".join(parts[:i]) in frozen_module_names for i in range(1, len(parts)))

    names = set(trainable_names)
    names.update(name for name, _ in model.named_buffers() if not under_frozen(name))
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k in names}


class FixedMaskValidationCallback(pl.Callback):
    """Evaluate immutable validation-mask items and restore the best state.

    Validation uses the predictive mean across ``n_samples`` stochastic draws.
    The callback never evaluates the final test masks. It stores a compact state
    excluding frozen pretrained backbones, so STPath/GigaPath validation does
    not create multi-gigabyte in-memory copies.
    """

    def __init__(self, items: list[dict], every_n_steps: int = 1000,
                 patience_checks: int = 5, min_delta: float = 1e-4,
                 metric: str = "rmse", n_samples: int = 4,
                 history_path: str | Path | None = None, seed: int = 12345):
        super().__init__()
        if metric not in {"rmse", "pcc"}:
            raise ValueError("validation metric must be 'rmse' or 'pcc'")
        self.items = items
        self.every_n_steps = max(1, int(every_n_steps))
        self.patience_checks = max(1, int(patience_checks))
        self.min_delta = float(min_delta)
        self.metric = metric
        self.n_samples = max(1, int(n_samples))
        self.history_path = Path(history_path) if history_path else None
        self.seed = int(seed)
        self.best_score = float("inf") if metric == "rmse" else -float("inf")
        self.best_state: dict[str, torch.Tensor] | None = None
        self.bad_checks = 0
        self.history: list[dict] = []

    @torch.no_grad()
    def _score(self, model: torch.nn.Module, step: int) -> float:
        values = []
        was_training = model.training
        model.eval()
        for i, cpu_item in enumerate(self.items):
            item = move_to_device(cpu_item, model.device)
            samples = predictive_samples(
                model, item["context"], item["query"], self.n_samples,
                # Keep Monte-Carlo draws fixed across validation checks so a
                # checkpoint is never rewarded or rejected merely because it
                # received an easier random sample set at a later step.
                seed=self.seed + i,
            )
            pred = samples.mean(dim=0)
            target = item["target_expression"]
            decoder_idx = getattr(model, "_decoder_target_col_idx", None)
            if decoder_idx is not None:
                target = target[:, decoder_idx]
            if self.metric == "rmse":
                value = torch.sqrt(torch.mean((pred - target) ** 2)).item()
            else:
                pred_c = pred - pred.mean(dim=0, keepdim=True)
                target_c = target - target.mean(dim=0, keepdim=True)
                denom = torch.sqrt((pred_c**2).sum(0) * (target_c**2).sum(0)).clamp_min(1e-8)
                value = ((pred_c * target_c).sum(0) / denom).nanmean().item()
            values.append(value)
        model.train(was_training)
        return float(np.mean(values))

    def _improved(self, score: float) -> bool:
        if self.metric == "rmse":
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

    def _write_history(self) -> None:
        if self.history_path is None:
            return
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.history_path.with_suffix(self.history_path.suffix + ".tmp")
        tmp.write_text(json.dumps({
            "metric": self.metric,
            "best_score": self.best_score,
            "n_samples": self.n_samples,
            "history": self.history,
        }, indent=2))
        tmp.replace(self.history_path)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step == 0 or step % self.every_n_steps:
            return
        score = self._score(pl_module, step)
        improved = self._improved(score)
        self.history.append({"step": step, self.metric: score, "improved": improved})
        if improved:
            self.best_score = score
            self.best_state = compact_state_dict(pl_module)
            self.bad_checks = 0
        else:
            self.bad_checks += 1
        self._write_history()
        print(f"validation step={step} {self.metric}={score:.6f} best={self.best_score:.6f}")
        if self.bad_checks >= self.patience_checks:
            print(f"early stopping after {self.bad_checks} validation checks without improvement")
            trainer.should_stop = True

    def on_train_end(self, trainer, pl_module):
        # Very short smoke runs may finish before every_n_steps. They still
        # receive one real validation check rather than silently saving the
        # last training state under a "best" checkpoint name.
        if self.best_state is None and self.items:
            step = int(trainer.global_step)
            score = self._score(pl_module, step)
            self.best_score = score
            self.best_state = compact_state_dict(pl_module)
            self.history.append({"step": step, self.metric: score, "improved": True, "final_check": True})
        if self.best_state is not None:
            pl_module.load_state_dict(self.best_state, strict=False)
        self._write_history()
