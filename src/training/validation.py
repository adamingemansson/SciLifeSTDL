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
    """Return ``[S, N, G]`` samples with deterministic RNG isolation.

    Models may expose ``prepare_sampling_conditioning`` and
    ``sample_from_prepared_conditioning`` to cache deterministic conditioning
    across draws.  FlowMatchingOT uses this to avoid running STPath/StormLite
    once per Monte-Carlo sample.  The fast path is eval-only: callers that
    intentionally sample while the model is in training mode retain the old
    one-shot behavior, including any stochastic conditioner layers.
    """
    outputs = []
    devices = [model.device] if model.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if model.device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        prepare = getattr(model, "prepare_sampling_conditioning", None)
        sample_prepared = getattr(model, "sample_from_prepared_conditioning", None)
        use_prepared = not model.training and callable(prepare) and callable(sample_prepared)
        prepared = prepare(context, query) if use_prepared else None
        for _ in range(max(1, int(n_samples))):
            output = sample_prepared(prepared) if use_prepared else model.sample(context, query)
            outputs.append(output["expression"])
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
                 history_path: str | Path | None = None, seed: int = 12345,
                 early_stopping_min_steps: int = 0,
                 require_anchor_improvement: bool = False,
                 anchor_min_delta: float = 0.0,
                 min_correction_rms: float = 0.0,
                 quality_gate_path: str | Path | None = None):
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
        self.early_stopping_min_steps = max(0, int(early_stopping_min_steps))
        self.require_anchor_improvement = bool(require_anchor_improvement)
        self.anchor_min_delta = float(anchor_min_delta)
        self.min_correction_rms = max(0.0, float(min_correction_rms))
        self.quality_gate_path = Path(quality_gate_path) if quality_gate_path else None
        self.best_score = float("inf") if metric == "rmse" else -float("inf")
        self.patience_score = float("inf") if metric == "rmse" else -float("inf")
        self.best_state: dict[str, torch.Tensor] | None = None
        self.anchor_score: float | None = None
        self.correction_rms: float | None = None
        self.quality_gate_passed: bool | None = None
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

    @torch.no_grad()
    def _score_anchor(self, model: torch.nn.Module) -> float | None:
        values = []
        was_training = model.training
        model.eval()
        for cpu_item in self.items:
            item = move_to_device(cpu_item, model.device)
            output = model.sample(item["context"], item["query"])
            pred = output.get("anchor_expression")
            if pred is None:
                model.train(was_training)
                return None
            target = item["target_expression"]
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
        """Whether ``score`` is the numerically best checkpoint seen.

        Checkpoint selection must not use ``min_delta``.  ``min_delta`` is an
        early-stopping tolerance: using it here can retain an older, worse
        checkpoint whenever several individually small improvements add up.
        """
        if self.metric == "rmse":
            return score < self.best_score
        return score > self.best_score

    def _meaningfully_improved(self, score: float) -> bool:
        """Whether ``score`` clears the tolerance used to reset patience."""
        if self.metric == "rmse":
            return score < self.patience_score - self.min_delta
        return score > self.patience_score + self.min_delta

    @torch.no_grad()
    def _score_correction_rms(self, model: torch.nn.Module) -> float | None:
        values = []
        was_training = model.training
        model.eval()
        for cpu_item in self.items:
            item = move_to_device(cpu_item, model.device)
            output = model.sample(item["context"], item["query"])
            correction = output.get("residual_expression")
            if correction is None:
                model.train(was_training)
                return None
            values.append(correction.square().mean().sqrt().item())
        model.train(was_training)
        return float(np.mean(values))

    def _write_history(self) -> None:
        if self.history_path is None:
            return
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.history_path.with_suffix(self.history_path.suffix + ".tmp")
        tmp.write_text(json.dumps({
            "metric": self.metric,
            "best_score": self.best_score,
            "anchor_score": self.anchor_score,
            "correction_rms": self.correction_rms,
            "minimum_correction_rms": self.min_correction_rms,
            "n_samples": self.n_samples,
            "history": self.history,
        }, indent=2))
        tmp.replace(self.history_path)

    def _write_quality_gate(self) -> None:
        if self.quality_gate_path is None:
            return
        self.quality_gate_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "required": self.require_anchor_improvement,
            "passed": self.quality_gate_passed,
            "metric": self.metric,
            "best_score": self.best_score,
            "anchor_score": self.anchor_score,
            "required_improvement": self.anchor_min_delta,
            "correction_rms": self.correction_rms,
            "minimum_correction_rms": self.min_correction_rms,
        }
        tmp = self.quality_gate_path.with_suffix(self.quality_gate_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.quality_gate_path)

    def on_train_start(self, trainer, pl_module):
        self.anchor_score = self._score_anchor(pl_module)
        if self.anchor_score is not None:
            self.history.append({"step": 0, f"anchor_{self.metric}": self.anchor_score})
            self._write_history()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step == 0 or step % self.every_n_steps:
            return
        score = self._score(pl_module, step)
        meaningfully_improved = self._meaningfully_improved(score)
        improved = self._improved(score)
        self.history.append({
            "step": step,
            self.metric: score,
            "improved": improved,
            "patience_reset": meaningfully_improved,
        })
        if improved:
            self.best_score = score
            self.best_state = compact_state_dict(pl_module)
        if meaningfully_improved:
            self.patience_score = score
            self.bad_checks = 0
        elif step >= self.early_stopping_min_steps:
            self.bad_checks += 1
        self._write_history()
        print(f"validation step={step} {self.metric}={score:.6f} best={self.best_score:.6f}")
        if (step >= self.early_stopping_min_steps
                and self.bad_checks >= self.patience_checks):
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
        self.correction_rms = self._score_correction_rms(pl_module)
        if self.anchor_score is None:
            self.quality_gate_passed = None
        elif self.metric == "rmse":
            self.quality_gate_passed = self.best_score < self.anchor_score - self.anchor_min_delta
        else:
            self.quality_gate_passed = self.best_score > self.anchor_score + self.anchor_min_delta
        if (self.quality_gate_passed is True and self.min_correction_rms > 0
                and (self.correction_rms is None
                     or self.correction_rms < self.min_correction_rms)):
            self.quality_gate_passed = False
        self._write_history()
        self._write_quality_gate()

    def raise_if_quality_gate_failed(self) -> None:
        if self.require_anchor_improvement and self.quality_gate_passed is not True:
            raise RuntimeError(
                "model quality gate failed: the best validation checkpoint "
                f"did not beat its declared anchor by {self.anchor_min_delta:g} "
                f"{self.metric} (best={self.best_score:.6f}, anchor={self.anchor_score}). "
                f"correction_rms={self.correction_rms}, required>={self.min_correction_rms:g}. "
                "Do not promote this run to later experiment stages."
            )
