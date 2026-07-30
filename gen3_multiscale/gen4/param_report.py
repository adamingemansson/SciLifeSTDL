"""Parameter-count / frozen-vs-trainable report for a Gen4 model instance
-- GEN4_CONTRACT.md section 3/12. Pure introspection over a real,
already-constructed `nn.Module`; no training or weight loading happens
here.
"""
from __future__ import annotations

import torch.nn as nn


def report_parameters(model: nn.Module) -> dict:
    """Per-top-level-submodule (and overall) trainable/frozen parameter
    counts. A submodule with ALL parameters frozen (requires_grad=False)
    is reported as `"frozen"`; one with a mix is `"mixed"` -- e.g.
    Gen4Conditioner's own `gene_encoder` for a frozen_context arm (built
    but never called, gradient disabled -- GEN4_CONTRACT.md section 4)."""
    per_module: dict[str, dict] = {}
    total_trainable, total_frozen = 0, 0
    for name, submodule in model.named_children():
        trainable = sum(p.numel() for p in submodule.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in submodule.parameters() if not p.requires_grad)
        total_trainable += trainable
        total_frozen += frozen
        if trainable and frozen:
            status = "mixed"
        elif trainable:
            status = "trainable"
        elif frozen:
            status = "frozen"
        else:
            status = "empty"
        per_module[name] = {"trainable_params": trainable, "frozen_params": frozen, "status": status}
    return {
        "model_class": type(model).__name__,
        "total_trainable_params": total_trainable,
        "total_frozen_params": total_frozen,
        "total_params": total_trainable + total_frozen,
        "by_submodule": per_module,
    }


def format_report_table(report: dict) -> str:
    lines = [f"{report['model_class']}: {report['total_trainable_params']:,} trainable / {report['total_frozen_params']:,} frozen"]
    for name, entry in report["by_submodule"].items():
        lines.append(f"  {name:24s} {entry['status']:10s} trainable={entry['trainable_params']:,} frozen={entry['frozen_params']:,}")
    return "\n".join(lines)
