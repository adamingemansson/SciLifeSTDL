"""Tests for gen3_multiscale/config_identity.py -- Codex re-audit of
commit 57f0e3c: "Prefer moving config fingerprint/loading utilities into
a neutral module to avoid a circular import between the trainer and
resolver." Confirms the neutral module works standalone, that
training/train.py's own names are genuine re-exports (not
reimplementations that could drift apart), and that the resolver now
imports from the neutral module directly, never from train.py."""
from __future__ import annotations

import ast
from pathlib import Path

from gen3_multiscale.config_identity import config_fingerprint, config_identity_fingerprint, resolved_config

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_train_py_re_exports_are_the_same_function_objects_not_reimplementations():
    from gen3_multiscale.training import train as train_module

    assert train_module.config_fingerprint is config_fingerprint
    assert train_module.config_identity_fingerprint is config_identity_fingerprint
    assert train_module.resolved_config is resolved_config


def test_config_fingerprint_is_deterministic_and_content_sensitive():
    a = {"model": {"architecture": "1"}, "training": {"lr": 1e-4}}
    b = {"model": {"architecture": "1"}, "training": {"lr": 1e-4}}
    c = {"model": {"architecture": "2"}, "training": {"lr": 1e-4}}
    assert config_fingerprint(a) == config_fingerprint(b)
    assert config_fingerprint(a) != config_fingerprint(c)


def test_config_identity_fingerprint_ignores_resume_excluded_and_evaluation_fields():
    a = {"training": {"total_steps": 100, "lr": 1e-4}, "evaluation": {"n_masks_per_sample": 4}}
    b = {"training": {"total_steps": 999999, "lr": 1e-4}, "evaluation": {"n_masks_per_sample": 999}}
    c = {"training": {"total_steps": 100, "lr": 2e-4}, "evaluation": {"n_masks_per_sample": 4}}
    assert config_identity_fingerprint(a) == config_identity_fingerprint(b)
    assert config_identity_fingerprint(a) != config_identity_fingerprint(c)


def test_resolve_experiment_config_module_does_not_import_from_training_train():
    """Static AST check -- Codex re-audit of commit 57f0e3c's explicit
    ask: the resolver must import config fingerprint/loading utilities
    from the neutral module, never from training.train (which would
    reintroduce the exact circular-import risk this refactor closes)."""
    source = (_REPO_ROOT / "gen3_multiscale" / "scripts" / "resolve_experiment_config.py").read_text()
    tree = ast.parse(source)
    imported_modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "gen3_multiscale.training.train" not in imported_modules
    assert "gen3_multiscale.config_identity" in imported_modules
