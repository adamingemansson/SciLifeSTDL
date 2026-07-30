"""Shared full-expression autoencoder -- GEN5_CONTRACT.md section 3.

Row-independent (no spot ever attends to another spot inside this module)
-- spatial reasoning belongs entirely to the conditioner/flow, matching
this codebase's established "concatenate/project per-branch, spatial
mixing happens once, elsewhere" discipline (models/tokens.py's own
module docstring).
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from gen3_multiscale.evaluation.metrics import pearson_per_gene, rmse

_SCHEMA_VERSION = 1


def _gene_names_hash(gene_names: list[str]) -> str:
    return hashlib.sha256("\0".join(gene_names).encode()).hexdigest()


class ExpressionEncoder(nn.Module):
    def __init__(self, n_genes: int, latent_dim: int = 256, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        if expression.ndim != 2:
            raise ValueError(f"expression must be [N, n_genes], got shape {tuple(expression.shape)}")
        return self.net(expression)


class ExpressionDecoder(nn.Module):
    def __init__(self, n_genes: int, latent_dim: int = 256, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_genes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError(f"z must be [N, latent_dim], got shape {tuple(z.shape)}")
        return self.net(z)


class ExpressionAutoencoder(nn.Module):
    """`gene_names`/`gene_names_hash` are metadata carried on the module
    (not used in any forward pass) so a caller always has one object to
    both run and verify -- mirrors GeneResidualBasis's identical
    (basis-tensor, gene_names, gene_names_hash) shape (models/gene_basis.py)."""

    def __init__(self, n_genes: int, gene_names: list[str], latent_dim: int = 256, hidden_dim: int = 1024):
        super().__init__()
        if len(gene_names) != n_genes:
            raise ValueError(f"gene_names has {len(gene_names)} entries, expected n_genes={n_genes}")
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.gene_names = tuple(gene_names)
        self.gene_names_hash = _gene_names_hash(list(gene_names))
        self.encoder = ExpressionEncoder(n_genes, latent_dim, hidden_dim)
        self.decoder = ExpressionDecoder(n_genes, latent_dim, hidden_dim)

    def encode(self, expression: torch.Tensor) -> torch.Tensor:
        return self.encoder(expression)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(expression))


def verify_expression_autoencoder_gene_names(autoencoder: ExpressionAutoencoder, gene_names: list[str]) -> None:
    """Fail-closed check mirroring gene_basis.verify_gene_residual_basis."""
    current_hash = _gene_names_hash(list(gene_names))
    if current_hash != autoencoder.gene_names_hash:
        raise ValueError(
            "gene panel does not match the panel this ExpressionAutoencoder was fit against "
            f"(expected hash {autoencoder.gene_names_hash}, got {current_hash}) -- refusing to use an "
            "autoencoder fit on a different, possibly reordered, gene panel"
        )


def save_expression_autoencoder_checkpoint(
    autoencoder: ExpressionAutoencoder, path: str | Path, *,
    dataset_manifest_fingerprint: str, preprocessing_spec: str, code_identity: str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "kind": "gen5_expression_autoencoder",
        "n_genes": autoencoder.n_genes,
        "latent_dim": autoencoder.latent_dim,
        "hidden_dim": autoencoder.encoder.net[0].out_features,
        "gene_names": list(autoencoder.gene_names),
        "gene_names_hash": autoencoder.gene_names_hash,
        "dataset_manifest_fingerprint": dataset_manifest_fingerprint,
        "preprocessing_spec": preprocessing_spec,
        "code_identity": code_identity,
        "state_dict": autoencoder.state_dict(),
    }
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_expression_autoencoder_checkpoint(
    path: str | Path, *,
    dataset_manifest_fingerprint: str | None = None, code_identity: str | None = None, allow_code_drift: bool = False,
) -> tuple[ExpressionAutoencoder, dict]:
    """Load-and-verify. `dataset_manifest_fingerprint`, when given, must
    match the saved value exactly (fail closed -- a checkpoint fit
    against a different dataset must never be silently reused).
    `code_identity`, when given, must match unless `allow_code_drift=True`
    -- the same explicit, named override this codebase already uses
    elsewhere (`training/train.py`'s `allow_code_drift`) rather than a
    silent bypass."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"expression autoencoder checkpoint not found at {path}")
    payload = torch.load(path, map_location="cpu")
    if payload.get("kind") != "gen5_expression_autoencoder" or payload.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(f"{path} is not a valid gen5_expression_autoencoder checkpoint")
    gene_names = list(payload["gene_names"])
    expected_hash = _gene_names_hash(gene_names)
    if payload.get("gene_names_hash") != expected_hash:
        raise ValueError(f"{path}: gene_names_hash does not match its own saved gene_names -- corrupted or hand-edited file")
    if dataset_manifest_fingerprint is not None and payload.get("dataset_manifest_fingerprint") != dataset_manifest_fingerprint:
        raise ValueError(
            f"{path}: dataset_manifest_fingerprint {payload.get('dataset_manifest_fingerprint')!r} does not match "
            f"the live manifest {dataset_manifest_fingerprint!r} -- refusing to load an autoencoder fit on "
            "different data"
        )
    if code_identity is not None and payload.get("code_identity") != code_identity and not allow_code_drift:
        raise ValueError(
            f"{path}: code_identity {payload.get('code_identity')!r} does not match the current "
            f"{code_identity!r} -- pass allow_code_drift=True to explicitly override"
        )
    autoencoder = ExpressionAutoencoder(payload["n_genes"], gene_names, payload["latent_dim"], payload.get("hidden_dim", 1024))
    state = payload["state_dict"]
    fresh_state = autoencoder.state_dict()
    if set(state) != set(fresh_state):
        raise ValueError(f"{path}: checkpoint state_dict keys do not match a freshly-constructed ExpressionAutoencoder")
    shape_mismatches = [
        name for name in state if tuple(state[name].shape) != tuple(fresh_state[name].shape)
    ]
    if shape_mismatches:
        raise ValueError(
            f"{path}: checkpoint state_dict has shape mismatches for {shape_mismatches[:5]} against the "
            "reconstructed model -- corrupted or hand-edited checkpoint metadata"
        )
    autoencoder.load_state_dict(state)
    return autoencoder, payload


@dataclass(frozen=True)
class ReconstructionReport:
    n_rows: int
    n_genes: int
    rmse: float
    pcc_mean: float
    pcc_per_gene: np.ndarray


def evaluate_autoencoder_reconstruction(
    autoencoder: ExpressionAutoencoder, expression: np.ndarray, gene_names: list[str],
) -> ReconstructionReport:
    """Full-gene reconstruction PCC/RMSE -- the "reconstruction ceiling"
    every Gen5 flow arm's decoded output is bounded by. `expression` is
    caller-provided (held-out training rows or real validation rows,
    GEN5_CONTRACT.md section 3) -- this function has no knowledge of
    which split it came from."""
    verify_expression_autoencoder_gene_names(autoencoder, gene_names)
    autoencoder.eval()
    with torch.no_grad():
        reconstructed = autoencoder(torch.as_tensor(expression, dtype=torch.float32)).cpu().numpy()
    per_gene = pearson_per_gene(reconstructed, np.asarray(expression, dtype=np.float32))
    return ReconstructionReport(
        n_rows=int(expression.shape[0]), n_genes=int(expression.shape[1]),
        rmse=float(rmse(reconstructed, np.asarray(expression, dtype=np.float32))),
        pcc_mean=float(np.nanmean(per_gene)), pcc_per_gene=per_gene,
    )
