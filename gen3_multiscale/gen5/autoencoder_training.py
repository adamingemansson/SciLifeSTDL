"""Training-only expression-autoencoder trainer -- GEN5_CONTRACT.md
section 3. `train_expression_autoencoder` accepts a single, caller-supplied
training-expression matrix -- there is no argument through which
validation/test expression could reach it, the same "caller's
responsibility, no other-split argument exists" limit
`models/gene_basis.py::fit_gene_residual_basis` already has.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from gen3_multiscale.gen5.autoencoder import ExpressionAutoencoder


@dataclass(frozen=True)
class AutoencoderTrainingReport:
    n_epochs: int
    n_rows: int
    final_train_loss: float
    loss_per_epoch: list[float] = field(default_factory=list)


def train_expression_autoencoder(
    train_expression: np.ndarray, gene_names: list[str], *,
    latent_dim: int = 256, hidden_dim: int = 1024, n_epochs: int = 50, batch_size: int = 64,
    lr: float = 1e-3, weight_decay: float = 0.0, device: str = "cpu", seed: int = 0,
) -> tuple[ExpressionAutoencoder, AutoencoderTrainingReport]:
    train_expression = np.asarray(train_expression, dtype=np.float32)
    if train_expression.ndim != 2:
        raise ValueError(f"train_expression must be [N, n_genes], got shape {train_expression.shape}")
    n_rows, n_genes = train_expression.shape
    if n_genes != len(gene_names):
        raise ValueError(f"train_expression has {n_genes} gene columns but {len(gene_names)} gene_names were given")
    if n_rows < 2:
        raise ValueError("need at least 2 training rows to fit an autoencoder")
    if not np.all(np.isfinite(train_expression)):
        raise ValueError("train_expression contains non-finite values")
    if latent_dim <= 0 or hidden_dim <= 0 or n_epochs <= 0 or batch_size <= 0:
        raise ValueError("latent_dim, hidden_dim, n_epochs, and batch_size must all be positive")

    torch.manual_seed(seed)
    autoencoder = ExpressionAutoencoder(n_genes, gene_names, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)

    autoencoder.train()
    loss_per_epoch = []
    for _epoch in range(n_epochs):
        perm = torch.randperm(n_rows, generator=generator)
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, n_rows, batch_size):
            # Slice on CPU first, then transfer only one batch. A real
            # HEST manifest can contain several GB of full-gene
            # expression; moving the complete matrix to CUDA defeated
            # the purpose of batching and could OOM before epoch 1.
            rows = perm[start:start + batch_size].numpy()
            batch = torch.as_tensor(
                np.asarray(train_expression[rows], dtype=np.float32),
                dtype=torch.float32,
                device=device,
            )
            reconstructed = autoencoder(batch)
            loss = torch.nn.functional.mse_loss(reconstructed, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        loss_per_epoch.append(epoch_loss / max(n_batches, 1))

    report = AutoencoderTrainingReport(
        n_epochs=n_epochs, n_rows=n_rows, final_train_loss=loss_per_epoch[-1] if loss_per_epoch else float("nan"),
        loss_per_epoch=loss_per_epoch,
    )
    return autoencoder, report
