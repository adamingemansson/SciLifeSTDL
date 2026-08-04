"""Bounded TensorBoard diagnostics for conditional-WAE validation.

The posterior embedding uses held-out target GEX only as a diagnostic.  It is
never passed to the image-conditioned predictor and never affects checkpoint
selection.  All cohorts are fixed by the deterministic validation schedule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


METADATA_HEADER = [
    "sample_id", "patient_id", "organ", "stratum", "query_fingerprint",
    "barcode", "x", "y", "image_available", "query_status",
]


def _even_positions(length: int, limit: int) -> np.ndarray:
    if length <= limit:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, limit, dtype=np.int64))


def _pca_rgb(values: np.ndarray) -> np.ndarray:
    """Deterministic three-component PCA rendered with robust RGB scaling."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not len(values):
        raise ValueError("PCA values must be a non-empty rank-2 matrix")
    if not np.isfinite(values).all():
        raise ValueError("PCA values must be finite")
    centered = values - values.mean(axis=0, keepdims=True)
    rank = min(3, centered.shape[0], centered.shape[1])
    if rank:
        # The fixed cohort is bounded; this avoids another runtime dependency.
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ vh[:rank].T
    else:  # pragma: no cover - guarded by non-empty input
        projected = np.zeros((len(values), 0), dtype=np.float32)
    rgb = np.zeros((len(values), 3), dtype=np.float32)
    rgb[:, :rank] = projected
    for channel in range(3):
        low, high = np.percentile(rgb[:, channel], [1.0, 99.0])
        if high > low:
            rgb[:, channel] = np.clip((rgb[:, channel] - low) / (high - low), 0.0, 1.0)
        else:
            rgb[:, channel] = 0.5
    return rgb


@dataclass
class ConditionalWAESnapshotAccumulator:
    """Collect a deterministic, memory-bounded validation cohort."""

    sample_records: dict
    gene_names: list[str]
    logged_gene_indices: list[int]
    max_points: int = 5000
    max_points_per_item: int = 64
    thumbnail_max_points: int = 512
    thumbnail_size: int = 48
    metadata: list[list[str]] = field(default_factory=list)
    posterior_z: list[np.ndarray] = field(default_factory=list)
    context: list[np.ndarray] = field(default_factory=list)
    coords: list[np.ndarray] = field(default_factory=list)
    sample_ids: list[np.ndarray] = field(default_factory=list)
    point_rmse: list[np.ndarray] = field(default_factory=list)
    prediction_genes: list[np.ndarray] = field(default_factory=list)
    target_genes: list[np.ndarray] = field(default_factory=list)
    thumbnails: list[np.ndarray] = field(default_factory=list)
    points_by_sample: dict[str, int] = field(default_factory=dict)
    n_points: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("max_points", self.max_points),
            ("max_points_per_item", self.max_points_per_item),
            ("thumbnail_max_points", self.thumbnail_max_points),
            ("thumbnail_size", self.thumbnail_size),
        ):
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
        if any(index < 0 or index >= len(self.gene_names) for index in self.logged_gene_indices):
            raise ValueError("logged gene index is outside the frozen gene panel")

    @property
    def full(self) -> bool:
        return self.n_points >= self.max_points

    def add(self, *, inputs, target: torch.Tensor, identity: dict,
            prediction: dict, posterior_z: torch.Tensor, sample) -> None:
        if self.full:
            return
        sample_id = str(sample.sample_id)
        per_sample_limit = int(np.ceil(self.max_points / max(1, len(self.sample_records))))
        sample_remaining = per_sample_limit - self.points_by_sample.get(sample_id, 0)
        if sample_remaining <= 0:
            return
        query_full_indices = np.flatnonzero(np.asarray(inputs.query_mask, dtype=bool))
        n_query = len(query_full_indices)
        if target.shape[0] != n_query:
            raise ValueError("snapshot target rows do not align with query mask")
        take = min(
            self.max_points_per_item, self.max_points - self.n_points, sample_remaining,
        )
        local = _even_positions(n_query, take)
        full = query_full_indices[local]
        count = len(local)
        record = self.sample_records.get(sample_id, {})
        raw_coords = np.asarray(sample.full_sample_coords, dtype=np.float32)[full]
        available = np.asarray(sample.image_source_available, dtype=bool)[full]
        barcodes = np.asarray(sample.adata.obs_names, dtype=str)[full]
        stratum = str(identity.get("stratum", "unknown"))
        fingerprint = str(identity.get("query_fingerprint", "unknown"))
        for row in range(count):
            self.metadata.append([
                sample_id, str(sample.patient_id), str(record.get("organ", "unknown")),
                stratum, fingerprint, str(barcodes[row]),
                f"{raw_coords[row, 0]:.6g}", f"{raw_coords[row, 1]:.6g}",
                str(bool(available[row])), "query",
            ])
        local_index = torch.as_tensor(local, dtype=torch.long, device=target.device)
        predictive_mean = prediction["predictive_mean"].detach().index_select(0, local_index)
        selected_target = target.detach().index_select(0, local_index)
        self.posterior_z.append(
            posterior_z.detach().index_select(0, local_index).cpu().float().numpy()
        )
        self.context.append(
            prediction["image_context"].detach().index_select(0, local_index)
            .cpu().float().numpy()
        )
        self.coords.append(raw_coords)
        self.sample_ids.append(np.full(count, sample_id, dtype=object))
        self.point_rmse.append(
            torch.sqrt(torch.mean((predictive_mean - selected_target).square(), dim=1))
            .cpu().float().numpy()
        )
        if self.logged_gene_indices:
            gene_index = torch.as_tensor(
                self.logged_gene_indices, dtype=torch.long, device=predictive_mean.device,
            )
            self.prediction_genes.append(
                predictive_mean.index_select(1, gene_index).cpu().float().numpy()
            )
            self.target_genes.append(
                selected_target.index_select(1, gene_index).cpu().float().numpy()
            )
        if self.n_points < self.thumbnail_max_points:
            thumbnail_take = min(count, self.thumbnail_max_points - self.n_points)
            patches = np.asarray(sample.patches)[full[:thumbnail_take]]
            if patches.ndim != 4 or patches.shape[-1] != 3:
                raise ValueError("aligned H&E patches must have shape [N, H, W, 3]")
            tensor = torch.as_tensor(patches).permute(0, 3, 1, 2).float() / 255.0
            tensor = torch.nn.functional.interpolate(
                tensor, size=(self.thumbnail_size, self.thumbnail_size),
                mode="bilinear", align_corners=False,
            )
            self.thumbnails.append(tensor.clamp(0.0, 1.0).numpy())
        self.n_points += count
        self.points_by_sample[sample_id] = self.points_by_sample.get(sample_id, 0) + count

    def arrays(self) -> dict:
        if not self.n_points:
            raise ValueError("cannot log an empty validation snapshot")
        if len(self.metadata) != self.n_points:
            raise ValueError("snapshot metadata and embedding rows are misaligned")
        result = {
            "posterior_z": np.concatenate(self.posterior_z),
            "context": np.concatenate(self.context),
            "coords": np.concatenate(self.coords),
            "sample_ids": np.concatenate(self.sample_ids),
            "point_rmse": np.concatenate(self.point_rmse),
        }
        result["prediction_genes"] = (
            np.concatenate(self.prediction_genes)
            if self.prediction_genes else np.empty((self.n_points, 0), dtype=np.float32)
        )
        result["target_genes"] = (
            np.concatenate(self.target_genes)
            if self.target_genes else np.empty((self.n_points, 0), dtype=np.float32)
        )
        result["thumbnails"] = (
            np.concatenate(self.thumbnails)
            if self.thumbnails else np.empty((0, 3, self.thumbnail_size, self.thumbnail_size))
        )
        for name in (
            "posterior_z", "context", "coords", "point_rmse",
            "prediction_genes", "target_genes", "thumbnails",
        ):
            if not np.isfinite(result[name]).all():
                raise ValueError(f"snapshot {name} contains a non-finite value")
        return result


class ConditionalWAETensorBoardLogger:
    """Write scalars, Projector cohorts, H&E thumbnails and spatial maps."""

    def __init__(self, log_dir: str | Path, *, writer=None,
                 max_spatial_samples: int = 4, purge_step: int | None = None):
        self.log_dir = Path(log_dir)
        self.max_spatial_samples = int(max_spatial_samples)
        if self.max_spatial_samples < 1:
            raise ValueError("max_spatial_samples must be positive")
        if writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:  # fail clearly when explicitly enabled
                raise RuntimeError(
                    "TensorBoard logging is enabled but tensorboard is not installed"
                ) from exc
            writer = SummaryWriter(
                log_dir=str(self.log_dir),
                purge_step=(int(purge_step) if purge_step is not None else None),
            )
        self.writer = writer

    def add_train_scalars(self, step: int, losses: dict, *, grad_norm,
                          discriminator_loss=None, learning_rate=None) -> None:
        values = {
            "train/total": losses["total"],
            "train/reconstruction": losses["reconstruction_loss"],
            "train/reconstruction_rmse": losses["reconstruction_rmse"],
            "train/reconstruction_pcc_loss": losses["reconstruction_pcc_loss"],
            "train/conditional_mean": losses["conditional_mean_loss"],
            "train/conditional_mean_rmse": losses["conditional_mean_rmse"],
            "train/prior": losses["prior_loss"],
            "train/generator_grad_norm": grad_norm,
        }
        if discriminator_loss is not None:
            values["train/discriminator"] = discriminator_loss
        if learning_rate is not None:
            values["train/learning_rate"] = learning_rate
        for tag, value in values.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            self.writer.add_scalar(tag, float(value), int(step))

    def add_validation_scalars(self, step: int, entry: dict) -> None:
        for key in ("total", "rmse", "pcc_loss", "conditional_mean_rmse"):
            self.writer.add_scalar(f"validation/{key}", float(entry[key]), int(step))

    def add_snapshot(self, step: int, snapshot: ConditionalWAESnapshotAccumulator) -> None:
        arrays = snapshot.arrays()
        metadata = snapshot.metadata
        posterior = torch.as_tensor(arrays["posterior_z"])
        context = torch.as_tensor(arrays["context"])
        self.writer.add_embedding(
            posterior, metadata=metadata, metadata_header=METADATA_HEADER,
            global_step=int(step), tag="embeddings/posterior_z_target_diagnostic",
        )
        self.writer.add_embedding(
            context, metadata=metadata, metadata_header=METADATA_HEADER,
            global_step=int(step), tag="embeddings/conditional_context",
        )
        thumbnails = torch.as_tensor(arrays["thumbnails"])
        n_thumbnails = len(thumbnails)
        if n_thumbnails:
            thumbnail_metadata = metadata[:n_thumbnails]
            self.writer.add_embedding(
                posterior[:n_thumbnails], metadata=thumbnail_metadata,
                metadata_header=METADATA_HEADER, label_img=thumbnails,
                global_step=int(step), tag="embeddings/posterior_z_with_he",
            )
            self.writer.add_embedding(
                context[:n_thumbnails], metadata=thumbnail_metadata,
                metadata_header=METADATA_HEADER, label_img=thumbnails,
                global_step=int(step), tag="embeddings/context_with_he",
            )
        self._add_spatial_figures(step, snapshot, arrays)
        self.writer.flush()

    def _add_spatial_figures(self, step: int, snapshot, arrays: dict) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cohort_limit = min(len(arrays["context"]), 2000)
        context_rgb = _pca_rgb(arrays["context"][:cohort_limit])
        posterior_rgb = _pca_rgb(arrays["posterior_z"][:cohort_limit])
        sample_ids = arrays["sample_ids"][:cohort_limit]
        coords = arrays["coords"][:cohort_limit]
        ordered_samples = list(dict.fromkeys(map(str, sample_ids)))[:self.max_spatial_samples]
        for sample_id in ordered_samples:
            mask = sample_ids == sample_id
            for name, colors in (("context_pca_rgb", context_rgb),
                                 ("posterior_z_pca_rgb", posterior_rgb)):
                fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
                ax.scatter(coords[mask, 0], coords[mask, 1], c=colors[mask], s=12)
                ax.invert_yaxis()
                ax.set_aspect("equal", adjustable="datalim")
                ax.set_title(f"{sample_id}: {name}")
                ax.set_axis_off()
                self.writer.add_figure(f"spatial/{sample_id}/{name}", fig, step, close=True)
            all_mask = arrays["sample_ids"] == sample_id
            full_coords = arrays["coords"][all_mask]
            fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
            plot = ax.scatter(
                full_coords[:, 0], full_coords[:, 1], c=arrays["point_rmse"][all_mask],
                cmap="magma", s=12,
            )
            fig.colorbar(plot, ax=ax, label="per-spot RMSE")
            ax.invert_yaxis()
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_title(f"{sample_id}: predictive error")
            ax.set_axis_off()
            self.writer.add_figure(f"spatial/{sample_id}/predictive_rmse", fig, step, close=True)
            for gene_column, gene_index in enumerate(snapshot.logged_gene_indices):
                true = arrays["target_genes"][all_mask, gene_column]
                predicted = arrays["prediction_genes"][all_mask, gene_column]
                low = float(min(true.min(), predicted.min()))
                high = float(max(true.max(), predicted.max()))
                gene = snapshot.gene_names[gene_index]
                fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
                for ax, values, title, cmap, limits in (
                    (axes[0], true, "target", "viridis", (low, high)),
                    (axes[1], predicted, "prediction", "viridis", (low, high)),
                    (axes[2], np.abs(predicted - true), "absolute error", "magma", (None, None)),
                ):
                    kwargs = {"cmap": cmap, "s": 12}
                    if limits[0] is not None and limits[1] > limits[0]:
                        kwargs.update(vmin=limits[0], vmax=limits[1])
                    scatter = ax.scatter(full_coords[:, 0], full_coords[:, 1], c=values, **kwargs)
                    fig.colorbar(scatter, ax=ax)
                    ax.invert_yaxis()
                    ax.set_aspect("equal", adjustable="datalim")
                    ax.set_title(title)
                    ax.set_axis_off()
                fig.suptitle(f"{sample_id}: {gene}")
                self.writer.add_figure(
                    f"spatial/{sample_id}/genes/{gene}", fig, step, close=True,
                )

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()


class ConditionalFlowTensorBoardLogger(ConditionalWAETensorBoardLogger):
    """The same bounded MK diagnostics with flow-specific training scalars."""

    def add_train_scalars(self, step: int, losses: dict, *, grad_norm,
                          learning_rate=None) -> None:
        values = {
            "train/total": losses["total"],
            "train/reconstruction": losses["reconstruction_loss"],
            "train/reconstruction_rmse": losses["reconstruction_rmse"],
            "train/reconstruction_pcc_loss": losses["reconstruction_pcc_loss"],
            "train/conditional_mean": losses["conditional_mean_loss"],
            "train/conditional_mean_rmse": losses["conditional_mean_rmse"],
            "train/flow": losses["flow_loss"],
            "train/gradient_norm": grad_norm,
        }
        if learning_rate is not None:
            values["train/learning_rate"] = learning_rate
        for tag, value in values.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            self.writer.add_scalar(tag, float(value), int(step))
