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

from gen3_multiscale.conditional_wae.reference_projection import assign_clusters, project_onto_reference
from gen3_multiscale.evaluation.metrics import pearson_per_gene


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


def _pca_2d(values: np.ndarray) -> np.ndarray:
    """Deterministic 2-component PCA (no fixed/persisted basis -- this is a
    per-snapshot diagnostic view of the CURRENT latent geometry, not a
    quantity compared across steps like `_fixed_pca_rgb`'s spatial maps
    are). Reuses `_pca_rgb`'s own SVD-based projection machinery, just
    keeping 2 components instead of 3 and skipping the RGB rescale."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not len(values):
        raise ValueError("PCA values must be a non-empty rank-2 matrix")
    if not np.isfinite(values).all():
        raise ValueError("PCA values must be finite")
    centered = values - values.mean(axis=0, keepdims=True)
    rank = min(2, centered.shape[0], centered.shape[1])
    projected = np.zeros((len(values), 2), dtype=np.float32)
    if rank:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        projected[:, :rank] = centered @ vh[:rank].T
    return projected


def _fixed_pca_rgb(pca_coords: np.ndarray, pc_ranges: np.ndarray) -> np.ndarray:
    """RGB from up to 3 PCA coordinates, scaled by FIXED, persisted ranges --
    never percentile-rescaled per call like `_pca_rgb` above. A color change
    between snapshots then only ever reflects a real change in the
    underlying coordinates, not a re-fit/re-oriented/re-scaled basis."""
    pca_coords = np.asarray(pca_coords, dtype=np.float32)
    pc_ranges = np.asarray(pc_ranges, dtype=np.float32)
    n_components = min(3, pca_coords.shape[1])
    rgb = np.full((len(pca_coords), 3), 0.5, dtype=np.float32)
    for channel in range(n_components):
        low, high = pc_ranges[channel]
        if high > low:
            rgb[:, channel] = np.clip((pca_coords[:, channel] - low) / (high - low), 0.0, 1.0)
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
        point_prediction = prediction["point_prediction"].detach().index_select(0, local_index)
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
            torch.sqrt(torch.mean((point_prediction - selected_target).square(), dim=1))
            .cpu().float().numpy()
        )
        if self.logged_gene_indices:
            gene_index = torch.as_tensor(
                self.logged_gene_indices, dtype=torch.long, device=point_prediction.device,
            )
            self.prediction_genes.append(
                point_prediction.index_select(1, gene_index).cpu().float().numpy()
            )
            self.target_genes.append(
                selected_target.index_select(1, gene_index).cpu().float().numpy()
            )
        # H&E thumbnails are the only consumer of raw pixels after the
        # spot-feature cache has verified them, so they are skipped when a run
        # released the patch array (data.retain_patches_in_memory=false).
        # Every scalar, panel and spatial-map diagnostic is unaffected; only
        # the thumbnail strip is absent, which is the intended trade for
        # ~75 GB of resident pixels per process.
        if self.n_points < self.thumbnail_max_points and sample.patches is not None:
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
                          discriminator_loss=None, discriminator_grad_norm=None,
                          learning_rate=None, masks_seen=None,
                          gradient_accumulation_steps=None) -> None:
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
        if discriminator_grad_norm is not None:
            values["train/discriminator_grad_norm"] = discriminator_grad_norm
        if learning_rate is not None:
            values["train/learning_rate"] = learning_rate
        for tag, value in values.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            self.writer.add_scalar(tag, float(value), int(step))
        # Adam's four-arm WAE-GAN ablation: gradient_accumulation_steps
        # changes how many raw masks one optimizer step represents, so
        # `masks_seen` is logged as its own scalar (plotted against the
        # SAME optimizer-step x-axis every other train/* tag uses) --
        # comparing arms with different accumulation by masks_seen means
        # reading this tag, not just the x-axis, which only ever counts
        # optimizer steps.
        if masks_seen is not None:
            self.writer.add_scalar("train/masks_seen", float(masks_seen), int(step))
        self.writer.add_scalar("train/optimizer_step", float(step), int(step))
        if gradient_accumulation_steps is not None:
            self.writer.add_scalar(
                "train/gradient_accumulation_steps", float(gradient_accumulation_steps), int(step),
            )

    def add_validation_scalars(self, step: int, entry: dict, *,
                               best_total: float | None = None, best_step: int | None = None) -> None:
        for key in ("total", "rmse", "pcc_loss"):
            self.writer.add_scalar(f"validation/{key}", float(entry[key]), int(step))
            self.writer.add_scalar(f"validation/point_{key}", float(entry[key]), int(step))
        for key in ("wae_prior_total", "wae_prior_rmse", "wae_prior_pcc_loss"):
            if key in entry:
                self.writer.add_scalar(f"validation/{key}", float(entry[key]), int(step))
        if best_total is not None:
            self.writer.add_scalar("validation/best_total", float(best_total), int(step))
        if best_step is not None:
            self.writer.add_scalar("validation/best_step", float(best_step), int(step))

    def add_whole_slide_scalars(
        self, step: int, *, whole_slide_total: float, whole_slide_rmse: float, whole_slide_pcc_loss: float,
        whole_slide_hvg50_pcc_loss: float | None = None,
        best_whole_slide_total: float | None = None, best_whole_slide_step: int | None = None,
    ) -> None:
        """Same total/rmse/pcc_loss shape as `add_validation_scalars`, pooled
        over every predicted whole-slide spot. Diagnostic only -- never read
        back for loss or checkpoint selection beyond `whole_slide_total`,
        itself a diagnostic-only secondary criterion (see best_whole_slide/
        in the checkpoint dir). `whole_slide_hvg50_pcc_loss` is the SAME
        pooled pcc_loss, restricted to the `train_log1p_variance_top50`
        panel's gene columns -- a single summary number for "how well are
        we doing on the genes that actually vary," which a huge
        low-variance gene majority can otherwise dilute in the full-panel
        pcc_loss above."""
        self.writer.add_scalar("whole_slide/total", float(whole_slide_total), int(step))
        self.writer.add_scalar("whole_slide/rmse", float(whole_slide_rmse), int(step))
        self.writer.add_scalar("whole_slide/pcc_loss", float(whole_slide_pcc_loss), int(step))
        self.writer.add_scalar("whole_slide/point_total", float(whole_slide_total), int(step))
        self.writer.add_scalar("whole_slide/point_rmse", float(whole_slide_rmse), int(step))
        self.writer.add_scalar("whole_slide/point_pcc_loss", float(whole_slide_pcc_loss), int(step))
        if whole_slide_hvg50_pcc_loss is not None:
            self.writer.add_scalar(
                "whole_slide/hvg50_pcc_loss", float(whole_slide_hvg50_pcc_loss), int(step),
            )
        if best_whole_slide_total is not None:
            self.writer.add_scalar(
                "whole_slide/best_total", float(best_whole_slide_total), int(step),
            )
        if best_whole_slide_step is not None:
            self.writer.add_scalar(
                "whole_slide/best_step", float(best_whole_slide_step), int(step),
            )

    def add_whole_slide_spatial_maps(
        self, step: int, sample_id: str, coords: np.ndarray, true_gex: np.ndarray,
        predicted_gex: np.ndarray, gene_names: list[str], reference_projection: dict,
        *, gene_indices: list[int] | None = None,
    ) -> None:
        """Whole-slide spatial diagnostics: PC1/PC2/PC3 as three SEPARATE
        maps with a fixed diverging scale + explained variance in the
        title (primary), an RGB composite from the same frozen basis
        (secondary), and a fixed-cluster categorical map -- for both true
        and predicted GEX, so they are directly visually comparable. Every
        tissue spot is plotted (whole-slide coverage is already 100%, so
        there is no query-only subset to distinguish from a background
        lattice). `reference_projection` MUST be the same frozen basis
        (reference_projection.ensure_reference_gex_projection) across every
        step/architecture/WAE dimensionality being compared.

        `gene_indices` (optional) additionally plots the RAW target/
        predicted/absolute-error value for each given gene column over
        every spot on the slide -- the whole-slide analogue of
        `_add_spatial_figures`'s per-gene panel, which only ever covers
        the masked validation query subset. Each such gene also gets a
        scalar `whole_slide/{sample_id}/genes/{gene}/pcc` (this slide's
        real per-gene Pearson correlation, same convention as
        evaluation.metrics.pearson_per_gene: undefined/skipped when the
        true values are constant) so PCC can be tracked as a line chart
        over training steps alongside watching the spatial maps evolve."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        coords = np.asarray(coords, dtype=np.float32)
        pc_ranges = np.asarray(reference_projection["pc_ranges"], dtype=np.float32)
        explained = np.asarray(reference_projection["explained_variance_ratio"], dtype=np.float32)
        n_clusters = int(reference_projection["n_clusters"])
        n_components = pc_ranges.shape[0]

        for label, gex in (("true", true_gex), ("predicted", predicted_gex)):
            pca_coords = project_onto_reference(gex, gene_names, reference_projection)
            for component in range(n_components):
                low, high = pc_ranges[component]
                fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
                scatter = ax.scatter(
                    coords[:, 0], coords[:, 1], c=pca_coords[:, component],
                    cmap="RdBu_r", vmin=low, vmax=high, s=8,
                )
                fig.colorbar(scatter, ax=ax)
                ax.invert_yaxis()
                ax.set_aspect("equal", adjustable="datalim")
                ax.set_title(
                    f"{sample_id}: {label} PC{component + 1} "
                    f"(explained variance {explained[component]:.1%})"
                )
                ax.set_axis_off()
                self.writer.add_figure(
                    f"whole_slide/{sample_id}/{label}/pc{component + 1}", fig, step, close=True,
                )

            rgb = _fixed_pca_rgb(pca_coords, pc_ranges)
            fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
            ax.scatter(coords[:, 0], coords[:, 1], c=rgb, s=8)
            ax.invert_yaxis()
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_title(f"{sample_id}: {label} PCA RGB composite (fixed scale, secondary)")
            ax.set_axis_off()
            self.writer.add_figure(
                f"whole_slide/{sample_id}/{label}/pca_rgb_composite", fig, step, close=True,
            )

            clusters = assign_clusters(pca_coords, reference_projection)
            fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
            scatter = ax.scatter(
                coords[:, 0], coords[:, 1], c=clusters, cmap="tab10",
                vmin=0, vmax=max(9, n_clusters - 1), s=8,
            )
            fig.colorbar(scatter, ax=ax, ticks=range(n_clusters))
            ax.invert_yaxis()
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_title(f"{sample_id}: {label} fixed clusters")
            ax.set_axis_off()
            self.writer.add_figure(
                f"whole_slide/{sample_id}/{label}/clusters", fig, step, close=True,
            )

        if gene_indices:
            true_gex = np.asarray(true_gex, dtype=np.float32)
            predicted_gex = np.asarray(predicted_gex, dtype=np.float32)
            for gene_index in gene_indices:
                gene = gene_names[gene_index]
                true_values = true_gex[:, gene_index]
                predicted_values = predicted_gex[:, gene_index]
                gene_pcc = float(
                    pearson_per_gene(predicted_values[:, None], true_values[:, None])[0]
                )
                if np.isfinite(gene_pcc):
                    self.writer.add_scalar(f"whole_slide/{sample_id}/genes/{gene}/pcc", gene_pcc, step)
                # Real bug (user report, Aug 2026): raw min/max let a single
                # outlier spot (real biology -- a sharp local expression
                # spike is normal, not an artifact) stretch vmax and crush
                # every other spot toward the dark end of viridis, making
                # genuinely present signal look like near-total sparsity.
                # 2nd-98th percentile, matching the same percentile-clip
                # convention `_pca_rgb` already uses above in this file.
                combined = np.concatenate([true_values, predicted_values])
                low, high = (float(v) for v in np.percentile(combined, [2.0, 98.0]))
                fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
                for ax, values, title, cmap, limits in (
                    (axes[0], true_values, "target", "viridis", (low, high)),
                    (axes[1], predicted_values, "prediction", "viridis", (low, high)),
                    (axes[2], np.abs(predicted_values - true_values), "absolute error", "magma", (None, None)),
                ):
                    kwargs = {"cmap": cmap, "s": 8}
                    if limits[0] is not None and limits[1] > limits[0]:
                        kwargs.update(vmin=limits[0], vmax=limits[1])
                    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=values, **kwargs)
                    fig.colorbar(scatter, ax=ax)
                    ax.invert_yaxis()
                    ax.set_aspect("equal", adjustable="datalim")
                    ax.set_title(title)
                    ax.set_axis_off()
                fig.suptitle(f"{sample_id}: whole-slide {gene}")
                self.writer.add_figure(
                    f"whole_slide/{sample_id}/genes/{gene}", fig, step, close=True,
                )
        self.writer.flush()

    def add_film_diagnostics(self, step: int, diagnostics: dict) -> None:
        """Collapse-watching scalars/histograms for a FiLM-conditioned
        encoder's validation cohort. Diagnostic only."""
        self.writer.add_scalar(
            "film/posterior_effective_rank", float(diagnostics["effective_rank"]), int(step),
        )
        self.writer.add_scalar(
            "film/posterior_active_dimensions", float(diagnostics["active_dimensions"]), int(step),
        )
        self.writer.add_scalar(
            "film/predictive_std_mean", float(diagnostics["predictive_std_mean"]), int(step),
        )
        self.writer.add_scalar(
            "film/stochastic_vs_conditional_mean_diff",
            float(diagnostics["stochastic_vs_conditional_mean_diff"]), int(step),
        )
        self.writer.add_histogram(
            "film/latent_dim_mean", np.asarray(diagnostics["latent_dim_mean"]), int(step),
        )
        self.writer.add_histogram(
            "film/latent_dim_std", np.asarray(diagnostics["latent_dim_std"]), int(step),
        )
        for layer, (gamma, beta) in diagnostics["gamma_beta"].items():
            self.writer.add_histogram(f"film/gamma_{layer}", np.asarray(gamma), int(step))
            self.writer.add_histogram(f"film/beta_{layer}", np.asarray(beta), int(step))
        self.writer.flush()

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
        self._add_latent_scatter(step, metadata, arrays)
        self.writer.flush()

    def _add_latent_scatter(self, step: int, metadata: list[list[str]], arrays: dict) -> None:
        """A quick-glance 2D PCA scatter of the posterior latent (z), colored
        by organ -- cheaper and more directly readable than the high-dim
        Projector snapshot above for "is the latent space actually
        organizing by tissue" at a glance. `organ` is METADATA_HEADER[2]."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        coords_2d = _pca_2d(arrays["posterior_z"])
        organs = np.asarray([row[2] for row in metadata], dtype=str)
        unique_organs = sorted(set(organs.tolist()))
        organ_index = {organ: index for index, organ in enumerate(unique_organs)}
        colors = np.asarray([organ_index[organ] for organ in organs])

        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        scatter = ax.scatter(
            coords_2d[:, 0], coords_2d[:, 1], c=colors, cmap="tab20",
            vmin=0, vmax=max(19, len(unique_organs) - 1), s=10,
        )
        handles = [
            plt.Line2D(
                [0], [0], marker="o", linestyle="", color=scatter.cmap(scatter.norm(index)),
                label=organ,
            )
            for organ, index in organ_index.items()
        ]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
        ax.set_title("posterior z: 2D PCA, colored by organ")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        self.writer.add_figure("embeddings/posterior_z_pca_2d", fig, step, close=True)

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
