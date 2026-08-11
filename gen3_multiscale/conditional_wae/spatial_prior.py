"""Self-supervised pretraining of the spatial refiner on EXPRESSION ALONE.

The idea (Adam, Aug 2026): learn what real expression FIELDS look like -- their
spatial autocorrelation, tissue-domain structure, gene-gene coupling and
realistic sparsity -- from ST data with no image involved, then hand those
weights to the image-conditioned refiner as an initialisation.

Two properties make this worth a separate stage rather than more paired
training:

* **Far more supervision per slide.** The paired task yields one supervised
  example per masked hole. Masking spots and predicting them from their
  neighbours yields a fresh example for every masking pattern over every spot
  of every slide, and it never consumes the image-pairing budget.
* **It is a different kind of knowledge.** The image can say "this region is
  stroma"; only expression can say "a spot whose neighbours look like THIS
  usually looks like THAT". Measured evidence says the image pathway already
  saturates around tissue-compartment identity, so the gene-gene/spatial
  structure has to come from somewhere else.

The honest limitation, stated up front. At inference in the ``he_to_st`` task
there is no observed neighbour expression, so the pretrained module attends to
the model's OWN predictions -- this is a better-initialised iterative refiner,
not an oracle. It becomes far more powerful in the ``he_plus_st_to_st``
imputation arms, where real neighbour expression IS visible; those arms already
exist in the contract and are the natural second use of this artifact.

Leakage discipline: pretraining reads TRAIN-split expression only. A spot's own
expression is always masked out of its own input, so the task can never be
solved by copying the answer, and validation/test slides are never opened.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from gen3_multiscale.conditional_wae.spatial_refinement import (
    SpatialExpressionRefiner,
    padded_neighbor_graph,
)

SPATIAL_PRIOR_FORMAT = "conditional_wae_spatial_expression_prior_v1"


def mask_spots(n_spots: int, mask_fraction: float, generator: torch.Generator
               ) -> torch.Tensor:
    """Boolean [n_spots] mask selecting the spots to hide and predict."""
    if not 0.0 < mask_fraction < 1.0:
        raise ValueError("mask_fraction must be in (0, 1)")
    if n_spots < 2:
        raise ValueError("masked-spot pretraining needs at least two spots")
    n_masked = max(1, min(n_spots - 1, int(round(n_spots * mask_fraction))))
    order = torch.randperm(n_spots, generator=generator)
    mask = torch.zeros(n_spots, dtype=torch.bool)
    mask[order[:n_masked]] = True
    return mask


def masked_spot_loss(refiner: SpatialExpressionRefiner, expression: torch.Tensor,
                     coords: torch.Tensor, spot_mask: torch.Tensor) -> torch.Tensor:
    """One self-supervised step: hide `spot_mask`'s spots, predict them back.

    The masked spots enter the refiner with their expression ZEROED, so their
    own values cannot leak into their own prediction -- the refiner may only
    reach them through the neighbour attention. ``context`` is zeros because
    this stage is deliberately image-free; the same module later receives real
    image context, and a zero context is exactly the "no image information"
    limit of that input.

    A consequence worth stating rather than discovering later: with a zero
    context, ``context_proj``'s WEIGHT receives no gradient here (only its
    bias does), so the image pathway of the refiner leaves this stage at its
    random initialisation. That is the intended division of labour -- this
    stage teaches the neighbour attention and the update head what real
    expression fields look like; the image pathway is learned in the paired
    stage, where a real context exists.
    """
    if expression.ndim != 2:
        raise ValueError("expression must be [n_spots, n_genes]")
    if spot_mask.shape[0] != expression.shape[0]:
        raise ValueError("spot_mask must have one entry per spot")
    if not bool(spot_mask.any()):
        raise ValueError("spot_mask must select at least one spot")
    visible = expression.clone()
    visible[spot_mask] = 0.0
    context = torch.zeros(
        expression.shape[0], refiner.context_proj.in_features,
        dtype=expression.dtype, device=expression.device,
    )
    neighbor_indices, neighbor_mask = refiner.cached_neighbor_graph(coords)
    predicted = refiner(visible, context, coords, neighbor_indices, neighbor_mask)
    return torch.nn.functional.mse_loss(predicted[spot_mask], expression[spot_mask])


def save_spatial_prior(refiner: SpatialExpressionRefiner, path: str | Path, *,
                       gene_names: list[str], provenance: dict) -> Path:
    """Atomically write the pretrained weights with an identity record.

    The gene-name hash is stored so loading into a model with a different gene
    panel fails loudly rather than silently mismatching columns -- the same
    fail-closed discipline the spot-feature and coexpression-basis caches use.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": SPATIAL_PRIOR_FORMAT,
        "state_dict": refiner.state_dict(),
        "n_genes": int(refiner.n_genes),
        "k_neighbors": int(refiner.k_neighbors),
        "context_dim": int(refiner.context_proj.in_features),
        "gex_feature_dim": int(refiner.gene_encoder.projection.out_features),
        "hidden_dim": int(refiner.context_proj.out_features),
        "geometry_dim": int(refiner.geometry_mlp[0].out_features),
        "gene_names_sha256": hashlib.sha256(
            json.dumps(list(gene_names), sort_keys=False).encode()
        ).hexdigest(),
        "provenance": dict(provenance),
    }
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_spatial_prior_into(refiner: SpatialExpressionRefiner, path: str | Path, *,
                            gene_names: list[str]) -> dict:
    """Load pretrained weights into `refiner`, failing closed on any mismatch."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format") != SPATIAL_PRIOR_FORMAT:
        raise ValueError(
            f"{path}: not a {SPATIAL_PRIOR_FORMAT} artifact (got {payload.get('format')!r})"
        )
    expected = hashlib.sha256(
        json.dumps(list(gene_names), sort_keys=False).encode()
    ).hexdigest()
    if payload["gene_names_sha256"] != expected:
        raise ValueError(
            f"{path}: gene panel differs from the one this prior was trained on -- "
            "the gene columns would be silently misaligned"
        )
    for field, actual in (
        ("n_genes", refiner.n_genes),
        ("k_neighbors", refiner.k_neighbors),
        ("context_dim", refiner.context_proj.in_features),
        ("gex_feature_dim", refiner.gene_encoder.projection.out_features),
        ("hidden_dim", refiner.context_proj.out_features),
        ("geometry_dim", refiner.geometry_mlp[0].out_features),
    ):
        # Every dimension that shapes a parameter is checked here, so a
        # mismatch raises this readable error rather than surfacing as
        # load_state_dict's wall of size-mismatch text.
        if int(payload[field]) != int(actual):
            raise ValueError(
                f"{path}: {field}={payload[field]} does not match the model's {actual}"
            )
    refiner.load_state_dict(payload["state_dict"])
    return {
        "format": payload["format"],
        "gene_names_sha256": payload["gene_names_sha256"],
        "provenance": payload.get("provenance", {}),
    }


def masked_spot_pearson(refiner: SpatialExpressionRefiner, expression: torch.Tensor,
                        coords: torch.Tensor, spot_mask: torch.Tensor, *,
                        n_top_genes: int = 200) -> dict[str, float]:
    """Per-gene Pearson across the MASKED spots, over two gene sets.

    Reported alongside the MSE so the pretraining log is readable in the same
    currency as every downstream number, and so a run that merely drives MSE
    down by predicting each gene's slide mean is visible as ~0 here (genes
    with no variance across the masked spots are excluded, exactly as
    ``pearson_per_gene`` does).

    Both a full-panel and a top-variance-gene average are returned. The
    full-panel mean over ~17,000 genes is dominated by near-silent ones and
    moves very little even when the model is genuinely learning; the headline
    metrics in this study are HVG panels, so ``top_variance_genes`` is the
    number to watch. They differ by roughly an order of magnitude on the same
    data -- a parameter-free neighbour-mean predictor scores 0.056 full-panel
    against 0.392 on the top 200 -- so quoting the wrong one badly misreads a
    run's progress.
    """
    with torch.no_grad():
        visible = expression.clone()
        visible[spot_mask] = 0.0
        context = torch.zeros(
            expression.shape[0], refiner.context_proj.in_features,
            dtype=expression.dtype, device=expression.device,
        )
        neighbor_indices, neighbor_mask = refiner.cached_neighbor_graph(coords)
        predicted = refiner(visible, context, coords, neighbor_indices, neighbor_mask)
        truth = expression[spot_mask].double()
        estimate = predicted[spot_mask].double()
        truth_centered = truth - truth.mean(dim=0, keepdim=True)
        estimate_centered = estimate - estimate.mean(dim=0, keepdim=True)
        truth_norm = truth_centered.norm(dim=0)
        estimate_norm = estimate_centered.norm(dim=0)
        valid = (truth_norm > 1e-12) & (estimate_norm > 1e-12)
        if not bool(valid.any()):
            return {"all_genes": float("nan"), "top_variance_genes": float("nan")}
        correlation = (truth_centered * estimate_centered).sum(dim=0) / (truth_norm * estimate_norm)
        top_count = max(1, min(int(n_top_genes), int(valid.sum())))
        # Rank by variance in the TRUTH, so the gene set never depends on what
        # the model happened to predict.
        ranked = torch.argsort(torch.where(valid, truth_norm, truth_norm.new_zeros(())),
                               descending=True)[:top_count]
        return {
            "all_genes": float(correlation[valid].mean()),
            "top_variance_genes": float(correlation[ranked].mean()),
        }


def _spatial_prior_step(refiner: SpatialExpressionRefiner,
                        optimizer: torch.optim.Optimizer,
                        expression: torch.Tensor, coords: torch.Tensor, *,
                        mask_fraction: float, generator: torch.Generator) -> float:
    spot_mask = mask_spots(expression.shape[0], mask_fraction, generator).to(expression.device)
    loss = masked_spot_loss(refiner, expression, coords, spot_mask)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite spatial-prior loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


def pretrain_spatial_prior(refiner: SpatialExpressionRefiner,
                           slides: list[tuple[np.ndarray, np.ndarray]], *,
                           steps: int, mask_fraction: float = 0.25,
                           learning_rate: float = 1e-4, seed: int = 0,
                           device: torch.device | None = None,
                           log_every: int = 50) -> list[float]:
    """Run `steps` masked-spot updates over in-memory `slides`.

    Each slide is a ``(expression [S, G], coords [S, >=2])`` pair. Every slide
    is held in memory for the whole run, so this entrypoint is for tests and
    small panels; the production path is ``pretrain_spatial_prior_streaming``,
    which holds one slide at a time.

    Returns the per-logged-step losses so a caller can assert the task is
    actually being learned rather than merely running.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    if not slides:
        raise ValueError("at least one slide is required")
    device = device or torch.device("cpu")
    refiner = refiner.to(device)
    optimizer = torch.optim.Adam(refiner.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(int(seed))
    history: list[float] = []
    for step in range(steps):
        expression_np, coords_np = slides[step % len(slides)]
        expression = torch.as_tensor(expression_np, dtype=torch.float32, device=device)
        coords = torch.as_tensor(coords_np, dtype=torch.float32, device=device)
        value = _spatial_prior_step(
            refiner, optimizer, expression, coords,
            mask_fraction=mask_fraction, generator=generator,
        )
        if step % max(1, int(log_every)) == 0 or step == steps - 1:
            history.append(value)
            print(f"spatial-prior pretraining step {step + 1}/{steps} loss={value:.6f}", flush=True)
    return history


def pretrain_spatial_prior_streaming(refiner: SpatialExpressionRefiner,
                                     sample_ids: list[str],
                                     load_slide, *, rounds: int,
                                     steps_per_slide: int = 4,
                                     mask_fraction: float = 0.25,
                                     learning_rate: float = 1e-4, seed: int = 0,
                                     n_top_genes: int = 200,
                                     validation_ids: list[str] | None = None,
                                     device: torch.device | None = None) -> list[dict]:
    """Pretrain over slides loaded ONE AT A TIME, holding at most one in memory.

    ``load_slide(sample_id)`` returns ``(expression [S, G], coords [S, >=2])``.
    A full-panel slide is ~2,500 spots x 17,189 genes x 4 bytes ~= 170 MB, so
    keeping a two-hundred-slide corpus resident would cost ~34 GB per process;
    this loop's peak is one slide plus the model. That matters here because
    this exact RAM pattern (raw H&E patches held for every slide, in eight
    concurrent processes) is what previously pushed the shared box to its
    limit.

    Consecutive steps on one slide are correlated, so ``steps_per_slide`` is
    kept small and the slide list is re-shuffled each round: the corpus is
    revisited ``rounds`` times in a different order rather than trained to
    convergence slide by slide. Returns one record per slide visit.

    ``validation_ids`` names slides that are NEVER trained on and are scored
    once at the end of every round. This is the number that means something:
    the per-slide metric recorded during training is measured on the very
    slides just optimised, so a module that memorised slide-specific structure
    would look identical to one that learned transferable spatial structure.
    The held-out column separates them. Its evaluation mask is drawn from a
    fixed seed so the same spots are scored every round and the trajectory is
    comparable across rounds rather than re-randomised each time.
    """
    if rounds < 1:
        raise ValueError("rounds must be positive")
    if steps_per_slide < 1:
        raise ValueError("steps_per_slide must be positive")
    if not sample_ids:
        raise ValueError("at least one sample_id is required")
    device = device or torch.device("cpu")
    refiner = refiner.to(device)
    optimizer = torch.optim.Adam(refiner.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(int(seed))
    order_rng = np.random.default_rng(int(seed))
    history: list[dict] = []
    for round_index in range(rounds):
        order = list(sample_ids)
        order_rng.shuffle(order)
        for position, sample_id in enumerate(order, start=1):
            expression_np, coords_np = load_slide(sample_id)
            expression = torch.as_tensor(np.asarray(expression_np), dtype=torch.float32, device=device)
            coords = torch.as_tensor(np.asarray(coords_np), dtype=torch.float32, device=device)
            losses = [
                _spatial_prior_step(
                    refiner, optimizer, expression, coords,
                    mask_fraction=mask_fraction, generator=generator,
                )
                for _ in range(steps_per_slide)
            ]
            evaluation_mask = mask_spots(expression.shape[0], mask_fraction, generator).to(device)
            pearson = masked_spot_pearson(
                refiner, expression, coords, evaluation_mask, n_top_genes=n_top_genes,
            )
            record = {
                "round": round_index + 1,
                "split": "train",
                "sample_id": sample_id,
                "n_spots": int(expression.shape[0]),
                "first_loss": losses[0],
                "last_loss": losses[-1],
                "masked_spot_pearson": pearson["all_genes"],
                "masked_spot_pearson_top_genes": pearson["top_variance_genes"],
            }
            history.append(record)
            print(
                f"spatial-prior round {round_index + 1}/{rounds} "
                f"slide {position}/{len(order)} {sample_id} "
                f"loss {losses[0]:.6f}->{losses[-1]:.6f} "
                f"pcc_all={pearson['all_genes']:+.4f} "
                f"pcc_top{n_top_genes}={pearson['top_variance_genes']:+.4f}",
                flush=True,
            )
            del expression, coords, expression_np, coords_np
        for sample_id in (validation_ids or []):
            expression_np, coords_np = load_slide(sample_id)
            expression = torch.as_tensor(np.asarray(expression_np), dtype=torch.float32, device=device)
            coords = torch.as_tensor(np.asarray(coords_np), dtype=torch.float32, device=device)
            # Fixed seed per slide: the same spots are hidden every round, so
            # the round-to-round trajectory reflects the model changing rather
            # than the mask changing.
            held_out_mask = mask_spots(
                expression.shape[0], mask_fraction,
                torch.Generator().manual_seed(int(seed) + 1_000_003),
            ).to(device)
            pearson = masked_spot_pearson(
                refiner, expression, coords, held_out_mask, n_top_genes=n_top_genes,
            )
            history.append({
                "round": round_index + 1,
                "split": "validation",
                "sample_id": sample_id,
                "n_spots": int(expression.shape[0]),
                "masked_spot_pearson": pearson["all_genes"],
                "masked_spot_pearson_top_genes": pearson["top_variance_genes"],
            })
            print(
                f"spatial-prior round {round_index + 1}/{rounds} HELD-OUT {sample_id} "
                f"pcc_all={pearson['all_genes']:+.4f} "
                f"pcc_top{n_top_genes}={pearson['top_variance_genes']:+.4f}",
                flush=True,
            )
            del expression, coords, expression_np, coords_np
    return history
