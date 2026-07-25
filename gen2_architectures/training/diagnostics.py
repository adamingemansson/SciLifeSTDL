"""Cheap per-step training diagnostics — GPT's second-round code audit
suggested logging these five signals during training because "those five
plots can catch many silent failures long before validation metrics do":
latent norm, decoder norm, query token norm, attention entropy, gene
embedding norm.

Implemented via forward hooks on whichever well-known submodules a given
model actually has, so model forward() code stays untouched and the extra
cost is one .norm() call on a tensor that was already computed anyway
(negligible — no extra forward/backward passes). Coverage genuinely
differs by architecture:

- Architecture 1/2 (LocalNeighborhoodTransformer): gene_embedding_norm
  (gene_encoder's output), query_token_norm (the post-transformer,
  per-query hidden state — decoder's input), decoder_output_norm
  (predicted expression). No true bottleneck "latent", so latent_norm is
  not reported (architecturally honest — there isn't one).
- Architecture 3 Stage A (DenoisingTranscriptomeAutoencoder): latent_norm
  IS the encoder's output (that's the whole point of an autoencoder) and
  decoder_output_norm is the reconstruction. No spatial query token here.
- Architecture 3 Stage B: gene_embedding_norm is the context spots'
  encoded latent (via the reused Stage-A encoder), query_token_norm is the
  post-transformer hidden state read by latent_head, decoder_output_norm
  is the decoded predicted_expression. latent_norm (predicted_latent) is
  already directly available as forward()'s own return value in the
  training loop, so no hook is needed for it there.
- Architecture 4 (frozen STPath + scFoundation residual): almost
  everything is frozen inside STPathContextEncoder, so the ONLY thing
  worth watching is the one part that's actually trainable — the
  scFoundation residual (new_gene_encoder) and its injection
  (residual_proj). Reported as gene_embedding_norm / decoder_output_norm
  respectively so the same log line format applies everywhere; a
  perpetually-zero decoder_output_norm here would mean the residual isn't
  learning to move away from its zero-initialization (see
  stpath_encoder.py's own docstring on why it's zero-initialized).

attention_entropy is deliberately NOT collected by these hooks: PyTorch's
nn.TransformerEncoderLayer skips returning attention weights on its fast
path for performance, and forcing need_weights=True on every step would
cost real throughput on the actual training runs. Call
compute_attention_entropy() explicitly and only periodically (e.g.
alongside eval_every_n_steps, not every log step).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _resolve_attr(model: nn.Module, path: str) -> nn.Module | None:
    obj = model
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj if isinstance(obj, nn.Module) else None


def _first_present(model: nn.Module, paths: tuple[str, ...]) -> nn.Module | None:
    for path in paths:
        module = _resolve_attr(model, path)
        if module is not None:
            return module
    return None


def _row_norm(tensor: torch.Tensor) -> float:
    flat = tensor.detach().reshape(tensor.shape[0], -1).float()
    return flat.norm(dim=-1).mean().item()


def attach_diagnostic_hooks(model: nn.Module) -> dict:
    """Register forward hooks on whichever of the well-known submodules
    this model actually has. Returns a plain dict that is overwritten (not
    appended to) on every forward pass — read it via collect_diagnostics()
    immediately after each model(...) call, before the next one runs."""
    stats: dict[str, float] = {}

    def _post_hook(name: str):
        def _hook(module, inputs, output):
            t = output[0] if isinstance(output, (tuple, list)) else output
            if torch.is_tensor(t) and t.numel() > 0:
                stats[name] = _row_norm(t)
        return _hook

    def _pre_hook(name: str):
        def _hook(module, inputs):
            t = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
            if torch.is_tensor(t) and t.numel() > 0:
                stats[name] = _row_norm(t)
        return _hook

    # "gene_embedding_norm" (a per-context-spot input embedding feeding a
    # further transformer/spatial stage) vs "latent_norm" (a genuine
    # terminal bottleneck, nothing downstream but a decoder) name
    # DIFFERENT concepts, but for architectures without a `gene_encoder`
    # attribute the same "encoder" module can play either role depending
    # on whether there's a `transformer` stage consuming its output —
    # Stage B's reused Stage-A encoder feeds its transformer
    # (gene_embedding_norm); Stage A's own encoder feeds nothing but its
    # own decoder (latent_norm, the real thing GPT meant by that name).
    gene_embedding_source = _first_present(model, ("gene_encoder", "autoencoder.encoder", "stpath.new_gene_encoder"))
    if gene_embedding_source is not None:
        gene_embedding_source.register_forward_hook(_post_hook("gene_embedding_norm"))
    elif _resolve_attr(model, "encoder") is not None:
        _resolve_attr(model, "encoder").register_forward_hook(_post_hook("latent_norm"))

    decoder_source = _first_present(
        model, ("decoder", "autoencoder.decoder", "stpath.residual_proj")
    )
    if decoder_source is not None:
        decoder_source.register_forward_hook(_post_hook("decoder_output_norm"))

    query_readout = _first_present(model, ("latent_head",)) or (
        _resolve_attr(model, "decoder") if _resolve_attr(model, "transformer") is not None else None
    )
    if query_readout is not None:
        query_readout.register_forward_pre_hook(_pre_hook("query_token_norm"))

    transformer = _resolve_attr(model, "transformer")
    if transformer is not None:
        def _capture_transformer_input(module, inputs):
            stats["_transformer_input"] = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
        transformer.register_forward_pre_hook(_capture_transformer_input)

    return stats


def collect_diagnostics(model: nn.Module, stats: dict) -> dict:
    """Merge hook-captured stats with cheap direct-parameter reads into a
    flat dict of plain floats ready for logging. query_token_param_norm is
    the LEARNED initial query token itself (a raw nn.Parameter, not a
    module output — hooks can't see it), separate from query_token_norm
    (the per-step, per-query hidden state after attending over neighbors)."""
    out = {k: v for k, v in stats.items() if not k.startswith("_")}
    query_token = getattr(model, "query_token", None)
    if isinstance(query_token, nn.Parameter):
        out["query_token_param_norm"] = query_token.detach().float().norm().item()
    return out


def compute_attention_entropy(model: nn.Module, stats: dict) -> float | None:
    """Opt-in, higher-cost diagnostic — call only periodically (e.g.
    alongside validation), never every training step. Recomputes the local
    transformer's first-layer self-attention weights via a second call to
    that layer's self_attn with need_weights=True on the SAME tokens the
    real forward pass just used (captured by attach_diagnostic_hooks'
    transformer pre-hook) — this redundant call roughly doubles that one
    layer's attention compute for this step only, which is why it isn't
    run by default. Returns None if this model has no local transformer
    (e.g. Architecture 4) or no forward pass has run yet."""
    transformer = _resolve_attr(model, "transformer")
    tokens = stats.get("_transformer_input")
    if transformer is None or tokens is None:
        return None
    layer = transformer.layers[0]
    with torch.no_grad():
        normed = layer.norm1(tokens) if getattr(layer, "norm_first", False) else tokens
        _, attn_weights = layer.self_attn(
            normed, normed, normed, need_weights=True, average_attn_weights=True
        )
        probs = attn_weights.clamp_min(1e-12)
        entropy = -(probs * probs.log()).sum(dim=-1).mean()
    return entropy.item()


def format_diagnostics(diagnostics: dict) -> str:
    if not diagnostics:
        return ""
    return " ".join(f"{k}={v:.3f}" for k, v in sorted(diagnostics.items()))
