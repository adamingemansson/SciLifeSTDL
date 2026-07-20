#!/usr/bin/env python3
"""Precompute clean context-only Novae features for a finite mask schedule.

This is intended for configs using ``data.novae_mode: context_only`` together
with ``training.unique_mask_count``. It populates the same cache consumed by
training, so several matched Novae jobs can start without racing to recompute
the same graph embeddings.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict

import numpy as np
from omegaconf import OmegaConf

from src.data.mask_bank import record_masks, split_records
from src.training.train import (
    _cap_context_mask,
    _load_data,
    _mask_bank_for_config,
    _training_seed_bank_for_config,
    make_context_query_split,
    prepare_novae_inputs,
)


def _belongs_to_shard(index: int, shard_index: int, num_shards: int) -> bool:
    return index % num_shards == shard_index


def main(config_path: str, include_test: bool = True,
         shard_index: int = 0, num_shards: int = 1) -> None:
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    cfg = OmegaConf.load(config_path)
    model_params = OmegaConf.to_container(cfg.model.get("params", {}), resolve=True)
    adata, coords3d, _expr, slice_ids, _images = _load_data(cfg)
    inputs = prepare_novae_inputs(
        cfg, adata, model_params, coords3d, slice_ids,
        sample_id=str(cfg.data.get("sample_id", "sample")),
    )
    provider = inputs.get("context_novae_feature_provider") or inputs.get("context_gene_feature_provider")
    if provider is None:
        raise RuntimeError("config does not request context-only Novae features")

    training_bank, training_path = _training_seed_bank_for_config(cfg, adata.obs_names)
    unique_seeds = list(OrderedDict.fromkeys(int(x) for x in training_bank["seeds"]))
    selected_seeds = [
        seed for i, seed in enumerate(unique_seeds)
        if _belongs_to_shard(i, shard_index, num_shards)
    ]
    print(
        f"precomputing shard {shard_index + 1}/{num_shards}: "
        f"{len(selected_seeds)}/{len(unique_seeds)} unique training masks "
        f"from {training_path} ({len(training_bank['seeds'])} total steps)"
    )
    for i, seed in enumerate(selected_seeds, start=1):
        context_mask, _query_mask = make_context_query_split(coords3d, slice_ids, cfg.masking, seed)
        context_mask = _cap_context_mask(
            context_mask, getattr(cfg.masking, "max_context_points", None), seed
        )
        provider(np.asarray(context_mask, dtype=bool))
        if i == 1 or i % 25 == 0 or i == len(selected_seeds):
            print(f"  training cache {i}/{len(selected_seeds)}")

    eval_bank, eval_path = _mask_bank_for_config(cfg, adata, coords3d, slice_ids)
    splits = ["validation", "test"] if include_test else ["validation"]
    eval_tasks = [
        (split, record)
        for split in splits
        for record in split_records(eval_bank, split)
    ]
    selected_eval_tasks = [
        task for i, task in enumerate(eval_tasks)
        if _belongs_to_shard(i, shard_index, num_shards)
    ]
    total = len(selected_eval_tasks)
    done = 0
    print(
        f"precomputing shard {shard_index + 1}/{num_shards}: "
        f"{total}/{len(eval_tasks)} evaluation masks from {eval_path}"
    )
    for split, record in selected_eval_tasks:
        context_mask, _query_mask = record_masks(record, adata.obs_names)
        provider(context_mask)
        done += 1
        print(f"  evaluation cache {done}/{total} ({split})")

    print("context-only Novae cache complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    main(
        args.config,
        include_test=not args.validation_only,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
