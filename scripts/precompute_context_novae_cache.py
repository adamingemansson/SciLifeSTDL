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
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from src.data.mask_bank import ensure_mask_bank, record_masks, split_records
from src.training.train import (
    _cap_context_mask,
    _load_data,
    _mask_bank_for_config,
    _training_seed_bank_for_config,
    _validated_sample_groups,
    evaluation_query_exclusion_mask,
    load_multi_sample_data,
    make_context_query_split,
    make_training_context_query_split,
    prepare_novae_inputs,
)


def _belongs_to_shard(index: int, shard_index: int, num_shards: int) -> bool:
    return index % num_shards == shard_index


def _provider(sample):
    provider = sample[9] or sample[8]
    if provider is None:
        raise RuntimeError("config does not request context-only Novae features")
    return provider


def _cache_one(sample, masking_cfg, seed: int) -> None:
    coords3d, _expr, slice_ids = sample[:3]
    context_mask, query_mask = make_context_query_split(
        coords3d, slice_ids, masking_cfg, seed
    )
    context_mask = _cap_context_mask(
        context_mask,
        getattr(masking_cfg, "max_context_points", None),
        seed,
        coords3d=coords3d,
        query_mask=query_mask,
        selection=str(getattr(masking_cfg, "context_selection", "random")),
    )
    _provider(sample)(np.asarray(context_mask, dtype=bool))


def _main_multi_sample(cfg, include_test: bool, shard_index: int, num_shards: int) -> None:
    train_ids, validation_ids, test_ids = _validated_sample_groups(cfg)
    train_samples, train_adatas = load_multi_sample_data(cfg, sample_ids=train_ids)
    composite_names = [
        f"{sid}:{name}"
        for sid, adata in zip(train_ids, train_adatas)
        for name in adata.obs_names
    ]
    training_bank, training_path = _training_seed_bank_for_config(cfg, composite_names)
    unique_seeds = list(OrderedDict.fromkeys(int(x) for x in training_bank["seeds"]))
    selected = [
        (i, seed) for i, seed in enumerate(unique_seeds)
        if _belongs_to_shard(i, shard_index, num_shards)
    ]
    print(
        f"precomputing multi-sample shard {shard_index + 1}/{num_shards}: "
        f"{len(selected)}/{len(unique_seeds)} unique training masks from {training_path}"
    )
    for done, (_index, seed) in enumerate(selected, start=1):
        sample_index = int(np.random.default_rng(seed).integers(len(train_samples)))
        # Must exactly match MultiSampleMaskedContextQueryDataset.__getitem__:
        # seed selects the sample, seed+1 selects its missing region.
        _cache_one(train_samples[sample_index], cfg.masking, seed + 1)
        if done == 1 or done % 25 == 0 or done == len(selected):
            print(f"  training cache {done}/{len(selected)}")

    eval_ids = [*validation_ids, *(test_ids if include_test else [])]
    if not eval_ids:
        return
    reference_genes = train_adatas[0].var_names.tolist()
    eval_samples, eval_adatas = load_multi_sample_data(
        cfg, sample_ids=eval_ids, reference_gene_names=reference_genes
    )
    evaluation = cfg.get("evaluation", {})
    mask_dir = Path(evaluation.get("mask_bank_dir", "results/mask_banks"))
    counts = {
        "validation": int(evaluation.get("n_validation_masks", 4)),
        "test": int(evaluation.get("n_test_masks", 8)),
    }
    seeds = {
        "validation": int(evaluation.get("validation_seed", 700_000)),
        "test": int(evaluation.get("test_seed", 900_000)),
    }
    tasks = []
    for sid, sample, adata in zip(eval_ids, eval_samples, eval_adatas):
        coords3d, _expr, slice_ids = sample[:3]
        bank = ensure_mask_bank(
            mask_dir / f"{sid}.json",
            coords3d,
            slice_ids,
            adata.obs_names,
            cfg.masking,
            counts,
            seeds,
        )
        wanted_split = "validation" if sid in validation_ids else "test"
        for record in split_records(bank, wanted_split):
            tasks.append((sid, sample, adata, record))
    selected_tasks = [
        task for i, task in enumerate(tasks)
        if _belongs_to_shard(i, shard_index, num_shards)
    ]
    print(
        f"precomputing multi-sample shard {shard_index + 1}/{num_shards}: "
        f"{len(selected_tasks)}/{len(tasks)} held-out evaluation masks"
    )
    for done, (sid, sample, adata, record) in enumerate(selected_tasks, start=1):
        context_mask, _ = record_masks(record, adata.obs_names)
        _provider(sample)(context_mask)
        print(f"  evaluation cache {done}/{len(selected_tasks)} ({sid})")


def main(config_path: str, include_test: bool = True,
         shard_index: int = 0, num_shards: int = 1) -> None:
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    cfg = OmegaConf.load(config_path)
    if cfg.data.get("train_sample_ids") is not None:
        _main_multi_sample(cfg, include_test, shard_index, num_shards)
        print("context-only Novae cache complete")
        return
    model_params = OmegaConf.to_container(cfg.model.get("params", {}), resolve=True)
    adata, coords3d, _expr, slice_ids, _images = _load_data(cfg)
    inputs = prepare_novae_inputs(
        cfg, adata, model_params, coords3d, slice_ids,
        sample_id=str(cfg.data.get("sample_id", "sample")),
    )
    provider = inputs.get("context_novae_feature_provider") or inputs.get("context_gene_feature_provider")
    if provider is None:
        raise RuntimeError("config does not request context-only Novae features")

    eval_bank, eval_path = _mask_bank_for_config(cfg, adata, coords3d, slice_ids)
    excluded_training_mask = None
    if bool(cfg.training.get("exclude_evaluation_query_spots", False)):
        excluded_training_mask = evaluation_query_exclusion_mask(eval_bank, adata.obs_names)
        print(
            f"within-slide leakage guard: excluding "
            f"{int(excluded_training_mask.sum())}/{adata.n_obs} evaluation query spots "
            f"from every precomputed training graph"
        )

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
        context_mask, _query_mask = make_training_context_query_split(
            coords3d, slice_ids, cfg.masking, seed,
            excluded_training_mask=excluded_training_mask,
        )
        provider(np.asarray(context_mask, dtype=bool))
        if i == 1 or i % 25 == 0 or i == len(selected_seeds):
            print(f"  training cache {i}/{len(selected_seeds)}")

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
