"""Persistent train/validation/test masking banks keyed by observation names."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Iterable


import numpy as np

from src.data import masking


def _cfg_get(obj, key, default=None):
    if hasattr(obj, "get"):
        return obj.get(key, default)
    return getattr(obj, key, default)


def make_split(coords3d: np.ndarray, slice_ids: np.ndarray, masking_cfg, seed: int):
    """Standalone equivalent of the training split dispatcher."""
    strategy = _cfg_get(masking_cfg, "strategy")
    params = _cfg_get(masking_cfg, "params", {})
    params = dict(params)
    if strategy == "hold_out_slice":
        held_out = np.random.default_rng(seed).choice(np.unique(slice_ids))
        return masking.hold_out_slice(coords3d[:, 2], held_out, slice_ids)
    if strategy == "random_dropout_patches":
        return masking.random_dropout_patches(coords3d[:, :2], slice_ids, seed=seed, **params)
    if strategy == "sparse_spot_dropout":
        return masking.sparse_spot_dropout(coords3d[:, :2], slice_ids, seed=seed, **params)
    if strategy == "mixed_dropout":
        return masking.mixed_dropout(coords3d[:, :2], slice_ids, seed=seed, **params)
    raise ValueError(f"unknown masking strategy {strategy!r}")


def cap_context_mask(context_mask: np.ndarray, max_context_points, seed: int) -> np.ndarray:
    if max_context_points is None:
        return context_mask
    idx = np.flatnonzero(context_mask)
    if len(idx) <= int(max_context_points):
        return context_mask
    keep = np.random.default_rng(seed + 3).choice(idx, size=int(max_context_points), replace=False)
    out = np.zeros_like(context_mask, dtype=bool)
    out[keep] = True
    return out


def dataset_fingerprint(obs_names: Iterable[str]) -> str:
    names = [str(x) for x in obs_names]
    return sha256("\n".join(names).encode("utf-8")).hexdigest()


def _jsonable(value):
    """Convert OmegaConf/numpy containers into a stable JSON representation."""
    if hasattr(value, "items"):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    # OmegaConf ListConfig and similar sequence wrappers are iterable but
    # are not subclasses of list/tuple.
    if (not isinstance(value, (str, bytes)) and hasattr(value, "__iter__")
            and not isinstance(value, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def masking_fingerprint(masking_cfg, split_counts=None, split_seeds=None) -> str:
    payload = {
        "masking": _jsonable(masking_cfg),
        "split_counts": _jsonable(split_counts or {"validation": 4, "test": 8}),
        "split_seeds": _jsonable(split_seeds or {"validation": 700_000, "test": 900_000}),
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def spatial_fingerprint(coords3d: np.ndarray, slice_ids: np.ndarray) -> str:
    coords = np.ascontiguousarray(np.asarray(coords3d, dtype=np.float64))
    slices = np.asarray(slice_ids, dtype=str)
    digest = sha256()
    digest.update(str(coords.shape).encode())
    digest.update(coords.tobytes())
    digest.update("\n".join(slices.tolist()).encode())
    return digest.hexdigest()


def build_mask_bank(
    coords3d: np.ndarray,
    slice_ids: np.ndarray,
    obs_names: Iterable[str],
    masking_cfg,
    split_counts: dict[str, int] | None = None,
    split_seeds: dict[str, int] | None = None,
) -> dict:
    """Create reproducible masks and store them by barcode/name, not row index."""
    split_counts = split_counts or {"validation": 4, "test": 8}
    split_seeds = split_seeds or {"validation": 700_000, "test": 900_000}
    names = np.asarray([str(x) for x in obs_names])
    records = []
    max_context = _cfg_get(masking_cfg, "max_context_points", None)
    for split, count in split_counts.items():
        base_seed = int(split_seeds[split])
        for i in range(int(count)):
            seed = base_seed + i
            context, query = make_split(coords3d, slice_ids, masking_cfg, seed)
            context = cap_context_mask(context, max_context, seed)
            if not query.any() or not context.any():
                raise ValueError(f"mask seed {seed} produced an empty context or query")
            records.append({
                "split": split,
                "index": i,
                "seed": seed,
                "context_obs_names": names[context].tolist(),
                "query_obs_names": names[query].tolist(),
            })
    return {
        "version": 2,
        "dataset_fingerprint": dataset_fingerprint(names),
        "spatial_fingerprint": spatial_fingerprint(coords3d, slice_ids),
        "masking_fingerprint": masking_fingerprint(masking_cfg, split_counts, split_seeds),
        "masking": _jsonable(masking_cfg),
        "split_counts": _jsonable(split_counts),
        "split_seeds": _jsonable(split_seeds),
        "n_obs": int(len(names)),
        "records": records,
    }


def save_mask_bank(bank: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Multiple single-GPU jobs may start together and all request the same
    # immutable mask bank. Use a process-specific temporary file followed by
    # atomic os.replace so concurrent writers can never expose partial JSON or
    # collide on one shared .tmp filename. All writers deterministically build
    # identical content for the same config/fingerprint.
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(bank, indent=2, sort_keys=True))
    os.replace(tmp, path)
    return path


def load_mask_bank(
    path: str | Path,
    obs_names: Iterable[str],
    *,
    coords3d: np.ndarray | None = None,
    slice_ids: np.ndarray | None = None,
    masking_cfg=None,
    split_counts: dict[str, int] | None = None,
    split_seeds: dict[str, int] | None = None,
) -> dict:
    path = Path(path)
    bank = json.loads(path.read_text())
    expected = dataset_fingerprint(obs_names)
    if bank.get("dataset_fingerprint") != expected:
        raise ValueError(
            f"mask bank {path} was built for a different observation set; "
            "regenerate it for this exact QC/alignment result"
        )
    supplied = [coords3d is not None, slice_ids is not None, masking_cfg is not None]
    if any(supplied) and not all(supplied):
        raise ValueError("coords3d, slice_ids and masking_cfg must be supplied together")
    if all(supplied):
        expected_spatial = spatial_fingerprint(coords3d, slice_ids)
        expected_masking = masking_fingerprint(masking_cfg, split_counts, split_seeds)
        if bank.get("version", 1) < 2:
            raise ValueError(
                f"mask bank {path} predates spatial/config fingerprints; remove it once "
                "and regenerate so changed masking settings cannot silently reuse stale masks"
            )
        if bank.get("spatial_fingerprint") != expected_spatial:
            raise ValueError(
                f"mask bank {path} was built for different coordinates or slice IDs; regenerate it"
            )
        if bank.get("masking_fingerprint") != expected_masking:
            raise ValueError(
                f"mask bank {path} was built with different masking parameters, split counts, "
                "or split seeds; use a distinct path or regenerate it"
            )
    return bank


def record_masks(record: dict, obs_names: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    names = np.asarray([str(x) for x in obs_names])
    context_names = set(record["context_obs_names"])
    query_names = set(record["query_obs_names"])
    context = np.asarray([x in context_names for x in names], dtype=bool)
    query = np.asarray([x in query_names for x in names], dtype=bool)
    if int(context.sum()) != len(context_names) or int(query.sum()) != len(query_names):
        raise ValueError("mask bank record contains observation names missing from current data")
    if np.any(context & query):
        raise ValueError("mask bank record has overlapping context/query rows")
    return context, query


def split_records(bank: dict, split: str) -> list[dict]:
    records = [r for r in bank["records"] if r["split"] == split]
    return sorted(records, key=lambda r: int(r["index"]))


def ensure_mask_bank(
    path: str | Path,
    coords3d: np.ndarray,
    slice_ids: np.ndarray,
    obs_names: Iterable[str],
    masking_cfg,
    split_counts: dict[str, int] | None = None,
    split_seeds: dict[str, int] | None = None,
) -> dict:
    path = Path(path)
    if path.exists():
        return load_mask_bank(
            path, obs_names, coords3d=coords3d, slice_ids=slice_ids,
            masking_cfg=masking_cfg, split_counts=split_counts, split_seeds=split_seeds,
        )
    bank = build_mask_bank(
        coords3d, slice_ids, obs_names, masking_cfg,
        split_counts=split_counts, split_seeds=split_seeds,
    )
    save_mask_bank(bank, path)
    # Re-read and validate after the atomic write. If another concurrent job
    # wrote a differently configured bank to the same path, fail loudly.
    return load_mask_bank(
        path, obs_names, coords3d=coords3d, slice_ids=slice_ids,
        masking_cfg=masking_cfg, split_counts=split_counts, split_seeds=split_seeds,
    )


def build_training_seed_bank(
    obs_names: Iterable[str], n_items: int, base_seed: int, masking_cfg=None
) -> dict:
    """Persist the exact training-mask seed schedule for one observation set.

    Storing every context/query barcode list for tens of thousands of training
    draws would create multi-gigabyte JSON. A deterministic seed schedule plus
    the immutable observation fingerprint is lossless: ``make_split`` and
    context capping are pure functions of the data order, masking config and
    seed. Validation/test masks remain stored explicitly by barcode.
    """
    n_items = int(n_items)
    base_seed = int(base_seed)
    if n_items < 1:
        raise ValueError("training seed bank requires at least one item")
    names = [str(x) for x in obs_names]
    return {
        "version": 1,
        "kind": "training_seed_schedule",
        "dataset_fingerprint": dataset_fingerprint(names),
        "n_obs": len(names),
        "n_items": n_items,
        "base_seed": base_seed,
        "masking_fingerprint": (
            sha256(json.dumps(_jsonable(masking_cfg), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if masking_cfg is not None else None
        ),
        "seeds": list(range(base_seed, base_seed + n_items)),
    }


def ensure_training_seed_bank(
    path: str | Path,
    obs_names: Iterable[str],
    n_items: int,
    base_seed: int,
    masking_cfg=None,
) -> tuple[dict, Path]:
    """Load or atomically create an immutable training seed schedule."""
    path = Path(path)
    expected = build_training_seed_bank(obs_names, n_items, base_seed, masking_cfg)
    if path.exists():
        bank = json.loads(path.read_text())
        keys = ["kind", "dataset_fingerprint", "n_obs", "n_items", "base_seed"]
        # Version-1 training banks contained the same lossless seed list but
        # no masking fingerprint. They can be upgraded safely after validating
        # all original fields and the exact schedule; explicit but different
        # fingerprints still fail below.
        legacy_without_masking = masking_cfg is not None and bank.get("masking_fingerprint") is None
        if masking_cfg is not None and not legacy_without_masking:
            keys.append("masking_fingerprint")
        for key in keys:
            if bank.get(key) != expected.get(key):
                raise ValueError(
                    f"training seed bank {path} does not match the current data/run "
                    f"for field {key!r}; use a new path or remove the stale bank"
                )
        seeds = [int(x) for x in bank.get("seeds", [])]
        if seeds != expected["seeds"]:
            raise ValueError(f"training seed bank {path} contains an altered seed schedule")
        if legacy_without_masking:
            save_mask_bank(expected, path)
            return expected, path
        bank["seeds"] = seeds
        return bank, path
    save_mask_bank(expected, path)
    return expected, path
