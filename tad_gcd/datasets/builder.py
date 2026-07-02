#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset builder for TAD-GCD.

This file provides unified dataset / dataloader construction APIs.

Expected project structure:

    datasets/
        __init__.py
        builder.py
        thumos14_tad_gcd.py

Typical usage in training code:

    from datasets.builder import build_dataset, build_dataloader

    cfg = {
        "dataset_name": "thumos14_tad_gcd",
        "index_json": "data/thumos14_tad_gcd_code2114_train50_split0.json",
        "feature_dir": "/data/ysc/action/thumos14/features/i3d",
        "feature_ext": ".npy",
        "load_features": True,
        "feature_norm": "l2",
        "batch_size": 2,
        "num_workers": 4,
    }

    train_set = build_dataset(cfg, mode="train_labeled")
    train_loader = build_dataloader(cfg, mode="train_labeled")

Supported dataset_name:

    thumos14_tad_gcd
    thumos14
    tad_gcd

Supported modes:

    train_labeled
    train_unlabeled
    test
    test_old
    test_new

This builder is intentionally lightweight. It should not contain model logic,
loss logic, proposal generation, or evaluation logic.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset


# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------


def cfg_get(cfg: Union[Dict[str, Any], Any], key: str, default: Any = None) -> Any:
    """
    Read config from dict or object-style config.

    Examples:
        cfg_get({"a": 1}, "a") -> 1
        cfg_get(namespace, "a") -> namespace.a
    """
    if cfg is None:
        return default

    if isinstance(cfg, dict):
        return cfg.get(key, default)

    return getattr(cfg, key, default)


def cfg_to_dict(cfg: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    """
    Convert config object / argparse Namespace / dict to plain dict.
    """
    if cfg is None:
        return {}

    if isinstance(cfg, dict):
        return dict(cfg)

    if hasattr(cfg, "__dict__"):
        return dict(vars(cfg))

    # Fallback: collect public attributes.
    out = {}
    for k in dir(cfg):
        if k.startswith("_"):
            continue
        try:
            v = getattr(cfg, k)
        except Exception:
            continue
        if callable(v):
            continue
        out[k] = v
    return out


def normalize_dataset_name(name: str) -> str:
    """
    Normalize dataset name aliases.
    """
    name = str(name).lower().strip()

    aliases = {
        "thumos14_tad_gcd": "thumos14_tad_gcd",
        "thumos14": "thumos14_tad_gcd",
        "tad_gcd": "thumos14_tad_gcd",
        "thumos": "thumos14_tad_gcd",
    }

    if name not in aliases:
        raise ValueError(
            f"Unsupported dataset_name={name}. "
            f"Supported aliases: {sorted(aliases.keys())}"
        )

    return aliases[name]


def infer_shuffle(mode: str, shuffle: Optional[bool] = None) -> bool:
    """
    Infer default shuffle behavior.

    train_labeled:
        True

    train_unlabeled:
        True

    test/test_old/test_new:
        False
    """
    if shuffle is not None:
        return bool(shuffle)

    return mode in {"train_labeled", "train_unlabeled"}


def infer_drop_last(mode: str, drop_last: Optional[bool] = None) -> bool:
    """
    Infer default drop_last behavior.

    For detection/evaluation, default is False.
    """
    if drop_last is not None:
        return bool(drop_last)

    return False


# -----------------------------------------------------------------------------
# Dataset construction
# -----------------------------------------------------------------------------


def build_dataset(
    cfg: Union[Dict[str, Any], Any],
    mode: str,
) -> Dataset:
    """
    Build dataset by config.

    Required config:
        dataset_name:
            default "thumos14_tad_gcd"

        index_json:
            path to split json.

    Optional config for THUMOS14TADGCDDataset:
        feature_dir
        feature_ext
        load_features
        feature_norm
        return_video_path
        return_raw_annotations
        strict
        allow_missing_video_path
        allow_missing_feature
        copy_annotations

    Example:
        dataset = build_dataset(cfg, mode="train_labeled")
    """
    dataset_name = normalize_dataset_name(
        cfg_get(cfg, "dataset_name", "thumos14_tad_gcd")
    )

    if dataset_name == "thumos14_tad_gcd":
        # Local import keeps this builder independent and avoids circular imports.
        try:
            from .thumos14_tad_gcd import THUMOS14TADGCDDataset
        except ImportError:
            # Allows running builder.py directly from project root.
            from thumos14_tad_gcd import THUMOS14TADGCDDataset

        index_json = cfg_get(cfg, "index_json", None)
        if index_json is None:
            raise KeyError(
                "Missing required config key: index_json. "
                "Example: data/thumos14_tad_gcd_code2114_train50_split0.json"
            )

        dataset = THUMOS14TADGCDDataset(
            index_json=index_json,
            mode=mode,
            feature_dir=cfg_get(cfg, "feature_dir", None),
            feature_ext=cfg_get(cfg, "feature_ext", ".npy"),
            load_features=cfg_get(cfg, "load_features", False),
            feature_norm=cfg_get(cfg, "feature_norm", "none"),
            return_video_path=cfg_get(cfg, "return_video_path", True),
            return_raw_annotations=cfg_get(cfg, "return_raw_annotations", True),
            strict=cfg_get(cfg, "strict", True),
            allow_missing_video_path=cfg_get(cfg, "allow_missing_video_path", True),
            allow_missing_feature=cfg_get(cfg, "allow_missing_feature", True),
            copy_annotations=cfg_get(cfg, "copy_annotations", True),
        )

        return dataset

    raise RuntimeError(f"Unexpected normalized dataset_name={dataset_name}")


def build_dataloader(
    cfg: Union[Dict[str, Any], Any],
    mode: str,
    dataset: Optional[Dataset] = None,
) -> DataLoader:
    """
    Build dataloader by config.

    Config keys:
        batch_size:
            default 2

        train_batch_size:
            optional override for train_labeled / train_unlabeled

        test_batch_size:
            optional override for test / test_old / test_new

        num_workers:
            default 0

        shuffle:
            optional global shuffle override

        pin_memory:
            default False

        drop_last:
            optional global drop_last override

        persistent_workers:
            default False

        prefetch_factor:
            optional, only valid when num_workers > 0

    Example:
        loader = build_dataloader(cfg, mode="train_labeled")
    """
    dataset_name = normalize_dataset_name(
        cfg_get(cfg, "dataset_name", "thumos14_tad_gcd")
    )

    if dataset is None:
        dataset = build_dataset(cfg, mode=mode)

    if dataset_name == "thumos14_tad_gcd":
        try:
            from .thumos14_tad_gcd import tad_gcd_collate_fn
        except ImportError:
            from thumos14_tad_gcd import tad_gcd_collate_fn

        collate_fn = tad_gcd_collate_fn
    else:
        collate_fn = None

    default_batch_size = cfg_get(cfg, "batch_size", 2)

    if mode in {"train_labeled", "train_unlabeled"}:
        batch_size = cfg_get(cfg, "train_batch_size", default_batch_size)
    else:
        batch_size = cfg_get(cfg, "test_batch_size", default_batch_size)

    num_workers = int(cfg_get(cfg, "num_workers", 0))
    pin_memory = bool(cfg_get(cfg, "pin_memory", False))
    shuffle = infer_shuffle(mode, cfg_get(cfg, "shuffle", None))
    drop_last = infer_drop_last(mode, cfg_get(cfg, "drop_last", None))

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }

    persistent_workers = bool(cfg_get(cfg, "persistent_workers", False))
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers

        prefetch_factor = cfg_get(cfg, "prefetch_factor", None)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    return DataLoader(**loader_kwargs)


def build_dataset_and_loader(
    cfg: Union[Dict[str, Any], Any],
    mode: str,
) -> Tuple[Dataset, DataLoader]:
    """
    Build dataset and dataloader together.
    """
    dataset = build_dataset(cfg, mode=mode)
    loader = build_dataloader(cfg, mode=mode, dataset=dataset)
    return dataset, loader


# -----------------------------------------------------------------------------
# Multi-loader helpers
# -----------------------------------------------------------------------------


def build_train_loaders(
    cfg: Union[Dict[str, Any], Any],
) -> Dict[str, DataLoader]:
    """
    Build training loaders.

    Usually TAD-GCD training uses:
        train_labeled:
            supervised old-class detection.

        train_unlabeled:
            proposal mining / unknown discovery / unsupervised objective.
    """
    loaders = {
        "train_labeled": build_dataloader(cfg, mode="train_labeled"),
        "train_unlabeled": build_dataloader(cfg, mode="train_unlabeled"),
    }
    return loaders


def build_eval_loaders(
    cfg: Union[Dict[str, Any], Any],
    include_test_old_new: bool = True,
) -> Dict[str, DataLoader]:
    """
    Build evaluation loaders.

    test:
        old + new.

    test_old:
        old only.

    test_new:
        new only.
    """
    loaders = {
        "test": build_dataloader(cfg, mode="test"),
    }

    if include_test_old_new:
        # Some old split json may not contain test_old / test_new.
        index_json = cfg_get(cfg, "index_json", None)
        if index_json is not None and Path(index_json).exists():
            with open(index_json, "r", encoding="utf-8") as f:
                raw = json.load(f)

            for mode in ["test_old", "test_new"]:
                if mode in raw and isinstance(raw[mode], list) and len(raw[mode]) > 0:
                    loaders[mode] = build_dataloader(cfg, mode=mode)

    return loaders


def build_all_loaders(
    cfg: Union[Dict[str, Any], Any],
    include_test_old_new: bool = True,
) -> Dict[str, DataLoader]:
    """
    Build train_labeled, train_unlabeled, test, optionally test_old/test_new.
    """
    loaders = {}
    loaders.update(build_train_loaders(cfg))
    loaders.update(build_eval_loaders(cfg, include_test_old_new=include_test_old_new))
    return loaders


# -----------------------------------------------------------------------------
# Sanity check / debug utilities
# -----------------------------------------------------------------------------


def summarize_dataset(dataset: Dataset) -> Dict[str, Any]:
    """
    Return dataset summary if available.
    """
    if hasattr(dataset, "summary") and callable(getattr(dataset, "summary")):
        return dataset.summary()

    return {
        "dataset_class": dataset.__class__.__name__,
        "num_samples": len(dataset),
    }


def print_dataset_summary(dataset: Dataset, title: str = "Dataset") -> None:
    """
    Print summary for one dataset.
    """
    print("=" * 80)
    print(title)
    print("=" * 80)

    summary = summarize_dataset(dataset)
    for k, v in summary.items():
        print(f"{k}: {v}")

    if hasattr(dataset, "get_old_classes"):
        print("-" * 80)
        print("old_classes:", getattr(dataset, "get_old_classes")())

    if hasattr(dataset, "get_new_classes"):
        print("new_classes:", getattr(dataset, "get_new_classes")())

    print("=" * 80)


def inspect_loader(loader: DataLoader, max_batches: int = 1) -> None:
    """
    Print several batches from dataloader.
    """
    print("-" * 80)
    print("Inspect DataLoader")
    print("-" * 80)

    for batch_idx, batch in enumerate(loader):
        print(f"[Batch {batch_idx}]")
        if isinstance(batch, dict):
            print("keys:", list(batch.keys()))

            if "video_id" in batch:
                print("video_id:", batch["video_id"])

            if "features" in batch:
                features = batch["features"]
                if features is None:
                    print("features: None")
                else:
                    print("features shape:", tuple(features.shape))

            if "feature_mask" in batch:
                mask = batch["feature_mask"]
                if mask is None:
                    print("feature_mask: None")
                else:
                    print("feature_mask shape:", tuple(mask.shape))
                    print("feature_mask valid counts:", mask.sum(dim=1).tolist())

            if "segments" in batch:
                print("segments shapes:", [tuple(x.shape) for x in batch["segments"]])

            if "labels" in batch:
                print("labels shapes:", [tuple(x.shape) for x in batch["labels"]])

            if "annotations" in batch:
                print("annotations_per_video:", [len(x) for x in batch["annotations"]])

        else:
            print("batch type:", type(batch))

        if batch_idx + 1 >= max_batches:
            break

    print("-" * 80)


def check_mode(
    cfg: Union[Dict[str, Any], Any],
    mode: str,
    inspect_batches: int = 1,
) -> None:
    """
    Build one dataset/loader and print summary.
    """
    dataset, loader = build_dataset_and_loader(cfg, mode=mode)
    print_dataset_summary(dataset, title=f"Dataset mode: {mode}")
    inspect_loader(loader, max_batches=inspect_batches)


def check_all_modes(
    cfg: Union[Dict[str, Any], Any],
    inspect_batches: int = 1,
) -> None:
    """
    Check all available modes.
    """
    index_json = cfg_get(cfg, "index_json", None)
    modes = ["train_labeled", "train_unlabeled", "test"]

    if index_json is not None and Path(index_json).exists():
        with open(index_json, "r", encoding="utf-8") as f:
            raw = json.load(f)

        for mode in ["test_old", "test_new"]:
            if mode in raw and isinstance(raw[mode], list) and len(raw[mode]) > 0:
                modes.append(mode)

    for mode in modes:
        print("\n\n")
        print("#" * 80)
        print(f"Checking mode: {mode}")
        print("#" * 80)
        check_mode(cfg, mode=mode, inspect_batches=inspect_batches)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("TAD-GCD dataset builder")

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="thumos14_tad_gcd",
        help="Dataset name. Supported: thumos14_tad_gcd.",
    )

    parser.add_argument(
        "--index_json",
        type=str,
        default="data/thumos14_tad_gcd_code2114_train50_split0.json",
        help="Path to split json.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="train_labeled",
        choices=["train_labeled", "train_unlabeled", "test", "test_old", "test_new"],
    )

    parser.add_argument("--feature_dir", type=str, default=None)
    parser.add_argument("--feature_ext", type=str, default=".npy")
    parser.add_argument("--load_features", action="store_true")
    parser.add_argument("--feature_norm", type=str, default="none", choices=["none", "l2", "zscore"])

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--test_batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=None)

    parser.add_argument("--shuffle", type=str, default=None, choices=["true", "false", "none"])
    parser.add_argument("--drop_last", type=str, default=None, choices=["true", "false", "none"])

    parser.add_argument("--no_strict", action="store_true")
    parser.add_argument("--require_video_path", action="store_true")
    parser.add_argument("--require_feature", action="store_true")
    parser.add_argument("--no_video_path", action="store_true")
    parser.add_argument("--no_raw_annotations", action="store_true")

    parser.add_argument("--check_all_modes", action="store_true")
    parser.add_argument("--inspect_batches", type=int, default=1)

    return parser.parse_args()


def str_to_optional_bool(x: Optional[str]) -> Optional[bool]:
    if x is None or x == "none":
        return None
    return x == "true"


def args_to_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = vars(args).copy()

    cfg["strict"] = not args.no_strict
    cfg["allow_missing_video_path"] = not args.require_video_path
    cfg["allow_missing_feature"] = not args.require_feature
    cfg["return_video_path"] = not args.no_video_path
    cfg["return_raw_annotations"] = not args.no_raw_annotations

    cfg["shuffle"] = str_to_optional_bool(args.shuffle)
    cfg["drop_last"] = str_to_optional_bool(args.drop_last)

    # Remove CLI-only keys.
    cfg.pop("mode", None)
    cfg.pop("check_all_modes", None)
    cfg.pop("inspect_batches", None)

    cfg.pop("no_strict", None)
    cfg.pop("require_video_path", None)
    cfg.pop("require_feature", None)
    cfg.pop("no_video_path", None)
    cfg.pop("no_raw_annotations", None)

    return cfg


def main() -> None:
    args = parse_args()
    cfg = args_to_cfg(args)

    if args.check_all_modes:
        check_all_modes(cfg, inspect_batches=args.inspect_batches)
    else:
        check_mode(cfg, mode=args.mode, inspect_batches=args.inspect_batches)


if __name__ == "__main__":
    main()
