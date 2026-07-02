#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Datasets package for TAD-GCD.

This file exposes the main dataset and builder APIs.

Expected structure:

    datasets/
    ├── __init__.py
    ├── thumos14_tad_gcd.py
    ├── builder.py
    └── build_thumos14_tad_gcd_index_code2114.py

Recommended usage:

    from datasets import THUMOS14TADGCDDataset
    from datasets import build_dataset, build_dataloader

    cfg = {
        "dataset_name": "thumos14_tad_gcd",
        "index_json": "data/thumos14_tad_gcd_code2114_train50_split0.json",
        "feature_dir": "/data/ysc/action/thumos14/features/i3d",
        "feature_ext": ".npy",
        "load_features": True,
        "feature_norm": "l2",
    }

    train_set = build_dataset(cfg, mode="train_labeled")
    train_loader = build_dataloader(cfg, mode="train_labeled")

Available modes:

    train_labeled
    train_unlabeled
    test
    test_old
    test_new
"""

from .thumos14_tad_gcd import (
    THUMOS14TADGCDDataset,
    VALID_MODES,
    tad_gcd_collate_fn,
    build_tad_gcd_dataset,
    build_tad_gcd_dataloader,
    build_dataset as build_thumos14_tad_gcd_dataset,
    load_feature_file,
    normalize_feature,
)

from .builder import (
    build_dataset,
    build_dataloader,
    build_dataset_and_loader,
    build_train_loaders,
    build_eval_loaders,
    build_all_loaders,
    cfg_get,
    cfg_to_dict,
    normalize_dataset_name,
)


__all__ = [
    # Dataset class
    "THUMOS14TADGCDDataset",

    # Constants
    "VALID_MODES",

    # Collate
    "tad_gcd_collate_fn",

    # Low-level THUMOS14-TAD-GCD builders
    "build_tad_gcd_dataset",
    "build_tad_gcd_dataloader",
    "build_thumos14_tad_gcd_dataset",

    # Unified builders
    "build_dataset",
    "build_dataloader",
    "build_dataset_and_loader",
    "build_train_loaders",
    "build_eval_loaders",
    "build_all_loaders",

    # Config helpers
    "cfg_get",
    "cfg_to_dict",
    "normalize_dataset_name",

    # Feature helpers
    "load_feature_file",
    "normalize_feature",
]


def available_datasets():
    """
    Return supported dataset names.

    Example:
        from datasets import available_datasets
        print(available_datasets())
    """
    return [
        "thumos14_tad_gcd",
        "thumos14",
        "tad_gcd",
        "thumos",
    ]


def available_modes():
    """
    Return supported dataset modes.

    Example:
        from datasets import available_modes
        print(available_modes())
    """
    return sorted(list(VALID_MODES))
