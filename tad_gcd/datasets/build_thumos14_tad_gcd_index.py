#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build THUMOS14-TAD-GCD split index aligned with Code-2114 open-class splits.

Reference code inspected from Code-2114.zip:

    configs/thumos14_i3d.yaml
    libs/datasets/thumos14.py
    libs/modeling/text.py
    splits/train_50_test_50/THUMOS14/train/split_0.list
    splits/train_50_test_50/THUMOS14/test/split_0.list
    splits/train_75_test_25/THUMOS14/train/split_0.list
    splits/train_75_test_25/THUMOS14/test/split_0.list

Key finding:
    Code-2114 does open-class recognition by class-list split files.

    For THUMOS14:
        splits/train_50_test_50/THUMOS14/train/split_k.list
            = known / seen / old classes

        splits/train_50_test_50/THUMOS14/test/split_k.list
            = unknown / unseen / new classes

For TAD-GCD:
    old_classes = Code-2114 train split classes
    new_classes = Code-2114 test split classes

This script converts the original THUMOS14 annotation json into a TAD-GCD json:

    train_labeled:
        validation videos with old-class annotations only.

    train_unlabeled:
        validation videos with annotations=[].
        This is the mixed old/new unlabeled pool, but no annotations are exposed.

    test:
        test videos with old + new annotations for evaluation.

    test_old:
        test videos with old annotations only.

    test_new:
        test videos with new annotations only.

Default local THUMOS14 layout:

    /data/ysc/action/thumos14/
        video/
        thumos/
            annotations/
                thumos14.json

Default annotation json:
    /data/ysc/action/thumos14/thumos/annotations/thumos14.json

Default video directory:
    /data/ysc/action/thumos14/video

Example usage:

    python datasets/build_thumos14_tad_gcd_index_code2114.py \
        --annotation_json /data/ysc/action/thumos14/thumos/annotations/thumos14.json \
        --video_dir /data/ysc/action/thumos14/video \
        --split_root ./splits \
        --data_split 50 \
        --split_num 0 \
        --output data/thumos14_tad_gcd_split_code2114_50_split0.json

If you extracted Code-2114 somewhere else:

    python datasets/build_thumos14_tad_gcd_index_code2114.py \
        --split_root /path/to/Code-2114/splits \
        --data_split 50 \
        --split_num 0

Important:
    - Do not feed new-class annotations into training.
    - Do not use new class names/prompts during training.
    - new_label_to_eval_id is evaluation-only.
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_DATA_ROOT = "/data/ysc/action/thumos14"
DEFAULT_VIDEO_DIR = "/data/ysc/action/thumos14/video"
DEFAULT_ANNOTATION_JSON = "/data/ysc/action/thumos14/thumos/annotations/thumos14.json"

THUMOS14_CLASSES = [
    "BaseballPitch",
    "BasketballDunk",
    "Billiards",
    "CleanAndJerk",
    "CliffDiving",
    "CricketBowling",
    "CricketShot",
    "Diving",
    "FrisbeeCatch",
    "GolfSwing",
    "HammerThrow",
    "HighJump",
    "JavelinThrow",
    "LongJump",
    "PoleVault",
    "Shotput",
    "SoccerPenalty",
    "TennisSwing",
    "ThrowDiscus",
    "VolleyballSpiking",
]


def normalize_class_name(name: Any) -> str:
    if name is None:
        return ""

    s = str(name).strip()
    s = Path(s).stem

    for prefix in ["Annotation_", "annotation_", "annotations_", "Annotations_"]:
        if s.startswith(prefix):
            s = s[len(prefix):]

    for suffix in [
        "_test",
        "_validation",
        "_val",
        "_train",
        "-test",
        "-validation",
        "-val",
        "-train",
    ]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]

    compact = re.sub(r"[^a-zA-Z0-9]", "", s).lower()
    for cls in THUMOS14_CLASSES:
        cls_compact = re.sub(r"[^a-zA-Z0-9]", "", cls).lower()
        if compact == cls_compact:
            return cls

    return s


def normalize_subset_name(subset: Any) -> str:
    if subset is None:
        return "unknown"

    s = str(subset).strip().lower()

    if s in {"validation", "val", "valid", "train", "training"}:
        return "validation"
    if s in {"test", "testing"}:
        return "test"

    return s


def infer_subset_from_video_id(video_id: str) -> str:
    low = str(video_id).lower()
    if "validation" in low or low.startswith("video_validation") or "_val_" in low:
        return "validation"
    if "test" in low or low.startswith("video_test"):
        return "test"
    return "unknown"


def parse_label(ann: Dict[str, Any]) -> str:
    for key in ["label", "class", "class_name", "action", "action_name"]:
        if key in ann:
            return normalize_class_name(ann[key])
    return ""


def parse_segment(ann: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    if "segment" in ann:
        seg = ann["segment"]
        if isinstance(seg, (list, tuple)) and len(seg) >= 2:
            return float(seg[0]), float(seg[1])

    for ks, ke in [
        ("start", "end"),
        ("t_start", "t_end"),
        ("start_time", "end_time"),
        ("xmin", "xmax"),
    ]:
        if ks in ann and ke in ann:
            return float(ann[ks]), float(ann[ke])

    return None


def load_code2114_class_list(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Class split list does not exist: {p}")

    classes = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        cls = normalize_class_name(line)
        if cls not in THUMOS14_CLASSES:
            raise ValueError(f"Invalid THUMOS14 class in {p}: {line}")
        classes.append(cls)

    classes = list(dict.fromkeys(classes))
    if not classes:
        raise ValueError(f"Empty class split list: {p}")

    return classes


def get_code2114_split_paths(
    split_root: str,
    data_split: int,
    split_num: int,
    dataset_name: str = "THUMOS14",
) -> Tuple[Path, Path]:
    split_root = Path(split_root)

    train_file = (
        split_root
        / f"train_{data_split}_test_{100 - data_split}"
        / dataset_name
        / "train"
        / f"split_{split_num}.list"
    )

    test_file = (
        split_root
        / f"train_{data_split}_test_{100 - data_split}"
        / dataset_name
        / "test"
        / f"split_{split_num}.list"
    )

    return train_file, test_file


def build_class_split_from_code2114(
    split_root: str,
    data_split: int,
    split_num: int,
) -> Tuple[List[str], List[str], Dict[str, str]]:
    train_file, test_file = get_code2114_split_paths(
        split_root=split_root,
        data_split=data_split,
        split_num=split_num,
        dataset_name="THUMOS14",
    )

    old_classes = load_code2114_class_list(str(train_file))
    new_classes = load_code2114_class_list(str(test_file))

    overlap = sorted(set(old_classes).intersection(new_classes))
    if overlap:
        raise RuntimeError(f"Code-2114 split has old/new overlap: {overlap}")

    missing = sorted(set(THUMOS14_CLASSES) - set(old_classes) - set(new_classes))
    if missing:
        raise RuntimeError(f"Code-2114 split does not cover all THUMOS14 classes: {missing}")

    meta = {
        "old_classes_file": str(train_file),
        "new_classes_file": str(test_file),
    }

    return old_classes, new_classes, meta


def load_thumos14_json(annotation_json: str) -> Dict[str, Dict[str, Any]]:
    path = Path(annotation_json)
    if not path.exists():
        raise FileNotFoundError(f"Annotation json does not exist: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    database = raw.get("database", raw) if isinstance(raw, dict) else raw
    if not isinstance(database, dict):
        raise ValueError(f"Unsupported json format: {path}")

    output: Dict[str, Dict[str, Any]] = {}

    for raw_video_id, info in database.items():
        video_id = str(raw_video_id)

        if isinstance(info, list):
            subset = infer_subset_from_video_id(video_id)
            duration = None
            annotations = info

        elif isinstance(info, dict):
            subset = normalize_subset_name(info.get("subset", None))
            if subset == "unknown":
                subset = infer_subset_from_video_id(video_id)

            duration = info.get("duration", None)
            annotations = info.get("annotations", [])

        else:
            continue

        clean_anns: List[Dict[str, Any]] = []

        for ann in annotations:
            if not isinstance(ann, dict):
                continue

            label = parse_label(ann)
            if label not in THUMOS14_CLASSES:
                continue

            seg = parse_segment(ann)
            if seg is None:
                continue

            start, end = seg
            start = float(start)
            end = float(end)

            if end <= start:
                continue

            clean_anns.append(
                {
                    "start": start,
                    "end": end,
                    "label": label,
                }
            )

        output[video_id] = {
            "subset": subset,
            "duration": duration,
            "annotations": sorted(clean_anns, key=lambda x: (x["start"], x["end"], x["label"])),
        }

    return output


def split_database_by_subset(
    database: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    train_annotations: Dict[str, List[Dict[str, Any]]] = {}
    test_annotations: Dict[str, List[Dict[str, Any]]] = {}

    for video_id, info in database.items():
        subset = normalize_subset_name(info.get("subset", "unknown"))
        anns = info.get("annotations", [])

        if subset == "validation":
            train_annotations[video_id] = anns
        elif subset == "test":
            test_annotations[video_id] = anns
        else:
            inferred = infer_subset_from_video_id(video_id)
            if inferred == "validation":
                train_annotations[video_id] = anns
            elif inferred == "test":
                test_annotations[video_id] = anns

    return train_annotations, test_annotations


def discover_video_files(video_dir: str) -> Dict[str, str]:
    root = Path(video_dir)
    if not root.exists():
        return {}

    exts = {".mp4", ".avi", ".mkv", ".webm"}
    video_map: Dict[str, str] = {}

    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            video_map[p.stem] = str(p)

    return dict(sorted(video_map.items(), key=lambda x: x[0]))


def discover_feature_files(
    feat_folder: Optional[str],
    file_ext: str = ".npy",
    feature_type: str = "",
    file_prefix: str = "",
) -> Dict[str, str]:
    if feat_folder is None:
        return {}

    root = Path(feat_folder)
    if not root.exists():
        return {}

    feat_map: Dict[str, str] = {}
    for p in root.rglob(f"*{file_ext}"):
        stem = p.stem

        video_id = stem
        if file_prefix and video_id.startswith(file_prefix):
            video_id = video_id[len(file_prefix):]
        if feature_type and video_id.endswith(feature_type):
            video_id = video_id[: -len(feature_type)]

        feat_map[video_id] = str(p)

    return dict(sorted(feat_map.items(), key=lambda x: x[0]))


def build_label_mapping(
    old_classes: List[str],
    new_classes: List[str],
) -> Dict[str, Dict[str, int]]:
    return {
        "old_label_to_id": {name: idx for idx, name in enumerate(old_classes)},
        "new_label_to_eval_id": {name: idx for idx, name in enumerate(new_classes)},
    }


def group_old_instances_by_class(
    train_annotations: Dict[str, List[Dict[str, Any]]],
    old_classes: List[str],
) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
    old_set = set(old_classes)
    grouped: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {c: [] for c in old_classes}

    for video_id in sorted(train_annotations.keys()):
        for ann in sorted(train_annotations[video_id], key=lambda x: (x["start"], x["end"], x["label"])):
            label = normalize_class_name(ann["label"])
            if label in old_set:
                grouped[label].append((video_id, ann))

    return grouped


def select_labeled_old_instances(
    train_annotations: Dict[str, List[Dict[str, Any]]],
    old_classes: List[str],
    labeled_ratio: float,
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    if not (0.0 < labeled_ratio <= 1.0):
        raise ValueError("--labeled_ratio must be in (0, 1].")

    grouped = group_old_instances_by_class(train_annotations, old_classes)
    rng = random.Random(seed)

    selected_by_video: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for cls in old_classes:
        items = grouped.get(cls, [])
        if not items:
            continue

        order = list(range(len(items)))
        rng.shuffle(order)

        if labeled_ratio >= 1.0:
            keep = set(order)
        else:
            keep_n = max(1, int(round(len(items) * labeled_ratio)))
            keep = set(order[:keep_n])

        for idx, (video_id, ann) in enumerate(items):
            if idx in keep:
                selected_by_video[video_id].append(ann)

    for video_id in selected_by_video:
        selected_by_video[video_id] = sorted(
            selected_by_video[video_id],
            key=lambda x: (x["start"], x["end"], x["label"]),
        )

    return dict(selected_by_video)


def add_paths(
    item: Dict[str, Any],
    video_map: Dict[str, str],
    feat_map: Dict[str, str],
    include_video_path: bool,
    include_feat_path: bool,
) -> Dict[str, Any]:
    video_id = item["video_id"]

    if include_video_path and video_id in video_map:
        item["video_path"] = video_map[video_id]

    if include_feat_path and video_id in feat_map:
        item["feat_path"] = feat_map[video_id]

    return item


def build_train_labeled(
    selected_old_annotations: Dict[str, List[Dict[str, Any]]],
    old_label_to_id: Dict[str, int],
    video_map: Dict[str, str],
    feat_map: Dict[str, str],
    include_video_path: bool,
    include_feat_path: bool,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for video_id in sorted(selected_old_annotations.keys()):
        out_anns = []

        for ann in selected_old_annotations[video_id]:
            label = normalize_class_name(ann["label"])
            if label not in old_label_to_id:
                continue

            out_anns.append(
                {
                    "start": float(ann["start"]),
                    "end": float(ann["end"]),
                    "label": label,
                    "label_id": int(old_label_to_id[label]),
                    "class_type": "old",
                }
            )

        if not out_anns:
            continue

        item: Dict[str, Any] = {
            "video_id": video_id,
            "subset": "train_labeled",
            "annotations": out_anns,
        }

        results.append(
            add_paths(
                item,
                video_map=video_map,
                feat_map=feat_map,
                include_video_path=include_video_path,
                include_feat_path=include_feat_path,
            )
        )

    return results


def filter_video_ids_for_unlabeled(
    train_annotations: Dict[str, List[Dict[str, Any]]],
    selected_old_annotations: Dict[str, List[Dict[str, Any]]],
    old_classes: List[str],
    new_classes: List[str],
    policy: str,
) -> List[str]:
    all_train_ids = sorted(train_annotations.keys())
    labeled_video_ids = set(selected_old_annotations.keys())
    old_set = set(old_classes)
    new_set = set(new_classes)

    if policy == "all_train":
        return all_train_ids

    if policy == "complement_labeled":
        return [v for v in all_train_ids if v not in labeled_video_ids]

    if policy == "videos_with_new":
        result = []
        for video_id, anns in train_annotations.items():
            if any(normalize_class_name(a["label"]) in new_set for a in anns):
                result.append(video_id)
        return sorted(result)

    if policy == "mixed_old_new":
        result = []
        for video_id, anns in train_annotations.items():
            if any(
                normalize_class_name(a["label"]) in old_set or normalize_class_name(a["label"]) in new_set
                for a in anns
            ):
                result.append(video_id)
        return sorted(result)

    raise ValueError(
        f"Unknown unlabeled_policy={policy}. "
        "Expected all_train, complement_labeled, videos_with_new, or mixed_old_new."
    )


def build_train_unlabeled(
    train_video_ids: List[str],
    video_map: Dict[str, str],
    feat_map: Dict[str, str],
    include_video_path: bool,
    include_feat_path: bool,
) -> List[Dict[str, Any]]:
    results = []

    for video_id in sorted(set(train_video_ids)):
        item: Dict[str, Any] = {
            "video_id": video_id,
            "subset": "train_unlabeled",
            "annotations": [],
        }

        results.append(
            add_paths(
                item,
                video_map=video_map,
                feat_map=feat_map,
                include_video_path=include_video_path,
                include_feat_path=include_feat_path,
            )
        )

    return results


def build_eval_split(
    annotations: Dict[str, List[Dict[str, Any]]],
    old_classes: List[str],
    new_classes: List[str],
    old_label_to_id: Dict[str, int],
    new_label_to_eval_id: Dict[str, int],
    video_map: Dict[str, str],
    feat_map: Dict[str, str],
    include_video_path: bool,
    include_feat_path: bool,
    keep_types: str,
    subset_name: str,
) -> List[Dict[str, Any]]:
    old_set = set(old_classes)
    new_set = set(new_classes)
    results: List[Dict[str, Any]] = []

    for video_id in sorted(annotations.keys()):
        out_anns = []

        for ann in sorted(annotations[video_id], key=lambda x: (x["start"], x["end"], x["label"])):
            label = normalize_class_name(ann["label"])

            if label in old_set and keep_types in {"all", "old"}:
                out_anns.append(
                    {
                        "start": float(ann["start"]),
                        "end": float(ann["end"]),
                        "label": label,
                        "label_id": int(old_label_to_id[label]),
                        "class_type": "old",
                    }
                )

            elif label in new_set and keep_types in {"all", "new"}:
                out_anns.append(
                    {
                        "start": float(ann["start"]),
                        "end": float(ann["end"]),
                        "label": label,
                        "eval_id": int(new_label_to_eval_id[label]),
                        "class_type": "new",
                    }
                )

        if not out_anns:
            continue

        item: Dict[str, Any] = {
            "video_id": video_id,
            "subset": subset_name,
            "annotations": out_anns,
        }

        results.append(
            add_paths(
                item,
                video_map=video_map,
                feat_map=feat_map,
                include_video_path=include_video_path,
                include_feat_path=include_feat_path,
            )
        )

    return results


def count_instances(entries: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = defaultdict(int)
    total = 0

    for item in entries:
        for ann in item.get("annotations", []):
            label = ann.get("label", "UNKNOWN")
            counts[label] += 1
            total += 1

    out = dict(sorted(counts.items(), key=lambda x: x[0]))
    out["_total"] = total
    return out


def count_class_types(entries: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = defaultdict(int)

    for item in entries:
        for ann in item.get("annotations", []):
            counts[ann.get("class_type", "unknown")] += 1

    return dict(counts)


def count_paths(entries: List[Dict[str, Any]], key: str) -> int:
    return sum(1 for item in entries if key in item and item[key] is not None)


def summarize_hidden_train_new(
    train_annotations: Dict[str, List[Dict[str, Any]]],
    new_classes: List[str],
) -> Dict[str, Any]:
    new_set = set(new_classes)
    counts = defaultdict(int)
    videos = set()

    for video_id, anns in train_annotations.items():
        for ann in anns:
            label = normalize_class_name(ann["label"])
            if label in new_set:
                counts[label] += 1
                videos.add(video_id)

    out = dict(sorted(counts.items(), key=lambda x: x[0]))
    out["_total"] = sum(v for k, v in out.items() if not k.startswith("_"))

    return {
        "num_videos_with_hidden_new": len(videos),
        "hidden_new_instance_count_by_class": out,
        "warning": "This summary is for debugging/reporting only and must not be used by training code.",
    }


def validate_no_new_leakage(
    train_labeled: List[Dict[str, Any]],
    train_unlabeled: List[Dict[str, Any]],
    new_classes: List[str],
) -> None:
    new_set = set(new_classes)

    leaked = []
    for item in train_labeled:
        for ann in item.get("annotations", []):
            if ann.get("label") in new_set or ann.get("class_type") == "new" or "eval_id" in ann:
                leaked.append((item.get("video_id"), ann))

    if leaked:
        raise RuntimeError(
            "New-class leakage detected in train_labeled. "
            f"First leaked item: {leaked[0]}"
        )

    bad_unlabeled = []
    for item in train_unlabeled:
        if item.get("annotations", None) != []:
            bad_unlabeled.append(item.get("video_id"))

    if bad_unlabeled:
        raise RuntimeError(
            "train_unlabeled must have empty annotations. "
            f"First bad video: {bad_unlabeled[0]}"
        )


def validate_split(
    old_classes: List[str],
    new_classes: List[str],
    train_labeled: List[Dict[str, Any]],
    train_unlabeled: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
    strict: bool,
) -> Dict[str, Any]:
    old_set = set(old_classes)
    new_set = set(new_classes)

    overlap = sorted(old_set.intersection(new_set))
    if overlap:
        raise RuntimeError(f"old/new classes overlap: {overlap}")

    missing_taxonomy = sorted(set(THUMOS14_CLASSES) - old_set - new_set)
    if missing_taxonomy:
        raise RuntimeError(f"old/new classes do not cover THUMOS14 taxonomy: {missing_taxonomy}")

    train_counts = count_instances(train_labeled)
    test_counts = count_instances(test)

    missing_train_old = [c for c in old_classes if train_counts.get(c, 0) == 0]
    missing_test_old = [c for c in old_classes if test_counts.get(c, 0) == 0]
    missing_test_new = [c for c in new_classes if test_counts.get(c, 0) == 0]

    validate_no_new_leakage(train_labeled, train_unlabeled, new_classes)

    if strict:
        errors = []

        if not train_labeled:
            errors.append("train_labeled is empty.")
        if not train_unlabeled:
            errors.append("train_unlabeled is empty.")
        if not test:
            errors.append("test is empty.")
        if missing_train_old:
            errors.append(f"missing_train_old_classes: {missing_train_old}")
        if missing_test_old:
            errors.append(f"missing_test_old_classes: {missing_test_old}")
        if missing_test_new:
            errors.append(f"missing_test_new_classes: {missing_test_new}")

        if errors:
            raise RuntimeError("\n".join(errors))

    return {
        "missing_train_old_classes": missing_train_old,
        "missing_test_old_classes": missing_test_old,
        "missing_test_new_classes": missing_test_new,
    }


def preview_json_structure(annotation_json: str, n: int = 2) -> None:
    path = Path(annotation_json)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    print("=" * 80)
    print("Annotation JSON preview")
    print("=" * 80)

    if isinstance(raw, dict):
        print("top-level keys:", list(raw.keys())[:20])
        database = raw.get("database", raw)

        if isinstance(database, dict):
            keys = list(database.keys())[:n]
            print("num videos:", len(database))
            print("first video ids:", keys)

            for vid in keys:
                print("-" * 80)
                print("video_id:", vid)
                info = database[vid]
                print("type:", type(info))

                if isinstance(info, dict):
                    print("keys:", list(info.keys()))
                    print("subset:", info.get("subset", None))
                    anns = info.get("annotations", [])
                    print("num annotations:", len(anns) if isinstance(anns, list) else "not-list")
                    if isinstance(anns, list) and anns:
                        print("first annotation:", anns[0])

                elif isinstance(info, list):
                    print("annotation-list length:", len(info))
                    if info:
                        print("first annotation:", info[0])

    else:
        print("unsupported top-level type:", type(raw))

    print("=" * 80)


def write_json(obj: Dict[str, Any], output: str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build THUMOS14-TAD-GCD split aligned with Code-2114 open-class splits."
    )

    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--annotation_json", type=str, default=DEFAULT_ANNOTATION_JSON)
    parser.add_argument("--video_dir", type=str, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--feat_folder", type=str, default=None)
    parser.add_argument("--file_ext", type=str, default=".npy")
    parser.add_argument("--feature_type", type=str, default="")
    parser.add_argument("--file_prefix", type=str, default="")

    parser.add_argument("--split_root", type=str, default="./splits")
    parser.add_argument("--data_split", type=int, default=50, choices=[50, 75])
    parser.add_argument("--split_num", type=int, default=0)

    parser.add_argument("--output", type=str, default=None)

    parser.add_argument(
        "--labeled_ratio",
        type=float,
        default=1.0,
        help="Visible old-class instance ratio in train_labeled.",
    )

    parser.add_argument(
        "--unlabeled_policy",
        type=str,
        default="all_train",
        choices=["all_train", "complement_labeled", "videos_with_new", "mixed_old_new"],
    )

    parser.add_argument("--no_video_path", action="store_true")
    parser.add_argument("--include_feat_path", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--preview_json", action="store_true")

    args = parser.parse_args()

    if args.preview_json:
        preview_json_structure(args.annotation_json)
        return

    if args.output is None:
        ratio_tag = int(round(args.labeled_ratio * 100))
        args.output = (
            f"data/thumos14_tad_gcd_code2114_"
            f"train{args.data_split}_split{args.split_num}_ratio{ratio_tag}.json"
        )

    include_video_path = not args.no_video_path
    include_feat_path = args.include_feat_path

    old_classes, new_classes, split_meta = build_class_split_from_code2114(
        split_root=args.split_root,
        data_split=args.data_split,
        split_num=args.split_num,
    )

    label_mapping = build_label_mapping(old_classes, new_classes)
    old_label_to_id = label_mapping["old_label_to_id"]
    new_label_to_eval_id = label_mapping["new_label_to_eval_id"]

    print("=" * 80)
    print("Build THUMOS14-TAD-GCD split aligned with Code-2114")
    print("=" * 80)
    print(f"data_root        : {args.data_root}")
    print(f"annotation_json  : {args.annotation_json}")
    print(f"video_dir        : {args.video_dir}")
    print(f"feat_folder      : {args.feat_folder}")
    print(f"split_root       : {args.split_root}")
    print(f"data_split       : {args.data_split}")
    print(f"split_num        : {args.split_num}")
    print(f"old list         : {split_meta['old_classes_file']}")
    print(f"new list         : {split_meta['new_classes_file']}")
    print(f"output           : {args.output}")
    print(f"labeled_ratio    : {args.labeled_ratio}")
    print(f"unlabeled_policy : {args.unlabeled_policy}")
    print("-" * 80)
    print("old_classes:", old_classes)
    print("new_classes:", new_classes)
    print("-" * 80)

    database = load_thumos14_json(args.annotation_json)
    train_annotations, test_annotations = split_database_by_subset(database)
    video_map = discover_video_files(args.video_dir)
    feat_map = discover_feature_files(
        feat_folder=args.feat_folder,
        file_ext=args.file_ext,
        feature_type=args.feature_type,
        file_prefix=args.file_prefix,
    )

    selected_old_annotations = select_labeled_old_instances(
        train_annotations=train_annotations,
        old_classes=old_classes,
        labeled_ratio=args.labeled_ratio,
        seed=args.split_num,
    )

    train_labeled = build_train_labeled(
        selected_old_annotations=selected_old_annotations,
        old_label_to_id=old_label_to_id,
        video_map=video_map,
        feat_map=feat_map,
        include_video_path=include_video_path,
        include_feat_path=include_feat_path,
    )

    train_unlabeled_video_ids = filter_video_ids_for_unlabeled(
        train_annotations=train_annotations,
        selected_old_annotations=selected_old_annotations,
        old_classes=old_classes,
        new_classes=new_classes,
        policy=args.unlabeled_policy,
    )

    train_unlabeled = build_train_unlabeled(
        train_video_ids=train_unlabeled_video_ids,
        video_map=video_map,
        feat_map=feat_map,
        include_video_path=include_video_path,
        include_feat_path=include_feat_path,
    )

    test = build_eval_split(
        annotations=test_annotations,
        old_classes=old_classes,
        new_classes=new_classes,
        old_label_to_id=old_label_to_id,
        new_label_to_eval_id=new_label_to_eval_id,
        video_map=video_map,
        feat_map=feat_map,
        include_video_path=include_video_path,
        include_feat_path=include_feat_path,
        keep_types="all",
        subset_name="test",
    )

    test_old = build_eval_split(
        annotations=test_annotations,
        old_classes=old_classes,
        new_classes=new_classes,
        old_label_to_id=old_label_to_id,
        new_label_to_eval_id=new_label_to_eval_id,
        video_map=video_map,
        feat_map=feat_map,
        include_video_path=include_video_path,
        include_feat_path=include_feat_path,
        keep_types="old",
        subset_name="test_old",
    )

    test_new = build_eval_split(
        annotations=test_annotations,
        old_classes=old_classes,
        new_classes=new_classes,
        old_label_to_id=old_label_to_id,
        new_label_to_eval_id=new_label_to_eval_id,
        video_map=video_map,
        feat_map=feat_map,
        include_video_path=include_video_path,
        include_feat_path=include_feat_path,
        keep_types="new",
        subset_name="test_new",
    )

    missing_info = validate_split(
        old_classes=old_classes,
        new_classes=new_classes,
        train_labeled=train_labeled,
        train_unlabeled=train_unlabeled,
        test=test,
        strict=args.strict,
    )

    train_counts = count_instances(train_labeled)
    test_counts = count_instances(test)
    test_old_counts = count_instances(test_old)
    test_new_counts = count_instances(test_new)
    hidden_new_summary = summarize_hidden_train_new(train_annotations, new_classes)

    output_data: Dict[str, Any] = {
        "dataset": "THUMOS14",
        "task": "TAD-GCD",
        "seed": args.split_num,
        "protocol_version": "v0.4_code2114_aligned",
        "data_root": args.data_root,
        "annotation_json": args.annotation_json,
        "video_dir": args.video_dir,
        "feat_folder": args.feat_folder,
        "split_config": {
            "source": "Code-2114 open-class split lists",
            "split_root": args.split_root,
            "data_split": args.data_split,
            "split_num": args.split_num,
            "old_classes_file": split_meta["old_classes_file"],
            "new_classes_file": split_meta["new_classes_file"],
            "labeled_ratio": args.labeled_ratio,
            "unlabeled_policy": args.unlabeled_policy,
        },
        "class_split": {
            "old_classes": old_classes,
            "new_classes": new_classes,
        },
        "label_mapping": label_mapping,
        "train_labeled": train_labeled,
        "train_unlabeled": train_unlabeled,
        "test": test,
        "test_old": test_old,
        "test_new": test_new,
        "train_hidden_new_summary": hidden_new_summary,
        "statistics": {
            "num_videos_in_json": len(database),
            "num_video_files_found": len(video_map),
            "num_feature_files_found": len(feat_map),
            "num_train_annotation_videos": len(train_annotations),
            "num_test_annotation_videos": len(test_annotations),
            "num_train_labeled_videos": len(train_labeled),
            "num_train_unlabeled_videos": len(train_unlabeled),
            "num_test_videos": len(test),
            "num_test_old_videos": len(test_old),
            "num_test_new_videos": len(test_new),
            "num_train_labeled_videos_with_video_path": count_paths(train_labeled, "video_path"),
            "num_train_unlabeled_videos_with_video_path": count_paths(train_unlabeled, "video_path"),
            "num_test_videos_with_video_path": count_paths(test, "video_path"),
            "num_train_labeled_videos_with_feat_path": count_paths(train_labeled, "feat_path"),
            "num_train_unlabeled_videos_with_feat_path": count_paths(train_unlabeled, "feat_path"),
            "num_test_videos_with_feat_path": count_paths(test, "feat_path"),
            "train_labeled_instance_count_by_class": train_counts,
            "test_instance_count_by_class": test_counts,
            "test_old_instance_count_by_class": test_old_counts,
            "test_new_instance_count_by_class": test_new_counts,
            "train_labeled_class_type_counts": count_class_types(train_labeled),
            "test_class_type_counts": count_class_types(test),
            **missing_info,
        },
        "notes": {
            "unknown_definition": (
                "Unknown classes are Code-2114 test split classes, i.e. held-out THUMOS14 action classes. "
                "They belong to the benchmark taxonomy but are removed from the supervised training label space."
            ),
            "training_restriction": (
                "train_labeled contains only old-class annotations. "
                "train_unlabeled exposes video ids only and keeps annotations empty."
            ),
            "new_class_forbidden_information": [
                "new-class labels",
                "new-class temporal boundaries",
                "new-class names",
                "new-class text descriptions",
                "new-class prompts",
                "new_label_to_eval_id in training",
            ],
            "evaluation": (
                "test contains old and new annotations. "
                "test_old and test_new are convenience evaluation subsets. "
                "New-class cluster IDs should be aligned to eval_id by Hungarian matching."
            ),
            "warning": (
                "train_hidden_new_summary is for debugging/reporting only. "
                "Do not feed it into training code."
            ),
        },
    }

    write_json(output_data, args.output)

    print("Done.")
    print("-" * 80)
    print(f"Saved to: {args.output}")
    print(f"videos in json                  : {len(database)}")
    print(f"video files found               : {len(video_map)}")
    print(f"feature files found             : {len(feat_map)}")
    print(f"train annotation videos         : {len(train_annotations)}")
    print(f"test annotation videos          : {len(test_annotations)}")
    print(f"train_labeled videos            : {len(train_labeled)}")
    print(f"train_unlabeled videos          : {len(train_unlabeled)}")
    print(f"test videos                     : {len(test)}")
    print(f"test_old videos                 : {len(test_old)}")
    print(f"test_new videos                 : {len(test_new)}")
    print(f"train_labeled instances         : {train_counts.get('_total', 0)}")
    print(f"test instances                  : {test_counts.get('_total', 0)}")
    print(f"test_old instances              : {test_old_counts.get('_total', 0)}")
    print(f"test_new instances              : {test_new_counts.get('_total', 0)}")
    print(f"hidden train new instances      : {hidden_new_summary['hidden_new_instance_count_by_class'].get('_total', 0)}")
    print("-" * 80)
    print("Leakage check:")
    print("  train_labeled contains only old annotations: OK")
    print("  train_unlabeled annotations are empty      : OK")
    print("-" * 80)
    print("Missing class checks:")
    print(f"  missing_train_old: {missing_info['missing_train_old_classes']}")
    print(f"  missing_test_old : {missing_info['missing_test_old_classes']}")
    print(f"  missing_test_new : {missing_info['missing_test_new_classes']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
