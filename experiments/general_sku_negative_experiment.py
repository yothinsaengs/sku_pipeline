import argparse
import copy
import glob
import math
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


Box = Dict[str, Any]
ImageRecord = Dict[str, Any]


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_class_names(data_dir: str, class_ids: Iterable[int], config: Dict[str, Any]) -> Dict[int, str]:
    ids = set(class_ids)
    names = {cls["id"]: cls["name"] for cls in config.get("classes", []) if cls["id"] in ids}
    candidates = [
        os.path.join(data_dir, "classes.txt"),
        os.path.join(data_dir, "obj.names"),
        os.path.join(os.path.dirname(data_dir), "classes.txt"),
    ]

    for path in candidates:
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        for class_id in ids:
            if class_id < len(lines):
                names[class_id] = lines[class_id]
        break

    for class_id in ids:
        names.setdefault(class_id, f"class_{class_id}")
    return names


def discover_dataset(data_path: str):
    from sku_clf.utils.io import extract_dataset, resolve_data_paths

    data_dir = extract_dataset(data_path)
    images_dir, labels_dir = resolve_data_paths(data_dir)
    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"Expected YOLO images/ and labels/ directories under: {data_dir}")
    return data_dir, images_dir, labels_dir


def parse_label_file(label_path: str) -> List[Box]:
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            bbox = [float(value) for value in parts[1:]]
            boxes.append({"class_id": class_id, "bbox": bbox})
    return boxes


def load_image_records(data_path: str, config: Dict[str, Any]) -> Tuple[List[ImageRecord], Dict[int, str]]:
    data_dir, images_dir, labels_dir = discover_dataset(data_path)
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    image_files.sort()

    records = []
    class_ids = set()
    for image_path in image_files:
        basename = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(labels_dir, basename + ".txt")
        if not os.path.exists(label_path):
            continue
        boxes = parse_label_file(label_path)
        if not boxes:
            continue
        class_ids.update(box["class_id"] for box in boxes)
        records.append({"image_path": image_path, "label_path": label_path, "boxes": boxes})

    if not records:
        raise ValueError(f"No labeled images found under: {data_dir}")

    return records, load_class_names(data_dir, class_ids, config)


def resolve_class_ref(
    ref: Optional[str],
    id_to_name: Dict[int, str],
    description: str,
) -> int:
    if ref is None:
        raise ValueError(f"{description} is required")

    try:
        class_id = int(ref)
    except ValueError:
        matches = [class_id for class_id, name in id_to_name.items() if name == ref]
        if not matches:
            available = ", ".join(f"{class_id}:{name}" for class_id, name in sorted(id_to_name.items()))
            raise ValueError(f"Unknown {description} '{ref}'. Available classes: {available}")
        return matches[0]

    if class_id not in id_to_name:
        available = ", ".join(f"{cid}:{name}" for cid, name in sorted(id_to_name.items()))
        raise ValueError(f"Unknown {description} id '{class_id}'. Available classes: {available}")
    return class_id


def resolve_class_refs(refs: Sequence[str], id_to_name: Dict[int, str]) -> List[int]:
    return list(dict.fromkeys(resolve_class_ref(ref, id_to_name, "positive class") for ref in refs))


def yolo_to_xyxy(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    xc, yc, bw, bh = bbox
    return xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = yolo_to_xyxy(box_a)
    bx1, by1, bx2, by2 = yolo_to_xyxy(box_b)

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def split_records(
    records: Sequence[ImageRecord],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[ImageRecord], List[ImageRecord], List[ImageRecord]]:
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)

    num_train = int(len(indices) * train_ratio)
    num_val = int(len(indices) * val_ratio)

    train = [records[i] for i in indices[:num_train]]
    val = [records[i] for i in indices[num_train:num_train + num_val]]
    test = [records[i] for i in indices[num_train + num_val:]]
    return train, val, test


def sample_negative_boxes(boxes: Sequence[Box], fraction: float, rng: random.Random) -> List[Box]:
    if fraction <= 0 or not boxes:
        return []
    if fraction >= 1:
        return list(boxes)

    count = max(1, math.ceil(len(boxes) * fraction))
    return rng.sample(list(boxes), min(count, len(boxes)))


def record_to_samples(
    records: Sequence[ImageRecord],
    positive_ids: Sequence[int],
    gss_class_id: int,
    gss_fraction: float,
    overlap_iou: float,
    filter_overlap: bool,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rng = random.Random(seed)
    positive_set = set(positive_ids)
    samples = []
    stats = {
        "positive_boxes": 0,
        "gss_boxes_total": 0,
        "gss_boxes_after_overlap_filter": 0,
        "gss_boxes_sampled": 0,
    }
    gss_candidates = []

    for record in records:
        positive_boxes = [box for box in record["boxes"] if box["class_id"] in positive_set]
        gss_boxes = [box for box in record["boxes"] if box["class_id"] == gss_class_id]

        stats["positive_boxes"] += len(positive_boxes)
        stats["gss_boxes_total"] += len(gss_boxes)

        if filter_overlap:
            gss_boxes = [
                box
                for box in gss_boxes
                if all(iou(box["bbox"], pos["bbox"]) < overlap_iou for pos in positive_boxes)
            ]

        stats["gss_boxes_after_overlap_filter"] += len(gss_boxes)

        for box in positive_boxes:
            samples.append({
                "image_path": record["image_path"],
                "bbox": box["bbox"],
                "class_id": box["class_id"],
                "role": "positive",
            })
        for box in gss_boxes:
            gss_candidates.append({
                "image_path": record["image_path"],
                "bbox": box["bbox"],
                "class_id": box["class_id"],
                "role": "negative",
            })

    sampled_gss = sample_negative_boxes(gss_candidates, gss_fraction, rng)
    stats["gss_boxes_sampled"] = len(sampled_gss)
    samples.extend(sampled_gss)
    return samples, stats


def build_experiment_config(
    base_config: Dict[str, Any],
    id_to_name: Dict[int, str],
    positive_ids: Sequence[int],
    gss_class_id: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    config = copy.deepcopy(base_config)
    config.pop("class_split_experiment", None)
    config["classes"] = [
        {"id": class_id, "name": id_to_name[class_id], "role": "positive"}
        for class_id in positive_ids
    ]
    config["classes"].append({"id": gss_class_id, "name": id_to_name[gss_class_id], "role": "negative"})
    config["model"]["num_classes"] = len(positive_ids)
    config["dataset"]["split"] = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    config["dataset"]["split_seed"] = args.seed

    if args.lr is not None:
        config["training"]["lr"] = args.lr
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.pos_neg_ratio is not None:
        config.setdefault("sampling", {})["pos_neg_ratio"] = args.pos_neg_ratio

    config["experiment"] = {
        "name": "general_sku_negative",
        "gss_class_id": gss_class_id,
        "gss_class_name": id_to_name[gss_class_id],
        "train_gss_fraction": args.train_gss_fraction,
        "val_gss_fraction": args.val_gss_fraction,
        "test_gss_fraction": args.test_gss_fraction,
        "pos_neg_ratio": config.get("sampling", {}).get("pos_neg_ratio", "1:1"),
        "filter_gss_overlap": args.filter_gss_overlap,
        "overlap_iou": args.overlap_iou,
    }
    return config


class SampleListSKUROIDataset:
    def __new__(cls, samples, config, transform, is_training):
        from sku_clf.data.dataset import SKUROIDataset

        class _Dataset(SKUROIDataset):
            def __init__(self, provided_samples, dataset_config, dataset_transform, training):
                self._provided_samples = list(provided_samples)
                super().__init__([], [], dataset_config, transform=dataset_transform, is_training=training)

            def _prepare_samples(self):
                return self._provided_samples

        return _Dataset(samples, config, transform, is_training)


def build_dataloaders(
    config: Dict[str, Any],
    train_samples: Sequence[Dict[str, Any]],
    val_samples: Sequence[Dict[str, Any]],
    test_samples: Sequence[Dict[str, Any]],
):
    from torch.utils.data import DataLoader

    from sku_clf.data.augment import get_transforms
    from sku_clf.data.sampler import BatchRatioSampler

    train_ds = SampleListSKUROIDataset(train_samples, config, get_transforms(config, True), True)
    val_ds = SampleListSKUROIDataset(val_samples, config, get_transforms(config, False), False)
    test_ds = SampleListSKUROIDataset(test_samples, config, get_transforms(config, False), False)

    batch_size = config["training"]["batch_size"]
    ratio = config["sampling"].get("pos_neg_ratio", "1:1")
    num_workers = config["training"].get("num_workers", 0)
    train_sampler = BatchRatioSampler(train_ds.samples, batch_size, ratio)

    return (
        DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=False),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False),
    )


def print_summary(title: str, samples: Sequence[Dict[str, Any]], stats: Dict[str, int]) -> None:
    positives = sum(1 for sample in samples if sample["role"] == "positive")
    negatives = sum(1 for sample in samples if sample["role"] == "negative")
    print(f"{title}:")
    print(f"  positives: {positives}")
    print(f"  GSS negatives sampled: {negatives}")
    print(f"  GSS boxes total: {stats['gss_boxes_total']}")
    print(f"  GSS boxes after overlap filter: {stats['gss_boxes_after_overlap_filter']}")


def print_sampler_expectation(config: Dict[str, Any], train_samples: Sequence[Dict[str, Any]]) -> None:
    from sku_clf.data.sampler import BatchRatioSampler

    sampler = BatchRatioSampler(
        list(train_samples),
        config["training"]["batch_size"],
        config.get("sampling", {}).get("pos_neg_ratio", "1:1"),
    )
    print(
        "Train sampler:"
        f" expected batch mix {sampler.num_pos_per_batch} positive / {sampler.num_neg_per_batch} negative;"
        " repeats samples with replacement when a side is scarce"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train all known SKU classes as positives with sampled GENERAL_SKU_SINGLE negatives.",
    )
    parser.add_argument("--config", default="sku_classifier/config.yaml", help="Base package config path")
    parser.add_argument("--data", required=True, help="YOLO dataset zip or directory")
    parser.add_argument("--gss-class", default="GENERAL_SKU_SINGLE", help="GSS class name or id")
    parser.add_argument("--positive", nargs="*", default=[], help="Optional positive class names or ids")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--train-gss-fraction", type=float, default=0.2)
    parser.add_argument("--val-gss-fraction", type=float, default=1.0)
    parser.add_argument("--test-gss-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--filter-gss-overlap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overlap-iou", type=float, default=0.5)
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--pos-neg-ratio", type=str, help="Override positive:negative sampler ratio, e.g. 1:1 or 1:2")
    parser.add_argument("--device", help="Override device: cpu, cuda, or mps")
    parser.add_argument("--dry-run", action="store_true", help="Print split/sample counts without training")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("--train-ratio + --val-ratio + --test-ratio must equal 1.0")

    for name in ("train_gss_fraction", "val_gss_fraction", "test_gss_fraction"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")


def main() -> None:
    args = parse_args()
    validate_args(args)

    from sku_clf.utils.validators import validate_config

    base_config = load_config(args.config)
    records, id_to_name = load_image_records(args.data, base_config)
    gss_class_id = resolve_class_ref(args.gss_class, id_to_name, "GSS class")

    if args.positive:
        positive_ids = resolve_class_refs(args.positive, id_to_name)
    else:
        positive_ids = [class_id for class_id in sorted(id_to_name) if class_id != gss_class_id]

    if not positive_ids:
        raise ValueError("No positive classes selected")

    config = build_experiment_config(base_config, id_to_name, positive_ids, gss_class_id, args)
    validate_config(config)

    train_records, val_records, test_records = split_records(
        records,
        args.train_ratio,
        args.val_ratio,
        args.seed,
    )

    train_samples, train_stats = record_to_samples(
        train_records,
        positive_ids,
        gss_class_id,
        args.train_gss_fraction,
        args.overlap_iou,
        args.filter_gss_overlap,
        args.seed,
    )
    val_samples, val_stats = record_to_samples(
        val_records,
        positive_ids,
        gss_class_id,
        args.val_gss_fraction,
        args.overlap_iou,
        args.filter_gss_overlap,
        args.seed + 1,
    )
    test_samples, test_stats = record_to_samples(
        test_records,
        positive_ids,
        gss_class_id,
        args.test_gss_fraction,
        args.overlap_iou,
        args.filter_gss_overlap,
        args.seed + 2,
    )

    print("General SKU negative experiment")
    print(f"  GSS class: {gss_class_id}:{id_to_name[gss_class_id]}")
    print(f"  positive classes: {len(positive_ids)}")
    print(f"  image split ratio: {args.train_ratio:.2f}/{args.val_ratio:.2f}/{args.test_ratio:.2f}")
    print(f"  image split count: train={len(train_records)}, val={len(val_records)}, test={len(test_records)}")
    print(f"  train GSS fraction: {args.train_gss_fraction:.2f}")
    print(f"  batch pos:neg ratio: {config.get('sampling', {}).get('pos_neg_ratio', '1:1')}")
    print(f"  filter overlap: {args.filter_gss_overlap} at IoU {args.overlap_iou:.2f}")
    print_summary("Train", train_samples, train_stats)
    print_summary("Val", val_samples, val_stats)
    print_summary("Test", test_samples, test_stats)
    print_sampler_expectation(config, train_samples)

    if args.dry_run:
        return

    if not any(sample["role"] == "positive" for sample in train_samples):
        raise ValueError("Training split has no positive samples")
    if not any(sample["role"] == "negative" for sample in train_samples):
        raise ValueError("Training split has no sampled GSS negatives")

    from sku_clf.train import train_with_dataloaders

    train_loader, val_loader, test_loader = build_dataloaders(
        config,
        train_samples,
        val_samples,
        test_samples,
    )
    train_with_dataloaders(config, train_loader, val_loader, test_loader)


if __name__ == "__main__":
    main()
