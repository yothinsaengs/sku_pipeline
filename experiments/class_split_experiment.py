import argparse
import copy
import glob
import os
import random
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def class_label(cls: Dict[str, Any]) -> str:
    return f"{cls['id']}:{cls['name']}"


def resolve_class_refs(classes: Sequence[Dict[str, Any]], refs: Iterable[str]) -> List[int]:
    by_id = {int(cls["id"]): cls for cls in classes}
    by_name = {cls["name"]: cls for cls in classes}
    resolved = []

    for ref in refs:
        if ref in by_name:
            resolved.append(by_name[ref]["id"])
            continue

        try:
            class_id = int(ref)
        except ValueError as exc:
            choices = ", ".join(class_label(cls) for cls in classes)
            raise ValueError(f"Unknown class reference '{ref}'. Available classes: {choices}") from exc

        if class_id not in by_id:
            choices = ", ".join(class_label(cls) for cls in classes)
            raise ValueError(f"Unknown class id '{class_id}'. Available classes: {choices}")
        resolved.append(class_id)

    return list(dict.fromkeys(resolved))


def choose_class_split(args: argparse.Namespace, classes: Sequence[Dict[str, Any]]) -> Tuple[List[int], List[int]]:
    all_ids = [cls["id"] for cls in classes]

    if args.mode == "explicit":
        if not args.positive:
            raise ValueError("--positive is required for explicit class split experiments")

        pos_ids = resolve_class_refs(classes, args.positive)
        negative_refs = args.negative or []
        negative_all_remaining = args.negative_all_remaining or "ALL_REMAIN" in negative_refs

        if negative_all_remaining:
            neg_ids = [class_id for class_id in all_ids if class_id not in pos_ids]
        else:
            neg_ids = resolve_class_refs(classes, negative_refs)
    else:
        if not 0 < args.pos_fraction <= 1:
            raise ValueError("--pos-fraction must be in (0, 1]")
        if not 0 <= args.neg_fraction <= 1:
            raise ValueError("--neg-fraction must be in [0, 1]")

        shuffled = all_ids.copy()
        random.Random(args.seed).shuffle(shuffled)
        num_pos = max(1, int(len(shuffled) * args.pos_fraction))
        num_neg = int(len(shuffled) * args.neg_fraction)
        if num_pos + num_neg > len(shuffled):
            raise ValueError("--pos-fraction + --neg-fraction cannot select more than all classes")

        pos_ids = shuffled[:num_pos]
        neg_ids = shuffled[num_pos:num_pos + num_neg]

    overlap = set(pos_ids) & set(neg_ids)
    if overlap:
        raise ValueError(f"Classes cannot be both positive and negative: {sorted(overlap)}")

    return pos_ids, neg_ids


def build_config(base_config: Dict[str, Any], positive_ids: Sequence[int], negative_ids: Sequence[int]) -> Dict[str, Any]:
    by_id = {cls["id"]: cls for cls in base_config["classes"]}
    config = copy.deepcopy(base_config)
    config.pop("class_split_experiment", None)

    classes = []
    for class_id in positive_ids:
        cls = by_id[class_id]
        classes.append({"id": cls["id"], "name": cls["name"], "role": "positive"})
    for class_id in negative_ids:
        cls = by_id[class_id]
        classes.append({"id": cls["id"], "name": cls["name"], "role": "negative"})

    config["classes"] = classes
    config["model"]["num_classes"] = len(positive_ids)
    return config


def apply_training_overrides(config: Dict[str, Any], args: argparse.Namespace) -> None:
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


def discover_image_label_pairs(data_path: str) -> List[Tuple[str, str]]:
    from sku_clf.utils.io import extract_dataset, resolve_data_paths

    data_dir = extract_dataset(data_path)
    images_dir, labels_dir = resolve_data_paths(data_dir)

    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"Expected YOLO images/ and labels/ directories under: {data_dir}")

    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    image_files.sort()

    pairs = []
    for image_path in image_files:
        basename = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(labels_dir, basename + ".txt")
        if os.path.exists(label_path):
            pairs.append((image_path, label_path))

    if not pairs:
        raise ValueError(f"No image/label pairs found under: {data_dir}")
    return pairs


def split_pairs(
    pairs: Sequence[Tuple[str, str]],
    split_config: Dict[str, float],
    seed: int,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    indices = list(range(len(pairs)))
    random.Random(seed).shuffle(indices)

    num_train = int(len(indices) * split_config["train"])
    num_val = int(len(indices) * split_config["val"])

    train_pairs = [pairs[i] for i in indices[:num_train]]
    val_pairs = [pairs[i] for i in indices[num_train:num_train + num_val]]
    test_pairs = [pairs[i] for i in indices[num_train + num_val:]]
    return train_pairs, val_pairs, test_pairs


def unzip_pairs(pairs: Sequence[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
    if not pairs:
        return [], []
    images, labels = zip(*pairs)
    return list(images), list(labels)


def build_dataloaders(train_config: Dict[str, Any], eval_config: Dict[str, Any], data_path: str):
    from torch.utils.data import DataLoader

    from sku_clf.data.augment import get_transforms
    from sku_clf.data.dataset import SKUROIDataset
    from sku_clf.data.sampler import BatchRatioSampler

    pairs = discover_image_label_pairs(data_path)
    train_pairs, val_pairs, test_pairs = split_pairs(
        pairs,
        train_config["dataset"]["split"],
        train_config["dataset"].get("split_seed", 42),
    )

    train_imgs, train_lbls = unzip_pairs(train_pairs)
    val_imgs, val_lbls = unzip_pairs(val_pairs)
    test_imgs, test_lbls = unzip_pairs(test_pairs)

    train_ds = SKUROIDataset(
        train_imgs,
        train_lbls,
        train_config,
        transform=get_transforms(train_config, True),
        is_training=True,
    )
    val_ds = SKUROIDataset(
        val_imgs,
        val_lbls,
        eval_config,
        transform=get_transforms(eval_config, False),
        is_training=False,
    )
    test_ds = SKUROIDataset(
        test_imgs,
        test_lbls,
        eval_config,
        transform=get_transforms(eval_config, False),
        is_training=False,
    )

    train_pos = sum(1 for sample in train_ds.samples if sample["role"] == "positive")
    train_neg = sum(1 for sample in train_ds.samples if sample["role"] in {"negative", "synthetic"})
    if train_pos == 0:
        raise ValueError("The training split has no positive ROI samples after class filtering")
    if train_neg == 0:
        raise ValueError("The training split has no negative ROI samples after class filtering")

    batch_size = train_config["training"]["batch_size"]
    ratio = train_config["sampling"].get("pos_neg_ratio", "1:1")
    train_sampler = BatchRatioSampler(train_ds.samples, batch_size, ratio)
    num_workers = train_config["training"].get("num_workers", 0)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader, test_loader


def summarize_split(
    base_classes: Sequence[Dict[str, Any]],
    positive_ids: Sequence[int],
    train_negative_ids: Sequence[int],
    eval_extra_negative_ids: Sequence[int],
) -> None:
    by_id = {cls["id"]: cls for cls in base_classes}

    def names(ids: Sequence[int]) -> str:
        return ", ".join(by_id[class_id]["name"] for class_id in ids) or "(none)"

    print("Class split experiment")
    print(f"  positive: {names(positive_ids)}")
    print(f"  train negatives: {names(train_negative_ids)}")
    print(f"  eval-only negatives: {names(eval_extra_negative_ids)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run class split experiments using the sku_clf package.")
    parser.add_argument("--config", default="sku_classifier/config.yaml", help="Base package config path")
    parser.add_argument("--data", required=True, help="YOLO dataset zip or directory")
    parser.add_argument("--mode", choices=["explicit", "random"], default="explicit")
    parser.add_argument("--positive", nargs="*", default=[], help="Positive class names or ids for explicit mode")
    parser.add_argument("--negative", nargs="*", default=[], help="Negative class names or ids for explicit mode")
    parser.add_argument(
        "--negative-all-remaining",
        action="store_true",
        help="Use every non-positive class as a training negative",
    )
    parser.add_argument("--eval-unselected-as-negative", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pos-fraction", type=float, default=0.25)
    parser.add_argument("--neg-fraction", type=float, default=0.25)
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--pos-neg-ratio", type=str, help="Override positive:negative sampler ratio, e.g. 1:1 or 1:2")
    parser.add_argument("--device", help="Override device: cpu, cuda, or mps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from sku_clf.train import train_with_dataloaders
    from sku_clf.utils.validators import validate_config

    base_config = load_config(args.config)
    apply_training_overrides(base_config, args)

    positive_ids, train_negative_ids = choose_class_split(args, base_config["classes"])
    all_ids = [cls["id"] for cls in base_config["classes"]]
    eval_extra_negative_ids = [
        class_id
        for class_id in all_ids
        if class_id not in positive_ids and class_id not in train_negative_ids
    ]
    eval_negative_ids = list(train_negative_ids)
    if args.eval_unselected_as_negative:
        eval_negative_ids.extend(eval_extra_negative_ids)

    train_config = build_config(base_config, positive_ids, train_negative_ids)
    eval_config = build_config(base_config, positive_ids, eval_negative_ids)
    train_config["experiment"] = {
        "name": "class_split",
        "mode": args.mode,
        "positive_ids": list(positive_ids),
        "train_negative_ids": list(train_negative_ids),
        "eval_extra_negative_ids": list(eval_extra_negative_ids),
        "eval_unselected_as_negative": args.eval_unselected_as_negative,
    }

    validate_config(train_config)
    validate_config(eval_config)
    summarize_split(base_config["classes"], positive_ids, train_negative_ids, eval_extra_negative_ids)

    train_loader, val_loader, test_loader = build_dataloaders(train_config, eval_config, args.data)
    train_with_dataloaders(train_config, train_loader, val_loader, test_loader)


if __name__ == "__main__":
    main()
