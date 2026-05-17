import argparse
import copy
import datetime
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
import yaml

from sku_clf.data.dataset import build_samples, discover_records, resolve_negative_ids, split_records
from sku_clf.logging_utils import append_log
from sku_clf.merge import merge_human_and_roi_dataset
from sku_clf.train import train_fraction_run
from sku_clf.utils.io import extract_dataset, resolve_data_paths
from sku_clf.validation import assert_dir, assert_file, validate_samples, validate_split_map, validate_training_samples


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def sample_negatives(candidates: Sequence[Dict[str, Any]], fraction: float, seed: int) -> List[Dict[str, Any]]:
    if not candidates or fraction <= 0:
        return []
    if fraction >= 1:
        return list(candidates)
    count = max(1, math.ceil(len(candidates) * fraction))
    return random.Random(seed).sample(list(candidates), min(count, len(candidates)))


def sample_to_row(sample: Dict[str, Any], split: str, index: int) -> Dict[str, Any]:
    xc, yc, bw, bh = sample["bbox"]
    return {
        "split": split,
        "index": index,
        "stem": sample.get("stem", ""),
        "image_path": sample["image_path"],
        "role": sample["role"],
        "source": sample.get("source", ""),
        "class_id": sample.get("class_id", -1),
        "x_center": xc,
        "y_center": yc,
        "width": bw,
        "height": bh,
    }


def write_manifest(path: Path, samples: Sequence[Dict[str, Any]], split: str) -> None:
    rows = [sample_to_row(sample, split, idx) for idx, sample in enumerate(samples)]
    pd.DataFrame(rows).to_csv(path, index=False)


def write_split_manifest(path: Path, split_map: Dict[str, List[Dict[str, Any]]]) -> None:
    rows = []
    for split_name, records in split_map.items():
        rows.extend({
            "split": split_name,
            "stem": record["stem"],
            "image_path": record["image_path"],
            "label_path": record["label_path"],
            "box_count": len(record["boxes"]),
        } for record in records)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_dataset_summary(
    path: Path,
    split_map: Dict[str, List[Dict[str, Any]]],
    split_samples: Dict[str, Dict[str, Any]],
    positive_ids: Sequence[int],
    negative_ids: Sequence[int],
    id_to_name: Dict[int, str],
) -> None:
    rows = []
    for split_name, records in split_map.items():
        class_box_counts = {}
        for record in records:
            for box in record["boxes"]:
                class_box_counts[box["class_id"]] = class_box_counts.get(box["class_id"], 0) + 1
        sample_info = split_samples[split_name]
        rows.append({
            "split": split_name,
            "image_count": len(records),
            "positive_class_count": len(positive_ids),
            "negative_class_count": len(negative_ids),
            "positive_box_count": sample_info["stats"]["positive_boxes"],
            "negative_box_total": sample_info["stats"]["negative_boxes_total"],
            "negative_box_kept": sample_info["stats"]["negative_boxes_kept"],
            "background_candidate_count": sample_info["stats"]["background_candidates"],
            "positive_classes": "|".join(f"{class_id}:{id_to_name[class_id]}" for class_id in positive_ids),
            "negative_classes": "|".join(f"{class_id}:{id_to_name[class_id]}" for class_id in negative_ids),
            "box_counts_by_class": "|".join(
                f"{class_id}:{id_to_name.get(class_id, f'class_{class_id}')}={count}"
                for class_id, count in sorted(class_box_counts.items())
            ),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def validate_raw_input_dataset(data_path: str, extract_to: Path) -> str:
    data_dir = extract_dataset(data_path, extract_to=str(extract_to))
    images_dir, labels_dir = resolve_data_paths(data_dir)
    rmon_dir = Path(data_dir) / "rmon"
    classes_path = Path(data_dir) / "classes.txt"
    assert_dir(images_dir, "Input images")
    assert_dir(labels_dir, "Input labels")
    assert_dir(str(rmon_dir), "Input rmon")
    assert_file(str(classes_path), "Input classes.txt")
    return data_dir


def run_experiment(config: Dict[str, Any], data_path: str, output_dir: str) -> None:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(output_dir) / timestamp
    output_root.mkdir(parents=True, exist_ok=True)
    append_log(output_root, f"Run started: {timestamp}")
    append_log(output_root, f"Input data: {data_path}")
    append_log(output_root, f"Output root: {output_root}")
    raw_data_dir = validate_raw_input_dataset(data_path, output_root / "input_dataset")
    append_log(output_root, f"Validated raw dataset: {raw_data_dir}")
    merged_data_dir = output_root / "merged_dataset"
    append_log(output_root, f"Merging YOLO labels and rmon files into: {merged_data_dir}")
    merge_human_and_roi_dataset(
        human_dataset=raw_data_dir,
        roi_rmon_dir=str(Path(raw_data_dir) / "rmon"),
        out=str(merged_data_dir),
        roi_class_name=config.get("experiment", {}).get("negative_class_name", "GENERAL_SKU_SINGLE"),
        iou_threshold=float(config.get("experiment", {}).get("negative_iou_threshold", 0.3)),
    )
    append_log(output_root, "Merge complete")

    discovered = discover_records(str(merged_data_dir))
    records = discovered["records"]
    id_to_name = discovered["id_to_name"]
    negative_ids = resolve_negative_ids(config, id_to_name)
    positive_ids = [class_id for class_id in sorted(id_to_name) if class_id not in set(negative_ids)]
    if not positive_ids:
        raise ValueError("No positive classes remain after resolving negative_class_ids")
    append_log(output_root, f"Positive classes: {len(positive_ids)}")
    append_log(output_root, f"Negative class ids: {negative_ids}")

    positive_names = [id_to_name[class_id] for class_id in positive_ids]
    config = copy.deepcopy(config)
    config.setdefault("experiment", {})["negative_class_ids"] = negative_ids
    config["classes"] = [
        {"id": class_id, "name": id_to_name[class_id], "role": "positive"} for class_id in positive_ids
    ] + [
        {"id": class_id, "name": id_to_name[class_id], "role": "negative"} for class_id in negative_ids
    ]
    config.setdefault("model", {})["num_classes"] = len(positive_ids)

    split_cfg = config.get("dataset", {}).get("split", {"train": 0.7, "val": 0.15, "test": 0.15})
    seed = int(config.get("dataset", {}).get("split_seed", 42))
    split_map = split_records(records, split_cfg, seed)
    validate_split_map(split_map)
    write_split_manifest(output_root / "split_manifest.csv", split_map)
    append_log(output_root, f"Image split counts: train={len(split_map['train'])}, val={len(split_map['val'])}, test={len(split_map['test'])}")

    split_samples = {
        name: build_samples(records_for_split, positive_ids, negative_ids, config, name, seed + offset)
        for offset, (name, records_for_split) in enumerate(split_map.items())
    }
    pd.DataFrame([value["stats"] for value in split_samples.values()]).to_csv(output_root / "candidate_summary.csv", index=False)
    write_dataset_summary(output_root / "dataset_summary.csv", split_map, split_samples, positive_ids, negative_ids, id_to_name)
    append_log(output_root, "Wrote candidate_summary.csv and dataset_summary.csv")

    train_positive = split_samples["train"]["positives"]
    train_candidates = split_samples["train"]["negative_boxes"] + split_samples["train"]["background"]
    val_samples = split_samples["val"]["positives"] + split_samples["val"]["negative_boxes"] + split_samples["val"]["background"]
    test_samples = split_samples["test"]["positives"] + split_samples["test"]["negative_boxes"] + split_samples["test"]["background"]
    validate_samples(val_samples, "val")
    validate_samples(test_samples, "test")

    fractions = [float(value) for value in config.get("experiment", {}).get("negative_fractions", [0.1, 0.2, 0.5, 1.0])]
    comparison_rows = []
    for fraction in fractions:
        selected_negatives = sample_negatives(train_candidates, fraction, seed + int(fraction * 10000))
        train_samples = train_positive + selected_negatives
        validate_training_samples(train_samples, fraction)
        validate_samples(train_samples, f"train fraction {fraction}")
        run_dir = output_root / f"negative_fraction_{int(round(fraction * 100)):03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        append_log(output_root, f"Starting fraction {fraction}: selected_train_negatives={len(selected_negatives)}")
        write_manifest(run_dir / "negative_pool_manifest.csv", selected_negatives, "train")
        write_manifest(run_dir / "train_manifest.csv", train_samples, "train")
        summary = train_fraction_run(
            config=config,
            run_dir=run_dir,
            positive_ids=positive_ids,
            positive_names=positive_names,
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            fraction=fraction,
        )
        for row in summary["final_test_rows"]:
            comparison_rows.append({"run_dir": str(run_dir), **row})
        append_log(output_root, f"Finished fraction {fraction}: best_val_f1_macro={summary['best_val_f1_macro']}")

    pd.DataFrame(comparison_rows).to_csv(output_root / "negative_fraction_comparison.csv", index=False)
    append_log(output_root, "Wrote negative_fraction_comparison.csv")
    append_log(output_root, "Run complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge raw YOLO+rmon data, then run the SKU negative-fraction classifier experiment.")
    parser.add_argument("--data", required=True, help="Raw dataset folder or zip containing images/, labels/, rmon/, classes.txt")
    parser.add_argument("--config", default="sku_classifier/config.yaml", help="Experiment config YAML")
    parser.add_argument("--out", default="runs/negative_fraction_experiment", help="Base output directory; a timestamped child folder is created automatically")
    parser.add_argument("--epochs", type=int, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--device", help="Override device, e.g. auto, cpu, cuda, mps")
    parser.add_argument("--num-workers", type=int, help="Override DataLoader workers")
    parser.add_argument("--backbone", help="Override timm backbone")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, help="Override pretrained backbone flag")
    parser.add_argument("--negative-fractions", nargs="+", type=float, help="Override negative fractions, e.g. 0.1 0.2 0.5 1.0")
    parser.add_argument("--scales", nargs="+", help="Override input scales, e.g. --scales 56 or --scales 56,112 or --scales 112 224")
    parser.add_argument("--per-class-interval", type=int, help="Override per-class metrics save interval")
    parser.add_argument("--macro-min-support", type=int, help="Ignore classes with lower support for main macro metrics")
    parser.add_argument("--class-pos-weight", action=argparse.BooleanOptionalAction, help="Enable/disable positive class loss weighting")
    parser.add_argument("--tune-thresholds", action=argparse.BooleanOptionalAction, help="Enable/disable validation threshold tuning")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args)
    run_experiment(config, args.data, args.out)


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> None:
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.device is not None:
        config.setdefault("training", {})["device"] = args.device
    if args.num_workers is not None:
        config.setdefault("training", {})["num_workers"] = args.num_workers
    if args.backbone is not None:
        config.setdefault("model", {})["backbone"] = args.backbone
    if args.pretrained is not None:
        config.setdefault("model", {})["pretrained"] = args.pretrained
    if args.negative_fractions is not None:
        config.setdefault("experiment", {})["negative_fractions"] = args.negative_fractions
    if args.scales is not None:
        config.setdefault("input", {})["sizes"] = parse_scales(args.scales)
    if args.per_class_interval is not None:
        config.setdefault("logging", {})["per_class_interval"] = args.per_class_interval
    if args.macro_min_support is not None:
        config.setdefault("metrics", {})["macro_min_support"] = args.macro_min_support
    if args.class_pos_weight is not None:
        config.setdefault("loss", {}).setdefault("class_pos_weight", {})["enabled"] = args.class_pos_weight
    if args.tune_thresholds is not None:
        config.setdefault("metrics", {}).setdefault("tune_thresholds", {})["enabled"] = args.tune_thresholds


def parse_scales(values: Sequence[str]) -> List[int]:
    scales = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            scale = int(part)
            if scale <= 0:
                raise ValueError(f"Scale must be positive, got {scale}")
            scales.append(scale)
    scales = list(dict.fromkeys(scales))
    if not scales:
        raise ValueError("--scales did not contain any valid scale")
    return scales


if __name__ == "__main__":
    main()
