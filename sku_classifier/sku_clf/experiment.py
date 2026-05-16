import argparse
import copy
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
import yaml

from sku_clf.data.dataset import build_samples, discover_records, resolve_negative_ids, split_records
from sku_clf.train import train_fraction_run


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


def run_experiment(config: Dict[str, Any], data_path: str, output_dir: str) -> None:
    discovered = discover_records(data_path)
    records = discovered["records"]
    id_to_name = discovered["id_to_name"]
    negative_ids = resolve_negative_ids(config, id_to_name)
    positive_ids = [class_id for class_id in sorted(id_to_name) if class_id not in set(negative_ids)]
    if not positive_ids:
        raise ValueError("No positive classes remain after resolving negative_class_ids")

    positive_names = [id_to_name[class_id] for class_id in positive_ids]
    config = copy.deepcopy(config)
    config.setdefault("experiment", {})["negative_class_ids"] = negative_ids
    config["classes"] = [
        {"id": class_id, "name": id_to_name[class_id], "role": "positive"} for class_id in positive_ids
    ] + [
        {"id": class_id, "name": id_to_name[class_id], "role": "negative"} for class_id in negative_ids
    ]
    config.setdefault("model", {})["num_classes"] = len(positive_ids)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    split_cfg = config.get("dataset", {}).get("split", {"train": 0.7, "val": 0.15, "test": 0.15})
    seed = int(config.get("dataset", {}).get("split_seed", 42))
    split_map = split_records(records, split_cfg, seed)
    write_split_manifest(output_root / "split_manifest.csv", split_map)

    split_samples = {
        name: build_samples(records_for_split, positive_ids, negative_ids, config, name, seed + offset)
        for offset, (name, records_for_split) in enumerate(split_map.items())
    }
    pd.DataFrame([value["stats"] for value in split_samples.values()]).to_csv(output_root / "candidate_summary.csv", index=False)

    train_positive = split_samples["train"]["positives"]
    train_candidates = split_samples["train"]["negative_boxes"] + split_samples["train"]["background"]
    val_samples = split_samples["val"]["positives"] + split_samples["val"]["negative_boxes"] + split_samples["val"]["background"]
    test_samples = split_samples["test"]["positives"] + split_samples["test"]["negative_boxes"] + split_samples["test"]["background"]

    fractions = [float(value) for value in config.get("experiment", {}).get("negative_fractions", [0.1, 0.2, 0.5, 1.0])]
    comparison_rows = []
    for fraction in fractions:
        selected_negatives = sample_negatives(train_candidates, fraction, seed + int(fraction * 10000))
        train_samples = train_positive + selected_negatives
        run_dir = output_root / f"negative_fraction_{int(round(fraction * 100)):03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
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

    pd.DataFrame(comparison_rows).to_csv(output_root / "negative_fraction_comparison.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SKU negative-fraction classifier experiment.")
    parser.add_argument("--data", required=True, help="Merged YOLO dataset directory or zip")
    parser.add_argument("--config", default="sku_classifier/config.yaml", help="Experiment config YAML")
    parser.add_argument("--out", default="runs/negative_fraction_experiment", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(load_config(args.config), args.data, args.out)


if __name__ == "__main__":
    main()
