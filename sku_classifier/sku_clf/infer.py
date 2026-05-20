import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import yaml

from sku_clf.data.dataset import SKUExperimentDataset
from sku_clf.geometry import yolo_to_xyxy
from sku_clf.merge import detect_label_folder, image_size, load_roi_boxes
from sku_clf.models.backbone import get_model
from sku_clf.train import quantize_model_int8, resolve_device
from sku_clf.utils.io import extract_dataset, find_images
from sku_clf.validation import assert_dir


def load_inference_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        if Path(path).suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(f)
        return json.load(f)


def resolve_inference_paths(data_dir: str) -> tuple[str, str]:
    root = Path(data_dir)
    images_dir = root / "images"
    roi_dir = root / "roi"
    if images_dir.exists() and roi_dir.exists():
        return str(images_dir), str(roi_dir)

    found_images = None
    found_roi = None
    for current_root, dirs, _ in os.walk(data_dir):
        if "images" in dirs and found_images is None:
            found_images = Path(current_root) / "images"
        if "roi" in dirs and found_roi is None:
            found_roi = Path(current_root) / "roi"
        if found_images and found_roi:
            return str(found_images), str(found_roi)
    return str(images_dir), str(roi_dir)


def class_index_from_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "class_index" in config:
        return [
            {
                "index": int(item["index"]),
                "class_id": int(item.get("class_id", item["index"])),
                "class_name": str(item.get("class_name", f"class_{item['index']}")),
            }
            for item in config["class_index"]
        ]
    classes = [item for item in config.get("classes", []) if item.get("role") == "positive"]
    return [
        {"index": index, "class_id": int(item["id"]), "class_name": str(item.get("name", f"class_{item['id']}"))}
        for index, item in enumerate(classes)
    ]


def thresholds_from_config(config: Dict[str, Any], class_count: int, override: float | None) -> np.ndarray:
    if override is not None:
        return np.full(class_count, float(override), dtype=np.float32)
    inference_cfg = config.get("inference", {})
    thresholds = inference_cfg.get("thresholds", {})
    fallback = float(inference_cfg.get("best_confidence", config.get("threshold", 0.5)))
    return np.array([float(thresholds.get(str(index), fallback)) for index in range(class_count)], dtype=np.float32)


def scale_from_config(config: Dict[str, Any], override: str | None) -> int | str:
    if override:
        return override if override == "dynamic" else int(override)
    value = config.get("inference", {}).get("scale")
    if value is not None:
        return value if value == "dynamic" else int(value)
    sizes = [int(size) for size in config.get("input", {}).get("sizes", [56, 112, 224])]
    return max(sizes)


def discover_roi_samples(data_path: str, config: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[str]]:
    data_dir = extract_dataset(data_path)
    images_dir, roi_dir = resolve_inference_paths(data_dir)
    assert_dir(images_dir, "Images")
    assert_dir(roi_dir, "ROI")
    roi_format, roi_files = detect_label_folder(roi_dir, "ROI")
    images = find_images(images_dir)
    if not images:
        raise ValueError(f"No images found under {images_dir}")

    image_stems = {Path(path).stem for path in images}
    missing_roi = sorted(image_stems - set(roi_files))
    extra_roi = sorted(set(roi_files) - image_stems)
    if missing_roi:
        raise FileNotFoundError(f"Missing ROI files for image stems: {missing_roi[:10]}")
    if extra_roi:
        raise ValueError(f"ROI files without matching image stems: {extra_roi[:10]}")

    negative_cfg = config.get("negative_class", {})
    roi_class_name = negative_cfg.get("class_name", config.get("experiment", {}).get("negative_class_name", "GENERAL_SKU_SINGLE"))
    samples = []
    for image_path in images:
        stem = Path(image_path).stem
        width, height = image_size(image_path)
        roi_boxes, _, _ = load_roi_boxes(
            roi_format=roi_format,
            roi_path=roi_files[stem],
            width=width,
            height=height,
            roi_class_id=-1,
            roi_class_name=roi_class_name,
        )
        for roi_index, box in enumerate(roi_boxes):
            samples.append({
                "image_path": image_path,
                "stem": stem,
                "roi_index": roi_index,
                "bbox": box["bbox"],
                "image_dim": [width, height],
                "class_id": -1,
                "role": "negative",
                "source": "roi",
            })
    if not samples:
        raise ValueError(f"No ROI boxes found under {roi_dir}")
    return samples, images


def load_model(weights_path: str, config: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(weights_path, map_location="cpu")
    model_config = dict(config)
    model_config["model"] = dict(config.get("model", {}))
    model_config["model"]["pretrained"] = False
    model = get_model(model_config)
    if checkpoint.get("quantization") == "dynamic_int8":
        model = quantize_model_int8(model)
        device = torch.device("cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict(
    model: torch.nn.Module,
    dataset: SKUExperimentDataset,
    class_index: Sequence[Dict[str, Any]],
    thresholds: np.ndarray,
    scale: int | str,
    batch_size: int,
    device: torch.device,
) -> List[Dict[str, Any]]:
    rows = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch_indices = list(range(start, min(start + batch_size, len(dataset))))
            grouped: Dict[tuple[int, ...], List[tuple[int, Any]]] = {}
            for index in batch_indices:
                item = dataset.get_item_at_size(index, scale)
                grouped.setdefault(tuple(item[0].shape), []).append((index, item))
            for group in grouped.values():
                indices = [index for index, _ in group]
                inputs = torch.stack([item[0] for _, item in group], dim=0).to(device)
                outputs = model(inputs).cpu().numpy()
                for index, scores in zip(indices, outputs):
                    sample = dataset.samples[index]
                    selected_index = int(np.argmax(scores))
                    selected_confidence = float(scores[selected_index])
                    is_positive = bool(selected_confidence >= thresholds[selected_index])
                    selected_class = class_index[selected_index]
                    rows.append({
                        "image_path": sample["image_path"],
                        "stem": sample["stem"],
                        "roi_index": int(sample.get("roi_index", index)),
                        "bbox": [float(value) for value in sample["bbox"]],
                        "predicted_index": selected_index if is_positive else len(class_index),
                        "predicted_class_id": int(selected_class["class_id"]) if is_positive else -1,
                        "predicted_class_name": selected_class["class_name"] if is_positive else "NEGATIVE",
                        "confidence": selected_confidence,
                        "threshold": float(thresholds[selected_index]),
                        "is_positive": is_positive,
                        "scores": {
                            str(item["index"]): {
                                "class_id": int(item["class_id"]),
                                "class_name": item["class_name"],
                                "confidence": float(scores[int(item["index"])]),
                                "threshold": float(thresholds[int(item["index"])]),
                            }
                            for item in class_index
                        },
                    })
    return rows


def prediction_to_rmon_annotation(row: Dict[str, Any], width: int, height: int) -> Dict[str, Any]:
    x1, y1, x2, y2 = yolo_to_xyxy(row["bbox"], width, height)
    return {
        "type": "rectangle",
        "xmin": float(x1),
        "ymin": float(y1),
        "xmax": float(x2),
        "ymax": float(y2),
        "classname": row["predicted_class_name"],
        "confidence": float(row["confidence"]),
        "threshold": float(row["threshold"]),
        "class_id": int(row["predicted_class_id"]),
        "class_index": int(row["predicted_index"]),
        "roi_index": int(row["roi_index"]),
        "is_positive": bool(row["is_positive"]),
        "scores": row["scores"],
    }


def write_rmon_outputs(rows: Sequence[Dict[str, Any]], image_paths: Sequence[str], out_dir: str, include_negative: bool = False) -> None:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    rows_by_stem: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if include_negative or row["is_positive"]:
            rows_by_stem.setdefault(row["stem"], []).append(row)

    for image_path in image_paths:
        stem = Path(image_path).stem
        width, height = image_size(image_path)
        annotations = [
            prediction_to_rmon_annotation(row, width, height)
            for row in rows_by_stem.get(stem, [])
        ]
        with (path / f"{stem}.rmon").open("w") as f:
            json.dump({"image_dim": [width, height], "annotations": annotations}, f, indent=2)


def write_outputs(
    rows: Sequence[Dict[str, Any]],
    out_path: str,
    image_paths: Sequence[str],
    include_negative_rmon: bool = False,
) -> None:
    path = Path(out_path)
    if path.suffix.lower() == ".rmon":
        path.parent.mkdir(parents=True, exist_ok=True)
        if len(image_paths) != 1:
            raise ValueError("--out ending in .rmon is only valid when the input has exactly one image")
        write_rmon_outputs(rows, image_paths, str(path.parent), include_negative=include_negative_rmon)
        generated = path.parent / f"{Path(image_paths[0]).stem}.rmon"
        if generated != path:
            generated.replace(path)
        return
    if path.suffix.lower() not in {".json", ".csv"}:
        write_rmon_outputs(rows, image_paths, out_path, include_negative=include_negative_rmon)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        fields = [
            "image_path",
            "stem",
            "roi_index",
            "bbox",
            "predicted_index",
            "predicted_class_id",
            "predicted_class_name",
            "confidence",
            "threshold",
            "is_positive",
        ]
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(row[key]) if key == "bbox" else row[key] for key in fields})
        return
    with path.open("w") as f:
        json.dump(list(rows), f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SKU classifier inference on images/ plus roi/ without labels/.")
    parser.add_argument("--data", required=True, help="Dataset folder or zip containing images/ and roi/")
    parser.add_argument("--config", required=True, help="Inference JSON written by training, or a resolved train YAML/JSON with class metadata")
    parser.add_argument("--weights", required=True, help="Model checkpoint .pth")
    parser.add_argument("--out", default="predictions", help="Output directory for .rmon files, or a .json/.csv path")
    parser.add_argument("--device", default=None, help="Override device, e.g. cpu, cuda, mps")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None, help="Override all confidence thresholds")
    parser.add_argument("--scale", default=None, help="Override inference scale, e.g. 224 or dynamic")
    parser.add_argument("--include-negative-rmon", action="store_true", help="Include below-threshold ROI predictions as NEGATIVE annotations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_inference_config(args.config)
    class_index = class_index_from_config(config)
    if not class_index:
        raise ValueError("Inference config has no class_index/classes metadata")
    config.setdefault("model", {})["num_classes"] = len(class_index)
    samples, image_paths = discover_roi_samples(args.data, config)
    thresholds = thresholds_from_config(config, len(class_index), args.threshold)
    scale = scale_from_config(config, args.scale)
    if args.device:
        config.setdefault("training", {})["device"] = args.device
    device = resolve_device(config)
    model = load_model(args.weights, config, device)
    if next(model.parameters()).device.type == "cpu":
        device = torch.device("cpu")
    batch_size = int(args.batch_size or config.get("training", {}).get("batch_size", 32))
    dataset = SKUExperimentDataset(samples, [item["class_id"] for item in class_index], config, is_training=False)
    rows = predict(model, dataset, class_index, thresholds, scale, batch_size, device)
    write_outputs(rows, args.out, image_paths, include_negative_rmon=args.include_negative_rmon)
    print(f"Wrote {len(rows)} predictions to {args.out}", flush=True)


if __name__ == "__main__":
    main()
