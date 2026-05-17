import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import pandas as pd

from sku_clf.geometry import iou_xyxy, xyxy_to_yolo, yolo_to_xyxy
from sku_clf.utils.io import find_images, read_class_names, read_yolo_labels, resolve_data_paths, write_class_names, write_yolo_labels
from sku_clf.validation import assert_dir, assert_file, validate_class_names, validate_rmon_annotation


SUPPORTED_LABEL_SUFFIXES = {".txt", ".rmon"}


def detect_label_folder(folder: str, description: str) -> Tuple[str, Dict[str, Path]]:
    assert_dir(folder, description)
    paths = [path for path in Path(folder).iterdir() if path.is_file() and path.suffix in SUPPORTED_LABEL_SUFFIXES]
    if not paths:
        raise ValueError(f"No supported label files found in {folder}; expected .txt or .rmon")
    suffixes = sorted({path.suffix for path in paths})
    if len(suffixes) != 1:
        raise ValueError(f"Mixed label formats in {folder}: {suffixes}; use only one suffix per folder")
    by_stem = {}
    duplicates = []
    for path in paths:
        if path.stem in by_stem:
            duplicates.append(path.stem)
        by_stem[path.stem] = path
    if duplicates:
        raise ValueError(f"Duplicate label stems in {folder}: {sorted(set(duplicates))[:10]}")
    return suffixes[0].lstrip("."), by_stem


def load_rmon(path: Path) -> Dict:
    with path.open("r") as f:
        data = json.load(f)
    if "annotations" not in data or not isinstance(data["annotations"], list):
        raise ValueError(f"Missing annotations list in {path}")
    if "image_dim" not in data:
        raise ValueError(f"Missing image_dim in {path}")
    for index, annotation in enumerate(data["annotations"]):
        validate_rmon_annotation(annotation, str(path), index)
    return data


def image_size(image_path: str, label_data: Dict | None = None) -> Tuple[int, int]:
    image = cv2.imread(image_path)
    if image is not None:
        h, w = image.shape[:2]
        return w, h
    dims = label_data.get("image_dim") if label_data else None
    if dims and len(dims) == 2:
        return int(dims[0]), int(dims[1])
    raise ValueError(f"Could not resolve dimensions for {image_path}")


def assert_rmon_dim_matches(stem: str, image_width: int, image_height: int, data: Dict, path: Path) -> None:
    dims = data.get("image_dim")
    if dims and [int(dims[0]), int(dims[1])] != [image_width, image_height]:
        raise ValueError(f"Dimension mismatch for {stem}: image is {[image_width, image_height]}, {path} says {dims}")


def rmon_annotations_to_boxes(
    data: Dict,
    width: int,
    height: int,
    class_to_id: Dict[str, int] | None = None,
    forced_class_id: int | None = None,
    required_classname: str | None = None,
) -> Tuple[List[Dict], int]:
    boxes = []
    total = 0
    for ann in data.get("annotations", []):
        if ann.get("type") != "rectangle":
            continue
        if required_classname is not None and ann.get("classname") != required_classname:
            continue
        total += 1
        if forced_class_id is None:
            classname = ann.get("classname")
            if classname not in class_to_id:
                raise ValueError(f"RMON classname '{classname}' not found in classes.txt")
            class_id = class_to_id[classname]
        else:
            class_id = forced_class_id
        xyxy = (float(ann["xmin"]), float(ann["ymin"]), float(ann["xmax"]), float(ann["ymax"]))
        boxes.append({"class_id": class_id, "bbox": list(xyxy_to_yolo(xyxy, width, height))})
    return boxes, total


def load_human_boxes(
    label_format: str,
    label_path: Path,
    width: int,
    height: int,
    class_to_id: Dict[str, int],
) -> Tuple[List[Dict], Dict | None]:
    if label_format == "txt":
        return read_yolo_labels(str(label_path)), None
    data = load_rmon(label_path)
    assert_rmon_dim_matches(label_path.stem, width, height, data, label_path)
    boxes, _ = rmon_annotations_to_boxes(data, width, height, class_to_id=class_to_id)
    return boxes, data


def load_roi_boxes(
    roi_format: str,
    roi_path: Path,
    width: int,
    height: int,
    roi_class_id: int,
    roi_class_name: str,
) -> Tuple[List[Dict], int, Dict | None]:
    if roi_format == "txt":
        raw_boxes = read_yolo_labels(str(roi_path))
        return [{"class_id": roi_class_id, "bbox": box["bbox"]} for box in raw_boxes], len(raw_boxes), None
    data = load_rmon(roi_path)
    assert_rmon_dim_matches(roi_path.stem, width, height, data, roi_path)
    boxes, total = rmon_annotations_to_boxes(
        data,
        width,
        height,
        forced_class_id=roi_class_id,
        required_classname=roi_class_name,
    )
    return boxes, total, data


def filter_roi_boxes(
    roi_boxes: List[Dict],
    human_boxes: List[Dict],
    width: int,
    height: int,
    iou_threshold: float,
) -> Tuple[List[Dict], int]:
    human_xyxy = [yolo_to_xyxy(box["bbox"], width, height) for box in human_boxes]
    kept = []
    removed = 0
    for box in roi_boxes:
        xyxy = yolo_to_xyxy(box["bbox"], width, height)
        if any(iou_xyxy(xyxy, human_box) >= iou_threshold for human_box in human_xyxy):
            removed += 1
            continue
        kept.append(box)
    return kept, removed


def merge_human_and_roi_dataset(
    human_dataset: str,
    roi_dir: str,
    out: str,
    roi_class_name: str = "GENERAL_SKU_SINGLE",
    iou_threshold: float = 0.3,
) -> None:
    human_root = Path(human_dataset)
    images_dir, labels_dir = resolve_data_paths(str(human_root))
    assert_dir(images_dir, "Human images")
    assert_dir(labels_dir, "Human labels")
    label_format, label_files = detect_label_folder(labels_dir, "Human labels")
    roi_format, roi_files = detect_label_folder(roi_dir, "ROI labels")

    out_root = Path(out)
    out_images = out_root / "images"
    out_labels = out_root / "labels"
    if out_images.exists():
        shutil.rmtree(out_images)
    if out_labels.exists():
        shutil.rmtree(out_labels)
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    class_names = read_class_names(str(human_root))
    if not class_names:
        raise ValueError(f"classes.txt is required for both YOLO and RMON human labels: {human_root}")
    validate_class_names(class_names)
    if roi_class_name in class_names:
        raise ValueError(f"{roi_class_name} already exists in human classes.txt; refusing to append duplicate negative class")
    class_to_id = {name: index for index, name in enumerate(class_names)}

    image_paths = find_images(images_dir)
    if not image_paths:
        raise ValueError(f"No human images found under {images_dir}")
    image_stems = [Path(path).stem for path in image_paths]
    missing_labels = sorted(set(image_stems) - set(label_files))
    missing_roi = sorted(set(image_stems) - set(roi_files))
    extra_labels = sorted(set(label_files) - set(image_stems))
    extra_roi = sorted(set(roi_files) - set(image_stems))
    if missing_labels:
        raise FileNotFoundError(f"Missing human label files for image stems: {missing_labels[:10]}")
    if missing_roi:
        raise FileNotFoundError(f"Missing ROI label files for image stems: {missing_roi[:10]}")
    if extra_labels:
        raise ValueError(f"Human label files without matching image stems: {extra_labels[:10]}")
    if extra_roi:
        raise ValueError(f"ROI label files without matching image stems: {extra_roi[:10]}")

    roi_class_id = len(class_names)
    class_names.append(roi_class_name)
    write_class_names(str(out_root / "classes.txt"), class_names)

    summary_rows = []
    for image_path in image_paths:
        stem = Path(image_path).stem
        width, height = image_size(image_path)
        human_label_path = label_files[stem]
        roi_label_path = roi_files[stem]
        human_boxes, _ = load_human_boxes(label_format, human_label_path, width, height, class_to_id)
        roi_boxes_raw, roi_total, _ = load_roi_boxes(roi_format, roi_label_path, width, height, roi_class_id, roi_class_name)
        roi_boxes, roi_removed = filter_roi_boxes(roi_boxes_raw, human_boxes, width, height, iou_threshold)
        merged = human_boxes + roi_boxes
        shutil.copy2(image_path, out_images / Path(image_path).name)
        write_yolo_labels(str(out_labels / f"{stem}.txt"), merged)
        summary_rows.append({
            "stem": stem,
            "image_path": image_path,
            "human_label_path": str(human_label_path),
            "roi_label_path": str(roi_label_path),
            "human_label_format": label_format,
            "roi_label_format": roi_format,
            "human_boxes": len(human_boxes),
            "roi_boxes_total": roi_total,
            "roi_boxes_kept": len(roi_boxes),
            "roi_boxes_removed_iou": roi_removed,
            "merged_boxes": len(merged),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_root / "merge_summary.csv", index=False)
    (out_root / "merge_summary.json").write_text(summary.to_json(orient="records", indent=2))
