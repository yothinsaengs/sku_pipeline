import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import pandas as pd

from sku_clf.geometry import iou_xyxy, xyxy_to_yolo, yolo_to_xyxy
from sku_clf.utils.io import find_images, read_class_names, read_yolo_labels, resolve_data_paths, write_class_names, write_yolo_labels
from sku_clf.validation import assert_dir, assert_file, validate_class_names, validate_rmon_annotation


def load_rmon_files(rmon_dir: str) -> Dict[str, Dict]:
    assert_dir(rmon_dir, "ROI .rmon")
    files = {}
    for path in Path(rmon_dir).glob("*.rmon"):
        with path.open("r") as f:
            data = json.load(f)
        if "annotations" not in data or not isinstance(data["annotations"], list):
            raise ValueError(f"Missing annotations list in {path}")
        if "image_dim" not in data:
            raise ValueError(f"Missing image_dim in {path}")
        for index, annotation in enumerate(data["annotations"]):
            validate_rmon_annotation(annotation, str(path), index)
        stem = data.get("filestem") or path.stem
        files[stem] = {"path": str(path), "data": data}
        files[path.stem] = {"path": str(path), "data": data}
    if not files:
        raise ValueError(f"No .rmon files found under {rmon_dir}")
    return files


def image_size(image_path: str, rmon_data: Dict) -> Tuple[int, int]:
    image = cv2.imread(image_path)
    if image is not None:
        h, w = image.shape[:2]
        return w, h
    dims = rmon_data.get("image_dim") if rmon_data else None
    if dims and len(dims) == 2:
        return int(dims[0]), int(dims[1])
    raise ValueError(f"Could not resolve dimensions for {image_path}")


def human_boxes_xyxy(human_boxes: List[Dict], width: int, height: int) -> List[Tuple[float, float, float, float]]:
    return [yolo_to_xyxy(box["bbox"], width, height) for box in human_boxes]


def kept_roi_boxes(
    rmon_data: Dict,
    human_xyxy: List[Tuple[float, float, float, float]],
    width: int,
    height: int,
    roi_class_id: int,
    roi_class_name: str,
    iou_threshold: float,
) -> Tuple[List[Dict], int, int]:
    kept = []
    total = 0
    removed = 0
    for ann in rmon_data.get("annotations", []) if rmon_data else []:
        if ann.get("type") != "rectangle" or ann.get("classname") != roi_class_name:
            continue
        total += 1
        xyxy = (float(ann["xmin"]), float(ann["ymin"]), float(ann["xmax"]), float(ann["ymax"]))
        if any(iou_xyxy(xyxy, human_box) >= iou_threshold for human_box in human_xyxy):
            removed += 1
            continue
        kept.append({"class_id": roi_class_id, "bbox": list(xyxy_to_yolo(xyxy, width, height))})
    return kept, total, removed


def merge_human_and_roi_dataset(
    human_dataset: str,
    roi_rmon_dir: str,
    out: str,
    roi_class_name: str = "GENERAL_SKU_SINGLE",
    iou_threshold: float = 0.3,
) -> None:
    human_root = Path(human_dataset)
    images_dir, labels_dir = resolve_data_paths(str(human_root))
    assert_dir(images_dir, "Human images")
    assert_dir(labels_dir, "Human labels")
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
    validate_class_names(class_names)
    if roi_class_name in class_names:
        raise ValueError(f"{roi_class_name} already exists in human classes.txt; refusing to append duplicate negative class")
    image_paths = find_images(images_dir)
    if not image_paths:
        raise ValueError(f"No human images found under {images_dir}")
    max_human_class = -1
    for image_path in image_paths:
        label_path = Path(labels_dir) / f"{Path(image_path).stem}.txt"
        assert_file(str(label_path), "Human YOLO label")
        for box in read_yolo_labels(str(label_path)):
            max_human_class = max(max_human_class, int(box["class_id"]))
    if not class_names:
        class_names = [f"class_{idx}" for idx in range(max_human_class + 1)]
    while len(class_names) <= max_human_class:
        class_names.append(f"class_{len(class_names)}")

    roi_class_id = len(class_names)
    class_names.append(roi_class_name)
    write_class_names(str(out_root / "classes.txt"), class_names)

    rmon_files = load_rmon_files(roi_rmon_dir)
    summary_rows = []
    for image_path in image_paths:
        stem = Path(image_path).stem
        if stem not in rmon_files:
            raise FileNotFoundError(f"Missing .rmon for image stem: {stem}")
        rmon = rmon_files[stem]
        rmon_data = rmon.get("data", {})
        width, height = image_size(image_path, rmon_data)
        rmon_dim = rmon_data.get("image_dim")
        if rmon_dim and [int(rmon_dim[0]), int(rmon_dim[1])] != [width, height]:
            raise ValueError(f"Dimension mismatch for {stem}: image is {[width, height]}, .rmon says {rmon_dim}")
        human_label_path = Path(labels_dir) / f"{stem}.txt"
        human_boxes = read_yolo_labels(str(human_label_path))
        human_xyxy = human_boxes_xyxy(human_boxes, width, height)
        roi_boxes, roi_total, roi_removed = kept_roi_boxes(
            rmon_data,
            human_xyxy,
            width,
            height,
            roi_class_id,
            roi_class_name,
            iou_threshold,
        )
        merged = human_boxes + roi_boxes
        shutil.copy2(image_path, out_images / Path(image_path).name)
        write_yolo_labels(str(out_labels / f"{stem}.txt"), merged)
        summary_rows.append({
            "stem": stem,
            "image_path": image_path,
            "rmon_path": rmon.get("path", ""),
            "human_boxes": len(human_boxes),
            "roi_boxes_total": roi_total,
            "roi_boxes_kept": len(roi_boxes),
            "roi_boxes_removed_iou": roi_removed,
            "merged_boxes": len(merged),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_root / "merge_summary.csv", index=False)
    (out_root / "merge_summary.json").write_text(summary.to_json(orient="records", indent=2))
