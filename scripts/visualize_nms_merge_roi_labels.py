"""
Hardcoded visual QA script for merging human labels with ROI boxes.

Edit the constants below, then run this file directly. It expects the same
dataset shape as training:

  DATASET_DIR/
    images/
    labels/
    roi/
    classes.txt

Output:
  OUTPUT_DIR/
    images with red positive boxes and gray negative ROI boxes
    merged_labels/ YOLO txt files after ROI suppression
    merge_summary.csv
"""

import csv
import shutil
from pathlib import Path

import cv2

from sku_clf.geometry import iou_xyxy, yolo_to_xyxy
from sku_clf.merge import (
    detect_label_folder,
    image_size,
    load_human_boxes,
    load_roi_boxes,
)
from sku_clf.utils.io import find_images, read_class_names, resolve_data_paths, write_yolo_labels
from sku_clf.validation import assert_dir, validate_class_names


# =========================
# Hardcoded Config
# =========================

DATASET_DIR = Path("/Users/yothinsaengsakoon/Downloads/project-1361-at-2026-05-19-07-19-311cbc2b/")
ROI_DIR = DATASET_DIR / "roi"
OUTPUT_DIR = Path("outputs/nms_merge_visualization")

ROI_CLASS_NAME = "GENERAL_SKU_SINGLE"
IOU_THRESHOLD = 0.30

IMAGE_EXTENSIONS_TO_COPY = True
DRAW_LINE_THICKNESS = 2
POSITIVE_COLOR_BGR = (0, 0, 255)
NEGATIVE_COLOR_BGR = (145, 145, 145)


# =========================
# Helpers
# =========================

def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_class_map(dataset_dir: Path) -> tuple[list[str], dict[str, int], int]:
    class_names = read_class_names(str(dataset_dir))
    if not class_names:
        raise ValueError(f"classes.txt or obj.names is required under {dataset_dir}")
    validate_class_names(class_names)
    if ROI_CLASS_NAME in class_names:
        raise ValueError(f"{ROI_CLASS_NAME} already exists in classes.txt; choose a negative class name not used by positives")
    class_to_id = {name: index for index, name in enumerate(class_names)}
    roi_class_id = len(class_names)
    return class_names, class_to_id, roi_class_id


def suppress_roi_boxes(roi_boxes: list[dict], positive_boxes: list[dict], image_width: int, image_height: int) -> tuple[list[dict], int]:
    positive_xyxy = [yolo_to_xyxy(box["bbox"], image_width, image_height) for box in positive_boxes]
    kept = []
    removed = 0
    for roi_box in roi_boxes:
        roi_xyxy = yolo_to_xyxy(roi_box["bbox"], image_width, image_height)
        if any(iou_xyxy(roi_xyxy, positive_xyxy_box) >= IOU_THRESHOLD for positive_xyxy_box in positive_xyxy):
            removed += 1
            continue
        kept.append(roi_box)
    return kept, removed


def draw_box(image, bbox: list[float], image_width: int, image_height: int, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = yolo_to_xyxy(bbox, image_width, image_height)
    pt1 = (max(0, int(round(x1))), max(0, int(round(y1))))
    pt2 = (min(image_width - 1, int(round(x2))), min(image_height - 1, int(round(y2))))
    cv2.rectangle(image, pt1, pt2, color, DRAW_LINE_THICKNESS)


def visualize_image(
    image_path: str,
    out_path: Path,
    positive_boxes: list[dict],
    negative_boxes: list[dict],
    image_width: int,
    image_height: int,
) -> None:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)
    for box in negative_boxes:
        draw_box(image, box["bbox"], image_width, image_height, NEGATIVE_COLOR_BGR)
    for box in positive_boxes:
        draw_box(image, box["bbox"], image_width, image_height, POSITIVE_COLOR_BGR)
    cv2.imwrite(str(out_path), image)


def assert_matching_stems(image_paths: list[str], label_files: dict[str, Path], roi_files: dict[str, Path]) -> None:
    image_stems = {Path(path).stem for path in image_paths}
    missing_labels = sorted(image_stems - set(label_files))
    missing_roi = sorted(image_stems - set(roi_files))
    extra_labels = sorted(set(label_files) - image_stems)
    extra_roi = sorted(set(roi_files) - image_stems)
    if missing_labels:
        raise FileNotFoundError(f"Missing label files for image stems: {missing_labels[:10]}")
    if missing_roi:
        raise FileNotFoundError(f"Missing ROI files for image stems: {missing_roi[:10]}")
    if extra_labels:
        raise ValueError(f"Label files without matching image stems: {extra_labels[:10]}")
    if extra_roi:
        raise ValueError(f"ROI files without matching image stems: {extra_roi[:10]}")


def write_summary(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# =========================
# Main Script
# =========================

def main() -> None:
    images_dir, labels_dir = resolve_data_paths(str(DATASET_DIR))
    assert_dir(images_dir, "Images")
    assert_dir(labels_dir, "Labels")
    assert_dir(str(ROI_DIR), "ROI")

    class_names, class_to_id, roi_class_id = load_class_map(DATASET_DIR)
    label_format, label_files = detect_label_folder(labels_dir, "Labels")
    roi_format, roi_files = detect_label_folder(str(ROI_DIR), "ROI")
    image_paths = find_images(images_dir)
    if not image_paths:
        raise ValueError(f"No images found under {images_dir}")
    assert_matching_stems(image_paths, label_files, roi_files)

    ensure_clean_dir(OUTPUT_DIR)
    visual_dir = OUTPUT_DIR / "visualized"
    merged_label_dir = OUTPUT_DIR / "merged_labels"
    visual_dir.mkdir(parents=True, exist_ok=True)
    merged_label_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for image_path in image_paths:
        stem = Path(image_path).stem
        image_width, image_height = image_size(image_path)

        positive_boxes, _ = load_human_boxes(
            label_format=label_format,
            label_path=label_files[stem],
            width=image_width,
            height=image_height,
            class_to_id=class_to_id,
        )
        roi_boxes, roi_total, _ = load_roi_boxes(
            roi_format=roi_format,
            roi_path=roi_files[stem],
            width=image_width,
            height=image_height,
            roi_class_id=roi_class_id,
            roi_class_name=ROI_CLASS_NAME,
        )
        kept_roi_boxes, removed_roi_boxes = suppress_roi_boxes(
            roi_boxes=roi_boxes,
            positive_boxes=positive_boxes,
            image_width=image_width,
            image_height=image_height,
        )
        merged_boxes = positive_boxes + kept_roi_boxes
        write_yolo_labels(str(merged_label_dir / f"{stem}.txt"), merged_boxes)

        out_image_path = visual_dir / Path(image_path).name
        visualize_image(
            image_path=image_path,
            out_path=out_image_path,
            positive_boxes=positive_boxes,
            negative_boxes=kept_roi_boxes,
            image_width=image_width,
            image_height=image_height,
        )

        summary_rows.append({
            "stem": stem,
            "image_path": image_path,
            "label_path": str(label_files[stem]),
            "roi_path": str(roi_files[stem]),
            "label_format": label_format,
            "roi_format": roi_format,
            "positive_boxes": len(positive_boxes),
            "roi_boxes_total": roi_total,
            "roi_boxes_kept": len(kept_roi_boxes),
            "roi_boxes_removed_by_iou": removed_roi_boxes,
            "merged_boxes": len(merged_boxes),
            "visualized_path": str(out_image_path),
        })

    write_summary(OUTPUT_DIR / "merge_summary.csv", summary_rows)
    print(f"Done. Wrote {len(summary_rows)} visualizations to {visual_dir}")
    print(f"Positive boxes: red | Negative ROI boxes kept after IoU suppression: gray")
    print(f"Known positive classes: {len(class_names)} | Negative class id in merged labels: {roi_class_id}")


if __name__ == "__main__":
    main()
