import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from sku_clf.geometry import iou_yolo, yolo_to_xyxy
from sku_clf.utils.io import extract_dataset, find_images, read_class_names, read_yolo_labels, resolve_data_paths
from sku_clf.validation import assert_dir, assert_file, validate_records


def discover_records(data_path: str) -> Dict[str, Any]:
    data_dir = extract_dataset(data_path)
    images_dir, labels_dir = resolve_data_paths(data_dir)
    assert_dir(images_dir, "Images")
    assert_dir(labels_dir, "Labels")
    images = find_images(images_dir)
    if not images:
        raise ValueError(f"No images found under {images_dir}")
    records = []
    class_ids = set()
    for image_path in images:
        stem = Path(image_path).stem
        label_path = str(Path(labels_dir) / f"{stem}.txt")
        assert_file(label_path, "YOLO label")
        boxes = read_yolo_labels(label_path)
        if not boxes:
            continue
        class_ids.update(box["class_id"] for box in boxes)
        records.append({"image_path": image_path, "label_path": label_path, "stem": stem, "boxes": boxes})
    if not records:
        raise ValueError(f"No labeled YOLO images found under {data_dir}")

    names = read_class_names(data_dir)
    id_to_name = {class_id: names[class_id] if class_id < len(names) else f"class_{class_id}" for class_id in sorted(class_ids)}
    validate_records(records)
    return {"data_dir": data_dir, "records": records, "id_to_name": id_to_name}


def split_records(records: Sequence[Dict[str, Any]], split: Dict[str, float], seed: int) -> Dict[str, List[Dict[str, Any]]]:
    total = split["train"] + split["val"] + split["test"]
    if abs(total - 1.0) > 1e-6:
        raise ValueError("dataset.split train + val + test must equal 1.0")
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    train_count = int(len(indices) * split["train"])
    val_count = int(len(indices) * split["val"])
    return {
        "train": [records[i] for i in indices[:train_count]],
        "val": [records[i] for i in indices[train_count:train_count + val_count]],
        "test": [records[i] for i in indices[train_count + val_count:]],
    }


def resolve_negative_ids(config: Dict[str, Any], id_to_name: Dict[int, str]) -> List[int]:
    configured = config.get("experiment", {}).get("negative_class_ids", [])
    if configured:
        return sorted({int(value) for value in configured})
    for class_id, name in id_to_name.items():
        if name == config.get("experiment", {}).get("negative_class_name", "GENERAL_SKU_SINGLE"):
            return [class_id]
    raise ValueError("No negative_class_ids configured and GENERAL_SKU_SINGLE was not found in classes")


def build_samples(
    records: Sequence[Dict[str, Any]],
    positive_ids: Sequence[int],
    negative_ids: Sequence[int],
    config: Dict[str, Any],
    split_name: str,
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    positive_set = set(positive_ids)
    negative_set = set(negative_ids)
    overlap_iou = config.get("experiment", {}).get("negative_iou_threshold", 0.3)
    positives = []
    negative_boxes = []
    stats = {
        "split": split_name,
        "images": len(records),
        "positive_boxes": 0,
        "negative_boxes_total": 0,
        "negative_boxes_kept": 0,
        "background_candidates": 0,
    }

    for record in records:
        pos_boxes = [box for box in record["boxes"] if box["class_id"] in positive_set]
        stats["positive_boxes"] += len(pos_boxes)
        for box in pos_boxes:
            positives.append({
                "image_path": record["image_path"],
                "stem": record["stem"],
                "bbox": box["bbox"],
                "class_id": box["class_id"],
                "role": "positive",
                "source": "human_box",
            })

        for box in record["boxes"]:
            if box["class_id"] not in negative_set:
                continue
            stats["negative_boxes_total"] += 1
            if any(iou_yolo(box["bbox"], pos["bbox"]) >= overlap_iou for pos in pos_boxes):
                continue
            stats["negative_boxes_kept"] += 1
            negative_boxes.append({
                "image_path": record["image_path"],
                "stem": record["stem"],
                "bbox": box["bbox"],
                "class_id": box["class_id"],
                "role": "negative",
                "source": "negative_box",
            })

    background = generate_background_candidates(records, positive_set, len(negative_boxes), overlap_iou, rng)
    stats["background_candidates"] = len(background)
    return {"positives": positives, "negative_boxes": negative_boxes, "background": background, "stats": stats}


def generate_background_candidates(
    records: Sequence[Dict[str, Any]],
    positive_ids: set,
    target_count: int,
    overlap_iou: float,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if target_count <= 0 or not records:
        return []
    candidates = []
    attempts = 0
    max_attempts = max(100, target_count * 50)
    while len(candidates) < target_count and attempts < max_attempts:
        attempts += 1
        record = rng.choice(list(records))
        pos_boxes = [box["bbox"] for box in record["boxes"] if box["class_id"] in positive_ids]
        scale = rng.uniform(0.12, 0.45)
        aspect = rng.uniform(0.6, 1.6)
        bw = min(0.95, scale * aspect)
        bh = min(0.95, scale / aspect)
        if bw <= 0 or bh <= 0 or bw >= 1 or bh >= 1:
            continue
        xc = rng.uniform(bw / 2.0, 1.0 - bw / 2.0)
        yc = rng.uniform(bh / 2.0, 1.0 - bh / 2.0)
        bbox = [xc, yc, bw, bh]
        if any(iou_yolo(bbox, pos_box) >= overlap_iou for pos_box in pos_boxes):
            continue
        candidates.append({
            "image_path": record["image_path"],
            "stem": record["stem"],
            "bbox": bbox,
            "class_id": -1,
            "role": "negative",
            "source": "background",
        })
    return candidates


class SKUExperimentDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Dict[str, Any]],
        positive_ids: Sequence[int],
        config: Dict[str, Any],
        is_training: bool,
        fixed_size: Optional[int] = None,
    ):
        self.samples = list(samples)
        self.positive_ids = list(positive_ids)
        self.pos_id_to_idx = {class_id: idx for idx, class_id in enumerate(self.positive_ids)}
        self.config = config
        self.is_training = is_training
        self.fixed_size = fixed_size
        input_cfg = config.get("input", {})
        self.sizes = [int(size) for size in input_cfg.get("sizes", [56, 112, 224])]
        self.pad_color = int(input_cfg.get("pad_color", 114))
        self.context_margin = float(input_cfg.get("context_margin", 0.0))
        self.min_size = int(input_cfg.get("min_size", 4))
        self.label_smoothing = float(config.get("label_smoothing", 0.0))

    def __len__(self) -> int:
        return len(self.samples)

    def set_fixed_size(self, size: Optional[int]) -> None:
        self.fixed_size = size

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        size = self.fixed_size or (random.choice(self.sizes) if self.is_training else max(self.sizes))
        image = cv2.imread(sample["image_path"])
        if image is None:
            raise FileNotFoundError(sample["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        crop = self._crop(image, sample["bbox"])
        crop = self._letterbox(crop, size)
        tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        target, hard_target = self._targets(sample)
        return tensor, torch.from_numpy(target), torch.from_numpy(hard_target)

    def _targets(self, sample: Dict[str, Any]):
        num_classes = len(self.positive_ids)
        smooth_floor = self.label_smoothing / num_classes if self.label_smoothing > 0 and num_classes else 0.0
        target = np.full(num_classes, smooth_floor, dtype=np.float32)
        hard_target = np.zeros(num_classes, dtype=np.float32)
        if sample["role"] == "positive":
            idx = self.pos_id_to_idx[sample["class_id"]]
            target[idx] = 1.0 - self.label_smoothing
            hard_target[idx] = 1.0
        return target, hard_target

    def _crop(self, image: np.ndarray, bbox: Sequence[float]) -> np.ndarray:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = yolo_to_xyxy(bbox, w, h)
        if self.context_margin:
            bw, bh = x2 - x1, y2 - y1
            x1 -= bw * self.context_margin / 2.0
            x2 += bw * self.context_margin / 2.0
            y1 -= bh * self.context_margin / 2.0
            y2 += bh * self.context_margin / 2.0
        x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        x2, y2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
        if x2 - x1 < self.min_size or y2 - y1 < self.min_size:
            return np.zeros((self.min_size, self.min_size, 3), dtype=np.uint8)
        return image[y1:y2, x1:x2]

    def _letterbox(self, image: np.ndarray, size: int) -> np.ndarray:
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return np.full((size, size, 3), self.pad_color, dtype=np.uint8)
        ratio = min(size / h, size / w)
        new_w = max(1, int(round(w * ratio)))
        new_h = max(1, int(round(h * ratio)))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), self.pad_color, dtype=np.uint8)
        x = (size - new_w) // 2
        y = (size - new_h) // 2
        canvas[y:y + new_h, x:x + new_w] = resized
        return canvas
