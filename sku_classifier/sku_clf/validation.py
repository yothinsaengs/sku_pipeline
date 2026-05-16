from pathlib import Path
from typing import Any, Dict, Iterable, Sequence


def assert_dir(path: str, description: str) -> None:
    if not Path(path).is_dir():
        raise FileNotFoundError(f"{description} directory not found: {path}")


def assert_file(path: str, description: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{description} file not found: {path}")


def validate_yolo_box(box: Dict[str, Any], label_path: str, line_context: str = "") -> None:
    class_id = box.get("class_id")
    bbox = box.get("bbox")
    where = f"{label_path}{line_context}"
    if not isinstance(class_id, int) or class_id < 0:
        raise ValueError(f"Invalid class_id in {where}: {class_id}")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Invalid bbox in {where}: {bbox}")
    xc, yc, bw, bh = bbox
    values = [xc, yc, bw, bh]
    if any(not isinstance(value, (int, float)) for value in values):
        raise ValueError(f"Non-numeric YOLO bbox in {where}: {bbox}")
    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
        raise ValueError(f"YOLO center outside [0,1] in {where}: {bbox}")
    if not (0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
        raise ValueError(f"YOLO width/height must be in (0,1] in {where}: {bbox}")


def validate_records(records: Sequence[Dict[str, Any]]) -> None:
    if not records:
        raise ValueError("No labeled records found")
    seen = set()
    for record in records:
        stem = record.get("stem")
        if stem in seen:
            raise ValueError(f"Duplicate image stem found: {stem}")
        seen.add(stem)
        assert_file(record["image_path"], "Image")
        assert_file(record["label_path"], "YOLO label")
        if not record.get("boxes"):
            raise ValueError(f"Label file has no boxes: {record['label_path']}")
        for box in record["boxes"]:
            validate_yolo_box(box, record["label_path"])


def validate_split_map(split_map: Dict[str, Sequence[Dict[str, Any]]]) -> None:
    all_stems = {}
    for split_name in ("train", "val", "test"):
        records = split_map.get(split_name, [])
        if not records:
            raise ValueError(f"{split_name} split is empty; adjust split ratios or dataset size")
        for record in records:
            stem = record["stem"]
            if stem in all_stems:
                raise ValueError(f"Image split leakage: {stem} appears in {all_stems[stem]} and {split_name}")
            all_stems[stem] = split_name


def validate_samples(samples: Sequence[Dict[str, Any]], split_name: str) -> None:
    if not samples:
        raise ValueError(f"{split_name} samples are empty")
    for sample in samples:
        assert_file(sample["image_path"], "Sample image")
        validate_yolo_box({"class_id": max(0, int(sample.get("class_id", 0))), "bbox": sample["bbox"]}, sample["image_path"], f" sample {sample.get('stem', '')}")


def validate_training_samples(train_samples: Sequence[Dict[str, Any]], fraction: float) -> None:
    positives = sum(1 for sample in train_samples if sample.get("role") == "positive")
    negatives = sum(1 for sample in train_samples if sample.get("role") == "negative")
    if positives == 0:
        raise ValueError(f"Train fraction {fraction} has no positive samples")
    if negatives == 0:
        raise ValueError(f"Train fraction {fraction} has no negative samples")


def validate_rmon_annotation(annotation: Dict[str, Any], path: str, index: int) -> None:
    for key in ("type", "xmin", "ymin", "xmax", "ymax", "classname"):
        if key not in annotation:
            raise ValueError(f"Missing '{key}' in {path} annotation #{index}")
    if annotation["type"] != "rectangle":
        return
    coords = [annotation["xmin"], annotation["ymin"], annotation["xmax"], annotation["ymax"]]
    if any(not isinstance(value, (int, float)) for value in coords):
        raise ValueError(f"Non-numeric ROI coordinates in {path} annotation #{index}: {coords}")
    if annotation["xmax"] <= annotation["xmin"] or annotation["ymax"] <= annotation["ymin"]:
        raise ValueError(f"Invalid ROI box in {path} annotation #{index}: {coords}")


def validate_class_names(names: Iterable[str]) -> None:
    seen = set()
    for name in names:
        if name in seen:
            raise ValueError(f"Duplicate class name in classes.txt: {name}")
        seen.add(name)
