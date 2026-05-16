import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG")


def extract_dataset(data_path: str, extract_to: str = "temp_dataset") -> str:
    if not zipfile.is_zipfile(data_path):
        return data_path
    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)
    with zipfile.ZipFile(data_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    return extract_to


def resolve_data_paths(data_dir: str) -> Tuple[str, str]:
    root = Path(data_dir)
    images_dir = root / "images"
    labels_dir = root / "labels"
    if images_dir.exists() and labels_dir.exists():
        return str(images_dir), str(labels_dir)

    found_images = None
    found_labels = None
    for current_root, dirs, _ in os.walk(data_dir):
        if "images" in dirs and found_images is None:
            found_images = Path(current_root) / "images"
        if "labels" in dirs and found_labels is None:
            found_labels = Path(current_root) / "labels"
        if found_images and found_labels:
            return str(found_images), str(found_labels)
    return str(images_dir), str(labels_dir)


def read_class_names(data_dir: str) -> List[str]:
    for name in ("classes.txt", "obj.names"):
        path = Path(data_dir) / name
        if path.exists():
            return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return []


def write_class_names(path: str, names: Iterable[str]) -> None:
    Path(path).write_text("\n".join(names) + "\n")


def find_images(images_dir: str) -> List[str]:
    return sorted(str(path) for path in Path(images_dir).iterdir() if path.suffix in IMAGE_EXTENSIONS)


def read_yolo_labels(label_path: str) -> List[Dict]:
    path = Path(label_path)
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        boxes.append({
            "class_id": int(parts[0]),
            "bbox": [float(value) for value in parts[1:]],
        })
    return boxes


def write_yolo_labels(label_path: str, boxes: Iterable[Dict]) -> None:
    lines = []
    for box in boxes:
        xc, yc, bw, bh = box["bbox"]
        lines.append(f"{int(box['class_id'])} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")
    Path(label_path).write_text("\n".join(lines) + ("\n" if lines else ""))
