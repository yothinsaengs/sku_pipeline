from typing import Sequence, Tuple


Box = Tuple[float, float, float, float]


def clamp_box_xyxy(box: Sequence[float], width: float, height: float) -> Box:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(width), float(x1)))
    y1 = max(0.0, min(float(height), float(y1)))
    x2 = max(0.0, min(float(width), float(x2)))
    y2 = max(0.0, min(float(height), float(y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def yolo_to_xyxy(bbox: Sequence[float], width: float = 1.0, height: float = 1.0) -> Box:
    xc, yc, bw, bh = [float(v) for v in bbox]
    x1 = (xc - bw / 2.0) * width
    y1 = (yc - bh / 2.0) * height
    x2 = (xc + bw / 2.0) * width
    y2 = (yc + bh / 2.0) * height
    return clamp_box_xyxy((x1, y1, x2, y2), width, height)


def xyxy_to_yolo(box: Sequence[float], width: float, height: float) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = clamp_box_xyxy(box, width, height)
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    return xc / width, yc / height, bw / width, bh / height


def iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def iou_yolo(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    return iou_xyxy(yolo_to_xyxy(box_a), yolo_to_xyxy(box_b))
