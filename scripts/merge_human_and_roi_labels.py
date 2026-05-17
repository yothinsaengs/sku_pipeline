#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sku_classifier"))

from sku_clf.merge import merge_human_and_roi_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge human labels with ROI detector labels. labels/ and roi/ may be YOLO .txt or .rmon.")
    parser.add_argument("--human-dataset", required=True, help="Dataset with images/, labels/, classes.txt")
    parser.add_argument("--roi-dir", required=True, help="Directory containing one ROI .txt or .rmon file per image")
    parser.add_argument("--out", required=True, help="Output merged YOLO dataset directory")
    parser.add_argument("--roi-class-name", default="GENERAL_SKU_SINGLE")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_human_and_roi_dataset(
        human_dataset=args.human_dataset,
        roi_dir=args.roi_dir,
        out=args.out,
        roi_class_name=args.roi_class_name,
        iou_threshold=args.iou_threshold,
    )


if __name__ == "__main__":
    main()
