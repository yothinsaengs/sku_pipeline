#!/bin/bash
# scripts/infer.sh
#
# PURPOSE:
#   Runs inference on a single image using a trained model. Supports ROI extraction or direct crops.
#
# USAGE:
#   ./scripts/infer.sh <checkpoint_pth> <mode_specific_args>
#
# ARGUMENTS:
#   <checkpoint_pth>     : Path to the trained .pth model.
#   <mode_specific_args> : Use Mode A (Image+Boxes) or Mode B (Crops).
#
# EXAMPLES (Mode A - Full Image + YOLO Boxes):
#   ./scripts/infer.sh runs/.../best_model.pth --image img.jpg --boxes img.txt --save_vis out.jpg
#
# EXAMPLES (Mode B - Pre-cropped ROI):
#   ./scripts/infer.sh runs/.../best_model.pth --crops crop.jpg
#

set -e
cd "$(dirname "$0")/.."

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: ./scripts/infer.sh path/to/checkpoint.pth --image img.jpg --boxes img.txt"
    echo "   or: ./scripts/infer.sh path/to/checkpoint.pth --crops crop.jpg"
    exit 1
fi

CHECKPOINT=$1
shift

source venv/bin/activate
sku-clf infer --config sku_classifier/config.yaml --checkpoint "$CHECKPOINT" "$@"
