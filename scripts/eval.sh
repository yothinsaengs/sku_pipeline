#!/bin/bash
# scripts/eval.sh
#
# PURPOSE:
#   Evaluates a trained model checkpoint on a specific dataset and generates a metrics report.
#
# USAGE:
#   ./scripts/eval.sh <checkpoint_pth> <path_to_dataset>
#
# ARGUMENTS:
#   <checkpoint_pth>  : Path to the .pth file (usually in runs/<folder>/checkpoints/best_model.pth).
#   <path_to_dataset> : Path to the evaluation data (ZIP or directory).
#
# EXAMPLES:
#   ./scripts/eval.sh runs/20240101_120000/checkpoints/best_model.pth data/test_split.zip
#

set -e
cd "$(dirname "$0")/.."

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: ./scripts/eval.sh path/to/checkpoint.pth path/to/dataset.zip"
    exit 1
fi

CHECKPOINT=$1
DATA=$2

source venv/bin/activate
sku-clf eval --config sku_classifier/config.yaml --checkpoint "$CHECKPOINT" --data "$DATA"
