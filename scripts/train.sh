#!/bin/bash
# scripts/train.sh
#
# PURPOSE:
#   Starts the training process for the Stage 2 ROI Classifier.
#
# USAGE:
#   ./scripts/train.sh <path_to_dataset> [extra_cli_args]
#
# ARGUMENTS:
#   <path_to_dataset> : Path to your training data (ZIP or directory).
#   [extra_cli_args]  : (Optional) Override config values.
#
# EXAMPLES:
#   ./scripts/train.sh data/sku_dataset.zip
#   ./scripts/train.sh data/sku_dataset.zip --epochs 100 --lr 0.001 --batch_size 64 --pos-neg-ratio 1:1
#   ./scripts/train.sh data/sku_dataset.zip --device cuda --wandb
#

set -e
cd "$(dirname "$0")/.."

if [ -z "$1" ]; then
    echo "Usage: ./scripts/train.sh path/to/dataset.zip [extra_args]"
    exit 1
fi

DATA_PATH=$1
shift

source venv/bin/activate
echo "Starting training..."
sku-clf train --config sku_classifier/config.yaml --data "$DATA_PATH" "$@"
