#!/bin/bash
# scripts/general_sku_negative_experiment.sh
#
# PURPOSE:
#   Runs the experiment where all known SKU classes are positives and a sampled
#   fraction of GENERAL_SKU_SINGLE boxes are used as negatives.
#
# USAGE:
#   ./scripts/general_sku_negative_experiment.sh <path_to_dataset> [extra_args]
#
# EXAMPLES:
#   ./scripts/general_sku_negative_experiment.sh data/sku_dataset.zip --train-gss-fraction 0.1 --pos-neg-ratio 1:1
#   ./scripts/general_sku_negative_experiment.sh data/sku_dataset.zip --gss-class GENERAL_SKU_SINGLE --epochs 20
#

set -e
cd "$(dirname "$0")/.."

if [ -z "$1" ]; then
    echo "Usage: ./scripts/general_sku_negative_experiment.sh path/to/dataset.zip [extra_args]"
    exit 1
fi

DATA_PATH=$1
shift

source venv/bin/activate
PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}sku_classifier" \
    python experiments/general_sku_negative_experiment.py \
    --config sku_classifier/config.yaml \
    --data "$DATA_PATH" \
    "$@"
