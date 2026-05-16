#!/bin/bash
set -e
cd "$(dirname "$0")/.."

if [ -z "$1" ]; then
    echo "Usage: ./scripts/run_negative_fraction_experiment.sh /path/to/merged_yolo_dataset [extra_args]"
    exit 1
fi

DATA_PATH=$1
shift

PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}sku_classifier" \
    python -m sku_clf.experiment \
    --data "$DATA_PATH" \
    --config sku_classifier/config.yaml \
    "$@"
