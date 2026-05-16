#!/bin/bash
# scripts/infer_batch.sh
#
# PURPOSE:
#   Processes a whole folder of images or crops and saves results to a summary CSV.
#
# USAGE:
#   ./scripts/infer_batch.sh <checkpoint_pth> <mode_specific_args>
#
# ARGUMENTS:
#   <checkpoint_pth>     : Path to the trained .pth model.
#   <mode_specific_args> : Directories for images/boxes or crops.
#
# EXAMPLES (Mode A - Folder of Images + Folder of Boxes):
#   ./scripts/infer_batch.sh runs/.../best.pth --images_dir ./imgs --boxes_dir ./txts --output_dir ./results
#
# EXAMPLES (Mode B - Folder of Crops):
#   ./scripts/infer_batch.sh runs/.../best.pth --crops_dir ./crops --output_dir ./results
#

set -e
cd "$(dirname "$0")/.."

if [ -z "$1" ]; then
    echo "Usage: ./scripts/infer_batch.sh path/to/checkpoint.pth [extra_args]"
    exit 1
fi

CHECKPOINT=$1
shift

source venv/bin/activate
sku-clf infer-batch --config sku_classifier/config.yaml --checkpoint "$CHECKPOINT" "$@"
