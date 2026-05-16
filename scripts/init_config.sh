#!/bin/bash
# scripts/init_config.sh
#
# PURPOSE:
#   Scans a dataset and interactively helps you choose positive/negative classes to update config.yaml.
#
# USAGE:
#   ./scripts/init_config.sh <path_to_dataset>
#
# ARGUMENTS:
#   <path_to_dataset> : Path to a ZIP file or directory containing 'images/' and 'labels/'.
#
# EXAMPLES:
#   ./scripts/init_config.sh data/my_dataset.zip
#   ./scripts/init_config.sh ./raw_data_folder
#

set -e
cd "$(dirname "$0")/.."

if [ -z "$1" ]; then
    echo "Usage: ./scripts/init_config.sh path/to/dataset.zip"
    exit 1
fi

source venv/bin/activate
sku-clf init-config --data "$1" --config sku_classifier/config.yaml
