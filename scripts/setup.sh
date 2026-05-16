#!/bin/bash
# scripts/setup.sh
#
# PURPOSE:
#   Automates the creation of a Python virtual environment and installs the sku_classifier package.
#
# USAGE:
#   ./scripts/setup.sh
#
# DESCRIPTION:
#   1. Creates a 'venv' directory if it doesn't exist.
#   2. Activates the virtual environment.
#   3. Upgrades pip to the latest version.
#   4. Installs the project in 'editable' mode (-e).
#
# AFTER RUNNING:
#   You must activate the environment in your shell to use the commands:
#   source venv/bin/activate
#

set -e
cd "$(dirname "$0")/.."

echo "--- SKU Classifier Setup ---"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment 'venv' already exists."
fi

source venv/bin/activate
echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing sku_classifier package..."
pip install -e ./sku_classifier

echo "----------------------------------------"
echo "Setup complete!"
echo "Run 'source venv/bin/activate' to start."
echo "----------------------------------------"
