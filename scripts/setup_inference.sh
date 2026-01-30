#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$SCRIPT_DIR"

# Use Python 3.9.25 via pyenv
PYTHON_VERSION="3.9.25"
PYTHON_BIN="$(pyenv root)/versions/${PYTHON_VERSION}/bin/python"

if [ ! -f "$PYTHON_BIN" ]; then
    echo "Error: Python ${PYTHON_VERSION} not found in pyenv"
    echo "Install it with: pyenv install ${PYTHON_VERSION}"
    exit 1
fi

echo "Using Python: $PYTHON_BIN"
echo "Version: $($PYTHON_BIN --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_BIN -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

echo "Installing dependencies..."

# Upgrade pip first
pip install --upgrade pip wheel setuptools

# Install PyTorch (CPU version for compatibility, GPU will auto-upgrade if available)
pip install torch

# Install compatible transformers with pinned packaging version to avoid version parsing issues
pip install 'packaging<22'
pip install transformers==4.5.1

# Install other dependencies
pip install numpy tqdm regex requests filelock sacremoses gdown

# Install ilm package from training folder
ILM_DIR="$PROJECT_ROOT/training/ilm"
if [ -d "$ILM_DIR" ]; then
    echo "Installing ilm package from training/ilm..."
    pip install -e "$ILM_DIR"
else
    echo "Error: ILM package not found at $ILM_DIR"
    echo "Please clone the repo: git clone https://github.com/chrisdonahue/ilm.git $ILM_DIR"
    exit 1
fi

# Download pretrained model using gdown (handles Google Drive large files)
MODEL_DIR="$PROJECT_ROOT/models/sto_ilm"
if [ ! -f "$MODEL_DIR/pytorch_model.bin" ] || [ $(stat -c%s "$MODEL_DIR/pytorch_model.bin" 2>/dev/null || echo 0) -lt 1000000 ]; then
    echo "Downloading pretrained model to $MODEL_DIR..."
    rm -rf "$MODEL_DIR"
    mkdir -p "$MODEL_DIR"
    cd "$MODEL_DIR"
    gdown 1oYFLxkX6mWbmpEwQH8BmgE7iKix2W0zU -O pytorch_model.bin
    gdown 15JnXi7L6LeEB2fq4dFK2WRvDKyX46hVi -O config.json
    gdown 1nTQVe2tfkWV8dumbrLIHzMgPwpLIbYUd -O additional_ids_to_tokens.pkl
    cd "$SCRIPT_DIR"
fi

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "Project structure:"
echo "  $PROJECT_ROOT/"
echo "  ├── inference/    <- You are here"
echo "  ├── training/ilm/ <- ILM source code"
echo "  ├── models/       <- Pretrained models"
echo "  └── data/         <- Datasets"
echo ""
echo "To run inference:"
echo "  cd $SCRIPT_DIR"
echo "  source venv/bin/activate"
echo "  python ilm_interactive.py -i <csv_file>"
echo "============================================"
