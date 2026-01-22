# ILM - Infilling Language Models

Project for training and inference with Infilling Language Models (ILM).

Based on: [Enabling Language Models to Fill in the Blanks](https://arxiv.org/abs/2005.05339) (ACL 2020)

## Project Structure

```
ilms/
├── inference/          # Inference scripts and tools
│   ├── setup.sh        # Setup script for inference environment
│   ├── ilm_inference.py        # Basic inference example
│   ├── ilm_interactive.py      # Interactive CSV-based inference
│   └── venv/           # Python virtual environment
├── training/           # Training code and data preparation
│   └── ilm/            # Cloned ILM repository
├── models/             # Pretrained and fine-tuned models
│   └── sto_ilm/        # Story infilling model
└── data/               # Datasets
```

## Quick Start

### 1. Setup Inference Environment

```bash
cd inference
./setup.sh
```

This will:
- Create a Python 3.9 virtual environment
- Install dependencies (PyTorch, transformers, etc.)
- Install the ILM package from training/ilm
- Download the pretrained story infilling model

### 2. Run Interactive Inference

```bash
cd inference
source venv/bin/activate
python ilm_interactive.py -i /path/to/your/data.csv --n-masks 5 --shuffle
```

Options:
- `-i, --input`: Path to CSV file with 'text' column (required)
- `--n-masks`: Number of words to randomly mask (default: 3)
- `--num-infills`: Number of predictions to generate (default: 2)
- `--shuffle`: Randomize text order
- `--seed`: Random seed for reproducibility
- `--start-idx`: Start from specific index

Controls:
- **Enter**: Next text
- **r**: Re-mask current text
- **q**: Quit
- **number**: Jump to specific index

### 3. Basic Inference Example

```bash
cd inference
source venv/bin/activate
python ilm_inference.py
```

## Training

The training code is in `training/ilm/`. See the [original ILM repository](https://github.com/chrisdonahue/ilm) for training instructions.

## Requirements

- Python 3.9.x (via pyenv)
- PyTorch
- transformers==4.5.1
- CUDA (optional, for GPU acceleration)
