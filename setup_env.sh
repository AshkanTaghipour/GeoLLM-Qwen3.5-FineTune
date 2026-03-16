#!/bin/bash
# setup_env.sh — Set up the conda environment for Qwen 3.5 fine-tuning
#
# Usage:
#   bash setup_env.sh
#
# This script:
#   1. Loads miniforge3 module
#   2. Creates a conda env in ./qwen_finetune_env (if it doesn't exist)
#   3. Installs PyTorch with CUDA 12.4 support
#   4. Installs all dependencies from requirements.txt
#
# After running, activate with:
#   module load miniforge3
#   conda activate ./qwen_finetune_env

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${SCRIPT_DIR}/qwen_finetune_env"

echo "============================================"
echo "Setting up Qwen 3.5 fine-tuning environment"
echo "============================================"

# Load conda
module load miniforge3

# Create env if it doesn't exist
if [ ! -d "${ENV_DIR}" ]; then
    echo "[1/4] Creating conda environment at ${ENV_DIR} ..."
    conda create -p "${ENV_DIR}" python=3.11 -y
else
    echo "[1/4] Conda environment already exists at ${ENV_DIR}"
fi

# Activate
conda activate "${ENV_DIR}"

# Install PyTorch with CUDA 12.4 (compatible with CUDA 12.8 drivers)
echo "[2/4] Installing PyTorch with CUDA 12.4 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install all other dependencies
echo "[3/4] Installing training dependencies..."
pip install unsloth datasets trl peft accelerate bitsandbytes tensorboard pytest reportlab sentencepiece protobuf

echo "[4/4] Verifying installation..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
import transformers
print(f'  Transformers: {transformers.__version__}')
import datasets
print(f'  Datasets: {datasets.__version__}')
import trl
print(f'  TRL: {trl.__version__}')
import peft
print(f'  PEFT: {peft.__version__}')
print('  All imports successful!')
"

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "To activate the environment:"
echo "  module load miniforge3"
echo "  conda activate ${ENV_DIR}"
echo ""
echo "To run training (on a GPU node):"
echo "  python prepare_data.py"
echo "  python train.py"
echo ""
echo "To view training metrics:"
echo "  tensorboard --logdir ./logs"
echo "============================================"
