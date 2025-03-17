#!/bin/bash
# Setup script for BrainMamba virtual environment

# Set the name of the virtual environment
VENV_NAME="brainmamba_env"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment '$VENV_NAME'..."
python3 -m venv $VENV_NAME

# Activate virtual environment
echo "Activating virtual environment..."
source $VENV_NAME/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Install the BrainMamba package in development mode
echo "Installing BrainMamba in development mode..."
pip install -e .

# Create directories for ABIDE example
echo "Creating directories for ABIDE example..."
mkdir -p data/abide
mkdir -p results/abide
mkdir -p results/abide_h100

echo ""
echo "Virtual environment setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "source $VENV_NAME/bin/activate"
echo ""
echo "To download and process ABIDE data, run:"
echo "python examples/abide_dataset.py --data_dir data/abide --atlas cc200"
echo ""
echo "To train and evaluate BrainMamba on ABIDE, run:"
echo "python examples/abide_classification.py --data_dir data/abide --download_data --use_mixed_precision --save_model"
echo ""
echo "For optimal performance on H100 GPUs, run:"
echo "python examples/abide_classification.py --data_dir data/abide --download_data --atlas cc200 --d_model 128 --d_state 128 --n_ts_layers 3 --n_mpnn_layers 2 --n_ssm_layers 3 --batch_size 32 --num_epochs 100 --pretrain_epochs 20 --use_mixed_precision --save_model --output_dir results/abide_h100" 