#!/bin/bash
# Setup script for ABIDE example

# Create directories
mkdir -p data/abide
mkdir -p results/abide
mkdir -p results/abide_h100

echo "Directory structure created."
echo "To download and process ABIDE data, run:"
echo "python abide_dataset.py --data_dir data/abide --atlas cc200"
echo ""
echo "To train and evaluate BrainMamba on ABIDE, run:"
echo "python abide_classification.py --data_dir data/abide --download_data --use_mixed_precision --save_model"
echo ""
echo "For optimal performance on H100 GPUs, run:"
echo "python abide_classification.py --data_dir data/abide --download_data --atlas cc200 --d_model 128 --d_state 128 --n_ts_layers 3 --n_mpnn_layers 2 --n_ssm_layers 3 --batch_size 32 --num_epochs 100 --pretrain_epochs 20 --use_mixed_precision --save_model --output_dir results/abide_h100" 