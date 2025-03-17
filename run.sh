#!/bin/bash
# BrainMamba single-device execution script
# Optimized for H100 GPUs

# Check if virtual environment exists
if [ ! -d "brainmamba_env" ]; then
    echo "Virtual environment not found. Creating it now..."
    ./setup_venv.sh
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Exiting."
        exit 1
    fi
else
    echo "Using existing virtual environment."
fi

# Activate virtual environment
source brainmamba_env/bin/activate

# Set environment variables for optimal single GPU performance
export CUDA_VISIBLE_DEVICES=7
export OMP_NUM_THREADS=16  # Adjust based on your CPU cores
export MKL_NUM_THREADS=16   # Adjust based on your CPU cores

# Create data directory if it doesn't exist
mkdir -p data/abide

# Check if ABIDE dataset needs to be downloaded
if [ ! -f "data/abide/Phenotypic_V1_0b_preprocessed1.csv" ]; then
    echo "Downloading ABIDE dataset..."
    echo "NOTE: Due to changes in ABIDE data access, you may need to download data manually."
    echo "Attempting automatic download first..."
    
    python examples/abide_dataset.py --data_dir data/abide --atlas cc200
    
    # Check download status
    if [ $? -ne 0 ]; then
        echo ""
        echo "Automatic download failed. Please download the ABIDE dataset manually from one of these sources:"
        echo "1. NITRC-IR: https://nitrc.org/ir/app/template/Index.vm (select ABIDE)"
        echo "2. COINS: https://coins.trendscenter.org/dataexchange/"
        echo "3. LORIS: https://abide.loris.ca/ (login: abide, password: abide_2012)"
        echo ""
        echo "After downloading, place the files in data/abide/rois_cc200/"
        echo "Format: SITE_SUBJECTID_rois_cc200.1D (e.g., NYU_51456_rois_cc200.1D)"
        echo ""
        read -p "Press Enter to continue once you've downloaded the required files, or Ctrl+C to exit..."
    fi
else
    echo "ABIDE dataset already downloaded."
fi

# Set training parameters
MODEL_DIM=128
STATE_DIM=128
TS_LAYERS=3
MPNN_LAYERS=2
SSM_LAYERS=3
BATCH_SIZE=32
EPOCHS=100
PRETRAIN_EPOCHS=20
OUTPUT_DIR="results/abide_h100"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run training with optimized parameters for H100 GPU
echo "Starting training on single GPU..."
python examples/abide_classification.py \
    --data_dir data/abide \
    --atlas cc200 \
    --d_model $MODEL_DIM \
    --d_state $STATE_DIM \
    --n_ts_layers $TS_LAYERS \
    --n_mpnn_layers $MPNN_LAYERS \
    --n_ssm_layers $SSM_LAYERS \
    --batch_size $BATCH_SIZE \
    --num_epochs $EPOCHS \
    --pretrain_epochs $PRETRAIN_EPOCHS \
    --use_mixed_precision \
    --save_model \
    --output_dir $OUTPUT_DIR

# Save training configuration
echo "Saving configuration..."
CONFIG_FILE="$OUTPUT_DIR/training_config.txt"
echo "Training configuration:" > $CONFIG_FILE
echo "------------------------" >> $CONFIG_FILE
echo "Model dimension: $MODEL_DIM" >> $CONFIG_FILE
echo "State dimension: $STATE_DIM" >> $CONFIG_FILE
echo "Timeseries layers: $TS_LAYERS" >> $CONFIG_FILE
echo "MPNN layers: $MPNN_LAYERS" >> $CONFIG_FILE
echo "SSM layers: $SSM_LAYERS" >> $CONFIG_FILE
echo "Batch size: $BATCH_SIZE" >> $CONFIG_FILE
echo "Epochs: $EPOCHS" >> $CONFIG_FILE
echo "Pretrain epochs: $PRETRAIN_EPOCHS" >> $CONFIG_FILE
echo "------------------------" >> $CONFIG_FILE
echo "Device: H100 GPU" >> $CONFIG_FILE
echo "Mixed precision: Enabled" >> $CONFIG_FILE
date >> $CONFIG_FILE

echo "Training complete. Results saved to $OUTPUT_DIR" 