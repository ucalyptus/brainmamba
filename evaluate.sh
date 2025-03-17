#!/bin/bash
# BrainMamba evaluation script

# Check if arguments are provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_path> [output_dir]"
    echo "  model_path: Path to the trained model weights file (.pth)"
    echo "  output_dir: Directory to save evaluation results (default: results/evaluation)"
    exit 1
fi

MODEL_PATH=$1
OUTPUT_DIR=${2:-"results/evaluation"}

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file $MODEL_PATH not found."
    exit 1
fi

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
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16  # Adjust based on your CPU cores
export MKL_NUM_THREADS=16   # Adjust based on your CPU cores

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Get model directory for loading configuration
MODEL_DIR=$(dirname "$MODEL_PATH")
CONFIG_FILE="$MODEL_DIR/training_config.txt"

# Set default parameters
MODEL_DIM=128
STATE_DIM=128
TS_LAYERS=3
MPNN_LAYERS=2
SSM_LAYERS=3

# Load configuration if available
if [ -f "$CONFIG_FILE" ]; then
    echo "Loading configuration from $CONFIG_FILE"
    MODEL_DIM=$(grep "Model dimension" "$CONFIG_FILE" | awk '{print $3}')
    STATE_DIM=$(grep "State dimension" "$CONFIG_FILE" | awk '{print $3}')
    TS_LAYERS=$(grep "Timeseries layers" "$CONFIG_FILE" | awk '{print $3}')
    MPNN_LAYERS=$(grep "MPNN layers" "$CONFIG_FILE" | awk '{print $3}')
    SSM_LAYERS=$(grep "SSM layers" "$CONFIG_FILE" | awk '{print $3}')
else
    echo "Configuration file not found. Using default parameters."
fi

# Create a script to run the evaluation
EVAL_SCRIPT="$OUTPUT_DIR/evaluate.py"
cat > $EVAL_SCRIPT << EOL
#!/usr/bin/env python
"""
Evaluation script for BrainMamba model
"""
import argparse
import os
import sys
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Add parent directory to path
sys.path.append('.')

from brainmamba.models.brainmamba import BrainMamba
from examples.abide_dataset import load_abide_data, create_abide_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description='BrainMamba Evaluation Script')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the trained model weights')
    parser.add_argument('--data_dir', type=str, default='data/abide',
                        help='Directory containing the ABIDE dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save evaluation results')
    parser.add_argument('--d_model', type=int, default=128,
                        help='Model dimension')
    parser.add_argument('--d_state', type=int, default=128,
                        help='State dimension')
    parser.add_argument('--n_ts_layers', type=int, default=3,
                        help='Number of timeseries layers')
    parser.add_argument('--n_mpnn_layers', type=int, default=2,
                        help='Number of MPNN layers')
    parser.add_argument('--n_ssm_layers', type=int, default=3,
                        help='Number of SSM layers')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--use_mixed_precision', action='store_true',
                        help='Use mixed precision inference')
    parser.add_argument('--atlas', type=str, default='cc200',
                        help='Atlas used for ROI extraction')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['timeseries', 'connectivity', 'both'],
                        help='Data mode to use')
    return parser.parse_args()

def evaluate_model(model, test_loader, device, args):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Extract data based on mode
            if args.mode == 'timeseries':
                timeseries = batch['timeseries'].to(device)
                labels = batch['label']
                
                # Forward pass
                if args.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        predictions, probabilities = model.inference(timeseries)
                else:
                    predictions, probabilities = model.inference(timeseries)
            elif args.mode == 'connectivity':
                connectivity = batch['connectivity'].to(device)
                labels = batch['label']
                
                # Create dummy timeseries from connectivity
                dummy_timeseries = torch.randn(
                    connectivity.shape[0], connectivity.shape[1], 100, 
                    device=device
                )
                
                # Forward pass
                if args.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        logits = model(dummy_timeseries, connectivity)
                else:
                    logits = model(dummy_timeseries, connectivity)
                
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)
            else:  # 'both'
                timeseries = batch['timeseries'].to(device)
                connectivity = batch['connectivity'].to(device)
                labels = batch['label']
                functional_systems = batch['functional_systems'].to(device) if 'functional_systems' in batch else None
                
                # Forward pass
                if args.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        predictions, probabilities = model.inference(timeseries)
                else:
                    predictions, probabilities = model.inference(timeseries)
            
            # Update statistics
            all_preds.append(predictions.cpu().numpy())
            all_probs.append(probabilities.cpu().numpy())
            all_labels.append(labels.numpy())
    
    # Compute metrics
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)
    
    # Return metrics
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist()
    }
    
    return metrics, all_preds, all_probs, all_labels

def plot_confusion_matrix(cm, output_dir):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Control', 'ASD'], 
                yticklabels=['Control', 'ASD'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def plot_roc_curve(y_true, y_prob, output_dir):
    from sklearn.metrics import roc_curve, auc
    
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load ABIDE data
    print("Loading ABIDE data...")
    timeseries, connectivity, labels, subject_ids, phenotypic = load_abide_data(
        args.data_dir,
        atlas=args.atlas
    )
    
    # Create data loaders
    print("Creating data loaders...")
    _, _, test_loader, num_classes, num_nodes, seq_len = create_abide_dataloaders(
        timeseries, connectivity, labels,
        batch_size=args.batch_size,
        mode=args.mode
    )
    
    # Initialize model
    print("Initializing model...")
    model = BrainMamba(
        d_model=args.d_model,
        d_state=args.d_state,
        n_ts_layers=args.n_ts_layers,
        n_mpnn_layers=args.n_mpnn_layers,
        n_ssm_layers=args.n_ssm_layers,
        num_classes=num_classes,
        use_mixed_precision=args.use_mixed_precision,
    ).to(device)
    
    # Load model weights
    print(f"Loading weights from {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # Evaluate model
    print("Evaluating model...")
    metrics, all_preds, all_probs, all_labels = evaluate_model(model, test_loader, device, args)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save metrics
    print("Saving results...")
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Plot confusion matrix
    plot_confusion_matrix(np.array(metrics['confusion_matrix']), args.output_dir)
    
    # Plot ROC curve
    plot_roc_curve(all_labels, all_probs, args.output_dir)
    
    # Print results
    print(f"Results saved to {args.output_dir}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")

if __name__ == '__main__':
    main()
EOL

chmod +x $EVAL_SCRIPT

# Run the evaluation
echo "Running evaluation..."
python $EVAL_SCRIPT \
    --model_path $MODEL_PATH \
    --data_dir data/abide \
    --output_dir $OUTPUT_DIR \
    --d_model $MODEL_DIM \
    --d_state $STATE_DIM \
    --n_ts_layers $TS_LAYERS \
    --n_mpnn_layers $MPNN_LAYERS \
    --n_ssm_layers $SSM_LAYERS \
    --use_mixed_precision \
    --atlas cc200 \
    --mode both

echo "Evaluation complete. Results saved to $OUTPUT_DIR" 