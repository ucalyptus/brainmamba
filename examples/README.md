# BrainMamba Examples

This directory contains example scripts for using the BrainMamba model with various datasets.

## ABIDE Dataset Classification

The Autism Brain Imaging Data Exchange (ABIDE) dataset contains resting-state fMRI data from individuals with Autism Spectrum Disorder (ASD) and typically developing controls. This example demonstrates how to use BrainMamba for autism classification using the ABIDE dataset.

### Requirements

Install the required packages:

```bash
pip install numpy pandas torch scikit-learn requests nibabel nilearn tqdm matplotlib seaborn
```

### Data Preparation

The `abide_dataset.py` script handles downloading and preprocessing the ABIDE dataset. You can run it directly to download the data:

```bash
python abide_dataset.py --data_dir data/abide --atlas cc200
```

This will:
1. Download the phenotypic data and ROI time series for the specified atlas
2. Process the data to extract time series and compute functional connectivity matrices
3. Print summary statistics about the loaded data

### Training and Evaluation

The `abide_classification.py` script handles training and evaluating the BrainMamba model on the ABIDE dataset:

```bash
python abide_classification.py --data_dir data/abide --download_data --use_mixed_precision --save_model
```

Key arguments:
- `--data_dir`: Directory to store the data
- `--download_data`: Download ABIDE data if not already present
- `--pipeline`: Preprocessing pipeline (cpac, dparsf, or niak)
- `--atlas`: Atlas used for ROI extraction (cc200, aal, etc.)
- `--mode`: Data mode to use (timeseries, connectivity, or both)
- `--d_model`: Model dimension
- `--n_ts_layers`: Number of timeseries SSM layers
- `--n_mpnn_layers`: Number of message passing layers
- `--n_ssm_layers`: Number of graph SSM layers
- `--pretrain_epochs`: Number of pretraining epochs
- `--num_epochs`: Number of training epochs
- `--use_mixed_precision`: Use mixed precision training (recommended for H100 GPUs)
- `--save_model`: Save the trained model
- `--output_dir`: Directory to save results

### Results

The script will save the following results in the output directory:
- Training curves (loss and accuracy)
- Confusion matrix
- Test results (accuracy, precision, recall, F1 score)
- Best model weights (if `--save_model` is specified)

### Example Usage for H100 GPUs

For optimal performance on H100 GPUs, use the following configuration:

```bash
python abide_classification.py \
    --data_dir data/abide \
    --download_data \
    --atlas cc200 \
    --d_model 128 \
    --d_state 128 \
    --n_ts_layers 3 \
    --n_mpnn_layers 2 \
    --n_ssm_layers 3 \
    --batch_size 32 \
    --num_epochs 100 \
    --pretrain_epochs 20 \
    --use_mixed_precision \
    --save_model \
    --output_dir results/abide_h100
```

This configuration takes advantage of the H100's tensor cores and large memory capacity to train a larger model with mixed precision.

## Synthetic Data Example

The `classification_example.py` script demonstrates how to use BrainMamba with synthetic data:

```bash
python classification_example.py
```

This is useful for testing the model without downloading real data.

## Other Datasets

The BrainMamba model can be used with other brain imaging datasets as well. The implementation pattern is similar:
1. Create a dataset loader that handles downloading and preprocessing the data
2. Create a training script that uses the dataset loader and BrainMamba model
3. Train and evaluate the model

Some other datasets that can be used with BrainMamba:
- Human Connectome Project (HCP)
- OpenNeuro datasets
- ADHD-200
- UK Biobank 