# BrainMamba

This repository implements the BrainMamba architecture for brain activity modeling and analysis as described in the paper "BrainMamba: A Novel Brain Dynamics Modeling Framework via Selective State Space Models".

BrainMamba is a novel brain dynamics modeling framework that captures both short and long-range dependencies in brain activity data. It integrates timeseries and brain network data through a selective state space model (Mamba) architecture to model complex brain dynamics.

## Description

BrainMamba is a deep learning model for analyzing brain activity data, combining the power of Mamba (Selective State Space Models) with specialized components for processing brain timeseries and connectivity data.

## Features

- **BTMamba**: Brain Timeseries Mamba for encoding multivariate timeseries data from brain activity
- **BNMamba**: Brain Network Mamba for encoding brain networks (functional connectivity)
- **Fusion Gate**: Mechanism to combine timeseries and network encodings
- **Mutual Information Loss**: Pretraining objective to maximize information between encoders
- **Classification Head**: For downstream tasks such as disease classification

## Installation

To set up the environment and install the BrainMamba package, run:

```bash
./setup_venv.sh
```

This will create a virtual environment, install all dependencies, and set up the package in development mode.

## Usage

### Running the model

To train the BrainMamba model on a single device (optimized for H100 GPUs), run:

```bash
./run.sh
```

This script will:
1. Set up the environment if necessary
2. Download the ABIDE dataset if it's not already present
3. Train the BrainMamba model with optimized parameters
4. Save the trained model and results

### Evaluating a trained model

To evaluate a trained model, use:

```bash
./evaluate.sh <path_to_model_weights> [output_directory]
```

For example:
```bash
./evaluate.sh results/abide_h100/model_final.pth results/evaluation
```

The evaluation script will:
1. Load the trained model weights
2. Evaluate the model on the test set
3. Generate performance metrics including accuracy, precision, recall, and F1 score
4. Create visualizations (confusion matrix and ROC curve)
5. Save all results to the specified output directory

## Architecture

BrainMamba consists of two main components:

1. **Brain Timeseries Mamba (BTMamba)**: Encodes multivariate timeseries data from brain activity
   - Cross-Variate MLP for information fusion
   - Variate Encoder for encoding individual timeseries
   - Bidirectional Readout for brain-level encoding

2. **Brain Network Mamba (BNMamba)**: Encodes brain networks
   - Message Passing Neural Network for local dependencies
   - Functional Ordering for organizing brain units
   - Selective Graph SSM for long-range dependencies

These components are combined in the BrainMamba architecture for downstream tasks such as classification and prediction.

## Dataset

This implementation supports the ABIDE dataset for autism spectrum disorder (ASD) classification. The dataset will be automatically downloaded and processed when running the model.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)

For a complete list of dependencies, see setup.py.

## License

MIT 