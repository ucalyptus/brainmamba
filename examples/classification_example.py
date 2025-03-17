#!/usr/bin/env python
"""
BrainMamba Classification Example

This script demonstrates how to use the BrainMamba model for brain activity classification.
It includes data loading, preprocessing, model training, and evaluation.
Optimized for H100 GPUs with mixed precision training.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import sys
sys.path.append('..')
from brainmamba.models.brainmamba import BrainMamba
from brainmamba.utils.connectivity import construct_functional_connectivity, get_functional_systems


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='BrainMamba Classification Example')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='data/sample_data.npz',
                        help='Path to the data file')
    parser.add_argument('--num_nodes', type=int, default=116,
                        help='Number of brain regions/nodes')
    parser.add_argument('--seq_len', type=int, default=200,
                        help='Length of the time series')
    
    # Model parameters
    parser.add_argument('--d_model', type=int, default=64,
                        help='Model dimension')
    parser.add_argument('--d_state', type=int, default=64,
                        help='State dimension for the SSM')
    parser.add_argument('--n_ts_layers', type=int, default=2,
                        help='Number of timeseries SSM layers')
    parser.add_argument('--n_mpnn_layers', type=int, default=2,
                        help='Number of message passing layers')
    parser.add_argument('--n_ssm_layers', type=int, default=2,
                        help='Number of graph SSM layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--fc_threshold', type=float, default=0.5,
                        help='Threshold for functional connectivity construction')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--pretrain_epochs', type=int, default=10,
                        help='Number of pretraining epochs')
    parser.add_argument('--use_mixed_precision', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Device parameters
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    return parser.parse_args()


def load_data(args):
    """
    Load and preprocess the data.
    
    In a real application, this would load actual brain imaging data.
    For this example, we generate synthetic data.
    
    Args:
        args: Command line arguments
        
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        num_classes: Number of classes
    """
    # Generate synthetic data for demonstration
    # In a real application, you would load actual brain imaging data
    print("Generating synthetic data...")
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Generate synthetic timeseries data
    num_samples = 1000
    num_classes = 2
    
    # Generate timeseries with different patterns for each class
    timeseries = np.zeros((num_samples, args.num_nodes, args.seq_len))
    labels = np.zeros(num_samples, dtype=np.int64)
    
    for i in range(num_samples):
        # Randomly assign a class
        label = np.random.randint(0, num_classes)
        labels[i] = label
        
        # Generate timeseries with class-specific patterns
        if label == 0:
            # Class 0: Random walk with positive drift
            for j in range(args.num_nodes):
                timeseries[i, j, 0] = np.random.randn()
                for t in range(1, args.seq_len):
                    timeseries[i, j, t] = timeseries[i, j, t-1] + 0.01 + 0.1 * np.random.randn()
        else:
            # Class 1: Random walk with negative drift
            for j in range(args.num_nodes):
                timeseries[i, j, 0] = np.random.randn()
                for t in range(1, args.seq_len):
                    timeseries[i, j, t] = timeseries[i, j, t-1] - 0.01 + 0.1 * np.random.randn()
    
    # Normalize the timeseries
    for i in range(num_samples):
        for j in range(args.num_nodes):
            mean = np.mean(timeseries[i, j])
            std = np.std(timeseries[i, j])
            if std > 0:
                timeseries[i, j] = (timeseries[i, j] - mean) / std
    
    # Split the data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        timeseries, labels, test_size=0.2, random_state=args.seed
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=args.seed
    )
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    # Create datasets and data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True
    )
    
    print(f"Data generated: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test")
    
    return train_loader, val_loader, test_loader, num_classes


def pretrain_model(model, train_loader, args):
    """
    Pretrain the model using mutual information loss.
    
    Args:
        model: BrainMamba model
        train_loader: DataLoader for training data
        args: Command line arguments
    """
    print("Pretraining the model...")
    
    # Set up optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    
    # Set up gradient scaler for mixed precision training
    scaler = GradScaler() if args.use_mixed_precision else None
    
    # Pretraining loop
    model.train()
    for epoch in range(args.pretrain_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (timeseries, _) in enumerate(train_loader):
            # Move data to device
            timeseries = timeseries.to(args.device)
            
            # Perform pretraining step
            loss = model.pretraining_step(timeseries, optimizer, scaler)
            
            # Update statistics
            total_loss += loss
            num_batches += 1
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Pretrain Epoch: {epoch+1}/{args.pretrain_epochs} "
                      f"[{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss:.6f}")
        
        # Print epoch summary
        avg_loss = total_loss / num_batches
        print(f"Pretrain Epoch: {epoch+1}/{args.pretrain_epochs} "
              f"Average Loss: {avg_loss:.6f}")


def train_model(model, train_loader, val_loader, args):
    """
    Train the model for classification.
    
    Args:
        model: BrainMamba model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        args: Command line arguments
        
    Returns:
        best_model: Best model based on validation accuracy
    """
    print("Training the model...")
    
    # Set up optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    
    # Set up learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Set up gradient scaler for mixed precision training
    scaler = GradScaler() if args.use_mixed_precision else None
    
    # Training loop
    best_val_acc = 0.0
    best_model = None
    
    for epoch in range(args.num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_train_batches = 0
        
        for batch_idx, (timeseries, labels) in enumerate(train_loader):
            # Move data to device
            timeseries = timeseries.to(args.device)
            labels = labels.to(args.device)
            
            # Perform training step
            loss = model.training_step(timeseries, labels, optimizer, scaler)
            
            # Update statistics
            train_loss += loss
            num_train_batches += 1
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Train Epoch: {epoch+1}/{args.num_epochs} "
                      f"[{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss:.6f}")
        
        # Compute average training loss
        avg_train_loss = train_loss / num_train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for timeseries, labels in val_loader:
                # Move data to device
                timeseries = timeseries.to(args.device)
                labels = labels.to(args.device)
                
                # Forward pass
                if args.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        logits = model(timeseries)
                        loss = nn.CrossEntropyLoss()(logits, labels)
                else:
                    logits = model(timeseries)
                    loss = nn.CrossEntropyLoss()(logits, labels)
                
                # Compute predictions
                preds = torch.argmax(logits, dim=1)
                
                # Update statistics
                val_loss += loss.item()
                num_val_batches += 1
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Compute average validation loss and metrics
        avg_val_loss = val_loss / num_val_batches
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        val_acc = accuracy_score(all_labels, all_preds)
        val_precision = precision_score(all_labels, all_preds, average='macro')
        val_recall = recall_score(all_labels, all_preds, average='macro')
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        # Update learning rate scheduler
        scheduler.step(val_acc)
        
        # Print epoch summary
        print(f"Epoch: {epoch+1}/{args.num_epochs} "
              f"Train Loss: {avg_train_loss:.6f} "
              f"Val Loss: {avg_val_loss:.6f} "
              f"Val Acc: {val_acc:.4f} "
              f"Val Precision: {val_precision:.4f} "
              f"Val Recall: {val_recall:.4f} "
              f"Val F1: {val_f1:.4f}")
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict().copy()
            print(f"New best model with validation accuracy: {best_val_acc:.4f}")
    
    # Load the best model
    model.load_state_dict(best_model)
    
    return model


def evaluate_model(model, test_loader, args):
    """
    Evaluate the model on the test set.
    
    Args:
        model: BrainMamba model
        test_loader: DataLoader for test data
        args: Command line arguments
    """
    print("Evaluating the model...")
    
    # Evaluation phase
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for timeseries, labels in test_loader:
            # Move data to device
            timeseries = timeseries.to(args.device)
            
            # Forward pass
            predictions, _ = model.inference(timeseries)
            
            # Update statistics
            all_preds.append(predictions.cpu().numpy())
            all_labels.append(labels.numpy())
    
    # Compute metrics
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    test_acc = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds, average='macro')
    test_recall = recall_score(all_labels, all_preds, average='macro')
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    
    # Print results
    print(f"Test Results:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1 Score: {test_f1:.4f}")


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU instead.")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    args.device = device
    
    # Load data
    train_loader, val_loader, test_loader, num_classes = load_data(args)
    
    # Initialize model
    model = BrainMamba(
        d_model=args.d_model,
        d_state=args.d_state,
        n_ts_layers=args.n_ts_layers,
        n_mpnn_layers=args.n_mpnn_layers,
        n_ssm_layers=args.n_ssm_layers,
        num_classes=num_classes,
        dropout=args.dropout,
        fc_threshold=args.fc_threshold,
        use_parallel_scan=True,
        use_mixed_precision=args.use_mixed_precision,
    ).to(device)
    
    # Print model summary
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Pretrain the model
    if args.pretrain_epochs > 0:
        pretrain_model(model, train_loader, args)
    
    # Train the model
    model = train_model(model, train_loader, val_loader, args)
    
    # Evaluate the model
    evaluate_model(model, test_loader, args)


if __name__ == '__main__':
    main() 