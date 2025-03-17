#!/usr/bin/env python
"""
ABIDE Classification Example for BrainMamba

This script demonstrates how to use the BrainMamba model for autism classification
using the ABIDE dataset. It includes data loading, preprocessing, model training,
and evaluation.

Optimized for H100 GPUs with mixed precision training.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys

# Add parent directory to path to import BrainMamba modules
sys.path.append('..')
from brainmamba.models.brainmamba import BrainMamba
from abide_dataset import download_abide_preproc, load_abide_data, create_abide_dataloaders


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='BrainMamba ABIDE Classification Example')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/abide',
                        help='Directory to store the data')
    parser.add_argument('--pipeline', type=str, default='cpac',
                        help='Preprocessing pipeline (cpac, dparsf, or niak)')
    parser.add_argument('--strategy', type=str, default='filt_global',
                        help='Noise reduction strategy')
    parser.add_argument('--atlas', type=str, default='cc200',
                        help='Atlas used for ROI extraction (cc200, aal, etc.)')
    parser.add_argument('--min_scan_length', type=int, default=100,
                        help='Minimum number of time points required')
    parser.add_argument('--download_data', action='store_true',
                        help='Download ABIDE data if not already present')
    
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
    parser.add_argument('--batch_size', type=int, default=16,
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
    parser.add_argument('--mode', type=str, default='both',
                        choices=['timeseries', 'connectivity', 'both'],
                        help='Data mode to use (timeseries, connectivity, or both)')
    
    # Device parameters
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results/abide',
                        help='Directory to save results')
    parser.add_argument('--save_model', action='store_true',
                        help='Save the trained model')
    
    return parser.parse_args()


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
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}/{args.pretrain_epochs}")):
            # Extract data
            if args.mode == 'timeseries':
                timeseries = batch['timeseries'].to(args.device)
                # Perform pretraining step
                loss = model.pretraining_step(timeseries, optimizer, scaler)
            elif args.mode == 'connectivity':
                # For connectivity-only mode, we need to create a dummy timeseries
                # This is because the pretraining step requires timeseries data
                connectivity = batch['connectivity'].to(args.device)
                # Create dummy timeseries from connectivity
                dummy_timeseries = torch.randn(
                    connectivity.shape[0], connectivity.shape[1], 100, 
                    device=args.device
                )
                # Set the adjacency matrix directly
                loss = model.pretraining_step(dummy_timeseries, optimizer, scaler)
            else:  # 'both'
                timeseries = batch['timeseries'].to(args.device)
                connectivity = batch['connectivity'].to(args.device)
                functional_systems = batch['functional_systems'].to(args.device) if 'functional_systems' in batch else None
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
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    val_accs = []
    
    for epoch in range(args.num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_train_batches = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{args.num_epochs}")):
            # Extract data
            if args.mode == 'timeseries':
                timeseries = batch['timeseries'].to(args.device)
                labels = batch['label'].to(args.device)
                # Perform training step
                loss = model.training_step(timeseries, labels, optimizer, scaler)
            elif args.mode == 'connectivity':
                connectivity = batch['connectivity'].to(args.device)
                labels = batch['label'].to(args.device)
                # Create dummy timeseries from connectivity
                dummy_timeseries = torch.randn(
                    connectivity.shape[0], connectivity.shape[1], 100, 
                    device=args.device
                )
                # Override the forward method to use connectivity directly
                with torch.no_grad():
                    model.bnmamba.eval()  # Set to eval mode to avoid updating batch norm stats
                    # Forward pass with connectivity
                    if args.use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            logits = model(dummy_timeseries, connectivity)
                            loss = nn.CrossEntropyLoss()(logits, labels)
                    else:
                        logits = model(dummy_timeseries, connectivity)
                        loss = nn.CrossEntropyLoss()(logits, labels)
                
                # Backward pass
                optimizer.zero_grad()
                if args.use_mixed_precision and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                loss = loss.item()
            else:  # 'both'
                timeseries = batch['timeseries'].to(args.device)
                connectivity = batch['connectivity'].to(args.device)
                labels = batch['label'].to(args.device)
                functional_systems = batch['functional_systems'].to(args.device) if 'functional_systems' in batch else None
                
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
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{args.num_epochs}"):
                # Extract data
                if args.mode == 'timeseries':
                    timeseries = batch['timeseries'].to(args.device)
                    labels = batch['label'].to(args.device)
                    
                    # Forward pass
                    if args.use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            logits = model(timeseries)
                            loss = nn.CrossEntropyLoss()(logits, labels)
                    else:
                        logits = model(timeseries)
                        loss = nn.CrossEntropyLoss()(logits, labels)
                elif args.mode == 'connectivity':
                    connectivity = batch['connectivity'].to(args.device)
                    labels = batch['label'].to(args.device)
                    
                    # Create dummy timeseries from connectivity
                    dummy_timeseries = torch.randn(
                        connectivity.shape[0], connectivity.shape[1], 100, 
                        device=args.device
                    )
                    
                    # Forward pass with connectivity
                    if args.use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            logits = model(dummy_timeseries, connectivity)
                            loss = nn.CrossEntropyLoss()(logits, labels)
                    else:
                        logits = model(dummy_timeseries, connectivity)
                        loss = nn.CrossEntropyLoss()(logits, labels)
                else:  # 'both'
                    timeseries = batch['timeseries'].to(args.device)
                    connectivity = batch['connectivity'].to(args.device)
                    labels = batch['label'].to(args.device)
                    functional_systems = batch['functional_systems'].to(args.device) if 'functional_systems' in batch else None
                    
                    # Forward pass
                    if args.use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            logits = model(timeseries, connectivity, functional_systems)
                            loss = nn.CrossEntropyLoss()(logits, labels)
                    else:
                        logits = model(timeseries, connectivity, functional_systems)
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
        val_losses.append(avg_val_loss)
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        val_acc = accuracy_score(all_labels, all_preds)
        val_accs.append(val_acc)
        
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
            
            if args.save_model:
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'))
    
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
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Extract data
            if args.mode == 'timeseries':
                timeseries = batch['timeseries'].to(args.device)
                labels = batch['label']
                
                # Forward pass
                predictions, probabilities = model.inference(timeseries)
            elif args.mode == 'connectivity':
                connectivity = batch['connectivity'].to(args.device)
                labels = batch['label']
                
                # Create dummy timeseries from connectivity
                dummy_timeseries = torch.randn(
                    connectivity.shape[0], connectivity.shape[1], 100, 
                    device=args.device
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
                timeseries = batch['timeseries'].to(args.device)
                connectivity = batch['connectivity'].to(args.device)
                labels = batch['label']
                functional_systems = batch['functional_systems'].to(args.device) if 'functional_systems' in batch else None
                
                # Forward pass
                predictions, probabilities = model.inference(timeseries)
            
            # Update statistics
            all_preds.append(predictions.cpu().numpy())
            all_probs.append(probabilities.cpu().numpy())
            all_labels.append(labels.numpy())
    
    # Compute metrics
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    test_acc = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds, average='macro')
    test_recall = recall_score(all_labels, all_preds, average='macro')
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print results
    print(f"Test Results:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Control', 'ASD'], 
                yticklabels=['Control', 'ASD'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    # Save results to file
    results = {
        'accuracy': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'confusion_matrix': cm.tolist()
    }
    
    import json
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=4)


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
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Download data if requested
    if args.download_data:
        print("Downloading ABIDE data...")
        pheno_file = download_abide_preproc(
            args.data_dir, 
            pipeline=args.pipeline,
            strategy=args.strategy,
            derivatives=[f'rois_{args.atlas}']
        )
    
    # Load data
    print("Loading ABIDE data...")
    timeseries, connectivity, labels, subject_ids, phenotypic = load_abide_data(
        args.data_dir,
        pipeline=args.pipeline,
        strategy=args.strategy,
        atlas=args.atlas,
        min_scan_length=args.min_scan_length,
        fc_threshold=args.fc_threshold
    )
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader, num_classes, num_nodes, seq_len = create_abide_dataloaders(
        timeseries, connectivity, labels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        mode=args.mode
    )
    
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
    
    print("Done!")


if __name__ == '__main__':
    main() 