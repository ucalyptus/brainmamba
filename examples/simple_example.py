"""
Simple example demonstrating how to use BrainMamba.

This script creates a toy dataset of brain activity and demonstrates
how to use BrainMamba for classification.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_curve, auc

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brainmamba.models import BrainMamba, BrainMambaForClassification


def generate_toy_data(batch_size=32, num_nodes=10, seq_len=100, num_classes=2):
    """
    Generate toy brain activity data.
    
    Args:
        batch_size: Number of samples
        num_nodes: Number of brain units (nodes)
        seq_len: Length of the timeseries
        num_classes: Number of classes
        
    Returns:
        timeseries: Tensor of shape (batch_size, num_nodes, seq_len)
        adj_matrix: Tensor of shape (batch_size, num_nodes, num_nodes)
        labels: Tensor of shape (batch_size,)
    """
    # Generate random timeseries
    timeseries = torch.randn(batch_size, num_nodes, seq_len)
    
    # Add class-specific patterns
    labels = torch.randint(0, num_classes, (batch_size,))
    
    for i in range(batch_size):
        class_idx = labels[i].item()
        
        # Add class-specific frequency components
        t = torch.linspace(0, 10, seq_len)
        freq = 1.0 + 0.5 * class_idx
        
        # Add sine wave with class-specific frequency to some nodes
        for j in range(num_nodes // 2):
            timeseries[i, j] += torch.sin(2 * np.pi * freq * t) * 0.5
    
    # Generate random adjacency matrices (functional connectivity)
    adj_matrix = torch.zeros(batch_size, num_nodes, num_nodes)
    
    for i in range(batch_size):
        # Create a random symmetric adjacency matrix
        temp = torch.rand(num_nodes, num_nodes)
        temp = (temp + temp.T) / 2  # Make symmetric
        temp = (temp > 0.7).float()  # Threshold to create sparse matrix
        
        # Set diagonal to zero (no self-connections)
        temp.fill_diagonal_(0)
        
        # Add class-specific connectivity patterns
        class_idx = labels[i].item()
        if class_idx == 0:
            # Class 0: Stronger connections between first half of nodes
            temp[:num_nodes//2, :num_nodes//2] += 0.3
        else:
            # Class 1: Stronger connections between second half of nodes
            temp[num_nodes//2:, num_nodes//2:] += 0.3
        
        # Threshold again to ensure binary adjacency matrix
        temp = (temp > 0.7).float()
        temp.fill_diagonal_(0)
        
        adj_matrix[i] = temp
    
    return timeseries, adj_matrix, labels


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate toy data
    print("Generating toy data...")
    batch_size = 64
    num_nodes = 16
    seq_len = 128
    num_classes = 2
    
    timeseries, adj_matrix, labels = generate_toy_data(
        batch_size=batch_size,
        num_nodes=num_nodes,
        seq_len=seq_len,
        num_classes=num_classes
    )
    
    # Split into train and test sets
    train_size = int(0.8 * batch_size)
    
    train_timeseries = timeseries[:train_size]
    train_adj_matrix = adj_matrix[:train_size]
    train_labels = labels[:train_size]
    
    test_timeseries = timeseries[train_size:]
    test_adj_matrix = adj_matrix[train_size:]
    test_labels = labels[train_size:]
    
    # Create BrainMamba model
    print("Creating BrainMamba model...")
    model = BrainMambaForClassification(
        d_model=32,
        d_state=32,
        n_layers=2,
        n_mpnn_layers=2,
        num_classes=num_classes,
        dropout=0.1
    )
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    print("Training the model...")
    num_epochs = 20
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(train_timeseries, train_adj_matrix)
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(logits, train_labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Compute training accuracy
        _, predicted = torch.max(logits, 1)
        train_acc = (predicted == train_labels).float().mean().item()
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_logits = model(test_timeseries, test_adj_matrix)
            _, test_predicted = torch.max(test_logits, 1)
            test_acc = (test_predicted == test_labels).float().mean().item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, "
              f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    # Final evaluation
    print("\nFinal evaluation:")
    model.eval()
    with torch.no_grad():
        # Get predictions on test set
        test_logits = model(test_timeseries, test_adj_matrix)
        test_probs = torch.nn.functional.softmax(test_logits, dim=1)
        _, test_predicted = torch.max(test_logits, 1)
        
        # Compute accuracy
        test_acc = accuracy_score(test_labels.numpy(), test_predicted.numpy())
        print(f"Test Accuracy: {test_acc:.4f}")
        
        # Compute PR-AUC for binary classification
        if num_classes == 2:
            precision, recall, _ = precision_recall_curve(
                test_labels.numpy(), test_probs[:, 1].numpy()
            )
            pr_auc = auc(recall, precision)
            print(f"PR-AUC: {pr_auc:.4f}")
    
    # Visualize a sample
    plt.figure(figsize=(12, 8))
    
    # Plot timeseries
    plt.subplot(2, 1, 1)
    plt.imshow(timeseries[0].numpy(), aspect='auto', cmap='viridis')
    plt.colorbar(label='Amplitude')
    plt.title(f'Timeseries (Class {labels[0].item()})')
    plt.xlabel('Time')
    plt.ylabel('Brain Unit')
    
    # Plot adjacency matrix
    plt.subplot(2, 1, 2)
    plt.imshow(adj_matrix[0].numpy(), cmap='Blues')
    plt.colorbar(label='Connection Strength')
    plt.title(f'Adjacency Matrix (Class {labels[0].item()})')
    plt.xlabel('Brain Unit')
    plt.ylabel('Brain Unit')
    
    plt.tight_layout()
    plt.savefig('sample_visualization.png')
    print("Visualization saved to 'sample_visualization.png'")


if __name__ == "__main__":
    main() 