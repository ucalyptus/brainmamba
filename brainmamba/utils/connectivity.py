"""
Utilities for constructing functional connectivity graphs from timeseries data.
"""

import torch
import numpy as np


def pearson_correlation(x, y):
    """
    Compute Pearson correlation between two vectors.
    
    Args:
        x: First vector
        y: Second vector
        
    Returns:
        Pearson correlation coefficient
    """
    # Center the variables
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    
    # Compute correlation
    numerator = (x_centered * y_centered).sum()
    denominator = torch.sqrt((x_centered ** 2).sum() * (y_centered ** 2).sum())
    
    # Handle division by zero
    if denominator == 0:
        return 0
    
    return numerator / denominator


def construct_functional_connectivity(timeseries, threshold=0.5, absolute=True):
    """
    Construct functional connectivity graph from timeseries data using Pearson correlation.
    
    Args:
        timeseries: Tensor of shape (batch_size, num_nodes, seq_len)
        threshold: Correlation threshold for edge creation
        absolute: Whether to use absolute correlation values
        
    Returns:
        adj_matrix: Adjacency matrix of shape (batch_size, num_nodes, num_nodes)
    """
    batch_size, num_nodes, seq_len = timeseries.shape
    adj_matrix = torch.zeros(batch_size, num_nodes, num_nodes, device=timeseries.device)
    
    for b in range(batch_size):
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):  # Only compute upper triangle
                # Compute correlation
                corr = pearson_correlation(timeseries[b, i], timeseries[b, j])
                
                # Apply absolute if needed
                if absolute:
                    corr = torch.abs(corr)
                
                # Apply threshold
                if corr > threshold:
                    adj_matrix[b, i, j] = corr
                    adj_matrix[b, j, i] = corr  # Symmetric
    
    return adj_matrix


def get_functional_systems(num_nodes, num_systems=7):
    """
    Generate random functional systems for brain units.
    
    In a real application, this would be based on actual brain anatomy.
    
    Args:
        num_nodes: Number of brain units (nodes)
        num_systems: Number of functional systems
        
    Returns:
        functional_systems: Tensor of shape (num_nodes,) indicating the
                           functional system of each node
    """
    # Randomly assign nodes to functional systems
    functional_systems = torch.randint(0, num_systems, (num_nodes,))
    
    return functional_systems 