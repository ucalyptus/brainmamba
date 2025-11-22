"""
Utilities for constructing functional connectivity graphs from timeseries data.
"""

import torch
from torch import Tensor


def pearson_correlation(x: Tensor, y: Tensor) -> Tensor:
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
    denominator = torch.sqrt((x_centered**2).sum() * (y_centered**2).sum())

    # Handle division by zero
    if denominator == 0:
        return torch.tensor(0.0, device=x.device)

    return numerator / denominator


def construct_functional_connectivity(
    timeseries: Tensor, threshold: float = 0.5, absolute: bool = True
) -> Tensor:
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
    
    # Vectorized implementation

    timeseries_mean = timeseries.mean(dim=-1, keepdim=True)
    timeseries_centered = timeseries - timeseries_mean

    # Covariance matrix (unscaled)
    # (B, N, L) @ (B, L, N) -> (B, N, N)
    numerator = torch.matmul(timeseries_centered, timeseries_centered.transpose(1, 2))

    # Denominator
    # Sum of squares
    sum_sq = (timeseries_centered**2).sum(dim=-1, keepdim=True) # (B, N, 1)
    std = torch.sqrt(sum_sq)
    denominator = torch.matmul(std, std.transpose(1, 2))

    # Correlation matrix
    corr_matrix = numerator / (denominator + 1e-8)

    if absolute:
        corr_matrix = torch.abs(corr_matrix)

    # Threshold
    mask = corr_matrix > threshold
    adj_matrix = corr_matrix * mask.float()

    # Zero out diagonal
    identity_mask = torch.eye(num_nodes, device=timeseries.device).unsqueeze(0).expand(batch_size, -1, -1)
    adj_matrix = adj_matrix * (1 - identity_mask)
    
    return adj_matrix


def get_functional_systems(num_nodes: int, num_systems: int = 7) -> Tensor:
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
