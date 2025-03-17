"""
Brain Timeseries Mamba (BTMamba) implementation.

This module implements the BTMamba component of the BrainMamba architecture,
which is designed to encode multivariate timeseries data from brain activity.
Optimized for H100 GPUs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .selective_ssm import SelectiveSSMBlock


class CrossVariateMLP(nn.Module):
    """
    Cross-Variate MLP for fusing information across variates in multivariate timeseries.
    
    As described in the paper, this module uses a simple MLP to bind temporal
    information across variates (brain units).
    """
    
    def __init__(self, d_model, expansion_factor=2, dropout=0.0):
        """
        Initialize the Cross-Variate MLP.
        
        Args:
            d_model: Model dimension
            expansion_factor: Expansion factor for the hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_hidden = int(d_model * expansion_factor)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Two-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, self.d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_hidden, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Forward pass of the Cross-Variate MLP.
        
        Args:
            x: Input tensor of shape (batch_size, num_variates, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, num_variates, seq_len)
        """
        # Transpose to (batch_size, seq_len, num_variates)
        x_t = x.transpose(1, 2)
        
        # Apply layer normalization
        x_norm = self.norm(x_t)
        
        # Apply MLP
        y = self.mlp(x_norm)
        
        # Residual connection
        y = y + x_t
        
        # Transpose back to (batch_size, num_variates, seq_len)
        return y.transpose(1, 2)


class VariateEncoder(nn.Module):
    """
    Variate Encoder for encoding each variate (brain unit) timeseries.
    
    This module uses a selective SSM to encode each variate's timeseries.
    """
    
    def __init__(self, d_model, d_state=64, n_layers=2, dropout=0.0, use_parallel_scan=True):
        """
        Initialize the Variate Encoder.
        
        Args:
            d_model: Model dimension
            d_state: State dimension for the SSM
            n_layers: Number of SSM layers
            dropout: Dropout rate
            use_parallel_scan: Whether to use parallel scan for faster computation
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers
        
        # Stack of SSM blocks
        self.layers = nn.ModuleList([
            SelectiveSSMBlock(
                d_model=d_model,
                d_state=d_state,
                dropout=dropout,
                use_parallel_scan=use_parallel_scan
            )
            for _ in range(n_layers)
        ])
    
    def forward(self, x):
        """
        Forward pass of the Variate Encoder.
        
        Args:
            x: Input tensor of shape (batch_size, num_variates, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, num_variates, d_model)
        """
        # Process each variate separately
        batch_size, num_variates, seq_len = x.shape
        
        # Reshape to process each variate as a separate sequence
        x_reshaped = x.view(batch_size * num_variates, 1, seq_len)
        
        # Project to d_model dimension if needed
        if seq_len != self.d_model:
            x_reshaped = nn.Linear(seq_len, self.d_model)(x_reshaped.squeeze(1)).unsqueeze(1)
        
        # Apply SSM layers
        for layer in self.layers:
            x_reshaped = layer(x_reshaped)
        
        # Reshape back to (batch_size, num_variates, d_model)
        return x_reshaped.view(batch_size, num_variates, self.d_model)


class BidirectionalReadout(nn.Module):
    """
    Bidirectional Readout function for brain-level encoding.
    
    This module implements the readout function as shown in the architecture diagram,
    with multiple linear layers, activation functions, and multiply operations.
    """
    
    def __init__(self, d_model, d_state=64, dropout=0.0):
        """
        Initialize the Bidirectional Readout.
        
        Args:
            d_model: Model dimension
            d_state: State dimension for the SSM
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        
        # Forward and backward SSM blocks
        self.forward_ssm = SelectiveSSMBlock(
            d_model=d_model,
            d_state=d_state,
            dropout=dropout
        )
        
        self.backward_ssm = SelectiveSSMBlock(
            d_model=d_model,
            d_state=d_state,
            dropout=dropout
        )
        
        # Linear layers as shown in the diagram
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.linear3 = nn.Linear(d_model, d_model)
        
        # Activation functions
        self.activation1 = nn.GELU()
        self.activation2 = nn.GELU()
        self.activation3 = nn.GELU()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Final output projection
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        """
        Forward pass of the Bidirectional Readout.
        
        Args:
            x: Input tensor of shape (batch_size, num_variates, d_model)
            
        Returns:
            Output tensor of shape (batch_size, d_model)
        """
        # Process in forward direction
        x_forward = self.forward_ssm(x)
        
        # Process in backward direction
        x_backward = self.backward_ssm(torch.flip(x, dims=[1]))
        x_backward = torch.flip(x_backward, dims=[1])
        
        # Take the last token from forward and first token from backward
        x_forward_last = x_forward[:, -1, :]
        x_backward_first = x_backward[:, 0, :]
        
        # Combine forward and backward representations
        x_combined = (x_forward_last + x_backward_first) / 2
        
        # Apply the readout network as shown in the diagram
        # Linear1 -> Activation1
        h1 = self.activation1(self.linear1(x_combined))
        
        # Linear2 -> Activation2
        h2 = self.activation2(self.linear2(x_combined))
        
        # Linear3 -> Activation3
        h3 = self.activation3(self.linear3(x_combined))
        
        # Multiply operations
        m1 = h1 * x_combined
        m2 = h2 * x_combined
        m3 = h3 * x_combined
        
        # Sum
        sum_output = m1 + m2 + m3
        
        # Apply dropout
        sum_output = self.dropout(sum_output)
        
        # Final projection
        output = self.out_proj(sum_output)
        
        return output


class BTMamba(nn.Module):
    """
    Brain Timeseries Mamba (BTMamba) for encoding multivariate brain signals.
    
    This module implements the complete BTMamba architecture as described in the paper.
    Optimized for H100 GPUs.
    """
    
    def __init__(
        self,
        d_model=64,
        d_state=64,
        n_layers=2,
        dropout=0.0,
        use_parallel_scan=True,
    ):
        """
        Initialize the BTMamba.
        
        Args:
            d_model: Model dimension
            d_state: State dimension for the SSM
            n_layers: Number of SSM layers
            dropout: Dropout rate
            use_parallel_scan: Whether to use parallel scan for faster computation
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        
        # Cross-Variate MLP for inter-variate information fusing
        self.cross_variate_mlp = CrossVariateMLP(
            d_model=d_model,
            dropout=dropout
        )
        
        # Variate Encoder for encoding each variate
        self.variate_encoder = VariateEncoder(
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            dropout=dropout,
            use_parallel_scan=use_parallel_scan
        )
        
        # Bidirectional Readout for brain-level encoding
        self.readout = BidirectionalReadout(
            d_model=d_model,
            d_state=d_state,
            dropout=dropout
        )
    
    def forward(self, x, return_node_encodings=False):
        """
        Forward pass of the BTMamba.
        
        Args:
            x: Input tensor of shape (batch_size, num_variates, seq_len)
            return_node_encodings: Whether to return node-level encodings
            
        Returns:
            If return_node_encodings is True:
                node_encodings: Tensor of shape (batch_size, num_variates, d_model)
                brain_encoding: Tensor of shape (batch_size, d_model)
            Else:
                brain_encoding: Tensor of shape (batch_size, d_model)
        """
        # Apply Cross-Variate MLP
        z = self.cross_variate_mlp(x)
        
        # Apply Variate Encoder
        node_encodings = self.variate_encoder(z)
        
        # Apply Bidirectional Readout
        brain_encoding = self.readout(node_encodings)
        
        if return_node_encodings:
            return node_encodings, brain_encoding
        else:
            return brain_encoding 