"""
Brain Network Mamba (BNMamba) implementation.

This module implements the BNMamba component of the BrainMamba architecture,
which is designed to encode brain networks (functional connectivity graphs).
Optimized for H100 GPUs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .selective_ssm import SelectiveSSMBlock


class MessagePassingLayer(nn.Module):
    """
    Message Passing Neural Network layer for encoding local dependencies in brain networks.
    
    This module implements a graph neural network layer that updates node
    representations based on their neighbors in the brain network.
    """
    
    def __init__(self, d_model, dropout=0.0):
        """
        Initialize the Message Passing Layer.
        
        Args:
            d_model: Model dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Message function
        self.message_fn = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )
        
        # Update function
        self.update_fn = nn.GRUCell(d_model, d_model)
    
    def forward(self, x, adj_matrix):
        """
        Forward pass of the Message Passing Layer.
        
        Args:
            x: Node features of shape (batch_size, num_nodes, d_model)
            adj_matrix: Adjacency matrix of shape (batch_size, num_nodes, num_nodes)
            
        Returns:
            Updated node features of shape (batch_size, num_nodes, d_model)
        """
        batch_size, num_nodes, _ = x.shape
        
        # Apply layer normalization
        x_norm = self.norm(x)
        
        # Compute messages
        messages = torch.zeros_like(x_norm)
        
        for b in range(batch_size):
            for i in range(num_nodes):
                # Get neighbors
                neighbors = torch.nonzero(adj_matrix[b, i]).squeeze(-1)
                
                if neighbors.numel() > 0:
                    # Get neighbor features
                    neighbor_feats = x_norm[b, neighbors]
                    
                    # Repeat node features for each neighbor
                    node_feats_repeated = x_norm[b, i].unsqueeze(0).repeat(neighbors.numel(), 1)
                    
                    # Concatenate node and neighbor features
                    combined_feats = torch.cat([node_feats_repeated, neighbor_feats], dim=-1)
                    
                    # Compute messages
                    neighbor_messages = self.message_fn(combined_feats)
                    
                    # Aggregate messages (mean)
                    messages[b, i] = neighbor_messages.mean(dim=0)
        
        # Update node representations
        x_updated = torch.zeros_like(x)
        for b in range(batch_size):
            x_updated[b] = self.update_fn(
                messages[b].view(-1, self.d_model),
                x[b].view(-1, self.d_model)
            )
        
        # Residual connection
        return x + x_updated


class FunctionalOrdering(nn.Module):
    """
    Functional Ordering module for organizing brain units based on functional systems.
    
    This module implements the functional ordering component as shown in the diagram,
    which reorders brain units based on their functional systems for more effective
    sequential processing.
    """
    
    def __init__(self, d_model, dropout=0.0):
        """
        Initialize the Functional Ordering module.
        
        Args:
            d_model: Model dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        
        # System embedding
        self.system_embedding = nn.Embedding(10, d_model)  # Support up to 10 systems
        
        # Projection for combining node features with system embedding
        self.projection = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, functional_systems):
        """
        Forward pass of the Functional Ordering module.
        
        Args:
            x: Node features of shape (batch_size, num_nodes, d_model)
            functional_systems: System assignments of shape (batch_size, num_nodes)
            
        Returns:
            Reordered node features of shape (batch_size, num_nodes, d_model)
        """
        batch_size, num_nodes, _ = x.shape
        
        # Get system embeddings
        system_embeds = self.system_embedding(functional_systems)
        
        # Combine node features with system embeddings
        combined_features = torch.cat([x, system_embeds], dim=-1)
        enhanced_features = self.projection(combined_features)
        
        # Sort nodes by functional system
        sorted_indices = []
        sorted_features = torch.zeros_like(enhanced_features)
        
        for b in range(batch_size):
            # Get indices sorted by functional system
            _, indices = torch.sort(functional_systems[b])
            sorted_indices.append(indices)
            
            # Reorder features
            sorted_features[b] = enhanced_features[b, indices]
        
        return sorted_features, sorted_indices


class SelectiveGraphSSM(nn.Module):
    """
    Selective Graph SSM for encoding long-range dependencies in brain networks.
    
    This module applies a selective SSM to the reordered brain units to capture
    long-range dependencies across functional systems.
    """
    
    def __init__(self, d_model, d_state=64, n_layers=2, dropout=0.0, use_parallel_scan=True):
        """
        Initialize the Selective Graph SSM.
        
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
    
    def forward(self, x, sorted_indices):
        """
        Forward pass of the Selective Graph SSM.
        
        Args:
            x: Sorted node features of shape (batch_size, num_nodes, d_model)
            sorted_indices: Indices used for sorting
            
        Returns:
            Updated node features of shape (batch_size, num_nodes, d_model)
        """
        # Apply SSM layers
        for layer in self.layers:
            x = layer(x)
        
        # Reorder back to original order
        batch_size = x.shape[0]
        original_order = torch.zeros_like(x)
        
        for b in range(batch_size):
            # Create inverse mapping
            inverse_indices = torch.zeros_like(sorted_indices[b])
            inverse_indices[sorted_indices[b]] = torch.arange(len(sorted_indices[b]), device=x.device)
            
            # Reorder features back to original order
            original_order[b] = x[b, inverse_indices]
        
        return original_order


class BNMamba(nn.Module):
    """
    Brain Network Mamba (BNMamba) for encoding brain networks.
    
    This module implements the complete BNMamba architecture as described in the paper.
    Optimized for H100 GPUs.
    """
    
    def __init__(
        self,
        d_model=64,
        d_state=64,
        n_mpnn_layers=2,
        n_ssm_layers=2,
        dropout=0.0,
        use_parallel_scan=True,
    ):
        """
        Initialize the BNMamba.
        
        Args:
            d_model: Model dimension
            d_state: State dimension for the SSM
            n_mpnn_layers: Number of message passing layers
            n_ssm_layers: Number of SSM layers
            dropout: Dropout rate
            use_parallel_scan: Whether to use parallel scan for faster computation
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        
        # Initial node embedding
        self.node_embedding = nn.Linear(1, d_model)
        
        # Message Passing Neural Network for local dependencies
        self.mpnn_layers = nn.ModuleList([
            MessagePassingLayer(d_model=d_model, dropout=dropout)
            for _ in range(n_mpnn_layers)
        ])
        
        # Functional Ordering
        self.functional_ordering = FunctionalOrdering(d_model=d_model, dropout=dropout)
        
        # Selective Graph SSM for long-range dependencies
        self.selective_graph_ssm = SelectiveGraphSSM(
            d_model=d_model,
            d_state=d_state,
            n_layers=n_ssm_layers,
            dropout=dropout,
            use_parallel_scan=use_parallel_scan
        )
        
        # Readout function
        self.readout = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, adj_matrix, functional_systems=None, return_node_encodings=False):
        """
        Forward pass of the BNMamba.
        
        Args:
            adj_matrix: Adjacency matrix of shape (batch_size, num_nodes, num_nodes)
            functional_systems: Optional system assignments of shape (batch_size, num_nodes)
            return_node_encodings: Whether to return node-level encodings
            
        Returns:
            If return_node_encodings is True:
                node_encodings: Tensor of shape (batch_size, num_nodes, d_model)
                graph_encoding: Tensor of shape (batch_size, d_model)
            Else:
                graph_encoding: Tensor of shape (batch_size, d_model)
        """
        batch_size, num_nodes, _ = adj_matrix.shape
        
        # Generate random functional systems if not provided
        if functional_systems is None:
            functional_systems = torch.randint(0, 7, (batch_size, num_nodes), device=adj_matrix.device)
        
        # Initial node features (degree centrality)
        node_degrees = adj_matrix.sum(dim=-1, keepdim=True)
        node_features = self.node_embedding(node_degrees)
        
        # Apply Message Passing Neural Network
        for mpnn_layer in self.mpnn_layers:
            node_features = mpnn_layer(node_features, adj_matrix)
        
        # Apply Functional Ordering
        ordered_features, sorted_indices = self.functional_ordering(node_features, functional_systems)
        
        # Apply Selective Graph SSM
        node_encodings = self.selective_graph_ssm(ordered_features, sorted_indices)
        
        # Global readout (mean pooling)
        graph_encoding = node_encodings.mean(dim=1)
        graph_encoding = self.readout(graph_encoding)
        
        if return_node_encodings:
            return node_encodings, graph_encoding
        else:
            return graph_encoding 