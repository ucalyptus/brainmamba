"""
BrainMamba: A Selective State Space Model for Brain Dynamics.

This module implements the complete BrainMamba architecture as described in the paper,
combining BTMamba for timeseries encoding and BNMamba for network encoding.
Optimized for H100 GPUs with mixed precision training support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .btmamba import BTMamba
from .bnmamba import BNMamba
from ..utils.connectivity import construct_functional_connectivity, get_functional_systems


class FusionGate(nn.Module):
    """
    Fusion Gate for combining timeseries and network encodings.
    
    This module implements a gating mechanism to fuse the encodings from
    BTMamba and BNMamba, allowing the model to adaptively weight the importance
    of each modality.
    """
    
    def __init__(self, d_model, dropout=0.0):
        """
        Initialize the Fusion Gate.
        
        Args:
            d_model: Model dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Layer normalization
        self.norm_ts = nn.LayerNorm(d_model)
        self.norm_net = nn.LayerNorm(d_model)
        
        # Gate computation
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, ts_encoding, net_encoding):
        """
        Forward pass of the Fusion Gate.
        
        Args:
            ts_encoding: Timeseries encoding from BTMamba
            net_encoding: Network encoding from BNMamba
            
        Returns:
            Fused encoding
        """
        # Apply layer normalization
        ts_norm = self.norm_ts(ts_encoding)
        net_norm = self.norm_net(net_encoding)
        
        # Compute gate
        gate_input = torch.cat([ts_norm, net_norm], dim=-1)
        gate = self.gate(gate_input)
        
        # Apply gate
        fused = gate * ts_norm + (1 - gate) * net_norm
        
        # Apply output projection
        output = self.out_proj(fused)
        
        return output


class MutualInformationLoss(nn.Module):
    """
    Mutual Information Loss for pre-training BrainMamba.
    
    This module implements a contrastive loss that maximizes the mutual information
    between timeseries and network encodings, encouraging the model to learn
    complementary representations.
    """
    
    def __init__(self, d_model, temperature=0.1):
        """
        Initialize the Mutual Information Loss.
        
        Args:
            d_model: Model dimension
            temperature: Temperature parameter for the contrastive loss
        """
        super().__init__()
        
        self.d_model = d_model
        self.temperature = temperature
        
        # Projection heads
        self.proj_ts = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        self.proj_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, ts_encoding, net_encoding):
        """
        Forward pass of the Mutual Information Loss.
        
        Args:
            ts_encoding: Timeseries encoding from BTMamba
            net_encoding: Network encoding from BNMamba
            
        Returns:
            Mutual information loss
        """
        batch_size = ts_encoding.shape[0]
        
        # Project encodings
        ts_proj = self.proj_ts(ts_encoding)
        net_proj = self.proj_net(net_encoding)
        
        # Normalize projections
        ts_proj = F.normalize(ts_proj, dim=-1)
        net_proj = F.normalize(net_proj, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(ts_proj, net_proj.transpose(0, 1)) / self.temperature
        
        # Compute contrastive loss
        labels = torch.arange(batch_size, device=ts_encoding.device)
        loss_ts = F.cross_entropy(sim_matrix, labels)
        loss_net = F.cross_entropy(sim_matrix.transpose(0, 1), labels)
        
        return (loss_ts + loss_net) / 2


class ClassificationHead(nn.Module):
    """
    Classification Head for downstream tasks.
    
    This module implements a simple classification head for downstream tasks
    such as disease classification or cognitive state prediction.
    """
    
    def __init__(self, d_model, num_classes, dropout=0.0):
        """
        Initialize the Classification Head.
        
        Args:
            d_model: Model dimension
            num_classes: Number of classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass of the Classification Head.
        
        Args:
            x: Input tensor of shape (batch_size, d_model)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.classifier(x)


class BrainMamba(nn.Module):
    """
    BrainMamba: A Selective State Space Model for Brain Dynamics.
    
    This module implements the complete BrainMamba architecture as described in the paper,
    combining BTMamba for timeseries encoding and BNMamba for network encoding.
    Optimized for H100 GPUs with mixed precision training support.
    """
    
    def __init__(
        self,
        d_model=64,
        d_state=64,
        n_ts_layers=2,
        n_mpnn_layers=2,
        n_ssm_layers=2,
        num_classes=2,
        dropout=0.0,
        fc_threshold=0.5,
        use_parallel_scan=True,
        use_mixed_precision=True,
    ):
        """
        Initialize the BrainMamba.
        
        Args:
            d_model: Model dimension
            d_state: State dimension for the SSM
            n_ts_layers: Number of timeseries SSM layers
            n_mpnn_layers: Number of message passing layers
            n_ssm_layers: Number of graph SSM layers
            num_classes: Number of classes for downstream tasks
            dropout: Dropout rate
            fc_threshold: Threshold for functional connectivity construction
            use_parallel_scan: Whether to use parallel scan for faster computation
            use_mixed_precision: Whether to use mixed precision training
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.fc_threshold = fc_threshold
        self.use_mixed_precision = use_mixed_precision
        
        # BTMamba for timeseries encoding
        self.btmamba = BTMamba(
            d_model=d_model,
            d_state=d_state,
            n_layers=n_ts_layers,
            dropout=dropout,
            use_parallel_scan=use_parallel_scan
        )
        
        # BNMamba for network encoding
        self.bnmamba = BNMamba(
            d_model=d_model,
            d_state=d_state,
            n_mpnn_layers=n_mpnn_layers,
            n_ssm_layers=n_ssm_layers,
            dropout=dropout,
            use_parallel_scan=use_parallel_scan
        )
        
        # Fusion Gate
        self.fusion_gate = FusionGate(
            d_model=d_model,
            dropout=dropout
        )
        
        # Mutual Information Loss
        self.mi_loss = MutualInformationLoss(
            d_model=d_model
        )
        
        # Classification Head
        self.classification_head = ClassificationHead(
            d_model=d_model,
            num_classes=num_classes,
            dropout=dropout
        )
    
    def forward(self, timeseries, adj_matrix=None, functional_systems=None, return_mi_loss=False):
        """
        Forward pass of the BrainMamba.
        
        Args:
            timeseries: Timeseries data of shape (batch_size, num_nodes, seq_len)
            adj_matrix: Optional adjacency matrix of shape (batch_size, num_nodes, num_nodes)
            functional_systems: Optional system assignments of shape (batch_size, num_nodes)
            return_mi_loss: Whether to return mutual information loss
            
        Returns:
            If return_mi_loss is True:
                logits: Tensor of shape (batch_size, num_classes)
                mi_loss: Mutual information loss
            Else:
                logits: Tensor of shape (batch_size, num_classes)
        """
        batch_size, num_nodes, seq_len = timeseries.shape
        
        # Use mixed precision if enabled
        with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
            # Construct functional connectivity if not provided
            if adj_matrix is None:
                adj_matrix = construct_functional_connectivity(
                    timeseries, threshold=self.fc_threshold, absolute=True
                )
            
            # Generate functional systems if not provided
            if functional_systems is None:
                functional_systems = get_functional_systems(
                    num_nodes, num_systems=7
                ).expand(batch_size, -1)
            
            # Encode timeseries with BTMamba
            ts_encoding = self.btmamba(timeseries)
            
            # Encode network with BNMamba
            net_encoding = self.bnmamba(adj_matrix, functional_systems)
            
            # Compute mutual information loss if needed
            if return_mi_loss:
                mi_loss = self.mi_loss(ts_encoding, net_encoding)
            
            # Fuse encodings
            fused_encoding = self.fusion_gate(ts_encoding, net_encoding)
            
            # Apply classification head
            logits = self.classification_head(fused_encoding)
        
        if return_mi_loss:
            return logits, mi_loss
        else:
            return logits
    
    def pretraining_step(self, timeseries, optimizer, scaler=None):
        """
        Perform a pretraining step using mutual information loss.
        
        Args:
            timeseries: Timeseries data of shape (batch_size, num_nodes, seq_len)
            optimizer: PyTorch optimizer
            scaler: GradScaler for mixed precision training
            
        Returns:
            loss: Mutual information loss
        """
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mutual information loss
        if self.use_mixed_precision and scaler is not None:
            with torch.cuda.amp.autocast():
                _, mi_loss = self.forward(timeseries, return_mi_loss=True)
            
            # Backward pass with gradient scaling
            scaler.scale(mi_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            _, mi_loss = self.forward(timeseries, return_mi_loss=True)
            mi_loss.backward()
            optimizer.step()
        
        return mi_loss.item()
    
    def training_step(self, timeseries, labels, optimizer, scaler=None):
        """
        Perform a training step for downstream tasks.
        
        Args:
            timeseries: Timeseries data of shape (batch_size, num_nodes, seq_len)
            labels: Ground truth labels of shape (batch_size,)
            optimizer: PyTorch optimizer
            scaler: GradScaler for mixed precision training
            
        Returns:
            loss: Classification loss
        """
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        if self.use_mixed_precision and scaler is not None:
            with torch.cuda.amp.autocast():
                logits = self.forward(timeseries)
                loss = F.cross_entropy(logits, labels)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = self.forward(timeseries)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
        
        return loss.item()
    
    def inference(self, timeseries):
        """
        Perform inference on new data.
        
        Args:
            timeseries: Timeseries data of shape (batch_size, num_nodes, seq_len)
            
        Returns:
            predictions: Class predictions
            probabilities: Class probabilities
        """
        # Set model to evaluation mode
        self.eval()
        
        # Forward pass with no gradient computation
        with torch.no_grad():
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    logits = self.forward(timeseries)
            else:
                logits = self.forward(timeseries)
            
            # Compute probabilities and predictions
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
        
        return predictions, probabilities 