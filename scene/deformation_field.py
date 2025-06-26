# scene/deformation_field.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class LowRankDeformationField(nn.Module):
    """
    Low-rank deformation field for monocular 4D Gaussian Splatting.
    
    Produces time-dependent deformation coefficients b(t) that are combined
    with per-splat basis matrices W to generate deformations: Δμ = W @ b(t).
    """
    
    def __init__(
        self,
        rank: int = 4,
        time_embed_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 3,
        activation: str = 'relu',
        time_encoding: str = 'fourier',
        fourier_freq_max: float = 10.0,
        fourier_num_freqs: int = 6,
    ):
        """
        Initialize low-rank deformation field.
        
        Args:
            rank: Rank of deformation basis (r in W @ b(t))
            time_embed_dim: Dimension of time embedding
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP layers
            activation: Activation function ('relu', 'gelu', 'silu')
            time_encoding: Time encoding method ('fourier', 'sinusoidal', 'learned')
            fourier_freq_max: Maximum frequency for Fourier encoding
            fourier_num_freqs: Number of frequency bands for Fourier encoding
        """
        super().__init__()
        
        self.rank = rank
        self.time_embed_dim = time_embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.time_encoding = time_encoding
        
        # Time encoding
        if time_encoding == 'fourier':
            self.fourier_freqs = nn.Parameter(
                torch.logspace(0, np.log10(fourier_freq_max), fourier_num_freqs),
                requires_grad=False
            )
            time_input_dim = 1 + 2 * fourier_num_freqs  # 1 + 2*freqs (sin, cos)
        elif time_encoding == 'sinusoidal':
            self.pos_encoding_freqs = nn.Parameter(
                torch.pow(10000, -torch.arange(0, time_embed_dim, 2) / time_embed_dim),
                requires_grad=False
            )
            time_input_dim = time_embed_dim
        elif time_encoding == 'learned':
            self.time_embedding = nn.Embedding(1000, time_embed_dim)  # Support up to 1000 time steps
            time_input_dim = time_embed_dim
        else:
            time_input_dim = 1  # Raw time
            
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # MLP layers
        layers = []
        input_dim = time_input_dim
        
        for i in range(num_layers):
            output_dim = hidden_dim if i < num_layers - 1 else rank
            layers.append(nn.Linear(input_dim, output_dim))
            
            if i < num_layers - 1:
                layers.append(self.activation)
                
            input_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for hidden layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize final layer with small weights for stability
        if len(self.mlp) > 0:
            final_layer = self.mlp[-1] if not isinstance(self.mlp[-1], nn.ReLU) else self.mlp[-2]
            if isinstance(final_layer, nn.Linear):
                nn.init.normal_(final_layer.weight, std=0.01)
                if final_layer.bias is not None:
                    nn.init.zeros_(final_layer.bias)
    
    def encode_time(self, t: torch.Tensor) -> torch.Tensor:
        """
        Encode time using the specified encoding method.
        
        Args:
            t: Time values [B] or [B, 1]
            
        Returns:
            encoded_time: Encoded time features [B, time_embed_dim]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        if self.time_encoding == 'fourier':
            # Fourier encoding: [t, sin(2π*f*t), cos(2π*f*t), ...]
            freqs = self.fourier_freqs.unsqueeze(0)  # [1, num_freqs]
            angles = 2 * np.pi * freqs * t  # [B, num_freqs]
            encoded = torch.cat([
                t,  # Raw time
                torch.sin(angles),
                torch.cos(angles)
            ], dim=-1)  # [B, 1 + 2*num_freqs]
            
        elif self.time_encoding == 'sinusoidal':
            # Sinusoidal positional encoding like in Transformer
            angles = t * self.pos_encoding_freqs.unsqueeze(0)  # [B, embed_dim//2]
            encoded = torch.cat([
                torch.sin(angles),
                torch.cos(angles)
            ], dim=-1)  # [B, embed_dim]
            
        elif self.time_encoding == 'learned':
            # Learned embedding (requires integer time indices)
            t_int = torch.clamp(t.long().squeeze(-1), 0, 999)
            encoded = self.time_embedding(t_int)  # [B, embed_dim]
            
        else:
            # Raw time
            encoded = t
            
        return encoded
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate deformation coefficients.
        
        Args:
            t: Time values [B] or [B, 1]
            
        Returns:
            coefficients: Deformation coefficients [B, rank]
        """
        # Encode time
        time_features = self.encode_time(t)
        
        # Pass through MLP
        coefficients = self.mlp(time_features)
        
        return coefficients
    
    def get_regularization_loss(self) -> torch.Tensor:
        """
        Get regularization loss for the deformation field parameters.
        
        Returns:
            loss: L2 regularization loss
        """
        loss = 0.0
        for param in self.parameters():
            loss += torch.sum(param ** 2)
        return loss


class SplatDeformationManager:
    """
    Manager for per-splat deformation basis matrices W.
    
    Handles initialization, updates, and optimization of the low-rank
    deformation basis matrices for each Gaussian splat.
    """
    
    def __init__(
        self,
        num_splats: int,
        rank: int = 4,
        device: str = 'cuda',
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize splat deformation manager.
        
        Args:
            num_splats: Number of Gaussian splats
            rank: Rank of deformation basis
            device: Device to store tensors
            dtype: Data type for basis matrices (float16 for memory efficiency)
        """
        self.num_splats = num_splats
        self.rank = rank
        self.device = device
        self.dtype = dtype
        
        # Initialize deformation basis matrices W [N, 3, r]
        self.W = self._initialize_basis_matrices()
        
        # Track which splats have active deformation
        self.deformation_mask = torch.ones(num_splats, device=device, dtype=torch.bool)
    
    def _initialize_basis_matrices(self) -> torch.Tensor:
        """
        Initialize deformation basis matrices.
        
        Returns:
            W: Deformation basis matrices [N, 3, r]
        """
        # Initialize with small random values
        W = torch.randn(
            self.num_splats, 3, self.rank,
            device=self.device, dtype=self.dtype
        ) * 0.01
        
        return nn.Parameter(W)
    
    def add_splats(self, num_new_splats: int) -> None:
        """
        Add new splats and initialize their deformation matrices.
        
        Args:
            num_new_splats: Number of new splats to add
        """
        # Initialize new basis matrices
        new_W = torch.randn(
            num_new_splats, 3, self.rank,
            device=self.device, dtype=self.dtype
        ) * 0.01
        
        # Concatenate with existing matrices
        self.W = nn.Parameter(torch.cat([self.W.data, new_W], dim=0))
        
        # Update deformation mask
        new_mask = torch.ones(num_new_splats, device=self.device, dtype=torch.bool)
        self.deformation_mask = torch.cat([self.deformation_mask, new_mask], dim=0)
        
        # Update splat count
        self.num_splats += num_new_splats
    
    def remove_splats(self, mask: torch.Tensor) -> None:
        """
        Remove splats based on a boolean mask.
        
        Args:
            mask: Boolean mask [N] where True indicates splats to keep
        """
        self.W = nn.Parameter(self.W.data[mask])
        self.deformation_mask = self.deformation_mask[mask]
        self.num_splats = mask.sum().item()
    
    def get_basis_matrices(self, active_only: bool = False) -> torch.Tensor:
        """
        Get deformation basis matrices.
        
        Args:
            active_only: If True, return only matrices for active splats
            
        Returns:
            W: Deformation basis matrices [N, 3, r] or [N_active, 3, r]
        """
        if active_only:
            return self.W[self.deformation_mask]
        return self.W
    
    def set_deformation_mask(self, mask: torch.Tensor) -> None:
        """
        Set which splats have active deformation.
        
        Args:
            mask: Boolean mask [N] where True indicates active deformation
        """
        assert mask.shape[0] == self.num_splats
        self.deformation_mask = mask
    
    def get_regularization_loss(self) -> torch.Tensor:
        """
        Get regularization loss for deformation basis matrices.
        
        Returns:
            loss: Frobenius norm regularization loss
        """
        # L2 regularization on basis matrices
        W_active = self.W[self.deformation_mask]
        return torch.sum(W_active ** 2)
    
    def optimize_basis_via_svd(self, deformation_history: torch.Tensor) -> None:
        """
        Optimize basis matrices using SVD on deformation history.
        
        This can be called periodically to update the basis matrices
        based on observed deformations.
        
        Args:
            deformation_history: Historical deformations [N, 3, T] for T time steps
        """
        N, _, T = deformation_history.shape
        
        for i in range(N):
            if not self.deformation_mask[i]:
                continue
                
            # SVD on deformation history for splat i
            U, S, Vt = torch.svd(deformation_history[i])  # [3, T] -> U[3,3], S[3], Vt[T,3]
            
            # Keep top-r components
            self.W.data[i] = U[:, :self.rank].to(self.dtype)


def create_deformation_field(
    rank: int = 4,
    **kwargs
) -> Tuple[LowRankDeformationField, SplatDeformationManager]:
    """
    Factory function to create deformation field and splat manager.
    
    Args:
        rank: Rank of deformation basis
        **kwargs: Additional arguments for LowRankDeformationField
        
    Returns:
        deform_field: Time-dependent deformation field
        splat_manager: Per-splat deformation manager (initialized with 0 splats)
    """
    deform_field = LowRankDeformationField(rank=rank, **kwargs)
    
    # Splat manager will be initialized when splats are created
    splat_manager = None
    
    return deform_field, splat_manager