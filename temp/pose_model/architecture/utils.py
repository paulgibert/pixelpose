"""Utility blocks for neural network architectures."""

from typing import Tuple
from torch import nn


class DownBlock(nn.Module):
    """Downsampling convolution block with optional normalization.
    
    Structure: Conv2d → [GroupNorm] → SiLU
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int, norm_groups: int):
        super().__init__()

        padding = (kernel_size - 1) // 2
        
        layers = [
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding)
        ]
        
        if norm_groups > 0:
            layers.append(nn.GroupNorm(norm_groups, out_channels))
        
        layers.append(nn.SiLU())
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Upsampling transposed convolution block with optional normalization.
    
    Structure: ConvTranspose2d → [GroupNorm] → SiLU
    
    Padding is calculated to ensure output = input * stride for exact reversal
    of downsampling operations.
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int, norm_groups: int):
        super().__init__()

        # Calculate padding for exact upsampling
        if kernel_size % 2 == 0:  # Even kernel
            padding = (kernel_size - stride) // 2
            output_padding = 0
        else:  # Odd kernel
            padding = (kernel_size - 1) // 2
            output_padding = stride - 1

        layers = [
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               output_padding=output_padding)
        ]
        
        if norm_groups > 0:
            layers.append(nn.GroupNorm(norm_groups, out_channels))
        
        layers.append(nn.SiLU())
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class LazyMLP(nn.Module):
    """Multi-layer perceptron with lazy input dimension inference.
    
    First layer uses LazyLinear to infer input dimensions on first forward pass.
    Subsequent layers have fixed dimensions based on hidden_dims.
    """
    
    def __init__(self, out_features: int, hidden_dims: Tuple[int, ...]):
        super().__init__()

        hidden_dims = tuple(hidden_dims)  # Ensure it's a tuple
        
        if len(hidden_dims) > 0:
            layers = [nn.LazyLinear(hidden_dims[0]), nn.SiLU()]

            for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
                layers += [
                    nn.Linear(in_dim, out_dim),
                    nn.SiLU()
                ]
            
            layers.append(nn.Linear(hidden_dims[-1], out_features))
            self.layers = nn.Sequential(*layers)
        else:
            # No hidden layers - direct mapping
            self.layers = nn.LazyLinear(out_features)

    def forward(self, x):
        return self.layers(x)


class LazyFiLMAdapter(nn.Module):
    """Feature-wise Linear Modulation (FiLM) adapter with lazy input inference.
    
    Applies affine transformation to feature maps conditioned on input features:
        output = x * gamma + beta
    
    where gamma and beta are learned from input features via an MLP.
    """
    
    def __init__(self, out_channels: int, hidden_dims: Tuple[int, ...]):
        super().__init__()
        # MLP outputs 2x channels (for gamma and beta)
        self.mlp = LazyMLP(out_channels * 2, hidden_dims)

    def forward(self, x, features):
        """Apply FiLM conditioning.
        
        Args:
            x: Feature map [B, C, H, W]
            features: Conditioning features [B, D]
            
        Returns:
            Modulated features [B, C, H, W]
        """
        gamma, beta = self.mlp(features).chunk(2, dim=-1)
        # Broadcast to spatial dimensions
        return x * gamma[..., None, None] + beta[..., None, None]
