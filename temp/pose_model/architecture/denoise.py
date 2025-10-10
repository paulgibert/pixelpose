"""UNet denoiser with reference image conditioning via CLIP embeddings."""

from dataclasses import dataclass, asdict
from typing import Tuple
import torch
from torch import nn
from .utils import DownBlock, UpBlock, LazyFiLMAdapter


@dataclass
class ConvConfig:
    """Convolutional layer configuration."""
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    norm_groups: int  # 0 disables normalization


class ReferenceBlock(nn.Module):
    """Processes reference image conditioned on CLIP embeddings.
    
    Use encoder() or decoder() factory methods to create instances.
    """
    
    def __init__(self, conv: ConvConfig, clip_dims: Tuple[int, ...],
                 conv_module: nn.Module):
        super().__init__()
        self.adapter = LazyFiLMAdapter(conv.in_channels, clip_dims)
        self.conv = conv_module

    @classmethod
    def encoder(cls, conv: ConvConfig, clip_dims: Tuple[int, ...]):
        """Create encoder reference block (downsampling)."""
        conv_module = DownBlock(
            conv.in_channels, conv.out_channels, conv.kernel_size,
            conv.stride, conv.norm_groups
        )
        return cls(conv, clip_dims, conv_module)

    @classmethod
    def decoder(cls, conv: ConvConfig, clip_dims: Tuple[int, ...]):
        """Create decoder reference block (upsampling)."""
        conv_module = UpBlock(
            conv.in_channels, conv.out_channels, conv.kernel_size,
            conv.stride, conv.norm_groups
        )
        return cls(conv, clip_dims, conv_module)

    def forward(self, x, clip):
        """Apply CLIP conditioning then convolution."""
        x = self.adapter(x, clip)
        return self.conv(x)



class PoseEncoder(nn.Module):
    def __init__(self, conv_list: Tuple[ConvConfig]):
        super().__init__()
        # Note: Despite use of DownBlock, we aren't downsampling
        self.blocks = nn.Sequential(
            *[DownBlock(**asdict(conv)) for conv in conv_list]
        )

    def forward(self, x):
        return self.blocks(x)


@dataclass
class UNetBlockConfig:
    """Configuration for a single UNet encoder/decoder block."""
    ref_conv: ConvConfig      # Reference path convolution
    unet_conv: ConvConfig     # UNet path convolution
    clip_dims: Tuple[int, ...]  # Hidden dims for CLIP projection
    ref_dims: Tuple[int, ...]   # Hidden dims for reference projection


class UNetBlock(nn.Module):
    """UNet block with reference image conditioning.
    
    Processes both the UNet path and reference path in parallel,
    conditioning UNet features on flattened reference features.
    
    Use encoder() or decoder() factory methods to create instances.
    """
    
    def __init__(self, in_channels: int, ref_dims: Tuple[int, ...],
                 conv_module: nn.Module, ref_module: nn.Module):
        super().__init__()
        self.ref = ref_module
        self.adapter = LazyFiLMAdapter(in_channels, ref_dims)
        self.conv = conv_module
    
    @classmethod
    def encoder(cls, config: UNetBlockConfig):
        """Create encoder block (downsampling both paths)."""
        in_channels = config.unet_conv.in_channels
        ref_dims = config.ref_dims
        conv_module = DownBlock(**asdict(config.unet_conv))
        ref_module = ReferenceBlock.encoder(config.ref_conv, config.clip_dims)
        return cls(in_channels, ref_dims, conv_module, ref_module)
    
    @classmethod
    def decoder(cls, config: UNetBlockConfig):
        """Create decoder block (upsampling both paths)."""
        in_channels = config.unet_conv.in_channels
        ref_dims = config.ref_dims
        conv_module = UpBlock(**asdict(config.unet_conv))
        ref_module = ReferenceBlock.decoder(config.ref_conv, config.clip_dims)
        return cls(in_channels, ref_dims, conv_module, ref_module)
    
    def forward(self, x, latent, clip):
        """Process both paths and return updated features."""
        # Update reference path
        latent = self.ref(latent, clip)
        
        # Condition UNet path on reference features
        latent_flat = latent.view(latent.shape[0], -1)
        x = self.adapter(x, latent_flat)
        x = self.conv(x)
        
        return x, latent


@dataclass
class UNetConfig:
    """Full UNet denoiser configuration."""
    pose_encoder: Tuple[ConvConfig, ...]
    encoder: Tuple[UNetBlockConfig, ...]
    decoder: Tuple[UNetBlockConfig, ...]
    bottleneck_kernel: int


class UNetDenoiser(nn.Module):
    """UNet denoiser with CLIP-conditioned reference image guidance.
    
    Architecture:
        - Parallel encoder paths for UNet and reference image
        - Reference features condition UNet features via FiLM
        - Skip connections from encoder to decoder
        - Symmetric encoder/decoder structure
    
    Forward pass:
        x: noisy input image [B, 3, H, W]
        latent: reference image latent [B, C, H', W']
        clip: CLIP text/image embedding [B, D]
        
    Returns:
        denoised image latent [B, D', H, W]
    """
    
    def __init__(self, config: UNetConfig):
        super().__init__()

        # Pose Encoder: transform pose
        self.pose_encoder = PoseEncoder(config.pose_encoder)

        # Encoder: downsample both paths
        self.encoder = nn.Sequential(
            *[UNetBlock.encoder(c) for c in config.encoder]
        )
        
        # Bottleneck: process at lowest resolution
        bneck_channels = config.encoder[-1].unet_conv.out_channels
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bneck_channels, bneck_channels,
                      kernel_size=config.bottleneck_kernel,
                      stride=1,
                      padding=(config.bottleneck_kernel - 1) // 2),
            nn.SiLU()
        )
        
        # Decoder: upsample with skip connections
        self.decoder = nn.Sequential(
            *[UNetBlock.decoder(c) for c in config.decoder]
        )


    def forward(self, noisy, pose, latent, clip):
        # Encode pose and combine with noisy input
        pose = self.pose_encoder(pose)
        x = pose + noisy

        # Encoder with skip collection
        skips = []
        for enc in self.encoder:
            x, latent = enc(x, latent, clip)
            skips.insert(0, x.clone())
        
        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip injection
        for dec, s in zip(self.decoder, skips):
            x = torch.cat([x, s], dim=1)  # Concatenate skip
            x, latent = dec(x, latent, clip)
        
        return x
