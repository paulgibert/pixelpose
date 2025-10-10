from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .architecture.preprocess import Preprocessor
from .architecture.denoise import UNetBlockConfig, UNetDenoiser, UNetConfig, ConvConfig
from .dataset import load_and_split_preprocessed_dataset


REF_CHANNELS = [(16, 32), (32, 64), (64, 128), (128, 256), (256, 512), (512, 1024)]
UNET_CHANNELS = [(3, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256)]
MAX_DEPTH = len(REF_CHANNELS)


class NoiseScheduler:
    def __init__(self, total_steps: int):
        if total_steps <= 0:
            raise ValueError(f"total_steps must be > 0. Got {total_steps}")
        self.total_steps = total_steps
        
        betas = torch.linspace(1e-4, 0.02, total_steps)
        alphas = 1.0 - betas
        self._alphas_cumprod = torch.cumprod(alphas, dim=0)
        
    def add_noise_at_random(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = x.shape[0]
        steps = torch.randint(0, self.total_steps, (batch_size,))
        eps = torch.randn_like(x)
        a_t = self._alphas_cumprod[steps].view(batch_size, 1, 1, 1)
        noisy = torch.sqrt(a_t) * x + torch.sqrt(1 - a_t) * eps
        return noisy, eps
        

@dataclass
class TrainerConfig:
    lr: float
    batch_size: int
    total_steps: int
    denoiser_depth: int
    unet_kernels: int
    unet_strides: int
    ref_kernels: int
    ref_strides: int
    preprocess_dataset_dir: Path


def make_unet_config(config: TrainerConfig) -> UNetConfig:
    """Create complete UNet config"""
    depth = depth
    
    if not 1 <= depth <= MAX_DEPTH:
        raise ValueError(f"depth must be between 1 and {MAX_DEPTH}, got {depth}")
    
    ref_kernel = config.ref_kernels
    ref_stride = config.ref_strides

    unet_kernel = config.unet_kernels
    unet_stride = config.unet_strides

    # Pose encoder:
    pose_encoder = (
        ConvConfig(3, 16, 3, 1, 4),
        ConvConfig(16, 3, 3, 1, 4)
    )

    # Encoder: forward order
    encoder = tuple([
        UNetBlockConfig(
            ref_conv=ConvConfig(ref_in, ref_out, ref_kernel, ref_stride, 4),
            unet_conv=ConvConfig(unet_in, unet_out, unet_kernel, unet_stride, 4),
            clip_dims=(64,),
            ref_dims=(64,)
        ) 
        for (ref_in, ref_out), (unet_in, unet_out) 
        in zip(REF_CHANNELS[:depth], UNET_CHANNELS[:depth])
    ])
    
    # Decoder: reversed order with swapped in/out
    decoder = tuple([
        UNetBlockConfig(
            ref_conv=ConvConfig(ref_out, ref_in, ref_kernel, ref_stride, 4),
            unet_conv=ConvConfig(unet_out, unet_in, unet_kernel, unet_stride, 4),
            clip_dims=(64,),
            ref_dims=(64,)
        ) 
        for (ref_in, ref_out), (unet_in, unet_out) 
        in zip(reversed(REF_CHANNELS[:depth]), reversed(UNET_CHANNELS[:depth]))
    ])
    
    return UNetConfig(
        pose_encoder=pose_encoder,
        encoder=encoder,
        decoder=decoder,
        bottleneck_kernel=3,
    )


class Trainer:
    def __init__(self, config: TrainerConfig):
        self.prep = Preprocessor()
        self.model = UNetDenoiser(make_unet_config(config))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = NoiseScheduler(config.total_steps)
        self.batch_size = config.batch_size
        self.train_dl, self.eval_dl = load_and_split_preprocessed_dataset(config.preprocess_dataset_dir)

    def _optimize_one_epoch(self, dl: DataLoader, is_eval: bool) -> float:
        if is_eval:
            self.model.eval()
        else:
            self.model.train()
        
        total_loss = 0.0
        total_batches = 0

        for batch in dl:
            # Prepare inputs
            latent = batch['vae']
            clip = batch['clip']
            pose = batch['pose']

            # Prepare target
            target = batch['target']
            x, eps = self.scheduler.add_noise_at_random(target)

            # Forward
            pred_eps = self.model(x, pose, latent, clip)
            
            # Loss
            loss = F.mse_loss(pred_eps, eps)
            
            # Backprop
            if not is_eval:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

            # Log loss
            total_loss += loss.item()
            total_batches += 1

        return total_loss / total_batches   

    def train_one_epoch(self) -> float:
        return self._optimize_one_epoch(self.train_dl, False)

    def evaluate_one_epoch(self) -> float:
        return self._optimize_one_epoch(self.eval_dl, True)
