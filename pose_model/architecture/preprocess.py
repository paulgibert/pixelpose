"""Image preprocessing for VAE encoding and CLIP feature extraction."""

from typing import Tuple
from PIL import Image
import torch
from torch import Tensor
import torchvision.transforms as T
from diffusers import AutoencoderKL
import open_clip


VAE_MODEL_PATH = 'stabilityai/stable-diffusion-3.5-large'
CLIP_MODEL_PATH = 'hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K'


def image_to_tensor(image: Image) -> torch.Tensor:
    """Convert PIL Image to normalized tensor in range [-1, 1].
    
    Args:
        image: RGB PIL Image
        
    Returns:
        Tensor with shape [1, 3, H, W] in range [-1, 1]
    """
    transform = T.Compose([
        T.ToTensor(),  # [0, 1]
        T.Lambda(lambda t: 2.0 * t - 1.0)  # [-1, 1]
    ])
    x = transform(image)
    return x.unsqueeze(0)


class Preprocessor():
    """Encodes images into VAE latents and CLIP features.
    
    Uses Stable Diffusion 3.5 VAE for latent encoding and CLIP ViT-B/32
    for semantic feature extraction.
    """
    
    def __init__(self, vae_path: str = VAE_MODEL_PATH, clip_path: str = CLIP_MODEL_PATH):
        self.vae = AutoencoderKL.from_pretrained(vae_path, subfolder='vae')
        self.clip, self.clip_pre = open_clip.create_model_from_pretrained(clip_path)
    
    def preprocess(self, image: Image) -> Tuple[Tensor, Tensor]:
        """Encode image into VAE latent and CLIP features.
        
        Args:
            image: RGB PIL Image
            
        Returns:
            latent: VAE encoded latent [1, C, H/8, W/8]
            features: Normalized CLIP features [1, 512]
        """
        # Encode with VAE
        x = image_to_tensor(image)
        latent = self.vae.encode(x).latent_dist.sample()
        latent *= self.vae.config.scaling_factor
        
        # Extract CLIP features
        pre = self.clip_pre(image).unsqueeze(0)
        features = self.clip.encode_image(pre)
        features /= features.norm(dim=-1, keepdim=True)  # L2 normalize
        
        return latent, features
