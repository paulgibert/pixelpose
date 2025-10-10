from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image


def load_png(path) -> Image:
    """
    Loads an PNG image replacing the alpha background with a neutral gray.
    Result is returned in RGB
    """
    image = Image.open(path).convert('RGBA')
    bg = Image.new('RGB', image.size, (128, 128, 128))
    image_rgb = Image.alpha_composite(bg.convert('RGBA'), image).convert('RGB')
    return image_rgb


def image_to_tensor(image: Image) -> torch.Tensor:
    """Converts an RGB image to a Tensor"""
    transform = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda t: 2.0 * t - 1.0)
    ])
    x = transform(image)
    return x.unsqueeze(0)


def tensor_to_image(tensor):
    """Convert tensor to PIL Image"""
    # If batch, take first image
    if tensor.ndim == 4:
        tensor = tensor[0]
    
    # If values are in [-1, 1], scale to [0, 1]
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2
    
    # Clamp to [0, 1]
    tensor = tensor.clamp(0, 1)
    
    # Convert to PIL
    return to_pil_image(tensor.cpu())