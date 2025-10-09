import torch
import numpy as np
from PIL import Image
from ...architecture.preprocess import image_to_tensor, Preprocessor


def _make_dummy_image(width: int, height: int) -> Image:
    """Create dummy PIL Image for testing without disk I/O."""
    array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(array)


def test_image_to_tensor_shape():
    """Test image to tensor conversion produces correct shape and range."""
    image = _make_dummy_image(128, 128)
    x = image_to_tensor(image)
    assert x.shape == torch.Size((1, 3, 128, 128))
    assert x.min() >= -1.0 and x.max() <= 1.0  # Check range


class TestPreprocessor:
    def test_preprocess_shape(self):
        """Test preprocessor produces correct VAE latent and CLIP feature shapes."""
        pre = Preprocessor()
        image = _make_dummy_image(256, 256)
        latent, features = pre.preprocess(image)
        
        # VAE downsamples by 8x
        assert latent.shape == torch.Size((1, 16, 32, 32))
        
        # CLIP always outputs 512-dim features
        assert features.shape == torch.Size((1, 512))
    
    def test_clip_features_normalized(self):
        """Test CLIP features are L2 normalized."""
        pre = Preprocessor()
        image = _make_dummy_image(256, 256)
        _, features = pre.preprocess(image)
        
        # L2 norm should be 1.0 for normalized features
        norm = features.norm(dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)
