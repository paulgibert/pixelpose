import torch
from ...architecture.utils import (
    DownBlock,
    UpBlock,
    LazyMLP,
    LazyFiLMAdapter
)


class TestDownBlock:
    def test_forward_shape(self):
        """Test downsampling with normalization produces correct output shape."""
        block = DownBlock(32, 64,
                          kernel_size=3,
                          stride=2,
                          norm_groups=8)
        x = torch.randn(1, 32, 128, 128)
        y = block(x)
        assert y.shape == torch.Size((1, 64, 64, 64))

    def test_forward_shape_no_norm(self):
        """Test downsampling without normalization produces correct output shape."""
        block = DownBlock(32, 64,
                          kernel_size=3,
                          stride=2,
                          norm_groups=0)
        x = torch.randn(1, 32, 128, 128)
        y = block(x)
        assert y.shape == torch.Size((1, 64, 64, 64))


class TestUpBlock:
    def test_forward_shape_even_kernel(self):
        """Test upsampling with even kernel size produces correct output shape."""
        block = UpBlock(64, 32,
                        kernel_size=4,
                        stride=2,
                        norm_groups=8)
        x = torch.randn(1, 64, 128, 128)
        y = block(x)
        assert y.shape == torch.Size((1, 32, 256, 256))

    def test_forward_shape_odd_kernel(self):
        """Test upsampling with odd kernel size produces correct output shape."""
        block = UpBlock(64, 32,
                        kernel_size=3,
                        stride=2,
                        norm_groups=8)
        x = torch.randn(1, 64, 128, 128)
        y = block(x)
        assert y.shape == torch.Size((1, 32, 256, 256))

    def test_forward_shape_no_norm(self):
        """Test upsampling without normalization produces correct output shape."""
        block = UpBlock(64, 32,
                        kernel_size=3,
                        stride=2,
                        norm_groups=0)
        x = torch.randn(1, 64, 128, 128)
        y = block(x)
        assert y.shape == torch.Size((1, 32, 256, 256))


class TestLazyMLP:
    def test_forward_shape_with_hidden(self):
        """Test MLP with hidden layers produces correct output shape."""
        mlp = LazyMLP(32, (128, 64))
        x = torch.randn(1, 256)
        y = mlp(x)
        assert y.shape == torch.Size((1, 32))

    def test_forward_shape_no_hidden(self):
        """Test MLP without hidden layers produces correct output shape."""
        mlp = LazyMLP(32, ())
        x = torch.randn(1, 256)
        y = mlp(x)
        assert y.shape == torch.Size((1, 32))


class TestLazyFiLMAdapter:
    def test_forward_shape(self):
        """Test FiLM adapter preserves spatial dimensions while conditioning."""
        film = LazyFiLMAdapter(16, (16, 128))
        x = torch.randn(1, 16, 64, 64)
        features = torch.randn(1, 512)
        y = film(x, features)
        assert y.shape == torch.Size((1, 16, 64, 64))

    def test_conditioning_changes_output(self):
        """Test FiLM adapter actually modulates features based on conditioning."""
        film = LazyFiLMAdapter(16, (128,))
        x = torch.randn(1, 16, 32, 32)
        features1 = torch.randn(1, 512)
        features2 = torch.randn(1, 512)
        
        y1 = film(x, features1)
        y2 = film(x, features2)
        
        # Different conditioning should produce different outputs
        assert not torch.allclose(y1, y2)
