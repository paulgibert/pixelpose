import torch
from ...architecture.denoise import (
    PoseEncoder,
    ReferenceBlock,
    UNetBlock,
    ConvConfig,
    UNetBlockConfig,
    UNetConfig,
    UNetDenoiser
)


class TestReferenceBlock:
    def test_encoder_forward_shape(self):
        """Test reference encoder block produces correct output shape."""
        conv = ConvConfig(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=2,
            norm_groups=4
        )
        block = ReferenceBlock.encoder(conv, (128,))
        x = torch.randn(1, 16, 64, 64)
        clip = torch.randn(1, 512)
        y = block(x, clip)
        assert y.shape == torch.Size((1, 32, 32, 32))

    def test_decoder_forward_shape(self):
        """Test reference decoder block produces correct output shape."""
        conv = ConvConfig(
            in_channels=16,
            out_channels=8,
            kernel_size=3,
            stride=2,
            norm_groups=4
        )
        block = ReferenceBlock.decoder(conv, (128,))
        x = torch.randn(1, 16, 64, 64)
        clip = torch.randn(1, 512)
        y = block(x, clip)
        assert y.shape == torch.Size((1, 8, 128, 128))


class TestPoseEncoder:
    def test_pose_encoder_forward_shape(self):
        """Test pose encoder produces correct output shape."""
        conv_list = (
            ConvConfig(3, 32, 5, 1, 4),
            ConvConfig(32, 3, 3, 1, 0)
        )
        encoder = PoseEncoder(conv_list)
        x = torch.randn(1, 3, 128, 128)
        y = encoder(x)
        assert y.shape == torch.Size((1, 3, 128, 128))


class TestUNetBlock:
    def test_encoder_forward_shape(self):
        """Test UNet encoder block processes both paths correctly."""
        ref_conv = ConvConfig(
            in_channels=64,
            out_channels=32,
            kernel_size=5,
            stride=2,
            norm_groups=4
        )
        unet_conv = ConvConfig(
            in_channels=3,
            out_channels=16,
            kernel_size=6,
            stride=2,
            norm_groups=4
        )
        config = UNetBlockConfig(
            ref_conv=ref_conv,
            unet_conv=unet_conv,
            clip_dims=(64, 256),
            ref_dims=()
        )
        
        block = UNetBlock.encoder(config)
        
        x = torch.randn((1, 3, 128, 128))
        latent = torch.randn((1, 64, 16, 16))
        clip = torch.randn(1, 512)
        
        y, next_latent = block(x, latent, clip)
        assert y.shape == torch.Size([1, 16, 64, 64])
        assert next_latent.shape == torch.Size([1, 32, 8, 8])

    def test_decoder_forward_shape(self):
        """Test UNet decoder block processes both paths correctly."""
        ref_conv = ConvConfig(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            stride=2,
            norm_groups=0
        )
        unet_conv = ConvConfig(
            in_channels=16,
            out_channels=3,
            kernel_size=6,
            stride=2,
            norm_groups=3
        )
        config = UNetBlockConfig(
            ref_conv=ref_conv,
            unet_conv=unet_conv,
            clip_dims=(64, 256),
            ref_dims=()
        )
        
        block = UNetBlock.decoder(config)
        
        x = torch.randn((1, 16, 64, 64))
        latent = torch.randn((1, 32, 8, 8))
        clip = torch.randn(1, 512)
        
        y, next_latent = block(x, latent, clip)
        assert y.shape == torch.Size([1, 3, 128, 128])
        assert next_latent.shape == torch.Size([1, 64, 16, 16])


class TestUNetDenoiser:
    def test_forward_shape(self):
        """Test full UNet denoiser produces correct output shape."""
        # Pose encoder configuration
        pose_encoder_config = (
            ConvConfig(3, 16, 3, 1, 0),
            ConvConfig(16, 3, 3, 1, 0)
        )
        
        # Encoder configuration
        encoder_config = (
            UNetBlockConfig(
                ref_conv=ConvConfig(16, 32, 3, 2, 4),
                unet_conv=ConvConfig(3, 8, 3, 2, 4),
                clip_dims=(64,),
                ref_dims=(64,)
            ),
            UNetBlockConfig(
                ref_conv=ConvConfig(32, 64, 3, 2, 4),
                unet_conv=ConvConfig(8, 16, 3, 2, 4),
                clip_dims=(64,),
                ref_dims=(64,)
            )
        )
        
        # Decoder configuration
        decoder_config = (
            UNetBlockConfig(
                ref_conv=ConvConfig(64, 32, 3, 2, 4),
                unet_conv=ConvConfig(32, 8, 3, 2, 4),
                clip_dims=(64,),
                ref_dims=(64,)
            ),
            UNetBlockConfig(
                ref_conv=ConvConfig(32, 16, 3, 2, 4),
                unet_conv=ConvConfig(16, 4, 3, 2, 4),
                clip_dims=(64,),
                ref_dims=(64,)
            )
        )
        
        config = UNetConfig(
            pose_encoder=pose_encoder_config,
            encoder=encoder_config,
            decoder=decoder_config,
            bottleneck_kernel=5
        )
        
        unet = UNetDenoiser(config)
        
        noisy = torch.randn((1, 3, 128, 128))
        pose = torch.randn((1, 3, 128, 128))
        latent = torch.randn((1, 16, 16, 16))
        clip = torch.randn((1, 512))
        
        y = unet(noisy, pose, latent, clip)
        assert y.shape == torch.Size((1, 3, 128, 128))
