"""
encoder.depth
encoder.X.unet.kernel/stride
encoder.X.refnet.kernel/stride

decoder.depth
decoder.X.unet.kerernel/stride
decoder.X.refnet.kernel/stride
"""

def make_unet_config(encoder_kernels: Tuple[int], encoder_strides: Tuple[int],
                     decoder_kernels: Tuple[int], decoder_strides: Tuple[int],
                     bottleneck_kernel: int)