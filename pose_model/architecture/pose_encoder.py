from typing import List
from torch import nn
from convolution import ConvBlockArgs, UpBlock

# in_channels >= out_channels
# kernel_size < 64
# stride <= kernel_size

# TODO: Validate args
class PoseEncoder(nn.Module):
    def __init__(self, args_list: List[ConvBlockArgs]):
        super().__init__()
        self.blocks = nn.Sequential(
            *[UpBlock(args) for args in args_list]
        )

    def forward(self, x):
        return self.blocks(x)
