from typing import Tuple

import torch.nn.functional as functional
from compressai.layers.gdn import GDN
from torch import Tensor
from torch.nn import Conv2d, PixelShuffle, BatchNorm2d, Module, Linear, Sequential, LeakyReLU


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> Module:
    """
    1x1 convolution.
    """
    return Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(stride, stride))


def conv3x3(in_channels: int, out_channels: int, stride: int = 1, bias: bool = True) -> Module:
    """
    3x3 convolution with padding.
    """
    return Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(stride, stride), padding=1, bias=bias)


def subpel_conv3x3(in_channels: int, out_channels: int, multiplier: int = 2, bias: bool = True) -> Module:
    """
    3x3 sub-pixel convolution for up-sampling.
    """
    return Sequential(
        Conv2d(in_channels, out_channels * multiplier ** 2, kernel_size=(3, 3), padding=1, bias=bias),
        PixelShuffle(multiplier),
    )


class ResidualBlockUpsample(Module):
    """
    Residual block with sub-pixel upsampling on the last convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, upsample: int = 2) -> None:
        super().__init__()

        self.subpel_conv = subpel_conv3x3(in_channels, out_channels, upsample)
        self.conv = conv3x3(out_channels, out_channels)
        self.igdn = GDN(out_channels, inverse=True)
        self.upsample = subpel_conv3x3(in_channels, out_channels, upsample)

    def forward(self, x: Tensor) -> Tensor:
        out = self.subpel_conv(x)
        out = functional.leaky_relu(out)
        out = self.conv(out)
        out = self.igdn(out)
        out += self.upsample(x)

        return out


class ResidualBlockWithStride(Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super().__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.gdn = GDN(out_channels)

        self.skip = conv1x1(in_channels, out_channels, stride) if stride != 1 or in_channels != out_channels else None

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = functional.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)
        out += self.skip(x) if self.skip else x

        return out


class ResidualBlock(Module):
    """
    Simple residual block with two 3x3 convolutions.
    """
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = False) -> None:
        super().__init__()

        self.conv1 = conv3x3(in_channels, out_channels, bias=not normalize)
        self.conv2 = conv3x3(out_channels, out_channels, bias=not normalize)

        self.skip = conv1x1(in_channels, out_channels) if in_channels != out_channels else None
        self.batch_norm1 = BatchNorm2d(out_channels) if normalize else None
        self.batch_norm2 = BatchNorm2d(out_channels) if normalize else None

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.batch_norm1(out) if self.batch_norm1 else out
        out = functional.leaky_relu(out)
        out = self.conv2(out)
        out = self.batch_norm2(out) if self.batch_norm2 else out
        out = functional.leaky_relu(out)
        out += self.skip(x) if self.skip else x

        return out


def cheng_smaller_lst(in_channels: int, out_channels: int, num_channels: int = 192) -> Module:
    return Sequential(
        ResidualBlock(in_channels, num_channels),
        ResidualBlock(num_channels, num_channels),
        ResidualBlock(num_channels, num_channels),
        conv3x3(num_channels, out_channels),
    )


def anderson_lst_downsample(
        in_channels: int,
        out_channels: int,
        num_channels: Tuple[int, int, int, int] = (32, 96, 128, 192),
) -> Module:

    return Sequential(
        conv3x3(in_channels, num_channels[0]),
        ResidualBlockWithStride(num_channels[0], num_channels[0]),
        ResidualBlock(num_channels[0], num_channels[1]),
        ResidualBlockWithStride(num_channels[1], num_channels[1]),
        ResidualBlock(num_channels[1], num_channels[2]),
        ResidualBlockWithStride(num_channels[2], num_channels[2]),
        ResidualBlock(num_channels[2], num_channels[3]),
        ResidualBlockWithStride(num_channels[3], num_channels[3]),
        conv3x3(num_channels[3], out_channels),
    )


def anderson_lst_upsample(
        in_channels: int,
        out_channels: int,
        num_channels: Tuple[int, int, int, int] = (192, 128, 96, 32),
) -> Module:

    return Sequential(
        conv3x3(in_channels, num_channels[0]),
        LeakyReLU(),
        subpel_conv3x3(num_channels[0], num_channels[0]),
        LeakyReLU(),
        subpel_conv3x3(num_channels[0], num_channels[1]),
        LeakyReLU(),
        subpel_conv3x3(num_channels[1], num_channels[2]),
        LeakyReLU(),
        subpel_conv3x3(num_channels[2], num_channels[3]),
        LeakyReLU(),
        conv3x3(num_channels[3], out_channels),
    )


def transformer(in_channels: int, out_channels: int, num_channels: int) -> Module:
    return Sequential(
        anderson_lst_downsample(in_channels, num_channels),
        cheng_smaller_lst(num_channels, num_channels, out_channels),
    )


def kernel_encoder(
        in_features: int,
        num_features: int = 512,
) -> Module:

    return Sequential(
        Linear(in_features, num_features),
        LeakyReLU(),
        Linear(num_features, num_features),
        LeakyReLU(),
        Linear(num_features, num_features),
        LeakyReLU(),
        Linear(num_features, 1),
    )
