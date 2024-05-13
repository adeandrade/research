from typing import Optional, Tuple

import torch.nn.functional as functional
from torch import Tensor
from torch.nn import (
    ELU,
    Conv2d,
    ConvTranspose2d,
    Identity,
    LeakyReLU,
    Module,
    ReLU,
    Sequential,
    Tanh,
    Upsample,
)


class ResidualBottleneckBlock(Sequential):
    def __init__(
        self,
        num_channels: int,
        num_hidden_channels: Optional[int] = None,
        kernel_size: int = 3,
        activation: str = 'relu',
        groups: int = 1,
    ) -> None:
        num_hidden_channels = num_channels if num_hidden_channels is None else num_hidden_channels

        if activation == 'relu':
            activation_function = ReLU
        elif activation == 'elu':
            activation_function = ELU
        elif activation == 'leaky_relu':
            activation_function = LeakyReLU
        elif activation == 'tanh':
            activation_function = Tanh
        elif activation == 'identity':
            activation_function = Identity
        else:
            raise ValueError(f'Activation not supported {activation}')

        super().__init__(
            Conv2d(num_channels, num_hidden_channels, kernel_size=1, groups=groups),
            activation_function(),
            Conv2d(num_hidden_channels, num_hidden_channels, kernel_size=kernel_size, padding='same', groups=groups),
            activation_function(),
            Conv2d(num_hidden_channels, num_channels, kernel_size=1, groups=groups),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + super().forward(inputs)


class ResidualBottleneckBlockStack(Sequential):
    def __init__(
        self,
        num_channels: int,
        num_blocks: int = 3,
        num_hidden_channels: Optional[int] = None,
        kernel_size: int = 3,
        activation: str = 'relu',
        groups: int = 1,
    ) -> None:
        super().__init__(
            *(
                ResidualBottleneckBlock(num_channels, num_hidden_channels, kernel_size, activation, groups)
                for _ in range(num_blocks)
            )
        )


class DownSample(Module):
    def __init__(self, num_channels: int, factor: int, mode: str = 'bilinear', kernel_size: int = 5):
        super().__init__()

        self.factor = factor
        self.mode = mode

        self.layer = Conv2d(num_channels, num_channels, kernel_size, padding='same')

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = functional.interpolate(inputs, scale_factor=1 / self.factor, mode=self.mode)
        outputs = self.layer(outputs)

        return outputs


class UpSample(Sequential):
    def __init__(self, num_channels: int, factor: int, mode: str = 'nearest-exact', kernel_size: int = 5):
        super().__init__(
            Upsample(scale_factor=factor, mode=mode),
            Conv2d(num_channels, num_channels, kernel_size, padding='same'),
        )


def elic_lst_downsample(
    in_channels: int,
    out_channels: int,
    num_channels: Tuple[int, int, int] = (24, 48, 192),
    num_blocks: int = 3,
    activation: str = 'elu',
) -> Sequential:
    return Sequential(
        Conv2d(in_channels, num_channels[0], kernel_size=5, stride=2, padding=2),
        ResidualBottleneckBlockStack(num_channels[0], num_blocks=num_blocks, activation=activation),
        Conv2d(num_channels[0], num_channels[1], kernel_size=5, stride=2, padding=2),
        ResidualBottleneckBlockStack(num_channels[1], num_blocks=num_blocks, activation=activation),
        Conv2d(num_channels[1], num_channels[2], kernel_size=5, stride=2, padding=2),
        ResidualBottleneckBlockStack(num_channels[2], num_blocks=num_blocks, activation=activation),
        Conv2d(num_channels[2], out_channels, kernel_size=5, stride=2, padding=2),
    )


def elic_lst_upsample(
    in_channels: int,
    out_channels: int,
    num_channels: Tuple[int, int, int] = (192, 48, 24),
    num_blocks: int = 3,
    activation: str = 'elu',
    kernel_size: int = 5,
) -> Sequential:
    padding = kernel_size // 2

    return Sequential(
        ConvTranspose2d(in_channels, num_channels[0], kernel_size, stride=2, padding=padding, output_padding=1),
        ResidualBottleneckBlockStack(num_channels[0], num_blocks=num_blocks, activation=activation),
        ConvTranspose2d(num_channels[0], num_channels[1], kernel_size, stride=2, padding=padding, output_padding=1),
        ResidualBottleneckBlockStack(num_channels[1], num_blocks=num_blocks, activation=activation),
        ConvTranspose2d(num_channels[1], num_channels[2], kernel_size, stride=2, padding=padding, output_padding=1),
        ResidualBottleneckBlockStack(num_channels[2], num_blocks=num_blocks, activation=activation),
        ConvTranspose2d(num_channels[2], out_channels, kernel_size, stride=2, padding=padding, output_padding=1),
    )
