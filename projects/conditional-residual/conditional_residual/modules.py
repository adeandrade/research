from typing import Tuple

from torch import Tensor
from torch.nn import Sequential, Conv2d, ConvTranspose2d, Identity, LeakyReLU, ReLU, Tanh


class ResidualBottleneckBlock(Sequential):
    def __init__(self, num_channels: int, kernel_size: int = 3, activation: str = 'relu') -> None:
        if activation == 'relu':
            activation_function = ReLU
        elif activation == 'leaky_relu':
            activation_function = LeakyReLU
        elif activation == 'tanh':
            activation_function = Tanh
        elif activation == 'identity':
            activation_function = Identity
        else:
            raise ValueError(f'Activation not supported {activation}')

        super().__init__(
            Conv2d(num_channels, num_channels, kernel_size=1),
            activation_function(),
            Conv2d(num_channels, num_channels, kernel_size=kernel_size, padding='same'),
            activation_function(),
            Conv2d(num_channels, num_channels, kernel_size=1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + super().forward(inputs)


class ResidualBottleneckBlockStack(Sequential):
    def __init__(self, num_channels: int, num_blocks: int = 3, kernel_size: int = 3, activation: str = 'relu') -> None:
        super().__init__(
            *(
                ResidualBottleneckBlock(
                    num_channels=num_channels,
                    kernel_size=kernel_size,
                    activation=activation,
                )
                for _ in range(num_blocks)
            )
        )


def elic_lst_downscale(
        in_channels: int,
        out_channels: int,
        num_channels: Tuple[int, int, int] = (24, 48, 192),
        num_blocks: int = 3,
        activation: str = 'relu',
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


def elic_lst_upscale(
        in_channels: int,
        out_channels: int,
        num_channels: Tuple[int, int, int] = (192, 48, 24),
        num_blocks: int = 3,
        activation: str = 'relu',
) -> Sequential:

    return Sequential(
        ConvTranspose2d(in_channels, num_channels[0], kernel_size=5, stride=2, padding=2, output_padding=1),
        ResidualBottleneckBlockStack(num_channels[0], num_blocks=num_blocks, activation=activation),
        ConvTranspose2d(num_channels[0], num_channels[1], kernel_size=5, stride=2, padding=2, output_padding=1),
        ResidualBottleneckBlockStack(num_channels[1], num_blocks=num_blocks, activation=activation),
        ConvTranspose2d(num_channels[1], num_channels[2], kernel_size=5, stride=2, padding=2, output_padding=1),
        ResidualBottleneckBlockStack(num_channels[2], num_blocks=num_blocks, activation=activation),
        ConvTranspose2d(num_channels[2], out_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
    )
