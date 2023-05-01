import abc
import functools
from abc import ABC
from typing import List, Tuple, Callable, Optional

import torch
from torch import Tensor
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, Parameter, LeakyReLU, ReLU, Hardswish, Hardsigmoid
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation

import coding.models.functions as functions


class EntropyModel(ABC):
    @abc.abstractmethod
    def __call__(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        ...

    @abc.abstractmethod
    def encode(self, batch: Tensor) -> List[bytes]:
        ...

    @abc.abstractmethod
    def decode(self, byte_strings: List[bytes], shape: Tuple[int, int, int]) -> Tensor:
        ...

    @abc.abstractmethod
    def calculate_sizes(self, batch: Tensor) -> List[int]:
        pass


class ConditionalEntropyModel(ABC):
    @abc.abstractmethod
    def __call__(self, inputs: Tensor, conditionals: Tensor) -> Tuple[Tensor, Tensor]:
        ...

    @abc.abstractmethod
    def encode(self, batch: Tensor, conditionals: Tensor) -> List[bytes]:
        ...

    @abc.abstractmethod
    def decode(self, byte_strings: List[bytes], shape: Tuple[int, int, int], conditionals: Tensor) -> Tensor:
        ...

    @abc.abstractmethod
    def calculate_sizes(self, batch: Tensor, conditionals: Tensor) -> List[int]:
        pass


class MaskedConv2d(Conv2d):
    mask: Tensor

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            *args,
            in_group_size: int = 1,
            out_group_size: int = 1,
            in_prefix_size: int = 0,
            out_prefix_size: int = 0,
            order: int = -1,
            include_target: bool = False,
            reverse: bool = False,
            **kwargs,
    ) -> None:

        assert 'groups' not in kwargs
        assert 'padding' not in kwargs
        assert 'transpose' not in kwargs
        assert out_channels % in_channels == 0 or in_channels % out_channels == 0

        groups = in_channels // in_group_size if order == 0 else 1

        super().__init__(
            in_channels=in_prefix_size + in_channels,
            out_channels=out_prefix_size + out_channels,
            *args,
            **kwargs,
            padding='same',
            groups=groups,
        )

        self.in_group_size = in_group_size
        self.out_group_size = out_group_size
        self.in_prefix_size = in_prefix_size
        self.out_prefix_size = out_prefix_size
        self.order = order
        self.include_target = include_target
        self.reverse = reverse

        _, _, kernel_height, kernel_width = self.weight.data.size()
        padding_height = kernel_height // 2
        padding_width = kernel_width // 2

        mask = functions.create_triangular_mask(
            shape=self.weight.data.shape,
            padding_height=padding_height,
            padding_width=padding_width,
            in_group_size=in_group_size,
            out_group_size=out_group_size,
            in_prefix_size=in_prefix_size,
            out_prefix_size=out_prefix_size,
            order=order,
            include_target=include_target,
            reverse=reverse,
            dtype=kwargs.get('dtype', torch.float32),
            device=kwargs.get('device'),
        )

        self.register_buffer('mask', mask)

    def forward(self, x: Tensor) -> Tensor:
        self.weight.data *= self.mask
        output = super().forward(x)

        return output


class MaskedBlock(Module):
    def __init__(
            self,
            num_channels: int,
            multiplier: int,
            kernel_size: int,
            group_size: int = 1,
            prefix_size: int = 0,
            order: int = -1,
            scale: bool = True,
            dtype: torch.dtype = torch.float32,
            device: Optional[torch.device] = None,
    ) -> None:

        super().__init__()

        self.num_channels = num_channels
        self.multiplier = multiplier
        self.kernel_size = kernel_size
        self.group_size = group_size
        self.prefix_size = prefix_size
        self.order = order
        self.scale = scale

        factory_kwargs = {'device': device, 'dtype': dtype}

        self.block = Sequential(
            MaskedConv2d(
                in_channels=num_channels,
                out_channels=multiplier * num_channels,
                kernel_size=(kernel_size, kernel_size),
                in_group_size=group_size,
                out_group_size=multiplier * group_size,
                in_prefix_size=prefix_size,
                out_prefix_size=prefix_size,
                order=order,
                include_target=True,
                **factory_kwargs,
            ),
            LeakyReLU(),
            MaskedConv2d(
                in_channels=multiplier * num_channels,
                out_channels=multiplier * num_channels,
                kernel_size=(kernel_size, kernel_size),
                in_group_size=multiplier * group_size,
                out_group_size=multiplier * group_size,
                in_prefix_size=prefix_size,
                out_prefix_size=prefix_size,
                order=order,
                include_target=True,
                **factory_kwargs,
            ),
            LeakyReLU(),
            MaskedConv2d(
                in_channels=multiplier * num_channels,
                out_channels=num_channels,
                kernel_size=(kernel_size, kernel_size),
                in_group_size=multiplier * group_size,
                out_group_size=group_size,
                in_prefix_size=prefix_size,
                out_prefix_size=prefix_size,
                order=order,
                include_target=True,
                **factory_kwargs,
            ),
        )

        self.scales = Parameter(torch.ones(prefix_size + num_channels, **factory_kwargs)) if self.scale else None

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.block(inputs)

        if self.scales is not None:
            outputs = self.scales[None, :, None, None] * inputs + outputs

        return outputs


class InvertedResidual(Module):
    """
    Implemented as described at section 5 of the MobileNetV3 paper
    """
    def __init__(
            self,
            in_channels: int,
            expanded_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            dilation: int,
            use_se: bool,
            use_hs: bool,
            norm_layer: Optional[Callable[..., Module]] = None,
            se_layer: Callable[..., Module] = functools.partial(SqueezeExcitation, scale_activation=Hardsigmoid),
    ) -> None:

        super().__init__()

        assert 1 <= stride <= 2, ValueError('Illegal stride value')

        self.out_channels = out_channels
        self.use_res_connect = stride == 1 and in_channels == out_channels

        activation_layer = Hardswish if use_hs else ReLU
        layers: List[Module] = []

        # expand
        if expanded_channels != in_channels:
            layers.append(
                Conv2dNormActivation(
                    in_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depth-wise
        stride = 1 if dilation > 1 else stride

        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        if use_se:
            squeeze_channels = functions.make_divisible(expanded_channels // 4, 8)
            layers.append(se_layer(expanded_channels, squeeze_channels))

        # project
        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None,
            )
        )

        self.block = Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        result = self.block(inputs)

        if self.use_res_connect:
            result += inputs

        return result


def inverted_residual_lst(
        in_channels: int,
        out_channels: int,
        expansion_channels: int = 192,
        num_layers: int = 2,
        norm_layer: Optional[Callable[..., Module]] = BatchNorm2d,
) -> Sequential:

    return Sequential(
        *[
            InvertedResidual(in_channels, expansion_channels, in_channels, 5, 1, 1, False, False, norm_layer)
            for _ in range(num_layers)
        ],
        InvertedResidual(in_channels, expansion_channels, out_channels, 5, 1, 1, False, False, norm_layer),
        *[
            InvertedResidual(out_channels, expansion_channels, out_channels, 5, 1, 1, False, False, norm_layer)
            for _ in range(num_layers)
        ],
    )
