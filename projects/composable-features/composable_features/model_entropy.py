import math
from typing import Tuple, Optional, List

import torch
from compressai.ans import RansEncoder, RansDecoder
from compressai.entropy_models import GaussianConditional
from compressai.layers import MaskedConv2d
from torch import Tensor
from torch.nn import Module, Sequential, Conv2d, LeakyReLU

import composable_features.functions as functions


def get_scale_table(minimum: float = 0.11, maximum: float = 256., levels: int = 64):
    return torch.exp(torch.linspace(math.log(minimum), math.log(maximum), levels))


class AutoregressiveEntropyModel(Module):
    def __init__(
            self,
            num_channels: int,
            num_channels_per_factor: int = 2,
            kernel_size: int = 5,
            factorized: bool = False,
    ) -> None:

        super().__init__()

        self.num_channels = num_channels
        self.num_channels_factor = num_channels_per_factor
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

        self.entropy_parameters = Sequential(
            MaskedConv2d(
                in_channels=num_channels,
                out_channels=num_channels_per_factor * num_channels,
                kernel_size=(kernel_size, kernel_size),
                padding=(self.padding, self.padding),
                groups=num_channels if factorized else 1,
            ),
            Conv2d(
                in_channels=num_channels_per_factor * num_channels,
                out_channels=num_channels_per_factor * num_channels,
                kernel_size=(1, 1),
                groups=num_channels if factorized else 1,
            ),
            LeakyReLU(),
            Conv2d(
                in_channels=num_channels_per_factor * num_channels,
                out_channels=num_channels_per_factor * num_channels,
                kernel_size=(1, 1),
                groups=num_channels if factorized else 1,
            ),
            LeakyReLU(),
            Conv2d(
                in_channels=num_channels_per_factor * num_channels,
                out_channels=2 * num_channels,
                kernel_size=(1, 1),
                groups=num_channels if factorized else 1,
            ),
        )

        self.gaussian_conditional = GaussianConditional(scale_table=None)

    @staticmethod
    def calculate_standardized_cumulative(inputs: Tensor) -> Tensor:
        """
        Using the complementary error function maximizes numerical precision.
        """
        return .5 * torch.erfc(-(2 ** -.5) * inputs)

    def calculate_likelihoods(self, inputs: Tensor, scales: Tensor, means: Optional[Tensor] = None) -> Tensor:
        values = inputs - means if means is not None else inputs
        values = functions.perturb(values) if self.training else functions.quantize(values)

        scales = self.gaussian_conditional.lower_bound_scale(scales)

        values = torch.abs(values)
        upper = self.calculate_standardized_cumulative((.5 - values) / scales)
        lower = self.calculate_standardized_cumulative((-.5 - values) / scales)
        likelihoods = upper - lower

        if self.gaussian_conditional.use_likelihood_bound:
            likelihoods = self.gaussian_conditional.likelihood_lower_bound(likelihoods)

        return likelihoods

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        inputs_quantized = functions.perturb(inputs) if self.training else functions.quantize(inputs)

        parameters = self.entropy_parameters(inputs_quantized)
        means, scales = torch.chunk(parameters, chunks=2, dim=1)

        likelihoods = self.calculate_likelihoods(inputs_quantized, scales, means)

        return inputs_quantized, likelihoods

    def update(self, force: bool = False) -> None:
        scale_table = self.gaussian_conditional.scale_table
        scale_table = get_scale_table() if len(scale_table) == 0 else scale_table

        self.gaussian_conditional.update_scale_table(scale_table, force)

    def compress(self, batch: Tensor) -> List[str]:
        self.update()

        batch_size, _, _, _ = batch.shape

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        batch = torch.round(batch)

        parameters = self.entropy_parameters(batch)
        means, scales = torch.chunk(parameters, chunks=2, dim=1)

        indices = self.gaussian_conditional.build_indexes(scales)
        indices = torch.permute(indices, dims=(0, 2, 3, 1))
        indices = torch.reshape(indices, shape=(batch_size, -1))

        symbols = self.gaussian_conditional.quantize(batch, 'symbols', means)
        symbols = torch.permute(symbols, dims=(0, 2, 3, 1))
        symbols = torch.reshape(symbols, shape=(batch_size, -1))

        encoder = RansEncoder()
        strings = [
            encoder.encode_with_indexes(symbols[index].tolist(), indices[index].tolist(), cdf, cdf_lengths, offsets)
            for index in range(batch_size)
        ]

        return strings

    def decompress(self, strings: List[str], shape: Tuple[int, int, int]) -> Tensor:
        self.update()

        batch_size = len(strings)
        num_channels, height, width = shape
        device = self.entropy_parameters[0].weight.device

        batch = torch.zeros(batch_size, num_channels, height, width, dtype=torch.float32, device=device)

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()

        for batch_index, string in enumerate(strings):
            decoder.set_stream(string)

            for h in range(height):
                for w in range(width):
                    parameters = batch[batch_index]
                    parameters = self.entropy_parameters(parameters)
                    parameters = parameters[:, h, w]

                    means, scales = torch.chunk(parameters, chunks=2, dim=0)

                    indices = self.gaussian_conditional.build_indexes(scales)
                    indices = indices.tolist()

                    symbols = decoder.decode_stream(indices, cdf, cdf_lengths, offsets)
                    symbols = torch.tensor(symbols, dtype=torch.int32)

                    value = self.gaussian_conditional.dequantize(symbols, means)
                    value = torch.round(value)

                    batch[batch_index, :, h, w] = value

        return batch
