from typing import List, Optional, Tuple, Union

import torch
from compressai.entropy_models import GaussianConditional
from compressai.layers import MaskedConv2d
from torch import Tensor
from torch.nn import ELU, Conv2d, LeakyReLU, Module, Sequential

import compatible_representations.functions as functions
from compatible_representations.model_lst import ResidualBottleneckBlockStack


class GaussianConditionalQuantized(GaussianConditional):
    def __init__(self, scale_table: Optional[Union[List, Tuple]] = None, *args, **kwargs):
        super().__init__(scale_table, **kwargs)

    def _likelihood(self, inputs: Tensor, scales: Tensor, means: Tensor) -> Tensor:
        values = functions.quantize(inputs - means)

        scales = self.lower_bound_scale(scales)

        values = torch.abs(values)
        upper = self._standardized_cumulative((0.5 - values) / scales)
        lower = self._standardized_cumulative((-0.5 - values) / scales)
        likelihood = upper - lower

        return likelihood

    def forward(self, inputs: Tensor, scales: Tensor, means: Tensor) -> Tuple[Tensor, Tensor]:
        likelihood = self._likelihood(inputs, scales, means)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return inputs, likelihood


class GaussianEntropyModel(Module):
    def __init__(self, num_channels: int) -> None:
        super().__init__()

        self.entropy_parameters = Sequential(
            Conv2d(num_channels * 12 // 6, num_channels * 10 // 3, 1),
            LeakyReLU(inplace=True),
            Conv2d(num_channels * 10 // 3, num_channels * 8 // 3, 1),
            LeakyReLU(inplace=True),
            Conv2d(num_channels * 8 // 3, num_channels * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(num_channels, 2 * num_channels, kernel_size=5, padding=2, stride=1)

        self.gaussian_conditional = GaussianConditionalQuantized()

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        inputs_quantized = functions.quantize(inputs)

        ctx_params = self.context_prediction(inputs_quantized)
        gaussian_params = self.entropy_parameters(ctx_params)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, likelihoods = self.gaussian_conditional(inputs_quantized, scales_hat, means=means_hat)

        return inputs_quantized, likelihoods


class GaussianConditionalEntropyModel(Module):
    def __init__(self, num_channels: int, num_channels_conditional: int) -> None:
        super().__init__()

        self.entropy_parameters = Sequential(
            Conv2d(num_channels * 12 // 3, num_channels * 10 // 3, 1),
            ELU(inplace=True),
            Conv2d(num_channels * 10 // 3, num_channels * 8 // 3, 1),
            ELU(inplace=True),
            Conv2d(num_channels * 8 // 3, num_channels * 6 // 3, 1),
        )

        self.conditional_prediction = Sequential(
            ResidualBottleneckBlockStack(
                num_channels=num_channels_conditional,
                num_blocks=2,
                kernel_size=5,
                activation='elu',
            ),
            Conv2d(
                in_channels=num_channels_conditional,
                out_channels=2 * num_channels,
                kernel_size=5,
                padding='same',
            ),
            ResidualBottleneckBlockStack(
                num_channels=2 * num_channels,
                num_blocks=2,
                kernel_size=5,
                activation='elu',
            ),
        )

        self.context_prediction = MaskedConv2d(num_channels, 2 * num_channels, kernel_size=5, padding=2, stride=1)

        self.gaussian_conditional = GaussianConditionalQuantized()

    def forward(self, inputs: Tensor, conditional: Tensor) -> Tuple[Tensor, Tensor]:
        inputs_quantized = functions.quantize(inputs)

        hyper_params = self.conditional_prediction(conditional)
        context_params = self.context_prediction(inputs_quantized)
        gaussian_params = self.entropy_parameters(torch.concat((hyper_params, context_params), dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        _, likelihoods = self.gaussian_conditional(inputs_quantized, scales_hat, means=means_hat)

        return inputs_quantized, likelihoods
