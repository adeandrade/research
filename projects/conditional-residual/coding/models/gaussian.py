from typing import Tuple, List

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module, Sequential, Conv2d

import coding.conversion as conversion
import coding.frameworks.grouped as grouped
import coding.models.functions as functions
import coding.models.modules as modules
import coding.normal as normal
from coding.autoregression import NetworkParameters
from coding.models.modules import EntropyModel, ConditionalEntropyModel, MaskedConv2d, MaskedBlock
from coding.normal import SCALE_MIN, TAIL_MASS


class GaussianEntropyModel(Module, EntropyModel):
    def __init__(
        self,
        num_channels: int,
        num_layers: int = 5,
        multiplier: int = 1,
        kernel_size: int = 5,
        in_group_size: int = 1,
        pre_group_size: int = 1,
        prefix_size: int = 0,
        order: int = -1,
        scale_lower_bound: float = SCALE_MIN,
        tail_mass: float = TAIL_MASS,
    ) -> None:
        assert num_channels % pre_group_size == 0

        super().__init__()

        self.num_channels = num_channels
        self.num_layers = num_layers
        self.multiplier = multiplier
        self.kernel_size = kernel_size
        self.in_group_size = in_group_size
        self.pre_group_size = pre_group_size
        self.prefix_size = prefix_size
        self.order = order
        self.scale_lower_bound = scale_lower_bound
        self.tail_mass = tail_mass

        self.pdf_parameters = Sequential(
            MaskedConv2d(
                in_channels=num_channels,
                out_channels=2 * num_channels,
                kernel_size=(kernel_size, kernel_size),
                in_group_size=in_group_size * pre_group_size,
                out_group_size=2 * pre_group_size,
                in_prefix_size=prefix_size,
                out_prefix_size=prefix_size,
                order=order,
                include_target=False,
            ),
            *[
                MaskedBlock(
                    num_channels=2 * num_channels,
                    multiplier=multiplier,
                    kernel_size=kernel_size,
                    group_size=2 * pre_group_size,
                    prefix_size=prefix_size,
                    order=order,
                    scale=True,
                )
                for _ in range(num_layers)
            ],
            MaskedConv2d(
                in_channels=2 * num_channels,
                out_channels=2 * num_channels,
                kernel_size=(kernel_size, kernel_size),
                in_group_size=2 * pre_group_size,
                out_group_size=2 * pre_group_size,
                in_prefix_size=prefix_size,
                out_prefix_size=0,
                order=order,
                include_target=True,
            ),
        )

        self.scale_table, self.cdfs, self.cdf_sizes, self.offsets, self.coder_parameters = None, None, None, None, None

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        inputs_quantized = functions.perturb_or_quantize(inputs, self.training)

        parameters = self.pdf_parameters(inputs_quantized)
        means, scales = functions.disentangle(parameters)

        errors = functions.perturb_or_quantize(inputs - means, self.training)

        likelihoods = functions.calculate_gaussian_likelihoods(errors, scales, self.scale_lower_bound, self.tail_mass)

        return inputs_quantized, likelihoods

    def collect_parameters(self) -> NetworkParameters:
        parameters = [
            (
                functions.to_numpy(self.pdf_parameters[0].weight * self.pdf_parameters[0].mask),
                functions.to_numpy(self.pdf_parameters[0].bias),
                functions.to_numpy(torch.ones_like(self.pdf_parameters[0].bias)),
                -1,
                self.pdf_parameters[0].in_group_size,
                self.pdf_parameters[0].out_group_size,
                self.pdf_parameters[0].in_prefix_size,
                self.pdf_parameters[0].out_prefix_size,
                False,
            )
        ]

        layer_index = 0

        for block in self.pdf_parameters[1:-1]:
            scales_block = functions.to_numpy(block.scales)
            scales_one = functions.to_numpy(torch.ones_like(block.scales))
            residual_layer_index = layer_index

            for index, layer in enumerate(block.block):
                is_last = index + 1 == len(block.block)

                if isinstance(layer, Conv2d):
                    parameters.append((
                        functions.to_numpy(layer.weight * layer.mask),
                        functions.to_numpy(layer.bias),
                        scales_block if is_last else scales_one,
                        residual_layer_index if is_last else -1,
                        layer.in_group_size,
                        layer.out_group_size,
                        layer.in_prefix_size,
                        layer.out_prefix_size,
                        False if is_last else True,
                    ))

                    layer_index += 1

        parameters.append((
            functions.to_numpy(self.pdf_parameters[-1].weight * self.pdf_parameters[-1].mask),
            functions.to_numpy(self.pdf_parameters[-1].bias),
            functions.to_numpy(torch.ones_like(self.pdf_parameters[-1].bias)),
            -1,
            self.pdf_parameters[-1].in_group_size,
            self.pdf_parameters[-1].out_group_size,
            self.pdf_parameters[-1].in_prefix_size,
            self.pdf_parameters[-1].out_prefix_size,
            False,
        ))

        parameters = tuple(parameters)

        return parameters

    def update_coder_parameters(self, force: bool = False) -> None:
        if (
            not force
            and self.scale_table is not None
            and self.cdfs is not None
            and self.cdf_sizes is not None
            and self.offsets is not None
            and self.coder_parameters is not None
        ):
            return

        self.scale_table = normal.get_scale_table(self.scale_lower_bound)
        self.cdfs, self.cdf_sizes, self.offsets = normal.calculate_cdfs(self.scale_table, self.tail_mass)
        self.coder_parameters = self.collect_parameters()

    def encode(self, batch: Tensor) -> List[bytes]:
        self.update_coder_parameters()

        conditional = np.empty((0, batch.shape[2], batch.shape[3]), dtype=np.float32)

        byte_strings = [
            conversion.int_array_to_bytes(
                grouped.encode(
                    sample=functions.to_numpy(sample),
                    conditional=conditional,
                    parameters=self.coder_parameters,
                    cdfs=self.cdfs,
                    cdf_sizes=self.cdf_sizes,
                    offsets=self.offsets,
                    scale_table=self.scale_table,
                    scale_lower_bound=self.scale_lower_bound,
                )
            )
            for sample in batch
        ]

        return byte_strings

    def decode(self, byte_strings: List[bytes], shape: Tuple[int, int, int]) -> Tensor:
        self.update_coder_parameters()

        conditional = np.empty((0, shape[1], shape[2]), dtype=np.float32)

        tensors = [
            grouped.decode(
                codes=conversion.bytes_to_int_array(byte_string),
                conditional=conditional,
                parameters=self.coder_parameters,
                shape=tuple(shape),
                cdfs=self.cdfs,
                cdf_sizes=self.cdf_sizes,
                offsets=self.offsets,
                scale_table=self.scale_table,
                scale_lower_bound=self.scale_lower_bound,
            )
            for byte_string in byte_strings
        ]

        parameter = next(self.parameters())

        batch = torch.stack(
            tensors=[torch.tensor(tensor, dtype=parameter.dtype, device=parameter.device) for tensor in tensors],
            dim=0,
        )

        return batch

    def calculate_sizes(self, batch: Tensor) -> List[int]:
        return [len(byte_string) * 8 for byte_string in self.encode(batch)]


class GaussianConditionalEntropyModel(Module, ConditionalEntropyModel):
    def __init__(
        self,
        num_channels: int,
        num_channels_conditional: int,
        num_channels_conditional_expansion: int = 128,
        num_layers: int = 5,
        num_layers_conditional: int = 2,
        multiplier: int = 1,
        kernel_size: int = 5,
        in_group_size: int = 1,
        pre_group_size: int = 1,
        order: int = -1,
        scale_lower_bound: float = SCALE_MIN,
        tail_mass: float = TAIL_MASS,
    ) -> None:
        super().__init__()

        self.gaussian_entropy_model = GaussianEntropyModel(
            num_channels=num_channels,
            num_layers=num_layers,
            multiplier=multiplier,
            kernel_size=kernel_size,
            in_group_size=in_group_size,
            pre_group_size=pre_group_size,
            order=order,
            prefix_size=num_channels,
            scale_lower_bound=scale_lower_bound,
            tail_mass=tail_mass,
        )

        self.conditional_transformer = modules.inverted_residual_lst(
            in_channels=num_channels_conditional,
            out_channels=num_channels,
            expansion_channels=num_channels_conditional_expansion,
            num_layers=num_layers_conditional,
            norm_layer=None,
        )

    def forward(self, inputs: Tensor, conditionals: Tensor) -> Tuple[Tensor, Tensor]:
        inputs_quantized = functions.perturb_or_quantize(inputs, self.training)

        conditional_transformed = self.conditional_transformer(conditionals)
        inputs_conditioned = torch.concat((conditional_transformed, inputs_quantized), dim=1)

        parameters = self.gaussian_entropy_model.pdf_parameters(inputs_conditioned)
        means, scales = functions.disentangle(parameters)

        errors = functions.perturb_or_quantize(inputs - means, self.training)

        likelihoods = functions.calculate_gaussian_likelihoods(
            errors=errors,
            scales=scales,
            minimum_scales=self.gaussian_entropy_model.scale_lower_bound,
            minimum_likelihood=self.gaussian_entropy_model.tail_mass,
        )

        return inputs_quantized, likelihoods

    def encode(self, batch: Tensor, conditionals: Tensor) -> List[bytes]:
        self.gaussian_entropy_model.update_coder_parameters()

        byte_strings = [
            conversion.int_array_to_bytes(
                grouped.encode(
                    sample=functions.to_numpy(sample),
                    conditional=functions.to_numpy(conditional),
                    parameters=self.coder_parameters,
                    cdfs=self.cdfs,
                    cdf_sizes=self.cdf_sizes,
                    offsets=self.offsets,
                    scale_table=self.scale_table,
                    scale_lower_bound=self.scale_lower_bound,
                )
            )
            for sample, conditional in zip(batch, conditionals)
        ]

        return byte_strings

    def decode(self, byte_strings: List[bytes], shape: Tuple[int, int, int], conditionals: Tensor) -> Tensor:
        self.gaussian_entropy_model.update_coder_parameters()

        tensors = [
            grouped.decode(
                codes=conversion.bytes_to_int_array(byte_string),
                conditional=functions.to_numpy(conditional),
                parameters=self.coder_parameters,
                shape=tuple(shape),
                cdfs=self.cdfs,
                cdf_sizes=self.cdf_sizes,
                offsets=self.offsets,
                scale_table=self.scale_table,
                scale_lower_bound=self.scale_lower_bound,
            )
            for byte_string, conditional in zip(byte_strings, conditionals)
        ]

        parameter = next(self.parameters())

        batch = torch.stack(
            tensors=[torch.tensor(tensor, dtype=parameter.dtype, device=parameter.device) for tensor in tensors],
            dim=0,
        )

        return batch

    def calculate_sizes(self, batch: Tensor, conditionals: Tensor) -> List[int]:
        return [len(byte_string) * 8 for byte_string in self.encode(batch, conditionals)]
